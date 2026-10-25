#!/usr/bin/env python3
"""Exact parameter, weight-memory, and KV-cache accounting for the final five candidates.

Counting convention: total learned deployable parameters < 500,000,000, including tied
embedding (counted once), norms, QK norms, routers/connectors/adapters where present.
No linear biases. SwiGLU has gate/up/down matrices.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

@dataclass
class Candidate:
    rank_hint: int
    slug: str
    name: str
    vocab: int
    d_model: int
    layers: int
    d_ff: int
    q_heads: int
    kv_heads: int
    attn_layers: int
    kv_follower_layers: int = 0
    qk_norm: bool = True
    context: int = 8192
    local_layers: int = 0
    global_layers: int = 0
    local_window: int = 0
    # learned extras beyond the standard stack
    extra_params: int = 0
    # full-depth attention layers that produce independent KV states
    kv_producer_layers: int | None = None
    description: str = ""

    @property
    def head_dim(self):
        if self.d_model % self.q_heads:
            raise ValueError(f"{self.slug}: d_model must divide q_heads")
        return self.d_model // self.q_heads

    @property
    def kv_dim(self):
        return self.kv_heads * self.head_dim

    @property
    def embedding_params(self):
        return self.vocab * self.d_model

    @property
    def ffn_per_layer(self):
        return 3 * self.d_model * self.d_ff

    @property
    def attn_full_params(self):
        # Q + K + V + O
        return 2 * self.d_model * self.d_model + 2 * self.d_model * self.kv_dim

    @property
    def attn_follower_params(self):
        # Cross-layer-KV follower computes Q and O but reuses K,V from previous source layer.
        return 2 * self.d_model * self.d_model

    @property
    def norm_params(self):
        # One RMSNorm before every FFN, one before every present attention, plus final RMSNorm.
        return (self.layers + self.attn_layers + 1) * self.d_model

    @property
    def qk_norm_params(self):
        if not self.qk_norm:
            return 0
        # Source attention: Q and KV head-dimensional norm scales. Follower: Q only.
        src = self.attn_layers - self.kv_follower_layers
        return src * (self.q_heads + self.kv_heads) * self.head_dim + self.kv_follower_layers * self.q_heads * self.head_dim

    @property
    def total_params(self):
        src = self.attn_layers - self.kv_follower_layers
        return (
            self.embedding_params
            + self.layers * self.ffn_per_layer
            + src * self.attn_full_params
            + self.kv_follower_layers * self.attn_follower_params
            + self.norm_params
            + self.qk_norm_params
            + self.extra_params
        )

    def weight_bytes(self, bits: float, group: int | None = None, scale_bytes: int = 2):
        eff = bits
        if group:
            eff += 8 * scale_bytes / group
        return self.total_params * eff / 8

    def kv_bytes(self, tokens: int, kv_bits: int = 8):
        """Payload estimate, excluding scale metadata/padding.

        Local/global candidates cap local layers at local_window. Cross-layer KV uses only
        producer layers. Other candidates use kv_producer_layers or attn_layers.
        Recursion-specific extra cache is modeled separately in the final report.
        """
        per_state = 2 * self.kv_heads * self.head_dim * kv_bits / 8
        if self.local_layers or self.global_layers:
            return per_state * (
                self.local_layers * min(tokens, self.local_window)
                + self.global_layers * tokens
            )
        producers = self.kv_producer_layers if self.kv_producer_layers is not None else (self.attn_layers - self.kv_follower_layers)
        return per_state * producers * tokens

# Candidate 1 extras: 4-tap recurrent connector at bottleneck r=128, router d->128->3,
# 3 recursion embeddings, four tap RMS scales, and rank-16 output adapters for six repeated layers.
d=1280; r=128
rec_connector = 4*d*r + r*d + 3*r*r + 4*d
rec_router = d*128 + 128*3 + 3*d
rec_adapters = 6 * (2*d*16)
REC_EXTRAS = rec_connector + rec_router + rec_adapters

CANDIDATES = [
    Candidate(
        1,"edge_rec_482","EdgeRec-482 Adaptive Monotonic Recursive Transformer",
        49152,1280,25,3328,20,4,25,
        context=16384, extra_params=REC_EXTRAS, kv_producer_layers=25,
        description="25 physical layers; unique stem/tail; a 6-layer middle span can receive one extra token-selective pass; zero-init recurrent connector, rank-16 depth adapters, causal budget router."
    ),
    Candidate(
        2,"dense_deep_485","DenseDeep-485 Quality Baseline",
        49152,1024,36,3072,16,4,36,
        context=8192, kv_producer_layers=36,
        description="Deep-thin all-attention baseline inspired by MobileLLM; maximum architectural regularity and no parameter sharing."
    ),
    Candidate(
        3,"flash_skip_497","FlashSkip-497 Latency Transformer",
        65536,1280,16,6144,20,4,9,
        context=8192, kv_producer_layers=9,
        description="Shallow/wide, only 9 of 16 blocks contain global GQA; remaining blocks are FFN-only skip-attention blocks; standard mobile-runtime operators."
    ),
    Candidate(
        4,"cla_cache_491","CLA-Cache-491 Cross-Layer-KV Transformer",
        49152,1024,35,3328,16,4,35,kv_follower_layers=17,
        context=16384, kv_producer_layers=18,
        description="35 layers; adjacent layer pairs share K/V projections and cached states, approximately halving KV-producing depth while preserving a query/output attention transform at every layer."
    ),
    Candidate(
        5,"local_global_488","LocalGlobal-488 Portable Long-Context Transformer",
        49152,1280,24,3584,20,4,24,
        context=32768, local_layers=20, global_layers=4, local_window=2048,
        description="24 layers with 5:1 local/global GQA, 2K local window; conservative standard Transformer operations with bounded cache on most layers."
    ),
]


def mb(x): return x/1e6
def mib(x): return x/(1024**2)

rows=[]
for c in CANDIDATES:
    if c.total_params >= 500_000_000:
        raise AssertionError(f"{c.slug} violates parameter ceiling: {c.total_params}")
    row={
        "rank_hint":c.rank_hint,"slug":c.slug,"name":c.name,
        "params":c.total_params,"params_M":round(c.total_params/1e6,3),
        "vocab":c.vocab,"d_model":c.d_model,"layers_physical":c.layers,
        "d_ff":c.d_ff,"q_heads":c.q_heads,"kv_heads":c.kv_heads,"head_dim":c.head_dim,
        "attention_layers":c.attn_layers,"kv_producer_layers":c.kv_producer_layers if c.kv_producer_layers is not None else c.attn_layers-c.kv_follower_layers,
        "context":c.context,"local_layers":c.local_layers,"global_layers":c.global_layers,"local_window":c.local_window,
        "embedding_M":round(c.embedding_params/1e6,3),"ffn_per_layer_M":round(c.ffn_per_layer/1e6,3),
        "attn_source_M":round(c.attn_full_params/1e6,3),"extras_M":round(c.extra_params/1e6,3),
        "fp16_MB":round(mb(c.weight_bytes(16)),2),"int8_MB":round(mb(c.weight_bytes(8)),2),
        "w4_g128_MB":round(mb(c.weight_bytes(4,128)),2),"w4_g32_MB":round(mb(c.weight_bytes(4,32)),2),
        "kv8_2k_MiB":round(mib(c.kv_bytes(2048,8)),2),"kv4_2k_MiB":round(mib(c.kv_bytes(2048,4)),2),
        "kv8_8k_MiB":round(mib(c.kv_bytes(8192,8)),2),"kv4_8k_MiB":round(mib(c.kv_bytes(8192,4)),2),
        "kv8_32k_MiB":round(mib(c.kv_bytes(32768,8)),2),"kv4_32k_MiB":round(mib(c.kv_bytes(32768,4)),2),
        "description":c.description,
    }
    rows.append(row)

csv_path=TABLES/"architecture_candidates.csv"
with csv_path.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
json_path=TABLES/"architecture_candidates.json"
json_path.write_text(json.dumps(rows,indent=2),encoding="utf-8")

print("STRICT_PARAMETER_CEILING=500000000")
for r in rows:
    print(f"{r['rank_hint']}. {r['name']}: {r['params']:,} params ({r['params_M']}M), W4/g128={r['w4_g128_MB']} MB, KV4@8K={r['kv4_8k_MiB']} MiB")
print(f"WROTE {csv_path}")
print(f"WROTE {json_path}")
