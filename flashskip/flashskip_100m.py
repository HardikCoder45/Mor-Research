#!/usr/bin/env python3
"""FlashSkip-100M: reproducible T4 training/evaluation pilot.

Architecture: 99,182,336 learned parameters
- 12 decoder blocks, SwiGLU in every block
- GQA attention in 7 blocks: [0, 1, 3, 5, 7, 9, 11]
- d_model=640, d_ff=2640, 10 query heads, 2 KV heads, head_dim=64
- 49,152 tied token embeddings / LM head (SmolLM2 tokenizer)
- Pre-RMSNorm, per-head QK RMSNorm, RoPE, no linear biases

The script supports: smoke, prepare, train, evaluate, verify, package.
All persistent training artifacts are written beneath --output-dir.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def emit_live(kind: str, data: dict[str, Any]) -> None:
    """Best-effort authenticated telemetry; training never depends on the relay."""
    url = os.environ.get("FLASHSKIP_TELEMETRY_URL", "")
    token = os.environ.get("FLASHSKIP_TELEMETRY_TOKEN", "")
    if not url or not token:
        return
    payload = json.dumps({"type": kind, "data": data}, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/api/telemetry", data=payload,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            response.read(1)
    except Exception:
        pass


# ----------------------------- configuration -----------------------------

@dataclass
class FlashSkipConfig:
    vocab_size: int = 49152
    d_model: int = 640
    d_ff: int = 2640
    n_layers: int = 12
    n_q_heads: int = 10
    n_kv_heads: int = 2
    attention_layers: tuple[int, ...] = (0, 1, 3, 5, 7, 9, 11)
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 8192
    tie_embeddings: bool = True
    model_name: str = "FlashSkip-100M"
    architecture_version: str = "pilot-v1-2026-08-30"

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_q_heads == 0
        return self.d_model // self.n_q_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["attention_layers"] = list(self.attention_layers)
        d["head_dim"] = self.head_dim
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FlashSkipConfig":
        d = dict(d)
        d.pop("head_dim", None)
        if "attention_layers" in d:
            d["attention_layers"] = tuple(d["attention_layers"])
        return cls(**d)


# ----------------------------- model -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight


class HeadRMSNorm(nn.Module):
    """Per-head, per-channel RMS scale: shape [H, Dh]."""
    def __init__(self, heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,H,T,D]
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(dtype) * self.weight[None, :, None, :]


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x [B,H,T,D]; cos/sin [T,D/2]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GQAAttention(nn.Module):
    def __init__(self, cfg: FlashSkipConfig):
        super().__init__()
        self.cfg = cfg
        d, hd = cfg.d_model, cfg.head_dim
        self.q = nn.Linear(d, cfg.n_q_heads * hd, bias=False)
        self.k = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.v = nn.Linear(d, cfg.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.n_q_heads * hd, d, bias=False)
        self.q_norm = HeadRMSNorm(cfg.n_q_heads, hd, cfg.rms_eps)
        self.k_norm = HeadRMSNorm(cfg.n_kv_heads, hd, cfg.rms_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        hq, hk, hd = self.cfg.n_q_heads, self.cfg.n_kv_heads, self.cfg.head_dim
        q = self.q(x).view(b, t, hq, hd).transpose(1, 2)
        k = self.k(x).view(b, t, hk, hd).transpose(1, 2)
        v = self.v(x).view(b, t, hk, hd).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        repeat = hq // hk
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, hq * hd)
        return self.o(y)


class FlashSkipBlock(nn.Module):
    def __init__(self, cfg: FlashSkipConfig, has_attention: bool):
        super().__init__()
        self.has_attention = has_attention
        if has_attention:
            self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
            self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.has_attention:
            x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FlashSkipLM(nn.Module):
    def __init__(self, cfg: FlashSkipConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        attn_set = set(cfg.attention_layers)
        self.blocks = nn.ModuleList([FlashSkipBlock(cfg, i in attn_set) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.apply(self._init_weights)
        # GPT-style scaled residual projection initialization.
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=residual_std)
            if block.has_attention:
                nn.init.normal_(block.attn.o.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds configured max {self.cfg.max_seq_len}")
        x = self.tok_embeddings(input_ids)
        cos, sin = rope_cos_sin(t, self.cfg.head_dim, self.cfg.rope_theta, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)
        # exact weight tying: no separate LM-head Parameter.
        return F.linear(x, self.tok_embeddings.weight)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 48, temperature: float = 0.8, top_k: int = 50):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(ctx)[:, -1, :]
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < vals[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids


def exact_parameter_breakdown(cfg: FlashSkipConfig) -> dict[str, int]:
    d, hd = cfg.d_model, cfg.head_dim
    embed = cfg.vocab_size * d
    ffn = cfg.n_layers * 3 * d * cfg.d_ff
    per_attn = d * (cfg.n_q_heads * hd) + 2 * d * (cfg.n_kv_heads * hd) + (cfg.n_q_heads * hd) * d
    attn = len(cfg.attention_layers) * per_attn
    norms = (cfg.n_layers + len(cfg.attention_layers) + 1) * d
    qk_norm = len(cfg.attention_layers) * (cfg.n_q_heads + cfg.n_kv_heads) * hd
    return {"embedding": embed, "ffn": ffn, "attention": attn, "rms_norm": norms, "qk_norm": qk_norm,
            "total": embed + ffn + attn + norms + qk_norm}


# ----------------------------- utilities -----------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_dump(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def device_info() -> dict[str, Any]:
    out = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "python": sys.version}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        out.update({"device": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "vram_bytes": p.total_memory, "capability": torch.cuda.get_device_capability(0)})
    else:
        out["device"] = "cpu"
    return out


def autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_tokenizer(model_id: str, output_dir: Path):
    from transformers import AutoTokenizer
    tok_dir = output_dir / "tokenizer"
    if (tok_dir / "tokenizer.json").exists() or (tok_dir / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        tok_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(tok_dir))
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer must define eos_token_id")
    if len(tokenizer) != 49152:
        raise RuntimeError(f"Expected SmolLM2 tokenizer vocab=49152, got {len(tokenizer)}")
    return tokenizer


# ----------------------------- data preparation -----------------------------

def _load_fineweb_stream(seed: int):
    from datasets import load_dataset
    attempts = [
        ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
        ("HuggingFaceFW/fineweb-edu", None),
        ("HuggingFaceFW/fineweb", "sample-10BT"),
    ]
    errors = []
    for repo, name in attempts:
        try:
            kw = {"path": repo, "split": "train", "streaming": True}
            if name is not None:
                kw["name"] = name
            ds = load_dataset(**kw)
            ds = ds.shuffle(seed=seed, buffer_size=5000)
            return ds, {"repo": repo, "config": name}
        except Exception as e:
            errors.append(f"{repo}/{name}: {type(e).__name__}: {e}")
    raise RuntimeError("Could not load FineWeb stream:\n" + "\n".join(errors))


def prepare_data(args, cfg: FlashSkipConfig):
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer, output_dir)
    train_path, val_path = data_dir / "train_tokens.uint16", data_dir / "val_tokens.uint16"
    manifest_path = output_dir / "data_manifest.json"
    if train_path.exists() and val_path.exists() and manifest_path.exists() and not args.force_prepare:
        m = json.loads(manifest_path.read_text())
        if m.get("train_tokens", 0) >= args.prepare_train_tokens and m.get("val_tokens", 0) >= args.prepare_val_tokens:
            print("DATA_ALREADY_PREPARED", json.dumps(m, sort_keys=True), flush=True)
            return m

    ds, source = _load_fineweb_stream(args.seed)
    print("DATA_SOURCE", source, flush=True)
    targets = [("val", val_path, int(args.prepare_val_tokens)), ("train", train_path, int(args.prepare_train_tokens))]
    writers = {name: path.open("wb") for name, path, _ in targets}
    written = {"val": 0, "train": 0}
    phase_idx = 0
    batch_texts: list[str] = []
    docs = 0
    started = time.time()
    eos = int(tokenizer.eos_token_id)

    def emit(texts: list[str]):
        nonlocal phase_idx, docs
        if not texts or phase_idx >= len(targets):
            return
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
        for ids in encoded:
            docs += 1
            ids = list(ids) + [eos]
            pos = 0
            while pos < len(ids) and phase_idx < len(targets):
                name, _, target = targets[phase_idx]
                need = target - written[name]
                if need <= 0:
                    phase_idx += 1
                    continue
                take = min(need, len(ids) - pos)
                arr = np.asarray(ids[pos:pos + take], dtype=np.uint16)
                arr.tofile(writers[name])
                written[name] += int(take)
                pos += take
                if written[name] >= target:
                    writers[name].flush()
                    os.fsync(writers[name].fileno())
                    phase_idx += 1
            if docs % 1000 == 0:
                elapsed = max(1e-6, time.time() - started)
                total = written["val"] + written["train"]
                json_dump(output_dir / "logs" / "prepare_progress.json", {
                    "event": "prepare",
                    "documents": docs,
                    "val_tokens": written["val"],
                    "train_tokens": written["train"],
                    "target_val_tokens": int(args.prepare_val_tokens),
                    "target_train_tokens": int(args.prepare_train_tokens),
                    "tokens_per_second": total / elapsed,
                    "elapsed_seconds": elapsed,
                    "updated_at": time.time(),
                })
                emit_live("prepare", {
                    "documents": docs, "val_tokens": written["val"], "train_tokens": written["train"],
                    "target_val_tokens": int(args.prepare_val_tokens),
                    "target_train_tokens": int(args.prepare_train_tokens),
                    "tokens_per_second": total / elapsed, "elapsed_seconds": elapsed,
                })
                print(f"PREP docs={docs} val={written['val']:,} train={written['train']:,} tok/s={total/elapsed:,.0f}", flush=True)

    try:
        for ex in ds:
            text = ex.get("text") if isinstance(ex, dict) else None
            if not text or len(text) < 80:
                continue
            batch_texts.append(text)
            if len(batch_texts) >= 64:
                emit(batch_texts)
                batch_texts.clear()
            if phase_idx >= len(targets):
                break
        emit(batch_texts)
    finally:
        for f in writers.values():
            f.close()

    if written["val"] < args.prepare_val_tokens or written["train"] < args.prepare_train_tokens:
        raise RuntimeError(f"stream ended early: {written}")
    manifest = {
        "source": source,
        "tokenizer": args.tokenizer,
        "tokenizer_vocab": len(tokenizer),
        "eos_token_id": eos,
        "dtype": "uint16",
        "train_tokens": written["train"],
        "val_tokens": written["val"],
        "documents_consumed": docs,
        "seed": args.seed,
        "shuffle_buffer_documents": 5000,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_sha256": sha256_file(train_path),
        "val_sha256": sha256_file(val_path),
        "prepared_seconds": time.time() - started,
    }
    json_dump(manifest_path, manifest)
    print("PREPARED", json.dumps(manifest, sort_keys=True), flush=True)
    emit_live("prepared", manifest)
    return manifest


# ----------------------------- training/checkpoints -----------------------------

def memmap_tokens(path: Path):
    return np.memmap(path, mode="r", dtype=np.uint16)


def next_sequential_batch(tokens, cursor: int, batch_size: int, seq_len: int, device: torch.device,
                          *, allow_wrap: bool = True):
    span = batch_size * (seq_len + 1)
    if cursor < 0:
        raise RuntimeError(f"negative data cursor: {cursor}")
    if cursor + span > len(tokens):
        if not allow_wrap:
            raise RuntimeError(
                f"training data exhausted before target: cursor={cursor:,} span={span:,} "
                f"available={len(tokens):,}; refusing to wrap/replay old source tokens"
            )
        cursor = 0
    a = np.asarray(tokens[cursor:cursor + span], dtype=np.int64).reshape(batch_size, seq_len + 1)
    x = torch.from_numpy(a[:, :-1].copy()).to(device, non_blocking=True)
    y = torch.from_numpy(a[:, 1:].copy()).to(device, non_blocking=True)
    return x, y, cursor + span


def lr_for_step(step: int, total_steps: int, warmup_steps: int, max_lr: float, min_lr: float):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    frac = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    frac = min(max(frac, 0.0), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * frac))


@torch.no_grad()
def evaluate_memmap_loss(model: FlashSkipLM, tokens, device: torch.device, seq_len: int = 512, batches: int = 32, batch_size: int = 2):
    model.eval()
    losses, cursor, n_tok = [], 0, 0
    for _ in range(batches):
        x, y, cursor = next_sequential_batch(tokens, cursor, batch_size, seq_len, device)
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(float(loss))
        n_tok += y.numel()
    mean = float(sum(losses) / len(losses))
    return {"loss": mean, "perplexity": float(math.exp(min(mean, 20.0))), "tokens": n_tok}


def optimizer_state_to_half_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    out = {"param_groups": copy.deepcopy(state_dict["param_groups"]), "state": {}}
    for pid, st in state_dict["state"].items():
        ns = {}
        for k, v in st.items():
            if torch.is_tensor(v):
                v = v.detach().cpu()
                if v.is_floating_point() and v.numel() > 1:
                    v = v.to(torch.float16)
                ns[k] = v
            else:
                ns[k] = v
        out["state"][pid] = ns
    return out


def restore_optimizer_device_dtype(optimizer: torch.optim.Optimizer):
    for p, st in optimizer.state.items():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                if k in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                    st[k] = v.to(device=p.device, dtype=p.dtype)
                else:
                    st[k] = v.to(device=p.device)


def save_model_safetensors(model: nn.Module, path: Path):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().to(torch.float16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path), metadata={"format": "pt", "dtype": "float16", "architecture": "FlashSkipLM"})


def load_model_safetensors(model: nn.Module, path: Path, strict: bool = True):
    from safetensors.torch import load_file
    state = load_file(str(path), device="cpu")
    return model.load_state_dict(state, strict=strict)


def write_model_card(output_dir: Path, cfg: FlashSkipConfig, train_summary: dict[str, Any] | None = None):
    breakdown = exact_parameter_breakdown(cfg)
    lines = [
        "# FlashSkip-100M Training Artifact",
        "",
        f"Architecture version: `{cfg.architecture_version}`",
        f"Total learned parameters: **{breakdown['total']:,}**",
        "",
        "## Architecture",
        f"- {cfg.n_layers} decoder blocks; SwiGLU in every block",
        f"- attention blocks: {list(cfg.attention_layers)} ({len(cfg.attention_layers)}/{cfg.n_layers})",
        f"- d_model={cfg.d_model}, d_ff={cfg.d_ff}",
        f"- {cfg.n_q_heads} query heads / {cfg.n_kv_heads} KV heads / head_dim={cfg.head_dim}",
        f"- tied {cfg.vocab_size:,}-token embedding/output weights",
        "- pre-RMSNorm, per-head QK-RMSNorm, RoPE, no linear biases",
        "",
        "## Loading",
        "Use the saved `source/flashskip_100m.py` with `config.json`, `final/model.safetensors`, and `tokenizer/`.",
        "For training resume, use the corresponding `training_state.pt` in the checkpoint/final directory.",
        "",
        "## Important scope",
        "This is a bounded pilot run on a T4, not a fully pretrained production model. Check `training_summary.json` and `evaluation/report.md` for the achieved token budget and scores.",
    ]
    if train_summary:
        lines += ["", "## Pilot training", f"- tokens seen: {train_summary.get('tokens_seen', 0):,}",
                  f"- optimizer steps: {train_summary.get('steps', 0):,}",
                  f"- final held-out FineWeb-Edu loss: {train_summary.get('final_val', {}).get('loss')}"]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_checkpoint(output_dir: Path, model: FlashSkipLM, optimizer, scaler, cfg: FlashSkipConfig,
                    step: int, tokens_seen: int, data_cursor: int, metrics: dict[str, Any], seed: int):
    ckpt_name = f"checkpoint_step_{step:06d}_tokens_{tokens_seen/1e6:07.2f}M"
    parent = output_dir / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    ckpt = parent / ckpt_name
    partial = parent / (ckpt_name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    save_model_safetensors(model, partial / "model.safetensors")
    json_dump(partial / "config.json", cfg.to_jsonable())
    json_dump(partial / "metrics.json", metrics)
    training_state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "data_cursor": data_cursor,
        "optimizer_state_half_disk": optimizer_state_to_half_cpu(optimizer.state_dict()),
        "grad_scaler": scaler.state_dict() if scaler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "seed": seed,
        "note": "Optimizer moments are stored FP16 on disk and cast back to parameter dtype when resumed.",
    }
    torch.save(training_state, partial / "training_state.pt")
    files = {}
    for name in ("model.safetensors", "training_state.pt", "config.json", "metrics.json"):
        p = partial / name
        files[name] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    json_dump(partial / "checkpoint_manifest.json", {
        "checkpoint_name": ckpt_name,
        "step": step,
        "tokens_seen": tokens_seen,
        "data_cursor": data_cursor,
        "files": files,
    })
    if ckpt.exists():
        shutil.rmtree(partial)
    else:
        os.replace(partial, ckpt)
    json_dump(output_dir / "latest_checkpoint.json", {"path": str(ckpt), "step": step, "tokens_seen": tokens_seen})
    print(f"CHECKPOINT_SAVED {ckpt}", flush=True)
    return ckpt


def resume_checkpoint(model, optimizer, scaler, ckpt: Path, device: torch.device):
    load_model_safetensors(model, ckpt / "model.safetensors")
    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_half_disk"])
    restore_optimizer_device_dtype(optimizer)
    if scaler is not None and state.get("grad_scaler"):
        scaler.load_state_dict(state["grad_scaler"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    np.random.set_state(state["numpy_rng_state"])
    random.setstate(state["python_rng_state"])
    return int(state["step"]), int(state["tokens_seen"]), int(state["data_cursor"])


def train(args, cfg: FlashSkipConfig):
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    tokenizer = load_tokenizer(args.tokenizer, output_dir)
    # Save an exact source snapshot with the artifacts.
    src_dir = output_dir / "source"
    src_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(Path(__file__), src_dir / "flashskip_100m.py")
    except Exception:
        pass
    json_dump(output_dir / "config.json", cfg.to_jsonable())
    json_dump(output_dir / "parameter_breakdown.json", exact_parameter_breakdown(cfg))
    json_dump(output_dir / "environment.json", device_info())

    train_path, val_path = data_dir / "train_tokens.uint16", data_dir / "val_tokens.uint16"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Prepared token files missing; run prepare first")
    train_tokens, val_tokens = memmap_tokens(train_path), memmap_tokens(val_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu_train:
        raise RuntimeError("Training pilot expects CUDA; use --allow-cpu-train only for a smoke test")

    model = FlashSkipLM(cfg).to(device)
    actual_params = sum(p.numel() for p in model.parameters())
    expected = exact_parameter_breakdown(cfg)["total"]
    if actual_params != expected:
        raise RuntimeError(f"parameter mismatch actual={actual_params} expected={expected}")
    print(f"MODEL_PARAMS {actual_params:,}", flush=True)
    print("DEVICE_INFO", json.dumps(device_info(), sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    global_tokens = args.micro_batch * args.seq_len * args.grad_accum
    total_steps = math.ceil(args.target_tokens / global_tokens)
    schedule_target_tokens = args.schedule_target_tokens or args.target_tokens
    schedule_total_step = math.ceil(schedule_target_tokens / global_tokens)
    schedule_steps = max(1, schedule_total_step - args.schedule_start_step)
    warmup_steps = max(1, int(schedule_steps * args.warmup_fraction))
    step, tokens_seen, data_cursor = 0, 0, 0

    if args.resume:
        ckpt = Path(args.resume)
        step, tokens_seen, data_cursor = resume_checkpoint(model, optimizer, scaler, ckpt, device)
        print(f"RESUMED {ckpt} step={step} tokens={tokens_seen} data_cursor={data_cursor}", flush=True)
    if args.require_resume_cursor_at_least:
        if not args.resume:
            raise RuntimeError("--require-resume-cursor-at-least requires --resume")
        if data_cursor < args.require_resume_cursor_at_least:
            raise RuntimeError(
                f"resume data_cursor={data_cursor:,} is before required no-replay boundary "
                f"{args.require_resume_cursor_at_least:,}"
            )
    if data_cursor >= len(train_tokens):
        raise RuntimeError(f"resume data_cursor={data_cursor:,} is outside prepared train data ({len(train_tokens):,})")
    print("NO_REPLAY_GUARD", json.dumps({
        "resume_data_cursor": data_cursor,
        "required_min_cursor": args.require_resume_cursor_at_least,
        "prepared_train_tokens": len(train_tokens),
        "target_tokens_seen": args.target_tokens,
    }, sort_keys=True), flush=True)

    train_model: nn.Module = model
    if args.data_parallel and torch.cuda.device_count() > 1:
        class CausalLoss(nn.Module):
            def __init__(self, inner: nn.Module):
                super().__init__(); self.inner = inner
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                logits = self.inner(x)
                return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        train_model = nn.DataParallel(CausalLoss(model))
        print(f"DATA_PARALLEL devices={torch.cuda.device_count()}", flush=True)

    val0 = evaluate_memmap_loss(model, val_tokens, device, min(args.seq_len, 512), args.eval_batches, args.eval_batch_size)
    print("INITIAL_VAL", json.dumps(val0), flush=True)
    emit_live("initial_eval", {"event": "initial_eval", "step": step, "tokens_seen": tokens_seen, **val0})
    metrics_path = output_dir / "logs" / "train_metrics.jsonl"
    append_jsonl(metrics_path, {"event": "initial_eval", "step": step, "tokens_seen": tokens_seen, **val0})

    checkpoint_origin = int(args.checkpoint_origin_tokens)
    if tokens_seen < checkpoint_origin:
        next_ckpt = checkpoint_origin
    else:
        next_ckpt = checkpoint_origin + ((tokens_seen - checkpoint_origin) // args.checkpoint_tokens + 1) * args.checkpoint_tokens
    training_start = time.time()
    recent_times, recent_losses = [], []
    last_ckpt = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    while tokens_seen < args.target_tokens:
        step += 1
        schedule_step = max(0, step - 1 - args.schedule_start_step)
        lr = lr_for_step(schedule_step, schedule_steps, warmup_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_start = time.perf_counter()
        for _micro in range(args.grad_accum):
            x, y, data_cursor = next_sequential_batch(
                train_tokens, data_cursor, args.micro_batch, args.seq_len, device, allow_wrap=False
            )
            with autocast_ctx(device):
                if train_model is model:
                    logits = model(x)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                else:
                    ce = train_model(x, y).mean()
                loss = ce / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += float(ce.detach()) / args.grad_accum
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - step_start
        this_tokens = global_tokens
        tokens_seen += this_tokens
        recent_times.append(dt)
        recent_losses.append(step_loss)
        recent_times = recent_times[-20:]
        recent_losses = recent_losses[-20:]

        if step == 1 or step % args.log_every == 0:
            tok_s = this_tokens / max(dt, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            row = {"event": "train", "step": step, "tokens_seen": tokens_seen, "loss": step_loss,
                   "loss_20": float(sum(recent_losses) / len(recent_losses)), "lr": lr, "grad_norm": grad_norm,
                   "step_seconds": dt, "tokens_per_second": tok_s, "peak_cuda_GiB": peak,
                   "elapsed_seconds": time.time() - training_start}
            append_jsonl(metrics_path, row)
            print("TRAIN", json.dumps(row, sort_keys=True), flush=True)
            emit_live("train", row)

        should_ckpt = tokens_seen >= next_ckpt or tokens_seen >= args.target_tokens
        if should_ckpt:
            val = evaluate_memmap_loss(model, val_tokens, device, min(args.seq_len, 512), args.eval_batches, args.eval_batch_size)
            ck_metrics = {"step": step, "tokens_seen": tokens_seen, "train_loss_recent": float(sum(recent_losses)/len(recent_losses)),
                          "val": val, "lr": lr, "elapsed_seconds": time.time() - training_start,
                          "peak_cuda_GiB": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else None}
            append_jsonl(metrics_path, {"event": "checkpoint_eval", **ck_metrics})
            print("CHECKPOINT_EVAL", json.dumps(ck_metrics, sort_keys=True), flush=True)
            emit_live("checkpoint_eval", {"event": "checkpoint_eval", **ck_metrics})
            last_ckpt = save_checkpoint(output_dir, model, optimizer, scaler, cfg, step, tokens_seen, data_cursor, ck_metrics, args.seed)
            while next_ckpt <= tokens_seen:
                next_ckpt += args.checkpoint_tokens

    final_val = evaluate_memmap_loss(model, val_tokens, device, min(args.seq_len, 512), max(args.eval_batches, 48), args.eval_batch_size)
    summary = {
        "model": cfg.model_name,
        "parameters": actual_params,
        "steps": step,
        "tokens_seen": tokens_seen,
        "target_tokens": args.target_tokens,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "global_tokens_per_step": global_tokens,
        "max_lr": args.max_lr,
        "min_lr": args.min_lr,
        "warmup_steps": warmup_steps,
        "schedule_start_step": args.schedule_start_step,
        "schedule_target_tokens": schedule_target_tokens,
        "training_seconds": time.time() - training_start,
        "average_effective_tokens_per_second": tokens_seen / max(time.time() - training_start, 1e-9),
        "initial_val": val0,
        "final_val": final_val,
        "peak_cuda_GiB": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else None,
        "last_checkpoint": str(last_ckpt) if last_ckpt else None,
    }
    json_dump(output_dir / "training_summary.json", summary)
    write_model_card(output_dir, cfg, summary)

    # Final is a copy/hardlink of the last checkpoint so it remains independently discoverable.
    if last_ckpt is None:
        raise RuntimeError("No checkpoint was saved")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ["model.safetensors", "training_state.pt", "config.json", "metrics.json"]:
        src, dst = last_ckpt / name, final_dir / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    json_dump(final_dir / "final_manifest.json", {"source_checkpoint": str(last_ckpt), **summary})
    print("TRAINING_COMPLETE", json.dumps(summary, sort_keys=True), flush=True)
    return summary


# ----------------------------- evaluation -----------------------------

def load_final_model(output_dir: Path, device: torch.device):
    cfg = FlashSkipConfig.from_dict(json.loads((output_dir / "config.json").read_text()))
    model = FlashSkipLM(cfg)
    load_model_safetensors(model, output_dir / "final" / "model.safetensors")
    model.to(device).eval()
    tokenizer = load_tokenizer("HuggingFaceTB/SmolLM2-135M", output_dir)
    return model, tokenizer, cfg


@torch.no_grad()
def score_completion(model, tokenizer, prompt: str, completion: str, device: torch.device, max_len: int = 512):
    p = tokenizer.encode(prompt, add_special_tokens=False)
    c = tokenizer.encode(completion, add_special_tokens=False)
    if not c:
        return -1e9
    if len(p) + len(c) > max_len:
        keep_prompt = max(1, max_len - len(c))
        p = p[-keep_prompt:]
        c = c[:max_len - len(p)]
    ids = p + c
    if len(ids) < 2:
        return -1e9
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with autocast_ctx(device):
        logits = model(x)[0]
        logp = F.log_softmax(logits.float(), dim=-1)
    # Predictions corresponding to completion tokens begin at target index len(p)-1.
    start = max(0, len(p) - 1)
    idx = torch.arange(start, targets.numel(), device=device)
    selected_targets = targets[start:]
    if selected_targets.numel() == 0:
        return -1e9
    return float(logp[idx, selected_targets].mean())


@torch.no_grad()
def eval_wikitext(model, tokenizer, device: torch.device, max_tokens: int = 200_000, seq_len: int = 512):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = []
    eos = tokenizer.eos_token_id
    for ex in ds:
        t = ex.get("text", "")
        if not t.strip():
            continue
        ids.extend(tokenizer.encode(t, add_special_tokens=False))
        ids.append(eos)
        if len(ids) >= max_tokens + 1:
            break
    ids = ids[:max_tokens + 1]
    losses, count = [], 0
    for s in range(0, len(ids) - 1, seq_len):
        chunk = ids[s:s + seq_len + 1]
        if len(chunk) < 32:
            break
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device)[None, :]
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device)[None, :]
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")
        losses.append(float(loss) * y.numel())
        count += y.numel()
    mean = sum(losses) / max(1, count)
    return {"loss": mean, "perplexity": float(math.exp(min(mean, 20))), "tokens": count}


@torch.no_grad()
def eval_hellaswag(model, tokenizer, device: torch.device, n: int = 100):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0
    details = []
    for i, ex in enumerate(ds):
        if i >= n: break
        prompt = ex["ctx"]
        scores = [score_completion(model, tokenizer, prompt, " " + end, device) for end in ex["endings"]]
        pred = int(np.argmax(scores)); label = int(ex["label"])
        correct += pred == label
        if i < 10: details.append({"index": i, "pred": pred, "label": label, "scores": scores})
    return {"accuracy": correct / max(1, n), "correct": correct, "examples": n, "chance": 0.25, "details_first10": details}


@torch.no_grad()
def eval_arc_easy(model, tokenizer, device: torch.device, n: int = 100):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    correct = 0; seen = 0; details = []
    for ex in ds:
        if seen >= n: break
        labels = ex["choices"]["label"]; texts = ex["choices"]["text"]
        answer = str(ex["answerKey"])
        if answer not in labels: continue
        prompt = "Question: " + ex["question"] + "\nAnswer:"
        scores = [score_completion(model, tokenizer, prompt, " " + t, device) for t in texts]
        pred_i = int(np.argmax(scores)); pred = labels[pred_i]
        correct += pred == answer; seen += 1
        if seen <= 10: details.append({"pred": pred, "label": answer, "scores": scores})
    return {"accuracy": correct / max(1, seen), "correct": correct, "examples": seen, "details_first10": details}


@torch.no_grad()
def eval_boolq(model, tokenizer, device: torch.device, n: int = 100):
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    correct = 0; seen = 0; details=[]
    for ex in ds:
        if seen >= n: break
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nAnswer:"
        opts = [" no", " yes"]
        scores = [score_completion(model, tokenizer, prompt, o, device) for o in opts]
        pred = bool(int(np.argmax(scores))); label = bool(ex["answer"])
        correct += pred == label; seen += 1
        if seen <= 10: details.append({"pred": pred, "label": label, "scores": scores})
    return {"accuracy": correct/max(1,seen), "correct":correct, "examples":seen, "chance":0.5, "details_first10":details}


@torch.no_grad()
def prefill_benchmark(model, cfg: FlashSkipConfig, device: torch.device, lengths=(128,512,1024), iters=20):
    out = {}
    model.eval()
    for t in lengths:
        x = torch.randint(0, cfg.vocab_size, (1, t), device=device)
        for _ in range(4):
            with autocast_ctx(device): model(x)
        if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(iters):
            ts=time.perf_counter()
            with autocast_ctx(device): model(x)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter()-ts)*1000)
        out[str(t)]={"median_ms":statistics.median(times), "p95_ms":sorted(times)[max(0,int(.95*len(times))-1)],
                     "tokens_per_second_equiv":t/(statistics.median(times)/1000),
                     "peak_cuda_MiB":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None}
    return out


def evaluate(args, cfg: FlashSkipConfig):
    output_dir=Path(args.output_dir); data_dir=Path(args.data_dir)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, cfg=load_final_model(output_dir, device)
    val_tokens=memmap_tokens(data_dir/"val_tokens.uint16")
    results={"device":device_info(), "parameters":sum(p.numel() for p in model.parameters()), "parameter_breakdown":exact_parameter_breakdown(cfg)}
    results["fineweb_edu_heldout"]=evaluate_memmap_loss(model,val_tokens,device,512,64,2)
    for name,fn in [
        ("wikitext2",lambda:eval_wikitext(model,tokenizer,device,args.wikitext_tokens,512)),
        ("hellaswag",lambda:eval_hellaswag(model,tokenizer,device,args.eval_examples)),
        ("arc_easy",lambda:eval_arc_easy(model,tokenizer,device,args.eval_examples)),
        ("boolq",lambda:eval_boolq(model,tokenizer,device,args.eval_examples)),
    ]:
        try:
            print("EVAL_START",name,flush=True); results[name]=fn(); print("EVAL_DONE",name,json.dumps(results[name],default=str)[:1000],flush=True)
        except Exception as e:
            results[name]={"error":f"{type(e).__name__}: {e}"}; print("EVAL_ERROR",name,results[name],flush=True)
    results["prefill_benchmark_no_kv_cache"]=prefill_benchmark(model,cfg,device)
    prompts=[
        "The capital of France is",
        "Photosynthesis is the process by which plants",
        "In Python, a function that returns the Fibonacci sequence can be written as",
        "If Alice has 3 apples and buys 4 more, she has",
        "A good explanation of gravity for a student is",
    ]
    samples=[]
    for p in prompts:
        ids=tokenizer.encode(p,return_tensors="pt",add_special_tokens=False).to(device)
        out=model.generate(ids,max_new_tokens=args.generation_tokens,temperature=.8,top_k=50)
        text=tokenizer.decode(out[0].tolist(),skip_special_tokens=False)
        samples.append({"prompt":p,"text":text})
        print("SAMPLE",repr(text[:500]),flush=True)
    results["samples"]=samples
    eval_dir=output_dir/"evaluation"; eval_dir.mkdir(parents=True,exist_ok=True)
    json_dump(eval_dir/"results.json",results)
    report=["# FlashSkip-100M Pilot Evaluation","",f"Parameters: **{results['parameters']:,}**",f"Device: `{results['device'].get('device')}`","",
            "## Language modeling",f"- FineWeb-Edu held-out: loss {results['fineweb_edu_heldout']['loss']:.4f}, perplexity {results['fineweb_edu_heldout']['perplexity']:.2f}"]
    if "perplexity" in results.get("wikitext2",{}): report.append(f"- WikiText-2: loss {results['wikitext2']['loss']:.4f}, perplexity {results['wikitext2']['perplexity']:.2f}")
    report += ["","## Downstream probes"]
    for k in ["hellaswag","arc_easy","boolq"]:
        v=results.get(k,{})
        report.append(f"- {k}: " + (f"accuracy {v['accuracy']:.3f} ({v.get('correct')}/{v.get('examples')})" if "accuracy" in v else f"ERROR {v.get('error')}"))
    report += ["","## Scope warning","This is a short tens-of-millions-token pilot. A 99M model normally requires orders of magnitude more pretraining to make downstream benchmark scores representative. The probe results measure learning progress, not final architecture capability.","","## Prefill benchmark (full-sequence, no KV cache)","```json",json.dumps(results["prefill_benchmark_no_kv_cache"],indent=2),"```","","## Samples"]
    for s in samples: report += [f"### {s['prompt']}","```",s["text"],"```"]
    (eval_dir/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("EVALUATION_COMPLETE",json.dumps({k:v for k,v in results.items() if k not in {'samples'}},default=str)[:5000],flush=True)
    return results


# ----------------------------- smoke / verify / package -----------------------------

def smoke(args, cfg: FlashSkipConfig):
    seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FlashSkipLM(cfg).to(device)
    actual=sum(p.numel() for p in model.parameters()); expected=exact_parameter_breakdown(cfg)["total"]
    if actual!=expected: raise RuntimeError((actual,expected))
    b=args.micro_batch; t=args.seq_len
    # smoke can be lowered via CLI; use real vocab CE to expose memory pressure.
    x=torch.randint(0,cfg.vocab_size,(b,t),device=device); y=torch.randint(0,cfg.vocab_size,(b,t),device=device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats()
    ts=time.perf_counter(); opt.zero_grad(set_to_none=True)
    with autocast_ctx(device):
        logits=model(x); loss=F.cross_entropy(logits.reshape(-1,cfg.vocab_size),y.reshape(-1))
    loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)); opt.step()
    if device.type=="cuda": torch.cuda.synchronize()
    result={"parameters":actual,"loss":float(loss),"grad_norm":grad,"seconds":time.perf_counter()-ts,"batch":b,"seq":t,**device_info(),
            "peak_cuda_GiB":torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else None}
    print("SMOKE_OK",json.dumps(result,sort_keys=True),flush=True)
    return result


def verify(args, cfg: FlashSkipConfig):
    output_dir=Path(args.output_dir); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,tokenizer,cfg=load_final_model(output_dir,device)
    x=torch.tensor(tokenizer.encode("A short verification sentence",add_special_tokens=False)[:32],device=device)[None,:]
    with torch.no_grad(),autocast_ctx(device): logits=model(x)
    if not torch.isfinite(logits).all(): raise RuntimeError("nonfinite logits")
    state=torch.load(output_dir/"final"/"training_state.pt",map_location="cpu",weights_only=False)
    required={"step","tokens_seen","data_cursor","optimizer_state_half_disk"}
    missing=sorted(required-set(state))
    if missing: raise RuntimeError(f"resume fields missing: {missing}")
    result={"ok":True,"parameters":sum(p.numel() for p in model.parameters()),"logits_shape":list(logits.shape),
            "finite":True,"resume_step":state["step"],"resume_tokens":state["tokens_seen"],"optimizer_state_entries":len(state["optimizer_state_half_disk"]["state"]),"device":str(device)}
    json_dump(output_dir/"verification.json",result); print("VERIFY_OK",json.dumps(result,sort_keys=True),flush=True); return result


def package(args):
    output_dir=Path(args.output_dir).resolve(); archive=Path(args.archive).resolve()
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists(): archive.unlink()
    # gz trades some CPU for smaller transfer; hardlinks keep final/checkpoint duplicate compact.
    with tarfile.open(archive,"w:gz",compresslevel=3) as tar:
        tar.add(output_dir,arcname="models",recursive=True)
    result={"archive":str(archive),"bytes":archive.stat().st_size,"sha256":sha256_file(archive)}
    json_dump(output_dir/"package_manifest.json",result)
    print("PACKAGE_OK",json.dumps(result,sort_keys=True),flush=True); return result


def make_parser():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["smoke","prepare","train","evaluate","verify","package"])
    p.add_argument("--output-dir",default="/content/flashskip/models")
    p.add_argument("--data-dir",default="/content/flashskip/data")
    p.add_argument("--tokenizer",default="HuggingFaceTB/SmolLM2-135M")
    p.add_argument("--seed",type=int,default=20260830)
    p.add_argument("--seq-len",type=int,default=512)
    p.add_argument("--micro-batch",type=int,default=8)
    p.add_argument("--grad-accum",type=int,default=8)
    p.add_argument("--target-tokens",type=int,default=20_000_000)
    p.add_argument("--prepare-train-tokens",type=int,default=22_000_000)
    p.add_argument("--prepare-val-tokens",type=int,default=1_000_000)
    p.add_argument("--checkpoint-tokens",type=int,default=5_000_000)
    p.add_argument("--checkpoint-origin-tokens",type=int,default=0)
    p.add_argument("--max-lr",type=float,default=6e-4)
    p.add_argument("--min-lr",type=float,default=6e-5)
    p.add_argument("--warmup-fraction",type=float,default=.05)
    p.add_argument("--schedule-start-step",type=int,default=0)
    p.add_argument("--schedule-target-tokens",type=int,default=0)
    p.add_argument("--weight-decay",type=float,default=.1)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--eval-batches",type=int,default=24)
    p.add_argument("--eval-batch-size",type=int,default=2)
    p.add_argument("--resume",default="")
    p.add_argument("--require-resume-cursor-at-least",type=int,default=0)
    p.add_argument("--allow-cpu-train",action="store_true")
    p.add_argument("--data-parallel",action="store_true")
    p.add_argument("--force-prepare",action="store_true")
    p.add_argument("--eval-examples",type=int,default=100)
    p.add_argument("--wikitext-tokens",type=int,default=200_000)
    p.add_argument("--generation-tokens",type=int,default=48)
    p.add_argument("--archive",default="/content/flashskip/models_bundle.tar.gz")
    return p


def main():
    args=make_parser().parse_args(); cfg=FlashSkipConfig()
    breakdown=exact_parameter_breakdown(cfg)
    print("CONFIG",json.dumps(cfg.to_jsonable(),sort_keys=True),flush=True)
    print("PARAMETER_BREAKDOWN",json.dumps(breakdown,sort_keys=True),flush=True)
    if args.mode=="smoke": smoke(args,cfg)
    elif args.mode=="prepare": prepare_data(args,cfg)
    elif args.mode=="train": train(args,cfg)
    elif args.mode=="evaluate": evaluate(args,cfg)
    elif args.mode=="verify": verify(args,cfg)
    elif args.mode=="package": package(args)


if __name__=="__main__":
    main()
