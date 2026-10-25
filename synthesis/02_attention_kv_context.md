# Attention, KV Cache, and Context Synthesis

## Core objective
At <500M parameters and batch-1 device inference, KV-cache architecture can become as important as stored weight size. The target is not the longest nominal context; it is the most useful context that preserves memory, TTFT and steady decode speed.

## 1. Use GQA by default
For `L` KV-producing layers, context length `T`, `Hkv` KV heads, head dimension `dh`, and cache precision `b` bits:

`KV_bytes = 2 × L × T × Hkv × dh × b / 8`.

With 20 Q heads / 4 KV heads / head size 64, one 8-bit KV-producing layer consumes `512 bytes/token`; 4-bit consumes `256 bytes/token`. This makes attention-layer count itself a direct memory knob.

The winning FlashSkip candidate has only nine KV-producing layers:
- KV8 @ 8K ≈ 36 MiB;
- KV4 @ 8K ≈ 18 MiB.

A conventional 36-layer 16Q/4KV deep model in our table needs ~72 MiB KV4 at 8K. The difference is structural and requires no exotic cache compressor.

## 2. Attention skipping versus local attention
MobileLLM-Flash provides direct modern phone evidence that **skip-attention** can outperform sliding-window attention on the tested mobile CPU runtime. A block without attention still has a full SwiGLU FFN, so it can perform rich token-wise nonlinear transformations; global mixing is refreshed by nearby attention-bearing blocks.

The production pattern is therefore nine attention blocks across sixteen physical blocks, with no long attention-free run:

`attention layers = [0, 1, 3, 5, 7, 9, 11, 13, 15]` (zero-indexed).

This deliberately begins and ends with attention and then mostly alternates attention/FFN-only blocks. It is a project hypothesis derived from the MobileLLM-Flash design rule; exact indices must be searched on target hardware.

### Why keep a local/global candidate anyway?
A 5:1 local/global stack bounds cache growth more aggressively for 32K+ context and can preserve attention mixing in every layer. Gemma-style local/global stacks, MobileLLM-Pro and Opt.Gear provide supporting evidence. Our LocalGlobal-488 candidate uses 20 local layers with a 2K window and four global layers; its ideal KV4 payload is ~42 MiB even at 32K, much smaller than a full-global stack.

The Colab proxy also gave LocalGlobal the lowest held-out loss of the five tiny proxies, although the margin over EdgeRec was only ~0.006 and is not statistically decisive.

## 3. Cross-layer KV sharing
Cross-Layer Attention (CLA) reuses K/V across adjacent layers. Our CLA-Cache-491 candidate contains 35 attention transformations but only 18 KV-producing layers. This cuts cache roughly in half relative to 35 independent KV sets:
- ideal KV4 @ 8K ≈ 36 MiB;
- ideal KV4 @ 16K ≈ 72 MiB.

The benefit is attractive, but the deployment graph differs from a stock self-attention layer because a follower layer consumes a prior layer’s K/V. If the runtime materializes/copies tensors or loses fused attention, theoretical savings may not translate to latency. This is why CLA ranks below the skip-attention production candidate.

## 4. Recursion-specific caching
For EdgeRec-482, the base physical stack contains 25 KV-producing layers. The optional repeated middle span is six layers.

If 35% of tokens receive one extra pass through that six-layer span under recursion-wise caching, the effective cache layer-equivalents are approximately:

`25 + 0.35×6 = 27.1`.

Relative to the base candidate’s ideal KV4 @ 8K (50 MiB), this is ~54.2 MiB. At a 50% route budget it is ~56 MiB. First-pass KV sharing can avoid the additional stored depth states, but deeper queries still reread the full shared cache and therefore may not improve decode IO.

**Decision:** recursion-wise cache is the research-quality default; first-pass shared KV is the RAM/TTFT experiment.

## 5. KV precision ladder
### Production baseline: KV8
- easiest to support on current mobile runtimes;
- stable accuracy baseline;
- straightforward integer storage.

### Primary target: packed KV4
- approximately halves cache bytes and cache bandwidth versus KV8;
- only a latency win if attention consumes packed cache directly or fuses unpack/dequant efficiently;
- should be introduced with QAT or cache-aware recovery if PTQ changes instruction/long-context behavior.

### Research only: 2–3 bit KV
KIVI/KVQuant show algorithmic feasibility, but metadata and unpack cost matter much more for a small model. Do not make KV2 the default until a target CPU/GPU/NPU kernel demonstrates lower p50/p95 decode latency, not just lower allocated RAM.

## 6. Context target
The production model is frozen at **8,192 native tokens** for v1. That choice is intentional:
- enough for device chat, many code edits and moderate retrieval prompts;
- keeps TTFT tractable for a 500M-class model;
- avoids allocating training/data budget to a context length the model may not use intelligently;
- makes KV memory modest with nine GQA layers.

Longer context is a separate extension gate:
1. train/continue at 16K or 32K;
2. pass perplexity and retrieval/RULER-style evaluations;
3. verify KV memory/TTFT on target devices;
4. only then advertise the extended window.

## 7. RoPE and QK normalization
Use QK-RMSNorm before RoPE/dot products. Native pretraining begins at 4K, then continues at 8K. A RoPE base around `1e6` is a starting value, not a sacred constant; compare `1e4/1e5/1e6` in small predictive runs and check short-context regression.

## 8. Cache acceptance criteria
For every export measure:
- real allocated bytes at T={1K,2K,4K,8K};
- slope of memory vs tokens;
- cache update operator count;
- whether cache updates are in-place;
- packed-vs-dequantized representation inside attention;
- TTFT and p50/p95 decode separately;
- long-generation stability and retrieval accuracy.

A model does not “have a 4-bit KV cache” in the systems sense if it unpacks the whole cache into FP16 for each attention call.
