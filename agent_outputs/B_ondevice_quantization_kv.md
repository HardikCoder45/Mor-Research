# Track B — On-device inference efficiency for a strictly <500M-parameter language model

Research date: 2026-08-30  
Scope: batch-1 interactive inference on phones, tablets, laptops, and embedded/edge SoCs.  
Constraint: **strictly fewer than 500,000,000 total learned parameters**, including embeddings, routers, adapters, and any other learned deployment-time parameters.

Source IDs `[Sxx]` map to `B_sources.tsv`. Results quoted from papers are marked **Observed**; calculations and deployment choices derived here are marked **Engineering inference**.

---

## 1. Executive recommendation

For a new sub-500M on-device LM, I would optimize in this order:

1. **Architect the cache first:** use GQA with 4–8 KV heads (or MQA only if quality survives), then bound most attention with a local/sliding window. Cache growth, not just weight size, determines whether long interactive sessions remain fast. MQA was introduced specifically to reduce incremental-decoding KV bandwidth [S02]; GQA recovers much of MHA quality while retaining most of the bandwidth benefit [S03].
2. **Use a hardware-native W4 path before exotic <4-bit formats.** Default to groupwise INT4 weights with FP16/BF16 or dynamically quantized INT8 activations, choosing the exact format from the target backend. AWQ/GPTQ remain reliable PTQ starting points [S12,S13]. On current ExecuTorch/XNNPACK, `8da4w` is a concrete mobile CPU path; Qualcomm QNN exposes `16a4w`/`16a4w_block`; Core ML has INT4 per-block weight compression on newer OS versions; Vulkan has W4 and 8da4w linear paths [S26-S30].
3. **Quantize KV to 8-bit by default, 4-bit when context is material, and 2–3 bits only after quality + kernel validation.** KIVI shows tuning-free 2-bit K/V is feasible with asymmetric granularity [S07], KVQuant shows high-quality 3-bit KV plus custom kernels [S08], and newer work improves layer/block allocation [S09-S11]. But a packed 2-bit cache that is unpacked into FP16 by a slow generic kernel can save RAM while losing latency.
4. **Prefer local/global attention over an unnecessarily large global cache.** A fixed local window changes local-layer KV memory from `O(T)` to `O(W)`. Mistral demonstrated practical SWA [S05]; Gemma 3 moved to a 5:1 local:global pattern specifically to reduce KV-cache growth [S06]. For a <500M model, a 1K–2K local window with sparse global layers is a strong starting point.
5. **Treat MoR as a parameter-capacity technique plus adaptive-compute technique, not as a free latency win.** MoR reuses a shared stack and routes tokens to different recursive depths [S01]. Its recursion-wise cache reduces KV memory/IO in the recursive stack to `(N_r+1)/(2N_r)` of the unrolled vanilla stack, while first-recursion KV sharing reaches `1/N_r` memory but normalized KV-cache IO remains 1 [S01]. On batch-1 devices, repeated recursion can reread the same weights from DRAM; the server-oriented continuous depth-wise batching speedup does not automatically transfer to a single user/session.
6. **Do not use unstructured sparsity unless the deployed kernel consumes it natively.** SparseGPT/Wanda demonstrate algorithmic sparsity [S23,S24], but zero weights in a dense GEMM still cost dense latency. Structured channel/head/layer removal is useful when it produces smaller dense tensors; N:M sparsity is useful only if the target backend has a matching sparse kernel.
7. **Benchmark TTFT and decode separately.** Prefill is increasingly compute/attention-bound; decode at batch 1 is usually weight- and KV-bandwidth-bound. A quantizer that wins model size but adds unpack/dequant kernels can improve neither metric.

### Recommended starting design points

| Target | Weights | Activations | KV | Attention | Routing/recursion | Why |
|---|---:|---:|---:|---|---|---|
| Arm CPU, broad compatibility | W4 group 32–128 | dynamic A8 or FP16 | KV8 → KV4 after validation | GQA + 1K–2K local window | no dynamic recursion initially; add token-choice `N_r<=3` only if loop saves real work | Native-ish integer paths and predictable memory traffic matter more than theoretical compression |
| Mobile GPU / Metal / Vulkan | W4 per-group/per-block | FP16 or dynamic A8 where integer-dot kernel wins | KV4/KV8 | GQA + local/global | static or coarse token-choice recursion | Fused dequant+matmul and packed KV attention are decisive |
| Qualcomm NPU/HTP | QNN-supported W4A16 block or W8A8 | backend-native | KV8 first | GQA; static bounded cache | prefer fixed-depth or compile a small set of recursion profiles | QNN documents these quant schemes; dynamic routing/compaction can break delegation [S27] |
| Apple Neural Engine | test W8A8 and palettization; compare against W4 GPU | backend-native | stateful KV; quantization must be benchmarked | bounded cache; stateful model | avoid fine-grained dynamic control flow unless compiler keeps it on ANE | Core ML explicitly notes compression latency depends on decompression strategy and compute unit [S30,S31] |
| Memory-minimum experimental | 2–3-bit vector/codebook or hardware-aligned mixed precision | FP16/A8 | 2–4 bit | local/global | MoR parameter sharing | QuIP#/AQLM/QTIP/ScaleBITS can beat scalar PTQ on rate-distortion, but kernel availability is the gating constraint [S16,S17,S20,S21] |

---

## 2. Quantitative resource model

### 2.1 Weight footprint

For `P` quantized learned parameters and nominal weight precision `b_w` bits,

```text
M_weights_raw = P * b_w / 8                       [bytes]
```

For groupwise affine/symmetric quantization with group size `G`, `s` bytes of scale metadata and `z` bytes of zero-point metadata per group,

```text
M_weights ~= P*b_w/8 + (P/G)*(s+z)
b_eff      = b_w + 8*(s+z)/G                      [bits/weight]
```

With symmetric W4, group 128, and one FP16 scale (`s=2`, `z=0`), `b_eff = 4 + 16/128 = 4.125` bits/weight. With group 32, the same scale overhead is 0.5 bit/weight, so rate is 4.5 bits/weight. Smaller groups can improve accuracy but are not free.

**Strict 500M ceiling arithmetic** (raw payload, before scales/codebooks/runtime buffers):

| Total params | FP16 | W8 | W4 | W3 | W2 |
|---:|---:|---:|---:|---:|---:|
| 450M | 900 MB | 450 MB | 225 MB | 168.75 MB | 112.5 MB |
| 480M | 960 MB | 480 MB | 240 MB | 180 MB | 120 MB |
| 499M | 998 MB | 499 MB | 249.5 MB | 187.1 MB | 124.75 MB |

A 480M model at symmetric W4/group-128 is about **247.5 MB decimal (236.0 MiB)** including FP16 scales. This is why W4 is the practical default: it leaves hundreds of MB of headroom for KV, activations, tokenizer/runtime, graphics, and the application itself.

### 2.2 KV-cache footprint: GQA/MQA

For a decoder with `L` attention layers, sequence length `T`, `H_kv` KV heads, head dimension `d_h`, and KV precision `b_kv` bits,

```text
M_KV(T) = 2 * L * T * H_kv * d_h * b_kv / 8      [bytes]
            ^ K+V
```

Equivalently, for `H_q` query heads and `d_model = H_q*d_h`,

```text
M_KV(T) = 2 * L * T * d_model * (H_kv/H_q) * b_kv/8
```

Thus, relative to MHA (`H_kv = H_q`):

```text
GQA cache ratio = H_kv/H_q
MQA cache ratio = 1/H_q
```

This is the core MQA/GQA bandwidth advantage [S02,S03]. It reduces K/V projection parameters and cache traffic; it **does not eliminate the query-head attention dot products**, because all query heads still score keys.

#### Worked example: 24 layers, d_model=1024, Hq=16, Hkv=4

Here `d_h=64`. Cache payload per token is:

```text
BF16/FP16: 2*24*4*64*2 bytes = 24 KiB/token
KV8:                              12 KiB/token
KV4:                               6 KiB/token
```

At 4K context this is approximately 96 MiB (16-bit), 48 MiB (8-bit), or 24 MiB (4-bit). MQA (`H_kv=1`) would quarter those GQA figures, but GQA is usually a safer quality/performance compromise [S03].

### 2.3 Sliding/local attention caps cache growth

If `L_local` layers retain only a window `W` while `L_global` layers retain full length `T`,

```text
M_KV_hybrid ~= 2*H_kv*d_h*b_kv/8 * (L_local*min(T,W) + L_global*T)
```

A 5:1 local/global stack therefore makes only roughly one-sixth of layer caches grow with the full context once `T >> W`. Gemma 3 uses a 5:1 local:global interleave and a short local span specifically to reduce long-context KV memory [S06]. Mistral's SWA provides a simpler all-local reference [S05].

If the application is conversational rather than document-scale, an attention-sink scheme can retain a few initial sink tokens plus the recent window; StreamingLLM showed that naïve window eviction can fail and that retaining initial sink states stabilizes streaming [S04]. ExecuTorch exposes both `use_attention_sink` and local/global attention options [S26].

### 2.4 Decode bandwidth roofline

At batch 1, each generated token often streams a large fraction of the quantized weights plus the participating KV cache. A useful first-order lower bound on token time is

```text
t_decode >= max(
    (B_weight_step + B_KV_read(T) + B_misc) / BW_effective,
    FLOPs_step / FLOPs_effective
) + t_launch/control

tokens/s <= BW_effective / (B_weight_step + B_KV_read + B_misc)
```

where `BW_effective` is *measured sustained bandwidth for this inference kernel*, not peak DRAM marketing bandwidth.

For a dense 480M model at raw W4, `B_weight_step` is already about 240 MB/token if the weights are much larger than on-chip cache and are reread each decode step. Ignoring all other costs, bandwidth-only ceilings are:

| Sustained effective BW | W4 480M weights-only ceiling |
|---:|---:|
| 20 GB/s | 83 tok/s |
| 40 GB/s | 167 tok/s |
| 80 GB/s | 333 tok/s |

These are **upper bounds, not expected device results**. Add KV reads, dequant/unpack, nonlinearities, output head, kernel gaps, synchronization, thermal throttling, and bandwidth contention, and actual throughput falls. This model is useful because it exposes when a change is incapable of helping: e.g. halving arithmetic without reducing bytes does little in a memory-bound decode kernel.

For the illustrative 24-layer GQA example at 4K context, KV4 adds about 24 MiB of cache payload; KV16 adds about 96 MiB. At 20 GB/s, even idealized W4+KV4 traffic is already materially below the 83 tok/s weights-only ceiling.

### 2.5 TTFT / prefill model

For dense Transformer linear projections, a standard forward approximation is roughly two FLOPs per applied parameter per input token:

```text
F_linear_prefill ~= 2 * N * P_applied
```

`P_applied` is important for recursive networks: it counts **parameter applications**, not unique stored parameters.

For causal full attention, counting QK and AV matmuls,

```text
F_attn_full ~= 2 * L * N^2 * d_model
```

For a local window `W << N`,

```text
F_attn_local ~= 4 * L * N * W * d_model
```

(the full causal triangle has about `N^2/2` query-key pairs; a mature local region has about `N*W`).

For a 480M dense model, linear work alone is about 0.492 TFLOP for a 512-token prompt, 1.97 TFLOP for 2K, and 7.86 TFLOP for 8K. For `L=24,d=1024`, full-attention work adds about 0.206 TFLOP at 2K and 3.30 TFLOP at 8K. At 8K, a 2K local window reduces that attention term to roughly 1.65 TFLOP. This is why long-context TTFT eventually becomes attention-sensitive even when the model is small.

Do not convert these FLOP counts directly into wall-clock time with peak TOPS. Use measured effective FLOP/s for the exact precision and operator graph.

---

## 3. MoR (arXiv:2507.10524): what carries over to on-device inference

### 3.1 What the paper actually does

Mixture-of-Recursions (MoR) combines [S01]:

- **weight sharing:** a stack of layers is reused for multiple recursion steps;
- **token-level routing:** tokens receive different recursive depths;
- **depth-aware KV caching:** cache only states needed at each recursion depth, or share first-recursion K/V across depths.

The paper studies expert-choice routing (depth experts select top-k tokens) and token-choice routing (each token selects its depth up front). Under balanced `N_r`-way token choice, recursion `r` processes

```text
a_r = (N_r-r+1)/N_r
```

of tokens, so `sum_r a_r = (N_r+1)/2`. For `N_r=3`, active fractions are `{1, 2/3, 1/3}` and the mean recursion depth is 2.

### 3.2 Do not confuse the paper's non-embedding count with total parameters

The paper's 360M base configuration has [S01]:

```text
32 layers, d_model=960, 15 query heads, 5 KV heads, d_head=64,
47M embedding parameters, 315M non-embedding parameters.
```

For Middle-Cycle with `N_r=3`, MoR reports **118M non-embedding unique parameters**. Under this project's strict counting rule, add the 47M embeddings: the deployed unique total is therefore roughly **165M**, plus tiny router overhead if not already included in the reported count. It is safely below 500M, but it is not a “118M total-parameter” model.

This configuration is particularly relevant because it combines recursion with GQA (`H_kv/H_q = 5/15 = 1/3`).

### 3.3 MoR KV memory formulas

Let `L_s` be the number of layers in the shared recursive stack and `L_e` the number of non-shared edge layers. An unrolled `N_r`-recursion baseline has KV layer-equivalents

```text
C_vanilla = L_e + L_s*N_r
```

For recursion-wise caching,

```text
C_rw = L_e + L_s * sum_r a_r
     = L_e + L_s*(N_r+1)/2         [balanced routing]

ratio_full = C_rw / C_vanilla
```

Inside the recursion block alone this becomes the paper's `(N_r+1)/(2N_r)` ratio [S01].

For first-recursion KV sharing,

```text
C_share = L_e + L_s
ratio_full = C_share / C_vanilla
```

Inside the recursive stack, memory ratio is the paper's `1/N_r` [S01].

#### Worked MoR cache: the paper's 360M-base, N_r=3 Middle-Cycle

The effective depth is `1 + 10*3 + 1 = 32`; use `H_kv=5,d_h=64`.

- vanilla/unrolled: 32 KV layer-equivalents;
- recursion-wise: `2 + 10*(1+2/3+1/3) = 22` layer-equivalents;
- recursive sharing: `2 + 10 = 12` layer-equivalents.

At 2K context:

| Cache strategy | KV16 | ideal KV4 payload |
|---|---:|---:|
| unrolled 32-layer cache | 80 MiB | 20 MiB |
| recursion-wise MoR | 55 MiB | 13.75 MiB |
| first-recursion shared KV | 30 MiB | 7.5 MiB |

At 8K, multiply by four: 320/220/120 MiB at 16-bit, or 80/55/30 MiB at ideal 4-bit.

The values exclude quant scales/zeros, page tables, padding, and allocator fragmentation.

### 3.4 Memory saving is not the same as decode-IO saving

This is MoR's most important on-device nuance. In Table 2, the recursive stack's normalized costs are [S01]:

| Strategy | KV memory | KV cache IO | attention FLOPs |
|---|---:|---:|---:|
| recursion-wise cache | `(N_r+1)/(2N_r)` | `(N_r+1)/(2N_r)` | `(k/N_ctx)^2` per selected layer context |
| recursive KV sharing | `1/N_r` | `1` | `k/N_ctx` |

Recursive sharing is therefore a **RAM/prefill optimization**, not automatically a decode-bandwidth optimization: deeper queries still attend to the full first-recursion cache. The paper explicitly notes that high KV IO can remain a decoding bottleneck [S01].

For mobile interactive use, this makes recursion-wise caching the default if decode latency matters and RAM is adequate. First-recursion sharing is attractive when **TTFT or hard RAM capacity** is the primary problem.

### 3.5 Parameter sharing shrinks resident weights, but recursion can reread them

Let `B_s` be the shared stack's quantized weight bytes, `B_e` non-recursive edge bytes, and `R_t` the recursion depth selected for the newly generated token. If the shared stack does not fit on-chip,

```text
B_weight_step(t) ~= B_e + R_t * B_s
E[B_weight_step] ~= B_e + E[R] * B_s
```

This is different from resident model size `B_e+B_s`.

For balanced `N_r=3`, `E[R]=2`. So a shared block that saves 3x in **storage** can still be streamed about twice per average decode token. On a phone CPU/GPU, a tens-of-MB recursive stack is much larger than the private caches and usually larger than reusable on-chip SRAM, so assuming “read once, execute three times for free” is unsafe.

This is also why the paper's reported **up-to-2.06x** throughput improvement for MoR-4 should not be transplanted into a batch-1 mobile estimate. That experiment uses continuous depth-wise batching and batch sizes around 32 / maximum batch on GPU [S01]. A single interactive session has no queue of other sequences to fill vacated depth slots.

### 3.6 Routing choice for edge deployment

**Expert-choice + recursion-wise cache:** strongest paper-backed quality/efficiency default, but has the hardest systems path. Selection is reconsidered at each depth and needs top-k/capacity handling; the paper uses an auxiliary mechanism to address causality at inference [S01]. Dynamic token compaction can cause graph breaks or CPU fallback on NPU/static runtimes.

**Token-choice:** one up-front depth decision is easier to schedule. At batch 1, the server-side load-balancing drawback matters much less. If the runtime supports a genuine conditional loop, token-choice is the more deployment-friendly MoR form. The paper's token-choice quality is weaker in its studied settings, and recursive KV sharing can partly compensate [S01].

**Static fallback:** if the backend cannot skip work, a router that merely masks outputs while all recursion blocks execute saves no matrix-multiply time. In that case either (a) use fixed recursion, (b) compile 1/2/3-depth subgraphs and dispatch between them, or (c) omit routing.

### 3.7 Recommended MoR experiment for this project

A good test point is `N_r=3`, Middle-Cycle, GQA, with both cache modes:

- deployment A: expert-choice + recursion-wise cache for the quality reference;
- deployment B: token-choice + recursion-wise cache for control-flow simplicity;
- deployment C: token-choice + first-recursion KV sharing for minimum RAM/TTFT;
- quantize weights W4 and evaluate KV8, KV4, then KV2/3;
- report **batch-1** TTFT and p50/p95 ms/token separately from large-batch throughput.

The key acceptance criterion is not “router reduced theoretical FLOPs”; it is “router reduced wall-clock work without delegation loss or extra cache traffic erasing the gain.”

---

## 4. KV-cache quantization and compression

### 4.1 KIVI and KVQuant: complementary lessons

**KIVI [S07] — Observed.** KIVI finds that key tensors are better quantized per-channel while values are better per-token, and implements tuning-free asymmetric 2-bit KV. The paper reports 2.6x lower *peak memory including weights*, up to 4x larger batch, and 2.35–3.47x throughput on its LLM serving workloads. The lesson for edge is not the batching result; it is the **different statistical axis for K vs V** and the feasibility of 2-bit cache with an appropriate kernel.

**KVQuant [S08] — Observed.** KVQuant combines per-channel keys, pre-RoPE key quantization, non-uniform layer-aware datatypes, and dense+sparse outlier handling. It reports <0.1 perplexity degradation at 3-bit KV on tested models and up to ~1.7x custom-kernel speedup over FP16 matrix-vector baselines. The edge lesson is that **packing + attention kernel integration** matters as much as nominal bitwidth.

### 4.2 Stronger/recent work

- **KVTuner (2025) [S09]:** offline, hardware-friendly layer-wise mixed precision. It reports near-lossless average 3.25-bit settings on some models and up to 38.3% throughput improvement over KV8. This is attractive for edge because the precision decision is static at runtime.
- **KVLinC (2025) [S10]:** Hadamard rotation plus small linear correction for extreme low-bit KV; includes a custom attention kernel. Promising, but the correction path adds operators that must fuse/delegate on the target runtime.
- **Adaptive on-device KV quantization (2026) [S11]:** directly evaluates lightweight SmolLM-135M/360M/1.7B and dynamically selects {2,4,8,16} bits by token features. For SmolLM-360M/HellaSwag it reports 17.75% lower decode latency than static KV quantization while remaining 0.30 accuracy points from FP16. This is unusually relevant to the target scale, but the learned controller and variable layouts can complicate kernels.
- **Block-GTQ (2026) [S32]:** RoPE-aware bit allocation by 2D frequency blocks; reports a packed K3V3 path 1.34x faster than FP16 FlashAttention2 at 128K on H800 and strong long-context quality. This is a good algorithmic direction, but the demonstrated kernel is a datacenter GPU kernel, not a mobile backend.

### 4.3 Practical KV precision ladder

For this project, optimize in this order:

1. **KV8 baseline:** easiest to support; Qualcomm ExecuTorch documentation already uses int8 KV cache in its Llama path [S27].
2. **KV4:** target when context exceeds ~2K–4K or RAM bandwidth is limiting. Prefer per-channel K / per-token V or a rotated/robust scheme if accuracy needs it.
3. **KV3/KV2:** only after a packed attention kernel exists on the selected backend. At these bitwidths, scale/zero/outlier metadata is a meaningful fraction of total bytes and nibble/bit unpack cost can dominate for a small model.
4. **Mixed precision:** use static layer/head/block allocations before token-dynamic bitwidths. Static metadata keeps the attention kernel regular and is friendlier to NPU/GPU compilation.

### 4.4 Recent-token residual buffers

Many low-bit KV methods preserve a recent FP16/FP8 residual window before quantizing older entries. That improves quality but means memory is

```text
M_KV = M_recent_high_precision + M_old_quantized + metadata
```

On a small model with short contexts, the residual can erase much of the theoretical compression. Measure the **real allocated cache at T={1K,2K,4K,8K}**, not only average bits/element.

---

## 5. Weight and activation quantization

### 5.1 W4 weight-only / W4A8 is the deployment sweet spot

**AWQ [S12].** Activation-aware scaling protects salient weight channels without backprop reconstruction. The MLSys paper is explicitly about on-device compression and reports >3x speedup versus its Hugging Face FP16 baseline on desktop and mobile GPUs. Its key edge advantage is regular, hardware-friendly low-bit weights.

**GPTQ [S13].** Approximate second-order PTQ gives strong 3/4-bit weight accuracy and is widely implemented. The original paper reports substantial GPU speedups with custom inference. It is a strong offline quantizer, but the serialized GPTQ format itself does not guarantee the phone runtime has the matching kernel.

**Recommendation:** choose the runtime first, then use AWQ/GPTQ-style calibration to produce the exact group size/layout its kernels consume. For a <500M model, the calibration problem is modest enough that a small QAT or distillation recovery pass is also realistic if PTQ loses quality.

### 5.2 W8A8 when the accelerator has fast integer GEMM

**SmoothQuant [S14]** moves activation outlier difficulty into weights via an equivalent rescaling and enables W8A8. It reports up to 1.56x speedup and 2x memory reduction on its platforms.

W8A8 can beat W4 weight-only on an accelerator where INT8×INT8 GEMM is deeply optimized, despite doubling weight bytes. Core ML specifically notes W8A8 latency benefits on Neural Engine hardware with faster int8 paths [S30]. Therefore, never pick W4 based only on file size: compare W4A16/W4A8 and W8A8 on target silicon.

### 5.3 Rotation-based end-to-end low-bit

**QuaRot [S18]** uses rotations to suppress outliers and quantizes weights, activations, and KV end-to-end to 4 bits. **SpinQuant [S19]** learns the rotations and reports a 2.9-point zero-shot gap to full precision for W4A4KV4 on Llama-2 7B, improving on random rotations.

These methods are compelling if the backend truly executes low-bit activation GEMMs and low-bit KV attention. If the runtime inserts dequantize-to-FP16 between every operator, they become mostly a storage technique.

### 5.4 Extreme 2–3-bit weight compression

- **QuIP [S15] / QuIP# [S16]:** incoherence processing plus lattice/vector codebooks; QuIP# targets <=4-bit PTQ and fast inference.
- **AQLM [S17]:** additive multi-codebook quantization; the paper reports up to ~4x layer speedup on its CPU implementation and ~30% GPU layer speedup, with very strong 2-bit rate-distortion.
- **QTIP [S20]:** trellis-coded high-dimensional quantization designed around hardware-efficient decoding.
- **ScaleBITS (2026) [S21]:** hardware-aligned block-wise mixed precision; claims better ultra-low-bit quality without runtime overhead.

For a strictly <500M model, these methods are best viewed as **RAM/flash-pressure tools** when W4 is still too large, or as research candidates for a custom kernel stack. At 480M, moving from effective 4.125 b/w to ~3.125 b/w saves about 60 MB decimal; moving to ~2.1 b/w saves another ~60 MB. That can be valuable, but it is smaller than the multi-GB savings that motivated many original LLM papers, so kernel complexity has a higher relative cost here.

### 5.5 CPU-specific low-bit kernels

T-MAC [S22] is useful evidence for the core systems issue: low-bit weights with high-precision activations often suffer because hardware lacks a native mixed-precision GEMM and frameworks dequantize first. T-MAC uses LUT-based kernels to consume low-bit weights directly and reports up to 4x throughput and 70% energy reduction versus llama.cpp on tested edge/CPU systems.

The recommendation is broader than T-MAC itself: **inspect the generated operator path**. If W3/W2 adds `unpack -> dequant -> FP16 GEMM`, W4 with a mature dot-product kernel can be faster and lower-energy despite larger weights.

---

## 6. Pruning and structured sparsity

SparseGPT [S23] and Wanda [S24] show that large pretrained transformers can tolerate substantial unstructured sparsity, and SparseGPT also studies N:M patterns. LLM-Pruner [S25] removes coupled structures rather than isolated weights.

For this sub-500M edge target:

- **Do not count unstructured zeros as a latency gain** unless the mobile kernel is sparse-aware.
- **Prefer structural pruning** that physically reduces dense matrix dimensions: remove whole FFN channels, attention heads/KV heads, or layers, then export ordinary smaller GEMMs.
- **Be cautious pruning KV heads:** reducing `H_kv` is valuable because it directly reduces cache bytes, but that is effectively an architectural move toward stronger GQA/MQA and must be retrained/recovered for quality.
- **N:M sparsity is backend-specific.** If a target NPU/GPU documents a native pattern, co-design around it; otherwise W4 dense is simpler.
- **A model trained from scratch at the final shape is preferable** when possible. Post-hoc pruning is most valuable when starting from a stronger pretrained checkpoint that materially improves the compressed result.

Core ML's optimization guidance is an example of why runtime specificity matters: it states pruning can yield latency gains on Neural Engine and CPU compute units [S30]. That is not permission to assume the same benefit on Vulkan, QNN, or generic Arm GEMM.

---

## 7. Local/sliding-window attention recommendations

### 7.1 Why local attention is unusually attractive below 500M

Weight quantization can make a 450–499M model a few hundred MB, after which an unbounded FP16 KV cache can become comparable to the weights. Local attention puts a hard cap on most cache layers without requiring exotic compression.

For a layer with GQA, replacing global `T` by window `W` gives an approximate memory and decode-KV-read reduction of

```text
min(1, W/T)
```

for that layer once the window is full.

### 7.2 Recommended pattern

Start with:

```text
local window W = 1024 or 2048
local:global layer ratio = 3:1 to 5:1
GQA ratio H_kv/H_q = 1/4 to 1/3
KV precision = 8 bit baseline, 4 bit target
```

Train this pattern from scratch or include it during continued pretraining. Do not take a full-attention checkpoint, truncate its cache at deployment, and assume quality is preserved.

A few global layers keep long-range retrieval paths; local layers dominate the stack so cache scales mostly with `W`. The Gemma 3 5:1 design is evidence that this trade is viable at much larger scale [S06], though the exact ratio must be revalidated below 500M.

For infinite/very long chat, preserve sink/prefix tokens or a compact global summary; StreamingLLM shows why pure recent-window eviction can destabilize attention [S04].

---

## 8. Runtime and operator constraints

### 8.1 ExecuTorch / Arm CPU / XNNPACK

Current ExecuTorch LLM export documentation [S26] exposes:

- KV cache and SDPA-with-KV-cache paths;
- int8 KV-cache quantization;
- attention sinks and local/global attention;
- TorchAO `8da4w` (dynamic INT8 activation + INT4 weight);
- low-bit Arm CPU kernels for int1–8 weights;
- backend delegation diagnostics.

The XNNPACK quantization docs [S28] identify 4-bit weights + dynamic 8-bit activations as an LLM path. This makes **W4A8 + KV8 + local attention** the lowest-risk Android/iOS CPU baseline in this ecosystem.

Practical constraints:

- keep group sizes/tensor dimensions aligned to kernel tile requirements;
- avoid custom router ops that force repeated backend transitions;
- use one fused SDPA/KV-cache operator rather than Python-style slice/concat/update graphs;
- preallocate fixed maximum cache or a supported ring-buffer layout to avoid per-token allocations.

### 8.2 Qualcomm QNN / Hexagon HTP / Adreno

ExecuTorch's QNN backend currently documents `8a8w`, `16a16w`, `16a8w`, `16a4w`, and `16a4w_block` plus per-channel/per-block customization [S27]. Its Llama tutorial uses mixed precision and int8 KV-cache compression [S27].

Therefore:

- begin with **W4A16 block or W8A8**, not an arbitrary AWQ/GGUF layout;
- keep KV8 until a lower-bit QNN attention op is proven end-to-end;
- favor static maximum context and static local-window state shapes;
- dynamic MoR token compaction/top-k is a high-risk delegation feature. A fixed recursion depth or a dispatch among separately compiled depth profiles is more likely to stay on accelerator.

### 8.3 Vulkan mobile GPU

ExecuTorch Vulkan supports W8/W4 quantized linear layers and documents both weight-only W4 and dynamic-A8/W4. It notes that dynamic activation quantization can improve latency by using accelerated integer dot-product instructions [S29].

Recommended benchmark matrix: `W4A16 vs W4A8`, group 32/64/128, with and without fused KV4 attention. Do not assume weight-only wins.

### 8.4 Apple Core ML / Metal / Neural Engine

Core ML stateful models (iOS 18/macOS 15+) support persistent state suitable for KV cache and can update it in place across autoregressive calls [S31]. Apple's optimization guidance says [S30]:

- palettization can work particularly well on Neural Engine;
- W8A8 can provide latency benefits on newer Neural Engine hardware;
- INT4 per-block weight quantization can work well on GPU;
- compressed weights may be decompressed ahead of time or just-in-time depending on backend, so model-size compression is not guaranteed latency compression.

For Apple targets, benchmark two distinct deployments rather than force one universal file:

1. **ANE:** W8A8 or palettized weights, stateful bounded KV;
2. **GPU/Metal:** INT4 per-block weights, stateful KV, optionally lower-bit cache if the attention graph remains fused.

### 8.5 llama.cpp / GGML

The 2026 llama.cpp feature matrix reports K-quants, KV-cache quantization, Flash Attention, and MoE architecture support across ARM NEON and Metal, while I-quants are marked slower than comparable K-quants on ARM/Metal [S33].

For a portable prototype, this strongly favors a mature K-quant around 4 bits over an IQ2/IQ3 format when latency is the target. Extreme I-quants are useful for capacity experiments but should not be the performance baseline.

### 8.6 MLC LLM

MLC LLM provides Android/iOS paths and a standard `q4f16` family of compiled models [S34]. It is useful when the goal is a single compilation stack across mobile GPUs. As with every compiler runtime, verify that custom recursion/router/cache operations lower into fused kernels rather than fallback operators.

---

## 9. Device-oriented design recommendations

### 9.1 Memory-first phone profile

Goal: fit comfortably beside an application with long conversations.

- total learned params: 300–480M **or** a smaller unique MoR model if recursion improves quality;
- W4 group 64–128 (or backend-native block W4);
- GQA `H_kv/H_q <= 1/4` if quality permits;
- 4-bit KV for local layers, 4–8 bit for global layers;
- 1K–2K local window on 75–85% of attention layers;
- 2K–4K full/global retained context unless the product explicitly needs more;
- preallocated ring buffers; no KV concatenation per token.

Avoid 2-bit weights until W4+KV4 still violates RAM/flash goals. A 480M W4 model is only ~250 MB plus metadata; the engineering complexity of 2-bit may not be justified.

### 9.2 Latency-first phone profile

Goal: lowest p50/p95 ms/token.

- choose the precision with the fastest **native GEMV/GEMM**, even if it is W8A8 rather than W4;
- keep `H_kv` small and cap local KV reads;
- use fused dequant+matmul and fused attention-cache-update;
- avoid dynamic per-token precision and frequent CPU↔GPU/NPU crossings;
- if using MoR, cap at `N_r=2 or 3`, choose token-choice or compiled depth profiles, and verify skipped recursions disappear in traces;
- quantize the vocabulary/output head too: for a small model, a large vocabulary matrix is a nontrivial fraction of decode bandwidth.

### 9.3 TTFT-first profile

Goal: fast prompt ingestion.

- local attention pays increasingly as prompt length grows;
- batch/chunk prefill enough to hit GEMM kernels rather than token-by-token GEMV;
- consider MoR first-recursion KV sharing: the paper notes it can skip deeper KV projection/prefill work in compatible sharing layouts [S01];
- use W8A8 if the NPU/ANE provides much higher effective integer GEMM throughput than W4 mixed precision;
- avoid per-token adaptive KV quantization during prefill if it serializes what should be a large matrix operation.

### 9.4 Quality-first under the same parameter ceiling

If quality is the primary objective while storage/RAM are still acceptable:

- GQA rather than MQA;
- W4 with conservative group size and selective higher precision for embedding/output/norm-sensitive layers;
- KV8 or mixed 4/8 before KV2;
- local/global rather than all-local;
- MoR expert-choice + recursion-wise cache as the research-quality reference;
- use distillation/QAT recovery rather than pushing every tensor to the same extreme precision.

---

## 10. Worked end-to-end example: MoR-derived 360M-base configuration

Using the paper's 360M base dimensions [S01]:

```text
d_model = 960
H_q     = 15
H_kv    = 5
head dim = 64
Middle-Cycle N_r=3
unique non-embedding params ~=118M
embedding params ~=47M
strict total unique params ~=165M (+ router bookkeeping)
```

### Resident weights

Raw W4 payload:

```text
165M * 4/8 ~= 82.5 MB
```

At group-128 with one FP16 scale, approximate payload is

```text
165M * 4.125/8 ~= 85.1 MB decimal
```

### KV at 2K

From Section 3:

```text
unrolled KV16       ~=80 MiB
recursion-wise KV16 ~=55 MiB
shared KV16         ~=30 MiB
recursion-wise KV4  ~=13.75 MiB ideal payload
shared KV4          ~=7.5 MiB ideal payload
```

So a W4 + recursion-wise-KV4 model/cache payload can be on the order of **~100 MB before activations/runtime buffers** at a 2K context. This is an unusually comfortable memory envelope for modern mobile devices.

### But decode weight traffic is larger than resident weight size

Approximating the 118M non-embedding unique parameters as 10/12 shared-stack and 2/12 edge parameters, balanced `N_r=3` gives `E[R]=2`. If the output embedding/head (~47M params) is read once per decoded token, raw W4 parameter traffic is approximately

```text
shared stack:  (118M * 10/12) * 2 recursions
edge blocks:   (118M *  2/12) * 1
output/head:    47M * 1
------------------------------------------------
active weight applications ~=263.3M parameter-reads
raw W4 traffic ~=131.7 MB/token
```

versus only 82.5 MB resident W4 parameters. This is an **engineering estimate**, because exact tied-embedding/head behavior and kernel caching matter, but it illustrates the recursion rule: **stored parameters and bytes streamed per token are different quantities**.

At max depth 3, the same estimate rises to ~180.8 MB/token. Therefore route-depth distribution should be logged during device benchmarks and converted into actual bytes/token.

---

## 11. Benchmark protocol that will expose the real bottleneck

For each candidate runtime/precision, report:

### Model/accounting

- exact **total learned parameters**, including embeddings/router/adapters;
- serialized file bytes and in-memory resident weight bytes;
- effective bits/weight including scales/zeros/codebooks;
- `L, d_model, H_q, H_kv, d_h`, local/global layer pattern and window;
- for MoR: `N_r`, unique stack depth, mean/p50/p95 selected recursion depth.

### Memory

Measure RSS/device allocations at context `{0,1K,2K,4K,8K}` and fit

```text
M(T) = M_fixed + slope_KV * T
```

Compare measured `slope_KV` with the formula. A mismatch usually reveals padding, high-precision residuals, duplicate cache copies, or backend fallback.

### Latency

Use batch 1 and report separately:

- cold model load;
- TTFT for prompts `{32,128,512,2K,4K}`;
- steady decode ms/token at context `{128,1K,2K,4K,8K}`;
- p50/p95/p99, not only average;
- first 16 generated tokens separately from later steady state;
- thermal state and power if available.

### Kernel/delegation

Capture an operator trace and answer:

- Is low-bit GEMM native or `dequant -> FP16 GEMM`?
- Is KV attention reading packed bits directly?
- Are K/V cache updates in-place?
- Are router/top-k/compaction ops on the accelerator?
- Does each recursion trigger synchronization or command-buffer submission?
- Do dynamic shapes force recompiles or graph fallback?

### Quality

At minimum compare FP16/BF16, W4, W4+KV8, W4+KV4, and the chosen extreme configuration on:

- held-out perplexity;
- short commonsense/instruction tasks;
- long-context retrieval if global context is advertised;
- generation stability over long chats;
- MoR route-depth behavior by token type/difficulty.

A low-bit setting should not be accepted because one perplexity number is stable if instruction following or long-context retrieval collapses.

---

## 12. Priority experiment matrix

| Priority | Experiment | Expected information gain |
|---:|---|---|
| P0 | Dense/GQA W4 vs W8A8 on actual CPU/GPU/NPU backends | Establish which precision is genuinely fastest |
| P0 | KV16 vs KV8 vs KV4 at 1K–8K | Determine cache bandwidth crossover and accuracy sensitivity |
| P0 | global vs 3:1 vs 5:1 local/global, W={1K,2K} | Bound KV and long-context TTFT with modest architecture risk |
| P1 | MoR `N_r=3`: expert-choice vs token-choice, recursion-wise cache | Separate quality gain from control-flow overhead |
| P1 | MoR recursion-wise vs first-recursion KV sharing | Quantify TTFT/RAM win versus decode-IO/quality tradeoff |
| P1 | W4 group 32/64/128 | Tune accuracy against metadata/kernel efficiency |
| P2 | KVTuner-style static mixed KV precision | Gain below KV4 without dynamic kernel irregularity |
| P2 | QuaRot/SpinQuant W4A4KV4 | Test true end-to-end 4-bit path only if backend can consume it |
| P2 | AQLM/QTIP/ScaleBITS / W3-W2 | Evaluate only after a concrete mobile decode kernel exists |
| P3 | Unstructured sparsity | Only if target runtime exposes a demonstrably faster sparse kernel |

---

## 13. Bottom line

For a **strictly <500M** language model, the best on-device efficiency stack is not the most aggressive compression paper in isolation. It is the combination whose **stored representation matches the device's executable kernels**.

The strongest near-term configuration is:

```text
<500M total learned params
+ GQA (roughly 1 KV head per 3–4 query heads)
+ mostly local attention (1K–2K) with sparse global layers
+ backend-native W4 (or W8A8 where integer GEMM is faster)
+ KV8 baseline -> packed KV4 target
+ fused cache update / attention
+ fixed preallocated cache state
```

MoR is a high-value research addition because it can trade unique parameters for effective depth and can cut recursion-wise KV traffic. But its on-device success depends on two conditions that server throughput results do not prove: **(1) skipped recursions must actually skip kernels, and (2) repeated shared-weight reads plus routing/control overhead must remain below the compute/cache savings.** For batch-1 mobile inference, recursion-wise KV caching is usually the safer decode-latency choice; first-recursion KV sharing is the stronger RAM/TTFT choice.

The practical rule for every proposed optimization is therefore:

> **Count total parameters, count actual bytes moved per token, inspect the delegated kernels, and measure TTFT + decode on the target device.**
