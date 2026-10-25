# Quantization and Device Deployment Synthesis

## 1. Deployment principle
The correct low-bit format is the format the target device can execute **without expanding back to FP16 for every matrix multiply or attention operation**. File size is not latency.

## 2. Weight formats
### Primary portable target: INT4 weights
For FlashSkip-497 (496.810M learned params):
- raw W4: ~248.4 MB decimal;
- W4 group-128 + FP16 scale: ~256.2 MB;
- W4 group-32 + FP16 scale: ~279.5 MB;
- INT8: ~496.8 MB;
- FP16/BF16: ~993.6 MB.

The group-32 format is larger but can improve quantization fidelity and matches practical mobile precedents such as MobileLLM-Pro CPU QAT. Group 64/128 should be benchmarked because the smaller metadata and larger groups may be faster or more cache-friendly.

### W8A8 is not a fallback
On NPUs/ANE-class accelerators with excellent INT8 dot-product support, W8A8 can beat W4A16/W4A8 in wall-clock latency even though it stores twice as many weight bits. SmoothQuant and platform optimization guidance support treating W8A8 as a serious competitor.

### Exotic <4-bit weights
QuIP#, AQLM and related methods can produce strong compression below 4 bits. For a ~500M model the absolute storage saving versus mature W4 is only on the order of tens to low hundreds of MB, while kernels become far less portable. Keep W3/W2 as a research branch unless a specific device backend consumes the format natively.

## 3. Recommended export matrix
Always preserve one BF16 master checkpoint. Produce multiple deployment artifacts rather than forcing one universal quantization:

### Arm CPU / ExecuTorch-XNNPACK
- primary: dynamic-A8 / W4 (`8da4w`) in the layout expected by XNNPACK;
- compare group 32/64/128 where supported;
- KV8 first, packed KV4 only if fused attention path exists;
- preallocated in-place cache.

### Qualcomm QNN / HTP
- benchmark backend-native block W4/A16 and W8A8 rather than importing arbitrary GGUF/AWQ packing;
- KV8 baseline;
- static shapes and fixed max context improve delegation;
- FlashSkip is attractive because it has no dynamic router/control flow.

### Apple
Two profiles should be tested:
- ANE: W8A8 or supported palettized/channel-wise format, stateful KV;
- GPU/Metal: INT4 block/group weights, stateful KV, optional lower-bit packed cache.

### llama.cpp / Metal / broad prototyping
- mature Q4_K/K-style formats are a practical first benchmark;
- use quantized KV only with Flash Attention/cache kernels that consume it efficiently;
- do not assume tiny IQ/I-quant files are faster on ARM/Metal.

### MLC/Vulkan
- q4f16/W4 and dynamic-A8/W4 are useful portable GPU targets;
- inspect generated kernels and avoid graph fragments that dequantize repeatedly.

## 4. Why FlashSkip is quantization-friendly
The winner uses only standard components:
- embedding lookup;
- RMSNorm;
- GQA attention on a fixed set of blocks;
- SwiGLU FFN;
- residual add;
- RoPE/QK norm.

There is no token router, dynamic gather/scatter, cross-layer cache aliasing requirement, state-space scan or custom convolution. The compiler sees a regular static graph, and the seven attention-free layers simply omit attention subgraphs. This is exactly the kind of optimization current mobile stacks can preserve.

## 5. KV cache
For FlashSkip-497:
- 9 KV-producing layers;
- 4 KV heads;
- head dimension 64;
- KV8 @ 8K ≈ 36 MiB ideal payload;
- KV4 @ 8K ≈ 18 MiB ideal payload.

These figures exclude alignment, scales, allocator overhead and any high-precision residual window. Actual device memory must be measured.

## 6. Quantization-aware training
A practical final phase:
1. freeze a high-quality BF16 teacher checkpoint of the same model;
2. insert fake quant matching the **exact target runtime** layout;
3. self-distill logits/hidden outputs from BF16 teacher;
4. quantize all large linear weights to W4;
5. keep norm scales in FP16/FP32 as runtime expects;
6. test embedding/output matrix at W4 and selectively raise it to W8 only if W4 damages rare-token/code behavior;
7. add activation fake quant only for W4A8/W8A8 targets;
8. treat KV8→KV4 as a separate QAT/validation dimension.

## 7. Device benchmark protocol
### Environment metadata
Record SoC, OS, runtime/compiler commit, thermal state, power mode, thread count, accelerator delegation and model format.

### Workloads
Prompt lengths: 32, 128, 512, 2K, 4K, 8K.
Generated tokens: at least 128 after warm-up.
Batch size: 1 for interactive on-device measurements.

### Report
- cold load time;
- peak RSS / accelerator allocation;
- TTFT p50/p95;
- prefill tokens/s;
- decode ms/token and tokens/s p50/p95;
- energy per generated token where measurable;
- temperature/thermal-throttle state;
- operator delegation percentage;
- size of serialized model and actual resident weights;
- cache bytes at 1K/2K/4K/8K.

### Trace questions
- Is W4 matmul native or `unpack→FP16 GEMM`?
- Are skipped-attention blocks truly absent?
- Are KV updates in place?
- Is quantized KV read packed inside attention?
- Does output projection dominate decode because of vocabulary size?
- Are any ops silently running on CPU when the rest is on NPU/GPU?

## 8. Acceptance thresholds for v1
Before calling a quantized model production-ready:
- no catastrophic regression on held-out LM loss;
- ≤1–2 relative points aggregate loss on the selected core benchmark suite unless speed gain justifies it;
- instruction-format and code-generation stability checked separately;
- p95 latency close to p50 (no repeated compiler/delegation fallback);
- memory slope with context matches expected KV design;
- no thermal collapse over a multi-minute generation test.

## 9. Compression order
Optimize in this order because each stage is easier to validate than the next:
1. architecture-level removal of redundant attention/KV producers;
2. GQA;
3. W4 or backend-native W8A8;
4. KV8;
5. packed KV4;
6. QAT/self-distillation recovery;
7. only then W3/W2 or KV3/KV2.

The model should become small because its **computation graph and representations are efficient first**, not because a complex codec is hiding an inefficient architecture.
