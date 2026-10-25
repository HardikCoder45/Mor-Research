# Architecture and Parameter Allocation Synthesis

## Decision question
How should fewer than 500M learned parameters be allocated to maximize useful language capability while preserving actual on-device latency and memory efficiency?

## 1. What survives across the literature

### Tied embeddings are the default
At small scale, vocabulary matrices can consume a disproportionate share of the total budget. MobileLLM directly showed that tying input/output embeddings at ~125M saves a double-digit percentage of parameters with minimal immediate quality loss, and that reinvesting the saving into Transformer capacity improves the model. OpenELM, SmolLM2 and MobileLLM-Pro also use weight tying.

**Decision:** all five candidates use tied input/output embeddings. A separate LM-head matrix is rejected unless an ablation shows a statistically meaningful gain large enough to justify tens of millions of parameters and extra decode bandwidth.

### SwiGLU remains worth its parameter cost
MobileLLM’s small-scale ablation finds a gated FFN beneficial. Modern successful small models—including OpenELM, MobileLLM-Pro and LFM2—also retain gated FFNs.

**Decision:** use SwiGLU in every block. Do not economize by switching to a weak ReLU MLP.

### GQA is a baseline, not an optional trick
GQA reduces K/V projection parameters and KV-cache bytes while preserving many query heads. MobileLLM saw little loss with strong Q:KV grouping at 125M/350M; later on-device models consistently use it.

**Decision:** final candidates use approximately 4–5 query heads per KV head.

### Depth-vs-width has two objectives
MobileLLM’s controlled experiments show deeper/thinner geometry improves **quality per parameter** at 125M/350M. MobileLLM-Flash’s 2026 hardware-in-loop measurements show that for **latency at batch 1 on a current phone**, a shallower/wider point can be better because physical sequential depth and operator behavior dominate more than FLOPs predict.

This is not a contradiction. It is a multi-objective optimization:

- More physical depth provides more sequential nonlinear transformations and often improves quality at fixed storage.
- More width/FFN capacity reduces serial depth and can map better to modern GEMM kernels.
- Skipping attention in selected blocks recovers much of the depth/FFN capacity without paying softmax/KV cost at every block.

**Final production prior:** 16–24 physical layers rather than an extreme 30–40-layer stack, with a wide SwiGLU FFN and attention in only a subset of layers where latency is the primary product constraint.

## 2. Why the winning candidate uses d=1280 / dFF=6144
MobileLLM-Pro provides direct evidence that `d=1280`, `20` query heads, `4` KV heads and `dFF=6144` is a viable compact-model geometry. Those dimensions are also friendly to common matrix tile sizes. We reuse this width/FFN/head shape, but reduce the number of physical layers and attention-bearing blocks to fit below 500M.

For a 65,536 vocabulary with tied embeddings:

- embedding = `65,536 × 1,280 = 83.886M` parameters;
- each SwiGLU FFN = `3 × 1,280 × 6,144 = 23.593M`;
- each GQA attention source layer (20Q/4KV, 64-dimensional heads) ≈ `3.932M` linear weights;
- 16 FFNs + 9 attention modules + norms/QK-norm produce **496.810M total learned parameters**.

This spends ~76% of learned parameters on FFNs, ~17% on tied vocabulary embeddings, ~7% on attention and a negligible fraction on normalization. That allocation is intentional: attention is not the only source of language-model capacity, while it has disproportionate cache/runtime cost.

## 3. Vocabulary is an architectural variable
A tiny LM cannot treat tokenizer size as free.

At d=1280:
- 49,152 tied tokens cost ~62.9M params;
- 65,536 cost ~83.9M;
- 128k cost ~163.8M;
- 202,048 (MobileLLM-Pro scale) would cost ~258.6M—more than half of our 500M ceiling even with tying.

**Decision:** the production candidate uses **65,536 byte-level BPE tokens**. This is large enough for code, structured text and useful multilingual coverage without consuming the model. A 49k tokenizer remains a high-priority ablation if English-heavy quality/parameter efficiency dominates.

## 4. Why no MoE
The requirement is fewer than 500M **total deployable learned parameters**, not merely active parameters. MoE therefore cannot hide a larger weight bank behind a small active count. A 500M-total MoE also adds routing and multiple expert tensors without enough expert diversity to justify the systems complexity.

**Decision:** dense FFNs for v1.

## 5. Why not maximum layer sharing
Recursive sharing saves resident weights but reapplying a large shared block can reread those weights and add serial latency. MoR also shows aggressive tying is capacity-limited around the 135M base scale and only becomes more competitive by ~360M+. The target already allows nearly 500M unique parameters, so there is no need to sacrifice unique capacity everywhere.

**Decision:** recursion is Candidate 3/high-upside research challenger, not the production default. If used, share/reuse only a middle span and retain unique boundaries.

## 6. QK normalization and residual design
Recent compact models increasingly use QK normalization to stabilize attention score scale, especially under long-context or quantized training. The extra parameters are tiny. Pre-RMSNorm remains a well-supported stable default.

**Decision:**
- no linear biases;
- pre-RMSNorm, epsilon `1e-6`;
- QK RMS normalization per head before RoPE/dot product;
- final RMSNorm;
- standard residual additions for the production candidate;
- no learned absolute positional embeddings.

## 7. Position encoding
RoPE has the best ecosystem/runtime compatibility and works with long-context extension techniques. The production model targets 8K native context, not 128K marketing context: at <500M, long-context capability must be useful rather than merely syntactically accepted.

**Decision:** RoPE, train natively at 4K then continue/midtrain at 8K; use a high enough base (candidate `theta≈1e6`) and validate extrapolation. Do not deploy YaRN/NTK scaling beyond trained context until RULER/retrieval and device-cache tests pass.

## 8. Resulting design philosophy
The best <500M device Transformer is not a uniformly shrunken Llama. It is a **parameter-dense FFN backbone with selectively placed GQA**, tied vocabulary, strong normalization, and an architecture chosen by measured device behavior. Extra complexity—adaptive recursion, cross-layer KV sharing, or local windows—belongs in challengers and ablations until target-phone traces prove a net win.
