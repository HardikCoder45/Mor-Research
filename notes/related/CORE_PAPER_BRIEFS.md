# Core Paper Briefs for a <500M On-Device Transformer

Research snapshot: 2026-08-30. This file is the high-signal reading layer over the full local corpus. The complete PDFs and extracted text are in `papers/` and `metadata/extracted_text/`. **Evidence** means a direct paper result or stated design; **Synthesis** is the project’s interpretation; **Hypothesis** requires our own controlled experiment.

## Architecture / parameter allocation

### MobileLLM — arXiv:2402.14905
**Evidence.** Controlled 125M/350M experiments show sub-billion Transformers are sensitive to architecture at fixed parameter count. SwiGLU beats a weaker FFN; deeper/thinner variants outperform shallower/wider variants for quality-per-parameter; tied input/output embeddings recover a large small-model parameter budget; GQA with far fewer KV heads is nearly accuracy-neutral; block reuse can improve quality without increasing stored parameters. The final 125M/350M models were trained on roughly 1T tokens.

**Synthesis.** This is the strongest direct evidence that a <500M design should not blindly copy a 7B geometry. However, its “deep-thin” result optimizes capability per parameter, not current phone latency.

### MobileLLM-Flash — arXiv:2603.15954
**Evidence.** Hardware-in-the-loop architecture search on a Samsung Galaxy S25 finds parameter count and FLOPs are only moderate predictors of measured latency. On the device latency-quality frontier, shallower/wider configurations become attractive. Interleaved skip-attention is more mobile-friendly than sliding-window attention in the tested CPU stack, and the authors warn against long runs of consecutive attention-free layers. Their 350M model uses 12 layers, d=1024, dFF=4096, 32 Q / 8 KV heads of size 64, but only seven attention blocks. It uses standard operators and reports strong measured phone decode throughput. The final model uses continued pretraining plus distillation.

**Synthesis.** This is the single most relevant 2026 result for a deployable pure Transformer. It changes our final winner from “maximum depth” to a shallower/wider, selectively-attentive architecture.

### OpenELM — arXiv:2404.14619
**Evidence.** Decoder-only pre-RMSNorm/RoPE/GQA/SwiGLU model with tied weights and **layer-wise scaling**: attention-head allocation and FFN multiplier increase across depth rather than remaining uniform. The 270M model uses d=1280 and 16 layers; the family trained on ~1.8T public tokens. The paper also notes that higher benchmark quality did not automatically imply higher inference speed because implementation/runtime details matter.

**Synthesis.** Nonuniform parameter allocation is credible, but it complicates kernel regularity. For the first production candidate we prefer uniform d/dFF and selectively remove attention, which modern runtimes handle more predictably.

### SmolLM2 — arXiv:2502.02737
**Evidence.** The 135M/360M variants were trained on 2T/4T tokens. Unlike the 1.7B model’s multistage recipe, the smaller variants benefited from a **single-stage, consistently high-quality mixture** containing filtered web/educational data plus code, math and synthetic educational material. Their post-training data was also filtered to model capacity, removing overly hard/complex examples before DPO.

**Synthesis.** Small models are capacity-limited enough that data quality and learnability dominate raw corpus breadth. Our pretraining should start with high-quality data from token one rather than postponing code/math/education until a late phase.

### MobileLLM-Pro — arXiv:2511.06719
**Evidence.** A 1.08B reference model uses 30 layers, d=1280, dFF=6144, 20 Q / 4 KV heads, tied embeddings, local/global attention and a very large 202,048 vocabulary. It uses logit KD during pretraining, long-context positional distillation, specialist merging, and INT4 QAT with self-distillation. CPU deployment uses symmetric INT4 group size 32; accelerator deployment uses channel-wise 4-bit quantization.

**Synthesis.** Its `d=1280, dFF=6144, 20Q/4KV` geometry is an excellent tested width/FFN/head template. Our latency winner deliberately keeps those dimensions but reduces physical depth and attention count to stay below 500M.

### LFM2 — arXiv:2511.23404
**Evidence.** Hardware-in-the-loop search produces a 350M edge model with 16 layers, d=1024, dFF=4608 and only six GQA layers; most sequence mixing is done by gated short convolutions. It is pretrained for roughly 10T tokens plus ~1T long-context midtraining and uses decoupled tempered Top-K=32 logit distillation. It reports strong batch-1 CPU prefill/decode throughput.

**Synthesis.** LFM2 is not a pure Transformer, so it is not one of the final five. It independently supports the key systems conclusion: global softmax attention need not be present in every block, and hardware-in-loop architecture search is more reliable than FLOPs alone.

## Recursion / parameter sharing / adaptive depth

### Universal Transformer — arXiv:1807.03819
**Evidence.** Foundational recurrent application of shared Transformer computation across depth, with dynamic halting ideas.

**Synthesis.** Establishes that logical depth can exceed unique physical layers, but modern decoder-only deployment needs more careful cache and representation handling.

### Lessons on Parameter Sharing — arXiv:2104.06022
**Evidence.** Cycle/Sequence/Cycle-Reverse schedules relax full sharing and often outperform one-block universal tying at comparable stored parameter count.

**Synthesis.** Sharing schedule matters; repeated blocks should retain some depth specialization.

### Relaxed Recursive Transformers — arXiv:2410.20672
**Evidence.** Converts pretrained Transformers to recursive models, then adds depth-wise low-rank adapters to recover flexibility under tying. Uptraining and careful initialization outperform naive sharing.

**Synthesis.** If we recurse a <500M model, depth-specific LoRA-like adapters are a cheap way to reduce representational aliasing.

### Mixture-of-Depths — arXiv:2404.02258
**Evidence.** Routes a fixed token capacity through optional Transformer computation, making compute allocation token-dependent while preserving a known overall budget.

**Synthesis.** A static capacity budget is attractive for predictable deployment, but top-k sequence routing can create compiler/control-flow difficulty on phone accelerators.

### Mixture-of-Recursions — arXiv:2507.10524
**Evidence.** Combines shared recursive layers, token-level routing, and recursion-aware KV caching. Middle-Cycle sharing preserves unique boundary layers and is strongest among the studied sharing patterns. At the 135M base scale aggressive sharing is capacity-limited; around 360M+ it closes much of the gap. Recursion-wise KV cache reduces memory/IO in the shared stack to `(Nr+1)/(2Nr)` under balanced routing; first-pass KV sharing reaches `1/Nr` memory but does not similarly reduce decode cache IO. Server continuous-depth batching gains are not a phone batch-1 guarantee.

**Synthesis.** Recursion is a high-upside challenger for our size range, but should not be the default production winner until real phone routing/compaction shows skipped work actually disappears.

### RecursiveVLM — arXiv:2602.09080
**Evidence.** Naive final-state recursion suffers input-distribution mismatch. A multi-tap normalized connector with zero-initialized residual path improves recurrence, and a monotonic loss penalizes tokens that become worse after more recursive computation. More recursion is not automatically better; R=2 is often the best general trade.

**Synthesis.** A language-model recursive challenger should combine MoR’s adaptive middle recurrence with a zero-init connector and monotonic multi-depth objective rather than directly feeding the last hidden state back.

## Attention / cache architecture

### Multi-Query Attention — arXiv:1911.02150
**Evidence.** Shares one K/V head across multiple query heads to reduce autoregressive KV bandwidth.

### GQA — arXiv:2305.13245
**Evidence.** Grouped-query attention provides a middle point between MHA and MQA; it retains much of multi-head quality with substantially reduced K/V projections and cache.

**Synthesis.** GQA is the baseline for all five candidates. Our final candidates use one KV head per roughly 4–5 Q heads rather than MHA.

### Cross-Layer Attention — arXiv:2405.12981
**Evidence.** Adjacent layers can share K/V projections and cached states, reducing KV cache by another ~2x relative to an already cache-efficient attention baseline while maintaining near-comparable accuracy in the paper’s studied scale.

**Synthesis.** Strong memory idea, but direct evidence is at larger size than our target and mobile kernel support is less mature. It becomes Candidate 4 rather than the default.

### YOCO — arXiv:2405.05254
**Evidence.** Decoder-decoder architecture computes/caches a global KV representation once and reuses it through cross-attention, sharply reducing KV storage at long context.

**Synthesis.** High-value inspiration, but a larger architectural deviation than needed for v1. Cross-layer KV sharing and local/global attention are lower-risk.

### KIVI — arXiv:2402.02750 / KVQuant — arXiv:2401.18079
**Evidence.** Very low-bit KV is feasible when quantization axes and outlier handling match K/V statistics. KIVI’s asymmetric 2-bit scheme uses different granularity for K and V; KVQuant shows high-quality ~3-bit cache with custom kernels.

**Synthesis.** Nominal bitwidth is not enough: packed attention kernel support determines whether low-bit KV saves latency rather than merely RAM. Use KV8 first, then packed KV4.

### SnapKV / MiniCache / Dynamic Memory Compression
**Evidence.** Long-context cache can also be compressed by selecting important positions, merging across depth, or learned online compression.

**Synthesis.** These are second-generation optimizations. For a first 500M device model, a structurally small cache (few attention/KV-producing layers + GQA) is easier to reason about and deploy.

## Quantization / systems

### GPTQ — arXiv:2210.17323; AWQ — arXiv:2306.00978
**Evidence.** Strong post-training low-bit weight quantization baselines. AWQ is explicitly oriented toward practical edge/mobile acceleration when matched with kernels.

### SmoothQuant — arXiv:2211.10438
**Evidence.** Activation/weight rescaling makes W8A8 practical; W8A8 can outperform smaller weight formats on accelerators with excellent INT8 GEMM.

### QuaRot — arXiv:2404.00456; SpinQuant — arXiv:2405.16406
**Evidence.** Rotations reduce outlier difficulty and improve end-to-end W4A4KV4 feasibility.

### QuIP# / AQLM
**Evidence.** Vector/codebook techniques can move below scalar 4-bit quantization with strong rate-distortion.

**Synthesis.** For a <500M model the extra 50–120 MB saved by exotic 2–3-bit formats is not automatically worth custom-kernel complexity. W4 is the primary deployment target; W8A8 remains a platform-specific competitor.

### LLM in a Flash — arXiv:2312.11514
**Evidence.** Constrained-device inference can be dominated by movement between slower model storage and fast memory; data movement strategy can be as important as arithmetic.

**Synthesis.** This supports attention skipping and weight-reuse decisions based on real bytes moved/token, not theoretical FLOPs alone.

## Data / scaling / distillation

### Scaling Laws — arXiv:2001.08361; Chinchilla — arXiv:2203.15556
**Evidence.** Foundational parameter/data/compute scaling relationships.

**Synthesis.** A fixed deployable model is intentionally **overtrained relative to compute-optimal training** because inference size is the hard product constraint. Chinchilla is a training-cost baseline, not a reason to stop a 500M model after a small token budget.

### Textbooks Are All You Need — arXiv:2306.11644; TinyStories — arXiv:2305.07759
**Evidence.** High-quality synthetic/educational distributions can produce surprisingly capable very small models in constrained domains.

**Synthesis.** Synthetic material is useful when it raises information density and matches student capacity; it should not replace diverse real text.

### DataComp-LM — arXiv:2406.11794; FineWeb — arXiv:2406.17557
**Evidence.** Controlled filtering and data curation materially change downstream quality; educational-quality filtering is especially relevant to small models.

### MobileLLM-R1 — arXiv:2509.24945
**Evidence.** Reasoning-oriented sub-billion training can be far more token-efficient when corpus contribution is measured with capability probes and data is resampled accordingly. The report shows strong reasoning with ~4.2T training tokens rather than assuming >10T is mandatory.

**Synthesis.** Our data mixture should be treated as an optimizable model component. Run small predictive/leave-one-source-out experiments and adjust domain weights based on held-out capability losses.

### MiniLLM — arXiv:2306.08543
**Evidence.** Reverse-KL, on-policy generative distillation improves student behavior and is tested down to ~120M students.

### DistiLLM — arXiv:2402.03898
**Evidence.** Skew-KL plus adaptive off-policy student generations makes generative distillation more efficient/stable than a fixed teacher-only objective in the studied settings.

### Small Models Struggle to Learn from Strong Reasoners — arXiv:2502.12143
**Evidence.** Small models can learn worse from very long/complex reasoning traces and overly strong teacher demonstrations; shorter/simpler or mixed-complexity reasoning can be more learnable.

**Synthesis.** Do not fill the post-training corpus with giant teacher CoTs. Use capacity-matched short/medium reasoning, mix difficulties, and distill logits/process supervision where the student can actually represent it.

## Hybrid alternatives considered but not selected for the final five pure Transformers

### Mamba / Mamba-2 / Griffin / RecurrentGemma / Hymba
These works show attractive recurrent/SSM/attention hybrids and can drastically reduce sequence-state cost. However, runtime/kernel portability and apples-to-apples evidence below 500M are less uniform than standard Transformer operators.

### Opt.Gear — arXiv:2608.01034
**Evidence.** A 270M NPU-oriented system combines local/global GQA with a ConvKV-gated mixer, 65K context support and a large tokenizer. It is direct 2026 evidence that fixed-state/local mixing plus sparse global attention is strong on edge accelerators.

**Decision.** Keep as a future hybrid branch. The user asked for the five best **Transformer architectures**, so the final five remain attention/FFN Transformer variants that can be implemented with broadly standard kernels.

## Cross-paper conclusion
The most defensible 2026 architecture prior is no longer “put attention in every deep-thin block.” For a strict <500M on-device model, quality-per-parameter evidence favors strong gated FFNs, tied embeddings and adequate representational depth, while direct device evidence favors **fewer physical layers and fewer attention blocks**. The final solution therefore spends most of the 500M budget on a wide SwiGLU backbone, uses GQA, removes redundant attention at selected depths, and treats recurrence/cross-layer KV/local attention as specialized challengers rather than mandatory complexity.
