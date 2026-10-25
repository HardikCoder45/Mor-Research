# Evaluation and Ablation Framework

## 1. What must be proven
A candidate is not “best” because it has the lowest parameter count, the fewest FLOPs, the highest one-benchmark accuracy, or the smallest quantized file. The project target is the best **joint frontier of capability, latency, memory, energy and runtime portability** below 500M total learned parameters.

## 2. Comparison discipline
Every architecture comparison should control, where applicable:
- total learned parameters within ±2%;
- tokenizer/vocabulary or report its parameter difference explicitly;
- training token count;
- data ordering/mixture;
- optimizer and learning-rate schedule;
- teacher/KD signal;
- context length;
- evaluation harness version;
- quantization calibration/QAT budget;
- device runtime and thermal state.

When a factor cannot be controlled, label the comparison observational rather than causal.

## 3. Pretraining quality suite
Track checkpoints throughout training, not just the final model.

### Held-out losses
- high-quality general/educational web;
- code;
- math/science;
- long-form text;
- multilingual slices chosen for product scope;
- structured/JSON.

These capability-probe losses are useful for data-mix optimization and detect regressions earlier than noisy downstream tasks.

### Classic small-model continuity tasks
- HellaSwag;
- PIQA;
- ARC-Easy/Challenge;
- WinoGrande;
- BoolQ;
- OpenBookQA.

Keep because MobileLLM/SmolLM-family papers report them, but treat them as legacy signals with possible contamination.

### General knowledge/reasoning
- MMLU and MMLU-Pro (watch floor effects at <500M);
- TruthfulQA or a fresh factuality suite;
- GSM8K plus GSM-Symbolic;
- selected math/science problem sets that do not saturate at random chance.

### Code
- HumanEval+ / MBPP+ rather than raw minimal test suites alone;
- LiveCodeBench snapshot with version/date frozen;
- syntax/compilation failure rate and pass@1.

### Instruction following
- IFEval;
- capacity-matched internal instruction set with machine-checkable constraints;
- format-following/JSON validity.

### Freshness/contamination control
- LiveBench snapshot;
- LiveCodeBench;
- internally generated fresh transformations/questions held outside the data pipeline;
- exact benchmark release date recorded.

## 4. Context evaluation
For native 8K winner:
- perplexity by relative token position;
- retrieval needle at 1K/2K/4K/8K with multiple positions;
- RULER-like aggregation/retrieval tasks up to 8K;
- long-code dependency tests;
- multi-turn conversation retention;
- no claim beyond 8K until extension phase passes.

For LocalGlobal/CLA/EdgeRec longer-context candidates, repeat at 16K/32K according to each target.

## 5. Device suite
At minimum test one representative of each intended deployment class:
- modern Snapdragon Android phone (CPU and, if available, QNN/HTP);
- Pixel/Tensor-class Android if product-relevant;
- Apple iPhone/iPad (ANE and/or Metal);
- laptop CPU as a broad compatibility baseline.

Batch=1 interactive measurements:
- model load;
- TTFT at prompt {32,128,512,2K,4K,8K};
- steady decode at context {128,1K,2K,4K,8K};
- p50/p95/p99;
- peak memory/RSS;
- serialized model bytes;
- actual KV bytes versus context;
- energy/token and thermal behavior over ≥5 minutes;
- delegated operator fraction.

## 6. Phase-0 Colab proxy already completed
A proxy-scale implementation of all five architecture motifs was trained for 150 matched steps on Tiny Shakespeare on a Tesla T4 using the official Colab CLI. It validates implementation/trainability and reveals structural overhead. It is **not** a model-quality bakeoff.

Observed proxy prefill order (fastest first):
1. FlashSkip ~5.85 ms;
2. CLA ~9.79 ms;
3. LocalGlobal ~11.02 ms;
4. DenseDeep ~14.58 ms;
5. EdgeRec ~14.67 ms.

Observed proxy validation loss order (lower first): LocalGlobal ~2.341, EdgeRec ~2.347, FlashSkip ~2.432, DenseDeep ~2.484, CLA ~2.509. Differences are confounded by proxy geometry/initialization and the short character task; only the stability and structural-cost signals are retained.

## 7. Highest-value architecture ablations
### A1: physical depth vs width
At ~497M, compare roughly 12/16/20/24 physical blocks by resizing d/dFF while holding tokenizer and attention budget sensible. Device traces decide whether extra depth’s quality gain pays for serial latency.

### A2: attention count
For 16-layer FlashSkip, compare 7/9/11/full-16 attention blocks at fixed total params by reinvesting removed attention parameters into FFN width or small depth. This is probably the single highest-value architectural ablation.

### A3: attention placement
At fixed nine attention blocks compare:
- evenly interleaved;
- front-loaded;
- back-loaded;
- learned/search-derived placement.
Never accept >3 consecutive attention-free layers without strong evidence.

### A4: Q/KV grouping
Compare 20Q/4KV, 20Q/2KV and an 20Q/5KV-like configuration. Measure quality and real cache/decode behavior.

### A5: vocabulary
49,152 vs 65,536 vs ~96K tokenizer at fixed total learned params. Reinvest embedding savings into FFN/depth. Report both benchmark quality and average tokens/document because tokenizer changes sequence length.

### A6: FFN allocation
For fixed parameter total, compare dFF around 5,120 / 6,144 / 7,168 with compensating depth/attention changes.

### A7: QK norm / RoPE base
Ablate QK norm; RoPE theta around 1e4/1e5/1e6; measure both short-context quality and 8K stability.

### A8: skip-attention vs local/global
Compare FlashSkip against a 5:1 local/global model on the actual phone runtime. This tests whether the 2026 MobileLLM-Flash finding transfers to our larger 497M dimensions/runtime.

### A9: EdgeRec challenge
At matched resident W4 bytes and training tokens, compare FlashSkip and EdgeRec in economy/balanced/quality routing modes. EdgeRec only wins if extra compute produces substantial capability gains without breaking device latency.

### A10: cross-layer K/V
Compare CLA source/follower pair sharing versus ordinary GQA at matched total parameters and KV precision. Verify whether cache savings survive the exported runtime graph.

## 8. Training/data ablations
- CE only vs CE + Top-K teacher KD;
- Top-K KD vs full-logit KD on a short experiment;
- fixed teacher-forced KD vs periodic student/on-policy skew/reverse-KL;
- generic corpus mix vs capability-probe-optimized mix;
- 49K/65K tokenizer fertility;
- short/simple CoT vs strong long-CoT vs mixed complexity;
- SFT with/without capacity filtering;
- 4K-only vs 4K→8K context continuation.

## 9. Quantization ablations
- BF16 master;
- W8A8 where native;
- W4A16 or W4A8 group {32,64,128};
- KV16/KV8/KV4;
- PTQ vs short QAT/self-distillation;
- embedding/output W4 vs W8;
- only after above: W3/W2 and KV3/KV2.

## 10. Promotion gates for FlashSkip-497
Freeze as production architecture only after:
1. at least one 50M–300M token pilot shows no training instability;
2. nine-attention placement beats or matches nearby patterns at equal compute/params;
3. a 1–5B-token scaled-down or width-scaled experiment preserves the MobileLLM-Flash latency-quality trend;
4. full 497M model passes loss and benchmark gates after meaningful pretraining;
5. W4 export runs fully delegated on target runtime;
6. batch-1 phone p50/p95 latency is materially better than DenseDeep/LocalGlobal at similar quality;
7. KV memory slope matches nine-producer math;
8. no major code/math/instruction regression from 65K tokenizer or attention skipping.

## 11. Falsification mindset
The final winner is a **research recommendation**, not a theorem. If target-device search shows 20–24 layers or local/global attention dominates nine-block skip-attention after strong training, change the architecture. The knowledge base is designed to make that change evidence-driven rather than defending the current choice.
