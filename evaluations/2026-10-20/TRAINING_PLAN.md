# Evaluation and Kaggle further-training plan — 20 October 2026

## Scope and recommendation

“Best” means the strongest saved model in this workspace, with the original <500M total-parameter deployment target. It does not mean a global leaderboard winner. Evaluate all local families with the same FP32 CPU implementation, tokenizer per checkpoint, 10 prose passages, 66 candidate-ranking diagnostics, and six greedy continuations of up to 24 tokens. Full outputs, timing and checkpoint SHA-256 hashes are in results.json. These previously published handcrafted diagnostics are useful regression tests, not a blind standardized benchmark. Do not train on these items or select many experiments against them.

The leading 10M candidate is non-recursive FlashSkip-10M at 1,500,053,504 trained labels. It has 9,964,832 parameters. Recorded validation loss is 4.1321 (PPL 62.31), but recorded losses across families are not a controlled comparison: their data provenance and training budgets differ. In our common evaluation, compare prose loss and category scores jointly. The recursive model has stronger factual ranking on this tiny sample; no universal superiority is established. MoR has only completed a locally available 100M-token checkpoint and should not be compared as an equal-budget architecture experiment.

For the fastest route to a stronger <500M product, benchmark a pretrained public candidate before spending a week improving the 10M model. LiquidAI/LFM2.5-350M is a current candidate, with SmolLM2-360M-Instruct as an alternative. The former's official card reports 350M parameters, extended pretraining and instruction tuning, and recommends extraction, structured outputs and tools; it cautions against knowledge-intensive tasks and programming. This is publisher evidence, not a model we ran here. Its different tokenizer means raw token perplexity cannot be directly ranked against FlashSkip; use task accuracy and optionally byte-normalized likelihood. Check its license and training framework compatibility before choosing it. Qwen3-0.6B exceeds this project's strict <500M total-parameter target.

## Stage 1: establish a trustworthy baseline (maximum 2 GPU session hours)

1. Freeze the chosen checkpoint, tokenizer hash, source revision, package versions and dataset manifest. Preserve the original weights. Validate checkpoint hashes and strict state loading.
2. Build a new document-disjoint train/development/test split with exact and near-duplicate removal. Deduplicate against benchmarks where available. Keep the final test inaccessible during tuning. Historical FlashSkip recorded validation alone is not sufficient evidence of leakage-free generalization.
3. Add pinned external task subsets: ARC-Easy (500), HellaSwag (500), PIQA (500), and a separately held-out prose corpus (at least 100K tokens). Report sample counts, answer scoring rule, per-task accuracy, confidence intervals and byte-normalized likelihood when comparing tokenizers. These datasets are proposed; they were not run today.
4. Add 100 new task-specific prompts with explicit correctness rubrics, including factual questions, short instructions and structured outputs. Test greedy decoding and a fixed sampling configuration on the exact same prompts; distinguish candidate ranking from free-generation accuracy.
5. Run a 200-step training pilot on the actual assigned T4: 50 warmup steps then measure useful labels/second, step time, peak VRAM and validation loss. Verify all losses/gradients are finite and resume matches an uninterrupted run. Benchmark one GPU against two GPUs before choosing DDP.

## Primary path: continue the winning FlashSkip-100M checkpoint

The completed four-family run selects FlashSkip-100M (99,182,336 parameters), saved at `flashskip/models/final`, trained on 200,015,872 labels, as the overall local quality baseline. Common prose PPL is 132.08 and equal-category accuracy is 74.7%; 10M at 1.5B labels has PPL 145.49 and 72.3%. This does not establish a statistically significant universal win: factual and arithmetic samples are tiny, and the larger model is substantially slower. The recorded validation PPL of the larger model was worse, illustrating why a common evaluation matters.

Freeze this checkpoint and run Stage 1 first. For the initial continuation pilots use two peak LRs, 1e-5 and 3e-5, each for 25M fresh labels, with the same data ordering and the optimizer/scaler/schedule policy described below. Use length 512, global batch 32,768 labels, FP16 AMP with FP32 optimizer state, gradient clip 1 and measured microbatch/accumulation on one T4. Initial data mixture: 80% deduplicated prose and 20% educational explanations by labels. The mixture and learning rates are hypotheses, not validated improvements. Preserve an unchanged-distribution control before attributing gains to curation.

Select the pilot using the fresh development split. Extend the winner by 100M labels initially; add another 100M only if measured throughput fits the remaining quota and development improvement continues. Evaluate every 10M labels during pilots and every 25M during continuation; checkpoint every 15–30 minutes. Target 5% lower fresh development PPL with no meaningful task regression. Do not promise a 1B-token run from the 10M model's throughput: the measured local generation speed already indicates very different compute cost. After continuation, apply Stage 2B SFT as a separate branch, using the same low-LR pilots and explicit masking/replay implementation requirements. Compare every stage to the untouched 100M checkpoint.

Keep the 10M route below as an optional research branch when the deployable parameter budget is specifically about 10M. Do not divide the first week's GPU quota across both model sizes by default. The weekly continuation allocation applies to the selected 100M path, with token count computed from its own T4 pilot.

## Stage 2A: controlled continuation of the existing 10M model

Keep architecture, tokenizer and context unchanged initially. The nominal 8192 context in its config is not evidence that it was trained or evaluated at that length; use 512 tokens for the first comparison. Do not alter recurrence or vocabulary while resuming its weights.

Run two 50M-new-token pilots from the same 1.5B checkpoint and identical fresh data order: AdamW with peak LR 1e-5 versus 3e-5, beta=(0.9,0.95), epsilon 1e-8, weight decay 0.1, gradient clip 1. Use an explicit new continuation schedule with 2% warmup then cosine to 10% of peak LR. These are proposed starting values, not validated optima. Restore optimizer moments only if state is complete and compatible, while deliberately resetting the LR schedule for this new phase; otherwise label the run as a weights-only restart and use the same restart procedure in both pilots.

Use global batch 32,768 labels (64 sequences of 512 prediction targets), FP16 autocast with FP32 weights/Adam moments and GradScaler on T4. Select microbatch by the pilot and accumulate gradients. Count only successful optimizer-step labels. Retry overflows; checkpoint data cursor/RNG/scaler so resuming does not skip or replay data. Compute validation in FP32.

Start with 80% newly deduplicated high-quality prose and 20% concise educational explanations, measured by trained labels. Keep an unchanged-data continuation as a control if a new mixture wins: otherwise a gain cannot be attributed to the new mixture. Curate factual/commonsense text based on error categories, never the public diagnostic questions themselves. Verify redistribution/training permissions and provenance for every source.

If a pilot improves development loss without behavioral regression, extend in 100M-label segments, initially capped at 300M additional labels. Re-evaluate every 25M labels on development only. Save rolling state every 15–30 minutes plus segment boundaries. Preserve best and latest checkpoints separately. More pretraining is not guaranteed to resolve the capacity and generation problems.

## Stage 2B: improve useful behavior with supervised distillation

Only after a clean continuation comparison, train a separate branch using 20K–50K verified short instruction/answer examples, roughly 5M–15M supervised labels depending on lengths. Favor single-step reasoning, grounded short answers, extraction and simple structured output over long chains of thought. Use a stronger teacher only when access and usage terms permit; log teacher/version/prompts and filter correctness, duplication and formatting. No teacher generation or API spending was performed in this evaluation.

Use a consistent instruction format built from existing tokenizer tokens; add assistant-only loss masking and EOS handling. Do not assume the current pretraining trainer implements these features. Suggested full-weight SFT pilots: LR 5e-6 and 1e-5, one epoch maximum, length 512, 10–20% prose replay to reduce forgetting. Keep compute and replay ratios explicit. Stop if development prose loss rises >3% or task accuracy drops >2 percentage points. These are operational thresholds, not statistical significance claims.

Avoid preference optimization or RL until supervised learning gives measurable improvements and a reliable reward is available. Decoding sweeps can reduce looping but must be reported separately from weight improvements.

## Alternative: improve a pretrained 350M model

If model size can rise from 10M to 350M, compare the untouched public candidate first. For narrow tasks, try LoRA rank 8/16, alpha 16/32, dropout 0.05, LR 5e-5–1e-4, length 512 and one epoch on verified task data. Discover supported module names from the actual architecture; do not copy transformer projection names into a hybrid model blindly. Test adapter support and backward execution before a full run. Full FP16 fine-tuning may also fit at this size with small microbatches; measure first. Quantization-aware adapters are optional rather than the default for 350M. Compare against the unmodified base and against FlashSkip on actual tasks, memory and latency. This is the more promising route to absolute quality under 500M, but is not a validated winner today.

## Kaggle execution and budget

Kaggle announced P100 retirement effective 15 September 2026; use the currently offered T4 x2 option and inspect actual allocation at runtime. Two 16GB GPUs are separate memories, not one 32GB GPU. For 10M parameters, DDP overhead may outweigh its benefit. Use one T4 unless measured throughput justifies two; idle allocated GPUs may still count under account quota rules. Existing P100-specific metadata must be replaced in a new kernel, and API accelerator identifiers must be checked against the installed Kaggle CLI before submitting.

Kaggle documentation describes up to 12-hour CPU/GPU sessions and a weekly GPU quota commonly around 30 hours, subject to availability/account conditions. Actual account quota was not queried. Work in resumable chunks of at most 8 hours, save to /kaggle/working, commit notebook outputs, download and verify artifacts before resuming. Keep validation data, tokenizer, manifests and complete resume state with each private version. Do not rely on a local tunnel or supervisor as the only copy of training progress.

Tentative first weekly allocation, in session hours: 2 for baseline/throughput/resume checks; 6 for LR/data pilots; 12 for selected continuation; 4 for SFT pilots; 3 for final evaluation/export; 3 reserve. Reallocate after throughput measurements and the account's actual quota. Never infer remaining quota from historical telemetry.

Budget formula: planned labels = measured successful labels/sec × available training seconds × 0.8 reserve factor. At 10K labels/sec, 300M labels need about 8.33 hours before evaluation/checkpoint overhead; at 25K, 3.33 hours. These are arithmetic examples, not measured T4 rates. Historical P100 rates do not establish T4 performance.

## Promotion criteria

Pre-register one primary goal before training: for the 10M research branch, at least 5% lower fresh development perplexity; for a task adapter, at least 5 percentage points higher accuracy on the target rubric. Require no material regressions in other categories, and report paired bootstrap intervals on the blind final evaluation. Measure at least 200 free generations for repetition and correctness. Aim for 20% relative reduction in repeated trigrams at fixed decoding, and no more than 10% latency regression on the intended device. These are success targets, not promised outcomes. Run the blind test once for the selected candidate; if it fails, keep the baseline.

Export FP16 first, then evaluate any INT8/INT4 conversion separately. Record model RAM, prefill time, decode tokens/sec, context and runtime. Current local CPU timings recompute the entire prefix and use short generations; they are not mobile-device or KV-cached serving benchmarks.

## Sources checked

- Kaggle P100 retirement: https://www.kaggle.com/discussions/product-announcements/735239
- Kaggle notebook sessions: https://www.kaggle.com/docs/notebooks
- Kaggle GPU quota guidance: https://www.kaggle.com/docs/efficient-gpu-usage
- LFM2.5-350M official model card: https://huggingface.co/LiquidAI/LFM2.5-350M
- SmolLM2-360M-Instruct official model card: https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct

Training was not launched by this request; the deliverable is an executed local inference/evaluation and a further-training plan. Old telemetry is historical and does not prove a current Kaggle job is running.
