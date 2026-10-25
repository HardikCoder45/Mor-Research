# Training, Data, Distillation, and Post-Training Synthesis

## Goal
The final architecture only defines the function class. At ~500M parameters, **data quality, teacher signal and capacity-matched post-training are likely to determine more of final capability than another small architectural tweak**.

## 1. Training-regime conclusion
A fixed on-device model should be deliberately overtrained relative to classical compute-optimal scaling because deployment size—not training compute—is the hard constraint. Existing high-performing small models span very different token budgets:
- MobileLLM final 125M/350M models: ~1T tokens;
- OpenELM: ~1.8T;
- SmolLM2-135M/360M: 2T/4T;
- MobileLLM-R1 reasoning program: ~4.2T resampled high-quality tokens;
- LFM2 dense family: ~10T base + ~1T long-context midtraining.

There is no single proven optimum for our 497M architecture. The evidence says to optimize **information density and teacher signal first**, then extend tokens while capability-probe losses still improve.

### Recommended budget
- **Minimum serious run:** 3–4T tokens plus strong KD.
- **Preferred run:** 6–8T high-quality effective tokens, measured after deduplication/resampling, with online/offline teacher logits over a large fraction.
- **Stretch:** continue toward 10T only if held-out capability probes still show non-saturated marginal gains.

These are research budgets, not claims that 8T is mathematically optimal.

## 2. Data mixture philosophy
SmolLM2’s 135M/360M result is unusually important: smaller models did better when high-quality code/math/educational data was present **from the beginning**, rather than relying on a sophisticated late curriculum designed for a larger model.

Start with a single high-quality mixture containing:
- filtered educational/web text (FineWeb-Edu / DCLM/DataComp-LM-style filtering);
- code with quality/licensing/dedup filters;
- math/science derivations and problems;
- encyclopedic/reference text;
- books/long-form prose where licenses allow;
- structured formats (JSON, Markdown, tables);
- multilingual text only to the degree the 65K tokenizer and model capacity can support without damaging the main target capabilities;
- synthetic “textbook”/explanation material after diversity and factuality filtering.

Do not freeze arbitrary percentages based on another model. Use capability-probe validation losses to tune mixture weights.

### Initial mixture hypothesis for pilot training
This is a starting experiment, not a final truth:
- 50–60% high-quality general/educational web + reference;
- 15–20% code;
- 10–15% math/science/reasoning;
- 5–10% structured technical/Q&A;
- 5–10% multilingual/high-value domain text.

Run small 100–300M-token predictive experiments and leave-one-source-out comparisons. MobileLLM-R1 provides a useful precedent for measuring how each corpus changes capability-probe NLL rather than choosing data by reputation alone.

## 3. Tokenizer training
Train a 65,536-token byte-level BPE/SentencePiece-BPE tokenizer on the **same broad domain mixture** as pretraining, with:
- byte fallback / complete UTF-8 coverage;
- strong code and whitespace representation;
- reserved control/chat/FIM tokens;
- no huge language-specific vocabulary partition;
- tokenizer fertility audit on English, code, math notation and selected multilingual sets.

Ablate 49,152 versus 65,536 versus ~96K at fixed total model parameters. A larger vocabulary is accepted only if reduced sequence length and better multilingual/code tokenization outweigh the transformer parameters it consumes.

## 4. Pretraining objective
Primary objective: causal next-token prediction.

Use teacher distillation concurrently rather than only after pretraining. LFM2 and MobileLLM-Pro show that pretraining-time logit KD is practical; MiniLLM/DistiLLM show why pure teacher-forced forward-KL can be suboptimal for generative students.

### Recommended composite objective
`L = L_CE + λ_KD L_topK + λ_on L_student + λ_reg L_regularization`.

Where:
- `L_CE`: ground-truth token CE;
- `L_topK`: temperature-scaled teacher distribution over Top-K≈32–64 logits plus an explicit outside-Top-K mass term (LFM2-style decoupling is a strong implementation reference);
- `L_student`: periodic skew/reverse-KL or adaptive off-policy loss on student-generated continuations;
- `L_regularization`: architecture-specific terms (router/monotonic loss only for EdgeRec).

Do not store full 65K teacher logits for every token if Top-K plus residual mass preserves the useful signal; bandwidth/storage savings are enormous.

## 5. Teacher strategy
Use at least one strong 3B–8B teacher for general language/logits, potentially complemented by domain specialists for code/math. Teacher parameters do **not** count toward deployed model size.

Important constraints:
- teacher tokenizer should ideally match the student tokenizer for efficient token-level logit distillation;
- if tokenizers differ, sequence-level/hidden feature alignment becomes more expensive and noisier;
- teacher quality is not enough—teacher outputs must be learnable by a 500M student.

For reasoning, “strongest possible, longest possible CoT” is explicitly rejected. The small-model learnability-gap work shows smaller students can benefit more from short/simple or mixed-complexity chains.

## 6. Reasoning training
### Pretraining
Ensure the model sees genuine reasoning structure through:
- code execution patterns;
- mathematical derivations;
- proofs/explanations;
- science QA/expository chains;
- algorithmic text;
- structured transformations.

### SFT
Use capacity-aware instruction data:
- short/medium reasoning traces dominate;
- hard long traces are sampled, not the whole corpus;
- include answer-only examples so the model does not learn to emit verbose chains for everything;
- include code, tool-style structured output and constrained formatting only after the base student shows competence.

### Preference/post-training
Start with filtered SFT, then DPO/IPO-style preference optimization on high-confidence preferences. SmolLM2’s small variants filtered out some over-complex instructions before preference training; this is a useful warning against training a 500M model on tasks it cannot model coherently.

RL should be an optional later experiment, not required for v1. At this scale reward hacking/format gaming can overwhelm genuine capability.

## 7. Learning-rate and optimizer baseline
A stable baseline:
- AdamW, betas roughly `(0.9, 0.95)`;
- weight decay ~0.1 on matrix weights, not norm/bias (there are no linear biases);
- BF16 training;
- global gradient clipping 1.0;
- warmup then cosine or WSD-style stable/decay schedule;
- tune peak LR with short loss-scaling sweeps rather than importing one number from a different batch/token budget.

Small models often tolerate relatively high peak learning rates; SmolLM2 small variants used a high LR with WSD. The exact value should be determined by 5–20B-token pilot sweeps.

## 8. Sequence-length curriculum
For FlashSkip-497:
1. majority pretraining at 2K–4K for throughput;
2. a later 8K continuation phase using naturally long examples plus packed short sequences;
3. distill from a long-context teacher during that phase;
4. do not waste the whole 6–8T run on 8K padding/attention when most learning signal is local.

## 9. Specialist merging
MobileLLM-Pro reports specialist training/merging benefits, but it introduces another training branch and merging hyperparameters. Treat as Phase-2 research:
- branch code specialist, math/reasoning specialist and general-language specialist from a common late checkpoint;
- anneal at low LR;
- evaluate weighted parameter averaging/model soups;
- only keep if one merged checkpoint beats the common base broadly without regressions.

It is not required to train v1 successfully.

## 10. Quantization-aware recovery
After BF16 base and post-training quality are stable:
- create W4 target fake-quant graph matching the selected runtime’s group/block layout;
- self-distill from frozen BF16 model;
- keep norms and possibly small sensitive tensors at 8/16 bit if backend permits;
- introduce KV8 fake quant, then KV4 as a separate experiment;
- run a short high-quality recovery/QAT phase rather than retraining from scratch under low precision.

The target is not “all tensors 4-bit”; it is the fastest real delegated graph with minimal benchmark regression.

## 11. Stop criteria
Training should stop based on marginal capability gain, not a round token number. Track:
- general-language held-out loss;
- code/math/science capability-probe losses;
- fresh benchmark subsets inaccessible to the data pipeline;
- instruction-following and reasoning validation after checkpoints;
- teacher-student KL gap;
- token-domain saturation: if one domain’s loss no longer improves, resample data rather than continuing identical mixture.

## Bottom line
The best 500M model is a co-designed **architecture + data distribution + teacher**. A strong architecture trained on generic noisy web will lose to a slightly weaker architecture trained on information-dense, capacity-matched data with good distillation. The training program should therefore spend real experimental budget on data ablations and teacher objectives, not only layer/head sweeps.
