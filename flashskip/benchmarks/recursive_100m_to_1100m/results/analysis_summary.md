# FlashSkip Recursive 100M–1100M: Evaluation Analysis

## Bottom line

The checkpoint comparison shows two different notions of “best”:

- **900M and 1100M are tied for the best fair mean core rank (3.125)** across handcrafted held-out NLL, aggregate diagnostic accuracy, answer-confidence margin, and greedy 3-gram repetition.
- **1100M is the best language-model checkpoint by the training/validation objective**: logged validation perplexity is lowest at **73.98**, and its mean diagnostic answer margin is the highest.
- **900M is the most compute-efficient checkpoint with essentially the same observed basic-language capability**: it has the lowest handcrafted held-out perplexity (**161.37**) and exactly the same correct/incorrect outcome as 1100M on all **66** grammar/lexical/factual/commonsense/arithmetic-logic items.
- **600M produces the least repetitive greedy continuations** in this small generation test (mean repeated 3-gram fraction **16.7%**), but it is materially weaker on the capability diagnostics than 900M/1100M.

For a default checkpoint to keep training, fine-tune, or use as the strongest general LM, **1100M is the safest choice**. For the best capability-per-training-token point observed here, **900M is the efficiency sweet spot**. There is no evidence that 1100M is meaningfully worse; rather, the model has entered a strong plateau after about 900M tokens.

## Test suite

All checkpoints use the same `FlashSkip-10M-Recursive-MiddleCycle-Nr2` architecture, exact tokenizer, deterministic MPS evaluation, and strict checkpoint loading. The suite contains:

- 10 handcrafted held-out prose passages for teacher-forced NLL/perplexity
- 20 grammatical minimal pairs
- 12 lexical cloze items
- 12 basic factual knowledge items
- 12 commonsense items
- 10 arithmetic/logic items
- 6 greedy free-generation prompts, 24 new tokens each

This is a compact checkpoint diagnostic suite, not a substitute for large standardized benchmarks.

## Full checkpoint comparison

| Tokens | Logged val PPL | Held-out PPL | Grammar | Cloze | Facts | Commonsense | Arith/Logic | Mean category accuracy | Greedy repeat 3-gram | Fair core rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100M | 149.17 | 197.12 | 95% | 75% | 25% | 66.7% | 40% | 60.3% | 25.0% | 9.25 |
| 200M | 105.08 | 188.56 | 90% | 75% | 25% | 66.7% | 40% | 59.3% | 17.4% | 8.13 |
| 300M | 95.16 | 175.45 | 90% | 75% | 25% | 66.7% | 40% | 59.3% | 19.7% | 7.88 |
| 400M | 87.21 | 170.14 | 90% | 75% | 33.3% | 66.7% | 40% | 61.0% | 20.5% | 6.63 |
| 500M | 82.74 | 171.18 | 90% | 75% | 33.3% | 66.7% | 40% | 61.0% | 21.2% | 6.88 |
| 600M | 79.56 | 165.76 | 90% | 75% | 33.3% | 66.7% | 50% | 63.0% | **16.7%** | 4.50 |
| 700M | 77.48 | 166.74 | 90% | 75% | 41.7% | 75% | 40% | 64.3% | 29.5% | 6.50 |
| 800M | 75.87 | 164.35 | 90% | 75% | 50% | 75% | 50% | 68.0% | 31.1% | 5.75 |
| **900M** | 74.99 | **161.37** | 90% | **83.3%** | **50%** | **75%** | **50%** | **69.7%** | 27.3% | **3.13** |
| 1000M | 74.40 | 162.75 | 90% | **83.3%** | **50%** | **75%** | **50%** | **69.7%** | 28.8% | 4.25 |
| **1100M** | **73.98** | 161.56 | 90% | **83.3%** | **50%** | **75%** | **50%** | **69.7%** | 27.3% | **3.13** |

The “mean category accuracy” gives each of the five diagnostic categories equal weight. Because grammar is binary-choice while the other four are four-choice, its naive equal-category chance baseline is about 30%.

## What improved with more training

### 100M → 200M: large LM improvement, little capability movement

Logged validation perplexity drops from **149.17 to 105.08 (-29.6%)**, the largest single interval gain. The handcrafted held-out perplexity improves only **4.3%**, while diagnostic accuracy changes from 60.3% to 59.3%. Early training is therefore primarily improving broad token-distribution modeling rather than flipping these coarse language-test decisions.

### 200M → 600M: general language quality consolidates

Logged validation perplexity improves another **24.3%**, handcrafted held-out perplexity improves **12.1%**, and mean diagnostic accuracy rises from 59.3% to 63.0%. Arithmetic/logic first reaches 50% at 600M. The answer-margin score also rises substantially, so even where binary accuracy is unchanged, the model is separating preferred continuations more confidently.

### 600M → 900M: strongest useful capability gain

This is the most important behavioral phase. Logged validation perplexity improves only **5.7%**, yet mean diagnostic accuracy rises from **63.0% to 69.7%**. Factual accuracy rises from 33.3% to 50%, commonsense from 66.7% to 75%, and lexical cloze from 75% to 83.3%. This suggests that small reductions in next-token uncertainty near this stage are sufficient to cross decision boundaries on more semantic tasks.

### 900M → 1100M: plateau, not clear overtraining

Logged validation perplexity still improves from **74.99 to 73.98 (-1.3%)**, but handcrafted held-out perplexity is essentially flat (161.37 vs 161.56) and the two checkpoints have the **same outcome on every one of the 66 behavioral items**. The 1100M checkpoint has a slightly larger mean answer margin, while 900M has a slightly lower handcrafted NLL.

A paired bootstrap over the 10 handcrafted held-out passages gives a 95% interval for the 1100M-minus-900M mean NLL difference of approximately **[-0.0107, +0.0130]**, which includes zero. Seven passages favor 900M and three favor 1100M. On this small test, that difference is not strong enough to call a real generalization regression.

## Why the final model still generates repetitive text

The top checkpoints are much better at *ranking* plausible continuations than at producing long greedy text. For example, 900M and 1100M both generate a loop beginning with:

> `The capital of France is the capital of the city of the city ...`

The mean repeated 3-gram fraction is 27.3% at both 900M and 1100M. The 600M checkpoint happens to be less repetitive on these six greedy prompts, but its semantic diagnostic accuracy is lower.

This is consistent with three constraints:

1. **Capacity:** the model has only ~9.96M unique parameters. Recursion increases effective depth to 18 layers but does not add storage capacity for more facts or lexical distinctions.
2. **Objective:** it is a next-token pretrained LM, not an instruction-tuned or preference-tuned assistant. A prompt such as “The capital of France is” is not guaranteed to be treated as a QA task.
3. **Greedy decoding:** once a small model enters a locally high-probability phrase loop, argmax decoding repeatedly reinforces it. Sampling, repetition penalties, or later instruction tuning can improve the visible generation behavior without changing the underlying checkpoint ranking.

## Remaining weaknesses at 1100M

The final checkpoint scores:

- Grammar: **18/20**
- Lexical cloze: **10/12**
- Facts: **6/12**
- Commonsense: **9/12**
- Arithmetic/logic: **5/10**

Persistent factual misses include Paris vs London, Mars vs Jupiter, Portuguese vs Spanish, and Indian vs Arctic in the diagnostic candidate-ranking setup. The arithmetic suite remains at 50%. These are capacity/data/objective limitations rather than evidence that the checkpoint failed to load or that recursion is malfunctioning.

The apparently higher 95% grammar score at 100M is a one-item discrete-test effect, not evidence that grammar degraded broadly. Later checkpoints have stronger average grammar margins; the extra persistent miss is the minimal pair `These flowers smell wonderful` vs `These flowers smells wonderful`.

## Recommendation

- **Use 1100M as the default strongest checkpoint** when the goal is lowest validation loss, continued training, fine-tuning, or broad LM quality.
- **Treat 900M as the capability-efficiency sweet spot.** On this suite it is statistically indistinguishable from 1100M and reaches the same 66-item behavioral decisions with ~200M fewer training tokens.
- **Do not switch to 600M solely because greedy generation repeats less.** Its semantic capability is weaker. A better next experiment is to keep 900M/1100M and improve decoding or instruction tuning.
- The most useful next evaluation would be standardized external tasks (for example WikiText-style perplexity plus compact versions of HellaSwag/PIQA/ARC/LAMBADA) and a decoding sweep with temperature/top-k/repetition penalty. That would distinguish model-quality gains from decoder-induced looping more rigorously.
