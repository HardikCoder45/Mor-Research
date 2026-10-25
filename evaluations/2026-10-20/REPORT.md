# Local inference and evaluation — 20 October 2026

Same CPU FP32 backend, strict weight loading. Ten prose passages, 66 choice diagnostics, six greedy continuations up to 24 tokens. Small previously published diagnostic suite; not a standardized or blind benchmark.

| Model / trained tokens | Parameters | Prose PPL ↓ | Grammar | Cloze | Facts | Commonsense | Arithmetic | Category mean | Repeated 3-grams ↓ | CPU tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flashskip10m_1500m | 9,964,832 | 145.49 | 95.0% | 100.0% | 33.3% | 83.3% | 50.0% | 72.3% | 29.5% | 302.7 |
| recursive10m_1100m | 9,964,832 | 161.56 | 90.0% | 83.3% | 50.0% | 75.0% | 50.0% | 69.7% | 27.3% | 184.9 |
| mor10m_100m | 9,965,344 | 185.50 | 100.0% | 83.3% | 16.7% | 66.7% | 50.0% | 63.3% | 18.9% | 205.6 |
| flashskip100m_200m | 99,182,336 | 132.08 | 100.0% | 91.7% | 58.3% | 83.3% | 40.0% | 74.7% | 28.0% | 26.0 |

The 100M-parameter FlashSkip checkpoint leads prose perplexity and mean category accuracy. Use it as the quality baseline under the original <500M cap. The 10M non-recursive checkpoint leads the 10M families overall, while the recursive variant wins their factual subset. Training budgets differ; these results cannot isolate architecture effects.

## Actual greedy inference samples

### flashskip10m_1500m

Prompt: The capital of France is

```text
 the capital of the United States.
The capital of the United States is the capital of the United States.
The
```

### recursive10m_1100m

Prompt: The capital of France is

```text
 the capital of the city of the city of the city of the city of the city of the city of the city of
```

### mor10m_100m

Prompt: The capital of France is

```text
 the largest city of the world.
The city is the largest city of the country. It is the largest city of
```

### flashskip100m_200m

Prompt: The capital of France is

```text
 the capital of the city of the city of the city of the city of the city of the city of the city of
```

## Reproduce

```sh
flashskip/.venv-inference/bin/python evaluations/2026-10-20/evaluate_local.py
```

All raw item scores, outputs, timings and checkpoint hashes: results.json. Timing is short CPU generation with full-prefix recomputation, not a device serving benchmark.
