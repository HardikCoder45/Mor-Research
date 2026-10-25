# MoR-10M continuation results — 23 October 2026

The requested continuation reached **700,022,784 total trained labels**, adding **500,006,912 fresh labels** to the 200M checkpoint. The 9,965,344-parameter architecture, tokenizer, sequential data cursor and original learning-rate schedule were preserved. All five new checkpoints and their optimizer/resume states have been downloaded to the Mac and SHA-256 verified.

Language modeling improved, but the run did not demonstrate stronger overall diagnostic accuracy or parity with a 1B model. These are small, repeatedly used handcrafted diagnostics rather than a blind standardized test.

| Trained labels | Validation PPL ↓ | Diagnostic prose PPL ↓ | Mean category accuracy | Repeated trigrams ↓ |
|---:|---:|---:|---:|---:|
| 200,015,872 | 88.15 | 163.05 | 74.0% | 24.2% |
| 300,023,808 | 79.88 | 149.63 | 70.0% | 31.8% |
| 400,031,744 | 77.30 | 146.00 | 74.0% | 28.0% |
| 500,006,912 | 74.09 | 137.72 | 69.7% | 31.1% |
| 600,014,848 | 72.01 | 136.27 | 72.3% | 28.8% |
| 700,022,784 | 70.48 | 129.53 | 72.0% | 28.8% |

## Checkpoints to retain

- Best validation/prose language modeling and latest checkpoint: [700M checkpoint](/Users/hardikarora/Desktop/MAIN/research/mor10m/run/downloads/700000000/result/checkpoints/checkpoint_0700022784).
- Highest mean diagnostic accuracy: the 200M and 400M checkpoints tie at 74%; the 200M checkpoint also has stronger factual accuracy than the final model.
- Retain all five intermediate continuation checkpoints. Full paths and verification records are in [machine-readable summary](/Users/hardikarora/Desktop/MAIN/research/mor10m/run/continuation_final_summary.json).

## Final task scores

| Diagnostic | Final accuracy |
|---|---:|
| grammar | 100.0% |
| cloze | 83.3% |
| facts | 41.7% |
| commonsense | 75.0% |
| arithmetic | 60.0% |

## Measured training performance

| Segment | Effective labels/s | Median step labels/s | Peak CUDA GiB |
|---|---:|---:|---:|
| To 300M | 38,911 | 43,708 | 6.25 |
| To 400M | 41,179 | 43,694 | 6.16 |
| To 500M | 43,729 | 44,590 | 6.16 |
| To 600M | 40,055 | 44,791 | 6.19 |
| To 700M | 38,782 | 42,282 | 6.18 |

These rates are measured inside successful training segments; queue waits, downloads, restarts and failed submissions are excluded. All 3,055 logged successful-segment rows had finite loss, gradient norm and auxiliary loss. AMP overflow retries were not counted as trained labels.

## Actual final greedy inference

Prompt: `The capital of France is`

```text
 the capital of the United States.
The capital of the United States is the capital of the United States. The capital
```

The output is factually incorrect and repetitive. Reduced perplexity does not make this a useful general assistant. A next phase would need a defined specialization, domain data/distillation, and a fresh held-out evaluation against an exact roughly 1B comparator; no such training was launched.

The final gate authorizes completion only. No training beyond the nominal 700M target is approved. The supervisor confirmed completion and exited. The review heartbeat was deleted on completion.
