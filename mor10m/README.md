# MoR-10M / token-choice recursion

Paper: Bae et al., *Mixture-of-Recursions: Learning Dynamic Recursive Depths
for Adaptive Token-Level Computation*, NeurIPS 2025, arXiv:2507.10524.
https://arxiv.org/abs/2507.10524
Reference implementation: https://github.com/raymin0223/mixture_of_recursions
Local paper: `../papers/mixture_of_recursions_2507.10524.pdf`.

This is an independent small-scale adaptation, not a faithful reproduction of
the paper's large-model experiments and not the old fixed-depth FlashSkip.
Quality improvement at 10M parameters has not been established.

## Architecture

- Exactly **9,965,344 unique trainable parameters**, including the router.
- 49,152 vocabulary; tied embedding/output weights; width 160; SwiGLU 496.
- Seven unique full attention/MLP blocks: unique first and last; five shared
  middle blocks applied one, two, or three times (7 / 12 / 17 effective depth).
- Top-1 token-choice routing from the causal prefix block; differentiable
  selected-probability output weighting plus the prefix residual, as in the
  authors' token-choice implementation. Unfinished tokens pass unweighted
  hidden states to the next shared pass.
- Recursion-wise packed attention: exited tokens have no deeper K/V entries.
  Padding is masked; original RoPE positions retained; no sequence-wide top-k
  decisions that can leak future information. Actual active tokens are packed
  and scattered, not merely run through dense layers and masked afterward.
- Router load-balancing loss (0.01) and z-loss (0.001); routing fractions logged.
- GQA 5 query / 1 KV heads, head dimension 32, RMSNorm/QK-norm, context 512.
- Incremental generation KV caching and the paper's specialized serving
  scheduler are not implemented. No 2x speed or large-model quality claim.

## Training and integrity

Fresh random initialization; no incompatible FlashSkip weights loaded.
FineWeb-Edu `sample-10BT`, pinned revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, seed 20260904.
Exact duplicate documents rejected by SHA-256. Whole-document holdout prevents
the same document from straddling validation and training. This does not
establish absence of near-duplicates or benchmark contamination.

1,000,000,000 target training labels, with a final whole-step rounding to
1,000,013,824 labels. Packed boundary tokens are additional source tokens,
not counted as trained labels. No wrapping/replaying the training stream.

AdamW, beta=(0.9,0.95), 2% warmup, cosine 6e-4 to 6e-5 across the entire 1B
schedule, global batch 32,768, gradient clip 1, FP16 AMP on the verified P100.
Overflow batches are retried rather than incorrectly counted as trained.
Validation every 10M labels (~32K validation labels per check). Checkpoints
every nominal 100M boundary, rounded up to optimizer steps, with full FP32
weights/optimizer moments, scaler/RNG state, cursor, config, and hashes.
Segments are downloaded and SHA-256 verified on the Mac before the next one.

Data preparation is a reusable CPU Kaggle kernel. A separate short P100
validation/throughput kernel must pass before the first training segment.
The local supervisor checks GPU quota before each segment and pauses until
the quota refresh if insufficient. It must stay running on a connected Mac;
caffeinate inhibits idle sleep while it runs. It does not survive a reboot
automatically. Restart `supervisor.py` to inspect/resume recorded progress.

## Commands

From the research workspace:

```
PYTHONPATH=flashskip flashskip/.venv-inference/bin/python mor10m/tests.py
python3 mor10m/supervisor.py --launch
```

Dashboard: http://127.0.0.1:8771 . Remote POSTs require a secret token; the
public tunnel rejects dashboard GETs. Secrets live only in local runtime
files and the private Kaggle source dataset, not in the source code.

Current state: `run/control.json`; logs: `run/supervisor.log`;
checkpoints: `run/downloads/<nominal-token-boundary>/result/checkpoints/`.
All earlier FlashSkip runs and models remain unchanged.
