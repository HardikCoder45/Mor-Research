# FlashSkip-100M — 200M-token Colab continuation

This directory contains the durable Mac-side training supervisor for continuing the 99,182,336-parameter FlashSkip-100M model from its 10.027M-token checkpoint to approximately 200M total FineWeb-Edu tokens.

## What runs where

- **Mac:** durable checkpoints, supervisor process, logs, status, tokenizer/source code.
- **Google Colab T4:** disposable training compute and temporary tokenized FineWeb-Edu data.
- **Remote interface:** the official `colab` CLI only (`new`, `status`, `install`, `upload`, `exec`, `ls`, `download`, `stop`, `sessions`).
- No notebook UI or browser automation is used for training.

## Training configuration

- Model: FlashSkip-100M, exactly **99,182,336 learned parameters**.
- Resume anchor: **10,027,008 tokens / optimizer step 306**.
- Dataset: `HuggingFaceFW/fineweb-edu`, `sample-10BT`, seed `20260830`.
- Expanded prepared stream: **210M train + 1M held-out tokens**.
- Target: **200M total training tokens**.
- Sequence length: 512.
- Micro-batch: 16.
- Gradient accumulation: 4.
- Effective batch: 32,768 next-token labels per optimizer step.
- Optimizer: fused AdamW, betas 0.9/0.95, weight decay 0.1.
- Mixed precision: FP16 + GradScaler on Tesla T4.
- Gradient clipping: 1.0.
- Continued LR schedule: **3.5e-4 → 3.5e-5 cosine, 5% warmup**. At resume step 306 this is intentionally close to the prior checkpoint LR (3.526e-4), avoiding a large LR discontinuity.
- Checkpoint cadence: about every **20M total tokens**.

The supervisor regenerates the expanded FineWeb-Edu stream with the original seed/tokenizer and refuses to trust the old data cursor until both of these integrity checks match the original pilot exactly:

- first 22M training tokens SHA-256: `0997c490eb5f0fa279de527baec93606fe9d88999f59512386bb16d39c41dff0`
- 1M held-out tokens SHA-256: `e04d5c522333ca12aa64e9ac89a3c69319d07cc6bc3f88429a59b9b163e9dabe`

## Start / monitor / stop

From `/Users/hardikarora/Desktop/MAIN`:

```bash
python3 research/flashskip/colab_supervisor.py kick
python3 research/flashskip/colab_supervisor.py status
```

The `kick` command detaches one supervisor process. A file lock prevents a second supervisor from starting concurrently.

Request a graceful stop:

```bash
python3 research/flashskip/colab_supervisor.py request-stop
```

This asks the supervisor to terminate its active `colab exec`, release the Colab session, and retain the newest locally mirrored exact-resume state.

Manually pull any completed checkpoint visible in the active Colab session:

```bash
python3 research/flashskip/colab_supervisor.py mirror-once
```

## Logs and status

- Supervisor lifecycle: `research/flashskip/logs/supervisor.log`
- Raw Colab training CLI output: `research/flashskip/logs/colab_train_cli.log`
- Dataset preparation output: `research/flashskip/logs/data_prepare.log`
- Mirrored trainer metrics: `research/flashskip/logs/train_metrics.jsonl`
- Machine-readable supervisor state: `research/flashskip/supervisor_status.json`
- PID: `research/flashskip/supervisor.pid`

Useful commands:

```bash
tail -f research/flashskip/logs/supervisor.log
tail -f research/flashskip/logs/colab_train_cli.log
colab --auth=oauth2 sessions
colab --auth=oauth2 status -s flashskip200m
```

## Checkpoint storage policy

The Mac had only about 3.4 GiB free when this continuation was configured. A full FlashSkip checkpoint consists of roughly:

- ~189 MB FP16 `model.safetensors`
- ~378 MB exact optimizer/GradScaler/RNG/data-cursor `training_state.pt`

Keeping ten complete optimizer-state copies would exceed available disk. Therefore the durable layout is:

```text
research/flashskip/models/
├── checkpoints/
│   ├── checkpoint_step_000306_tokens_0010.03M/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   ├── metrics.json
│   │   └── training_state.pt       # permanent 10M recovery anchor
│   ├── checkpoint_~20M/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── metrics.json
│   ├── checkpoint_~40M/
│   │   └── ...
│   └── ...
├── resume_latest/
│   ├── training_state.pt           # exact state for newest mirrored model
│   └── resume_latest.json          # points to matching checkpoint model
└── final/
    └── hard links to final model/state where filesystem permits
```

Every 20M model checkpoint remains loadable and suitable as a weights checkpoint. The newest mirrored model additionally has its exact optimizer/scaler/RNG/data-cursor state in `resume_latest`, so a reclaimed Colab runtime can resume without resetting momentum. The original 10.03M full state remains as a second recovery anchor.

## Runtime recovery behavior

If Colab is reclaimed or a CLI training execution fails:

1. the supervisor keeps all locally mirrored checkpoints;
2. it releases/forgets the disposable failed session;
3. after a conservative retry delay it requests a new T4;
4. it installs the required packages using `colab install`;
5. it uploads the trainer and newest exact-resume model/state;
6. it deterministically regenerates the 210M-token FineWeb-Edu stream;
7. it verifies the old-data prefix hashes;
8. it resumes from the saved optimizer step and data cursor.

It never intentionally starts a second training process when the named session reports `BUSY`.

## Completion behavior

After a locally mirrored checkpoint reaches or exceeds 200M total tokens, the supervisor:

1. creates `models/final/` using hard links where possible;
2. runs the final evaluation suite on the still-live T4 (held-out FineWeb-Edu, WikiText-2, HellaSwag, ARC-Easy, BoolQ, generation samples, and prefill timing as supported by the trainer);
3. downloads the evaluation report/results;
4. stops the Colab session;
5. records `state=completed` in `supervisor_status.json`.

A 200M-token run is still an **early pretraining experiment** for a 99M-parameter language model. It is far more meaningful than the 10M-token pilot, but it should not be treated as the final capability ceiling of the architecture.

## Local streaming inference UI

The compact local UI automatically loads the checkpoint with the highest
`tokens_seen` value, uses the saved SmolLM2 tokenizer, and streams decoded tokens
to the browser as newline-delimited JSON.

```bash
cd /Users/hardikarora/Desktop/MAIN/research/flashskip
./start_inference_ui.sh
```

Open `http://127.0.0.1:7860`. The server uses CUDA, then Apple Metal (MPS), then
CPU in that order. Pass `--port 8000` to use a different port.
