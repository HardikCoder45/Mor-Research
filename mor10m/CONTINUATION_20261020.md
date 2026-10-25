# Authorized MoR continuation — 20 October 2026

Resume checkpoint: run/recovered_latest/result/checkpoints/checkpoint_0200015872 (200,015,872 labels; all manifest SHA-256 hashes verified).
Target: nominal 700,000,000 total labels, rounded upward to whole 32,768-label optimizer steps. This adds approximately 500M fresh labels. Sequential cursor and original pinned data are retained; no stream replay. Original 1B-token learning-rate schedule is retained.

Supervisor: python3 mor10m/supervisor.py --launch from the research workspace. A file lock prevents duplicate local supervisors. Dashboard: http://127.0.0.1:8771 . Caffeinate prevents idle sleep while the supervisor runs; the Mac must remain running and connected. It does not restart automatically after reboot. The chat heartbeat can inspect and safely resume it when available.

P100 was retired by Kaggle; accelerator now configured as NvidiaTeslaT4. Runtime verifies assigned GPU names and runs a fresh smoke/gradient/causality test before training. Only device 0 trains; this is not multi-GPU DDP.

Five segment boundaries: nominal 300M, 400M, 500M, 600M and 700M. Each segment exits remotely, then full weights/FP32 optimizer/scaler/RNG/cursor/config/metrics are downloaded to run/downloads/TARGET and verified. No next submission occurs until the boundary review file is approved. Historical 100M checkpoint and recovered 200M checkpoint remain available.

Review heartbeat: review-mor-every-100m-tokens, every 15 minutes in this chat. It inspects state, evaluates a downloaded checkpoint with review_checkpoint.py, compares against run/reviews/baseline_200m.json and previous results, then writes run/reviews/TARGET.json with checkpoint_tokens, action continue/stop and reasons. Review cadence is therefore within the next scheduler wake after a boundary, not an exact token-triggered agent wake. The supervisor waits in review_pending until the review occurs.

Stops: nonfinite/corrupt result, >5% validation-loss regression, two consecutive segments without meaningful validation improvement, or other concerning instability requiring investigation. Final evaluation occurs before finishing. Best and latest weights should be distinguished. Fixed handcrafted diagnostics are regression indicators, not evidence of 1B-model parity.

Live metrics: run/logs/train_metrics.jsonl and run/telemetry_status.json. Only rows generated after this launch are current. Training logs report step/effective labels per second, loss, rolling loss, validation PPL, route fractions, mean recursion count, gradient norm, learning rate and peak CUDA memory. The synthetic smoke rate is a separate startup benchmark. See run/control.json for current phase and requested cap.

Checks passed before launch: parameter count 9,965,344, causality (including forced depths), sparse inactive rows finite, router gradients, tiny-batch learning, safetensors round trip and FP32 optimizer-state serialization. New supervisor/kernel/review/dashboard sources compile.
