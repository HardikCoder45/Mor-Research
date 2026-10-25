# Additional-billion-token continuation — 24 October 2026

Resume: 700,022,784-label MoR checkpoint, 9,965,344 parameters. All model/config/optimizer-state hashes were verified before setup. Original completed run state is archived in run/control_completed_700m_20261024.json. The 700M baseline evaluation is copied to run/reviews/baseline_700m_extension.json.

Requested additional labels: 1,000,000,000. Whole optimizer steps require 1,000,013,824 labels, giving a final cap of **1,700,036,608** total labels. Ten exact targets are recorded in run/control.json under segment_targets. Targets are rounded relative to the starting checkpoint, so the last segment actually meets the request for at least one billion additional labels.

## Fresh-token integrity

A separate CPU notebook prepares 1,704,000,000 raw uint16 training tokens, enough for all trained labels and packed-sequence boundary tokens. It regenerates the identical pinned FineWeb-Edu stream using the original revision, shuffle seed and tokenizer. Before any GPU job, it must prove that its first 1,002,000,000 training tokens have the exact SHA-256 of the old prepared data and that the document-disjoint validation stream is unchanged. Full new-file hashes are checked by the GPU launcher. Resume restores the saved exact cursor, optimizer moments, scaler and RNG states. Wrapping/replaying the stream is disallowed and data capacity is checked before training.

The data are deduplicated exactly by document hash across the full prepared stream. Near-duplicate removal and benchmark contamination freedom are not established. This continuation is general educational-text pretraining; it is not yet a task-specific instruction-distillation run.

## Schedule and evaluation

The new cosine schedule starts at the saved learning rate **0.00017554701690914522**, ending at **0.00002** across the additional billion labels. There is no restart warmup or learning-rate reset at segment boundaries. The previous 1B schedule is not simply stretched, which would raise the LR at the resume point. Architecture, vocabulary, width, recursion mechanism and context length remain unchanged.

Validation expands from 32 to 128 batches (131,072 labels per evaluation). Compare losses/PPL against the new run's initial_eval at the original 700M weights with the same 128 batches; do not directly attribute the metric change versus the historical 32-batch PPL of 70.48 to learning. Initial evaluations are in remote kernel logs and authenticated telemetry. Per-segment metrics record validation batch count and schedule settings.

Each approximately 100M checkpoint is downloaded to the Mac and SHA-256 verified before a review gate can release the next segment. The review checks validation loss/PPL, prose likelihood, all existing diagnostic categories, repetition, routing, gradients/AMP, GPU memory and speed. Mean diagnostic accuracy at the starting checkpoint is 72%; these are small, repeatedly reused handcrafted tests, not blind standardized benchmarks. A higher accuracy score or equivalence to a roughly 1B model is not guaranteed.

Stop or investigate before further GPU submission for corrupt/nonfinite state, >5% comparable validation-loss regression, two consecutive segments with <0.5% relative validation-PPL improvement, or persistent behavioral decline. Hold the review gate if accuracy remains >3 percentage points below the 700M reference on two successive checkpoints. Any change requires evidence, tests and documentation. Keep best and latest checkpoints separately.

## Orchestration

Kaggle refs, all private:

- Source: hardikarora83i2492/mor-10m-token-choice-source-v2
- Resume: hardikarora83i2492/mor-10m-token-choice-resume-v2
- CPU preparation: hardikarora83i2492/mor-10m-token-choice-data-1704m
- T4 continuation: hardikarora83i2492/mor-10m-700m-to-1700m

The previous training notebook/output remains preserved. CPU preparation may proceed while Mac disk space is low; GPU submissions and transfers wait below 768 MiB free. Approximately 3 GB free is recommended for ten new complete resume checkpoints and staging. The local supervisor uses a lock and caffeinate, but requires the Mac to remain running and connected. Heartbeat automation review-mor-billion-token-extension checks every 15 minutes, safely recovers stopped supervisors, reviews checkpoints, and notifies meaningful results/failures/actions. It must delete itself after final verified completion.

At the historical ~40K labels/s training rate, one billion labels would require roughly seven hours of actual training, excluding data preparation, larger validation, startup, queues, transfers, failures and review waits. This is an estimate, not a current-run measurement. Kaggle reported 30 GPU hours available at setup. A separate preparation kernel uses CPU rather than spending GPU quota on tokenization.

Tests passed: existing causality/forced-depth/router-gradient/tiny-batch learning/state serialization suite; unchanged-prefix hash verification and rejection of changed/short prefixes; sufficient stream capacity and ten target boundaries; continuation LR starts at the saved value and decays monotonically across segment boundaries. Old completed checkpoint files are not modified.

## Preparation integrity investigation

CPU preparation version 2 is running with the original immutable Kaggle Docker image (`sha256:dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9`), confirmed by pulling its remote metadata. The original and current tokenizer.json and tokenizer_config.json hashes match. Telemetry currently reports 1,000,611 validation tokens versus the original 1,000,917; this may include the earlier version, so treat the full output prefix/hash verification as authoritative. Do not bypass either prefix or validation hash checks. If preparation fails, investigate environment/dataset shard ordering before resubmission; GPU training must remain blocked. No new GPU training has yet been verified. Mac free space after user cleanup was about 1 GiB; checkpoint guard remains active.
