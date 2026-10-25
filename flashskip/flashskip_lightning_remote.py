#!/usr/bin/env python3
"""Remote Lightning Studio runner for the 1B-token FlashSkip-10M continuation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "models"
DATA = ROOT / "data"
RESUME = ROOT / "resume_200m"
STATUS = ROOT / "remote_status.json"
TRAINER = ROOT / "flashskip_10m.py"

START_TOKENS = 200_015_872
ADDITIONAL_TOKENS = 1_000_000_000
TARGET_TOKENS = START_TOKENS + ADDITIONAL_TOKENS
PREP_TRAIN_TOKENS = 1_210_000_000
PREP_VAL_TOKENS = 1_000_000
CHECKPOINT_TOKENS = 100_000_000
ORIGINAL_22M_TRAIN_SHA256 = "0997c490eb5f0fa279de527baec93606fe9d88999f59512386bb16d39c41dff0"
ORIGINAL_210M_TRAIN_SHA256 = "bfcfb643f5a9b021896091286f40303322fd09b8c00b58d2686dcaae893d11d7"
ORIGINAL_VAL_SHA256 = "e04d5c522333ca12aa64e9ac89a3c69319d07cc6bc3f88429a59b9b163e9dabe"


def write_status(phase: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps({
        "phase": phase,
        "start_tokens": START_TOKENS,
        "additional_target_tokens": ADDITIONAL_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "checkpoint_interval_tokens": CHECKPOINT_TOKENS,
        "updated_at": time.time(),
        **extra,
    }, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, STATUS)


def sha256_prefix(path: Path, byte_count: int | None = None) -> str:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as handle:
        while True:
            want = 8 << 20 if remaining is None else min(8 << 20, remaining)
            if want <= 0:
                break
            block = handle.read(want)
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise RuntimeError(f"File ended {remaining} bytes before requested prefix: {path}")
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_resume() -> dict:
    manifest = json.loads((RESUME / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("tokens_seen", 0)) != START_TOKENS:
        raise RuntimeError(f"Unexpected resume token count: {manifest.get('tokens_seen')}")
    for name, expected in manifest["files"].items():
        path = RESUME / name
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Resume size mismatch: {name}")
        if sha256_prefix(path) != expected["sha256"]:
            raise RuntimeError(f"Resume SHA-256 mismatch: {name}")
    return manifest


def validate_data(resume_manifest: dict) -> None:
    train = DATA / "train_tokens.uint16"
    val = DATA / "val_tokens.uint16"
    checks = {
        "val": sha256_prefix(val),
        "train_22m": sha256_prefix(train, 22_000_000 * 2),
        "train_210m": sha256_prefix(train, 210_000_000 * 2),
    }
    expected = {
        "val": ORIGINAL_VAL_SHA256,
        "train_22m": ORIGINAL_22M_TRAIN_SHA256,
        "train_210m": ORIGINAL_210M_TRAIN_SHA256,
    }
    if checks != expected:
        raise RuntimeError(f"The regenerated stream does not match the prior run: {checks}")
    cursor = int(resume_manifest["data_cursor"])
    remaining_steps = (ADDITIONAL_TOKENS + (256 * 512) - 1) // (256 * 512)
    required_cursor = cursor + remaining_steps * 256 * 513
    if required_cursor > PREP_TRAIN_TOKENS:
        raise RuntimeError(f"Prepared stream too short: needs {required_cursor:,}, has {PREP_TRAIN_TOKENS:,}")
    write_status(
        "data_verified",
        data_integrity="same_210m_prefix_verified",
        resume_data_cursor=cursor,
        required_final_cursor=required_cursor,
        prepared_train_tokens=PREP_TRAIN_TOKENS,
    )


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    resume_manifest = validate_resume()
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA"
    except Exception:
        gpu = "UNKNOWN"
    write_status("preparing_data", gpu=gpu, resume_checkpoint=RESUME.name)
    run([
        sys.executable, str(TRAINER), "prepare",
        "--output-dir", str(OUTPUT),
        "--data-dir", str(DATA),
        "--prepare-train-tokens", str(PREP_TRAIN_TOKENS),
        "--prepare-val-tokens", str(PREP_VAL_TOKENS),
    ])
    validate_data(resume_manifest)
    write_status("training", gpu=gpu, data_integrity="same_210m_prefix_verified")
    run([
        sys.executable, str(TRAINER), "train",
        "--output-dir", str(OUTPUT),
        "--data-dir", str(DATA),
        "--resume", str(RESUME),
        "--micro-batch", "256",
        "--grad-accum", "1",
        "--seq-len", "512",
        "--target-tokens", str(TARGET_TOKENS),
        "--checkpoint-tokens", str(CHECKPOINT_TOKENS),
        "--max-lr", "1.5e-4",
        "--min-lr", "1.5e-5",
        "--warmup-fraction", "0.05",
        "--log-every", "2",
        "--eval-batches", "24",
        "--eval-batch-size", "8",
    ])
    summary = json.loads((OUTPUT / "training_summary.json").read_text(encoding="utf-8"))
    write_status("completed", gpu=gpu, data_integrity="same_210m_prefix_verified", summary=summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
