#!/usr/bin/env python3
"""Run one 100M-token FlashSkip-10M continuation segment on Kaggle P100."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input/flashskip-10m-checkpoint")
WORK = Path("/kaggle/working/flashskip")
OUTPUT = WORK / "models"
DATA = WORK / "data"
RESUME = WORK / "resume"
FINAL_TOTAL_TOKENS = 500_015_872
SEGMENT_TOKENS = 100_000_000
MICRO_BATCH = 32
GRAD_ACCUM = 2
SEQ_LEN = 512
GLOBAL_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
SOURCE_PER_STEP = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)
KAGGLE_BASELINE_TOKENS = 305_609_792
KAGGLE_BASELINE_TRAIN_SHA256 = "d6aab3d3d3814c164753497d91cbd6518fa352a2ae522be5ae1f785f93f7efc0"
KAGGLE_BASELINE_VAL_SHA256 = "fd850c8c62194b31538d839a4900743218f92ed8766d7d7a668ca1f28594ac1e"


def emit(kind: str, **data: object) -> None:
    print("KAGGLE_STATUS", json.dumps({"phase": kind, "time": time.time(), **data}, sort_keys=True), flush=True)


def sha256(path: Path, byte_count: int | None = None) -> str:
    digest = hashlib.sha256(); remaining = byte_count
    with path.open("rb") as handle:
        while True:
            want = 8 << 20 if remaining is None else min(8 << 20, remaining)
            if want <= 0: break
            block = handle.read(want)
            if not block: break
            digest.update(block)
            if remaining is not None: remaining -= len(block)
    if remaining not in (None, 0): raise RuntimeError(f"short file: {path}")
    return digest.hexdigest()


def find_input() -> Path:
    if INPUT_ROOT.exists(): return INPUT_ROOT
    input_root = Path("/kaggle/input")
    for manifest in input_root.rglob("checkpoint_manifest.json"):
        return manifest.parent
    raise FileNotFoundError("FlashSkip checkpoint dataset is not mounted")


def install_pascal_torch() -> None:
    emit("installing_pascal_torch", package="torch==2.7.1+cu126")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
        "torch==2.7.1+cu126", "--index-url", "https://download.pytorch.org/whl/cu126",
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps",
        "nvidia-cusparselt-cu12==0.6.3",
    ], check=True)
    smoke = (
        "import json,torch; "
        "assert torch.cuda.is_available(); "
        "x=torch.ones(1024,device='cuda'); torch.cuda.synchronize(); "
        "print('P100_CUDA_OK',json.dumps({'name':torch.cuda.get_device_name(0),"
        "'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'sum':float(x.sum())}))"
    )
    subprocess.run([sys.executable, "-c", smoke], check=True)


def stage_input(source: Path) -> dict:
    WORK.mkdir(parents=True, exist_ok=True); RESUME.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "checkpoint_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = source / name
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"input checkpoint integrity failure: {name}")
        shutil.copy2(path, RESUME / name)
    shutil.copy2(source / "checkpoint_manifest.json", RESUME / "checkpoint_manifest.json")
    for name in ("flashskip_100m.py", "flashskip_10m.py"):
        shutil.copy2(source / name, WORK / name)
    tok = OUTPUT / "tokenizer"; tok.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source / name, tok / name)
    return manifest


def run(args: list[str]) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=WORK, check=True)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    source = find_input()
    manifest = stage_input(source)
    install_pascal_torch()
    resume_tokens = int(manifest["tokens_seen"])
    resume_step = int(manifest["step"])
    resume_cursor = int(manifest["data_cursor"])
    segment_target = min(FINAL_TOTAL_TOKENS, resume_tokens + SEGMENT_TOKENS)
    segment_steps = math.ceil((segment_target - resume_tokens) / GLOBAL_TOKENS)
    required_cursor = resume_cursor + segment_steps * SOURCE_PER_STEP
    prepare_tokens = required_cursor + 5_000_000
    emit("preparing_data", resume_tokens=resume_tokens, segment_target=segment_target, prepare_tokens=prepare_tokens)
    run([sys.executable, "flashskip_10m.py", "prepare", "--output-dir", str(OUTPUT), "--data-dir", str(DATA),
         "--prepare-train-tokens", str(prepare_tokens), "--prepare-val-tokens", "1000000"])
    train = DATA / "train_tokens.uint16"; val = DATA / "val_tokens.uint16"
    if sha256(val) != KAGGLE_BASELINE_VAL_SHA256:
        raise RuntimeError("Kaggle validation stream changed from the pinned continuation baseline")
    if prepare_tokens >= KAGGLE_BASELINE_TOKENS and sha256(train, KAGGLE_BASELINE_TOKENS * 2) != KAGGLE_BASELINE_TRAIN_SHA256:
        raise RuntimeError("Kaggle training stream changed from the pinned 305.61M-token prefix")
    emit("training", resume_tokens=resume_tokens, segment_target=segment_target, data_cursor=resume_cursor,
         data_integrity="verified", gpu="NVIDIA Tesla P100")
    run([sys.executable, "flashskip_10m.py", "train", "--output-dir", str(OUTPUT), "--data-dir", str(DATA),
         "--resume", str(RESUME), "--micro-batch", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM),
         "--seq-len", str(SEQ_LEN), "--target-tokens", str(segment_target), "--checkpoint-tokens", "100000000",
         "--max-lr", "1.0e-4", "--min-lr", "1.0e-5", "--warmup-fraction", "0.02",
         "--schedule-start-step", "6104", "--schedule-target-tokens", str(FINAL_TOTAL_TOKENS),
         "--log-every", "5", "--eval-batches", "24", "--eval-batch-size", "2"])
    # Keep only durable model artifacts in Kaggle output; the regenerated token stream is disposable.
    shutil.rmtree(DATA)
    checkpoints = sorted((OUTPUT / "checkpoints").glob("checkpoint_*"))
    latest = checkpoints[-1]
    out = Path("/kaggle/working/result") / latest.name
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(latest, out)
    shutil.copy2(OUTPUT / "logs" / "train_metrics.jsonl", Path("/kaggle/working/result/train_metrics.jsonl"))
    emit("completed", checkpoint=latest.name, tokens_seen=json.loads((latest / "metrics.json").read_text())["tokens_seen"])


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        emit("failed", error=f"{type(exc).__name__}: {exc}")
        raise
