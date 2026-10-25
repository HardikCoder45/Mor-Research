#!/usr/bin/env python3
"""Run the FlashSkip-10M 500M -> 1.5B continuation on one Kaggle P100."""
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

INPUT_ROOT = Path("/kaggle/input/flashskip-10m-1b-checkpoint")
WORK = Path("/kaggle/working/flashskip")
OUTPUT = WORK / "models"
DATA = WORK / "data"
RESUME = WORK / "resume"
RESULT = Path("/kaggle/working/result")

START_TOKENS = 500_039_680
START_STEP = 15_260
NO_REPLAY_CURSOR = 501_016_320
NEW_TOKEN_TARGET = 1_000_000_000
NOMINAL_FINAL_TOKENS = START_TOKENS + NEW_TOKEN_TARGET
MICRO_BATCH = 32          # proven P100-safe micro-batch
GRAD_ACCUM = 2            # preserves the prior 32,768-token effective global batch
SEQ_LEN = 512
GLOBAL_TOKENS = MICRO_BATCH * GRAD_ACCUM * SEQ_LEN
SOURCE_PER_STEP = MICRO_BATCH * GRAD_ACCUM * (SEQ_LEN + 1)
NEW_STEPS = math.ceil(NEW_TOKEN_TARGET / GLOBAL_TOKENS)
STEP_ALIGNED_FINAL_TOKENS = START_TOKENS + NEW_STEPS * GLOBAL_TOKENS
PREPARE_TRAIN_TOKENS = NO_REPLAY_CURSOR + NEW_STEPS * SOURCE_PER_STEP + 5_000_000
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
    for manifest in Path("/kaggle/input").rglob("checkpoint_manifest.json"):
        return manifest.parent
    raise FileNotFoundError("FlashSkip 500M checkpoint dataset is not mounted")


def ensure_p100_compatible_torch() -> None:
    """Pin a CUDA build with Pascal sm_60 kernels when Kaggle's default cu128 build omits them."""
    import torch
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if count != 1 or "P100" not in names[0].upper():
        raise RuntimeError(f"exactly 1xP100 required before torch bootstrap; allocated count={count} names={names}")
    arches = set(torch.cuda.get_arch_list())
    if "sm_60" in arches:
        print("P100_TORCH_OK", json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "arches": sorted(arches)}), flush=True)
        return
    if os.environ.get("FLASHSKIP_P100_TORCH_PINNED") == "1":
        raise RuntimeError(f"pinned torch still lacks sm_60 support: torch={torch.__version__} cuda={torch.version.cuda} arches={sorted(arches)}")
    print("P100_TORCH_BOOTSTRAP", json.dumps({"from_torch": torch.__version__, "from_cuda": torch.version.cuda, "arches": sorted(arches)}), flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "--force-reinstall",
        "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu126",
    ], check=True)
    env = os.environ.copy(); env["FLASHSKIP_P100_TORCH_PINNED"] = "1"
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())], env)


def verify_accelerator() -> dict:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if count != 1 or "P100" not in names[0].upper():
        raise RuntimeError(f"exactly 1xP100 required for this run; allocated count={count} names={names}")
    x = torch.ones(1024, device="cuda"); torch.cuda.synchronize()
    info = {"count": count, "names": names, "torch": torch.__version__, "cuda": torch.version.cuda,
            "capability": torch.cuda.get_device_capability(0), "sum": float(x.sum())}
    print("P100_CUDA_OK", json.dumps(info, sort_keys=True), flush=True)
    return info


def stage_input(source: Path) -> dict:
    WORK.mkdir(parents=True, exist_ok=True); RESUME.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "checkpoint_manifest.json").read_text())
    if int(manifest.get("tokens_seen", -1)) != START_TOKENS or int(manifest.get("step", -1)) != START_STEP:
        raise RuntimeError(f"wrong starting checkpoint: {manifest.get('tokens_seen')} tokens step {manifest.get('step')}")
    if int(manifest.get("data_cursor", -1)) != NO_REPLAY_CURSOR:
        raise RuntimeError(f"wrong no-replay cursor: {manifest.get('data_cursor')} != {NO_REPLAY_CURSOR}")
    for name, expected in manifest["files"].items():
        path = source / name
        if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"input checkpoint integrity failure: {name}")
        shutil.copy2(path, RESUME / name)
    shutil.copy2(source / "checkpoint_manifest.json", RESUME / "checkpoint_manifest.json")
    for name in ("flashskip_100m.py", "flashskip_10m.py"):
        shutil.copy2(source / name, WORK / name)
    tok = OUTPUT / "tokenizer"; tok.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(source / name, tok / name)
    telemetry = source / "telemetry.json"
    if telemetry.exists():
        cfg = json.loads(telemetry.read_text())
        if cfg.get("url") and cfg.get("token"):
            os.environ["FLASHSKIP_TELEMETRY_URL"] = str(cfg["url"])
            os.environ["FLASHSKIP_TELEMETRY_TOKEN"] = str(cfg["token"])
    return manifest


def run(args: list[str]) -> None:
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=WORK, check=True)


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    source = find_input()
    manifest = stage_input(source)
    ensure_p100_compatible_torch()
    accel = verify_accelerator()
    emit("preparing_data", resume_tokens=START_TOKENS, resume_step=START_STEP,
         data_cursor=NO_REPLAY_CURSOR, prepare_train_tokens=PREPARE_TRAIN_TOKENS,
         nominal_final_tokens=NOMINAL_FINAL_TOKENS, step_aligned_final_tokens=STEP_ALIGNED_FINAL_TOKENS,
         accelerator=accel, data_integrity="checkpoint_verified")

    run([sys.executable, "flashskip_10m.py", "prepare", "--output-dir", str(OUTPUT), "--data-dir", str(DATA),
         "--prepare-train-tokens", str(PREPARE_TRAIN_TOKENS), "--prepare-val-tokens", "1000000"])
    train = DATA / "train_tokens.uint16"; val = DATA / "val_tokens.uint16"
    if sha256(val) != KAGGLE_BASELINE_VAL_SHA256:
        raise RuntimeError("validation stream changed from pinned continuation baseline")
    if sha256(train, KAGGLE_BASELINE_TOKENS * 2) != KAGGLE_BASELINE_TRAIN_SHA256:
        raise RuntimeError("training stream changed from pinned continuation baseline")
    emit("data_verified", train_tokens=PREPARE_TRAIN_TOKENS, data_cursor=NO_REPLAY_CURSOR,
         data_integrity="verified_no_replay", machine="NVIDIA Tesla P100")

    emit("training", resume_tokens=START_TOKENS, target_tokens=NOMINAL_FINAL_TOKENS,
         new_token_target=NEW_TOKEN_TARGET, checkpoint_interval_tokens=100_000_000,
         data_cursor=NO_REPLAY_CURSOR, data_integrity="verified_no_replay", machine="NVIDIA Tesla P100")
    run([sys.executable, "flashskip_10m.py", "train", "--output-dir", str(OUTPUT), "--data-dir", str(DATA),
         "--resume", str(RESUME), "--require-resume-cursor-at-least", str(NO_REPLAY_CURSOR),
         "--micro-batch", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM),
         "--seq-len", str(SEQ_LEN), "--target-tokens", str(NOMINAL_FINAL_TOKENS),
         "--checkpoint-tokens", "100000000", "--checkpoint-origin-tokens", str(START_TOKENS),
         "--max-lr", "6.0e-5", "--min-lr", "5.0e-6",
         "--warmup-fraction", "0.01", "--schedule-start-step", str(START_STEP),
         "--schedule-target-tokens", str(NOMINAL_FINAL_TOKENS), "--log-every", "5",
         "--eval-batches", "24", "--eval-batch-size", "4"])

    # The 1.5B prepared stream is disposable. Persist every ~100M checkpoint, not the token file.
    shutil.rmtree(DATA, ignore_errors=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    out_ckpts = RESULT / "checkpoints"; out_ckpts.mkdir(parents=True, exist_ok=True)
    copied = []
    for checkpoint in sorted((OUTPUT / "checkpoints").glob("checkpoint_*")):
        metrics = json.loads((checkpoint / "metrics.json").read_text())
        if int(metrics.get("tokens_seen", 0)) <= START_TOKENS: continue
        destination = out_ckpts / checkpoint.name
        shutil.copytree(checkpoint, destination, dirs_exist_ok=True)
        copied.append({"name": checkpoint.name, "tokens_seen": int(metrics["tokens_seen"])})
    shutil.copy2(OUTPUT / "logs" / "train_metrics.jsonl", RESULT / "train_metrics.jsonl")
    summary = {"start_tokens": START_TOKENS, "nominal_new_tokens": NEW_TOKEN_TARGET,
               "nominal_final_tokens": NOMINAL_FINAL_TOKENS, "step_aligned_final_tokens": STEP_ALIGNED_FINAL_TOKENS,
               "no_replay_cursor": NO_REPLAY_CURSOR, "checkpoints": copied, "accelerator": accel}
    (RESULT / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    emit("completed", **summary)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        emit("failed", error=f"{type(exc).__name__}: {exc}")
        raise
