#!/usr/bin/env python3
"""Run three Kaggle P100 segments and mirror every 100M-token checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "flashskip10m_kaggle300m"
MODELS = RUN_ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
LOGS = RUN_ROOT / "logs"
STATUS = RUN_ROOT / "status.json"
PID_FILE = RUN_ROOT / "supervisor.pid"
STOP_FILE = RUN_ROOT / "supervisor.stop"
LOCK_FILE = RUN_ROOT / "supervisor.lock"
DATASET_STAGE = ROOT / "kaggle_dataset"
KERNEL_STAGE = ROOT / "kaggle_kernel"
SOURCE_CHECKPOINT = ROOT / "flashskip10m" / "models" / "checkpoints" / "checkpoint_step_006104_tokens_0200.02M"
KERNEL_REF = "hardikarora83i2492/flashskip-10m-p100-continuation"
DATASET_REF = "hardikarora83i2492/flashskip-10m-checkpoint"
START_TOKENS = 200_015_872
FINAL_TOKENS = 500_015_872
SEGMENT_TOKENS = 100_000_000
CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)_tokens_([0-9.]+)M")
TRAIN_RE = re.compile(r"TRAIN\s+(\{[^\n]+\})")
EVAL_RE = re.compile(r"(INITIAL_VAL|CHECKPOINT_EVAL)\s+(\{[^\n]+\})")
STATUS_RE = re.compile(r"KAGGLE_STATUS\s+(\{[^\n]+\})")
PREP_RE = re.compile(r"PREP docs=(\d+) val=([\d,]+) train=([\d,]+) tok/s=([\d,]+)")
POLL_SECONDS = 25


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (RUN_ROOT, MODELS, CHECKPOINTS, LOGS, DATASET_STAGE, KERNEL_STAGE):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def update_status(**fields: Any) -> None:
    data = read_json(STATUS)
    data.update({
        "model": "FlashSkip-10M", "parameters": 9_964_832,
        "provider": "Kaggle", "machine": "NVIDIA Tesla P100",
        "kernel": KERNEL_REF, "dataset": DATASET_REF,
        "start_tokens": START_TOKENS, "target_tokens": FINAL_TOKENS,
        "additional_target_tokens": 300_000_000,
        "checkpoint_interval_tokens": SEGMENT_TOKENS,
        "gpu_quota_hours_total": 30.0,
        "updated_at": now(),
        "architecture": {
            "vocab_size": 49152, "d_model": 160, "d_ff": 360, "layers": 10,
            "query_heads": 5, "kv_heads": 1, "attention_layers": [0,1,3,5,7,9],
            "sequence_length": 512, "micro_batch": 32, "grad_accum": 2,
            "global_tokens_per_step": 32768,
        },
        **fields,
    })
    atomic_json(STATUS, data)


def log(message: str) -> None:
    ensure_dirs()
    with (LOGS / "supervisor.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def command(args: list[str], timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20): digest.update(block)
    return digest.hexdigest()


def validate_checkpoint(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "checkpoint_manifest.json")
    if not manifest: raise RuntimeError(f"missing checkpoint manifest: {path}")
    for name, expected in manifest["files"].items():
        item = path / name
        if item.stat().st_size != int(expected["bytes"]) or sha256(item) != expected["sha256"]:
            raise RuntimeError(f"checkpoint integrity failure: {path.name}/{name}")
    return manifest


def latest_checkpoint() -> Path:
    candidates: list[tuple[int, Path]] = []
    for path in [SOURCE_CHECKPOINT, *CHECKPOINTS.glob("checkpoint_*")]:
        with contextlib.suppress(Exception):
            manifest = validate_checkpoint(path)
            candidates.append((int(manifest["tokens_seen"]), path))
    if not candidates: raise RuntimeError("No valid FlashSkip-10M checkpoint")
    return max(candidates, key=lambda item: item[0])[1]


def link_or_copy(source: Path, destination: Path) -> None:
    with contextlib.suppress(FileNotFoundError): destination.unlink()
    try: os.link(source, destination)
    except OSError: shutil.copy2(source, destination)


def stage_dataset(checkpoint: Path) -> None:
    manifest = validate_checkpoint(checkpoint)
    dynamic = ["model.safetensors", "training_state.pt", "config.json", "metrics.json", "checkpoint_manifest.json",
               "flashskip_100m.py", "flashskip_10m.py", "tokenizer.json", "tokenizer_config.json"]
    for name in dynamic:
        with contextlib.suppress(FileNotFoundError): (DATASET_STAGE / name).unlink()
    for name in manifest["files"]:
        link_or_copy(checkpoint / name, DATASET_STAGE / name)
    link_or_copy(checkpoint / "checkpoint_manifest.json", DATASET_STAGE / "checkpoint_manifest.json")
    link_or_copy(ROOT / "flashskip_100m.py", DATASET_STAGE / "flashskip_100m.py")
    link_or_copy(ROOT / "flashskip_10m.py", DATASET_STAGE / "flashskip_10m.py")
    tokenizer = ROOT / "models" / "mac_test" / "tokenizer"
    link_or_copy(tokenizer / "tokenizer.json", DATASET_STAGE / "tokenizer.json")
    link_or_copy(tokenizer / "tokenizer_config.json", DATASET_STAGE / "tokenizer_config.json")
    link_or_copy(ROOT / "kaggle_train_100m_segment.py", KERNEL_STAGE / "kaggle_train_100m_segment.py")


def dataset_exists() -> bool:
    result = command(["kaggle", "datasets", "status", DATASET_REF, "--format", "json"])
    return result.returncode == 0


def publish_dataset(checkpoint: Path, first: bool) -> None:
    stage_dataset(checkpoint)
    tokens = validate_checkpoint(checkpoint)["tokens_seen"]
    if first:
        command(["kaggle", "datasets", "create", "-p", str(DATASET_STAGE), "-r", "zip", "-q"], timeout=1800, check=True)
    else:
        command(["kaggle", "datasets", "version", "-p", str(DATASET_STAGE), "-r", "zip", "-q", "-m", f"Resume checkpoint at {tokens} tokens"], timeout=1800, check=True)
    for _ in range(60):
        status = command(["kaggle", "datasets", "status", DATASET_REF, "--format", "json"])
        if status.returncode == 0 and "ready" in status.stdout.lower(): return
        time.sleep(5)
    raise RuntimeError("Kaggle checkpoint dataset did not become ready")


def parse_logs(text: str) -> None:
    metrics: list[dict[str, Any]] = []
    for match in TRAIN_RE.finditer(text):
        with contextlib.suppress(Exception): metrics.append(json.loads(match.group(1)))
    for match in EVAL_RE.finditer(text):
        with contextlib.suppress(Exception):
            row = json.loads(match.group(2)); row["event"] = "initial_eval" if match.group(1) == "INITIAL_VAL" else "checkpoint_eval"; metrics.append(row)
    metrics.sort(key=lambda row: (int(row.get("step", 0)), 0 if row.get("event") == "train" else 1))
    if metrics:
        temp = LOGS / "train_metrics.jsonl.tmp"
        temp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics), encoding="utf-8")
        os.replace(temp, LOGS / "train_metrics.jsonl")
    statuses = []
    for match in STATUS_RE.finditer(text):
        with contextlib.suppress(Exception): statuses.append(json.loads(match.group(1)))
    if statuses:
        remote = statuses[-1]
        cuda_verified = bool(re.search(r"^P100_CUDA_OK\s+\{", text, re.MULTILINE))
        update_status(remote_phase=remote.get("phase"), data_integrity=remote.get("data_integrity"), cuda_verified=cuda_verified)
    preps = list(PREP_RE.finditer(text))
    if preps:
        match = preps[-1]
        target = next((int(s.get("prepare_tokens")) for s in reversed(statuses) if s.get("prepare_tokens")), 510_000_000)
        atomic_json(LOGS / "prepare_progress.json", {
            "documents": int(match.group(1)), "val_tokens": int(match.group(2).replace(",", "")),
            "train_tokens": int(match.group(3).replace(",", "")), "target_val_tokens": 1_000_000,
            "target_train_tokens": target, "tokens_per_second": int(match.group(4).replace(",", "")),
        })


def kernel_status() -> str:
    result = command(["kaggle", "kernels", "status", KERNEL_REF], timeout=120)
    return ((result.stdout or "") + (result.stderr or "")).strip()


def sync_logs() -> str:
    result = command(["kaggle", "kernels", "logs", KERNEL_REF], timeout=180)
    raw = (result.stdout or "") + (result.stderr or "")
    text = raw
    with contextlib.suppress(Exception):
        records = json.loads(result.stdout)
        if isinstance(records, list):
            text = "".join(str(record.get("data", "")) for record in records)
    (LOGS / "kaggle_kernel.log").write_text(text, encoding="utf-8")
    parse_logs(text)
    return text


def push_kernel(segment: int) -> None:
    with contextlib.suppress(FileNotFoundError): (LOGS / "kaggle_kernel.log").unlink()
    update_status(
        phase="submitting", segment=segment, kernel_status="submitting", last_error=None,
        remote_phase=None, cuda_verified=False, data_integrity=None,
    )
    command(["kaggle", "kernels", "push", "-p", str(KERNEL_STAGE), "--accelerator", "NvidiaTeslaP100", "--timeout", "43200"], timeout=1800, check=True)
    log(f"Submitted Kaggle P100 segment {segment}/3")


def monitor_kernel(segment: int) -> None:
    while not STOP_FILE.exists():
        status = kernel_status()
        text = sync_logs()
        lowered = status.lower()
        remote_phase = read_json(STATUS).get("remote_phase")
        update_status(phase=remote_phase or "running", segment=segment, kernel_status=status)
        if "complete" in lowered: return
        if any(word in lowered for word in ("error", "failed", "cancel")):
            raise RuntimeError(f"Kaggle kernel failed: {status}\n{text[-3000:]}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("stop requested")


def download_segment(segment: int) -> Path:
    output = RUN_ROOT / f"segment_{segment}_output"
    with contextlib.suppress(FileNotFoundError): shutil.rmtree(output)
    output.mkdir(parents=True)
    command(["kaggle", "kernels", "output", KERNEL_REF, "-p", str(output), "-o", "-q"], timeout=1800, check=True)
    found = []
    for manifest_path in output.rglob("checkpoint_manifest.json"):
        checkpoint = manifest_path.parent
        with contextlib.suppress(Exception): found.append((int(validate_checkpoint(checkpoint)["tokens_seen"]), checkpoint))
    if not found: raise RuntimeError("No valid checkpoint found in Kaggle kernel output")
    _, source = max(found, key=lambda item: item[0])
    destination = CHECKPOINTS / source.name
    if destination.exists(): shutil.rmtree(destination)
    shutil.move(str(source), destination)
    manifest = validate_checkpoint(destination)
    update_status(latest_checkpoint=destination.name, latest_tokens=manifest["tokens_seen"], latest_step=manifest["step"])
    log(f"Downloaded and SHA-verified {destination.name} to the Mac")
    shutil.rmtree(output)
    return destination


def run() -> None:
    ensure_dirs(); LOCK_FILE.touch(exist_ok=True)
    lock = LOCK_FILE.open("r+"); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    with contextlib.suppress(FileNotFoundError): STOP_FILE.unlink()
    for stale in (LOGS / "prepare_progress.json", LOGS / "train_metrics.jsonl", LOGS / "kaggle_kernel.log"):
        with contextlib.suppress(FileNotFoundError): stale.unlink()
    signal.signal(signal.SIGTERM, lambda *_: STOP_FILE.touch())
    signal.signal(signal.SIGINT, lambda *_: STOP_FILE.touch())
    update_status(
        state="running", phase="starting",
        latest_tokens=validate_checkpoint(latest_checkpoint())["tokens_seen"],
        last_error=None, remote_phase=None, cuda_verified=False, data_integrity=None,
    )
    try:
        checkpoint = latest_checkpoint()
        dataset_ready = dataset_exists()
        segment = max(1, round((validate_checkpoint(checkpoint)["tokens_seen"] - START_TOKENS) / SEGMENT_TOKENS) + 1)
        while validate_checkpoint(checkpoint)["tokens_seen"] < FINAL_TOKENS and not STOP_FILE.exists():
            manifest = validate_checkpoint(checkpoint)
            tokens = int(manifest["tokens_seen"])
            segment_target = min(FINAL_TOKENS, tokens + SEGMENT_TOKENS)
            segment_steps = math.ceil((segment_target - tokens) / 32_768)
            prepare_target = int(manifest["data_cursor"]) + segment_steps * 32_832 + 5_000_000
            update_status(prepare_target_tokens=prepare_target)
            if segment == 1 and tokens == START_TOKENS and dataset_ready:
                update_status(phase="staging_retry", segment=segment)
                stage_dataset(checkpoint)
            else:
                update_status(phase="publishing_checkpoint", segment=segment)
                publish_dataset(checkpoint, first=not dataset_ready)
                dataset_ready = True
            push_kernel(segment)
            monitor_kernel(segment)
            checkpoint = download_segment(segment)
            segment += 1
        if validate_checkpoint(checkpoint)["tokens_seen"] >= FINAL_TOKENS:
            update_status(state="completed", phase="completed")
            log("All three P100 segments completed and every 100M checkpoint is durable on the Mac")
        else:
            update_status(state="stopped", phase="stopped")
    except Exception as exc:
        update_status(state="failed", phase="failed", last_error=f"{type(exc).__name__}: {exc}")
        log(f"Supervisor failed: {type(exc).__name__}: {exc}")
        raise
    finally:
        with contextlib.suppress(FileNotFoundError): PID_FILE.unlink()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def kick() -> None:
    ensure_dirs()
    pid = int(PID_FILE.read_text()) if PID_FILE.exists() and PID_FILE.read_text().strip().isdigit() else 0
    if pid:
        try: os.kill(pid, 0); print(json.dumps({"already_running": True, "pid": pid})); return
        except OSError: pass
    handle = (LOGS / "launcher.log").open("a", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "run"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"started": True, "pid": process.pid}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("kick", "run", "status", "request-stop"))
    args = parser.parse_args()
    if args.command == "kick": kick()
    elif args.command == "run": run()
    elif args.command == "status": print(json.dumps(read_json(STATUS), indent=2))
    else: ensure_dirs(); STOP_FILE.touch(); print(json.dumps({"stop_requested": True}))


if __name__ == "__main__": main()
