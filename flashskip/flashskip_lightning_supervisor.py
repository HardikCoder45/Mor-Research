#!/usr/bin/env python3
"""Mirror Lightning Studio telemetry/checkpoints and release compute on completion."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
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
RUN_ROOT = ROOT / "flashskip10m_1b"
MODELS = RUN_ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
LOGS = RUN_ROOT / "logs"
STATUS = RUN_ROOT / "status.json"
PID_FILE = RUN_ROOT / "supervisor.pid"
STOP_FILE = RUN_ROOT / "supervisor.stop"
LOCK_FILE = RUN_ROOT / "supervisor.lock"

TEAMSPACE = "arorahardik344/vision-model"
STUDIO = "flashskip10m-1b"
REMOTE_ROOT = f"lit://{TEAMSPACE}/studios/{STUDIO}/flashskip1b"
START_TOKENS = 200_015_872
TARGET_TOKENS = 1_200_015_872
CHECKPOINT_INTERVAL = 100_000_000
CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)_tokens_([0-9.]+)M")
POLL_SECONDS = 20


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (RUN_ROOT, MODELS, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def update_status(**fields: Any) -> None:
    status = read_json(STATUS)
    status.update({
        "model": "FlashSkip-10M",
        "parameters": 9_964_832,
        "provider": "Lightning AI",
        "machine": "H100 80GB",
        "hourly_cost_usd": 4.50,
        "studio": STUDIO,
        "teamspace": TEAMSPACE,
        "start_tokens": START_TOKENS,
        "target_tokens": TARGET_TOKENS,
        "additional_target_tokens": 1_000_000_000,
        "checkpoint_interval_tokens": CHECKPOINT_INTERVAL,
        "updated_at": now(),
        "architecture": {
            "vocab_size": 49152, "d_model": 160, "d_ff": 360, "layers": 10,
            "query_heads": 5, "kv_heads": 1, "attention_layers": [0,1,3,5,7,9],
            "sequence_length": 512, "micro_batch": 256, "grad_accum": 1,
            "global_tokens_per_step": 131072,
        },
        **fields,
    })
    atomic_json(STATUS, status)


def log(message: str) -> None:
    ensure_dirs()
    with (LOGS / "supervisor.log").open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def command(args: list[str], timeout: int = 300, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def copy_remote(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    result = command(["lightning", "cp", remote, str(local)], timeout=600)
    return result.returncode == 0 and local.exists()


def sync_file(remote_name: str, local: Path) -> None:
    temp = local.with_suffix(local.suffix + ".tmp")
    with contextlib.suppress(FileNotFoundError):
        temp.unlink()
    if copy_remote(f"{REMOTE_ROOT}/{remote_name}", temp):
        os.replace(temp, local)


def remote_checkpoint_names() -> list[str]:
    result = command(["lightning", "ls", f"{REMOTE_ROOT}/models/checkpoints"], timeout=180)
    text = (result.stdout or "") + (result.stderr or "")
    names = set(match.group(0) for match in CHECKPOINT_RE.finditer(text))
    return sorted(names, key=lambda name: int(CHECKPOINT_RE.fullmatch(name).group(1)))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def mirror_checkpoint(name: str) -> None:
    destination = CHECKPOINTS / name
    if (destination / "checkpoint_manifest.json").exists():
        return
    stage = CHECKPOINTS / (name + ".partial")
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(stage)
    result = command(["lightning", "cp", "-r", f"{REMOTE_ROOT}/models/checkpoints/{name}", str(stage)], timeout=1800)
    if result.returncode:
        raise RuntimeError(f"checkpoint download failed: {name}: {result.stderr or result.stdout}")
    # The CLI may preserve the source directory or copy its contents directly.
    nested = stage / name
    source = nested if nested.is_dir() else stage
    manifest = read_json(source / "checkpoint_manifest.json")
    if not manifest:
        raise RuntimeError(f"checkpoint manifest missing after download: {name}")
    for filename, expected in manifest["files"].items():
        path = source / filename
        if path.stat().st_size != int(expected["bytes"]) or file_sha(path) != expected["sha256"]:
            raise RuntimeError(f"checkpoint integrity mismatch: {name}/{filename}")
    if source != stage:
        os.replace(source, destination)
        shutil.rmtree(stage)
    else:
        os.replace(stage, destination)
    update_status(latest_checkpoint=name, latest_tokens=manifest["tokens_seen"], latest_step=manifest["step"])
    log(f"Downloaded and verified checkpoint {name}")


def stop_studio() -> None:
    command(["lightning", "studio", "stop", "--name", STUDIO, "--teamspace", TEAMSPACE, "--json"], timeout=300)


def run() -> None:
    ensure_dirs(); LOCK_FILE.touch(exist_ok=True)
    lock = LOCK_FILE.open("r+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    with contextlib.suppress(FileNotFoundError): STOP_FILE.unlink()
    update_status(state="running", phase="connecting", last_error=None)
    signal.signal(signal.SIGTERM, lambda *_: STOP_FILE.touch())
    signal.signal(signal.SIGINT, lambda *_: STOP_FILE.touch())
    try:
        while not STOP_FILE.exists():
            sync_file("remote_status.json", LOGS / "remote_status.json")
            sync_file("models/logs/prepare_progress.json", LOGS / "prepare_progress.json")
            sync_file("models/logs/train_metrics.jsonl", LOGS / "train_metrics.jsonl")
            sync_file("remote_training.log", LOGS / "remote_training.log")
            remote = read_json(LOGS / "remote_status.json")
            phase = remote.get("phase", "connecting")
            update_status(
                phase=phase,
                state="failed" if phase == "failed" else "running",
                gpu=remote.get("gpu"),
                data_integrity=remote.get("data_integrity"),
                last_error=remote.get("error"),
            )
            for name in remote_checkpoint_names():
                mirror_checkpoint(name)
            if phase == "completed":
                sync_file("models/training_summary.json", MODELS / "training_summary.json")
                stop_studio()
                update_status(state="completed", phase="completed", studio_stopped=True)
                log("One-billion-token continuation completed; Lightning Studio stopped")
                return
            if phase == "failed":
                stop_studio()
                update_status(state="failed", studio_stopped=True)
                log(f"Remote runner failed: {remote.get('error')}")
                return
            time.sleep(POLL_SECONDS)
        stop_studio(); update_status(state="stopped", phase="stopped", studio_stopped=True)
    finally:
        with contextlib.suppress(FileNotFoundError): PID_FILE.unlink()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN); lock.close()


def kick() -> None:
    ensure_dirs()
    pid = int(PID_FILE.read_text()) if PID_FILE.exists() and PID_FILE.read_text().strip().isdigit() else 0
    if pid:
        try:
            os.kill(pid, 0); print(json.dumps({"already_running": True, "pid": pid})); return
        except OSError:
            pass
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
    else:
        ensure_dirs(); STOP_FILE.touch(); print(json.dumps({"stop_requested": True}))


if __name__ == "__main__":
    main()
