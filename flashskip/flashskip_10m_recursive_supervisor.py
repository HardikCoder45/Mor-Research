#!/usr/bin/env python3
"""Durable Colab-CLI supervisor for FlashSkip-10M Recursive's 100M-token run."""

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
BASE_TRAINER = ROOT / "flashskip_100m.py"
TRAINER = ROOT / "flashskip_10m_recursive.py"
RUN_ROOT = ROOT / "flashskip10m_recursive100m"
MODELS = RUN_ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
LOGS = RUN_ROOT / "logs"
STATUS = RUN_ROOT / "status.json"
PID_FILE = RUN_ROOT / "supervisor.pid"
LOCK_FILE = RUN_ROOT / "supervisor.lock"
STOP_FILE = RUN_ROOT / "supervisor.stop"

SESSION = "flashskip10mrec100m"
REMOTE_ROOT = "/content/flashskip10mrec100m"
REMOTE_MODELS = f"{REMOTE_ROOT}/models"
REMOTE_DATA = f"{REMOTE_ROOT}/data"
REMOTE_CHECKPOINTS = f"{REMOTE_MODELS}/checkpoints"
REMOTE_BASE = f"{REMOTE_ROOT}/flashskip_100m.py"
REMOTE_TRAINER = f"{REMOTE_ROOT}/flashskip_10m_recursive.py"
REMOTE_RESUME = f"{REMOTE_ROOT}/resume"

TARGET_TOKENS = 100_000_000
PREP_TRAIN_TOKENS = 110_000_000
PREP_VAL_TOKENS = 1_000_000
CHECKPOINT_TOKENS = 20_000_000
SEQ_LEN = 512
MICRO_BATCH = 16
GRAD_ACCUM = 4
PARAMETERS = 9_964_832
MAX_LR = 6e-4
MIN_LR = 6e-5
POLL_SECONDS = 15
RETRY_SECONDS = 60
TRAIN_TIMEOUT = 21_600

ORIGINAL_VAL_SHA256 = "e04d5c522333ca12aa64e9ac89a3c69319d07cc6bc3f88429a59b9b163e9dabe"
ORIGINAL_22M_TRAIN_SHA256 = "0997c490eb5f0fa279de527baec93606fe9d88999f59512386bb16d39c41dff0"
ORIGINAL_210M_TRAIN_SHA256 = "bfcfb643f5a9b021896091286f40303322fd09b8c00b58d2686dcaae893d11d7"
CHECKPOINT_RE = re.compile(r"checkpoint_step_(\d+)_tokens_([0-9.]+)M")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (RUN_ROOT, MODELS, CHECKPOINTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def update_status(**fields: Any) -> None:
    data = read_json(STATUS)
    data.update({
        "model": "FlashSkip-10M-Recursive-MiddleCycle-Nr2",
        "parameters": PARAMETERS,
        "target_tokens": TARGET_TOKENS,
        "session": SESSION,
        "updated_at": now(),
        **fields,
    })
    atomic_json(STATUS, data)


def log(message: str) -> None:
    ensure_dirs()
    line = f"[{now()}] {message}"
    with (LOGS / "supervisor.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def colab(*args: str) -> list[str]:
    executable = shutil.which("colab") or str(Path.home() / ".local/bin/colab")
    return [executable, "--auth=oauth2", *args]


def command(args: list[str], timeout: int = 300, check: bool = True, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def sessions() -> str:
    result = command(colab("sessions"), timeout=90, check=False)
    return (result.stdout or "") + (result.stderr or "")


def session_exists() -> bool:
    return f"[{SESSION}]" in sessions()


def session_status() -> str:
    result = command(colab("status", "-s", SESSION), timeout=90, check=False)
    return ((result.stdout or "") + (result.stderr or "")).strip()


def stop_session() -> None:
    if session_exists():
        log(f"Releasing Colab session {SESSION}")
        command(colab("stop", "-s", SESSION), timeout=180, check=False)


def remote_exec(code: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return command(colab("exec", "-s", SESSION, "--timeout", str(timeout)), timeout=timeout + 90, input_text=code)


def upload(local: Path, remote: str) -> None:
    size = local.stat().st_size
    if size < 24 * 1024 * 1024:
        command(colab("upload", "-s", SESSION, str(local), remote), timeout=600)
        return
    digest = file_sha(local)
    parts_dir = f"{REMOTE_ROOT}/.parts/{local.name}-{digest[:10]}"
    remote_exec(f"from pathlib import Path\nPath({parts_dir!r}).mkdir(parents=True,exist_ok=True)\n", 120)
    temp = RUN_ROOT / ".upload-part"
    count = 0
    try:
        with local.open("rb") as source:
            while block := source.read(12 * 1024 * 1024):
                temp.write_bytes(block)
                command(colab("upload", "-s", SESSION, str(temp), f"{parts_dir}/p{count:04d}"), timeout=600)
                count += 1
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()
    code = f"""from pathlib import Path
import hashlib,os,shutil
parts=Path({parts_dir!r}); target=Path({remote!r}); target.parent.mkdir(parents=True,exist_ok=True)
tmp=target.with_suffix(target.suffix+'.tmp'); h=hashlib.sha256()
with tmp.open('wb') as out:
    for p in sorted(parts.iterdir()):
        b=p.read_bytes(); out.write(b); h.update(b)
assert h.hexdigest()=={digest!r}
os.replace(tmp,target); shutil.rmtree(parts); print('UPLOAD_OK',target)
"""
    remote_exec(code, 600)


def download(remote: str, local: Path, check: bool = True) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    result = command(colab("download", "-s", SESSION, remote, str(local)), timeout=900, check=False)
    if check and result.returncode:
        raise RuntimeError(f"download failed: {remote}\n{result.stdout}\n{result.stderr}")
    return result.returncode == 0 and local.exists()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def latest_local_checkpoint() -> Path | None:
    found = []
    for path in CHECKPOINTS.glob("checkpoint_*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match and (path / "checkpoint_manifest.json").exists():
            metrics = read_json(path / "metrics.json")
            found.append((int(metrics.get("tokens_seen", 0)), path))
    return max(found, default=(0, None), key=lambda item: item[0])[1]


def local_progress() -> int:
    path = latest_local_checkpoint()
    return int(read_json(path / "metrics.json").get("tokens_seen", 0)) if path else 0


def remote_checkpoint_names() -> list[str]:
    result = command(colab("ls", "-s", SESSION, REMOTE_CHECKPOINTS), timeout=120, check=False)
    text = (result.stdout or "") + (result.stderr or "")
    return sorted(set(match.group(0) for match in CHECKPOINT_RE.finditer(text)), key=lambda name: int(CHECKPOINT_RE.fullmatch(name).group(1)))


def mirror_latest_checkpoint() -> None:
    names = remote_checkpoint_names()
    if not names:
        return
    name = names[-1]
    current = latest_local_checkpoint()
    if current and current.name == name:
        return
    # Keep one durable exact-resume checkpoint because the Mac volume is nearly full.
    for old in CHECKPOINTS.glob("checkpoint_*"):
        if old.is_dir():
            shutil.rmtree(old)
    stage = CHECKPOINTS / (name + ".partial")
    with contextlib.suppress(FileNotFoundError):
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        for filename in ("checkpoint_manifest.json", "metrics.json", "config.json", "model.safetensors", "training_state.pt"):
            download(f"{REMOTE_CHECKPOINTS}/{name}/{filename}", stage / filename)
        manifest = read_json(stage / "checkpoint_manifest.json")
        for filename, expected in manifest.get("files", {}).items():
            path = stage / filename
            if path.stat().st_size != int(expected["bytes"]) or file_sha(path) != expected["sha256"]:
                raise RuntimeError(f"checkpoint integrity mismatch: {filename}")
        os.replace(stage, CHECKPOINTS / name)
        metrics = read_json(CHECKPOINTS / name / "metrics.json")
        update_status(latest_checkpoint=name, latest_tokens=metrics.get("tokens_seen"), latest_step=metrics.get("step"))
        log(f"Mirrored durable checkpoint {name}")
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(stage)
        raise


def sync_small_files() -> None:
    pairs = [
        (f"{REMOTE_MODELS}/logs/train_metrics.jsonl", LOGS / "train_metrics.jsonl"),
        (f"{REMOTE_MODELS}/logs/prepare_progress.json", LOGS / "prepare_progress.json"),
        (f"{REMOTE_MODELS}/data_manifest.json", MODELS / "data_manifest.json"),
    ]
    for remote, local in pairs:
        temp = local.with_suffix(local.suffix + ".tmp")
        if download(remote, temp, check=False):
            os.replace(temp, local)


def ensure_session() -> None:
    if not session_exists():
        update_status(phase="allocating_gpu", state="running")
        log(f"Allocating named T4 session {SESSION}")
        command(colab("new", "-s", SESSION, "--gpu", "T4"), timeout=300)
    status = session_status()
    if "Hardware: T4" not in status:
        raise RuntimeError(f"Expected T4, got: {status}")
    update_status(phase="setting_up_runtime", colab_status=status, last_error=None)


def setup_remote() -> None:
    log("Installing remote dependencies through colab install")
    command(colab("install", "-s", SESSION, "numpy<3", "datasets", "transformers", "safetensors", "sentencepiece"), timeout=900)
    remote_exec(f"from pathlib import Path\n[Path(p).mkdir(parents=True,exist_ok=True) for p in {[REMOTE_ROOT, REMOTE_MODELS, REMOTE_DATA, REMOTE_CHECKPOINTS, REMOTE_RESUME]!r}]\n", 120)
    upload(BASE_TRAINER, REMOTE_BASE)
    upload(TRAINER, REMOTE_TRAINER)


def start_exec(code: str, log_name: str, timeout: int) -> subprocess.Popen[Any]:
    handle = (LOGS / log_name).open("a", encoding="utf-8")
    handle.write(f"\n===== {now()} =====\n"); handle.flush()
    process = subprocess.Popen(colab("exec", "-s", SESSION, "--timeout", str(timeout)), stdin=subprocess.PIPE, stdout=handle, stderr=subprocess.STDOUT, text=True)
    assert process.stdin is not None
    process.stdin.write(code); process.stdin.close()
    setattr(process, "_log_handle", handle)
    return process


def monitor(process: subprocess.Popen[Any], phase: str, mirror: bool = False) -> int:
    while process.poll() is None:
        if STOP_FILE.exists():
            process.terminate()
            with contextlib.suppress(Exception):
                process.wait(20)
            return -15
        if not session_exists():
            log(f"Colab reclaimed session during {phase}; terminating stale local exec")
            process.terminate()
            with contextlib.suppress(Exception):
                process.wait(20)
            return 97
        sync_small_files()
        if mirror:
            with contextlib.suppress(Exception):
                mirror_latest_checkpoint()
        update_status(phase=phase, colab_status=session_status(), latest_tokens=local_progress())
        time.sleep(POLL_SECONDS)
    handle = getattr(process, "_log_handle", None)
    if handle:
        handle.close()
    sync_small_files()
    return int(process.returncode or 0)


def prepare_data() -> None:
    # A reclaimed runtime's partial stream is not reusable; reset its local telemetry.
    with contextlib.suppress(FileNotFoundError):
        (LOGS / "prepare_progress.json").unlink()
    update_status(phase="preparing_data")
    code = f"""import runpy,sys
sys.path.insert(0,{REMOTE_ROOT!r})
sys.argv=[{REMOTE_TRAINER!r},'prepare','--output-dir',{REMOTE_MODELS!r},'--data-dir',{REMOTE_DATA!r},'--prepare-train-tokens','{PREP_TRAIN_TOKENS}','--prepare-val-tokens','{PREP_VAL_TOKENS}']
runpy.run_path({REMOTE_TRAINER!r},run_name='__main__')
"""
    process = start_exec(code, "prepare.log", 3000)
    if monitor(process, "preparing_data") != 0:
        raise RuntimeError("Remote dataset preparation failed")
    verify = f"""import hashlib
def h(path,n=None):
 d=hashlib.sha256(); left=n
 with open(path,'rb') as f:
  while True:
   want=8<<20 if left is None else min(8<<20,left)
   if want<=0: break
   b=f.read(want)
   if not b: break
   d.update(b)
   if left is not None: left-=len(b)
 return d.hexdigest()
train={REMOTE_DATA!r}+'/train_tokens.uint16'; val={REMOTE_DATA!r}+'/val_tokens.uint16'
assert h(val)=={ORIGINAL_VAL_SHA256!r}
assert h(train,44000000)=={ORIGINAL_22M_TRAIN_SHA256!r}
print('SAME_STREAM_PREFIX_VERIFIED')
"""
    result = remote_exec(verify, 600)
    if "SAME_STREAM_PREFIX_VERIFIED" not in result.stdout:
        raise RuntimeError("Dataset integrity verification did not confirm")
    update_status(data_integrity="verified_same_stream_22m_prefix")


def upload_resume(path: Path) -> str:
    remote_exec(f"from pathlib import Path\nPath({REMOTE_RESUME!r}).mkdir(parents=True,exist_ok=True)\n", 120)
    for name in ("model.safetensors", "training_state.pt", "config.json", "metrics.json"):
        upload(path / name, f"{REMOTE_RESUME}/{name}")
    return REMOTE_RESUME


def train() -> int:
    resume = latest_local_checkpoint()
    remote_resume = upload_resume(resume) if resume else ""
    args = [REMOTE_TRAINER, "train", "--output-dir", REMOTE_MODELS, "--data-dir", REMOTE_DATA,
            "--micro-batch", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM), "--seq-len", str(SEQ_LEN),
            "--target-tokens", str(TARGET_TOKENS), "--checkpoint-tokens", str(CHECKPOINT_TOKENS),
            "--max-lr", str(MAX_LR), "--min-lr", str(MIN_LR), "--warmup-fraction", "0.05",
            "--log-every", "5", "--eval-batches", "24", "--eval-batch-size", "4"]
    if remote_resume:
        args += ["--resume", remote_resume]
    code = f"import runpy,sys\nsys.path.insert(0,{REMOTE_ROOT!r})\nsys.argv={args!r}\nrunpy.run_path({REMOTE_TRAINER!r},run_name='__main__')\n"
    update_status(phase="training", state="running", resumed_from=resume.name if resume else None)
    return monitor(start_exec(code, "training.log", TRAIN_TIMEOUT), "training", mirror=True)


def build_final() -> None:
    checkpoint = latest_local_checkpoint()
    if not checkpoint:
        return
    final = MODELS / "final"; final.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "training_state.pt", "config.json", "metrics.json"):
        destination = final / name
        with contextlib.suppress(FileNotFoundError):
            destination.unlink()
        os.link(checkpoint / name, destination)
    atomic_json(final / "final_manifest.json", {"checkpoint": checkpoint.name, "tokens_seen": local_progress(), "parameters": PARAMETERS, "created_at": now()})


def run_cycle() -> None:
    ensure_session()
    setup_remote()
    prepare_data()
    result = train()
    with contextlib.suppress(Exception):
        mirror_latest_checkpoint()
    if result != 0 and not STOP_FILE.exists():
        raise RuntimeError(f"Training exec exited with {result}")


def supervisor() -> None:
    ensure_dirs(); LOCK_FILE.touch(exist_ok=True)
    lock = LOCK_FILE.open("r+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    with contextlib.suppress(FileNotFoundError): STOP_FILE.unlink()
    update_status(state="running", phase="starting", architecture={
        "vocab_size": 49152, "d_model": 160, "d_ff": 360, "unique_layers": 10,
        "effective_layers": 18, "recursion_strategy": "middle-cycle", "recursion_steps": 2,
        "shared_middle_layers": [1,2,3,4,5,6,7,8],
        "query_heads": 5, "kv_heads": 1, "attention_layers": [0,1,3,5,7,9],
        "sequence_length": SEQ_LEN, "micro_batch": MICRO_BATCH, "grad_accum": GRAD_ACCUM,
        "global_tokens_per_step": MICRO_BATCH * GRAD_ACCUM * SEQ_LEN,
    })
    signal.signal(signal.SIGTERM, lambda *_: STOP_FILE.touch())
    signal.signal(signal.SIGINT, lambda *_: STOP_FILE.touch())
    try:
        while not STOP_FILE.exists() and local_progress() < TARGET_TOKENS:
            try:
                run_cycle()
            except Exception as exc:
                log(f"Cycle error: {type(exc).__name__}: {exc}")
                update_status(phase="retry_wait", last_error=f"{type(exc).__name__}: {exc}")
                stop_session()
                if not STOP_FILE.exists(): time.sleep(RETRY_SECONDS)
        if local_progress() >= TARGET_TOKENS:
            build_final(); stop_session()
            update_status(state="completed", phase="completed", latest_tokens=local_progress(), colab_status=sessions())
            log("Training complete; final checkpoint durable and Colab released")
        else:
            stop_session(); update_status(state="stopped", phase="stopped")
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
    log_handle = (LOGS / "launcher.log").open("a", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "run"], cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    print(json.dumps({"started": True, "pid": process.pid, "status": str(STATUS)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("kick", "run", "status", "request-stop"))
    args = parser.parse_args()
    if args.command == "kick": kick()
    elif args.command == "run": supervisor()
    elif args.command == "status": print(json.dumps(read_json(STATUS), indent=2))
    else:
        ensure_dirs(); STOP_FILE.touch(); print(json.dumps({"stop_requested": True}))


if __name__ == "__main__":
    main()
