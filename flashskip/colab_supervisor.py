#!/usr/bin/env python3
"""Durable Mac-side supervisor for FlashSkip-100M continued training on Colab T4.

Only the official `colab` CLI is used for remote allocation, execution, installation,
upload, download, listing, status, and cleanup. The Mac is the durable checkpoint store;
Colab `/content` is treated as disposable.

Storage policy:
- every ~20M-token checkpoint is a full checkpoint containing model.safetensors,
  training_state.pt, config.json, metrics.json, and checkpoint_manifest.json;
- remote checkpoints are written through a .partial directory and atomically promoted;
- Mac mirroring verifies SHA-256/byte counts before atomic promotion and keeps the full
  optimizer, GradScaler, RNG, step, and data-cursor state inside every checkpoint;
- training_state.pt may use transparent lossless APFS compression, with identical bytes
  and the same normal file path, to fit the full checkpoint series safely.

Commands:
  python3 colab_supervisor.py status
  python3 colab_supervisor.py kick
  python3 colab_supervisor.py run
  python3 colab_supervisor.py request-stop
  python3 colab_supervisor.py mirror-once
"""
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
TRAINER = ROOT / "flashskip_100m.py"
MODELS = ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
RESUME_DIR = MODELS / "resume_latest"
FINAL_DIR = MODELS / "final"
LOGS = ROOT / "logs"
SUPERVISOR_LOG = LOGS / "supervisor.log"
TRAIN_LOG = LOGS / "colab_train_cli.log"
STATUS_FILE = ROOT / "supervisor_status.json"
PID_FILE = ROOT / "supervisor.pid"
LOCK_FILE = ROOT / "supervisor.lock"
STOP_FILE = ROOT / "supervisor.stop"

SESSION = os.environ.get("FLASHSKIP_COLAB_SESSION", "flashskip400m")
REMOTE_ROOT = "/content/flashskip"
REMOTE_MODELS = f"{REMOTE_ROOT}/models"
REMOTE_DATA = f"{REMOTE_ROOT}/data"
REMOTE_TRAINER = f"{REMOTE_ROOT}/flashskip_100m.py"
REMOTE_RESUME = f"{REMOTE_ROOT}/resume_checkpoint"
REMOTE_CHECKPOINTS = f"{REMOTE_MODELS}/checkpoints"

TARGET_TOKENS = 400_000_000
PREP_TRAIN_TOKENS = 410_000_000
PREP_VAL_TOKENS = 1_000_000
CHECKPOINT_TOKENS = 20_000_000
NO_REPLAY_CURSOR = 200_406_528
PREVIOUS_210M_TRAIN_TOKENS = 210_000_000
PREVIOUS_210M_TRAIN_SHA256 = "bfcfb643f5a9b021896091286f40303322fd09b8c00b58d2686dcaae893d11d7"
SEQ_LEN = 512
MICRO_BATCH = 16
GRAD_ACCUM = 4
MAX_LR = 3.5e-4
MIN_LR = 3.5e-5
WARMUP_FRACTION = 0.05
POLL_SECONDS = int(os.environ.get("FLASHSKIP_POLL_SECONDS", "75"))
RETRY_SECONDS = int(os.environ.get("FLASHSKIP_RETRY_SECONDS", "120"))
TRAIN_TIMEOUT_SECONDS = int(os.environ.get("FLASHSKIP_TRAIN_TIMEOUT", "21600"))

# The original 1M-val / 22M-train preparation hashes. Recreating the larger stream
# must preserve these exact prefixes before we trust the old data_cursor.
ORIGINAL_VAL_SHA256 = "e04d5c522333ca12aa64e9ac89a3c69319d07cc6bc3f88429a59b9b163e9dabe"
ORIGINAL_22M_TRAIN_SHA256 = "0997c490eb5f0fa279de527baec93606fe9d88999f59512386bb16d39c41dff0"
ORIGINAL_22M_TRAIN_BYTES = 22_000_000 * 2

CHECKPOINT_RE = re.compile(r"^checkpoint_step_(\d+)_tokens_([0-9.]+)M$")
REQUIRED_MODEL_FILES = ("model.safetensors", "config.json", "metrics.json")
FULL_CHECKPOINT_FILES = REQUIRED_MODEL_FILES + ("training_state.pt", "checkpoint_manifest.json")
REMOTE_COMPLETE_FILES = FULL_CHECKPOINT_FILES
RUNTIME_TOKEN_RE = re.compile(r"(colab-runtime-proxy-token=)[^&\\s]+")
LARGE_UPLOAD_THRESHOLD = 32 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024
MIN_FREE_FOR_FULL_MIRROR = 650 * 1024 * 1024


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for p in (MODELS, CHECKPOINTS, RESUME_DIR, FINAL_DIR, LOGS):
        p.mkdir(parents=True, exist_ok=True)


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_log(message: str) -> None:
    ensure_dirs()
    line = f"[{now_iso()}] {message}"
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(**fields: Any) -> None:
    base: dict[str, Any] = {}
    if STATUS_FILE.exists():
        with contextlib.suppress(Exception):
            base = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    base.update(fields)
    base["updated_at"] = now_iso()
    base["session"] = SESSION
    base["target_tokens"] = TARGET_TOKENS
    base["checkpoint_interval_tokens"] = CHECKPOINT_TOKENS
    base["supervisor_pid"] = os.getpid()
    atomic_json(STATUS_FILE, base)


def sanitize_text(text: str) -> str:
    return RUNTIME_TOKEN_RE.sub(r"\1[REDACTED]", text)


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    remaining = limit_bytes
    with path.open("rb") as f:
        while True:
            want = 8 << 20 if remaining is None else min(8 << 20, remaining)
            if want <= 0:
                break
            block = f.read(want)
            if not block:
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
    if limit_bytes is not None and remaining != 0:
        raise RuntimeError(f"{path} ended before {limit_bytes} bytes")
    return h.hexdigest()


def checkpoint_info(path: Path) -> dict[str, Any] | None:
    m = CHECKPOINT_RE.match(path.name)
    if not m or not path.is_dir():
        return None
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokens = int(metrics.get("tokens_seen") or 0)
    step = int(metrics.get("step") or int(m.group(1)))
    if tokens <= 0:
        return None
    return {"path": path, "name": path.name, "tokens_seen": tokens, "step": step, "metrics": metrics}


def local_checkpoints() -> list[dict[str, Any]]:
    ensure_dirs()
    out = []
    for p in CHECKPOINTS.iterdir():
        info = checkpoint_info(p)
        if info and all((p / f).is_file() and (p / f).stat().st_size > 0 for f in FULL_CHECKPOINT_FILES):
            out.append(info)
    return sorted(out, key=lambda x: (x["tokens_seen"], x["step"]))


def resume_meta() -> dict[str, Any] | None:
    p = RESUME_DIR / "resume_latest.json"
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    state = RESUME_DIR / "training_state.pt"
    ckpt = CHECKPOINTS / str(meta.get("checkpoint_name", ""))
    if not state.is_file() or state.stat().st_size < 100_000_000:
        return None
    if not ckpt.is_dir() or not all((ckpt / f).is_file() for f in REQUIRED_MODEL_FILES):
        return None
    expected = meta.get("training_state_sha256")
    if expected and sha256_file(state) != expected:
        return None
    return meta


def latest_exact_resume() -> dict[str, Any]:
    """Return newest exact-resume model + optimizer state.

    The rolling state is preferred after the first new mirror. Otherwise the permanent
    10.03M anchor's own training_state.pt is used.
    """
    meta = resume_meta()
    if meta:
        ckpt = CHECKPOINTS / meta["checkpoint_name"]
        info = checkpoint_info(ckpt)
        if info and int(meta.get("tokens_seen", -1)) == info["tokens_seen"]:
            return {**info, "training_state": RESUME_DIR / "training_state.pt", "resume_source": "rolling"}

    candidates = []
    for info in local_checkpoints():
        state = info["path"] / "training_state.pt"
        if state.is_file() and state.stat().st_size > 100_000_000:
            candidates.append({**info, "training_state": state, "resume_source": "checkpoint"})
    if not candidates:
        raise RuntimeError("No exact-resume local checkpoint found")
    return candidates[-1]


def validate_checkpoint_dir(path: Path, require_state: bool = False) -> dict[str, Any]:
    info = checkpoint_info(path)
    if not info:
        raise RuntimeError(f"Invalid checkpoint metadata: {path}")
    if (path / "model.safetensors").stat().st_size < 100_000_000:
        raise RuntimeError(f"Model weights unexpectedly small: {path}")
    if require_state:
        state = path / "training_state.pt"
        if not state.is_file() or state.stat().st_size < 100_000_000:
            raise RuntimeError(f"Training state missing/small: {path}")
    return info


def colab_bin() -> str:
    found = shutil.which("colab")
    if found:
        return found
    fallback = Path.home() / ".local/bin/colab"
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("Official colab CLI was not found in PATH or ~/.local/bin/colab")


def colab_args(*args: str) -> list[str]:
    return [colab_bin(), "--auth=oauth2", *args]


def run_cmd(args: list[str], *, timeout: int = 300, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    p = subprocess.run(args, input=input_text, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode != 0:
        out = sanitize_text(p.stdout or "")
        err = sanitize_text(p.stderr or "")
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(args)}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return p


def sessions_text() -> str:
    p = run_cmd(colab_args("sessions"), timeout=90, check=False)
    return (p.stdout or "") + (p.stderr or "")


def session_exists() -> bool:
    return f"[{SESSION}]" in sessions_text()


def session_status() -> str:
    p = run_cmd(colab_args("status", "-s", SESSION), timeout=90, check=False)
    text = (p.stdout or "") + (p.stderr or "")
    return text.strip()


def session_is_busy() -> bool:
    return "Status: BUSY" in session_status()


def stop_session() -> None:
    if session_exists():
        append_log(f"Stopping Colab session {SESSION}")
        run_cmd(colab_args("stop", "-s", SESSION), timeout=180, check=False)


def ensure_session() -> None:
    if not session_exists():
        append_log(f"Allocating Colab T4 session {SESSION}")
        run_cmd(colab_args("new", "-s", SESSION, "--gpu", "T4"), timeout=300)
    status = session_status()
    if "Hardware: T4" not in status:
        raise RuntimeError(f"Session is not a T4:\n{status}")
    update_status(colab_status=status, phase="session_ready", last_error=None)


def remote_exec(code: str, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return run_cmd(colab_args("exec", "-s", SESSION, "--timeout", str(timeout)), timeout=timeout + 120, input_text=code)


def remote_ls(path: str, check: bool = False) -> str:
    p = run_cmd(colab_args("ls", "-s", SESSION, path), timeout=120, check=check)
    return ((p.stdout or "") + (p.stderr or "")).strip()


def remote_upload(local: Path, remote: str) -> None:
    """Upload a file through the official Colab CLI, chunking large artifacts.

    The CLI content API rejects very large single uploads on some runtimes. For large
    checkpoint files we stream 16 MiB temporary chunks through `colab upload`, then use
    `colab exec` to concatenate and SHA-256 verify the target on the remote VM.
    """
    local = Path(local)
    size = local.stat().st_size
    if size <= LARGE_UPLOAD_THRESHOLD:
        run_cmd(colab_args("upload", "-s", SESSION, str(local), remote), timeout=600)
        return

    digest = sha256_file(local)
    tag = f"{local.name}-{digest[:12]}"
    remote_parts = f"{REMOTE_ROOT}/.upload_parts/{tag}"
    parent = str(Path(remote).parent)
    remote_exec(
        "from pathlib import Path\n"
        f"Path({remote_parts!r}).mkdir(parents=True, exist_ok=True)\n"
        f"Path({parent!r}).mkdir(parents=True, exist_ok=True)\n",
        timeout=120,
    )
    temp_chunk = ROOT / ".colab_upload_chunk.tmp"
    count = 0
    try:
        with local.open("rb") as src:
            while True:
                block = src.read(UPLOAD_CHUNK_BYTES)
                if not block:
                    break
                temp_chunk.write_bytes(block)
                part_name = f"part_{count:05d}.bin"
                run_cmd(
                    colab_args("upload", "-s", SESSION, str(temp_chunk), f"{remote_parts}/{part_name}"),
                    timeout=600,
                )
                count += 1
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_chunk.unlink()

    assemble = f"""from pathlib import Path
import hashlib, os, shutil
parts=Path({remote_parts!r})
target=Path({remote!r})
tmp=target.with_name(target.name+'.assembling')
h=hashlib.sha256(); total=0
with tmp.open('wb') as out:
    for p in sorted(parts.glob('part_*.bin')):
        b=p.read_bytes(); out.write(b); h.update(b); total+=len(b)
assert total=={size}, (total,{size})
assert h.hexdigest()=={digest!r}, (h.hexdigest(),{digest!r})
os.replace(tmp,target)
shutil.rmtree(parts)
print('LARGE_UPLOAD_OK',target,total,h.hexdigest())
"""
    p = remote_exec(assemble, timeout=600)
    if "LARGE_UPLOAD_OK" not in ((p.stdout or "") + (p.stderr or "")):
        raise RuntimeError(f"Remote large-file assembly did not confirm success for {local}")


def remote_download(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(colab_args("download", "-s", SESSION, remote, str(local)), timeout=1200)


def setup_remote() -> None:
    append_log("Warming Colab kernel and installing Python dependencies")
    # `colab install` is deliberately first. On a just-created T4 it reliably warms the
    # Jupyter kernel before our first `colab exec`; issuing exec against a cold kernel can
    # hit the CLI's internal 10-second kernel-start read timeout.
    run_cmd(
        colab_args("install", "-s", SESSION, "numpy<3", "datasets", "transformers", "safetensors", "sentencepiece"),
        timeout=900,
    )
    append_log("Creating remote FlashSkip directories after kernel warm-up")
    code = (
        "from pathlib import Path\n"
        f"[Path(p).mkdir(parents=True, exist_ok=True) for p in {repr([REMOTE_ROOT, REMOTE_MODELS, REMOTE_DATA, REMOTE_RESUME, REMOTE_CHECKPOINTS])}]\n"
        "print('REMOTE_DIRS_OK')\n"
    )
    remote_exec(code, timeout=120)
    remote_upload(TRAINER, REMOTE_TRAINER)


def upload_resume_checkpoint(info: dict[str, Any]) -> str:
    append_log(f"Uploading exact resume checkpoint {info['name']} ({info['tokens_seen']:,} tokens)")
    remote_exec(
        "from pathlib import Path\n"
        f"Path({REMOTE_RESUME!r}).mkdir(parents=True, exist_ok=True)\n",
        timeout=120,
    )
    for fname in REQUIRED_MODEL_FILES:
        remote_upload(info["path"] / fname, f"{REMOTE_RESUME}/{fname}")
    remote_upload(Path(info["training_state"]), f"{REMOTE_RESUME}/training_state.pt")
    remote_exec(
        "from pathlib import Path\n"
        f"p=Path({REMOTE_RESUME!r})\n"
        "req=['model.safetensors','config.json','metrics.json','training_state.pt']\n"
        "assert all((p/x).is_file() and (p/x).stat().st_size>0 for x in req)\n"
        "print('REMOTE_RESUME_OK', {x:(p/x).stat().st_size for x in req})\n",
        timeout=120,
    )
    return REMOTE_RESUME


def prepare_expanded_data() -> None:
    append_log(f"Preparing deterministic FineWeb-Edu stream: {PREP_TRAIN_TOKENS:,} train + {PREP_VAL_TOKENS:,} val tokens")
    update_status(phase="preparing_data")
    code = f"""import runpy,sys
sys.argv=[{REMOTE_TRAINER!r},'prepare','--output-dir',{REMOTE_MODELS!r},'--data-dir',{REMOTE_DATA!r},'--prepare-train-tokens','{PREP_TRAIN_TOKENS}','--prepare-val-tokens','{PREP_VAL_TOKENS}']
runpy.run_path({REMOTE_TRAINER!r},run_name='__main__')
"""
    p = remote_exec(code, timeout=2400)
    with (LOGS / "data_prepare.log").open("a", encoding="utf-8") as f:
        f.write(p.stdout or "")
        f.write(p.stderr or "")

    # Verify that the expanded deterministic stream preserves the original 22M-token
    # training prefix and the entire original 1M-token validation split.
    verify = f"""import hashlib
from pathlib import Path
def h(path,n=None):
    x=hashlib.sha256(); left=n
    with open(path,'rb') as f:
        while True:
            want=8<<20 if left is None else min(8<<20,left)
            if want<=0: break
            b=f.read(want)
            if not b: break
            x.update(b)
            if left is not None: left-=len(b)
    if n is not None: assert left==0, (path,left)
    return x.hexdigest()
train={REMOTE_DATA!r}+'/train_tokens.uint16'; val={REMOTE_DATA!r}+'/val_tokens.uint16'
train_prefix=h(train,{ORIGINAL_22M_TRAIN_BYTES})
train_210m=h(train,{PREVIOUS_210M_TRAIN_TOKENS * 2})
val_hash=h(val)
print('TRAIN_PREFIX_SHA256',train_prefix)
print('TRAIN_210M_SHA256',train_210m)
print('VAL_SHA256',val_hash)
assert train_prefix=={ORIGINAL_22M_TRAIN_SHA256!r}, (train_prefix,'train-prefix-mismatch')
assert train_210m=={PREVIOUS_210M_TRAIN_SHA256!r}, (train_210m,'210m-prefix-mismatch')
assert val_hash=={ORIGINAL_VAL_SHA256!r}, (val_hash,'val-mismatch')
assert {NO_REPLAY_CURSOR} < {PREVIOUS_210M_TRAIN_TOKENS}, ('saved cursor must lie inside verified 210M prefix')
print('DATA_PREFIX_INTEGRITY_OK')
print('DATA_CONTINUATION_BOUNDARY_OK', {NO_REPLAY_CURSOR})
"""
    vp = remote_exec(verify, timeout=600)
    text = (vp.stdout or "") + (vp.stderr or "")
    if "DATA_PREFIX_INTEGRITY_OK" not in text or "DATA_CONTINUATION_BOUNDARY_OK" not in text:
        raise RuntimeError("Expanded dataset did not pass deterministic no-replay prefix verification")
    append_log("Expanded FineWeb-Edu data passed exact 210M-prefix SHA-256 verification; saved cursor is inside verified prefix")
    update_status(phase="data_ready", data_integrity="verified_210m_prefix", no_replay_cursor=NO_REPLAY_CURSOR)


def remote_checkpoint_names() -> list[str]:
    if not session_exists():
        return []
    text = remote_ls(REMOTE_CHECKPOINTS, check=False)
    names = []
    for raw in text.splitlines():
        name = raw.strip().rstrip("/")
        if CHECKPOINT_RE.match(name):
            names.append(name)
    return sorted(names, key=lambda n: int(CHECKPOINT_RE.match(n).group(1)))


def remote_checkpoint_complete(name: str) -> bool:
    text = remote_ls(f"{REMOTE_CHECKPOINTS}/{name}", check=False)
    if "Error:" in text:
        return False
    listed = {line.strip().rstrip("/") for line in text.splitlines()}
    return all(x in listed for x in REMOTE_COMPLETE_FILES)


def validate_checkpoint_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Checkpoint manifest missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or {}
    for fname in ("model.safetensors", "training_state.pt", "config.json", "metrics.json"):
        expected = files.get(fname) or {}
        p = path / fname
        if not p.is_file() or p.stat().st_size != int(expected.get("bytes") or -1):
            raise RuntimeError(f"Checkpoint file size mismatch: {p}")
        if sha256_file(p) != expected.get("sha256"):
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {p}")
    return manifest


def apfs_compress_file(path: Path) -> None:
    """Losslessly apply transparent APFS compression while keeping the normal file path."""
    ditto = shutil.which("ditto")
    if sys.platform != "darwin" or not ditto or not path.is_file():
        return
    before_sha = sha256_file(path)
    tmp = path.with_name(path.name + ".compressed-new")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    run_cmd([ditto, "--hfsCompression", str(path), str(tmp)], timeout=300)
    if sha256_file(tmp) != before_sha:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"APFS compression changed checkpoint bytes: {path}")
    os.replace(tmp, path)


def install_rolling_resume_state(checkpoint: dict[str, Any]) -> None:
    """Point the rolling resume slot at the newest full checkpoint without duplicating storage."""
    src = checkpoint["path"] / "training_state.pt"
    state_sha = sha256_file(src)
    stable = RESUME_DIR / "training_state.pt"
    with contextlib.suppress(FileNotFoundError):
        stable.unlink()
    try:
        os.link(src, stable)
    except OSError:
        shutil.copy2(src, stable)
    meta = {
        "checkpoint_name": checkpoint["name"],
        "tokens_seen": checkpoint["tokens_seen"],
        "step": checkpoint["step"],
        "training_state_sha256": state_sha,
        "training_state_bytes": stable.stat().st_size,
        "installed_at": now_iso(),
    }
    atomic_json(RESUME_DIR / "resume_latest.json", meta)


def mirror_checkpoint(name: str) -> dict[str, Any] | None:
    final = CHECKPOINTS / name
    if final.is_dir():
        info = validate_checkpoint_dir(final, require_state=True)
        validate_checkpoint_manifest(final)
        rolling = resume_meta()
        if not rolling or rolling.get("checkpoint_name") != name:
            install_rolling_resume_state(info)
        return None
    if not remote_checkpoint_complete(name):
        return None
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < MIN_FREE_FOR_FULL_MIRROR:
        append_log(
            f"Deferring mirror for {name}: only {free_bytes:,} free bytes; "
            f"need at least {MIN_FREE_FOR_FULL_MIRROR:,} for safe full-checkpoint transfer"
        )
        update_status(phase="checkpoint_waiting_for_local_space", pending_checkpoint=name, local_free_bytes=free_bytes)
        return None

    append_log(f"Mirroring complete full remote checkpoint {name} to Mac")
    partial = CHECKPOINTS / (name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)

    # Manifest first lets interrupted downloads reuse already verified files.
    remote_download(f"{REMOTE_CHECKPOINTS}/{name}/checkpoint_manifest.json", partial / "checkpoint_manifest.json")
    manifest = json.loads((partial / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    expected_files = manifest.get("files") or {}
    download_order = ("training_state.pt", "model.safetensors", "config.json", "metrics.json")
    for fname in download_order:
        expected = expected_files.get(fname) or {}
        dst = partial / fname
        if dst.is_file() and dst.stat().st_size == int(expected.get("bytes") or -1) and sha256_file(dst) == expected.get("sha256"):
            continue
        tmp = partial / (fname + ".new")
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        remote_download(f"{REMOTE_CHECKPOINTS}/{name}/{fname}", tmp)
        if tmp.stat().st_size != int(expected.get("bytes") or -1) or sha256_file(tmp) != expected.get("sha256"):
            raise RuntimeError(f"Downloaded checkpoint file failed manifest verification: {fname}")
        os.replace(tmp, dst)
        if fname == "training_state.pt":
            apfs_compress_file(dst)

    info = validate_checkpoint_dir(partial, require_state=True)
    checked_manifest = validate_checkpoint_manifest(partial)
    if int(checked_manifest.get("tokens_seen") or -1) != info["tokens_seen"]:
        raise RuntimeError(f"Checkpoint manifest/metrics token mismatch: {name}")
    os.replace(partial, final)
    info = validate_checkpoint_dir(final, require_state=True)
    validate_checkpoint_manifest(final)
    install_rolling_resume_state(info)

    marker = {
        "checkpoint_name": name,
        "tokens_seen": info["tokens_seen"],
        "step": info["step"],
        "data_cursor": checked_manifest.get("data_cursor"),
        "files": checked_manifest.get("files"),
        "mirrored_at": now_iso(),
        "resume_state_policy": "full state retained inside every checkpoint; resume_latest is a hard link",
    }
    atomic_json(final / ".mirrored.json", marker)
    append_log(f"Full checkpoint durable on Mac: {name} ({info['tokens_seen']:,} tokens), all hashes verified")
    update_status(latest_local_checkpoint=name, latest_tokens=info["tokens_seen"], latest_step=info["step"], phase="training")
    return info


def mirror_all_available() -> list[dict[str, Any]]:
    mirrored = []
    for name in remote_checkpoint_names():
        try:
            item = mirror_checkpoint(name)
            if item:
                mirrored.append(item)
        except Exception as e:
            append_log(f"Mirror warning for {name}: {type(e).__name__}: {e}")
    return mirrored


def sync_small_remote_artifacts() -> None:
    for remote, local in [
        (f"{REMOTE_MODELS}/logs/train_metrics.jsonl", LOGS / "train_metrics.jsonl"),
        (f"{REMOTE_MODELS}/training_summary.json", MODELS / "training_summary.json"),
        (f"{REMOTE_MODELS}/data_manifest.json", MODELS / "data_manifest.json"),
    ]:
        with contextlib.suppress(Exception):
            tmp = local.with_suffix(local.suffix + ".tmp")
            remote_download(remote, tmp)
            os.replace(tmp, local)


def latest_progress_from_local() -> int:
    cps = local_checkpoints()
    return cps[-1]["tokens_seen"] if cps else 0


def training_code(remote_resume: str) -> str:
    args = [
        REMOTE_TRAINER, "train",
        "--output-dir", REMOTE_MODELS,
        "--data-dir", REMOTE_DATA,
        "--micro-batch", str(MICRO_BATCH),
        "--grad-accum", str(GRAD_ACCUM),
        "--seq-len", str(SEQ_LEN),
        "--target-tokens", str(TARGET_TOKENS),
        "--checkpoint-tokens", str(CHECKPOINT_TOKENS),
        "--max-lr", str(MAX_LR),
        "--min-lr", str(MIN_LR),
        "--warmup-fraction", str(WARMUP_FRACTION),
        "--log-every", "10",
        "--eval-batches", "24",
        "--eval-batch-size", "2",
        "--resume", remote_resume,
        "--require-resume-cursor-at-least", str(NO_REPLAY_CURSOR),
    ]
    return "import runpy,sys\n" + f"sys.argv={args!r}\n" + f"runpy.run_path({REMOTE_TRAINER!r},run_name='__main__')\n"


def start_training_process(remote_resume: str) -> subprocess.Popen[Any]:
    ensure_dirs()
    log = TRAIN_LOG.open("a", encoding="utf-8")
    log.write(f"\n\n===== TRAIN EXEC START {now_iso()} =====\n")
    log.flush()
    cmd = colab_args("exec", "-s", SESSION, "--timeout", str(TRAIN_TIMEOUT_SECONDS))
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    assert p.stdin is not None
    p.stdin.write(training_code(remote_resume))
    p.stdin.close()
    setattr(p, "_flashskip_log_handle", log)
    return p


def close_training_process(p: subprocess.Popen[Any]) -> None:
    handle = getattr(p, "_flashskip_log_handle", None)
    if handle:
        with contextlib.suppress(Exception):
            handle.flush()
            handle.close()


def wait_busy_session_to_finish() -> None:
    append_log("Existing Colab session is BUSY; monitoring it instead of starting duplicate training")
    while session_exists() and session_is_busy() and not STOP_FILE.exists():
        mirror_all_available()
        sync_small_remote_artifacts()
        update_status(phase="monitoring_existing_busy_session", colab_status=session_status())
        time.sleep(POLL_SECONDS)


def run_training_once() -> str:
    ensure_session()
    if session_is_busy():
        wait_busy_session_to_finish()
        return "existing_busy_finished"

    setup_remote()
    resume = latest_exact_resume()
    resume_manifest = validate_checkpoint_manifest(resume["path"])
    resume_cursor = int(resume_manifest.get("data_cursor") or -1)
    if resume_cursor < NO_REPLAY_CURSOR:
        raise RuntimeError(
            f"Refusing continuation from data_cursor={resume_cursor:,}; no-replay boundary is {NO_REPLAY_CURSOR:,}"
        )
    if resume["tokens_seen"] >= TARGET_TOKENS:
        return "target_already_reached"
    global_labels = MICRO_BATCH * SEQ_LEN * GRAD_ACCUM
    source_per_step = MICRO_BATCH * (SEQ_LEN + 1) * GRAD_ACCUM
    remaining_steps = (TARGET_TOKENS - resume["tokens_seen"] + global_labels - 1) // global_labels
    required_final_cursor = resume_cursor + remaining_steps * source_per_step
    if required_final_cursor > PREP_TRAIN_TOKENS:
        raise RuntimeError(
            f"Prepared data is too short for no-wrap continuation: need cursor {required_final_cursor:,}, "
            f"prepared {PREP_TRAIN_TOKENS:,}"
        )
    remote_resume = upload_resume_checkpoint(resume)
    prepare_expanded_data()

    append_log(
        f"Launching T4 continuation from {resume['tokens_seen']:,} to {TARGET_TOKENS:,} total tokens; "
        f"checkpoint cadence={CHECKPOINT_TOKENS:,}; LR={MAX_LR:g}->{MIN_LR:g}"
    )
    update_status(
        phase="training",
        resume_checkpoint=resume["name"],
        resume_tokens=resume["tokens_seen"],
        max_lr=MAX_LR,
        min_lr=MIN_LR,
        data_train_tokens=PREP_TRAIN_TOKENS,
    )
    p = start_training_process(remote_resume)
    last_sync = 0.0
    try:
        while p.poll() is None:
            if STOP_FILE.exists():
                append_log("Stop requested; terminating local colab exec and releasing session")
                p.terminate()
                with contextlib.suppress(Exception):
                    p.wait(timeout=30)
                stop_session()
                return "stopped"
            mirror_all_available()
            if time.time() - last_sync > 300:
                sync_small_remote_artifacts()
                last_sync = time.time()
            update_status(colab_status=session_status(), local_free_bytes=shutil.disk_usage(ROOT).free)
            time.sleep(POLL_SECONDS)
        rc = p.returncode
    finally:
        close_training_process(p)

    # One final pull after the training command exits.
    mirror_all_available()
    sync_small_remote_artifacts()
    append_log(f"Colab training exec exited rc={rc}; latest local={latest_progress_from_local():,} tokens")
    if latest_progress_from_local() >= TARGET_TOKENS:
        return "target_reached"
    if rc == 0:
        return "ended_before_target"
    return f"training_failed_rc_{rc}"


def build_final_links() -> None:
    resume = latest_exact_resume()
    if resume["tokens_seen"] < TARGET_TOKENS:
        return
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    # Final references use hard links where possible so no extra model/state storage is consumed.
    sources = {
        "model.safetensors": resume["path"] / "model.safetensors",
        "config.json": resume["path"] / "config.json",
        "metrics.json": resume["path"] / "metrics.json",
        "training_state.pt": Path(resume["training_state"]),
    }
    for name, src in sources.items():
        dst = FINAL_DIR / name
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    atomic_json(
        FINAL_DIR / "final_manifest.json",
        {
            "checkpoint_name": resume["name"],
            "tokens_seen": resume["tokens_seen"],
            "step": resume["step"],
            "created_at": now_iso(),
            "hardlink_preferred": True,
        },
    )


def final_evaluate() -> None:
    """Run comprehensive evaluation on the final model while the T4 still exists."""
    if latest_progress_from_local() < TARGET_TOKENS or not session_exists():
        return
    append_log("Running final T4 evaluation suite at ~200M tokens")
    code = f"""import runpy,sys
sys.argv=[{REMOTE_TRAINER!r},'evaluate','--output-dir',{REMOTE_MODELS!r},'--data-dir',{REMOTE_DATA!r},'--eval-examples','200','--wikitext-tokens','250000','--generation-tokens','64']
runpy.run_path({REMOTE_TRAINER!r},run_name='__main__')
"""
    try:
        p = remote_exec(code, timeout=2400)
        (LOGS / "final_evaluation_cli.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
        eval_dir = MODELS / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        for name in ("results.json", "report.md"):
            with contextlib.suppress(Exception):
                remote_download(f"{REMOTE_MODELS}/evaluation/{name}", eval_dir / name)
        append_log("Final evaluation finished and available results were mirrored")
    except Exception as e:
        append_log(f"Final evaluation warning: {type(e).__name__}: {e}")


def supervisor_loop() -> None:
    ensure_dirs()
    LOCK_FILE.touch(exist_ok=True)
    lock_f = LOCK_FILE.open("r+")
    try:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another FlashSkip supervisor already holds the lock")

    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    with contextlib.suppress(FileNotFoundError):
        STOP_FILE.unlink()
    update_status(phase="starting", state="running")
    append_log(f"Supervisor started pid={os.getpid()} target={TARGET_TOKENS:,} tokens")

    def on_signal(signum: int, _frame: Any) -> None:
        append_log(f"Received signal {signum}; requesting graceful stop")
        STOP_FILE.touch()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    try:
        while not STOP_FILE.exists():
            progress = latest_progress_from_local()
            if progress >= TARGET_TOKENS:
                build_final_links()
                final_evaluate()
                stop_session()
                update_status(phase="completed", state="completed", latest_tokens=progress, colab_sessions=sessions_text())
                append_log(f"Target reached: {progress:,} tokens. Colab released.")
                return
            try:
                outcome = run_training_once()
                append_log(f"Training cycle outcome: {outcome}")
                if outcome == "stopped":
                    update_status(phase="stopped", state="stopped")
                    return
                if latest_progress_from_local() >= TARGET_TOKENS:
                    continue
            except Exception as e:
                append_log(f"Supervisor cycle error: {type(e).__name__}: {e}")
                update_status(phase="retry_wait", state="running", last_error=f"{type(e).__name__}: {e}")
                # A broken/reclaimed session is disposable. Avoid assuming its filesystem survived.
                with contextlib.suppress(Exception):
                    stop_session()
            if STOP_FILE.exists():
                break
            append_log(f"Retrying from newest durable Mac checkpoint after {RETRY_SECONDS}s")
            time.sleep(RETRY_SECONDS)
    finally:
        with contextlib.suppress(FileNotFoundError):
            PID_FILE.unlink()
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        lock_f.close()
        if STOP_FILE.exists():
            with contextlib.suppress(Exception):
                stop_session()


def pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def current_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return None
    return pid if pid_running(pid) else None


def kick() -> int:
    ensure_dirs()
    pid = current_pid()
    if pid:
        print(json.dumps({"already_running": True, "pid": pid, "status_file": str(STATUS_FILE)}, indent=2))
        return pid
    with contextlib.suppress(FileNotFoundError):
        STOP_FILE.unlink()
    log = SUPERVISOR_LOG.open("a", encoding="utf-8")
    p = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "run"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    log.close()
    # Give run-mode a short opportunity to acquire the lock and write its PID.
    for _ in range(20):
        time.sleep(0.25)
        pid = current_pid()
        if pid:
            print(json.dumps({"started": True, "pid": pid, "status_file": str(STATUS_FILE), "log": str(SUPERVISOR_LOG)}, indent=2))
            return pid
        if p.poll() is not None:
            raise RuntimeError(f"Supervisor exited immediately with rc={p.returncode}; see {SUPERVISOR_LOG}")
    raise RuntimeError("Supervisor process started but did not publish a PID")


def request_stop() -> None:
    ensure_dirs()
    STOP_FILE.touch()
    print(json.dumps({"stop_requested": True, "pid": current_pid(), "stop_file": str(STOP_FILE)}, indent=2))


def print_status() -> None:
    ensure_dirs()
    status: dict[str, Any] = {}
    if STATUS_FILE.exists():
        with contextlib.suppress(Exception):
            status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    cps = local_checkpoints()
    status.update(
        {
            "pid": current_pid(),
            "running": current_pid() is not None,
            "local_checkpoints": [
                {"name": x["name"], "tokens_seen": x["tokens_seen"], "step": x["step"], "val": x["metrics"].get("val")}
                for x in cps
            ],
            "rolling_resume": resume_meta(),
            "local_free_bytes": shutil.disk_usage(ROOT).free,
            "session_listing": sessions_text().strip(),
            "paths": {
                "models": str(MODELS),
                "supervisor_log": str(SUPERVISOR_LOG),
                "train_log": str(TRAIN_LOG),
                "status": str(STATUS_FILE),
            },
        }
    )
    print(json.dumps(status, indent=2, sort_keys=True, default=str))


def mirror_once() -> None:
    if not session_exists():
        print(json.dumps({"mirrored": [], "reason": "session_not_active"}, indent=2))
        return
    items = mirror_all_available()
    sync_small_remote_artifacts()
    print(json.dumps({"mirrored": [x["name"] for x in items], "latest_tokens": latest_progress_from_local()}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "kick", "status", "request-stop", "mirror-once"])
    args = ap.parse_args()
    ensure_dirs()
    if args.command == "run":
        supervisor_loop()
    elif args.command == "kick":
        kick()
    elif args.command == "status":
        print_status()
    elif args.command == "request-stop":
        request_stop()
    elif args.command == "mirror-once":
        mirror_once()


if __name__ == "__main__":
    main()
