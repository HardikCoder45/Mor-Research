#!/usr/bin/env python3
"""Submit and supervise the FlashSkip-10M Recursive one-billion-new-token Kaggle run."""
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
RUN_ROOT = ROOT / "flashskip10m_recursive_kaggle1b"
MODELS = RUN_ROOT / "models"
CHECKPOINTS = MODELS / "checkpoints"
LOGS = RUN_ROOT / "logs"
STATUS = RUN_ROOT / "status.json"
PID_FILE = RUN_ROOT / "supervisor.pid"
STOP_FILE = RUN_ROOT / "supervisor.stop"
LOCK_FILE = RUN_ROOT / "supervisor.lock"
DATASET_STAGE = ROOT / "kaggle_recursive_1b_dataset"
KERNEL_STAGE = ROOT / "kaggle_recursive_1b_kernel"
SOURCE_CHECKPOINT = ROOT / "flashskip10m_recursive100m" / "models" / "checkpoints" / "checkpoint_step_003052_tokens_0100.01M"
KERNEL_REF = "hardikarora83i2492/flashskip-10m-recursive-1b-new-tokens-p100"
DATASET_REF = "hardikarora83i2492/flashskip-10m-recursive-1b-checkpoint"
START_TOKENS = 100_007_936
START_STEP = 3_052
NO_REPLAY_CURSOR = 100_203_264
NEW_TOKEN_TARGET = 1_000_000_000
NOMINAL_FINAL_TOKENS = START_TOKENS + NEW_TOKEN_TARGET
GLOBAL_TOKENS = 32_768
NEW_STEPS = (NEW_TOKEN_TARGET + GLOBAL_TOKENS - 1) // GLOBAL_TOKENS
STEP_ALIGNED_FINAL_TOKENS = START_TOKENS + NEW_STEPS * GLOBAL_TOKENS
PREPARE_TRAIN_TOKENS = NO_REPLAY_CURSOR + NEW_STEPS * 32_832 + 5_000_000
POLL_SECONDS = 20
TRAIN_RE = re.compile(r"TRAIN\s+(\{[^\n]+\})")
EVAL_RE = re.compile(r"(INITIAL_VAL|CHECKPOINT_EVAL)\s+(\{[^\n]+\})")
STATUS_RE = re.compile(r"KAGGLE_STATUS\s+(\{[^\n]+\})")
PREP_RE = re.compile(r"PREP docs=(\d+) val=([\d,]+) train=([\d,]+) tok/s=([\d,]+)")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs() -> None:
    for path in (RUN_ROOT, MODELS, CHECKPOINTS, LOGS, DATASET_STAGE, KERNEL_STAGE): path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def update_status(**fields: Any) -> None:
    data = read_json(STATUS)
    data.update({
        "model": "FlashSkip-10M-Recursive-MiddleCycle-Nr2", "run_id": "flashskip10m-recursive1b-p100", "parameters": 9_964_832, "provider": "Kaggle",
        "machine": "NVIDIA Tesla P100 (required)", "kernel": KERNEL_REF, "dataset": DATASET_REF,
        "start_tokens": START_TOKENS, "target_tokens": NOMINAL_FINAL_TOKENS,
        "step_aligned_target_tokens": STEP_ALIGNED_FINAL_TOKENS, "additional_target_tokens": NEW_TOKEN_TARGET,
        "no_replay_cursor": NO_REPLAY_CURSOR, "prepare_target_tokens": PREPARE_TRAIN_TOKENS,
        "checkpoint_interval_tokens": 100_000_000, "updated_at": now(),
        "architecture": {"vocab_size":49152,"d_model":160,"d_ff":360,"layers":10,"unique_layers":10,"effective_layers":18,
                         "recursion_strategy":"middle-cycle","recursion_steps":2,"shared_start_layer":1,"shared_end_layer":9,
                         "query_heads":5,"kv_heads":1,"attention_layers":[0,1,3,5,7,9],"attention_executions":10,"sequence_length":512,
                         "micro_batch":32,"grad_accum":2,"global_tokens_per_step":32768},
        **fields,
    })
    atomic_json(STATUS, data)


def log(message: str) -> None:
    ensure_dirs()
    with (LOGS / "supervisor.log").open("a", encoding="utf-8") as h: h.write(f"[{now()}] {message}\n")


def command(args: list[str], timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        while block := h.read(8 << 20): d.update(block)
    return d.hexdigest()


def validate_checkpoint(path: Path, require_start: bool = False) -> dict[str, Any]:
    manifest = read_json(path / "checkpoint_manifest.json")
    if not manifest: raise RuntimeError(f"missing checkpoint manifest: {path}")
    for name, expected in manifest["files"].items():
        item = path / name
        if item.stat().st_size != int(expected["bytes"]) or sha256(item) != expected["sha256"]:
            raise RuntimeError(f"checkpoint integrity failure: {path.name}/{name}")
    if require_start:
        actual = (int(manifest["tokens_seen"]), int(manifest["step"]), int(manifest["data_cursor"]))
        expected = (START_TOKENS, START_STEP, NO_REPLAY_CURSOR)
        if actual != expected: raise RuntimeError(f"refusing wrong resume checkpoint {actual}; expected {expected}")
    return manifest


def link_or_copy(source: Path, destination: Path) -> None:
    with contextlib.suppress(FileNotFoundError): destination.unlink()
    try: os.link(source, destination)
    except OSError: shutil.copy2(source, destination)


def stage_submission() -> None:
    ensure_dirs(); manifest = validate_checkpoint(SOURCE_CHECKPOINT, require_start=True)
    for path in DATASET_STAGE.iterdir():
        if path.is_file(): path.unlink()
    for name in manifest["files"]: link_or_copy(SOURCE_CHECKPOINT / name, DATASET_STAGE / name)
    link_or_copy(SOURCE_CHECKPOINT / "checkpoint_manifest.json", DATASET_STAGE / "checkpoint_manifest.json")
    link_or_copy(ROOT / "flashskip_100m.py", DATASET_STAGE / "flashskip_100m.py")
    link_or_copy(ROOT / "flashskip_10m_recursive.py", DATASET_STAGE / "flashskip_10m_recursive.py")
    telemetry = RUN_ROOT / "telemetry.json"
    if telemetry.exists(): link_or_copy(telemetry, DATASET_STAGE / "telemetry.json")
    tokenizer = ROOT / "models" / "mac_test" / "tokenizer"
    link_or_copy(tokenizer / "tokenizer.json", DATASET_STAGE / "tokenizer.json")
    link_or_copy(tokenizer / "tokenizer_config.json", DATASET_STAGE / "tokenizer_config.json")
    atomic_json(DATASET_STAGE / "dataset-metadata.json", {"title":"FlashSkip 10M Recursive 1B Resume Checkpoint","id":DATASET_REF,"licenses":[{"name":"other"}]})
    for path in KERNEL_STAGE.iterdir():
        if path.is_file(): path.unlink()
    link_or_copy(ROOT / "kaggle_train_recursive_1b_run.py", KERNEL_STAGE / "kaggle_train_recursive_1b_run.py")
    atomic_json(KERNEL_STAGE / "kernel-metadata.json", {
        "id": KERNEL_REF, "title":"FlashSkip 10M Recursive - 1B New Tokens - P100", "code_file":"kaggle_train_recursive_1b_run.py",
        "language":"python", "kernel_type":"script", "is_private":True, "enable_gpu":True, "enable_internet":True,
        "machine_shape":"NvidiaTeslaP100", "dataset_sources":[DATASET_REF], "competition_sources":[], "kernel_sources":[], "model_sources":[]})


def publish_dataset() -> None:
    status = command(["kaggle","datasets","status",DATASET_REF,"--format","json"], timeout=120)
    if status.returncode == 0:
        command(["kaggle","datasets","version","-p",str(DATASET_STAGE),"-r","zip","-q","-m","Verified recursive 100M resume for 1B-new-token run"], timeout=1800, check=True)
    else:
        command(["kaggle","datasets","create","-p",str(DATASET_STAGE),"-r","zip","-q"], timeout=1800, check=True)
    for _ in range(120):
        s = command(["kaggle","datasets","status",DATASET_REF,"--format","json"], timeout=120)
        if s.returncode == 0 and "ready" in s.stdout.lower(): return
        time.sleep(5)
    raise RuntimeError("Kaggle resume dataset did not become ready")


def parse_logs(text: str) -> None:
    rows: list[dict[str, Any]] = []
    for m in TRAIN_RE.finditer(text):
        with contextlib.suppress(Exception): rows.append(json.loads(m.group(1)))
    for m in EVAL_RE.finditer(text):
        with contextlib.suppress(Exception):
            row=json.loads(m.group(2)); row["event"]="initial_eval" if m.group(1)=="INITIAL_VAL" else "checkpoint_eval"; rows.append(row)
    rows.sort(key=lambda r:(int(r.get("step",0)),0 if r.get("event")=="train" else 1))
    if rows:
        tmp=LOGS/"train_metrics.jsonl.tmp"; tmp.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows),encoding="utf-8"); os.replace(tmp,LOGS/"train_metrics.jsonl")
        trains=[r for r in rows if r.get("event")=="train"]
        if trains:
            last=trains[-1]; update_status(latest_tokens=last.get("tokens_seen"),latest_step=last.get("step"),last_live_at=time.time(),phase="training")
    statuses=[]
    for m in STATUS_RE.finditer(text):
        with contextlib.suppress(Exception): statuses.append(json.loads(m.group(1)))
    if statuses:
        remote=statuses[-1]; phase=remote.get("phase")
        machine=remote.get("machine") or ("NVIDIA Tesla P100" if "P100_CUDA_OK" in text else None)
        update_status(remote_phase=phase,phase=phase or read_json(STATUS).get("phase"),data_integrity=remote.get("data_integrity"),machine=machine or read_json(STATUS).get("machine"),cuda_verified="P100_CUDA_OK" in text,last_live_at=time.time(), selected_micro_batch=remote.get("micro_batch"), selected_grad_accum=remote.get("grad_accum"))
        if remote.get("error"): update_status(last_error=remote["error"])
    preps=list(PREP_RE.finditer(text))
    if preps:
        m=preps[-1]; atomic_json(LOGS/"prepare_progress.json", {"documents":int(m.group(1)),"val_tokens":int(m.group(2).replace(",","")),"train_tokens":int(m.group(3).replace(",","")),"target_val_tokens":1_000_000,"target_train_tokens":PREPARE_TRAIN_TOKENS,"tokens_per_second":int(m.group(4).replace(",",""))})


def kernel_status() -> str:
    r=command(["kaggle","kernels","status",KERNEL_REF],timeout=120); return ((r.stdout or "")+(r.stderr or "")).strip()


def sync_logs() -> str:
    r=command(["kaggle","kernels","logs",KERNEL_REF],timeout=180)
    text=(r.stdout or "")+(r.stderr or "")
    with contextlib.suppress(Exception):
        records=json.loads(r.stdout)
        if isinstance(records,list): text="".join(str(x.get("data","")) for x in records)
    (LOGS/"kaggle_kernel.log").write_text(text,encoding="utf-8"); parse_logs(text); return text


def submit() -> None:
    for stale in (LOGS / "prepare_progress.json", LOGS / "train_metrics.jsonl", LOGS / "kaggle_kernel.log"):
        with contextlib.suppress(FileNotFoundError): stale.unlink()
    update_status(state="running",phase="submitting",kernel_status="submitting",last_error=None,cuda_verified=False,data_integrity="checkpoint_verified",latest_tokens=START_TOKENS,latest_step=START_STEP,run_started_at=now())
    command(["kaggle","kernels","push","-p",str(KERNEL_STAGE),"--accelerator","NvidiaTeslaP100","--timeout","43200"],timeout=1800,check=True)
    log("Submitted recursive 1B-new-token continuation requesting exactly one Tesla P100")


def monitor() -> None:
    while not STOP_FILE.exists():
        s=kernel_status(); text=sync_logs(); lowered=s.lower(); update_status(kernel_status=s)
        if "complete" in lowered: return
        if any(w in lowered for w in ("error","failed","cancel")): raise RuntimeError(f"Kaggle kernel failed: {s}\n{text[-4000:]}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError("stop requested")


def download_outputs() -> None:
    output=RUN_ROOT/"kernel_output"
    if output.exists(): shutil.rmtree(output)
    output.mkdir(parents=True)
    command(["kaggle","kernels","output",KERNEL_REF,"-p",str(output),"-o","-q"],timeout=3600,check=True)
    found=[]
    for manifest_path in output.rglob("checkpoint_manifest.json"):
        cp=manifest_path.parent
        with contextlib.suppress(Exception): found.append((int(validate_checkpoint(cp)["tokens_seen"]),cp))
    found.sort()
    for tokens, source in found:
        if tokens <= START_TOKENS: continue
        dest=CHECKPOINTS/source.name
        if dest.exists(): shutil.rmtree(dest)
        shutil.copytree(source,dest); validate_checkpoint(dest)
        log(f"Downloaded and SHA-verified {dest.name}")
    if found:
        tokens, cp=found[-1]; m=validate_checkpoint(cp); update_status(latest_checkpoint=cp.name,latest_tokens=tokens,latest_step=m["step"])
    run_manifest=next(iter(output.rglob("run_manifest.json")),None)
    if run_manifest: shutil.copy2(run_manifest,RUN_ROOT/"run_manifest.json")


def run() -> None:
    ensure_dirs(); LOCK_FILE.touch(exist_ok=True); lock=LOCK_FILE.open("r+"); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    PID_FILE.write_text(str(os.getpid()),encoding="utf-8")
    with contextlib.suppress(FileNotFoundError): STOP_FILE.unlink()
    signal.signal(signal.SIGTERM,lambda *_:STOP_FILE.touch()); signal.signal(signal.SIGINT,lambda *_:STOP_FILE.touch())
    try:
        validate_checkpoint(SOURCE_CHECKPOINT,require_start=True); stage_submission(); update_status(state="running",phase="publishing_resume_dataset",latest_tokens=START_TOKENS,latest_step=START_STEP,last_error=None)
        publish_dataset(); update_status(phase="staged_p100_submission")
        submit(); monitor(); download_outputs(); final=read_json(STATUS).get("latest_tokens",0)
        if int(final) < NOMINAL_FINAL_TOKENS: raise RuntimeError(f"kernel completed below target: {final}")
        update_status(state="completed",phase="completed"); log("1B-new-token continuation completed; all returned checkpoints SHA-verified on Mac")
    except Exception as exc:
        update_status(state="failed",phase="failed",last_error=f"{type(exc).__name__}: {exc}"); log(f"Supervisor failed: {type(exc).__name__}: {exc}"); raise
    finally:
        with contextlib.suppress(FileNotFoundError): PID_FILE.unlink()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()


def attach() -> None:
    """Attach supervision to an already-submitted Kaggle run without resubmitting it."""
    ensure_dirs(); LOCK_FILE.touch(exist_ok=True); lock=LOCK_FILE.open("r+"); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    PID_FILE.write_text(str(os.getpid()),encoding="utf-8")
    with contextlib.suppress(FileNotFoundError): STOP_FILE.unlink()
    signal.signal(signal.SIGTERM,lambda *_:STOP_FILE.touch()); signal.signal(signal.SIGINT,lambda *_:STOP_FILE.touch())
    try:
        validate_checkpoint(SOURCE_CHECKPOINT,require_start=True)
        update_status(state="running",phase="attaching_existing_run",last_error=None,kernel=KERNEL_REF)
        monitor(); download_outputs(); final=read_json(STATUS).get("latest_tokens",0)
        if int(final) < NOMINAL_FINAL_TOKENS: raise RuntimeError(f"kernel completed below target: {final}")
        update_status(state="completed",phase="completed"); log("Attached 1B-new-token run completed; returned checkpoints SHA-verified on Mac")
    except Exception as exc:
        update_status(state="failed",phase="failed",last_error=f"{type(exc).__name__}: {exc}"); log(f"Attached supervisor failed: {type(exc).__name__}: {exc}"); raise
    finally:
        with contextlib.suppress(FileNotFoundError): PID_FILE.unlink()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN); lock.close()


def kick(mode: str = "run") -> None:
    ensure_dirs(); pid=int(PID_FILE.read_text()) if PID_FILE.exists() and PID_FILE.read_text().strip().isdigit() else 0
    if pid:
        try: os.kill(pid,0); print(json.dumps({"already_running":True,"pid":pid})); return
        except OSError: pass
    handle=(LOGS/"launcher.log").open("a",encoding="utf-8")
    p=subprocess.Popen([sys.executable,"-B",str(Path(__file__).resolve()),mode],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    print(json.dumps({"started":True,"mode":mode,"pid":p.pid}))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=("kick","kick-attach","run","attach","status","request-stop")); a=p.parse_args()
    if a.command=="kick": kick("run")
    elif a.command=="kick-attach": kick("attach")
    elif a.command=="run": run()
    elif a.command=="attach": attach()
    elif a.command=="status": print(json.dumps(read_json(STATUS),indent=2))
    else: ensure_dirs(); STOP_FILE.touch(); print(json.dumps({"stop_requested":True}))

if __name__=="__main__": main()
