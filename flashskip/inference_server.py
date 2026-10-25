#!/usr/bin/env python3
"""Small local streaming inference server for the latest FlashSkip checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoTokenizer

from flashskip_100m import FlashSkipConfig, FlashSkipLM
from flashskip_10m_recursive import RecursiveFlashSkipConfig, RecursiveFlashSkipLM
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mor10m.model import Config as MoRConfig, MoR


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
UI_DIR = ROOT / "ui"
MODEL_LOCK = threading.Lock()
MODEL_FAMILIES = {
    "mor10m": {
        "label": "MoR 10M · latest checkpoint",
        "checkpoints_dir": ROOT.parent / "mor10m" / "run" / "downloads",
        "nested_checkpoints": True,
        "tokenizer_dir": ROOT.parent / "mor10m" / "run" / "source_dataset",
        "config_cls": MoRConfig,
        "model_cls": MoR,
        "device": "cpu",
    },
    "recursive10m": {
        "label": "FlashSkip 10M Recursive (1B continuation)",
        "checkpoints_dir": ROOT / "flashskip10m_recursive_kaggle1b" / "models" / "checkpoints",
        "tokenizer_dir": ROOT / "flashskip10m_recursive_kaggle1b" / "models" / "tokenizer",
        "config_cls": RecursiveFlashSkipConfig,
        "model_cls": RecursiveFlashSkipLM,
    },
    "10m": {
        "label": "FlashSkip 10M",
        "checkpoints_dir": ROOT / "flashskip10m_kaggle300m" / "models" / "checkpoints",
        "config_cls": FlashSkipConfig,
        "model_cls": FlashSkipLM,
    },
    "100m": {
        "label": "FlashSkip 100M",
        "checkpoints_dir": MODELS_DIR / "checkpoints",
        "config_cls": FlashSkipConfig,
        "model_cls": FlashSkipLM,
    },
}
DEFAULT_MODEL_KEY = "mor10m"


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latest_checkpoint(model_key: str = DEFAULT_MODEL_KEY) -> Path:
    family = MODEL_FAMILIES.get(model_key)
    if family is None:
        raise ValueError(f"Unknown model key: {model_key}")
    checkpoints_dir = family["checkpoints_dir"]
    candidates = []
    pattern = "*/result/checkpoints/checkpoint_*" if family.get("nested_checkpoints") else "checkpoint_*"
    for path in checkpoints_dir.glob(pattern):
        manifest = path / "checkpoint_manifest.json"
        if not (path / "model.safetensors").exists() or not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            candidates.append((int(data.get("tokens_seen", 0)), path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(f"No valid {family['label']} checkpoint found under {checkpoints_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def tokenizer_path(model_key: str = DEFAULT_MODEL_KEY) -> Path:
    family = MODEL_FAMILIES.get(model_key) or {}
    candidates = [family.get("tokenizer_dir"), MODELS_DIR / "tokenizer", MODELS_DIR / "mac_test" / "tokenizer"]
    for path in candidates:
        if path is None:
            continue
        if (path / "tokenizer.json").exists():
            return path
    raise FileNotFoundError("No local tokenizer found")


class Runtime:
    def __init__(self, model_key: str = DEFAULT_MODEL_KEY, checkpoint: Path | None = None) -> None:
        family = MODEL_FAMILIES.get(model_key)
        if family is None:
            raise ValueError(f"Unknown model key: {model_key}")
        checkpoint = checkpoint or latest_checkpoint(model_key)
        config_cls = family["config_cls"]
        model_cls = family["model_cls"]
        config = config_cls.from_dict(
            json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        )
        self.device = torch.device(family["device"]) if family.get("device") else select_device()
        if self.device.type == "cpu":
            torch.set_num_threads(4)
        self.model_key = model_key
        self.checkpoint = checkpoint
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path(model_key)), use_fast=True, local_files_only=True
        )

        model = model_cls(config)
        if model_key == "mor10m":
            manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
            for name in ("config.json", "model.safetensors"):
                with (checkpoint / name).open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != manifest["files"][name]["sha256"]:
                    raise ValueError(f"Checkpoint checksum mismatch: {name}")
        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=True)
        # CPU attention is more portable in float32; accelerators benefit from fp16.
        if self.device.type != "cpu":
            model = model.to(dtype=torch.float16)
        self.model = model.to(self.device).eval()

        manifest_path = checkpoint / "checkpoint_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    @torch.inference_mode()
    def stream(self, request: "GenerationRequest") -> Iterator[bytes]:
        seed = request.seed if request.seed is not None else random.randrange(2**31)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        encoded = self.tokenizer.encode(
            request.prompt, add_special_tokens=False, return_tensors="pt"
        )
        if encoded.numel() == 0:
            raise ValueError("Prompt must contain at least one token")
        if encoded.shape[1] + request.max_new_tokens > self.config.max_seq_len:
            keep = self.config.max_seq_len - request.max_new_tokens
            encoded = encoded[:, -keep:]
        input_ids = encoded.to(self.device)
        prompt_tokens = input_ids.shape[1]
        generated: list[int] = []
        started = time.perf_counter()

        yield event("start", seed=seed, prompt_tokens=prompt_tokens)
        with MODEL_LOCK:
            for index in range(request.max_new_tokens):
                logits = self.model(input_ids)[:, -1, :].float()
                if request.temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits /= request.temperature
                    if request.top_k > 0:
                        values, _ = torch.topk(logits, min(request.top_k, logits.shape[-1]))
                        logits.masked_fill_(logits < values[:, [-1]], float("-inf"))
                    next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)

                token_id = int(next_token.item())
                generated.append(token_id)
                input_ids = torch.cat((input_ids, next_token), dim=1)
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                elapsed = max(time.perf_counter() - started, 1e-6)
                yield event(
                    "token",
                    token_id=token_id,
                    index=index + 1,
                    text=text,
                    tokens_per_second=(index + 1) / elapsed,
                )
                if token_id == self.tokenizer.eos_token_id:
                    break

        elapsed = time.perf_counter() - started
        yield event(
            "done",
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            generated_tokens=len(generated),
            elapsed_seconds=elapsed,
            tokens_per_second=len(generated) / max(elapsed, 1e-6),
        )


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    max_new_tokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=50, ge=0, le=49152)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class ModelSwitchRequest(BaseModel):
    model: str


def event(kind: str, **payload: object) -> bytes:
    return (json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n").encode("utf-8")


runtime: Runtime | None = None


def runtime_status(current: Runtime) -> dict[str, object]:
    return {
        "ready": True,
        "model_key": current.model_key,
        "model": current.config.model_name,
        "parameters": sum(p.numel() for p in current.model.parameters()),
        "checkpoint": current.checkpoint.name,
        "tokens_seen": current.manifest.get("tokens_seen"),
        "device": current.device.type.upper(),
        "available_models": [
            {"key": key, "label": family["label"]}
            for key, family in MODEL_FAMILIES.items()
        ],
    }


@asynccontextmanager
async def lifespan(application: FastAPI):
    global runtime
    runtime = Runtime(model_key=application.state.model_key, checkpoint=application.state.checkpoint)
    try:
        yield
    finally:
        runtime = None


app = FastAPI(title="FlashSkip Studio", lifespan=lifespan)
app.state.checkpoint = None
app.state.model_key = DEFAULT_MODEL_KEY


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(UI_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
def javascript() -> FileResponse:
    return FileResponse(UI_DIR / "app.js", media_type="text/javascript")


@app.get("/api/status")
def status() -> dict[str, object]:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    return runtime_status(runtime)


@app.post("/api/model")
def switch_model(request: ModelSwitchRequest) -> dict[str, object]:
    global runtime
    if request.model not in MODEL_FAMILIES:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
    if runtime is not None and runtime.model_key == request.model:
        return runtime_status(runtime)

    try:
        with MODEL_LOCK:
            if runtime is not None and runtime.model_key == request.model:
                return runtime_status(runtime)
            next_runtime = Runtime(model_key=request.model)
            previous_runtime = runtime
            runtime = next_runtime
            app.state.model_key = request.model
            app.state.checkpoint = None
            del previous_runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return runtime_status(next_runtime)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load {request.model}: {exc}") from exc


@app.get("/events")
def events() -> StreamingResponse:
    """Server-Sent Events stream — live model status + heartbeat."""
    def sse_stream() -> Iterator[bytes]:
        while runtime is None:
            time.sleep(0.5)
        while True:
            yield event(
                "status",
                model_key=runtime.model_key,
                model=runtime.config.model_name,
                parameters=sum(p.numel() for p in runtime.model.parameters()),
                checkpoint=runtime.checkpoint.name,
                tokens_seen=runtime.manifest.get("tokens_seen"),
                device=runtime.device.type.upper(),
            )
            time.sleep(2)

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@app.post("/api/generate")
def generate(request: GenerationRequest) -> StreamingResponse:
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    current_runtime = runtime

    def safe_stream() -> Iterator[bytes]:
        try:
            yield from current_runtime.stream(request)
        except Exception as exc:  # Keep stream failures machine-readable for the UI.
            yield event("error", message=f"{type(exc).__name__}: {exc}")

    return StreamingResponse(safe_stream(), media_type="application/x-ndjson")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", choices=MODEL_FAMILIES, default=DEFAULT_MODEL_KEY)
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint dir (overrides latest)")
    args = parser.parse_args()

    app.state.model_key = args.model
    app.state.checkpoint = Path(args.checkpoint) if args.checkpoint else None
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
