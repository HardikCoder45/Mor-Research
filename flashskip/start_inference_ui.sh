#!/bin/sh
set -eu
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

if [ -n "${FLASHSKIP_PYTHON:-}" ]; then
    PYTHON="$FLASHSKIP_PYTHON"
elif [ -x ".venv-inference/bin/python" ]; then
    PYTHON=".venv-inference/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi

if [ -z "$PYTHON" ] || ! "$PYTHON" -c 'import fastapi, numpy, safetensors, torch, transformers, uvicorn' >/dev/null 2>&1; then
    cat >&2 <<'EOF'
FlashSkip inference dependencies are not installed.

Set up the dedicated local environment once:
  uv venv .venv-inference --python 3.12
  uv pip install --python .venv-inference/bin/python -r requirements-inference.txt

Then run:
  ./start_inference_ui.sh
EOF
    exit 1
fi

exec "$PYTHON" -B inference_server.py "$@"
