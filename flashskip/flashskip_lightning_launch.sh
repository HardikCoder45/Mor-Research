#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f runner.pid ]] && kill -0 "$(cat runner.pid)" 2>/dev/null; then
  echo "Runner already active: $(cat runner.pid)"
  exit 0
fi
python -m pip install -q 'numpy<3' datasets transformers safetensors sentencepiece
nohup python -u flashskip_lightning_remote.py > remote_training.log 2>&1 < /dev/null &
echo $! > runner.pid
echo "Lightning training runner started: $(cat runner.pid)"
