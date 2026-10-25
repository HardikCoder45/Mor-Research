#!/bin/zsh
set -euo pipefail
ROOT="/Users/hardikarora/Desktop/MAIN/research/flashskip"
RUN="$ROOT/flashskip10m_1b"
mkdir -p "$RUN/logs"
export FLASHSKIP_DASHBOARD_ROOT="$RUN"
export FLASHSKIP_START_TOKENS=200015872
export FLASHSKIP_TARGET_TOKENS=1200015872
export FLASHSKIP_CHECKPOINT_TOKENS=100000000
export FLASHSKIP_DASHBOARD_PORT=8768
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 -B "$ROOT/flashskip_10m_dashboard.py"
