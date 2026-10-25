#!/bin/zsh
set -euo pipefail
ROOT="/Users/hardikarora/Desktop/MAIN/research/flashskip"
RUN="$ROOT/flashskip10m_recursive_kaggle1b"
mkdir -p "$RUN/logs"
export FLASHSKIP_DASHBOARD_ROOT="$RUN"
export FLASHSKIP_START_TOKENS=100007936
export FLASHSKIP_TARGET_TOKENS=1100007936
export FLASHSKIP_CHECKPOINT_TOKENS=100000000
export FLASHSKIP_DASHBOARD_PORT=8770
export PYTHONDONTWRITEBYTECODE=1
exec /usr/bin/python3 -B "$ROOT/flashskip_recursive_dashboard.py"
