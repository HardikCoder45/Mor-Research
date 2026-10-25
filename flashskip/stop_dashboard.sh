#!/bin/zsh
set -euo pipefail
ROOT="/Users/hardikarora/Desktop/MAIN/research/flashskip"
PIDFILE="$ROOT/live_dashboard.pid"
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
  fi
  rm -f "$PIDFILE"
fi
