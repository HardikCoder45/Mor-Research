#!/bin/zsh
set -euo pipefail
ROOT="/Users/hardikarora/Desktop/MAIN/research/flashskip"
PIDFILE="$ROOT/live_dashboard.pid"
LOGFILE="$ROOT/logs/live_dashboard.log"
mkdir -p "$ROOT/logs"
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    open "http://127.0.0.1:8766"
    exit 0
  fi
fi
nohup /usr/bin/python3 "$ROOT/live_dashboard.py" >>"$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
for _ in {1..20}; do
  if curl -fsS --max-time 0.1 http://127.0.0.1:8766/api/status >/dev/null 2>&1; then
    break
  fi
  sleep 0.05
done
open "http://127.0.0.1:8766"
echo "FlashSkip dashboard running: pid=$PID http://127.0.0.1:8766"
