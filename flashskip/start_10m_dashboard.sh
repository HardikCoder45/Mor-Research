#!/bin/zsh
set -euo pipefail
ROOT="/Users/hardikarora/Desktop/MAIN/research/flashskip"
RUN="$ROOT/flashskip10m"
PIDFILE="$RUN/dashboard.pid"
mkdir -p "$RUN/logs"
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "Dashboard already running: http://127.0.0.1:8767"
    exit 0
  fi
fi
PYTHONDONTWRITEBYTECODE=1 nohup /usr/bin/python3 -B "$ROOT/flashskip_10m_dashboard.py" >>"$RUN/logs/dashboard.log" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
for _ in {1..30}; do
  curl -fsS --max-time .2 http://127.0.0.1:8767/api/status >/dev/null 2>&1 && break
  sleep .1
done
echo "Dashboard running: pid=$PID http://127.0.0.1:8767"
