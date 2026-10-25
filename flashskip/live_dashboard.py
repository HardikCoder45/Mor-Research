#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "supervisor_status.json"
TRAIN_LOG = ROOT / "logs" / "colab_train_cli.log"
TARGET = 400_000_000
TRAIN_RE = re.compile(rb'^TRAIN\s+(\{.*\})$')

_status_cache = ({}, -1)
_train_cache = ({}, -1, -1)


def read_tail(path: Path, max_bytes: int = 131072) -> bytes:
    if not path.exists():
        return b""
    with path.open("rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(0, size - max_bytes))
        data = f.read()
    if size > max_bytes:
        cut = data.find(b"\n")
        if cut >= 0:
            data = data[cut + 1:]
    return data


def latest_train_event():
    global _train_cache
    try:
        st = TRAIN_LOG.stat()
    except FileNotFoundError:
        return {}
    sig = (st.st_mtime_ns, st.st_size)
    if sig == _train_cache[1:]:
        return _train_cache[0]
    data = read_tail(TRAIN_LOG)
    event = {}
    for line in reversed(data.splitlines()):
        m = TRAIN_RE.match(line.strip())
        if m:
            try:
                event = json.loads(m.group(1))
            except Exception:
                event = {}
            break
    _train_cache = (event, *sig)
    return event


def status_event():
    global _status_cache
    try:
        st = STATUS.stat()
    except FileNotFoundError:
        return {}
    if st.st_mtime_ns == _status_cache[1]:
        return _status_cache[0]
    try:
        data = json.loads(STATUS.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    _status_cache = (data, st.st_mtime_ns)
    return data


def snapshot():
    status = status_event()
    train = latest_train_event()
    tokens = int(train.get("tokens_seen") or status.get("latest_tokens") or status.get("resume_tokens") or 0)
    step = int(train.get("step") or status.get("latest_step") or 0)
    pct = max(0.0, min(100.0, tokens / TARGET * 100.0))
    return {
        "tokens": tokens,
        "target": TARGET,
        "percent": pct,
        "step": step,
        "loss": train.get("loss"),
        "loss20": train.get("loss_20"),
        "tokensPerSecond": train.get("tokens_per_second"),
        "lr": train.get("lr"),
        "gradNorm": train.get("grad_norm"),
        "peakCudaGiB": train.get("peak_cuda_GiB"),
        "phase": status.get("phase", "unknown"),
        "running": bool(status.get("running", False)) or "Status: BUSY" in str(status.get("colab_status", "")),
        "colabStatus": status.get("colab_status", ""),
        "updatedAt": status.get("updated_at"),
        "nextCheckpoint": ((tokens // 20_000_000) + 1) * 20_000_000 if tokens < TARGET else TARGET,
    }


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FlashSkip Live</title><style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#07111f;color:#edf6ff}*{box-sizing:border-box}body{margin:0;background:#07111f;padding:14px}.w{max-width:680px;margin:auto}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}.name{font-weight:750;font-size:14px}.live{font-size:11px;color:#74e0a7;display:flex;gap:6px;align-items:center}.dot{width:7px;height:7px;border-radius:50%;background:#5ee69a}.card{background:#0a1829;border:1px solid #17304a;border-radius:15px;padding:16px}.hero{display:flex;justify-content:space-between;align-items:end;gap:12px}.label,.k{font-size:10px;color:#7891aa;text-transform:uppercase;letter-spacing:.07em}.big{font-size:38px;font-weight:800;line-height:1;margin-top:4px;letter-spacing:-.035em}.pct{font-size:19px;font-weight:700;color:#a6dcff}.bar{height:14px;background:#13283d;border-radius:999px;overflow:hidden;margin:12px 0 14px}.fill{height:100%;background:linear-gradient(90deg,#30c2ff,#63e6a1);width:0;transition:width .18s linear}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.s{background:#0c1d30;border:1px solid #17304a;border-radius:10px;padding:10px;min-width:0}.v{font-size:16px;font-weight:700;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bottom{display:flex;justify-content:space-between;gap:10px;margin-top:10px;color:#7189a1;font-size:10px}.status{overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:75%}@media(max-width:500px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.big{font-size:30px}}@media(prefers-reduced-motion:reduce){.fill{transition:none}}
</style></head><body><div class="w"><div class="top"><div class="name">FlashSkip-100M · Live</div><div class="live"><span class="dot"></span><span id="live">live</span></div></div><div class="card"><div class="hero"><div><div class="label">Cumulative tokens</div><div class="big" id="tokens">__TOKENS__</div></div><div class="pct" id="pct">__PCT__%</div></div><div class="bar"><div class="fill" id="fill" style="width:__PCT__%"></div></div><div class="grid"><div class="s"><div class="k">Step</div><div class="v" id="step">__STEP__</div></div><div class="s"><div class="k">Loss</div><div class="v" id="loss">__LOSS__</div></div><div class="s"><div class="k">20-step loss</div><div class="v" id="loss20">__LOSS20__</div></div><div class="s"><div class="k">Throughput</div><div class="v" id="tps">__TPS__</div></div><div class="s"><div class="k">GPU</div><div class="v" id="gpu">__GPU__</div></div><div class="s"><div class="k">Next checkpoint</div><div class="v" id="next">__NEXT__</div></div></div><div class="bottom"><div class="status" id="status">__STATUS__</div><div id="age">instant stream</div></div></div></div><script>
const $=x=>document.getElementById(x),nf=new Intl.NumberFormat('en-US',{maximumFractionDigits:0});const c=n=>n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(Math.round(n||0));function draw(d){$('tokens').textContent=nf.format(d.tokens||0);$('pct').textContent=d.percent.toFixed(3)+'%';$('fill').style.width=d.percent+'%';$('step').textContent=nf.format(d.step||0);$('loss').textContent=Number.isFinite(d.loss)?d.loss.toFixed(4):'—';$('loss20').textContent=Number.isFinite(d.loss20)?d.loss20.toFixed(4):'—';$('tps').textContent=Number.isFinite(d.tokensPerSecond)?c(d.tokensPerSecond)+'/s':'—';$('gpu').textContent=Number.isFinite(d.peakCudaGiB)?d.peakCudaGiB.toFixed(2)+' GiB':'—';$('next').textContent=c(d.nextCheckpoint);$('status').textContent=(d.colabStatus||d.phase||'').replace(/\s+/g,' ');$('live').textContent=d.running?'training':'idle';$('age').textContent='updated '+new Date().toLocaleTimeString();}let es;function connect(){es=new EventSource('/events');es.onmessage=e=>{try{draw(JSON.parse(e.data))}catch{}};es.onerror=()=>{$('live').textContent='reconnecting';es.close();setTimeout(connect,250)}}connect();
</script></body></html>'''


def html_now() -> bytes:
    s = snapshot()
    def num(v, digits=3):
        return "—" if v is None else f"{v:.{digits}f}"
    page = HTML.replace("__TOKENS__", f"{s['tokens']:,}")
    page = page.replace("__PCT__", f"{s['percent']:.3f}")
    page = page.replace("__STEP__", f"{s['step']:,}")
    page = page.replace("__LOSS__", num(s['loss'], 4)).replace("__LOSS20__", num(s['loss20'], 4))
    page = page.replace("__TPS__", "—" if s['tokensPerSecond'] is None else f"{s['tokensPerSecond']/1000:.2f}K/s")
    page = page.replace("__GPU__", "—" if s['peakCudaGiB'] is None else f"{s['peakCudaGiB']:.2f} GiB")
    page = page.replace("__NEXT__", f"{s['nextCheckpoint']/1e6:.0f}M")
    page = page.replace("__STATUS__", str(s['phase']))
    return page.encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            body = json.dumps(snapshot(), separators=(",", ":")).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/events":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.end_headers()
            last = None
            try:
                while True:
                    body = json.dumps(snapshot(), separators=(",", ":"))
                    if body != last:
                        self.wfile.write(("data:" + body + "\n\n").encode()); self.wfile.flush(); last = body
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                return
        if path in ("/", "/index.html"):
            body = html_now()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8766), Handler)
    server.daemon_threads = True
    print("FlashSkip dashboard: http://127.0.0.1:8766", flush=True)
    server.serve_forever()
