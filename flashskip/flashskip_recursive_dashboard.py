#!/usr/bin/env python3
"""Dependency-free live dashboard for the FlashSkip-10M recursive Kaggle P100 run."""

from __future__ import annotations

import json
import hmac
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(os.environ.get("FLASHSKIP_DASHBOARD_ROOT", Path(__file__).resolve().parent / "flashskip10m"))
STATUS = ROOT / "status.json"
METRICS = ROOT / "logs" / "train_metrics.jsonl"
PREP = ROOT / "logs" / "prepare_progress.json"
TELEMETRY_TOKEN = ROOT / "telemetry_token"
WRITE_LOCK = threading.Lock()
START_TOKENS = int(os.environ.get("FLASHSKIP_START_TOKENS", "0"))
TARGET = int(os.environ.get("FLASHSKIP_TARGET_TOKENS", "200000000"))
CHECKPOINT_INTERVAL = int(os.environ.get("FLASHSKIP_CHECKPOINT_TOKENS", "20000000"))
PORT = int(os.environ.get("FLASHSKIP_DASHBOARD_PORT", "8767"))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_metrics() -> list[dict]:
    if not METRICS.exists():
        return []
    try:
        with METRICS.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - 256_000))
            raw = handle.read()
        if size > 256_000 and b"\n" in raw:
            raw = raw.split(b"\n", 1)[1]
        rows = []
        for line in raw.splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
        return rows
    except Exception:
        return []


def snapshot() -> dict:
    status = read_json(STATUS)
    prep = read_json(PREP)
    rows = read_metrics()
    trains = [row for row in rows if row.get("event") == "train"]
    evals = [row for row in rows if row.get("event") in ("initial_eval", "checkpoint_eval")]
    latest = trains[-1] if trains else {}
    latest_eval = evals[-1] if evals else {}
    val = latest_eval.get("val", latest_eval)
    tokens = max(
        int(latest.get("tokens_seen") or 0),
        int(status.get("latest_tokens") or START_TOKENS),
    )
    percent = min(100.0, max(0.0, (tokens - START_TOKENS) / max(1, TARGET - START_TOKENS) * 100))
    throughput = float(latest.get("tokens_per_second") or 0)
    remaining = max(0, TARGET - tokens)
    eta = remaining / throughput if throughput else None
    prep_done = int(prep.get("train_tokens") or 0) + int(prep.get("val_tokens") or 0)
    prep_target = int(prep.get("target_train_tokens") or status.get("prepare_target_tokens") or 210_000_000) + int(prep.get("target_val_tokens") or 1_000_000)
    prep_throughput = float(prep.get("tokens_per_second") or 0)
    if not throughput and prep_throughput:
        eta = max(0, prep_target - prep_done) / prep_throughput
    architecture = status.get("architecture") or {
        "vocab_size": 49152, "d_model": 160, "d_ff": 360, "layers": 10, "unique_layers": 10, "effective_layers": 18,
        "recursion_strategy": "middle-cycle", "recursion_steps": 2, "query_heads": 5, "kv_heads": 1, "attention_layers": [0,1,3,5,7,9],
        "sequence_length": 512, "micro_batch": 32, "grad_accum": 2,
        "global_tokens_per_step": 32768,
    }
    return {
        "model": status.get("model", "FlashSkip-10M-Recursive-MiddleCycle-Nr2"),
        "parameters": int(status.get("parameters") or 9_964_832),
        "phase": status.get("phase", "waiting"),
        "state": status.get("state", "waiting"),
        "segment": status.get("segment"),
        "kernelStatus": status.get("kernel_status"),
        "telemetryBuffered": status.get("provider") == "Kaggle" and status.get("state") == "running" and time.time() - float(status.get("last_live_at") or 0) > 45,
        "runStartedAt": status.get("run_started_at"),
        "session": status.get("session") or status.get("run_id") or "flashskip10m1b",
        "updatedAt": status.get("updated_at"),
        "tokens": tokens,
        "startTokens": START_TOKENS,
        "addedTokens": max(0, tokens - START_TOKENS),
        "target": TARGET,
        "percent": percent,
        "step": max(int(latest.get("step") or 0), int(status.get("latest_step") or 0)),
        "loss": latest.get("loss"),
        "loss20": latest.get("loss_20"),
        "valLoss": val.get("loss") if isinstance(val, dict) else None,
        "perplexity": val.get("perplexity") if isinstance(val, dict) else None,
        "lr": latest.get("lr"),
        "gradNorm": latest.get("grad_norm"),
        "tokensPerSecond": throughput or None,
        "peakCudaGiB": latest.get("peak_cuda_GiB"),
        "elapsedSeconds": latest.get("elapsed_seconds"),
        "etaSeconds": eta,
        "nextCheckpoint": min(TARGET, START_TOKENS + (((max(tokens, START_TOKENS) - START_TOKENS) // CHECKPOINT_INTERVAL) + 1) * CHECKPOINT_INTERVAL),
        "latestCheckpoint": status.get("latest_checkpoint"),
        "dataIntegrity": status.get("data_integrity"),
        "lastError": status.get("last_error"),
        "provider": status.get("provider", "Google Colab"),
        "machine": status.get("machine", "P100"),
        "hourlyCostUsd": status.get("hourly_cost_usd"),
        "prep": {
            "documents": prep.get("documents"),
            "tokens": prep_done,
            "target": prep_target,
            "percent": min(100, prep_done / prep_target * 100) if prep_target else 0,
            "tokensPerSecond": prep.get("tokens_per_second"),
        },
        "architecture": architecture,
        "lossHistory": [{"step": row.get("step"), "loss": row.get("loss_20") or row.get("loss")} for row in trains[-160:]],
    }


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#11110f"><title>FlashSkip Recursive 10M / Training Instrument</title>
<style>
:root{--black:#11110f;--panel:#181816;--cream:#eee9dc;--dim:#8b887f;--line:#373630;--amber:#ffb000;--lime:#c7f35b;--red:#ff644f;--serif:"Iowan Old Style",Baskerville,serif;--mono:"IBM Plex Mono","SFMono-Regular",Consolas,monospace}*{box-sizing:border-box}html{background:var(--black);color:var(--cream)}body{margin:0;font-family:var(--mono);background:radial-gradient(circle at 80% -20%,#3a2b0d 0,transparent 28rem),var(--black);min-height:100vh}.noise{position:fixed;inset:0;opacity:.09;pointer-events:none;z-index:5;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence baseFrequency='.9' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.35'/%3E%3C/svg%3E")}.wrap{width:min(1400px,calc(100% - 40px));margin:auto;padding-bottom:36px}header{height:72px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;font-size:10px;letter-spacing:.12em}.brand{display:flex;align-items:center;gap:14px}.glyph{display:grid;place-items:center;width:32px;height:32px;border:1px solid var(--amber);color:var(--amber);font-family:var(--serif);font-size:20px}.live{display:flex;align-items:center;gap:8px;text-transform:uppercase}.dot{width:7px;height:7px;background:var(--amber);border-radius:50%;box-shadow:0 0 0 5px #ffb00018;animation:pulse 1.5s infinite}.live.done .dot{background:var(--lime)}.live.error .dot{background:var(--red)}.topline{display:grid;grid-template-columns:1.4fr .6fr;border-bottom:1px solid var(--line)}.identity{padding:48px 34px 44px 0;border-right:1px solid var(--line)}.eyebrow{color:var(--amber);font-size:10px;letter-spacing:.18em}.identity h1{font:400 clamp(64px,9vw,132px)/.78 var(--serif);letter-spacing:-.05em;margin:18px 0 25px}.identity h1 i{color:var(--amber);font-weight:400}.identity p{margin:0;color:var(--dim);font-size:11px;line-height:1.8;max-width:55ch}.phase{padding:48px 0 44px 34px;display:flex;flex-direction:column;justify-content:space-between}.phase-label{font-size:10px;color:var(--dim);letter-spacing:.14em}.phase-name{font:400 34px/1.1 var(--serif);margin:12px 0;text-transform:capitalize}.session{font-size:9px;color:var(--dim)}.rail{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:30px}.rail span{height:4px;background:var(--line)}.rail span.on{background:var(--amber)}.progress{padding:30px 0 34px;border-bottom:1px solid var(--line)}.progress-head{display:flex;justify-content:space-between;align-items:end}.token-count{font:400 clamp(40px,5vw,72px)/1 var(--serif)}.target{color:var(--dim);font-size:10px}.percent{font:400 32px/1 var(--serif);color:var(--amber)}.bar{height:22px;border:1px solid var(--line);padding:3px;margin-top:17px;position:relative;overflow:hidden}.fill{height:100%;width:0;background:repeating-linear-gradient(90deg,var(--amber) 0,var(--amber) 8px,#d79000 8px,#d79000 10px);transition:width .5s ease}.bar::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent 60%,#fff3 70%,transparent 80%);transform:translateX(-100%);animation:scan 3s infinite}.prep-line{display:flex;gap:14px;align-items:center;margin-top:12px;color:var(--dim);font-size:9px}.mini{height:2px;background:var(--line);width:120px}.mini i{display:block;height:100%;background:var(--lime);width:0}.dashboard{display:grid;grid-template-columns:1.3fr .7fr;border-bottom:1px solid var(--line)}.chart-panel{padding:34px 34px 34px 0;border-right:1px solid var(--line)}.section-head{display:flex;justify-content:space-between;align-items:center;font-size:9px;letter-spacing:.14em;color:var(--dim)}.current-loss{color:var(--cream);font-size:10px}.chart-wrap{height:270px;margin-top:20px;position:relative;background:linear-gradient(var(--line) 1px,transparent 1px);background-size:100% 25%}.chart-wrap canvas{width:100%;height:100%}.empty-chart{position:absolute;inset:0;display:grid;place-items:center;color:var(--dim);font:italic 22px var(--serif)}.metrics{display:grid;grid-template-columns:repeat(2,1fr)}.metric{min-height:112px;padding:23px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.metric:nth-child(even){border-right:0}.metric:nth-last-child(-n+2){border-bottom:0}.metric small{display:block;color:var(--dim);font-size:8px;letter-spacing:.13em;text-transform:uppercase}.metric strong{display:block;font:400 25px/1 var(--serif);margin-top:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.architecture{padding:34px 0}.architecture-grid{display:grid;grid-template-columns:1.4fr repeat(7,1fr);margin-top:20px;border:1px solid var(--line)}.arch{padding:18px;border-right:1px solid var(--line)}.arch:last-child{border:0}.arch small{display:block;color:var(--dim);font-size:8px;text-transform:uppercase;letter-spacing:.1em}.arch b{display:block;font:400 19px var(--serif);margin-top:8px}.arch.primary{background:var(--cream);color:var(--black)}.arch.primary small{color:#625f57}.footer{display:flex;justify-content:space-between;padding-top:15px;color:var(--dim);font-size:8px;letter-spacing:.1em}.error-box{display:none;margin-top:16px;padding:12px;border:1px solid var(--red);color:var(--red);font-size:9px;line-height:1.5}.error-box.show{display:block}@keyframes pulse{50%{opacity:.45}}@keyframes scan{to{transform:translateX(200%)}}@media(max-width:850px){.topline,.dashboard{grid-template-columns:1fr}.identity,.chart-panel{border-right:0;border-bottom:1px solid var(--line);padding-right:0}.phase{padding-left:0}.architecture-grid{grid-template-columns:repeat(4,1fr)}.arch{border-bottom:1px solid var(--line)}.arch.primary{grid-column:span 4}.identity h1{font-size:82px}}@media(max-width:520px){.wrap{width:calc(100% - 24px)}.identity h1{font-size:64px}.metrics{grid-template-columns:1fr}.metric,.metric:nth-child(even){border-right:0;border-bottom:1px solid var(--line)}.architecture-grid{grid-template-columns:repeat(2,1fr)}.arch.primary{grid-column:span 2}.token-count{font-size:36px}.chart-wrap{height:220px}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body><div class="noise"></div><main class="wrap"><header><div class="brand"><span class="glyph">ƒ</span><span>FLASHSKIP RECURSIVE / TRAINING INSTRUMENT</span></div><div class="live" id="live"><span class="dot"></span><span id="state">CONNECTING</span></div></header><section class="topline"><div class="identity"><div class="eyebrow">EXPERIMENT 10M-REC—1.1B / MIDDLE-CYCLE</div><h1>Small model.<br><i>Long lesson.</i></h1><p>The 9.96M-parameter Middle-Cycle Nr=2 decoder reuses blocks 1–8 twice: 10 unique layers, 18 effective layers. It is continuing through one billion new sequential FineWeb-Edu tokens on one Kaggle P100.</p></div><aside class="phase"><div><div class="phase-label">CURRENT OPERATION</div><div class="phase-name" id="phase">Initializing</div><div class="session" id="session">KAGGLE / NVIDIA TESLA P100</div></div><div class="rail" id="rail"><span></span><span></span><span></span><span></span></div><div class="error-box" id="error"></div></aside></section><section class="progress"><div class="progress-head"><div><div class="token-count" id="tokens">100,007,936</div><div class="target">TOTAL TOKENS / 1,100,007,936 · ONE BILLION NEW</div></div><div class="percent" id="percent">0.000%</div></div><div class="bar"><div class="fill" id="fill"></div></div><div class="prep-line"><span>DATA STREAM</span><div class="mini"><i id="prepFill"></i></div><span id="prepText">WAITING</span><span id="eta">ETA —</span></div></section><section class="dashboard"><div class="chart-panel"><div class="section-head"><span>LOSS TRAJECTORY / 20-STEP MEAN</span><span class="current-loss" id="chartLoss">—</span></div><div class="chart-wrap"><canvas id="chart"></canvas><div class="empty-chart" id="emptyChart">Live loss appears with the first continuation step.</div></div></div><div class="metrics"><div class="metric"><small>Train loss</small><strong id="loss">—</strong></div><div class="metric"><small>Validation loss</small><strong id="valLoss">—</strong></div><div class="metric"><small>Perplexity</small><strong id="ppl">—</strong></div><div class="metric"><small>Step</small><strong id="step">0</strong></div><div class="metric"><small>Throughput</small><strong id="tps">—</strong></div><div class="metric"><small>Learning rate</small><strong id="lr">—</strong></div><div class="metric"><small>Gradient norm</small><strong id="grad">—</strong></div><div class="metric"><small>Peak GPU memory</small><strong id="gpu">—</strong></div></div></section><section class="architecture"><div class="section-head"><span>MODEL CONSTITUTION / LIVE RUN CONFIG</span><span id="checkpoint">NEXT CHECKPOINT / 600M</span></div><div class="architecture-grid"><div class="arch primary"><small>Unique parameters</small><b id="params">9,964,832</b></div><div class="arch"><small>Unique / Effective</small><b id="layers">10 / 18</b></div><div class="arch"><small>Recursion</small><b id="recursion">Middle ×2</b></div><div class="arch"><small>Width</small><b id="width">160</b></div><div class="arch"><small>SwiGLU</small><b id="ff">360</b></div><div class="arch"><small>Q / KV heads</small><b id="heads">5 / 1</b></div><div class="arch"><small>Sequence</small><b id="seq">512</b></div><div class="arch"><small>Global batch</small><b id="batch">32,768</b></div></div></section><div class="footer"><span id="integrity">DATA INTEGRITY — PENDING</span><span id="updated">AWAITING LIVE TELEMETRY</span></div></main>
<script>
const $=id=>document.getElementById(id),fmt=new Intl.NumberFormat('en-US');const n=(v,d=4)=>Number.isFinite(v)?Number(v).toFixed(d):'—';const compact=v=>v>=1e9?(v/1e9).toFixed(2)+'B':v>=1e6?(v/1e6).toFixed(2)+'M':v>=1e3?(v/1e3).toFixed(1)+'K':fmt.format(v||0);const span=s=>{if(!Number.isFinite(s))return'—';let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return`${h}H ${m}M`};function phaseIndex(p){if(/allocat|setting|start|connect/.test(p))return 0;if(/prepar|data_verified/.test(p))return 1;if(/train|retry|running/.test(p))return 2;if(/complete/.test(p))return 3;return 0}function chart(rows){const c=$('chart'),box=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=box.width*dpr;c.height=box.height*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,box.width,box.height);if(rows.length<2){$('emptyChart').style.display='grid';return}$('emptyChart').style.display='none';let vals=rows.map(r=>Number(r.loss)).filter(Number.isFinite),lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo){hi+=.1;lo-=.1}x.beginPath();rows.forEach((r,i)=>{let px=i/(rows.length-1)*box.width,py=12+(hi-r.loss)/(hi-lo)*(box.height-24);i?x.lineTo(px,py):x.moveTo(px,py)});x.strokeStyle='#ffb000';x.lineWidth=2;x.stroke();x.lineTo(box.width,box.height);x.lineTo(0,box.height);x.closePath();let g=x.createLinearGradient(0,0,0,box.height);g.addColorStop(0,'#ffb00044');g.addColorStop(1,'#ffb00000');x.fillStyle=g;x.fill()}function draw(d){let phase=d.telemetryBuffered?`SEGMENT ${d.segment} RUNNING`:((d.phase||'waiting').replaceAll('_',' '));$('state').textContent=d.state==='completed'?'COMPLETE':d.phase==='retry_wait'?'RECOVERING':`LIVE / ${d.machine}`;$('live').className='live '+(d.state==='completed'?'done':d.lastError?'error':'');$('phase').textContent=phase;$('session').textContent=d.provider.toUpperCase()+' / '+d.machine;$('tokens').textContent=fmt.format(d.tokens);$('percent').textContent=d.percent.toFixed(3)+'%';$('fill').style.width=d.percent+'%';$('prepFill').style.width=d.prep.percent+'%';$('prepText').textContent=d.telemetryBuffered?'KAGGLE METRICS BUFFERED':d.prep.percent?`${d.prep.percent.toFixed(1)}% · ${compact(d.prep.tokens)}`:d.dataIntegrity?'VERIFIED':'WAITING';let wall=d.runStartedAt?Math.max(0,(Date.now()-Date.parse(d.runStartedAt))/1000):NaN;$('eta').textContent=`ELAPSED ${span(wall)} · ETA ${span(d.etaSeconds)}`;$('loss').textContent=n(d.loss);$('chartLoss').textContent=n(d.loss20);$('valLoss').textContent=n(d.valLoss);$('ppl').textContent=n(d.perplexity,2);$('step').textContent=fmt.format(d.step);$('tps').textContent=d.tokensPerSecond?compact(d.tokensPerSecond)+'/S':'—';$('lr').textContent=d.lr?Number(d.lr).toExponential(2):'—';$('grad').textContent=n(d.gradNorm,3);$('gpu').textContent=d.peakCudaGiB?n(d.peakCudaGiB,2)+' GiB':'—';$('checkpoint').textContent=d.latestCheckpoint||`NEXT CHECKPOINT / ${compact(d.nextCheckpoint)}`;$('params').textContent=fmt.format(d.parameters);let a=d.architecture;$('layers').textContent=(a.unique_layers||a.layers)+' / '+(a.effective_layers||a.layers);$('recursion').textContent=(a.recursion_strategy||'middle-cycle')+' ×'+(a.recursion_steps||2);$('width').textContent=a.d_model;$('ff').textContent=a.d_ff;$('heads').textContent=a.query_heads+' / '+a.kv_heads;$('seq').textContent=fmt.format(a.sequence_length);$('batch').textContent=fmt.format(a.global_tokens_per_step);$('integrity').textContent=d.latestCheckpoint?'LATEST CHECKPOINT — SHA-256 VERIFIED':d.dataIntegrity?'DATA INTEGRITY — VERIFIED CONTINUATION':'DATA INTEGRITY — PENDING';$('updated').textContent=(d.hourlyCostUsd?'$'+d.hourlyCostUsd.toFixed(2)+'/HR · ':'')+'UPDATED '+new Date().toLocaleTimeString();$('error').className='error-box '+(d.lastError?'show':'');$('error').textContent=d.lastError||'';[...$('rail').children].forEach((e,i)=>e.classList.toggle('on',i<=phaseIndex(d.phase)));chart(d.lossHistory)}let es;function connect(){es=new EventSource('/events');es.onmessage=e=>{try{draw(JSON.parse(e.data))}catch(err){console.error(err)}};es.onerror=()=>{es.close();setTimeout(connect,500)}}addEventListener('resize',()=>fetch('/api/status').then(r=>r.json()).then(draw));connect();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/telemetry":
            self.send_error(404); return
        expected = TELEMETRY_TOKEN.read_text(encoding="utf-8").strip() if TELEMETRY_TOKEN.exists() else ""
        supplied = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not expected or not hmac.compare_digest(supplied, expected):
            self.send_error(401); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise ValueError("invalid payload size")
            payload = json.loads(self.rfile.read(length))
            kind, data = payload["type"], payload["data"]
            if not isinstance(data, dict):
                raise ValueError("data must be an object")
            with WRITE_LOCK:
                status = read_json(STATUS)
                if kind == "train":
                    METRICS.parent.mkdir(parents=True, exist_ok=True)
                    with METRICS.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(data, sort_keys=True) + "\n")
                    status.update({"phase": "training", "latest_tokens": data.get("tokens_seen"), "latest_step": data.get("step"), "last_live_at": time.time()})
                elif kind in ("initial_eval", "checkpoint_eval"):
                    METRICS.parent.mkdir(parents=True, exist_ok=True)
                    with METRICS.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(data, sort_keys=True) + "\n")
                elif kind == "prepare":
                    PREP.parent.mkdir(parents=True, exist_ok=True)
                    temp = PREP.with_suffix(".json.tmp")
                    temp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
                    os.replace(temp, PREP)
                    status.update({"phase": "preparing_data", "last_live_at": time.time()})
                elif kind in ("phase", "prepared"):
                    status.update({k: v for k, v in data.items() if k in {"phase", "machine", "segment", "accelerator", "data_integrity", "run_started_at"}})
                    status["last_live_at"] = time.time()
                else:
                    raise ValueError("unknown telemetry type")
                status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                temp = STATUS.with_suffix(".json.tmp")
                temp.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
                os.replace(temp, STATUS)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_response(204); self.send_header("Content-Length", "0"); self.end_headers()

    def do_GET(self) -> None:
        # The Cloudflare quick tunnel is telemetry-ingress only; keep the dashboard itself local.
        if self.headers.get("CF-Connecting-IP") or self.headers.get("X-Forwarded-For"):
            self.send_error(403); return
        path = urlparse(self.path).path
        if path == "/api/status":
            body = json.dumps(snapshot(), separators=(",", ":")).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/events":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Connection", "keep-alive"); self.end_headers()
            last = ""
            try:
                while True:
                    data = json.dumps(snapshot(), separators=(",", ":"))
                    if data != last:
                        self.wfile.write(("data:" + data + "\n\n").encode()); self.wfile.flush(); last = data
                    time.sleep(.35)
            except (BrokenPipeError, ConnectionResetError):
                return
        if path in ("/", "/index.html"):
            body = HTML.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_error(404)

    def log_message(self, *_: object) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.daemon_threads = True
    print(f"FlashSkip-10M Recursive dashboard: http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
