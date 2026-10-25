"""Local-only MoR instrument with authenticated telemetry ingress."""
import json
from pathlib import Path
import time
from http.server import ThreadingHTTPServer
import flashskip_recursive_dashboard as dashboard

ROOT=Path(__file__).resolve().parent/'run'
dashboard.ROOT=ROOT; dashboard.STATUS=ROOT/'telemetry_status.json'
dashboard.METRICS=ROOT/'logs/train_metrics.jsonl'; dashboard.PREP=ROOT/'logs/prepare_progress.json'
dashboard.TELEMETRY_TOKEN=ROOT/'telemetry_token'


def snapshot():
    control=dashboard.read_json(ROOT/'control.json'); remote=dashboard.read_json(dashboard.STATUS)
    prep=dashboard.read_json(dashboard.PREP); rows=dashboard.read_metrics()
    trains=[x for x in rows if x.get('event')=='train' and not x.get('smoke')]
    last=trains[-1] if trains else {}; ev=[x for x in rows if x.get('event') in ('initial_eval','checkpoint_eval')]
    val=ev[-1].get('val',ev[-1]).get('loss') if ev else None
    age=time.time()-remote.get('last_live_at',0)
    tps=last.get('effective_tokens_per_second') or last.get('tokens_per_second') or control.get('benchmark_tokens_per_second') or 0
    tokens=max(control.get('tokens',0),last.get('tokens_seen',0))
    phase=remote.get('phase') if age<45 and control.get('phase') in ('preparing_data','training') else control.get('phase','initializing')
    target=control.get('final_target',1_000_000_000)
    return {**control,'phase':phase,'tokens':tokens,'target':target,'live':age<45,'age':age,'latest':last,'val_loss':val,'prep':prep,
            'elapsed':time.time()-control.get('started_at',time.time()),'eta':(target-tokens)/tps if tps else None}


dashboard.snapshot=snapshot
dashboard.HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MoR / 10M Training</title>
<style>:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#111310;color:#e6e7d9;font:14px Menlo,monospace}main{max-width:1150px;margin:auto;padding:42px 26px}header{display:flex;justify-content:space-between;border-bottom:1px solid #3d4236;padding-bottom:18px;font-size:12px;letter-spacing:.12em}h1{font:68px Georgia,serif;letter-spacing:-.05em;margin:45px 0 12px}h1 i{color:#d1ee80}.sub{max-width:740px;color:#b3b8a7;line-height:1.7}.progress{margin:38px 0 28px}.big{font-size:38px;letter-spacing:-.05em}.bar{height:10px;background:#30362a;margin:16px 0}.bar span{display:block;height:100%;background:#d1ee80;transition:width .3s}.small{font-size:11px;line-height:1.8;color:#b3b8a7}.grid{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid #3d4236;border-left:1px solid #3d4236}.card{padding:20px;border-right:1px solid #3d4236;border-bottom:1px solid #3d4236}.card label{display:block;color:#a4ac98;font-size:10px;text-transform:uppercase;letter-spacing:.1em}.card b{display:block;font-size:23px;margin-top:12px;font-weight:400}.chart{margin:28px 0;border-bottom:1px solid #3d4236}canvas{width:100%;height:170px}.routes{display:flex;gap:25px;margin:24px 0}.route{flex:1;border-top:2px solid #d1ee80;padding-top:12px}footer{color:#8c9580;line-height:1.8;margin-top:35px}a{color:#d1ee80}#error{color:#ffae93;white-space:pre-wrap}.live{color:#d1ee80}@media(max-width:650px){h1{font-size:48px}.grid{grid-template-columns:repeat(2,1fr)}.big{font-size:29px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}</style>
<main><header><span>MoR / RECURSIVE LANGUAGE LAB</span><span id="signal">CONNECTING</span></header>
<h1>Same weights.<br><i>Different depths.</i></h1><p class="sub">9,965,344 learned parameters. One router assigns each token 1–3 passes through five shared layers. A new model, trained from scratch on FineWeb-Edu.</p>
<div class="progress"><div class="big"><span id="tokens">0</span> <span class="small">/ <span id="target">1,000,000,000</span> TOKENS</span></div><div class="bar"><span id="fill" style="width:0"></span></div><div class="small" id="phase">Initializing</div><div class="small" id="prep"></div></div>
<section class="grid" id="metrics"></section><div class="chart"><p class="small">CROSS-ENTROPY LOSS / RECENT TRAINING STEPS</p><canvas id="plot"></canvas></div><div class="routes" id="routes"></div>
<div id="error"></div><footer>7 unique blocks · up to 17 effective layers · 160 width · 496 SwiGLU · 512 context · 32,768 labels/step<br>100M checkpoint cadence · FP32 optimizer state · document-disjoint validation · no data wrap<br><a href="https://arxiv.org/abs/2507.10524">Mixture-of-Recursions paper</a> · 10M is an experimental adaptation, not a published paper result.</footer></main>
<script>const $=x=>document.getElementById(x),num=x=>x==null?'—':Number(x).toLocaleString(undefined,{maximumFractionDigits:3}),hours=x=>x==null?'—':(x/3600).toFixed(2)+' h';let losses=[];let previous=-1;function draw(d){$('signal').textContent=d.live?'● LIVE TELEMETRY':'○ WAITING / STALE';$('signal').className=d.live?'live':'';$('tokens').textContent=num(d.tokens);$('target').textContent=num(d.target);$('fill').style.width=Math.min(100,d.tokens/d.target*100)+'%';$('phase').textContent=(d.phase||'waiting').replaceAll('_',' ').toUpperCase()+' · '+(d.kernel_status||'');let p=d.prep||{};$('prep').textContent=p.train_tokens?'DATA PREPARATION '+num(p.train_tokens)+' / '+num(p.target_train_tokens)+' · '+num(p.tokens_per_second)+' tokens/s':'';let l=d.latest||{};let cards=[['Train loss',num(l.loss)],['Validation loss',num(d.val_loss)],['Tokens / second',num(l.tokens_per_second)],['Training ETA',hours(d.eta)],['Run elapsed',hours(d.elapsed)],['Mean recursions',num(l.mean_recursions)],['Learning rate',l.lr?l.lr.toExponential(2):'—'],['Gradient norm',num(l.grad_norm)]];$('metrics').replaceChildren(...cards.map(([k,v])=>{let x=document.createElement('div');x.className='card';let a=document.createElement('label');a.textContent=k;let b=document.createElement('b');b.textContent=v;x.append(a,b);return x}));$('routes').replaceChildren(...(l.depth_fractions||[0,0,0]).map((v,i)=>{let e=document.createElement('div');e.className='route';e.textContent=(i+1)+' PASS'+(i?'ES':'')+' / '+(v*100).toFixed(1)+'%';return e}));$('error').textContent=d.error||'';if(l.step&&l.step!==previous){losses.push(l.loss);losses=losses.slice(-300);previous=l.step}let c=$('plot'),r=c.getBoundingClientRect();c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;let ctx=c.getContext('2d');ctx.scale(devicePixelRatio,devicePixelRatio);if(losses.length>1){let lo=Math.min(...losses),hi=Math.max(...losses);ctx.beginPath();losses.forEach((v,i)=>{let x=i/(losses.length-1)*r.width,y=10+(hi-v)/Math.max(.01,hi-lo)*(r.height-20);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.strokeStyle='#d1ee80';ctx.lineWidth=2;ctx.stroke()}}let es=new EventSource('/events');es.onmessage=e=>draw(JSON.parse(e.data));es.onerror=()=>{$('signal').textContent='CONNECTION INTERRUPTED'};</script></html>'''

dashboard.HTML=dashboard.HTML.replace('100M checkpoint cadence','Compute ETA excludes preparation, transfers, and quota waits.<br>100M checkpoint cadence')

if __name__=='__main__':
    ROOT.mkdir(exist_ok=True)
    ThreadingHTTPServer(('127.0.0.1',8771),dashboard.Handler).serve_forever()
