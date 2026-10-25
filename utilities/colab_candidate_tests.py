#!/usr/bin/env python3
"""One-shot Colab proxy tests for the five <500M candidate Transformer designs.

This is intentionally a *proxy-scale* experiment (~1.7-2.1M learned params each), not a
claim about final 480-497M benchmark quality. It tests:
  - forward/backward correctness and finite gradients
  - short real next-character training on Tiny Shakespeare
  - held-out loss after a matched token/step budget
  - measured prefill forward latency on the allocated Colab device
  - peak CUDA memory during training
  - architecture-specific control flow (recursion, attention skip, CLA KV reuse, local/global)

The full-size parameter/KV results are computed separately by architecture_math.py.
"""
from __future__ import annotations
import json, math, os, random, statistics, time, urllib.request
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED=20260830
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision('high')
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE',DEVICE)
if torch.cuda.is_available():
    print('GPU',torch.cuda.get_device_name(0))
    print('CUDA',torch.version.cuda)
print('TORCH',torch.__version__)

# ---- data: public-domain Shakespeare, with deterministic fallback ----
URL='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
try:
    text=urllib.request.urlopen(URL,timeout=20).read().decode('utf-8')
    print('DATA_SOURCE',URL,'chars',len(text))
except Exception as e:
    base=("To be, or not to be, that is the question:\nWhether 'tis nobler in the mind to suffer\n"
          "The slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles.\n")
    text=base*8000
    print('DATA_SOURCE fallback_embedded chars',len(text),'error',repr(e))
chars=sorted(set(text)); stoi={c:i for i,c in enumerate(chars)}; itos={i:c for c,i in stoi.items()}
data=torch.tensor([stoi[c] for c in text],dtype=torch.long)
n=int(0.9*len(data)); train_data=data[:n]; val_data=data[n:]
VOCAB=len(chars)
print('VOCAB',VOCAB,'train_tokens',len(train_data),'val_tokens',len(val_data))

SEQ=128; BATCH=16; STEPS=150; EVAL_BATCHES=20

def get_batch(split, batch=BATCH, seq=SEQ):
    src=train_data if split=='train' else val_data
    ix=torch.randint(len(src)-seq-1,(batch,))
    x=torch.stack([src[i:i+seq] for i in ix])
    y=torch.stack([src[i+1:i+seq+1] for i in ix])
    return x.to(DEVICE,non_blocking=True), y.to(DEVICE,non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self,d,dff):
        super().__init__(); self.g=nn.Linear(d,dff,bias=False); self.u=nn.Linear(d,dff,bias=False); self.o=nn.Linear(dff,d,bias=False)
    def forward(self,x): return self.o(F.silu(self.g(x))*self.u(x))

def rope(x,pos):
    # x: B,H,T,D, pos: B,T
    D=x.shape[-1]; half=D//2
    inv=1.0/(10000.0 ** (torch.arange(0,half,device=x.device,dtype=torch.float32)/half))
    freqs=pos.float().unsqueeze(-1)*inv
    cos=freqs.cos().to(x.dtype).unsqueeze(1); sin=freqs.sin().to(x.dtype).unsqueeze(1)
    x1,x2=x[...,:half],x[...,half:half*2]
    rot=torch.cat([x1*cos-x2*sin,x1*sin+x2*cos],dim=-1)
    if D>2*half: rot=torch.cat([rot,x[...,2*half:]],dim=-1)
    return rot

class GQAAttn(nn.Module):
    def __init__(self,d,qh,kvh,has_kv=True,local_window=0):
        super().__init__(); assert d%qh==0 and qh%kvh==0
        self.d=d; self.qh=qh; self.kvh=kvh; self.hd=d//qh; self.has_kv=has_kv; self.local_window=local_window
        self.q=nn.Linear(d,qh*self.hd,bias=False)
        self.k=nn.Linear(d,kvh*self.hd,bias=False) if has_kv else None
        self.v=nn.Linear(d,kvh*self.hd,bias=False) if has_kv else None
        self.o=nn.Linear(qh*self.hd,d,bias=False)
        self.qnorm=RMSNorm(self.hd); self.knorm=RMSNorm(self.hd) if has_kv else None
    def forward(self,x,pos,external_kv=None):
        B,T,_=x.shape
        q=self.q(x).view(B,T,self.qh,self.hd).transpose(1,2)
        q=rope(self.qnorm(q),pos)
        if self.has_kv:
            k=self.k(x).view(B,T,self.kvh,self.hd).transpose(1,2)
            v=self.v(x).view(B,T,self.kvh,self.hd).transpose(1,2)
            k=rope(self.knorm(k),pos)
            source=(k,v)
        else:
            if external_kv is None: raise RuntimeError('CLA follower missing external K/V')
            k,v=external_kv; source=None
        rep=self.qh//self.kvh
        kr=k.repeat_interleave(rep,dim=1); vr=v.repeat_interleave(rep,dim=1)
        if self.local_window and T>1:
            i=torch.arange(T,device=x.device)[:,None]; j=torch.arange(T,device=x.device)[None,:]
            ok=(j<=i)&(j>=i-self.local_window+1)
            mask=torch.zeros((T,T),device=x.device,dtype=x.dtype).masked_fill(~ok,float('-inf'))
            a=F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,is_causal=False)
        else:
            a=F.scaled_dot_product_attention(q,kr,vr,is_causal=True)
        a=a.transpose(1,2).contiguous().view(B,T,self.qh*self.hd)
        return self.o(a),source

class Block(nn.Module):
    def __init__(self,d,dff,qh,kvh,attn=True,has_kv=True,local_window=0):
        super().__init__(); self.has_attn=attn; self.has_kv=has_kv
        if attn:
            self.n1=RMSNorm(d); self.attn=GQAAttn(d,qh,kvh,has_kv=has_kv,local_window=local_window)
        self.n2=RMSNorm(d); self.ff=SwiGLU(d,dff)
    def forward(self,x,pos,external_kv=None):
        src=None
        if self.has_attn:
            a,src=self.attn(self.n1(x),pos,external_kv=external_kv); x=x+a
        x=x+self.ff(self.n2(x))
        return x,src

class BaseLM(nn.Module):
    def __init__(self,vocab,d): super().__init__(); self.tok=nn.Embedding(vocab,d); self.final=RMSNorm(d); self.d=d
    def logits(self,h): return F.linear(self.final(h),self.tok.weight)
    def positions(self,x): return torch.arange(x.shape[1],device=x.device).unsqueeze(0).expand(x.shape[0],-1)

class DenseDeep(BaseLM):
    def __init__(self,vocab):
        super().__init__(vocab,128); self.blocks=nn.ModuleList([Block(128,384,4,1) for _ in range(9)])
    def forward(self,x,targets=None):
        h=self.tok(x); pos=self.positions(x)
        for b in self.blocks: h,_=b(h,pos)
        return self.logits(h),torch.tensor(0.,device=h.device),{}

class FlashSkip(BaseLM):
    def __init__(self,vocab):
        super().__init__(vocab,192); attn={0,2,3}; self.pattern=sorted(attn)
        self.blocks=nn.ModuleList([Block(192,768,6,1,attn=(i in attn)) for i in range(4)])
    def forward(self,x,targets=None):
        h=self.tok(x); pos=self.positions(x)
        for b in self.blocks: h,_=b(h,pos)
        return self.logits(h),torch.tensor(0.,device=h.device),{'attention_layers':len(self.pattern)}

class LocalGlobal(BaseLM):
    def __init__(self,vocab):
        super().__init__(vocab,160); self.blocks=nn.ModuleList([Block(160,480,5,1,local_window=(64 if i<5 else 0)) for i in range(6)])
    def forward(self,x,targets=None):
        h=self.tok(x); pos=self.positions(x)
        for b in self.blocks: h,_=b(h,pos)
        return self.logits(h),torch.tensor(0.,device=h.device),{'local_layers':5,'window':64}

class CLACache(BaseLM):
    def __init__(self,vocab):
        super().__init__(vocab,160)
        # source/follower alternating, final source => 4 K/V producers across 7 attention layers
        self.blocks=nn.ModuleList([Block(160,400,5,1,has_kv=(i%2==0)) for i in range(7)])
    def forward(self,x,targets=None):
        h=self.tok(x); pos=self.positions(x); shared=None; producers=0
        for i,b in enumerate(self.blocks):
            if i%2==0:
                h,shared=b(h,pos); producers+=1
            else:
                h,_=b(h,pos,external_kv=shared); shared=None
        return self.logits(h),torch.tensor(0.,device=h.device),{'kv_producers':producers}

class TapConnector(nn.Module):
    def __init__(self,d=160,r=32,taps=4):
        super().__init__(); self.norms=nn.ModuleList([RMSNorm(d) for _ in range(taps)]); self.down=nn.ModuleList([nn.Linear(d,r,bias=False) for _ in range(taps)])
        self.mid=nn.Linear(r,r*2,bias=False); self.up=nn.Linear(r,d,bias=False); nn.init.zeros_(self.up.weight)
        self.scale=nn.Parameter(torch.zeros(()))
    def forward(self,taps):
        z=sum(p(n(x)) for p,n,x in zip(self.down,self.norms,taps))/len(taps)
        a,b=self.mid(z).chunk(2,-1); z=F.silu(a)*b
        return self.scale*self.up(z)

class EdgeRec(BaseLM):
    def __init__(self,vocab):
        super().__init__(vocab,160)
        self.stem=Block(160,480,5,1); self.middle=nn.ModuleList([Block(160,480,5,1) for _ in range(4)]); self.tail=Block(160,480,5,1)
        self.router=nn.Sequential(nn.Linear(160,32,bias=False),nn.SiLU(),nn.Linear(32,1,bias=False))
        self.conn=TapConnector(160,32,4)
        self.adapters=nn.ModuleList([nn.Sequential(nn.Linear(160,16,bias=False),nn.Linear(16,160,bias=False)) for _ in range(2)])
        for a in self.adapters: nn.init.zeros_(a[1].weight)
        self.route_frac=.5
    def forward(self,x,targets=None):
        emb=self.tok(x); pos=self.positions(x); h,_=self.stem(emb,pos); taps=[]
        for b in self.middle:
            h,_=b(h,pos); taps.append(h)
        router_logits=self.router(h).squeeze(-1)
        B,T,_=h.shape; k=max(1,int(T*self.route_frac)); idx=torch.topk(router_logits,k,dim=1).indices.sort(dim=1).values
        gather_idx=idx.unsqueeze(-1).expand(-1,-1,h.shape[-1]); hp=torch.gather(h,1,gather_idx); pp=torch.gather(pos,1,idx)
        tap_sel=[torch.gather(t,1,gather_idx) for t in taps]
        hp=hp+self.conn(tap_sel)
        # Reuse two physical middle blocks: actual parameter sharing / logical-depth extension.
        for b,ad in zip((self.middle[1],self.middle[2]),self.adapters):
            hp,_=b(hp,pp); hp=hp+ad(hp)
        h=h.scatter(1,gather_idx,hp)
        aux=torch.tensor(0.,device=h.device)
        if targets is not None:
            with torch.no_grad():
                shallow=F.linear(self.final(taps[-1]),self.tok.weight)
                ce=F.cross_entropy(shallow.reshape(-1,shallow.size(-1)),targets.reshape(-1),reduction='none').view(B,T)
                threshold=ce.median(dim=1,keepdim=True).values
                hard=(ce>=threshold).float()
            aux=F.binary_cross_entropy_with_logits(router_logits,hard)
        h,_=self.tail(h,pos)
        return self.logits(h),aux,{'route_fraction':k/T,'recursive_tokens':B*k}

MODELS={
 'edge_rec':EdgeRec,
 'dense_deep':DenseDeep,
 'flash_skip':FlashSkip,
 'cla_cache':CLACache,
 'local_global':LocalGlobal,
}

def nparams(m): return sum(p.numel() for p in m.parameters())

@torch.no_grad()
def eval_loss(model):
    model.eval(); vals=[]
    for _ in range(EVAL_BATCHES):
        x,y=get_batch('val'); logits,aux,_=model(x,y); vals.append(F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)).item())
    return sum(vals)/len(vals)

@torch.no_grad()
def prefill_bench(model,seq=256,iters=40):
    model.eval(); x,_=get_batch('val',batch=1,seq=seq)
    # warmup
    for _ in range(8): model(x,None)
    if DEVICE=='cuda': torch.cuda.synchronize()
    times=[]
    for _ in range(iters):
        t=time.perf_counter(); model(x,None)
        if DEVICE=='cuda': torch.cuda.synchronize()
        times.append((time.perf_counter()-t)*1000)
    return statistics.median(times), statistics.quantiles(times,n=20)[18]

def train_one(name,ctor):
    # reset seed so data sampling/order is repeatable across candidates; model shapes still differ.
    torch.manual_seed(SEED+17); random.seed(SEED+17)
    model=ctor(VOCAB).to(DEVICE)
    P=nparams(model)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,betas=(0.9,0.95),weight_decay=.1)
    if DEVICE=='cuda': torch.cuda.reset_peak_memory_stats()
    start_loss=eval_loss(model)
    step_times=[]; last_loss=None; aux_val=0.; meta={}
    for step in range(STEPS):
        model.train(); x,y=get_batch('train')
        lr=2e-3 * min(1.0,(step+1)/10) * (0.1+0.9*0.5*(1+math.cos(math.pi*step/STEPS)))
        for g in opt.param_groups:g['lr']=lr
        if DEVICE=='cuda': torch.cuda.synchronize()
        t=time.perf_counter(); opt.zero_grad(set_to_none=True)
        logits,aux,meta=model(x,y)
        ce=F.cross_entropy(logits.view(-1,VOCAB),y.view(-1)); loss=ce+0.05*aux
        if not torch.isfinite(loss): raise RuntimeError(f'{name}: non-finite loss at step {step}: {loss}')
        loss.backward(); grad=nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(grad): raise RuntimeError(f'{name}: non-finite gradient at step {step}')
        opt.step()
        if DEVICE=='cuda': torch.cuda.synchronize()
        dt=(time.perf_counter()-t)*1000
        if step>=20: step_times.append(dt)
        last_loss=ce.item(); aux_val=float(aux.detach())
        if step in (0,49,99,149): print('TRAIN',name,'step',step+1,'ce',round(last_loss,4),'aux',round(aux_val,4),'ms',round(dt,2))
    val=eval_loss(model); med,p95=prefill_bench(model)
    peak=(torch.cuda.max_memory_allocated()/1024**2) if DEVICE=='cuda' else 0.0
    return {
      'name':name,'proxy_params':P,'start_val_loss':start_loss,'end_val_loss':val,'val_loss_drop':start_loss-val,
      'last_train_ce':last_loss,'router_aux':aux_val,'train_step_ms_median':statistics.median(step_times),
      'prefill_256_ms_median':med,'prefill_256_ms_p95':p95,'peak_cuda_alloc_MiB':peak,'meta':meta,
    }

results=[]
for name,ctor in MODELS.items():
    print('\n===',name,'===')
    r=train_one(name,ctor); results.append(r); print('DONE',json.dumps(r,sort_keys=True))

# Derived comparison only; lower validation loss is a small-data proxy, not final capability.
rank_loss=sorted(results,key=lambda r:r['end_val_loss'])
rank_prefill=sorted(results,key=lambda r:r['prefill_256_ms_median'])
summary={
 'seed':SEED,'device':DEVICE,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'torch':torch.__version__,'dataset':'Tiny Shakespeare char-level','seq_len':SEQ,'batch':BATCH,'steps':STEPS,
 'results':results,'proxy_rank_by_val_loss':[r['name'] for r in rank_loss],
 'proxy_rank_by_prefill_latency':[r['name'] for r in rank_prefill],
 'interpretation_warning':'Proxy-scale short character training validates trainability/relative structural overhead only; it is not evidence that the same ordering holds at 480-497M after trillion-token pretraining.'
}
print('\nRESULT_JSON='+json.dumps(summary,sort_keys=True))
