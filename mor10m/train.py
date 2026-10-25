"""Resumable 100M segments of a 1B-token MoR pretraining schedule."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
import flashskip_100m as base
from model import Config, MoR

TARGET=1_000_000_000
GLOBAL=32768


def emit(kind,data):
    print(kind.upper(),json.dumps(data),flush=True)
    base.emit_live(kind,data)


def checkpoint(root,model,opt,scaler,step,tokens,cursor,metrics):
    path=root/"checkpoints"/f"checkpoint_{tokens:010d}"
    tmp=path.with_name(path.name+".partial"); tmp.mkdir(parents=True,exist_ok=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp/"model.safetensors"))
    # Keep FP32 weights AND optimizer moments. No lossy resume serialization.
    torch.save({"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"step":step,"tokens":tokens,"cursor":cursor,
                "rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng":random.getstate(),"numpy_rng":np.random.get_state()},tmp/"training_state.pt")
    base.json_dump(tmp/"config.json",model.cfg.to_jsonable())
    base.json_dump(tmp/"metrics.json",metrics)
    files={p.name:{"sha256":base.sha256_file(p),"bytes":p.stat().st_size} for p in tmp.iterdir() if p.is_file()}
    base.json_dump(tmp/"checkpoint_manifest.json",{"tokens_seen":tokens,"step":step,"data_cursor":cursor,"files":files})
    if path.exists(): raise RuntimeError(f"refusing overwrite {path}")
    tmp.rename(path)
    emit("phase",{"phase":"checkpoint_saved","latest_checkpoint":path.name,"latest_tokens":tokens})
    return path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--output",required=True)
    p.add_argument("--resume"); p.add_argument("--target",type=int,default=100_000_000)
    p.add_argument("--micro",type=int,default=16); p.add_argument("--smoke",action="store_true")
    p.add_argument('--schedule-start-tokens',type=int,default=0); p.add_argument('--schedule-end-tokens',type=int,default=TARGET)
    p.add_argument('--max-lr',type=float,default=6e-4); p.add_argument('--min-lr',type=float,default=6e-5)
    p.add_argument('--warmup-fraction',type=float,default=.02); p.add_argument('--val-batches',type=int,default=32)
    a=p.parse_args(); base.seed_all(20260904)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda" and not a.smoke: raise RuntimeError("CUDA required")
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    cfg=Config(); model=MoR(cfg).to(device)
    opt=torch.optim.AdamW([{"params":[v for v in model.parameters() if v.ndim>=2],"weight_decay":.1},
                          {"params":[v for v in model.parameters() if v.ndim<2],"weight_decay":0}],
                         lr=6e-4,betas=(.9,.95),eps=1e-8,fused=device.type=="cuda")
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    step=tokens=cursor=0
    if a.resume:
        cp=Path(a.resume); m=json.loads((cp/"checkpoint_manifest.json").read_text())
        for name,info in m["files"].items():
            if base.sha256_file(cp/name)!=info["sha256"]: raise RuntimeError("resume checksum mismatch")
        model.load_state_dict(load_file(str(cp/"model.safetensors")))
        s=torch.load(cp/"training_state.pt",map_location="cpu",weights_only=False)
        opt.load_state_dict(s["optimizer"]); base.restore_optimizer_device_dtype(opt); scaler.load_state_dict(s["scaler"])
        step,tokens,cursor=s["step"],s["tokens"],s["cursor"]
        torch.set_rng_state(s["rng"])
        if s["cuda_rng"] and device.type=="cuda": torch.cuda.set_rng_state_all(s["cuda_rng"])
        random.setstate(s["python_rng"]); np.random.set_state(s["numpy_rng"])
    data=Path(a.data); train=base.memmap_tokens(data/"train_tokens.uint16"); val=base.memmap_tokens(data/"val_tokens.uint16")
    seq=512; accum=GLOBAL//(a.micro*seq)
    assert a.micro*seq*accum==GLOBAL
    assert cursor==step*(GLOBAL+GLOBAL//seq),"data cursor/step mismatch"
    if len(train)<math.ceil(a.target/GLOBAL)*(GLOBAL+GLOBAL//seq): raise RuntimeError('training stream too short; refusing token replay')
    if not (0<=a.schedule_start_tokens<=tokens<a.target<=a.schedule_end_tokens): raise ValueError('invalid continuation schedule bounds')
    if a.schedule_start_tokens%GLOBAL or not (0<=a.warmup_fraction<1) or not (0<a.min_lr<=a.max_lr): raise ValueError('invalid schedule settings')
    if a.val_batches<1: raise ValueError('validation batches must be positive')
    nsteps=math.ceil((a.schedule_end_tokens-a.schedule_start_tokens)/GLOBAL); started=time.time(); start_tokens=tokens
    emit("phase",{"phase":"training","parameters":sum(v.numel() for v in model.parameters()),"machine":torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU", "latest_tokens":tokens})
    evaluation=base.evaluate_memmap_loss(model,val,device,seq,4 if a.smoke else a.val_batches,2)
    emit("initial_eval",{"event":"initial_eval","step":step,"tokens_seen":tokens,**evaluation})
    recent=[]; next_eval=((tokens//10_000_000)+1)*10_000_000
    while tokens<a.target:
        model.train() # evaluation sets eval mode; always restore explicitly
        step_start=time.perf_counter(); opt.zero_grad(set_to_none=True)
        lr=base.lr_for_step(step-a.schedule_start_tokens//GLOBAL,nsteps,math.ceil(nsteps*a.warmup_fraction),a.max_lr,a.min_lr)
        for group in opt.param_groups: group["lr"]=lr
        ce_sum=aux_sum=0.; fractions=torch.zeros(cfg.recursions,device=device)
        before_cursor=cursor
        for _ in range(accum):
            x,y,cursor=base.next_sequential_batch(train,cursor,a.micro,seq,device,allow_wrap=False)
            with base.autocast_ctx(device):
                output,aux,route=model(x,return_aux=True)
                ce=F.cross_entropy(output.reshape(-1,cfg.vocab_size),y.reshape(-1))
                loss=(ce+aux)/accum
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss; stopped before corrupting checkpoint")
            scaler.scale(loss).backward(); ce_sum+=float(ce.detach())/accum; aux_sum+=float(aux.detach())/accum
            fractions+=route["depth_fractions"]/accum
        scaler.unscale_(opt); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        old_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()<old_scale:
            cursor=before_cursor # overflow did NOT train these tokens; retry same batch
            emit("phase",{"phase":"amp_overflow_retry"}); continue
        if not torch.isfinite(grad): raise RuntimeError("non-finite gradient")
        step+=1; tokens+=GLOBAL
        if device.type=="cuda": torch.cuda.synchronize()
        dt=time.perf_counter()-step_start; recent.append(ce_sum); recent=recent[-20:]
        if step%5==0 or tokens-start_tokens==GLOBAL:
            row={"event":"train","step":step,"tokens_seen":tokens,"loss":ce_sum,"loss_20":sum(recent)/len(recent),
                 "aux_loss":aux_sum,"depth_fractions":fractions.tolist(),"mean_recursions":float((fractions*torch.arange(1,4,device=device)).sum()),
                 "grad_norm":float(grad),"lr":lr,"tokens_per_second":GLOBAL/dt,"effective_tokens_per_second":(tokens-start_tokens)/(time.time()-started),
                 "elapsed_seconds":time.time()-started,"peak_cuda_GiB":torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0}
            base.append_jsonl(root/"metrics.jsonl",row); emit("train",row)
        if tokens>=next_eval or tokens>=a.target:
            evaluation=base.evaluate_memmap_loss(model,val,device,seq,4 if a.smoke else a.val_batches,2)
            emit("checkpoint_eval",{"event":"checkpoint_eval","step":step,"tokens_seen":tokens,"val":evaluation})
            next_eval+=10_000_000
    cp=checkpoint(root,model,opt,scaler,step,tokens,cursor,{"val":evaluation,"loss":sum(recent)/len(recent),"tokens_seen":tokens,
        "schedule":{"start_tokens":a.schedule_start_tokens,"end_tokens":a.schedule_end_tokens,"max_lr":a.max_lr,"min_lr":a.min_lr,"warmup_fraction":a.warmup_fraction},"validation_batches":a.val_batches})
    emit("phase",{"phase":"segment_completed","latest_tokens":tokens,"latest_checkpoint":cp.name})

if __name__=="__main__": main()
