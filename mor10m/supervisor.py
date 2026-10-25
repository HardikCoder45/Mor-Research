"""Durable local orchestration: CPU data, GPU test, ten checkpointed segments."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent; RUN=ROOT/'run'; FLASH=ROOT.parent/'flashskip'
OWNER='hardikarora83i2492'
SOURCE=OWNER+'/mor-10m-token-choice-source'
RESUME=OWNER+'/mor-10m-token-choice-resume'
PREP=OWNER+'/mor-10m-token-choice-data'
SMOKE=OWNER+'/mor-10m-token-choice-gpu-test'
TRAIN=OWNER+'/mor-10m-token-choice-1b'
ACCELERATOR=os.environ.get('MOR_ACCELERATOR','NvidiaTeslaT4')


def read(path):
    try: return json.loads(path.read_text())
    except (OSError,ValueError): return {}


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)


def state(**fields):
    d=read(RUN/'control.json'); d.update(fields); d['updated_at']=time.time(); write(RUN/'control.json',d)


def command(args,timeout=1800):
    result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
    if result.returncode: raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout


def status(ref):
    try: return command(['kaggle','kernels','status',ref],120)
    except Exception: return 'not_found'


def wait(ref,phase):
    while True:
        s=status(ref); state(phase=phase,kernel_status=s.strip())
        if 'COMPLETE' in s: return
        if any(x in s for x in ('ERROR','CANCEL','FAILED')):
            logs=command(['kaggle','kernels','logs',ref],180)
            (RUN/'failed_kernel.log').write_text(logs)
            raise RuntimeError(f'{ref}: {s}; logs saved locally')
        time.sleep(30)


def publish(folder,ref,title):
    write(folder/'dataset-metadata.json',{'id':ref,'title':title,'licenses':[{'name':'other'}]})
    exists=subprocess.run(['kaggle','datasets','status',ref,'--format','json'],capture_output=True).returncode==0
    cmd=['kaggle','datasets','version' if exists else 'create','-p',str(folder),'-r','zip','-q']
    if exists: cmd+=['-m','Verified MoR source/checkpoint for next segment']
    command(cmd)
    for _ in range(180):
        try:
            if 'ready' in command(['kaggle','datasets','status',ref,'--format','json'],120).lower(): return
        except RuntimeError:
            # A new private dataset can briefly return 403 while indexing.
            pass
        time.sleep(5)
    raise RuntimeError('dataset not ready')


def push(ref,mode,target=None,resume=False):
    folder=RUN/('kernel_'+mode); folder.mkdir(exist_ok=True)
    job={'mode':mode,'target':target,'resume':resume,'accelerator':ACCELERATOR,
         'expected_resume_tokens':read(RUN/'control.json').get('tokens') if resume else None}
    config=read(RUN/'control.json')
    for key in ('prepare_train_tokens','verify_data_prefix','schedule_start_tokens','schedule_end_tokens','max_lr','min_lr','warmup_fraction','val_batches'):
        if key in config: job[key]=config[key]
    # Generated submission artifact, rather than editing any existing run.
    code=(ROOT/'kernel.py').read_text().replace('JOB = {} # injected configuration','JOB = '+repr(job))
    (folder/'kernel.py').write_text(code)
    datasets=[SOURCE]+([RESUME] if resume else [])
    write(folder/'kernel-metadata.json',{'id':ref,'title':ref.split('/')[-1].replace('-',' '),'code_file':'kernel.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':mode!='prepare','enable_internet':True,'dataset_sources':datasets,'kernel_sources':[PREP] if mode=='train' else [],'competition_sources':[],'model_sources':[]})
    if mode=='prepare' and config.get('prepare_docker_image'):
        metadata=read(folder/'kernel-metadata.json')
        metadata.update(docker_image=config['prepare_docker_image'],docker_image_pinning_type='original')
        write(folder/'kernel-metadata.json',metadata)
    cmd=['kaggle','kernels','push','-p',str(folder),'--timeout','10800']
    if mode!='prepare': cmd+=['--accelerator',ACCELERATOR]
    print(command(cmd),flush=True)


def services():
    RUN.mkdir(exist_ok=True); (RUN/'logs').mkdir(exist_ok=True)
    token=RUN/'telemetry_token'
    if not token.exists(): token.write_text(secrets.token_urlsafe(40)); token.chmod(0o600)
    env=os.environ.copy(); env['PYTHONPATH']=str(FLASH)
    try: urllib.request.urlopen('http://127.0.0.1:8771/api/status',timeout=2)
    except Exception:
        handle=(RUN/'dashboard.log').open('a')
        p=subprocess.Popen([sys.executable,str(ROOT/'dashboard.py')],env=env,stdout=handle,stderr=handle,start_new_session=True)
        state(dashboard_pid=p.pid)
    telemetry=read(RUN/'telemetry.json')
    alive=False
    if telemetry.get('pid'):
        try: os.kill(telemetry['pid'],0); alive=True
        except OSError: pass
    if not alive:
        log=RUN/'tunnel.log'; handle=log.open('w')
        p=subprocess.Popen(['/opt/homebrew/bin/cloudflared','tunnel','--no-autoupdate','--url','http://127.0.0.1:8771'],stdout=handle,stderr=handle,start_new_session=True)
        for _ in range(60):
            matches=re.findall(r'https://[a-z0-9-]+\.trycloudflare.com',log.read_text())
            if matches:
                telemetry={'url':matches[-1],'token':token.read_text().strip(),'pid':p.pid}; write(RUN/'telemetry.json',telemetry); (RUN/'telemetry.json').chmod(0o600); break
            time.sleep(1)
        else: raise RuntimeError('telemetry tunnel failed')
    # Test the complete public ingress path without exposing the dashboard.
    request=urllib.request.Request(telemetry['url']+'/api/telemetry',data=json.dumps({'type':'phase','data':{'phase':'connected'}}).encode(),headers={'Authorization':'Bearer '+telemetry['token'],'Content-Type':'application/json'})
    for attempt in range(3):
        try: urllib.request.urlopen(request,timeout=10); break
        except Exception:
            if attempt==2:
                # Some local resolvers cache NXDOMAIN for fresh tunnel names.
                # Resolve with DoH, keeping HTTPS hostname/certificate verification.
                host=urlparse(telemetry['url']).hostname
                dns=json.loads(command(['curl','-fsS','https://dns.google/resolve?name='+host+'&type=A'],30))
                ip=next(x['data'] for x in dns.get('Answer',[]) if x['type']==1)
                cfg='url = '+json.dumps(telemetry['url']+'/api/telemetry')+'\n'
                cfg+='resolve = '+json.dumps(host+':443:'+ip)+'\n'
                cfg+='header = '+json.dumps('Authorization: Bearer '+telemetry['token'])+'\n'
                cfg+='header = "Content-Type: application/json"\n'
                cfg+='data = '+json.dumps('{"type":"phase","data":{"phase":"connected"}}')+'\n'
                check=subprocess.run(['curl','--config','-','--fail','--silent','--show-error','--max-time','15'],input=cfg,text=True,capture_output=True)
                if check.returncode: raise RuntimeError('public telemetry ingress verification failed: '+check.stderr)
                break
            time.sleep(5)


def cli_json(output):
    # Kaggle CLI may prepend its version notice even with --format json.
    notice="Warning: Looks like you're using an outdated `kaggle` version"
    cleaned='\n'.join(line for line in output.splitlines() if not line.startswith(notice))
    return json.loads(cleaned)


def quota_wait(required):
    while True:
        q=cli_json(command(['kaggle','quota','--format','json'],120))
        # Parse the CLI's actual structured schema, not a guessed remaining limit.
        rows=q if isinstance(q,list) else q.get('quotas',[])
        gpu=next((r for r in rows if str(r.get('resource','')).upper()=='GPU'),None)
        if gpu is None: raise RuntimeError(f'Unknown quota schema: {q}')
        remaining=float(str(gpu['remaining']).removesuffix('h'))
        state(gpu_hours_remaining=remaining,quota_refresh=gpu.get('refreshAt'))
        if remaining>=required: return
        state(phase='quota_wait',kernel_status=f'{remaining:.2f} GPU hours left; waiting for quota refresh')
        time.sleep(300)


def download(ref,folder,pattern=None):
    folder.mkdir(parents=True,exist_ok=True)
    cmd=['kaggle','kernels','output',ref,'-p',str(folder)]
    if pattern: cmd+=['--file-pattern',pattern]
    command(cmd,1800)


def storage_wait():
    # Leave room for full FP32 resume state, download staging and metadata.
    while shutil.disk_usage(RUN).free < 768 * 2**20:
        state(phase='storage_wait',error='Less than 768 MiB free on Mac; waiting before checkpoint transfer or new GPU submission')
        time.sleep(60)
    state(error=None)


def run():
    global SOURCE,RESUME,PREP,TRAIN
    config=read(RUN/'control.json')
    SOURCE=config.get('source_ref',SOURCE); RESUME=config.get('resume_ref',RESUME)
    PREP=config.get('prepare_ref',PREP); TRAIN=config.get('train_ref',TRAIN)
    RUN.mkdir(exist_ok=True); lock=(RUN/'supervisor.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state(pid=os.getpid(),started_at=read(RUN/'control.json').get('started_at',time.time()),phase='initializing',error=None)
    subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    services()
    source=RUN/'source_dataset'; source.mkdir(exist_ok=True)
    for name in ('model.py','train.py','prepare.py','tests.py'): shutil.copy2(ROOT/name,source/name)
    shutil.copy2(FLASH/'flashskip_100m.py',source/'flashskip_100m.py')
    shutil.copy2(RUN/'telemetry.json',source/'telemetry.json')
    for name in ('tokenizer.json','tokenizer_config.json'): shutil.copy2(FLASH/'models/mac_test/tokenizer'/name,source/name)
    fingerprint=hashlib.sha256(b''.join((source/name).read_bytes() for name in ('model.py','train.py','prepare.py','tests.py','telemetry.json'))).hexdigest()
    if read(RUN/'control.json').get('source_fingerprint')!=fingerprint:
        state(phase='publishing_source'); publish(source,SOURCE,'MoR 10M Token Choice Source'); state(source_published=True,source_fingerprint=fingerprint)
    if not read(RUN/'control.json').get('prep_submitted'):
        push(PREP,'prepare'); state(prep_submitted=True)
    if not (RUN/'benchmark/benchmark.json').exists():
        if not read(RUN/'control.json').get('smoke_submitted'):
            quota_wait(.15); push(SMOKE,'smoke'); state(smoke_submitted=True)
        wait(SMOKE,'gpu_validation'); download(SMOKE,RUN/'benchmark','benchmark.json')
    benchmark=read(RUN/'benchmark/benchmark.json')
    if not benchmark.get('passed'): raise RuntimeError('GPU validation failed')
    tps=benchmark['last']['tokens_per_second']; state(benchmark_tokens_per_second=tps,estimated_training_hours=1e9/tps/3600)
    wait(PREP,'preparing_data')
    tokens=read(RUN/'control.json').get('tokens',0)
    final_target=int(read(RUN/'control.json').get('final_target',1_000_000_000))
    targets=read(RUN/'control.json').get('segment_targets',list(range(100_000_000,final_target+1,100_000_000)))
    if not targets or targets[-1]!=final_target or targets!=sorted(set(targets)): raise RuntimeError('invalid segment targets')
    for target in targets:
        if tokens>=target: continue
        previous=read(RUN/'control.json')
        if previous.get('submitted_target')!=target:
            storage_wait()
            if tokens and previous.get('published_resume_tokens')!=tokens:
                cp=Path(previous['latest_checkpoint']); manifest=read(cp/'checkpoint_manifest.json')
                resume=RUN/'resume_dataset'; resume.mkdir(exist_ok=True)
                for name in (*manifest['files'],'checkpoint_manifest.json'): shutil.copy2(cp/name,resume/name)
                publish(resume,RESUME,'MoR 10M Token Choice Resume'); state(published_resume_tokens=tokens)
            quota_wait(max(.4,100_000_000/tps/3600*1.3+.15))
            push(TRAIN,'train',target,bool(tokens)); state(submitted_target=target,phase='training')
        wait(TRAIN,'training')
        storage_wait()
        folder=RUN/'downloads'/str(target); download(TRAIN,folder)
        cp=next(folder.rglob('checkpoint_manifest.json')).parent
        manifest=read(cp/'checkpoint_manifest.json')
        for name,info in manifest['files'].items():
            if hashlib.sha256((cp/name).read_bytes()).hexdigest()!=info['sha256']: raise RuntimeError('download checksum mismatch')
        tokens=int(manifest['tokens_seen'])
        if tokens<target: raise RuntimeError('segment finished below target')
        state(tokens=tokens,latest_checkpoint=str(cp),phase='checkpoint_downloaded')
        if read(RUN/'control.json').get('require_review'):
            decision_path=RUN/'reviews'/f'{target}.json'
            state(phase='review_pending',review_target=target,review_checkpoint=str(cp))
            while True:
                decision=read(decision_path)
                if decision.get('checkpoint_tokens')==tokens and decision.get('action') in ('continue','stop'):
                    break
                time.sleep(30)
            if decision['action']=='stop':
                state(phase='stopped_after_review',review_decision=decision)
                return
            state(phase='review_approved',review_decision=decision)
        if target<final_target:
            resume=RUN/'resume_dataset'; resume.mkdir(exist_ok=True)
            for name in (*manifest['files'],'checkpoint_manifest.json'): shutil.copy2(cp/name,resume/name)
            publish(resume,RESUME,'MoR 10M Token Choice Resume')
            state(published_resume_tokens=tokens)
    state(phase='completed',kernel_status=f'Target {final_target:,} reached; checkpoints downloaded and verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--launch',action='store_true'); a=p.parse_args()
    if a.launch:
        RUN.mkdir(exist_ok=True); handle=(RUN/'supervisor.log').open('a')
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],stdout=handle,stderr=handle,start_new_session=True)
        print(json.dumps({'started':True,'pid':child.pid,'dashboard':'http://127.0.0.1:8771'}))
    else:
        try: run()
        except Exception as e: state(phase='failed',error=f'{type(e).__name__}: {e}'); raise
