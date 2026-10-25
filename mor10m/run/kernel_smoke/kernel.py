"""Kaggle entrypoint. JOB is filled by the local supervisor at submission."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

JOB = {'mode': 'smoke', 'target': None, 'resume': False, 'accelerator': 'NvidiaTeslaT4'}


def main():
    source=next(Path('/kaggle/input').rglob('model.py')).parent
    work=Path('/kaggle/working/code'); work.mkdir(exist_ok=True)
    for name in ('model.py','train.py','prepare.py','tests.py','flashskip_100m.py'):
        shutil.copy2(source/name,work/name)
    cfg=json.loads((source/'telemetry.json').read_text())
    os.environ['FLASHSKIP_TELEMETRY_URL']=cfg['url']; os.environ['FLASHSKIP_TELEMETRY_TOKEN']=cfg['token']
    os.environ['TOKENIZERS_PARALLELISM']='true'
    os.environ['PYTHONPATH']=str(work); sys.path.insert(0,str(work))
    import torch
    if JOB['mode']!='prepare':
        names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        expected='P100' if JOB.get('accelerator')=='NvidiaTeslaP100' else 'T4'
        if not names or not all(expected in name for name in names): raise RuntimeError(f'Expected {expected}, got {names}')
        if expected=='P100' and 'sm_60' not in torch.cuda.get_arch_list():
            if os.environ.get('MOR_TORCH_PINNED'): raise RuntimeError('torch still lacks P100 support')
            subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--upgrade','torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu126'],check=True)
            os.environ['MOR_TORCH_PINNED']='1'; os.execv(sys.executable,[sys.executable,__file__])
        torch.ones(8,device='cuda').sum().item()
        print('ACCELERATOR',json.dumps({'names':names,'torch':torch.__version__}),flush=True)
    if JOB['mode']=='prepare':
        subprocess.run([sys.executable,str(work/'prepare.py')],check=True); return
    if JOB['mode']=='smoke':
        subprocess.run([sys.executable,str(work/'tests.py')],check=True)
        # Synthetic benchmark tokens must never appear as real training progress.
        os.environ.pop('FLASHSKIP_TELEMETRY_URL',None)
        import numpy as np
        data=Path('/kaggle/working/smoke_data'); data.mkdir(exist_ok=True)
        rng=np.random.default_rng(42)
        for name in ('train','val'): rng.integers(0,49152,400000,dtype=np.uint16).tofile(data/f'{name}_tokens.uint16')
        output=Path('/kaggle/working/smoke_result')
        started=time.time()
        subprocess.run([sys.executable,str(work/'train.py'),'--data',str(data),'--output',str(output),'--target','163840','--micro','16','--smoke'],check=True)
        rows=[json.loads(x) for x in (output/'metrics.jsonl').read_text().splitlines()]
        benchmark={'passed':True,'wall_seconds':time.time()-started,'last':rows[-1]}
        Path('/kaggle/working/benchmark.json').write_text(json.dumps(benchmark))
        print('BENCHMARK',json.dumps(benchmark),flush=True)
        return
    data=next(Path('/kaggle/input').rglob('data_manifest.json')).parent
    manifest=json.loads((data/'data_manifest.json').read_text())
    import flashskip_100m as base
    for name,digest in manifest['sha256'].items():
        if base.sha256_file(data/name)!=digest: raise RuntimeError('data hash mismatch')
    cmd=[sys.executable,str(work/'train.py'),'--data',str(data),'--output','/kaggle/working/result','--target',str(JOB['target']),'--micro','16']
    if JOB.get('resume'):
        checkpoint=next(Path('/kaggle/input').rglob('checkpoint_manifest.json')).parent
        cmd += ['--resume',str(checkpoint)]
    subprocess.run(cmd,check=True)

if __name__=='__main__': main()
