"""Run the fixed local diagnostics on a verified MoR checkpoint."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flashskip'))
sys.path.insert(0,str(ROOT/'mor10m'))
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from model import Config, MoR

def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(42)
    manifest=json.loads((a.checkpoint/'checkpoint_manifest.json').read_text())
    for name,info in manifest['files'].items():
        with (a.checkpoint/name).open('rb') as f:
            digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=info['sha256']: raise RuntimeError('Checksum mismatch: '+name)
    cfg=Config.from_dict(json.loads((a.checkpoint/'config.json').read_text()))
    model=MoR(cfg).eval(); model.load_state_dict(load_file(str(a.checkpoint/'model.safetensors')),strict=True)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'mor10m/run/source_dataset'),local_files_only=True)
    spec=importlib.util.spec_from_file_location('diagnostics',ROOT/'flashskip/benchmarks/recursive_100m_to_1100m/benchmark_recursive_checkpoints.py')
    b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    device=torch.device('cpu'); start=time.time()
    result={'checkpoint':str(a.checkpoint.resolve()),'tokens':manifest['tokens_seen'],'recorded':json.loads((a.checkpoint/'metrics.json').read_text()),'heldout':b.eval_heldout(model,tokenizer,device),'grammar':b.eval_grammar(model,tokenizer,device)}
    for label,items in [('cloze',b.CLOZE),('facts',b.FACTS),('commonsense',b.COMMONSENSE),('arithmetic',b.ARITHMETIC_LOGIC)]:
        result[label]=b.eval_mc_category(model,tokenizer,device,items)
    result['generation']=b.eval_generation(model,tokenizer,device,24)
    counts=torch.zeros(3); total=0
    with torch.inference_mode():
        for text in b.HELDOUT_TEXTS:
            ids=tokenizer(text,return_tensors='pt')['input_ids']
            _,_,aux=model(ids,return_aux=True)
            counts+=aux['depth_fractions'].cpu()*ids.numel(); total+=ids.numel()
    result['depth_fractions']=(counts/total).tolist(); result['seconds']=time.time()-start
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'tokens':result['tokens'],'heldout_ppl':result['heldout']['perplexity'],'depth_fractions':result['depth_fractions'],'report':str(a.output)}))

if __name__=='__main__': main()
