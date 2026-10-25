"""Pinned FineWeb-Edu; document-disjoint holdout; exact duplicate rejection."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import numpy as np
from transformers import AutoTokenizer
import flashskip_100m as base

REVISION="87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
TRAIN_TOKENS=1_002_000_000 # 1B labels plus the packed sequence boundary tokens


def prefix_sha256(path, size):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        while size:
            block=handle.read(min(size,8<<20))
            if not block: raise RuntimeError('file shorter than required prefix')
            digest.update(block); size-=len(block)
    return digest.hexdigest()


def main():
    from datasets import load_dataset
    parser=argparse.ArgumentParser(); parser.add_argument('--train-tokens',type=int,default=TRAIN_TOKENS)
    parser.add_argument('--verify-prefix-manifest'); args=parser.parse_args()
    target=args.train_tokens
    if target<TRAIN_TOKENS: raise ValueError('extended stream cannot be shorter than original')
    root=Path('/kaggle/working/prepared'); root.mkdir(parents=True,exist_ok=True)
    tokpath=next(Path('/kaggle/input').rglob('tokenizer.json')).parent
    tok=AutoTokenizer.from_pretrained(str(tokpath),local_files_only=True)
    stream=load_dataset('HuggingFaceFW/fineweb-edu','sample-10BT',revision=REVISION,split='train',streaming=True).shuffle(seed=20260904,buffer_size=5000)
    seen=set(); counts={'train':0,'val':0}; docs=duplicates=0; started=time.time(); batch=[]
    train=(root/'train_tokens.uint16').open('wb'); val=(root/'val_tokens.uint16').open('wb')
    def consume(texts):
        nonlocal docs,duplicates
        unique=[]
        for text in texts:
            digest=hashlib.sha256(text.encode()).digest()
            if digest in seen: duplicates+=1; continue
            seen.add(digest); unique.append(text)
        for ids in tok(unique,add_special_tokens=False)['input_ids'] if unique else []:
            docs+=1; ids=ids+[tok.eos_token_id]
            split='val' if counts['val']<1_000_000 else 'train'
            # A document is NEVER split across holdout and training.
            if split=='train': ids=ids[:max(0,target-counts['train'])]
            np.asarray(ids,dtype=np.uint16).tofile(val if split=='val' else train)
            counts[split]+=len(ids)
    try:
        for row in stream:
            text=row.get('text','').strip()
            if len(text)<100: continue
            batch.append(text)
            if len(batch)<256: continue
            consume(batch); batch=[]
            if docs//1024!=(docs-256)//1024:
                progress={'documents':docs,'train_tokens':counts['train'],'val_tokens':counts['val'],'target_train_tokens':target,'target_val_tokens':1_000_000,'tokens_per_second':sum(counts.values())/(time.time()-started),'duplicates_rejected':duplicates}
                print('PREPARE',json.dumps(progress),flush=True); base.emit_live('prepare',progress)
            if counts['train']>=target: break
    finally: train.close(); val.close()
    if counts['train']<target: raise RuntimeError('source exhausted before target')
    manifest={'source':'HuggingFaceFW/fineweb-edu','config':'sample-10BT','revision':REVISION,'seed':20260904,'tokenizer':'SmolLM2-135M','vocab':len(tok),'counts':counts,'documents':docs,'duplicates_rejected':duplicates,'holdout_document_disjoint':True,'sha256':{p.name:base.sha256_file(p) for p in (root/'train_tokens.uint16',root/'val_tokens.uint16')}}
    if args.verify_prefix_manifest:
        old=json.loads(Path(args.verify_prefix_manifest).read_text())
        if old['revision']!=REVISION or old['seed']!=20260904: raise RuntimeError('data provenance mismatch')
        prefix=prefix_sha256(root/'train_tokens.uint16',old['counts']['train']*2)
        if prefix!=old['sha256']['train_tokens.uint16']: raise RuntimeError('extended training prefix changed')
        if manifest['sha256']['val_tokens.uint16']!=old['sha256']['val_tokens.uint16']: raise RuntimeError('validation stream changed')
        manifest.update(prefix_verified=True,verified_prefix_tokens=old['counts']['train'],verified_prefix_sha256=prefix)
    base.json_dump(root/'data_manifest.json',manifest)
    base.emit_live('phase',{'phase':'data_prepared','data_integrity':'sha256_verified_document_disjoint'})
    print('PREPARED',json.dumps(manifest),flush=True)

if __name__=='__main__': main()
