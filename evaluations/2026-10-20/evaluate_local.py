"""Compare saved local model families using the existing diagnostic suite."""
import gc
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'flashskip'))
sys.path.insert(0, str(ROOT / 'mor10m'))
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from flashskip_100m import FlashSkipConfig, FlashSkipLM
from flashskip_10m_recursive import RecursiveFlashSkipConfig, RecursiveFlashSkipLM
from model import Config, MoR

spec = importlib.util.spec_from_file_location('diagnostics', ROOT / 'flashskip/benchmarks/recursive_100m_to_1100m/benchmark_recursive_checkpoints.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device('cpu')  # Same FP32 backend for every architecture.
    candidates = [
        ('flashskip10m_1500m', 'flashskip/flashskip10m_kaggle1b/models/checkpoints/checkpoint_step_045778_tokens_1500.05M', FlashSkipConfig, FlashSkipLM, 'flashskip/flashskip10m_kaggle1b/kernel_output/flashskip/models/tokenizer'),
        ('recursive10m_1100m', 'flashskip/flashskip10m_recursive_kaggle1b/models/checkpoints/checkpoint_step_033570_tokens_1100.02M', RecursiveFlashSkipConfig, RecursiveFlashSkipLM, 'flashskip/flashskip10m_recursive_kaggle1b/models/tokenizer'),
        ('mor10m_100m', 'mor10m/run/resume_dataset', Config, MoR, 'mor10m/run/source_dataset'),
        ('flashskip100m_200m', 'flashskip/models/final', FlashSkipConfig, FlashSkipLM, 'flashskip/models/mac_test/tokenizer'),
    ]
    results = []
    for name, relative, config_cls, model_cls, tok_path in candidates:
        path = ROOT / relative
        cfg = config_cls.from_dict(json.loads((path / 'config.json').read_text()))
        model = model_cls(cfg).to(device).eval()
        weights = path / 'model.safetensors'
        model.load_state_dict(load_file(str(weights)), strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / tok_path), local_files_only=True)
        start = time.perf_counter()
        result = {'name': name, 'checkpoint': str(path), 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest(), 'parameters': sum(p.numel() for p in model.parameters()), 'device': str(device), 'dtype': 'float32', 'recorded_metrics': json.loads((path / 'metrics.json').read_text())}
        result['heldout'] = bench.eval_heldout(model, tokenizer, device)
        result['grammar'] = bench.eval_grammar(model, tokenizer, device)
        for label, items in [('cloze', bench.CLOZE), ('facts', bench.FACTS), ('commonsense', bench.COMMONSENSE), ('arithmetic', bench.ARITHMETIC_LOGIC)]:
            result[label] = bench.eval_mc_category(model, tokenizer, device, items)
        result['generation'] = bench.eval_generation(model, tokenizer, device, 24)
        result['seconds'] = time.perf_counter() - start
        results.append(result)
        (Path(__file__).parent / 'results.json').write_text(json.dumps({'date': '2026-10-20', 'suite': 'existing public handcrafted diagnostics; not a blind test', 'results': results}, indent=2))
        print(name, result['heldout']['perplexity'], {k: result[k]['accuracy'] for k in ['grammar','cloze','facts','commonsense','arithmetic']}, flush=True)
        del model
        gc.collect()

if __name__ == '__main__':
    main()
