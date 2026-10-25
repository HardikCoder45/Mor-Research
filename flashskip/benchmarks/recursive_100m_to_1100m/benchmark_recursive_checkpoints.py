#!/usr/bin/env python3
"""Deterministic diagnostic benchmark for FlashSkip recursive checkpoints.

This is intentionally compact and local. It compares checkpoints under one
fixed tokenizer and one fixed evaluation suite. It is a diagnostic regression
benchmark, not a replacement for standard external benchmark suites.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "flashskip10m_recursive_kaggle1b"
SOURCE_DIR = RUN_ROOT / "models" / "source"
TOKENIZER_DIR = RUN_ROOT / "models" / "tokenizer"
sys.path.insert(0, str(SOURCE_DIR))

from flashskip_10m_recursive import RecursiveFlashSkipConfig, RecursiveFlashSkipLM  # noqa: E402


HELDOUT_TEXTS = [
    "At sunrise, the baker opened the wooden shutters and placed fresh bread on the front counter. A cyclist stopped outside, checked the map on her phone, and asked for directions to the river path.",
    "The laboratory team repeated the measurement three times before recording the result. When the values agreed within a small margin, they cleaned the equipment and prepared the next sample.",
    "Mira missed the early bus, so she walked to the station instead. The sky looked gray, but the rain did not begin until she was safely under the platform roof.",
    "A healthy garden depends on more than sunlight. Soil must hold enough water for roots while still allowing air to move through the spaces between particles.",
    "The museum closed at six in the evening. Ten minutes before closing, a guard politely reminded the remaining visitors that the main doors would soon be locked.",
    "During the meeting, the engineer drew a simple diagram of the bridge on the whiteboard. She explained how the load would move from the deck through the supports and into the ground.",
    "Ravi poured hot tea into two cups and left one beside the window to cool. After several minutes, that cup was noticeably cooler than the one he had just filled.",
    "A small software team fixed the most serious bug before releasing the update. They also added a test so that the same failure would be detected automatically in the future.",
    "The old library was quiet except for the soft sound of pages turning. Near the back wall, a student compared notes from two books and highlighted the facts that appeared in both.",
    "When the power returned, the refrigerator started humming again and the hallway lights came on. Everyone checked the clocks because several of them were flashing the wrong time.",
]

GRAMMAR_PAIRS = [
    ("The cat is sleeping on the sofa.", "The cat are sleeping on the sofa."),
    ("Those apples are ripe.", "Those apples is ripe."),
    ("She has finished her homework.", "She have finished her homework."),
    ("He does not like cold coffee.", "He do not like cold coffee."),
    ("I saw an elephant at the zoo.", "I saw a elephant at the zoo."),
    ("There are three books on the desk.", "There is three books on the desk."),
    ("Each student has a pencil.", "Each student have a pencil."),
    ("The children were playing outside.", "The children was playing outside."),
    ("If it rains, we will stay inside.", "If it rain, we will stay inside."),
    ("Neither answer is correct.", "Neither answer are correct."),
    ("She went to the store yesterday.", "She go to the store yesterday."),
    ("The keys are on the table.", "The keys is on the table."),
    ("My brother and I are ready.", "My brother and I is ready."),
    ("This book belongs to me.", "This book belong to me."),
    ("The woman who lives next door is a doctor.", "The woman who lives next door are a doctor."),
    ("We have been waiting for an hour.", "We has been waiting for an hour."),
    ("The dog chased its tail.", "The dog chased it's tail."),
    ("She speaks English fluently.", "She speaks English fluent."),
    ("The red car moves quickly.", "The red car move quickly."),
    ("These flowers smell wonderful.", "These flowers smells wonderful."),
]

CLOZE = [
    ("A physician usually works in a", [" hospital", " bakery", " forest", " garage"], 0),
    ("A person uses a toothbrush to clean their", [" teeth", " shoes", " windows", " books"], 0),
    ("The opposite of cold is", [" hot", " narrow", " quiet", " empty"], 0),
    ("A bird normally builds a", [" nest", " engine", " ladder", " bottle"], 0),
    ("You wear gloves on your", [" hands", " ears", " knees", " teeth"], 0),
    ("A chef uses an oven to", [" cook", " swim", " write", " drive"], 0),
    ("A library is a place where people borrow", [" books", " cars", " hammers", " shoes"], 0),
    ("A clock is used to tell", [" time", " weight", " color", " distance"], 0),
    ("A farmer may grow wheat in a", [" field", " bathtub", " suitcase", " hallway"], 0),
    ("People usually sleep in a", [" bed", " sink", " bicycle", " oven"], 0),
    ("A pencil is commonly used for", [" writing", " boiling", " driving", " swimming"], 0),
    ("The color of fresh grass is usually", [" green", " purple", " silver", " black"], 0),
]

FACTS = [
    ("The capital of France is", [" Paris", " London", " Rome", " Madrid"], 0),
    ("The planet known as the Red Planet is", [" Mars", " Venus", " Jupiter", " Mercury"], 0),
    ("Water freezes at zero degrees", [" Celsius", " Fahrenheit", " Kelvin", " meters"], 0),
    ("The largest planet in the Solar System is", [" Jupiter", " Earth", " Mars", " Venus"], 0),
    ("The primary language of Brazil is", [" Portuguese", " Spanish", " French", " German"], 0),
    ("H2O is the chemical formula for", [" water", " oxygen", " salt", " iron"], 0),
    ("There are seven days in a", [" week", " year", " century", " minute"], 0),
    ("The Earth travels around the", [" Sun", " Moon", " Mars", " ocean"], 0),
    ("The author of Hamlet was William", [" Shakespeare", " Darwin", " Newton", " Edison"], 0),
    ("The ocean between Africa and Australia is the", [" Indian", " Arctic", " Atlantic", " Southern"], 0),
    ("A whale is a", [" mammal", " reptile", " insect", " plant"], 0),
    ("The gas humans need to breathe is", [" oxygen", " helium", " neon", " argon"], 0),
]

COMMONSENSE = [
    ("If a glass falls onto a hard floor, it may", [" break", " grow", " sing", " freeze"], 0),
    ("When people are hungry, they usually want to", [" eat", " sleep", " paint", " whisper"], 0),
    ("If a room is dark, you can turn on a", [" light", " pillow", " spoon", " shoe"], 0),
    ("An umbrella is useful when it is", [" raining", " sleeping", " cooking", " reading"], 0),
    ("Ice left in warm sunlight will usually", [" melt", " grow", " sharpen", " multiply"], 0),
    ("A key can be used to open a", [" lock", " cloud", " sandwich", " river"], 0),
    ("A tired person often needs", [" sleep", " paint", " sand", " smoke"], 0),
    ("A sharp knife can be used to", [" cut", " listen", " count", " float"], 0),
    ("A dry plant often needs", [" water", " metal", " paper", " glass"], 0),
    ("If you drop a ball, gravity makes it", [" fall", " glow", " read", " disappear"], 0),
    ("Before crossing a busy road, a careful person should", [" look", " sleep", " shout", " dance"], 0),
    ("If food is too hot to eat, you can wait for it to", [" cool", " grow", " harden", " vanish"], 0),
]

ARITHMETIC_LOGIC = [
    ("Two plus two equals", [" four", " five", " six", " seven"], 0),
    ("Five minus two equals", [" three", " two", " four", " eight"], 0),
    ("Three times three equals", [" nine", " six", " twelve", " five"], 0),
    ("Half of ten is", [" five", " ten", " two", " twenty"], 0),
    ("The next number after 2, 4, 6 is", [" 8", " 7", " 9", " 10"], 0),
    ("One dozen means", [" twelve", " ten", " twenty", " five"], 0),
    ("A triangle has", [" three sides", " four sides", " five sides", " six sides"], 0),
    ("Ten is greater than seven, so the statement is", [" true", " false", " unknown", " impossible"], 0),
    ("If all cats are animals and Mia is a cat, then Mia is an", [" animal", " planet", " number", " building"], 0),
    ("The opposite truth value of true is", [" false", " large", " equal", " round"], 0),
]

GENERATION_PROMPTS = [
    "The capital of France is",
    "Once upon a time, a small robot",
    "To make a cup of tea,",
    "Scientists discovered that the new material",
    "Alice went to the market because",
    "The rain stopped, so the children",
]


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_paths() -> list[Path]:
    paths = [ROOT / "flashskip10m_recursive100m/models/checkpoints/checkpoint_step_003052_tokens_0100.01M"]
    recovered_200 = RUN_ROOT / "recovered_200m/result/checkpoints/checkpoint_step_006104_tokens_0200.02M"
    temp_root = RUN_ROOT / "kaggle_output_latest.tmp/result/checkpoints"
    paths.append(recovered_200)
    paths.extend(
        p for p in sorted(temp_root.glob("checkpoint_step_*"))
        if "0200.02M" not in p.name and "1100.02M" not in p.name and (p / "model.safetensors").exists()
    )
    paths.append(RUN_ROOT / "models/checkpoints/checkpoint_step_033570_tokens_1100.02M")
    paths.sort(key=lambda p: json.loads((p / "checkpoint_manifest.json").read_text())["tokens_seen"])
    if len(paths) != 11:
        raise RuntimeError(f"Expected 11 checkpoints, found {len(paths)}: {[p.name for p in paths]}")
    return paths


def load_model(checkpoint: Path, device: torch.device) -> tuple[RecursiveFlashSkipLM, dict[str, Any], dict[str, Any]]:
    cfg = RecursiveFlashSkipConfig.from_dict(json.loads((checkpoint / "config.json").read_text()))
    model = RecursiveFlashSkipLM(cfg)
    state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    if device.type != "cpu":
        model = model.to(dtype=torch.float16)
    model = model.to(device).eval()
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    metrics = json.loads((checkpoint / "metrics.json").read_text())
    return model, manifest, metrics


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


@torch.inference_mode()
def full_text_mean_logprob(model, tokenizer, text: str, device: torch.device) -> tuple[float, int]:
    ids = encode(tokenizer, text)
    if len(ids) < 2:
        return float("nan"), 0
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x)[:, :-1, :].float()
    targets = x[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
    n = targets.numel()
    return -float(nll.item()) / n, n


@torch.inference_mode()
def conditional_mean_logprob(model, tokenizer, prompt: str, option: str, device: torch.device) -> tuple[float, int, int]:
    prefix = encode(tokenizer, prompt)
    full = encode(tokenizer, prompt + option)
    common = 0
    for a, b in zip(prefix, full):
        if a != b:
            break
        common += 1
    # The first scorable target token has index 1. If tokenization changes at
    # the boundary, include that changed suffix in the candidate score.
    start_target = max(1, common)
    if start_target >= len(full):
        return float("-inf"), 0, common
    x = torch.tensor([full], dtype=torch.long, device=device)
    logits = model(x).float()[0]
    target_ids = torch.tensor(full[start_target:], dtype=torch.long, device=device)
    pred = logits[start_target - 1 : len(full) - 1]
    logp = F.log_softmax(pred, dim=-1).gather(1, target_ids[:, None]).squeeze(1)
    return float(logp.mean().item()), int(logp.numel()), common


def eval_heldout(model, tokenizer, device: torch.device) -> dict[str, Any]:
    total_nll = 0.0
    total_tokens = 0
    per_text = []
    for text in HELDOUT_TEXTS:
        mean_lp, n = full_text_mean_logprob(model, tokenizer, text, device)
        total_nll += -mean_lp * n
        total_tokens += n
        per_text.append({"tokens": n, "nll": -mean_lp})
    nll = total_nll / total_tokens
    return {"nll": nll, "perplexity": math.exp(min(nll, 50)), "tokens": total_tokens, "per_text": per_text}


def eval_grammar(model, tokenizer, device: torch.device) -> dict[str, Any]:
    correct = 0
    margins = []
    details = []
    for good, bad in GRAMMAR_PAIRS:
        good_lp, _ = full_text_mean_logprob(model, tokenizer, good, device)
        bad_lp, _ = full_text_mean_logprob(model, tokenizer, bad, device)
        margin = good_lp - bad_lp
        correct += int(margin > 0)
        margins.append(margin)
        details.append({"good": good, "bad": bad, "good_mean_logp": good_lp, "bad_mean_logp": bad_lp, "margin": margin})
    return {"accuracy": correct / len(GRAMMAR_PAIRS), "correct": correct, "total": len(GRAMMAR_PAIRS), "mean_margin": statistics.mean(margins), "details": details}


def eval_mc_category(model, tokenizer, device: torch.device, items) -> dict[str, Any]:
    correct = 0
    margins = []
    details = []
    boundary_changes = 0
    for prompt, choices, answer in items:
        scores = []
        counts = []
        commons = []
        prefix_len = len(encode(tokenizer, prompt))
        for option in choices:
            score, n, common = conditional_mean_logprob(model, tokenizer, prompt, option, device)
            scores.append(score)
            counts.append(n)
            commons.append(common)
            boundary_changes += int(common < prefix_len)
        pred = max(range(len(scores)), key=scores.__getitem__)
        wrong_best = max(scores[i] for i in range(len(scores)) if i != answer)
        margin = scores[answer] - wrong_best
        correct += int(pred == answer)
        margins.append(margin)
        details.append({"prompt": prompt, "choices": choices, "answer": answer, "prediction": pred, "scores": scores, "candidate_token_counts": counts, "common_prefix_tokens": commons, "margin": margin})
    return {"accuracy": correct / len(items), "correct": correct, "total": len(items), "mean_margin": statistics.mean(margins), "boundary_changes": boundary_changes, "details": details}


def repeated_ngram_fraction(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    ids = encode(tokenizer, prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated: list[int] = []
    top1_probs: list[float] = []
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        top1_probs.append(float(probs[0, token_id].item()))
        x = torch.cat((x, next_token), dim=1)
        if token_id == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - started
    unique_ratio = len(set(generated)) / max(1, len(generated))
    adjacent_repeat = sum(a == b for a, b in zip(generated, generated[1:])) / max(1, len(generated) - 1)
    return {
        "prompt": prompt,
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "tokens": generated,
        "generated_tokens": len(generated),
        "tokens_per_second": len(generated) / max(elapsed, 1e-9),
        "unique_token_ratio": unique_ratio,
        "adjacent_repeat_rate": adjacent_repeat,
        "repeat_bigram_fraction": repeated_ngram_fraction(generated, 2),
        "repeat_trigram_fraction": repeated_ngram_fraction(generated, 3),
        "mean_top1_probability": statistics.mean(top1_probs) if top1_probs else float("nan"),
    }


def eval_generation(model, tokenizer, device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    samples = [greedy_generate(model, tokenizer, p, device, max_new_tokens) for p in GENERATION_PROMPTS]
    return {
        "mean_unique_token_ratio": statistics.mean(s["unique_token_ratio"] for s in samples),
        "mean_adjacent_repeat_rate": statistics.mean(s["adjacent_repeat_rate"] for s in samples),
        "mean_repeat_bigram_fraction": statistics.mean(s["repeat_bigram_fraction"] for s in samples),
        "mean_repeat_trigram_fraction": statistics.mean(s["repeat_trigram_fraction"] for s in samples),
        "mean_tokens_per_second": statistics.mean(s["tokens_per_second"] for s in samples),
        "mean_top1_probability": statistics.mean(s["mean_top1_probability"] for s in samples),
        "samples": samples,
    }


def flat_row(result: dict[str, Any]) -> dict[str, Any]:
    cats = ["grammar", "cloze", "facts", "commonsense", "arithmetic_logic"]
    behavioral_accuracy = statistics.mean(result[c]["accuracy"] for c in cats)
    behavioral_margin = statistics.mean(result[c]["mean_margin"] for c in cats)
    return {
        "tokens_m": result["tokens_seen"] / 1e6,
        "tokens_seen": result["tokens_seen"],
        "step": result["step"],
        "logged_val_loss": result["logged_val_loss"],
        "logged_val_ppl": result["logged_val_ppl"],
        "heldout_nll": result["heldout"]["nll"],
        "heldout_ppl": result["heldout"]["perplexity"],
        "grammar_accuracy": result["grammar"]["accuracy"],
        "cloze_accuracy": result["cloze"]["accuracy"],
        "facts_accuracy": result["facts"]["accuracy"],
        "commonsense_accuracy": result["commonsense"]["accuracy"],
        "arithmetic_logic_accuracy": result["arithmetic_logic"]["accuracy"],
        "behavioral_accuracy": behavioral_accuracy,
        "behavioral_mean_margin": behavioral_margin,
        "generation_unique_ratio": result["generation"]["mean_unique_token_ratio"],
        "generation_repeat_bigram": result["generation"]["mean_repeat_bigram_fraction"],
        "generation_repeat_trigram": result["generation"]["mean_repeat_trigram_fraction"],
        "generation_adjacent_repeat": result["generation"]["mean_adjacent_repeat_rate"],
        "generation_top1_probability": result["generation"]["mean_top1_probability"],
        "generation_tokens_per_second": result["generation"]["mean_tokens_per_second"],
        "checkpoint": result["checkpoint"],
    }


def rank_rows(rows: list[dict[str, Any]]) -> None:
    def ranks(key: str, reverse: bool) -> dict[int, float]:
        order = sorted(range(len(rows)), key=lambda i: rows[i][key], reverse=reverse)
        out: dict[int, float] = {}
        pos = 0
        while pos < len(order):
            end = pos + 1
            value = rows[order[pos]][key]
            while end < len(order) and math.isclose(rows[order[end]][key], value, rel_tol=0.0, abs_tol=1e-12):
                end += 1
            # Standard competition ranking with average rank for exact ties.
            average_rank = ((pos + 1) + end) / 2.0
            for j in range(pos, end):
                out[order[j]] = average_rank
            pos = end
        return out
    metric_ranks = {
        "heldout": ranks("heldout_nll", False),
        "behavior": ranks("behavioral_accuracy", True),
        "margin": ranks("behavioral_mean_margin", True),
        "generation": ranks("generation_repeat_trigram", False),
    }
    for i, row in enumerate(rows):
        row["heldout_rank"] = metric_ranks["heldout"][i]
        row["behavior_rank"] = metric_ranks["behavior"][i]
        row["margin_rank"] = metric_ranks["margin"][i]
        row["generation_rank"] = metric_ranks["generation"][i]
        row["mean_core_rank"] = statistics.mean(r[i] for r in metric_ranks.values())


def markdown_report(rows: list[dict[str, Any]], results: list[dict[str, Any]]) -> str:
    best_rank_value = min(r["mean_core_rank"] for r in rows)
    best_rank_rows = [r for r in rows if math.isclose(r["mean_core_rank"], best_rank_value, rel_tol=0.0, abs_tol=1e-12)]
    best_rank_label = " and ".join(f"{r['tokens_m']:.0f}M" for r in best_rank_rows)
    best_ppl = min(rows, key=lambda r: r["heldout_ppl"])
    best_behavior_value = max(r["behavioral_accuracy"] for r in rows)
    best_behavior_rows = [r for r in rows if math.isclose(r["behavioral_accuracy"], best_behavior_value, rel_tol=0.0, abs_tol=1e-12)]
    best_behavior_label = ", ".join(f"{r['tokens_m']:.0f}M" for r in best_behavior_rows)
    lines = [
        "# FlashSkip Recursive Checkpoint Benchmark: 100M–1100M",
        "",
        "This is a deterministic local diagnostic suite using one shared tokenizer and the same recursive architecture at every checkpoint. It is intended for checkpoint comparison, not as a substitute for standardized external benchmarks. Core ranks use average ranks for exact ties.",
        "",
        f"**Best mean core rank:** {best_rank_label} (mean rank {best_rank_value:.2f})",
        f"**Best handcrafted held-out perplexity:** {best_ppl['tokens_m']:.0f}M tokens ({best_ppl['heldout_ppl']:.2f})",
        f"**Best aggregate behavioral accuracy:** {best_behavior_label} ({best_behavior_value*100:.1f}%)",
        "",
        "| Tokens | Logged val PPL | Held-out PPL | Grammar | Cloze | Facts | Common | Arith/Logic | Behavior avg | Repeat 3-gram | Core rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tokens_m']:.0f}M | {r['logged_val_ppl']:.2f} | {r['heldout_ppl']:.2f} | "
            f"{r['grammar_accuracy']*100:.0f}% | {r['cloze_accuracy']*100:.0f}% | {r['facts_accuracy']*100:.0f}% | "
            f"{r['commonsense_accuracy']*100:.0f}% | {r['arithmetic_logic_accuracy']*100:.0f}% | "
            f"{r['behavioral_accuracy']*100:.1f}% | {r['generation_repeat_trigram']*100:.1f}% | {r['mean_core_rank']:.2f} |"
        )
    lines += ["", "## Greedy generation samples", ""]
    for result in results:
        lines.append(f"### {result['tokens_seen']/1e6:.0f}M tokens")
        for sample in result["generation"]["samples"]:
            clean = sample["text"].replace("\n", " ")
            lines.append(f"- **{sample['prompt']}** → {clean}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N checkpoints for smoke testing")
    args = parser.parse_args()

    torch.manual_seed(12345)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True, local_files_only=True)
    checkpoints = checkpoint_paths()
    if args.limit:
        checkpoints = checkpoints[: args.limit]
    print(json.dumps({"event": "start", "device": str(device), "checkpoints": len(checkpoints)}, sort_keys=True), flush=True)

    results: list[dict[str, Any]] = []
    for index, checkpoint in enumerate(checkpoints, 1):
        started = time.perf_counter()
        model, manifest, logged = load_model(checkpoint, device)
        val = logged.get("val", {})
        result: dict[str, Any] = {
            "checkpoint": str(checkpoint),
            "checkpoint_name": checkpoint.name,
            "step": int(manifest["step"]),
            "tokens_seen": int(manifest["tokens_seen"]),
            "logged_val_loss": float(logged.get("val_loss", val.get("loss"))),
            "logged_val_ppl": float(logged.get("val_perplexity", val.get("perplexity"))),
            "heldout": eval_heldout(model, tokenizer, device),
            "grammar": eval_grammar(model, tokenizer, device),
            "cloze": eval_mc_category(model, tokenizer, device, CLOZE),
            "facts": eval_mc_category(model, tokenizer, device, FACTS),
            "commonsense": eval_mc_category(model, tokenizer, device, COMMONSENSE),
            "arithmetic_logic": eval_mc_category(model, tokenizer, device, ARITHMETIC_LOGIC),
            "generation": eval_generation(model, tokenizer, device, args.max_new_tokens),
        }
        result["elapsed_seconds"] = time.perf_counter() - started
        results.append(result)
        row = flat_row(result)
        print(json.dumps({"event": "checkpoint_done", "index": index, **{k: row[k] for k in ["tokens_m", "heldout_ppl", "behavioral_accuracy", "generation_repeat_trigram"]}, "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    rows = [flat_row(r) for r in results]
    rank_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark_results.json").write_text(json.dumps({"device": str(device), "suite": {"heldout_texts": len(HELDOUT_TEXTS), "grammar_pairs": len(GRAMMAR_PAIRS), "cloze": len(CLOZE), "facts": len(FACTS), "commonsense": len(COMMONSENSE), "arithmetic_logic": len(ARITHMETIC_LOGIC), "generation_prompts": len(GENERATION_PROMPTS), "max_new_tokens": args.max_new_tokens}, "results": results, "summary_rows": rows}, indent=2, ensure_ascii=False))
    with (args.output_dir / "benchmark_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "benchmark_report.md").write_text(markdown_report(rows, results))
    best = min(rows, key=lambda r: r["mean_core_rank"])
    print(json.dumps({"event": "done", "best_tokens_m": best["tokens_m"], "best_mean_core_rank": best["mean_core_rank"], "output_dir": str(args.output_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
