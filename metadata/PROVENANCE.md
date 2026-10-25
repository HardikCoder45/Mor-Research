# Provenance and Evidence Policy

Research snapshot date: 2026-08-30

## Source hierarchy
1. Primary paper PDF / official arXiv abstract page / conference proceedings.
2. Official model or project repository / technical report.
3. Author-maintained project page.
4. Secondary sources only for discovery; important claims must be checked against a primary source.

## Per-paper record
Each curated paper should record, where available:
- title, authors, year, venue
- arXiv ID / DOI
- canonical abstract and PDF URLs
- local PDF path
- research tags
- why it was selected
- architecture/method contribution
- empirical evidence and benchmark context
- limitations / confounders
- actionable implications for <500M on-device LMs

## Claim labels
- **Evidence:** directly supported by a cited experiment, table, figure, or explicit paper statement.
- **Synthesis:** conclusion produced by combining multiple sources.
- **Hypothesis:** plausible but requires a controlled experiment in this project.

## Constraint semantics
“Under 500M” means **total learned deployable model parameters < 500,000,000**, not merely active MoE parameters. External teacher parameters are not part of the deployed student but must be identified separately.

## On-device evaluation dimensions
Quality alone is insufficient. Track total model bytes, working-set/peak RAM, KV-cache bytes/token, prompt processing throughput, decode tokens/s, TTFT, energy/power when measurable, thermal behavior, supported operator/kernel stack, and quantization sensitivity.

## Reproducibility
Record calculation assumptions, tokenizer/vocabulary size, tied-vs-untied embeddings, context length, precision, KV dtype, and architecture dimensions whenever reporting parameter or memory figures.
