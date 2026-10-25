# Sub-500M On-Device Language Model Research Knowledge Base

Research date: 2026-08-30

## Goal
Build an evidence-backed architecture and research program for an exceptionally capable on-device language model with **fewer than 500M total parameters**, anchored on arXiv:2602.09080 and arXiv:2507.10524 and expanded through related primary literature.

## Scope
This knowledge base separates:
- **Observed evidence** from papers and released models.
- **Engineering inference** derived from multiple sources.
- **Research hypotheses** that require ablation or device benchmarking.

The deployable model target is strictly <500M total parameters. Larger models may appear only as teachers, upper-bound baselines, or architectural references.

## Directory layout
- `papers/seed/` — the two seed PDFs.
- `papers/related/` — curated related primary papers.
- `notes/seed/` — deep seed-paper analyses.
- `notes/related/` — structured per-paper notes.
- `synthesis/` — subsystem and final architecture syntheses.
- `tables/` — model/paper matrices and quantitative comparisons.
- `bibliography/` — references and citation exports.
- `metadata/` — machine-readable paper manifests and provenance.
- `utilities/` — parameter/memory calculation scripts and helper material.
- `qa/` — final validation and artifact inventory.

## Research principles
1. Prefer arXiv, conference proceedings, official repositories, and model technical reports as primary evidence.
2. Record stable identifiers (arXiv IDs / DOI / venue) and source URLs.
3. Do not equate active parameters with total parameters.
4. Evaluate quality jointly with RAM, bandwidth, KV-cache growth, latency, power, and device-runtime compatibility.
5. Treat claims of a universally “best” architecture as unproven until validated under a common training budget and device stack.
6. Every final design choice should map to evidence, a quantitative constraint, or a clearly marked hypothesis.

## Seed papers
- arXiv:2602.09080 — https://arxiv.org/abs/2602.09080
- arXiv:2507.10524 — https://arxiv.org/abs/2507.10524

See `INDEX.md` as the knowledge base grows.
# Mor-Research
