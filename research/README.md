# Research artifacts

This directory contains synthetic, non-private artifacts from the prototype. It intentionally excludes user documents, model caches, vector databases, and raw generations that may reproduce private text.

## Included

- `results/architecture-comparison-summary.json`: exact aggregate metrics from ten distinct question categories.
- `results/architecture-comparison-cases.csv`: per-case metrics for all three retrieval modes.
- `results/reliability-observation.json`: the bounded reliability run and its one observed failure.
- `results/open-package-scaling-smoke-100.json`: a 100-node smoke test proving the public benchmark entry point runs; it is not a performance conclusion.
- `SCALING_STUDY.md`: a proposed protocol for measuring the curves we could not afford to establish locally.

The original architecture experiment used a local Qwen3 14B generation model and BGE-M3 embeddings on one Windows consumer workstation (AMD Ryzen 7 9800X3D, 64 GB system memory). It was designed to find implementation failures and large trade-offs, not to establish statistically general claims.
