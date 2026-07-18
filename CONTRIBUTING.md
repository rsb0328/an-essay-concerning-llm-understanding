# Contributing

Contributions are welcome, including negative benchmark results and implementations that replace the default components.

## Development setup

```bash
python -m venv .venv
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

## Principles

- Keep the core independent of any particular generation or embedding model.
- Preserve canonical text and structured procedures outside vector indexes.
- Treat semantic similarity as candidate generation, not logical judgment.
- Do not silently discard contradictory or lower-confidence sources.
- Validate all model-proposed IDs and relations before persistence.
- Keep shortcut activation, reliability, fallback, and retirement under application control.
- Add tests for new providers and traversal behavior.

## Benchmark contributions

Follow `research/SCALING_STUDY.md`. Include raw machine-readable results, commit hash, hardware, software versions, model identifiers, quantization, vector-store configuration, graph density, warm-up procedure, and failures. Do not submit proprietary datasets or model outputs you are not permitted to publish.

## Pull requests

Explain what changed, why it belongs in the model-independent core or an adapter, how it was validated, and whether it changes stored-data compatibility. Small, reviewable changes are preferred.

