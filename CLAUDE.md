# CLAUDE.md

Use `AGENTS.md` as the authoritative project guide.

This repository now has one canonical implementation:

- Package: `lightning_trainer`
- Model: ConvNeXt-Tiny only
- Data path: ImageFolder or preprocessed tensor cache
- Tests: `tests/test_lightning_trainer.py`
- Utility scripts: `scripts/prepare_tensor_cache.py` and
  `scripts/benchmark_lightning_throughput.py`
- Current recommended GPU throughput settings:
  `batch_size=128`, `num_workers=4`, `image_size=128`, tensor cache enabled.
- These settings are now the training CLI defaults. Use `--cache-dir ""`,
  or `--no-compile` for fallback runs.
- Keep the training CLI small. Lightning owns logger, checkpoint, LR monitor,
  precision, optimizer stepping, and scheduler stepping.

Validate with:

```bash
uv run ruff check .
uv run --extra dev python -m pytest -q
```
