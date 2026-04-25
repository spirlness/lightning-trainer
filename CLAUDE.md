# CLAUDE.md

Use `AGENTS.md` as the authoritative project guide.

This repository now has one canonical implementation:

- Package: `lightning_trainer`
- Model: ConvNeXt-Tiny only
- Data path: ImageFolder or preprocessed tensor cache
- Tests: `tests/test_lightning_trainer.py`
- Utility scripts: `scripts/prepare_tensor_cache.py` and
  `scripts/benchmark_lightning_throughput.py`

Validate with:

```bash
uv run ruff check .
uv run --extra dev python -m pytest -q
```
