# CLAUDE.md

Use `AGENTS.md` as the authoritative project guide.

This repository now has one canonical implementation:

- Package: `lightning_trainer`
- Model: ConvNeXt-Tiny only (frozen dataclass config)
- Data path: ImageFolder or preprocessed tensor cache
- Training config: `batch_size=128`, `num_workers=4`, `image_size=128`,
  tensor cache enabled, `torch.compile`, fused AdamW, `cudnn.benchmark`,
  TF32 matmul, AMP mixed precision, `channels_last`
- Tests: `tests/test_lightning_trainer.py` (16 tests)
- Utility scripts:
  - `scripts/download_data.py` — dataset download
  - `scripts/prepare_tensor_cache.py` — JPEG to tensor cache
  - `scripts/benchmark_lightning_throughput.py` — GPU throughput benchmark
  - `scripts/profile_run.py` — PyTorch profiler
- CLI defaults use the recommended settings. Use `--cache-dir ""`,
  `--no-compile`, or `--no-pretrained` for fallback runs.
- Keep the training CLI small. Lightning owns logger, checkpoint, LR monitor,
  precision, optimizer stepping, and scheduler stepping.

Validate with:

```bash
uv run ruff check .
uv run --extra dev python -m pytest -q
```
