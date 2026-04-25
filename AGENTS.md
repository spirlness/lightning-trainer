# AGENTS.md

This repository keeps one canonical PyTorch Lightning Tiny-ImageNet trainer.
Do not reintroduce the old `tiny_imagenet_trainer` package, old benchmark
scripts, or tests that import it.

## Commands

```bash
uv sync
uv run ruff check .
uv run --extra dev python -m pytest -q
```

Useful CPU smoke command:

```bash
python -m lightning_trainer.train \
  --data-dir data/tiny_imagenet_local \
  --max-epochs 0 \
  --num-workers 0 \
  --batch-size 2 \
  --no-pretrained
```

Best current GPU benchmark command on this machine:

```bash
uv run --extra dev python scripts/benchmark_lightning_throughput.py \
  --data-dir data/tiny_imagenet_local \
  --cache-dir data/tiny_imagenet_cache_128 \
  --batch-size 128 \
  --image-size 128 \
  --num-workers 4 \
  --full-epoch \
  --no-pretrained
```

## Project Shape

- `lightning_trainer/data.py`: Lightning `DataModule`, ImageFolder loading,
  and memory-mapped tensor cache loading.
- `lightning_trainer/model.py`: ConvNeXt-Tiny Lightning classifier.
- `lightning_trainer/train.py`: training CLI entry point.
- `download_data.py`: Tiny-ImageNet download and ImageFolder conversion helper.
- `scripts/prepare_tensor_cache.py`: converts ImageFolder JPEG data to a
  memory-mapped uint8 tensor cache.
- `scripts/benchmark_lightning_throughput.py`: GPU throughput benchmark.
- `tests/test_lightning_trainer.py`: canonical tests.

## Current Design

- Model is fixed to ConvNeXt-Tiny.
- Training uses fixed resize, not random cropping.
- Train DataLoader uses `drop_last=True` to avoid dynamic last-batch shapes
  with `torch.compile`.
- Tensor cache uses fixed resized CHW uint8 tensors and trades disk space for
  faster input throughput.
- `--gradient-checkpointing` only takes effect for models exposing
  `gradient_checkpointing_enable`; current torchvision ConvNeXt-Tiny does not,
  so the flag warns and continues.
- Use `uv run --extra dev python -m pytest -q`, not bare external `pytest`, so
  tests run inside the project environment.
