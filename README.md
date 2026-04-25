# Lightning Trainer

Minimal PyTorch Lightning trainer for Tiny-ImageNet classification on
ConvNeXt-Tiny.

## Project Layout

```text
lightning_trainer/
  data.py      # ImageFolder DataModule and tensor-cache Dataset
  model.py     # ConvNeXt-Tiny LightningModule
  train.py     # training CLI
scripts/
  prepare_tensor_cache.py          # JPEG ImageFolder -> uint8 tensor cache
  benchmark_lightning_throughput.py # GPU throughput benchmark
tests/
  test_lightning_trainer.py
download_data.py
```

The old `tiny_imagenet_trainer` package and old benchmark scripts are no longer
part of this project.

## Install

```bash
uv sync
```

## Data

```bash
# Download/convert Tiny-ImageNet through Hugging Face
python download_data.py --method huggingface

# Or create a small local subset from an existing full dataset
python download_data.py --method subset --num-classes 10 --images-per-class 100
```

Expected ImageFolder layout:

```text
data/tiny-imagenet-200/
  train/<class>/*.JPEG
  val/<class>/*.JPEG
```

## Tensor Cache

For the best current throughput, convert JPEG data to a memory-mapped uint8
tensor cache. This avoids per-epoch JPEG decode and PIL resize.

```bash
uv run --extra dev python scripts/prepare_tensor_cache.py \
  --data-dir data/tiny_imagenet_local \
  --cache-dir data/tiny_imagenet_cache_128 \
  --image-size 128 \
  --splits train val \
  --overwrite
```

The cache stores fixed resized CHW uint8 images in `images.bin`, plus
`labels.pt` and `manifest.json` for each split.

## Train

Basic ImageFolder training:

```bash
python -m lightning_trainer.train \
  --data-dir data/tiny-imagenet-200 \
  --batch-size 128 \
  --image-size 128 \
  --num-workers 4 \
  --compile \
  --fused-optimizer
```

Cached-data training:

```bash
python -m lightning_trainer.train \
  --data-dir data/tiny_imagenet_local \
  --cache-dir data/tiny_imagenet_cache_128 \
  --batch-size 128 \
  --image-size 128 \
  --num-workers 4 \
  --compile \
  --fused-optimizer \
  --no-pretrained
```

## Benchmark

Best current benchmark command on this machine:

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

Recent full-epoch result with tensor cache:

```text
batch_size=128, num_workers=4
throughput ~= 1082 samples/s
peak memory ~= 2.8 GB
elapsed ~= 142 s
```

## Design Notes

- Model is fixed to ConvNeXt-Tiny.
- Training uses fixed resize and horizontal flip; random crop is intentionally
  removed.
- Train DataLoader uses `drop_last=True` to avoid dynamic final-batch shapes
  with `torch.compile`.
- AMP, `channels_last`, fused AdamW, and `torch.compile` are supported.
- `--gradient-checkpointing` only has an effect for models exposing a native
  `gradient_checkpointing_enable` method. Current torchvision ConvNeXt-Tiny
  does not, so the flag warns and continues.

## Validate

```bash
uv run ruff check .
uv run --extra dev python -m pytest -q
```
