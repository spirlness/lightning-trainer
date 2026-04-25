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
  download_data.py                 # Dataset download script
tests/
  test_lightning_trainer.py
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
uv run --extra dev python scripts/download_data.py --method huggingface

# Or create a small local subset from an existing full dataset
uv run --extra dev python scripts/download_data.py --method subset --num-classes 10 --images-per-class 100
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

For the current local Tiny-ImageNet cache at `image_size=128`, the generated
files are approximately:

```text
train/images.bin  4.92 GB
val/images.bin    0.49 GB
```

## Train

Basic ImageFolder training:

```bash
python -m lightning_trainer.train \
  --data-dir data/tiny-imagenet-200 \
  --cache-dir "" \
  --batch-size 128 \
  --max-epochs 10
```

Cached-data training:

```bash
python -m lightning_trainer.train
```

The default training configuration uses the current recommended local setup:
`data/tiny_imagenet_local`, `data/tiny_imagenet_cache_128`, `batch_size=128`,
`image_size=128`, `num_workers=4`, `torch.compile`, fused AdamW, AMP,
`channels_last`, CSV logging, checkpointing, LR monitoring, and cosine LR
scheduling.

## Benchmark

Best current full-epoch benchmark command on this machine:

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

Measured environment:

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
PyTorch: 2.11.0+cu128
Dataset: 200 classes, 100000 train images, 10000 val images
Input: tensor cache, image_size=128, precision=16-mixed
Optimizations: torch.compile, channels_last, fused AdamW, drop_last=True
```

Recent benchmark results:

| batch size | workers | step throughput | epoch elapsed | peak memory | note |
|---:|---:|---:|---:|---:|---|
| 128 | 4 | 1082.89 samples/s | 141.53 s | 2807.3 MB | Recommended |
| 192 | 4 | 1085.36 samples/s | 143.89 s | 3998.2 MB | Similar speed, higher memory |
| 256 | 8 | 1028.74 samples/s | short-window only | 5167.8 MB | Not recommended |

Recommended default for this machine:

```text
batch_size=128
num_workers=4
image_size=128
cache_dir=data/tiny_imagenet_cache_128
```

## Design Notes

- Model is fixed to ConvNeXt-Tiny.
- Training uses fixed resize and horizontal flip; random crop is intentionally
  removed.
- Train DataLoader uses `drop_last=True` to avoid dynamic final-batch shapes
  with `torch.compile`.
- AMP, `channels_last`, fused AdamW, and `torch.compile` are supported.
- The training CLI intentionally exposes only business parameters:
  `data_dir/cache_dir/batch_size/max_epochs/no_compile`.
- Lightning handles CSV logging, checkpointing, LR monitoring, precision,
  device placement, optimizer stepping, and scheduler stepping.
- The training CLI enables `torch.compile` and fused AdamW by default. Use
  `--no-compile` to disable compile for fallback runs.
- `--gradient-checkpointing` only has an effect for models exposing a native
  `gradient_checkpointing_enable` method. Current torchvision ConvNeXt-Tiny
  does not, so the flag warns and continues.

## Validate

```bash
uv run ruff check .
uv run --extra dev python -m pytest -q
```
