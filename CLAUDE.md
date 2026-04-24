# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal PyTorch Lightning training framework for Tiny-ImageNet image classification. Focuses on simplicity and core optimizations.

## Architecture

```
lightning_trainer/
├── data.py    # TinyImageNetDataModule - handles data loading
├── model.py   # ImageClassifier - LightningModule with optimizations
└── train.py   # CLI entry point
```

## Common Commands

### Training

```bash
# Basic training
python -m lightning_trainer.train --data-dir data/tiny-imagenet-200

# With all optimizations
python -m lightning_trainer.train \
    --data-dir data/tiny-imagenet-200 \
    --batch-size 128 \
    --compile \
    --gradient-checkpointing \
    --fused-optimizer

# With test evaluation
python -m lightning_trainer.train --data-dir data/tiny-imagenet-200 --test
```

### Data

```bash
# Download full dataset
python download_data.py --method huggingface

# Create test subset
python download_data.py --method subset --num-classes 10 --images-per-class 100
```

## Key Design Decisions

1. **PyTorch Lightning**: Handles device management, AMP, checkpointing automatically
2. **Opt-in optimizations**: All advanced features (compile, checkpointing, fused) are opt-in
3. **Minimal configuration**: Simple argparse, no complex config classes
4. **Windows compatible**: MSVC setup via environment variables

## Environment Variables (Windows)

- `MSVC_PATH`: Visual Studio installation path
- `MSVC_VERSION`: MSVC version number (auto-detected if not set)
- `WINDOWS_SDK_PATH`: Windows SDK path

## Dependencies

- pytorch-lightning >= 2.0
- torch, torchvision
- triton-windows (Windows only)
