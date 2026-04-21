# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A minimal, beginner-friendly PyTorch training pipeline for Tiny-ImageNet image classification using ResNet18. The project emphasizes code clarity, modularity, and local-first data handling.

## Common Commands

### Training

```bash
# Basic training run
python main.py --data-dir data/tiny_imagenet_local --num-epochs 10

# Full parameter example
python main.py \
  --data-dir data/tiny_imagenet_local \
  --output-root outputs \
  --experiment-name tiny-imagenet-resnet18 \
  --batch-size 128 \
  --num-epochs 10 \
  --learning-rate 1e-4 \
  --weight-decay 1e-4 \
  --warmup-steps 50 \
  --num-workers 2 \
  --device cuda \
  --seed 42

# CPU training
python main.py --data-dir data/tiny_imagenet_local --device cpu --num-epochs 5

# Disable pretrained weights
python main.py --data-dir data/tiny_imagenet_local --no-pretrained

# Disable automatic mixed precision
python main.py --data-dir data/tiny_imagenet_local --no-amp
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py
pytest tests/test_data.py
pytest tests/test_trainer.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=tiny_imagenet_trainer --cov-report=term-missing
```

### Installation

```bash
# Install in editable mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Use the CLI entry point after installation
tiny-imagenet-train --data-dir data/tiny_imagenet_local --num-epochs 10
```

## Architecture Overview

### Module Responsibilities

The codebase follows a clean separation of concerns with each module having a single, well-defined responsibility:

**`config.py`** - Configuration management using `@dataclass(slots=True)` for memory efficiency. Defines `TrainingConfig` (hyperparameters, paths, training settings) and `RunPaths` (runtime directory structure). Handles path serialization and validation in `__post_init__`.

**`data.py`** - Data pipeline using torchvision's `ImageFolder`. Key abstraction: `DictClassificationDataset` wraps tuple-based datasets to return dict batches `{"pixel_values": tensor, "label": tensor}`. Handles ImageNet normalization, data augmentation (train), center crop (eval), and validation of dataset layout.

**`modeling.py`** - Model factory. Builds ResNet18 from torchvision, handles pretrained weight loading with fallback, and replaces the final FC layer to match `num_classes`.

**`trainer.py`** - Core training orchestration. Contains the `Trainer` class with:
- Training loop with AMP (Automatic Mixed Precision) support
- Validation/evaluation loop
- Checkpoint saving (best/last) via `checkpointing.py`
- LR scheduling (warmup + cosine annealing)
- Gradient clipping and scaling

**`checkpointing.py`** - PyTorch checkpoint I/O. Saves/loads model state, optimizer, scheduler, scaler, and metadata.

**`environment.py`** - Runtime utilities: device selection (auto/cpu/cuda), seeding (Python, torch, numpy), logger configuration (file + console), JSON I/O.

**`cli.py`** - Argument parsing with `argparse`, config instantiation, and main entry orchestration.

### Data Flow

```
main.py
  └── cli.main()
        ├── config.TrainingConfig (parse args)
        ├── environment.prepare_run() (create run_dir, logger)
        ├── data.build_dataloaders() (ImageFolder → DataLoader)
        ├── modeling.build_model() (ResNet18)
        └── trainer.Trainer.train() (train loop)
              ├── _train_one_epoch()
              ├── evaluate()
              ├── _save_last_checkpoint()
              └── _save_best_checkpoint()
```

### Run Directory Structure

Each training run creates a timestamped directory under `outputs/`:

```
outputs/
└── tiny-imagenet-resnet18-20250101_120000/
    ├── config.json          # Serialized TrainingConfig
    ├── history.json         # Per-epoch metrics
    ├── checkpoints/
    │   ├── best.pt          # Best val_loss checkpoint
    │   └── last.pt          # Most recent checkpoint
    └── logs/
        └── train.log        # Training logs
```

## Key Design Patterns

1. **Dataclasses with `slots=True`**: Memory-efficient configuration objects with validation in `__post_init__`.

2. **Dict-style datasets**: The `DictClassificationDataset` wrapper converts tuple-based torchvision datasets into dict batches, enabling consistent batch handling.

3. **Run isolation**: Each execution creates a fresh run directory with all artifacts (configs, logs, checkpoints, metrics), enabling reproducibility and easy comparison.

4. **AMP-aware training**: The trainer dynamically selects bfloat16/float16 based on CUDA capability and handles gradient scaling automatically.

5. **Defensive validation**: Dataset layout, class count matching, and device availability are validated early with clear error messages.

## Testing Strategy

Tests use `tmp_path` pytest fixtures for isolated filesystem operations. Focus areas:
- Config serialization/deserialization
- CLI argument parsing
- Dataset validation and DataLoader construction
- Checkpoint save/load roundtrip

Run tests with: `pytest` or `pytest -v` for verbose output.

## Dependencies

Core: `torch>=2.6`, `torchvision>=0.21`, `pillow>=10.0`

Test: `pytest>=8.0`

Python: `>=3.11`

## Notes

- The trainer expects ImageFolder structure: `data_dir/train/{class}/image.jpg` and `data_dir/val/{class}/image.jpg`
- Class folders must match between train/val splits
- `num_classes` in config must match actual class folder count
- Pretrained weights are downloaded on first use (cached by torchvision)
- Mixed precision is automatically disabled on CPU
