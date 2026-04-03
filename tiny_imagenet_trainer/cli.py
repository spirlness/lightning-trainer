"""Command line interface for training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.environment import (
    prepare_run,
    seed_everything,
    select_device,
)
from tiny_imagenet_trainer.modeling import build_model
from tiny_imagenet_trainer.trainer import Trainer


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{raw} must be > 0")
    return value


def _non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{raw} must be >= 0")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{raw} must be > 0")
    return value


def _non_negative_float(raw: str) -> float:
    value = float(raw)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{raw} must be >= 0")
    return value


def _optional_positive_float(raw: str) -> float | None:
    if raw.lower() == "none":
        return None
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{raw} must be > 0")
    return value


def build_parser() -> argparse.ArgumentParser:
    """Build the training argument parser."""
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        prog="tiny-imagenet-train",
        description="Train a ResNet18 classifier on Tiny-ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    experiment = parser.add_argument_group("Experiment")
    experiment.add_argument(
        "--experiment-name",
        default=defaults.experiment_name,
        help="Name for this experiment",
    )
    experiment.add_argument(
        "--output-root",
        type=Path,
        default=defaults.output_root,
        help="Root directory for all experiment outputs",
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.data_dir,
        help="Path to dataset directory with train/ and val/ subdirectories",
    )
    data.add_argument(
        "--num-classes",
        type=_positive_int,
        default=defaults.num_classes,
        help="Number of classes in the dataset",
    )
    data.add_argument(
        "--image-size",
        type=_positive_int,
        default=defaults.image_size,
        help="Input image size (square)",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=defaults.pretrained,
        help="Use pretrained ImageNet weights",
    )
    model.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Disable pretrained ImageNet weights",
    )

    training = parser.add_argument_group("Training")
    training.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=defaults.learning_rate,
        help="Base learning rate",
    )
    training.add_argument(
        "--weight-decay",
        type=_non_negative_float,
        default=defaults.weight_decay,
        help="Weight decay (L2 regularization)",
    )
    training.add_argument(
        "--batch-size",
        type=_positive_int,
        default=defaults.batch_size,
        help="Batch size for training and validation",
    )
    training.add_argument(
        "--num-epochs",
        type=_positive_int,
        default=defaults.num_epochs,
        help="Number of training epochs",
    )
    training.add_argument(
        "--warmup-steps",
        type=_non_negative_int,
        default=defaults.warmup_steps,
        help="Number of warmup steps for learning rate",
    )
    training.add_argument(
        "--gradient-clip-norm",
        type=_optional_positive_float,
        default=defaults.gradient_clip_norm,
        help="Gradient clipping norm (use 'none' to disable)",
    )
    training.add_argument(
        "--no-gradient-clip-norm",
        dest="gradient_clip_norm",
        action="store_const",
        const=None,
        help="Disable gradient clipping",
    )

    data_loading = parser.add_argument_group("Data Loading")
    data_loading.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=defaults.num_workers,
        help="Number of data loading workers",
    )

    hardware = parser.add_argument_group("Hardware")
    hardware.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=defaults.device,
        help="Device to use for training",
    )
    hardware.add_argument(
        "--amp",
        dest="enable_amp",
        action="store_true",
        default=defaults.enable_amp,
        help="Enable Automatic Mixed Precision",
    )
    hardware.add_argument(
        "--enable-amp",
        dest="enable_amp",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    hardware.add_argument(
        "--no-amp",
        dest="enable_amp",
        action="store_false",
        help="Disable Automatic Mixed Precision",
    )

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-every-n-steps",
        type=_positive_int,
        default=defaults.log_every_n_steps,
        help="Log training metrics every N steps",
    )

    reproducibility = parser.add_argument_group("Reproducibility")
    reproducibility.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed for reproducibility",
    )

    return parser


def namespace_to_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert parsed arguments to TrainingConfig."""
    return TrainingConfig(**vars(args))


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point for training."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    config = namespace_to_config(parsed_args)
    device_info = select_device(config.device)

    seed_everything(config.seed)
    run_paths, logger = prepare_run(config)
    logger.info("Starting training: %s", config.experiment_name)
    logger.info("Run directory: %s", run_paths.run_dir.resolve())
    logger.info("Using device: %s (%s)", device_info.device, device_info.name)

    dataloaders = build_dataloaders(config, logger)
    model = build_model(config, logger)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=dataloaders.train_loader,
        val_loader=dataloaders.val_loader,
        run_paths=run_paths,
        device=device_info.device,
        logger=logger,
    )

    history = trainer.train()
    logger.info("Training complete. Artifacts saved to: %s", run_paths.run_dir)
    logger.info("Best validation loss: %.4f", min(h["val_loss"] for h in history))

    return 0


if __name__ == "__main__":
    sys.exit(main())
