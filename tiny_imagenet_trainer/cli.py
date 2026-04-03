from __future__ import annotations

import argparse

from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.environment import prepare_run, seed_everything, select_device
from tiny_imagenet_trainer.modeling import build_model
from tiny_imagenet_trainer.trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Tiny-ImageNet classifier with a minimal, beginner-friendly pipeline."
    )
    parser.add_argument("--experiment-name", default="tiny-imagenet-resnet18")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--data-dir", type=str, default="data/tiny_imagenet_local")

    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--no-amp", dest="enable_amp", action="store_false")
    parser.set_defaults(pretrained=True, enable_amp=True)
    return parser


def namespace_to_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = namespace_to_config(args)

    seed_everything(config.seed)
    run_paths, logger = prepare_run(config)
    logger.info("Run directory: %s", run_paths.run_dir.resolve())

    dataloaders = build_dataloaders(config, logger)
    model = build_model(config, logger)
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=dataloaders.train_loader,
        val_loader=dataloaders.val_loader,
        run_paths=run_paths,
        device=select_device(config.device),
        logger=logger,
    )
    trainer.train()
    logger.info("Artifacts written to %s", run_paths.run_dir.resolve())
    return 0
