"""Command line interface for training with optimized argument parsing."""
from __future__ import annotations

import argparse
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Callable, Sequence, get_args, get_origin, get_type_hints

from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.environment import (
    configure_logger,
    prepare_run,
    seed_everything,
    select_device,
)
from tiny_imagenet_trainer.modeling import build_model
from tiny_imagenet_trainer.trainer import Trainer


CLI_GROUP_DESCRIPTIONS: dict[str, str] = {
    "Experiment": "Experiment configuration",
    "Data": "Data configuration",
    "Model": "Model configuration",
    "Training": "Training hyperparameters",
    "Data Loading": "Data loading configuration",
    "Hardware": "Hardware configuration",
    "Logging": "Logging configuration",
    "Reproducibility": "Reproducibility configuration",
}


def _flag_name(field_name: str) -> str:
    return field_name.replace("_", "-")


def _field_default(field_info) -> Any:
    if field_info.default is not MISSING:
        return field_info.default
    if field_info.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field_info.default_factory()
    raise ValueError(f"Field {field_info.name} requires a default")


def _resolve_annotation(field_info) -> tuple[Any, bool]:
    type_hints = get_type_hints(TrainingConfig, include_extras=True)
    annotated_type = type_hints.get(field_info.name, field_info.type)
    origin = get_origin(annotated_type)
    if origin is None:
        return annotated_type, False

    args = get_args(annotated_type)
    non_none = [arg for arg in args if arg is not type(None)]
    if len(non_none) == 1:
        return non_none[0], True
    return annotated_type, False


def _numeric_converter_factory(
    caster: Callable[[str], Any],
    metadata: dict[str, Any],
    allow_none: bool,
) -> Callable[[str], Any]:
    minimum = metadata.get("cli_min")
    inclusive = metadata.get("cli_min_inclusive", True)
    field_label = metadata.get("help") or "value"

    def _convert(raw: str) -> Any:
        if allow_none and raw.lower() == "none":
            return None
        try:
            value = caster(raw)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"{raw} is not a valid {caster.__name__}") from exc
        if minimum is not None:
            if inclusive and value < minimum:
                raise argparse.ArgumentTypeError(f"{field_label} must be >= {minimum}")
            if not inclusive and value <= minimum:
                raise argparse.ArgumentTypeError(f"{field_label} must be > {minimum}")
        return value

    return _convert


def _resolve_converter(
    base_type: Any,
    metadata: dict[str, Any],
    allow_none: bool,
) -> Callable[[str], Any] | Path | None:
    if base_type is int:
        return _numeric_converter_factory(int, metadata, allow_none)
    if base_type is float:
        return _numeric_converter_factory(float, metadata, allow_none)
    if base_type is Path:
        def _convert(raw: str) -> Path:
            return Path(raw)
        return _convert
    return None


def _add_bool_argument(
    group: argparse._ArgumentGroup,
    field_info,
    help_text: str,
    default: bool,
) -> None:
    dest = field_info.name
    flag = f"--{_flag_name(dest)}"
    neg_flag = f"--no-{_flag_name(dest)}"

    group.add_argument(
        flag,
        dest=dest,
        action="store_true",
        default=default,
        help=help_text,
    )
    group.add_argument(
        neg_flag,
        dest=dest,
        action="store_false",
        default=argparse.SUPPRESS,
        help=f"Disable {help_text}",
    )

    # Backwards compatibility for legacy AMP flag names
    if dest == "enable_amp":
        group.add_argument(
            "--amp",
            dest=dest,
            action="store_true",
            default=argparse.SUPPRESS,
            help="Enable Automatic Mixed Precision (alias)",
        )
        group.add_argument(
            "--no-amp",
            dest=dest,
            action="store_false",
            default=argparse.SUPPRESS,
            help="Disable Automatic Mixed Precision (alias)",
        )


def _add_standard_argument(
    group: argparse._ArgumentGroup,
    field_info,
) -> None:
    metadata = dict(field_info.metadata)
    base_type, optional_from_type = _resolve_annotation(field_info)
    allow_none = metadata.get("cli_allow_none", False) or optional_from_type

    if allow_none:
        metadata = {**metadata, "cli_allow_none": True}

    help_text = metadata.get("help", field_info.name.replace("_", " "))
    default = _field_default(field_info)
    dest = field_info.name
    flag = f"--{_flag_name(dest)}"

    kwargs: dict[str, Any] = {
        "dest": dest,
        "default": default,
        "help": help_text,
    }

    if "choices" in metadata:
        kwargs["choices"] = metadata["choices"]

    converter = _resolve_converter(base_type, metadata, allow_none)
    if converter is not None:
        kwargs["type"] = converter

    group.add_argument(flag, **kwargs)

    if allow_none and base_type is not bool:
        group.add_argument(
            f"--no-{_flag_name(dest)}",
            dest=dest,
            action="store_const",
            const=None,
            default=argparse.SUPPRESS,
            help=f"Disable {help_text}",
        )


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser using TrainingConfig metadata."""
    parser = argparse.ArgumentParser(
        prog="tiny-imagenet-train",
        description="Train a ResNet18 classifier on Tiny-ImageNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    groups: dict[str, argparse._ArgumentGroup] = {}

    for field_info in fields(TrainingConfig):
        group_name = field_info.metadata.get("group", "General")
        if group_name not in groups:
            description = CLI_GROUP_DESCRIPTIONS.get(group_name)
            groups[group_name] = parser.add_argument_group(
                group_name,
                description if description else None,
            )
        group = groups[group_name]

        base_type, _ = _resolve_annotation(field_info)
        if base_type is bool:
            _add_bool_argument(group, field_info, field_info.metadata.get("help", field_info.name), _field_default(field_info))
        else:
            _add_standard_argument(group, field_info)

    return parser


def namespace_to_config(args: argparse.Namespace) -> TrainingConfig:
    """Convert parsed arguments to TrainingConfig."""
    return TrainingConfig(**vars(args))


def main(args: Sequence[str] | None = None) -> int:
    """Main entry point for training."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    config = namespace_to_config(parsed_args)

    seed_everything(config.seed)
    run_paths, logger = prepare_run(config)
    logger.info("Starting training: %s", config.experiment_name)
    logger.info("Run directory: %s", run_paths.run_dir.resolve())

    dataloaders = build_dataloaders(config, logger)
    model = build_model(config, logger)
    device_info = select_device(config.device)
    logger.info("Using device: %s (%s)", device_info.device, device_info.name)

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
