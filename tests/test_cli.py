import argparse

import pytest

from tiny_imagenet_trainer.cli import build_parser, namespace_to_config
from tiny_imagenet_trainer.config import TrainingConfig


def _parse(parser: argparse.ArgumentParser, *args: str) -> TrainingConfig:
    return namespace_to_config(parser.parse_args(list(args)))


def test_parser_defaults_match_training_config():
    parser = build_parser()

    config_from_cli = _parse(parser)
    config_default = TrainingConfig()

    assert config_from_cli == config_default


def test_parser_allows_zero_for_non_negative_fields(tmp_path):
    parser = build_parser()

    config = _parse(
        parser,
        "--data-dir",
        str(tmp_path / "dataset"),
        "--output-root",
        str(tmp_path),
        "--warmup-steps",
        "0",
        "--num-workers",
        "0",
    )

    assert config.warmup_steps == 0
    assert config.num_workers == 0


def test_parser_still_rejects_invalid_positive_fields(tmp_path):
    parser = build_parser()

    with pytest.raises(SystemExit):
        _parse(
            parser,
            "--data-dir",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path),
            "--batch-size",
            "0",
        )
