import pytest
from pathlib import Path
from tiny_imagenet_trainer.config import TrainingConfig, build_parser, parse_args


def test_default_config():
    config = TrainingConfig()
    assert config.num_classes == 200
    assert config.batch_size == 256
    assert config.device in {"auto", "cpu", "cuda"}


def test_config_validation():
    with pytest.raises(ValueError, match="num_classes must be positive"):
        TrainingConfig(num_classes=0)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        TrainingConfig(batch_size=-1)

    with pytest.raises(ValueError, match="Invalid device"):
        TrainingConfig(device="tpu")


def test_config_serialization():
    config = TrainingConfig(data_dir=Path("test/path"))
    d = config.to_dict()
    assert isinstance(d["data_dir"], str)
    assert d["data_dir"] == str(Path("test/path"))


def test_cli_parser():
    parser = build_parser()
    args = parser.parse_args(["--batch-size", "64", "--no-pretrained"])
    assert args.batch_size == 64
    assert args.pretrained is False


def test_parse_args():
    config = parse_args(["--learning-rate", "0.01", "--device", "cpu"])
    assert config.learning_rate == 0.01
    assert config.device == "cpu"
