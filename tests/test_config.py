from tiny_imagenet_trainer.cli import build_parser, namespace_to_config
from tiny_imagenet_trainer.config import TrainingConfig


def test_training_config_serializes_paths(tmp_path):
    config = TrainingConfig(output_root=tmp_path, data_dir=tmp_path / "dataset")

    payload = config.to_dict()

    assert payload["output_root"] == str(tmp_path)
    assert payload["data_dir"] == str(tmp_path / "dataset")


def test_cli_overrides_are_applied(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--data-dir",
            str(tmp_path / "dataset"),
            "--output-root",
            str(tmp_path),
            "--num-epochs",
            "2",
            "--no-pretrained",
        ]
    )

    config = namespace_to_config(args)

    assert config.data_dir == tmp_path / "dataset"
    assert config.output_root == tmp_path
    assert config.num_epochs == 2
    assert config.pretrained is False


def test_default_training_config_is_minimal_local_setup():
    config = TrainingConfig()

    assert config.data_dir.name == "tiny_imagenet_local"
    assert config.batch_size == 128
    assert config.num_workers == 2
