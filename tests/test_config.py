from tiny_imagenet_trainer.cli import build_parser, namespace_to_config
from tiny_imagenet_trainer.config import TrainingConfig


def test_training_config_serializes_paths(tmp_path):
    config = TrainingConfig(output_root=tmp_path, dataset_source="mock")

    payload = config.to_dict()

    assert payload["output_root"] == str(tmp_path)
    assert payload["dataset_source"] == "mock"


def test_cli_overrides_are_applied(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-source",
            "local",
            "--output-root",
            str(tmp_path),
            "--num-epochs",
            "2",
            "--no-compile",
            "--no-pretrained",
        ]
    )

    config = namespace_to_config(args)

    assert config.dataset_source == "local"
    assert config.output_root == tmp_path
    assert config.num_epochs == 2
    assert config.compile_model is False
    assert config.pretrained is False


def test_default_training_config_prefers_local_loading():
    config = TrainingConfig()

    assert config.dataset_source == "local"
    assert config.batch_size == 128
    assert config.num_workers == 2
