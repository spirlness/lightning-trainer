import logging

from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.environment import prepare_run


def test_prepare_run_returns_logger_with_handlers(tmp_path):
    config = TrainingConfig(
        output_root=tmp_path / "outputs",
        data_dir=tmp_path / "data",
        device="cpu",
        enable_amp=False,
        num_epochs=1,
        num_workers=0,
    )

    run_paths, logger = prepare_run(config)

    assert logger.handlers, "Logger should have handlers configured"
    logger.info("hello world")

    log_text = run_paths.log_file.read_text(encoding="utf-8")
    assert "hello world" in log_text
