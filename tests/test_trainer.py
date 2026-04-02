import logging

from torch import nn

from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.trainer import Trainer


def test_trainer_completes_one_epoch_on_mock_data(tmp_path):
    run_paths = RunPaths.from_root(tmp_path / "run")
    config = TrainingConfig(
        output_root=tmp_path,
        dataset_source="mock",
        device="cpu",
        num_classes=4,
        image_size=64,
        batch_size=4,
        num_workers=0,
        num_epochs=1,
        max_train_batches=2,
        max_val_batches=1,
        mock_train_samples=16,
        mock_val_samples=8,
        pretrained=False,
        compile_model=False,
        enable_amp=False,
        log_every_n_steps=1,
    )
    logger = logging.getLogger("trainer-test")
    bundle = build_dataloaders(config, logger)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, config.num_classes))

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        run_paths=run_paths,
        device="cpu",
        logger=logger,
    )

    history = trainer.train()

    assert len(history) == 1
    assert (run_paths.checkpoints_dir / "last.pt").exists()
    assert run_paths.history_file.exists()
