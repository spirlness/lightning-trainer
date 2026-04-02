import logging

from torch import nn

from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.trainer import Trainer


def test_trainer_can_resume_from_last_checkpoint(tmp_path):
    run_paths = RunPaths.from_root(tmp_path / "run")
    base_config = TrainingConfig(
        output_root=tmp_path,
        dataset_source="mock",
        device="cpu",
        num_classes=4,
        image_size=64,
        batch_size=4,
        num_workers=0,
        num_epochs=1,
        max_train_batches=1,
        max_val_batches=1,
        mock_train_samples=12,
        mock_val_samples=8,
        pretrained=False,
        compile_model=False,
        enable_amp=False,
    )
    logger = logging.getLogger("resume-test")
    bundle = build_dataloaders(base_config, logger)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, base_config.num_classes))

    trainer = Trainer(
        config=base_config,
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        run_paths=run_paths,
        device="cpu",
        logger=logger,
    )
    trainer.train()

    resume_config = TrainingConfig(
        output_root=tmp_path,
        dataset_source="mock",
        device="cpu",
        num_classes=4,
        image_size=64,
        batch_size=4,
        num_workers=0,
        num_epochs=2,
        max_train_batches=1,
        max_val_batches=1,
        mock_train_samples=12,
        mock_val_samples=8,
        pretrained=False,
        compile_model=False,
        enable_amp=False,
        resume_from=run_paths.checkpoints_dir / "last.pt",
    )
    resumed_trainer = Trainer(
        config=resume_config,
        model=nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, resume_config.num_classes)),
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        run_paths=run_paths,
        device="cpu",
        logger=logger,
    )

    assert resumed_trainer.state.completed_epochs == 1
