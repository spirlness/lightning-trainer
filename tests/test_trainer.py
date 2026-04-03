import logging
from pathlib import Path

from PIL import Image
from torch import nn

from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders
from tiny_imagenet_trainer.trainer import Trainer


def test_trainer_completes_one_epoch_and_saves_core_artifacts(tmp_path):
    data_dir = tmp_path / "tiny-imagenet-local"
    _create_local_imagefolder_dataset(data_dir)

    run_paths = RunPaths.from_root(tmp_path / "run")
    config = TrainingConfig(
        output_root=tmp_path,
        data_dir=data_dir,
        device="cpu",
        num_classes=2,
        image_size=64,
        batch_size=2,
        num_workers=0,
        num_epochs=1,
        pretrained=False,
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
    assert (run_paths.checkpoints_dir / "best.pt").exists()
    assert run_paths.history_file.exists()


def _create_local_imagefolder_dataset(root: Path) -> None:
    for split in ("train", "val"):
        for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGB", (72, 72), color=color)
                image.save(class_dir / f"{split}_{class_name}_{index}.png")
