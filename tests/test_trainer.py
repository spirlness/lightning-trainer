import logging
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
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


def test_trainer_moves_labels_only_once_per_batch(tmp_path):
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
    logger = logging.getLogger("trainer-label-move-test")
    bundle = build_dataloaders(config, logger)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, config.num_classes))

    move_counter = _count_tensor_to_calls()

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=_wrap_loader(bundle.train_loader, move_counter),
        val_loader=_wrap_loader(bundle.val_loader, move_counter),
        run_paths=run_paths,
        device="cpu",
        logger=logger,
    )

    trainer.train()

    assert move_counter["count"] == _expected_label_transfers(bundle)


def _expected_label_transfers(bundle):
    return len(bundle.train_loader) + len(bundle.val_loader)


def _wrap_loader(loader, move_counter):
    def _gen():
        for batch in loader:
            batch = batch.copy()
            batch["label"] = _TrackedTensor(batch["label"], move_counter)
            yield batch

    return list(_gen())


class _TrackedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor, counter: dict[str, int]):
        obj = torch.Tensor._make_subclass(cls, tensor.clone(), require_grad=False)
        obj._counter = counter
        return obj

    def to(self, *args, **kwargs):
        self._counter["count"] += 1
        return super().to(*args, **kwargs)


def _count_tensor_to_calls():
    return {"count": 0}


def _create_local_imagefolder_dataset(root: Path) -> None:
    for split in ("train", "val"):
        for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGB", (72, 72), color=color)
                image.save(class_dir / f"{split}_{class_name}_{index}.png")
