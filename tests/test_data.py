import logging
from pathlib import Path

import pytest
from PIL import Image

from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders


def test_local_imagefolder_dataloaders_return_expected_shapes(tmp_path):
    _create_local_imagefolder_dataset(tmp_path / "tiny-imagenet-local")
    config = TrainingConfig(
        output_root=tmp_path,
        data_dir=tmp_path / "tiny-imagenet-local",
        device="cpu",
        num_classes=2,
        image_size=64,
        batch_size=2,
        num_workers=0,
        pretrained=False,
        enable_amp=False,
    )

    bundle = build_dataloaders(config, logging.getLogger("test"))
    batch = next(iter(bundle.train_loader))

    assert batch["pixel_values"].shape == (2, 3, 64, 64)
    assert batch["label"].shape == (2,)
    assert bundle.class_names == ["class_a", "class_b"]


def test_missing_local_dataset_raises_clear_error(tmp_path):
    config = TrainingConfig(output_root=tmp_path, data_dir=tmp_path / "missing-dataset")

    with pytest.raises(FileNotFoundError, match="Expected a local imagefolder dataset"):
        build_dataloaders(config, logging.getLogger("test"))


def _create_local_imagefolder_dataset(root: Path) -> None:
    for split in ("train", "val"):
        for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGB", (72, 72), color=color)
                image.save(class_dir / f"{split}_{class_name}_{index}.png")
