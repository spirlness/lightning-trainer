import importlib
import logging
from pathlib import Path

import pytest
from PIL import Image

from tiny_imagenet_trainer import data
from tiny_imagenet_trainer.config import TrainingConfig
from tiny_imagenet_trainer.data import build_dataloaders


def test_mock_dataloaders_return_expected_shapes(tmp_path):
    config = TrainingConfig(
        output_root=tmp_path,
        dataset_source="mock",
        device="cpu",
        num_classes=4,
        image_size=64,
        batch_size=8,
        num_workers=0,
        mock_train_samples=16,
        mock_val_samples=8,
        pretrained=False,
        compile_model=False,
    )

    bundle = build_dataloaders(config, logging.getLogger("test"))
    batch = next(iter(bundle.train_loader))

    assert batch["pixel_values"].shape == (8, 3, 64, 64)
    assert batch["label"].shape == (8,)


def test_huggingface_loader_requires_datasets_dependency(tmp_path, monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return original_import_module(name, package)

    monkeypatch.setattr(data.importlib, "import_module", fake_import_module)

    config = TrainingConfig(output_root=tmp_path, dataset_source="huggingface")

    with pytest.raises(RuntimeError, match="Install `pip install -e .\\[data\\]`"):
        build_dataloaders(config, logging.getLogger("test"))


def test_local_imagefolder_dataloaders_return_expected_shapes(tmp_path):
    _create_local_imagefolder_dataset(tmp_path / "tiny-imagenet-local")
    config = TrainingConfig(
        output_root=tmp_path,
        dataset_source="local",
        local_dataset_dir=tmp_path / "tiny-imagenet-local",
        device="cpu",
        num_classes=2,
        image_size=64,
        batch_size=2,
        num_workers=0,
        pretrained=False,
        compile_model=False,
    )

    bundle = build_dataloaders(config, logging.getLogger("test"))
    batch = next(iter(bundle.train_loader))

    assert batch["pixel_values"].shape == (2, 3, 64, 64)
    assert batch["label"].shape == (2,)
    assert bundle.class_names == ["class_a", "class_b"]


def _create_local_imagefolder_dataset(root: Path) -> None:
    for split in ("train", "val"):
        for class_name, color in (("class_a", (255, 0, 0)), ("class_b", (0, 255, 0))):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = Image.new("RGB", (72, 72), color=color)
                image.save(class_dir / f"{split}_{class_name}_{index}.png")
