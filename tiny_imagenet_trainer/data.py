from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tiny_imagenet_trainer.config import TrainingConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(slots=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_size: int
    val_size: int
    class_names: list[str]


class DictClassificationDataset(Dataset[dict[str, Any]]):
    """Adapt tuple-style torchvision datasets into dict batches."""

    def __init__(self, base_dataset: Dataset[Any]) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label = self.base_dataset[index]
        return {"pixel_values": image, "label": label}


def build_train_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(image_size: int) -> transforms.Compose:
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_dataloaders(config: TrainingConfig, logger) -> DatasetBundle:
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"
    _validate_dataset_layout(train_dir, val_dir)

    train_dataset = DictClassificationDataset(
        ImageFolder(train_dir, transform=build_train_transforms(config.image_size))
    )
    val_dataset = DictClassificationDataset(
        ImageFolder(val_dir, transform=build_eval_transforms(config.image_size))
    )

    class_names = train_dataset.base_dataset.classes
    if class_names != val_dataset.base_dataset.classes:
        raise ValueError("train and val splits must contain the same class folders")
    if len(class_names) != config.num_classes:
        raise ValueError(
            "num_classes does not match the local dataset class count. "
            f"Expected {len(class_names)}, got {config.num_classes}."
        )

    loader_kwargs = _common_loader_kwargs(config)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    logger.info(
        "Loaded local dataset from '%s' with %s train samples and %s validation samples.",
        config.data_dir,
        len(train_dataset),
        len(val_dataset),
    )
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        class_names=class_names,
    )


def _common_loader_kwargs(config: TrainingConfig) -> dict[str, Any]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available() and config.device != "cpu",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return loader_kwargs


def _validate_dataset_layout(train_dir: Path, val_dir: Path) -> None:
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected a local imagefolder dataset at "
            f"'{train_dir.parent}' with 'train/' and 'val/' subdirectories."
        )
    if not any(path.is_dir() for path in train_dir.iterdir()):
        raise FileNotFoundError(f"Training split '{train_dir}' does not contain class folders.")
    if not any(path.is_dir() for path in val_dir.iterdir()):
        raise FileNotFoundError(f"Validation split '{val_dir}' does not contain class folders.")
