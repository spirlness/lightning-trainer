"""Data loading and preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from tiny_imagenet_trainer.config import TrainingConfig

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class DataLoaders:
    """Container for train and validation data loaders."""
    train_loader: DataLoader
    val_loader: DataLoader
    num_classes: int
    class_names: list[str]


class DictDataset(Dataset[dict[str, Any]]):
    """Wrap tuple-style dataset to return dict samples.

    Converts (image, label) tuples to {"pixel_values": image, "label": label} dicts
    for consistent batch collation.
    """

    def __init__(self, dataset: Dataset[Any]) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image, label = self.dataset[idx]
        return {"pixel_values": image, "label": label}


def create_train_transforms(image_size: int) -> transforms.Compose:
    """Create training transforms with augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_val_transforms(image_size: int) -> transforms.Compose:
    """Create validation transforms (no augmentation)."""
    # Standard ImageNet validation resize followed by center crop
    resize_size = int(image_size * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_dataloaders(config: TrainingConfig, logger) -> DataLoaders:
    """Build training and validation data loaders.

    Args:
        config: Training configuration
        logger: Logger instance

    Returns:
        DataLoaders container with train/val loaders and class info

    Raises:
        FileNotFoundError: If dataset directories don't exist or are empty
        ValueError: If train/val classes don't match or num_classes mismatch
    """
    train_dir = config.data_dir / "train"
    val_dir = config.data_dir / "val"

    # Validate dataset structure
    _validate_dataset(train_dir, val_dir)

    # Create datasets with appropriate transforms
    train_ds = DictDataset(ImageFolder(
        train_dir,
        transform=create_train_transforms(config.image_size)
    ))
    val_ds = DictDataset(ImageFolder(
        val_dir,
        transform=create_val_transforms(config.image_size)
    ))

    # Validate class consistency
    train_classes = train_ds.dataset.classes
    val_classes = val_ds.dataset.classes

    if train_classes != val_classes:
        raise ValueError(
            f"Train and validation classes don't match. "
            f"Train: {len(train_classes)}, Val: {len(val_classes)}"
        )

    if len(train_classes) != config.num_classes:
        raise ValueError(
            f"config.num_classes ({config.num_classes}) doesn't match "
            f"dataset classes ({len(train_classes)})"
        )

    # Create data loaders
    loader_kwargs = _make_loader_kwargs(config)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    logger.info(
        "Loaded dataset: %d train, %d val samples, %d classes",
        len(train_ds), len(val_ds), len(train_classes)
    )

    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=len(train_classes),
        class_names=train_classes,
    )


def _validate_dataset(train_dir: Path, val_dir: Path) -> None:
    """Validate dataset directory structure."""
    missing_message = (
        "Expected a local imagefolder dataset with structure: "
        "data_dir/{split}/class_name/image.jpg"
    )
    for split_dir, name in [(train_dir, "train"), (val_dir, "val")]:
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{name} directory not found: {split_dir}\n{missing_message.format(split=name)}"
            )

        # Check for class subdirectories
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            raise FileNotFoundError(
                f"No class subdirectories found in {split_dir}\n{missing_message.format(split=name)}"
            )


def _make_loader_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Create DataLoader keyword arguments."""
    kwargs: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available() and config.device != "cpu",
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs
