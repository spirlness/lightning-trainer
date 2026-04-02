from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FakeData, ImageFolder

from tiny_imagenet_trainer.config import TrainingConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(slots=True)
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader | None
    train_size: int
    val_size: int | None
    class_names: list[str] | None = None


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


def build_dataloaders(config: TrainingConfig, logger: logging.Logger) -> DatasetBundle:
    if config.dataset_source == "mock":
        return _build_mock_dataloaders(config)
    if config.dataset_source == "local":
        return _build_local_dataloaders(config, logger)
    return _build_huggingface_dataloaders(config, logger)


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


def _build_local_dataloaders(config: TrainingConfig, logger: logging.Logger) -> DatasetBundle:
    dataset_root = _ensure_local_imagefolder_dataset(config, logger)
    train_dataset = DictClassificationDataset(
        ImageFolder(dataset_root / "train", transform=build_train_transforms(config.image_size))
    )
    val_dataset = DictClassificationDataset(
        ImageFolder(dataset_root / "val", transform=build_eval_transforms(config.image_size))
    )

    class_names = train_dataset.base_dataset.classes
    if len(class_names) != config.num_classes:
        raise ValueError(
            "num_classes does not match the local dataset class count. "
            f"Expected {len(class_names)}, got {config.num_classes}."
        )

    loader_kwargs = _common_loader_kwargs(config)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    logger.info(
        "Loaded local imagefolder dataset from '%s' with %s train samples and %s validation samples.",
        dataset_root,
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


def _build_mock_dataloaders(config: TrainingConfig) -> DatasetBundle:
    train_dataset = DictClassificationDataset(
        FakeData(
            size=config.mock_train_samples,
            image_size=(3, config.image_size, config.image_size),
            num_classes=config.num_classes,
            transform=build_train_transforms(config.image_size),
        )
    )
    val_dataset = DictClassificationDataset(
        FakeData(
            size=config.mock_val_samples,
            image_size=(3, config.image_size, config.image_size),
            num_classes=config.num_classes,
            transform=build_eval_transforms(config.image_size),
        )
    )

    loader_kwargs = _common_loader_kwargs(config)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
    )


def _build_huggingface_dataloaders(
    config: TrainingConfig, logger: logging.Logger
) -> DatasetBundle:
    raw_train, raw_val, class_names = _load_huggingface_splits(config, logger)
    if class_names and len(class_names) != config.num_classes:
        raise ValueError(
            "num_classes does not match the dataset label count. "
            f"Expected {len(class_names)}, got {config.num_classes}."
        )

    train_dataset = raw_train.with_transform(_make_transform(build_train_transforms(config.image_size)))
    val_dataset = raw_val.with_transform(_make_transform(build_eval_transforms(config.image_size)))

    loader_kwargs = _common_loader_kwargs(config)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    logger.info(
        "Loaded dataset '%s' with %s train samples and %s validation samples.",
        config.dataset_name,
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


def _ensure_local_imagefolder_dataset(
    config: TrainingConfig, logger: logging.Logger
) -> Path:
    metadata_path = config.local_dataset_dir / "metadata.json"
    train_dir = config.local_dataset_dir / "train"
    val_dir = config.local_dataset_dir / "val"

    if metadata_path.exists() and train_dir.exists() and val_dir.exists():
        return config.local_dataset_dir
    if _looks_like_imagefolder_dataset(train_dir, val_dir):
        return config.local_dataset_dir

    logger.info(
        "Preparing local imagefolder dataset at '%s'. This is a one-time export step.",
        config.local_dataset_dir,
    )
    raw_train, raw_val, class_names = _load_huggingface_splits(config, logger)
    class_names = class_names or [f"class_{index:03d}" for index in range(config.num_classes)]

    _materialize_split(raw_train, train_dir, class_names, split_name="train", logger=logger)
    _materialize_split(raw_val, val_dir, class_names, split_name="val", logger=logger)

    metadata = {
        "dataset_name": config.dataset_name,
        "train_split": config.train_split,
        "val_split": config.val_split,
        "num_classes": len(class_names),
        "class_names": class_names,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return config.local_dataset_dir


def _looks_like_imagefolder_dataset(train_dir: Path, val_dir: Path) -> bool:
    if not train_dir.exists() or not val_dir.exists():
        return False
    train_classes = [path for path in train_dir.iterdir() if path.is_dir()]
    val_classes = [path for path in val_dir.iterdir() if path.is_dir()]
    return bool(train_classes) and bool(val_classes)


def _materialize_split(
    dataset: Any,
    split_dir: Path,
    class_names: list[str],
    split_name: str,
    logger: logging.Logger,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(dataset):
        label = int(sample["label"])
        class_name = class_names[label]
        target_path = split_dir / class_name / f"{index:06d}.jpg"
        if target_path.exists():
            continue

        image = sample["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image.convert("RGB").save(target_path, format="JPEG", quality=95)

        if (index + 1) % 5000 == 0:
            logger.info(
                "Exported %s/%s images for split '%s'.",
                index + 1,
                len(dataset),
                split_name,
            )


def _load_huggingface_splits(
    config: TrainingConfig, logger: logging.Logger
) -> tuple[Any, Any, list[str] | None]:
    datasets_module = _require_datasets_dependency()
    load_dataset = datasets_module.load_dataset

    download_mode = "force_redownload" if config.force_redownload else None
    raw_train = load_dataset(
        config.dataset_name,
        split=config.train_split,
        cache_dir=str(config.dataset_cache_dir),
        download_mode=download_mode,
    )

    try:
        raw_val = load_dataset(
            config.dataset_name,
            split=config.val_split,
            cache_dir=str(config.dataset_cache_dir),
            download_mode=download_mode,
        )
    except Exception as exc:
        logger.warning(
            "Validation split '%s' is unavailable (%s). Falling back to train/validation split.",
            config.val_split,
            exc,
        )
        split = raw_train.train_test_split(
            test_size=config.validation_split_ratio,
            seed=config.seed,
        )
        raw_train = split["train"]
        raw_val = split["test"]

    return raw_train, raw_val, _extract_class_names(raw_train)


def _require_datasets_dependency():
    try:
        return importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "dataset_source='huggingface' requires the optional `datasets` package. "
            "Install `pip install -e .[data]` or `pip install datasets`."
        ) from exc


def _make_transform(transform: transforms.Compose):
    def apply(batch: dict[str, list[Any]]) -> dict[str, Any]:
        pixel_values = []
        labels = []

        for image, label in zip(batch["image"], batch["label"], strict=False):
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            pixel_values.append(transform(image.convert("RGB")))
            labels.append(int(label))

        return {"pixel_values": pixel_values, "label": labels}

    return apply


def _extract_class_names(dataset: Any) -> list[str] | None:
    label_feature = getattr(dataset, "features", {}).get("label")
    if label_feature is None:
        return None
    names = getattr(label_feature, "names", None)
    return list(names) if names else None
