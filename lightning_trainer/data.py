"""极简数据模块 - PyTorch Lightning DataModule."""

import json
from pathlib import Path
from typing import Protocol

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ClassificationDataset(Protocol):
    classes: list[str]


class CachedTensorDataset(Dataset):
    """Memory-mapped tensor cache for fixed-size RGB images."""

    def __init__(self, split_dir: Path, normalize: transforms.Normalize) -> None:
        manifest_path = split_dir / "manifest.json"
        labels_path = split_dir / "labels.pt"
        images_path = split_dir / "images.bin"
        if not manifest_path.exists():
            raise FileNotFoundError(f"缓存 manifest 不存在: {manifest_path}")
        if not labels_path.exists() or not images_path.exists():
            raise FileNotFoundError(f"缓存文件不完整: {split_dir}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.classes = list(manifest["classes"])
        self.num_samples = int(manifest["num_samples"])
        self.image_size = int(manifest["image_size"])
        self.normalize = normalize
        self.labels = torch.load(labels_path, map_location="cpu", weights_only=True)
        numel = self.num_samples * 3 * self.image_size * self.image_size
        self.images = torch.from_file(
            str(images_path),
            size=numel,
            dtype=torch.uint8,
        ).view(self.num_samples, 3, self.image_size, self.image_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].to(torch.float32).div_(255.0)
        label = self.labels[index]
        return self.normalize(image), label


class TinyImageNetDataModule(LightningDataModule):
    """Tiny-ImageNet 数据模块"""

    def __init__(
        self,
        data_dir: str | Path = "data/tiny-imagenet-200",
        batch_size: int = 128,
        num_workers: int = 2,
        image_size: int = 224,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes: int = 0  # 在 setup 中初始化

        # ImageNet 标准归一化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # 训练数据增强
        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        # 验证数据变换
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(image_size + 32),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def setup(self, stage: str | None = None) -> None:
        if self.cache_dir is not None:
            self._setup_cached()
            return

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"
        if not train_dir.exists():
            raise FileNotFoundError(f"训练数据目录不存在: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"验证数据目录不存在: {val_dir}")

        self.train_dataset = ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = ImageFolder(val_dir, transform=self.val_transform)
        if self.train_dataset.classes != self.val_dataset.classes:
            raise ValueError("训练集和验证集类别目录不一致")
        self.num_classes = len(self.train_dataset.classes)

        # 测试集（可选）
        self.test_dataset = None
        if test_dir.exists():
            self.test_dataset = ImageFolder(test_dir, transform=self.val_transform)
            if self.test_dataset.classes != self.train_dataset.classes:
                raise ValueError("测试集和训练集类别目录不一致")

    def _setup_cached(self) -> None:
        assert self.cache_dir is not None
        train_dir = self.cache_dir / "train"
        val_dir = self.cache_dir / "val"
        test_dir = self.cache_dir / "test"
        self.train_dataset = CachedTensorDataset(train_dir, self.normalize)
        self.val_dataset = CachedTensorDataset(val_dir, self.normalize)
        if self.train_dataset.classes != self.val_dataset.classes:
            raise ValueError("训练缓存和验证缓存类别不一致")
        if self.train_dataset.image_size != self.image_size:
            raise ValueError(
                f"缓存 image_size={self.train_dataset.image_size} "
                f"与当前 image_size={self.image_size} 不一致"
            )
        self.num_classes = len(self.train_dataset.classes)

        self.test_dataset = None
        if test_dir.exists():
            self.test_dataset = CachedTensorDataset(test_dir, self.normalize)
            if self.test_dataset.classes != self.train_dataset.classes:
                raise ValueError("测试缓存和训练缓存类别不一致")

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        return self._build_loader(self.test_dataset, shuffle=False, drop_last=False)

    def train_dataloader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def _build_loader(
        self,
        dataset: ClassificationDataset,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )
