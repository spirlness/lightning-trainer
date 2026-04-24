"""极简数据模块 - PyTorch Lightning DataModule"""
from pathlib import Path
from typing import Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TinyImageNetDataModule(LightningDataModule):
    """Tiny-ImageNet 数据模块"""

    def __init__(
        self,
        data_dir: str = "data/tiny-imagenet-200",
        batch_size: int = 128,
        num_workers: int = 2,
        image_size: int = 224,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.num_classes: int = 0  # 在 setup 中初始化

        # ImageNet 标准归一化
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # 训练数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])

        # 验证数据变换
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            self.normalize,
        ])

    def setup(self, stage: Optional[str] = None):
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        if not train_dir.exists():
            raise FileNotFoundError(f"训练数据目录不存在: {train_dir}")
        if not val_dir.exists():
            raise FileNotFoundError(f"验证数据目录不存在: {val_dir}")

        self.train_dataset = ImageFolder(train_dir, transform=self.train_transform)
        self.val_dataset = ImageFolder(val_dir, transform=self.val_transform)
        self.num_classes = len(self.train_dataset.classes)

        # 测试集（可选）
        self.test_dataset = None
        if test_dir.exists():
            self.test_dataset = ImageFolder(test_dir, transform=self.val_transform)

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )
