"""Tiny-ImageNet 数据加载流水线。
负责数据的增强处理、归一化以及 DataLoader 的构建。
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(is_train: bool, image_size: int = 224) -> transforms.Compose:
    """构建数据预处理和增强流水线。

    参数:
        is_train: 是否为训练模式。训练模式下会加入随机增强。
        image_size: 最终输入模型的图像尺寸。
    """
    # ImageNet 标准归一化参数
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    if is_train:
        # 训练集：加入随机裁剪、翻转和自动增强策略以提升泛化能力
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.TrivialAugmentWide(),  # 零超参数增强，自动选择最优策略
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        # 验证集：采用缩放后中心裁剪，确保测试一致性
        return transforms.Compose(
            [
                transforms.Resize(256),  # 先缩放到稍大的尺寸
                transforms.CenterCrop(image_size),  # 中心裁剪到目标尺寸
                transforms.ToTensor(),
                normalize,
            ]
        )


def build_dataloaders(
    data_dir: Path | str,
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 224,
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证集的 DataLoader。

    参数:
        data_dir: 数据集根目录，内部应包含 'train' 和 'val' 文件夹。
        batch_size: 批处理大小。
        num_workers: 多线程加载的线程数。
        image_size: 图像尺寸。
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    # 检查路径是否存在
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Expected 'train' and 'val' subdirectories."
        )

    # 使用 ImageFolder 自动根据文件夹名分配标签
    train_dataset = datasets.ImageFolder(
        train_dir, transform=build_transforms(is_train=True, image_size=image_size)
    )

    val_dataset = datasets.ImageFolder(
        val_dir, transform=build_transforms(is_train=False, image_size=image_size)
    )

    # 构建训练加载器：打乱顺序，丢弃最后一个不完整的 batch (防止 BatchNorm 报错)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,  # 避免每个 epoch 重建 worker 进程
        drop_last=True,
    )

    # 构建验证加载器：无需打乱
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader
