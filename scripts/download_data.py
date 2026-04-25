#!/usr/bin/env python
"""Tiny-ImageNet 数据集下载脚本。

支持多种下载方式：
1. Stanford 官方源
2. Hugging Face datasets
3. 手动下载指引
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Windows 需要特殊处理 SSL
if sys.platform == "win32":
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context


def download_from_stanford(data_dir: Path) -> bool:
    """从 Stanford 官方源下载 Tiny-ImageNet。

    Returns:
        True 如果下载成功
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_dir / "tiny-imagenet-200.zip"
    extract_dir = data_dir / "tiny-imagenet-200"

    print("=" * 60)
    print("从 Stanford 官方下载 Tiny-ImageNet")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"目标目录: {data_dir}")
    print()

    # 创建目录
    data_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已下载
    if extract_dir.exists() and (extract_dir / "train").exists():
        print(f"[跳过] 数据集已存在: {extract_dir}")
        return True

    # 下载
    if not zip_path.exists():
        print(f"[下载] {url}")
        print("文件大小约 237MB，请耐心等待...")
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=_download_progress)
            print("\n[完成] 下载完成")
        except Exception as e:
            print(f"\n[错误] 下载失败: {e}")
            print("可能是网络问题，请尝试使用 Hugging Face 方式或手动下载")
            return False

    # 解压
    print(f"\n[解压] {zip_path}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print("[完成] 解压完成")
    except Exception as e:
        print(f"[错误] 解压失败: {e}")
        return False

    if not _convert_stanford_val(extract_dir):
        return False

    # 清理
    zip_path.unlink()
    print(f"[清理] 已删除 {zip_path}")

    return True


def _convert_stanford_val(dataset_dir: Path) -> bool:
    """将 Stanford 官方验证集整理成 ImageFolder 目录结构。"""
    val_dir = dataset_dir / "val"
    annotations_path = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"
    if not annotations_path.exists() or not images_dir.exists():
        return True

    try:
        for line in annotations_path.read_text().splitlines():
            image_name, class_name, *_ = line.split("\t")
            src = images_dir / image_name
            if not src.exists():
                continue
            dst_dir = val_dir / class_name
            dst_dir.mkdir(exist_ok=True)
            shutil.move(str(src), dst_dir / image_name)
        shutil.rmtree(images_dir)
    except Exception as e:
        print(f"[错误] 验证集格式转换失败: {e}")
        return False

    print("[完成] 验证集已转换为 ImageFolder 格式")
    return True


def download_from_huggingface(data_dir: Path) -> bool:
    """从 Hugging Face datasets 下载 Tiny-ImageNet。

    Returns:
        True 如果下载成功
    """
    print("=" * 60)
    print("从 Hugging Face 下载 Tiny-ImageNet")
    print("=" * 60)

    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("[错误] 需要安装 datasets 和 pillow")
        print("  pip install datasets pillow")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / "tiny-imagenet-200"

    if output_dir.exists() and (output_dir / "train").exists():
        print(f"[跳过] 数据集已存在: {output_dir}")
        return True

    print("[下载] 加载 zh-plus/tiny-imagenet 数据集...")
    print("这可能需要几分钟...")

    try:
        dataset = load_dataset("zh-plus/tiny-imagenet", trust_remote_code=True)
    except Exception as e:
        print(f"[错误] 下载失败: {e}")
        return False

    print("\n[转换] 转换为 ImageFolder 格式...")

    # 获取类别映射
    label_names = dataset["train"].features["label"].names

    for split in ["train", "valid"]:
        split_name = "train" if split == "train" else "val"
        split_dir = output_dir / split_name

        print(f"  处理 {split_name}...")
        for i, example in enumerate(dataset[split]):
            label = example["label"]
            label_name = label_names[label]

            # 创建类别目录
            class_dir = split_dir / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # 保存图片
            img: Image.Image = example["image"]
            img_path = class_dir / f"{i:06d}.JPEG"
            img.save(img_path, "JPEG")

            if (i + 1) % 1000 == 0:
                print(f"    已处理 {i + 1} 张图片...")

    print(f"\n[完成] 数据集已保存到 {output_dir}")
    return True


def create_tiny_subset(
    data_dir: Path,
    num_classes: int = 10,
    images_per_class: int = 20,
) -> bool:
    """从完整数据集创建小型测试子集。

    Args:
        data_dir: 数据目录
        num_classes: 要提取的类别数
        images_per_class: 每个类别的图片数

    Returns:
        True 如果成功
    """
    import random

    full_dir = data_dir / "tiny-imagenet-200"
    subset_dir = data_dir / "tiny_imagenet_local"

    if not full_dir.exists():
        print("[错误] 请先下载完整数据集")
        return False

    print("=" * 60)
    print(f"创建测试子集 ({num_classes} 类 × {images_per_class} 张)")
    print("=" * 60)

    random.seed(42)

    for split in ["train", "val"]:
        src_split = full_dir / split
        dst_split = subset_dir / split

        if not src_split.exists():
            continue

        # 选择类别
        all_classes = sorted([d for d in src_split.iterdir() if d.is_dir()])
        selected_classes = random.sample(
            all_classes,
            min(num_classes, len(all_classes)),
        )

        for class_dir in selected_classes:
            src_class = src_split / class_dir.name
            dst_class = dst_split / class_dir.name
            dst_class.mkdir(parents=True, exist_ok=True)

            # 复制图片
            images = list(src_class.glob("*.JPEG")) + list(src_class.glob("*.jpg"))
            selected_images = random.sample(images, min(images_per_class, len(images)))

            for img in selected_images:
                shutil.copy(img, dst_class / img.name)

        print(f"  {split}: 已创建")

    print(f"\n[完成] 子集已保存到 {subset_dir}")
    return True


def _download_progress(count: int, block_size: int, total_size: int) -> None:
    """下载进度回调"""
    percent = min(100, int(count * block_size * 100 / total_size))
    bar_len = 40
    filled = int(bar_len * percent / 100)
    bar = "=" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {percent}% ({count * block_size / 1024 / 1024:.1f}MB)", end="")


def print_manual_instructions() -> None:
    """打印手动下载指引"""
    print("=" * 60)
    print("手动下载 Tiny-ImageNet")
    print("=" * 60)
    print("""
方法 1: Stanford 官方
  URL: http://cs231n.stanford.edu/tiny-imagenet-200.zip
  大小: ~237MB

方法 2: Kaggle
  1. 访问 https://www.kaggle.com/c/tiny-imagenet
  2. 登录并接受规则
  3. 下载数据

方法 3: 学术镜像
  - 搜索 "tiny-imagenet-200 download"

下载后:
  1. 解压到 data/tiny-imagenet-200/
  2. 目录结构应为:
     data/tiny-imagenet-200/
     ├── train/
     │   ├── n01443537/
     │   │   ├── n01443537_0.JPEG
     │   │   └── ...
     │   └── ...
     └── val/
         └── ...
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Tiny-ImageNet 数据集")
    parser.add_argument(
        "--method",
        choices=["stanford", "huggingface", "subset", "manual"],
        default="stanford",
        help="下载方式 (default: stanford)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="数据保存目录 (default: data)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="子集类别数 (仅用于 subset 方法)",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=20,
        help="子集每类图片数 (仅用于 subset 方法)",
    )

    args = parser.parse_args()

    print(f"数据目录: {args.data_dir.absolute()}\n")

    if args.method == "stanford":
        success = download_from_stanford(args.data_dir)
    elif args.method == "huggingface":
        success = download_from_huggingface(args.data_dir)
    elif args.method == "subset":
        success = create_tiny_subset(
            args.data_dir, args.num_classes, args.images_per_class
        )
    elif args.method == "manual":
        print_manual_instructions()
        success = True

    if success:
        print("\n" + "=" * 60)
        print("完成!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("失败，请尝试其他方式")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
