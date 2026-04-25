"""Prepare a memory-mapped tensor cache from an ImageFolder dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tiny-ImageNet tensor cache")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/tiny_imagenet_local"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/tiny_imagenet_cache_128"),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def process_sample(args: tuple[str, int, int]) -> tuple[int, bytes]:
    path, label, image_size = args
    tensor = convert_image(path, image_size)
    return label, tensor.numpy().tobytes()


def convert_image(path: str, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = F.resize(
            image,
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        return F.pil_to_tensor(image).contiguous()


def prepare_split(
    data_dir: Path,
    cache_dir: Path,
    split: str,
    image_size: int,
    overwrite: bool,
) -> None:
    source_dir = data_dir / split
    if not source_dir.exists():
        print(f"[skip] missing split: {source_dir}")
        return

    split_cache_dir = cache_dir / split
    if split_cache_dir.exists():
        if not overwrite:
            print(f"[skip] cache exists: {split_cache_dir}")
            return
        shutil.rmtree(split_cache_dir)
    split_cache_dir.mkdir(parents=True)

    dataset = ImageFolder(source_dir)
    labels = torch.empty(len(dataset.samples), dtype=torch.long)
    images_path = split_cache_dir / "images.bin"
    print(f"[build] {split}: {len(dataset.samples)} images -> {images_path}")

    with images_path.open("wb") as images_file:
        args_iter = ((path, label, image_size) for path, label in dataset.samples)
        torch.set_num_threads(1)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for index, (label, image_bytes) in enumerate(
                executor.map(process_sample, args_iter, chunksize=100)
            ):
                images_file.write(image_bytes)
                labels[index] = label
                if (index + 1) % 5000 == 0:
                    print(f"  {split}: {index + 1}/{len(dataset.samples)}")

    torch.save(labels, split_cache_dir / "labels.pt")
    manifest = {
        "format": "uint8_chw_bin_v1",
        "split": split,
        "num_samples": len(dataset.samples),
        "image_size": image_size,
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
    }
    (split_cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {split}: {split_cache_dir}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        prepare_split(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            split=split,
            image_size=args.image_size,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
