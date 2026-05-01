import json
import sys
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from PIL import Image

from lightning_trainer.data import TinyImageNetDataModule
from lightning_trainer.model import ImageClassifier, ImageClassifierConfig
from lightning_trainer.train import main


def make_imagefolder(root: Path, classes: list[str] | None = None) -> Path:
    classes = classes or ["class_a", "class_b"]
    for split in ["train", "val"]:
        for class_index, class_name in enumerate(classes):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(2):
                image = Image.new(
                    "RGB",
                    (32, 32),
                    color=(class_index * 80, image_index * 80, 128),
                )
                image.save(class_dir / f"{image_index}.jpg")
    return root


def test_datamodule_builds_loaders(tmp_path: Path) -> None:
    data_dir = make_imagefolder(tmp_path / "data")
    module = TinyImageNetDataModule(
        data_dir=data_dir,
        batch_size=2,
        image_size=32,
        num_workers=0,
    )

    module.setup("fit")
    images, labels = next(iter(module.train_dataloader()))

    assert module.num_classes == 2
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)


def test_train_dataloader_drops_last_batch(tmp_path: Path) -> None:
    data_dir = make_imagefolder(tmp_path / "data")
    module = TinyImageNetDataModule(
        data_dir=data_dir,
        batch_size=3,
        image_size=32,
        num_workers=0,
    )

    module.setup("fit")

    assert len(module.train_dataloader()) == 1
    assert len(module.val_dataloader()) == 2


def write_cache_split(
    cache_dir: Path,
    split: str,
    classes: list[str],
    image_size: int,
    num_samples: int,
) -> None:
    split_dir = cache_dir / split
    split_dir.mkdir(parents=True)
    images = torch.randint(
        0,
        256,
        (num_samples, 3, image_size, image_size),
        dtype=torch.uint8,
    )
    labels = torch.arange(num_samples, dtype=torch.long) % len(classes)
    (split_dir / "images.bin").write_bytes(images.numpy().tobytes())
    torch.save(labels, split_dir / "labels.pt")
    (split_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "uint8_chw_bin_v1",
                "split": split,
                "num_samples": num_samples,
                "image_size": image_size,
                "classes": classes,
            }
        ),
        encoding="utf-8",
    )


def test_datamodule_reads_tensor_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    classes = ["class_a", "class_b"]
    write_cache_split(cache_dir, "train", classes, image_size=32, num_samples=4)
    write_cache_split(cache_dir, "val", classes, image_size=32, num_samples=4)
    module = TinyImageNetDataModule(
        data_dir=tmp_path / "unused",
        cache_dir=cache_dir,
        batch_size=2,
        image_size=32,
        num_workers=0,
    )

    module.setup("fit")
    images, labels = next(iter(module.train_dataloader()))

    assert module.num_classes == 2
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)


def test_datamodule_cached_val_dataloader(tmp_path: Path) -> None:
    """Validate cached val_dataloader (regression: val_transform was None)."""
    cache_dir = tmp_path / "cache"
    classes = ["class_a", "class_b"]
    write_cache_split(cache_dir, "train", classes, image_size=32, num_samples=4)
    write_cache_split(cache_dir, "val", classes, image_size=32, num_samples=4)
    module = TinyImageNetDataModule(
        data_dir=tmp_path / "unused",
        cache_dir=cache_dir,
        batch_size=2,
        image_size=32,
        num_workers=0,
    )

    module.setup("fit")
    images, labels = next(iter(module.val_dataloader()))

    assert module.num_classes == 2
    assert images.shape == (2, 3, 32, 32)
    assert labels.shape == (2,)


def test_normalization_rejects_double_div(tmp_path: Path) -> None:
    """Verify cached path produces same output range (regression: double div_255)."""
    import json

    import torch.nn as nn

    from lightning_trainer.data import CachedTensorDataset

    # Build a tiny cached dataset with known uint8 values
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    images_tensor = torch.tensor([[[[100]], [[150]], [[200]]]], dtype=torch.uint8)
    labels_tensor = torch.zeros(1, dtype=torch.long)
    (cache_dir / "images.bin").write_bytes(images_tensor.numpy().tobytes())
    torch.save(labels_tensor, cache_dir / "labels.pt")
    (cache_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "uint8_chw_bin_v1",
                "split": "val",
                "num_samples": 1,
                "image_size": 1,
                "classes": ["class_a"],
            }
        ),
        encoding="utf-8",
    )

    dataset = CachedTensorDataset(cache_dir, nn.Identity())
    image, _ = dataset[0]

    # After fix: value is float32 [100, 200] (unchanged from uint8),
    # NOT [0.39, 0.78] (div_255 applied)
    assert image.dtype == torch.float32
    assert image[0, 0, 0].item() == 100.0
    assert image[1, 0, 0].item() == 150.0
    assert image[2, 0, 0].item() == 200.0


def test_datamodule_rejects_mismatched_classes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    make_imagefolder(data_dir)
    extra_val_class = data_dir / "val" / "class_c"
    extra_val_class.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(extra_val_class / "0.jpg")

    module = TinyImageNetDataModule(data_dir=data_dir, num_workers=0)

    with pytest.raises(ValueError, match="类别目录不一致"):
        module.setup("fit")


def test_image_classifier_forward() -> None:
    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=2,
            pretrained=False,
            compile_model=False,
            use_channels_last=False,
        )
    )

    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 2)


def test_test_step() -> None:
    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=2,
            pretrained=False,
            compile_model=False,
            use_channels_last=False,
        )
    )

    # Mock log to avoid missing trainer errors
    from unittest.mock import MagicMock

    model.log = MagicMock()

    batch = (torch.randn(2, 3, 32, 32), torch.randint(0, 2, (2,)))
    model.test_step(batch, 0)


def test_gradient_checkpointing_flag_does_not_crash_for_torchvision() -> None:
    with pytest.warns(UserWarning, match="not supported"):
        ImageClassifier(
            ImageClassifierConfig(
                pretrained=False,
                compile_model=False,
                use_gradient_checkpointing=True,
            )
        )


def test_lightning_cpu_smoke_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    data_dir = make_imagefolder(tmp_path / "data")
    module = TinyImageNetDataModule(
        data_dir=data_dir,
        batch_size=2,
        image_size=32,
        num_workers=0,
    )
    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=2,
            pretrained=False,
            compile_model=False,
            use_channels_last=False,
        )
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )

    trainer.fit(model, module)

    assert trainer.state.finished


def test_checkpointing(tmp_path: Path) -> None:
    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=2,
            pretrained=False,
            compile_model=False,
            use_channels_last=False,
        )
    )
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=True,
        default_root_dir=tmp_path,
    )
    # create a fake batch to save checkpoint without training
    trainer.strategy.connect(model)
    trainer.save_checkpoint(tmp_path / "checkpoint.ckpt")

    loaded_model = ImageClassifier.load_from_checkpoint(tmp_path / "checkpoint.ckpt")
    assert loaded_model.config.num_classes == 2


@pytest.mark.parametrize(
    "use_fused, is_available",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_configure_optimizers(
    monkeypatch: pytest.MonkeyPatch,
    use_fused: bool,
    is_available: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: is_available)

    model = ImageClassifier(
        ImageClassifierConfig(
            num_classes=2,
            pretrained=False,
            compile_model=False,
            use_fused_optimizer=use_fused,
        )
    )

    optim_config = model.configure_optimizers()

    # Assert return types
    assert isinstance(optim_config, dict)
    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config

    opt = optim_config["optimizer"]
    sched_dict = optim_config["lr_scheduler"]

    assert isinstance(opt, torch.optim.AdamW)
    assert isinstance(sched_dict, dict)
    assert isinstance(
        sched_dict["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR
    )
    assert sched_dict["interval"] == "epoch"

    # Assert fused branch is correctly covered
    if use_fused and is_available:
        assert opt.defaults.get("fused") is True
    else:
        assert opt.defaults.get("fused") is not True


def test_main_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = make_imagefolder(tmp_path / "data")

    # Mock CLI arguments
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data-dir",
            str(data_dir),
            "--cache-dir",
            "",
            "--batch-size",
            "2",
            "--max-epochs",
            "1",
            "--no-compile",
        ],
    )

    # Ensure outputs are written to the temp path
    monkeypatch.chdir(tmp_path)

    # Force CPU to make tests reliable and fast
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    # Call the main function
    main()

    # Verify outputs were created
    outputs_dir = tmp_path / "outputs"
    assert outputs_dir.exists()
    assert (outputs_dir / "lightning_trainer").exists()
