import pytest
from tiny_imagenet_trainer.data import build_dataloaders, build_transforms


def test_build_transforms():
    train_tf = build_transforms(is_train=True)
    val_tf = build_transforms(is_train=False)
    assert train_tf is not None
    assert val_tf is not None


def test_build_dataloaders_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        build_dataloaders(tmp_path)


def test_build_dataloaders_success(tmp_path):
    # Setup dummy directory structure
    train_dir = tmp_path / "train" / "class0"
    val_dir = tmp_path / "val" / "class0"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    # Create dummy images
    from PIL import Image

    img = Image.new("RGB", (64, 64), color="red")
    img.save(train_dir / "img1.JPEG")
    img.save(val_dir / "img1.JPEG")

    train_loader, val_loader = build_dataloaders(
        data_dir=tmp_path, batch_size=1, num_workers=0
    )

    assert len(train_loader) == 1
    assert len(val_loader) == 1
