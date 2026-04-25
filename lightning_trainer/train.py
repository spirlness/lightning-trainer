"""极简训练入口 - CLI."""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from lightning_trainer.data import TinyImageNetDataModule
from lightning_trainer.model import ImageClassifier


def setup_msvc() -> None:
    """设置 Windows MSVC 环境 (torch.compile inductor 需要)

    从环境变量读取配置:
    - MSVC_PATH: MSVC 安装路径
    - MSVC_VERSION: MSVC 版本号 (默认自动检测)
    - WINDOWS_SDK_PATH: Windows SDK 路径
    """
    if sys.platform != "win32":
        return

    # 从环境变量获取路径，支持用户自定义
    vs_path = os.environ.get(
        "MSVC_PATH",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
    )
    sdk_path = os.environ.get(
        "WINDOWS_SDK_PATH",
        r"C:\Program Files (x86)\Windows Kits\10",
    )

    # 自动检测 MSVC 版本
    msvc_base = f"{vs_path}\\VC\\Tools\\MSVC"
    msvc_version = os.environ.get("MSVC_VERSION")
    if not msvc_version and os.path.exists(msvc_base):
        versions = [
            d
            for d in os.listdir(msvc_base)
            if os.path.isdir(os.path.join(msvc_base, d))
        ]
        if versions:
            msvc_version = sorted(
                versions,
                key=lambda v: [int(x) if x.isdigit() else x for x in v.split(".")],
            )[-1]
    if not msvc_version:
        print("[Warning] 无法检测 MSVC 版本，torch.compile 可能失败")
        return

    msvc_path = f"{vs_path}\\VC\\Tools\\MSVC\\{msvc_version}"

    # 找到 SDK 版本
    sdk_include_base = f"{sdk_path}\\Include"
    sdk_version = "10.0.26100.0"
    if os.path.exists(sdk_include_base):
        versions = [
            d
            for d in os.listdir(sdk_include_base)
            if os.path.isdir(os.path.join(sdk_include_base, d))
        ]
        if versions:
            sdk_version = sorted(versions)[-1]

    # 设置环境变量
    include_paths = [
        f"{msvc_path}\\include",
        f"{sdk_path}\\Include\\{sdk_version}\\cppwinrt",
        f"{sdk_path}\\Include\\{sdk_version}\\shared",
        f"{sdk_path}\\Include\\{sdk_version}\\um",
        f"{sdk_path}\\Include\\{sdk_version}\\winrt",
        f"{sdk_path}\\Include\\{sdk_version}\\ucrt",
    ]
    valid_includes = [p for p in include_paths if os.path.exists(p)]
    if valid_includes:
        os.environ["INCLUDE"] = ";".join(valid_includes)

    lib_paths = [
        f"{msvc_path}\\lib\\x64",
        f"{sdk_path}\\Lib\\{sdk_version}\\um\\x64",
        f"{sdk_path}\\Lib\\{sdk_version}\\ucrt\\x64",
    ]
    valid_libs = [p for p in lib_paths if os.path.exists(p)]
    if valid_libs:
        os.environ["LIB"] = ";".join(valid_libs)
        os.environ["LIBPATH"] = f"{msvc_path}\\lib\\x64"

    cl_path = f"{msvc_path}\\bin\\Hostx64\\x64"
    if os.path.exists(cl_path):
        os.environ["PATH"] = cl_path + os.pathsep + os.environ.get("PATH", "")


def main() -> None:
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(
        description="Tiny-ImageNet 训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据参数
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tiny_imagenet_local",
        help="ImageFolder 数据目录",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/tiny_imagenet_cache_128",
        help="tensor 缓存目录；传空字符串可禁用缓存",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小")
    parser.add_argument("--max-epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--no-compile", action="store_true", help="禁用 torch.compile")
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="使用预训练权重",
    )

    args = parser.parse_args()

    compile_model = not args.no_compile
    if compile_model:
        setup_msvc()

    data_module = TinyImageNetDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir or None,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=128,
    )
    data_module.prepare_data()
    data_module.setup("fit")
    num_classes = data_module.num_classes

    model = ImageClassifier(
        num_classes=num_classes,
        lr=1e-4,
        max_epochs=args.max_epochs,
        compile_model=compile_model,
        use_fused_optimizer=True,
        pretrained=args.pretrained,
    )

    logger = CSVLogger("outputs", name="lightning_trainer")
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename="best-{epoch:02d}-{val_acc:.4f}",
            save_last=True,
            save_top_k=1,
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        default_root_dir="outputs",
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
