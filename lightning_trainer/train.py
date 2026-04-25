"""极简训练入口 - CLI."""

import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

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
            msvc_version = sorted(versions)[-1]
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
    parser = argparse.ArgumentParser(description="Tiny-ImageNet 训练")

    # 数据参数
    parser.add_argument("--data-dir", type=str, default="data/tiny-imagenet-200")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)

    # 模型参数
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=10)

    # 优化开关
    parser.add_argument("--compile", action="store_true", help="启用 torch.compile")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="启用模型支持时的梯度检查点",
    )
    parser.add_argument("--fused-optimizer", action="store_true", help="启用融合优化器")
    parser.add_argument("--no-pretrained", action="store_true", help="不使用预训练权重")
    parser.add_argument("--test", action="store_true", help="训练后运行测试集")

    args = parser.parse_args()

    # 设置 MSVC
    if args.compile:
        setup_msvc()

    # 数据模块
    data_module = TinyImageNetDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    # 预先 setup 获取类别数（Trainer.fit 时会再次调用，但 Lightning 会处理）
    data_module.prepare_data()
    data_module.setup("fit")
    num_classes = data_module.num_classes

    # 模型
    model = ImageClassifier(
        num_classes=num_classes,
        lr=args.lr,
        compile_model=args.compile,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_fused_optimizer=args.fused_optimizer,
        pretrained=not args.no_pretrained,
    )

    # 训练器
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                filename="best-{epoch:02d}-{val_acc:.2f}",
            ),
        ],
    )

    trainer.fit(model, data_module)

    # 测试集评估
    if args.test and data_module.test_dataloader() is not None:
        print("\n" + "=" * 50)
        print("测试集评估")
        print("=" * 50)
        trainer.test(model, data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
