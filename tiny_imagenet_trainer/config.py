"""配置管理与命令行参数解析模块。
本模块定义了训练所需的全部超参数，并支持从命令行自动生成解析器。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints


@dataclass
class TrainingConfig:
    """训练配置类，包含所有超参数及其验证逻辑。"""

    # 基础路径配置
    data_dir: Path = Path("data/tiny_imagenet_local")  # 数据集根目录
    output_root: Path = Path("outputs")  # 训练结果保存根目录
    experiment_name: str = "convnext_tiny_baseline"  # 实验名称，用于区分不同的运行

    # 模型与训练超参数
    model_name: str = "convnext_tiny"  # 模型架构: 'resnet18' 或 'convnext_tiny'
    num_classes: int = 200  # 分类类别数 (Tiny-ImageNet 默认为 200)
    image_size: int = 224  # 输入模型图像的尺寸
    batch_size: int = 64  # 每个批次的样本数 (由于 ConvNeXt 显存占用大，调小至 64)
    num_epochs: int = 1  # 总训练轮数
    learning_rate: float = 1e-4  # 学习率
    weight_decay: float = 1e-4  # 权重衰减 (L2 正则化)
    warmup_steps: int = 50  # 学习率预热步数
    gradient_clip_norm: float | None = 1.0  # 梯度裁剪阈值，防止梯度爆炸
    label_smoothing: float = 0.1  # 标签平滑系数，防止过拟合

    # 高级训练技术
    use_mixup: bool = True  # 启用 MixUp + CutMix 数据混合增强
    mixup_alpha: float = 0.2  # MixUp 混合强度
    cutmix_alpha: float = 1.0  # CutMix 混合强度
    use_ema: bool = True  # 启用指数移动平均 (EMA)
    ema_decay: float = 0.999  # EMA 衰减率
    use_compile: bool = True  # 启用 torch.compile (Windows 兼容性有限)
    use_channels_last: bool = True  # 启用 channels_last 内存格式加速卷积

    # 运行环境配置
    num_workers: int = 2  # 数据加载的线程数
    log_every_n_steps: int = 10  # 每隔多少个 step 记录一次日志
    seed: int = 42  # 随机种子，确保实验可复现
    device: str = "cuda"  # 运行设备: 'cuda', 'cpu' 或 'auto'
    pretrained: bool = False  # 是否加载 ImageNet 预训练权重
    enable_amp: bool = True  # 是否启用自动混合精度训练

    def __post_init__(self) -> None:
        """在 dataclass 初始化后执行的验证逻辑。"""
        self.output_root = Path(self.output_root)
        self.data_dir = Path(self.data_dir)

        # 验证设备与模型类型
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Invalid device: {self.device}")
        if self.model_name not in {"resnet18", "convnext_tiny"}:
            raise ValueError(f"Invalid model_name: {self.model_name}")

        # 数值合法性规则检查 (错误消息保持英文，便于调试和测试)
        rules = [
            (self.num_classes > 0, "num_classes must be positive"),
            (self.image_size > 0, "image_size must be positive"),
            (self.batch_size > 0, "batch_size must be positive"),
            (self.num_epochs > 0, "num_epochs must be positive"),
            (self.warmup_steps >= 0, "warmup_steps must be non-negative"),
            (self.num_workers >= 0, "num_workers must be non-negative"),
            (self.log_every_n_steps > 0, "log_every_n_steps must be positive"),
            (self.learning_rate > 0, "learning_rate must be positive"),
            (self.weight_decay >= 0, "weight_decay must be non-negative"),
            (0.0 <= self.label_smoothing < 1.0, "label_smoothing must be in [0, 1)"),
            (0.0 < self.ema_decay < 1.0, "ema_decay must be in (0, 1)"),
        ]

        for condition, message in rules:
            if not condition:
                raise ValueError(message)

    def to_dict(self) -> dict[str, Any]:
        """将配置转换为字典格式，并将 Path 对象序列化为字符串，便于保存为 JSON。"""
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }


def build_parser() -> argparse.ArgumentParser:
    """根据 TrainingConfig 的字段自动生成命令行参数解析器。
    该方法利用类型提示 (Type Hints) 自动映射 argparse 的 type 和 default。
    """
    parser = argparse.ArgumentParser(
        prog="tiny-imagenet-train",
        description="基于 ResNet18 的 Tiny-ImageNet 分类训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    defaults = TrainingConfig()
    hints = get_type_hints(TrainingConfig)

    for f in fields(TrainingConfig):
        # 将 python 风格的变量名 (batch_size) 转换为 CLI 风格 (--batch-size)
        name = f"--{f.name.replace('_', '-')}"
        tp = hints[f.name]
        default_val = getattr(defaults, f.name)

        # 提取 Union 类型中的参数 (例如处理 Optional[int])
        args = getattr(tp, "__args__", ())

        is_optional = False
        actual_type = tp
        if type(None) in args:
            is_optional = True
            actual_type = next(t for t in args if t is not type(None))

        # 特殊处理 choices
        if f.name == "device":
            parser.add_argument(
                name,
                type=str,
                choices=["auto", "cpu", "cuda"],
                default=default_val,
                help="运行计算的设备",
            )
        elif f.name == "model_name":
            parser.add_argument(
                name,
                type=str,
                choices=["resnet18", "convnext_tiny"],
                default=default_val,
                help="模型架构",
            )
        # 特殊处理布尔值：生成 --flag 和 --no-flag 两个选项
        elif actual_type is bool:
            dest = f.name
            parser.add_argument(
                name, dest=dest, action="store_true", default=default_val
            )
            parser.add_argument(
                f"--no-{f.name.replace('_', '-')}",
                dest=dest,
                action="store_false",
                help=argparse.SUPPRESS,
            )
        # 处理路径
        elif actual_type is Path:
            parser.add_argument(name, type=Path, default=default_val)
        # 处理可选类型 (Optional)
        elif is_optional:

            def optional_type(val: str, _type=actual_type) -> Any:
                if val.lower() == "none":
                    return None
                return _type(val)

            parser.add_argument(name, type=optional_type, default=default_val)
        # 普通数值和字符串
        else:
            parser.add_argument(name, type=actual_type, default=default_val)

    return parser


def parse_args(args: list[str] | None = None) -> TrainingConfig:
    """解析命令行参数并返回填充好的 TrainingConfig 对象。"""
    parser = build_parser()
    parsed_args = parser.parse_args(args)
    return TrainingConfig(**vars(parsed_args))
