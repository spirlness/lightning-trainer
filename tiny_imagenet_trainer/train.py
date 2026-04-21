"""模型训练核心逻辑。
包含训练循环、验证逻辑、模型构建以及 CLI 入口。
"""

import json
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
)
from torchvision.transforms import v2

from tiny_imagenet_trainer.config import TrainingConfig, parse_args
from tiny_imagenet_trainer.data import build_dataloaders

# 抑制 SequentialLR 内部传递 epoch 参数的弃用警告 (PyTorch 已知问题)
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step")


def setup_logger(output_dir: Path) -> logging.Logger:
    """配置结构化日志。
    同时将日志输出到控制台和文件。
    """
    logger = logging.getLogger("tiny_imagenet")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件处理器
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    """设置全局随机种子，确保实验结果的可复现性。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True  # 固定输入尺寸时自动选择最优卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+ GPU 矩阵乘法加速
        torch.backends.cudnn.allow_tf32 = True  # Ampere+ GPU 卷积加速


def select_device(config: TrainingConfig) -> torch.device:
    """根据配置选择执行设备。"""
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


class MetricTracker:
    """指标追踪器，用于计算每个 epoch 的平均 Loss 和准确率。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_loss = 0.0
        self.correct = 0
        self.total = 0

    def update(self, loss: float, preds: torch.Tensor, targets: torch.Tensor) -> None:
        batch_size = targets.size(0)
        self.total_loss += loss * batch_size
        self.correct += (preds.argmax(dim=1) == targets).sum().item()
        self.total += batch_size

    @property
    def avg_loss(self) -> float:
        """返回平均 Loss。"""
        return self.total_loss / max(self.total, 1)

    @property
    def accuracy(self) -> float:
        """返回准确率 (百分比)。"""
        return self.correct / max(self.total, 1) * 100.0


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    val_acc: float,
    config: TrainingConfig,
) -> None:
    """保存模型检查点到磁盘。"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_acc": val_acc,
        "config": config.to_dict(),
    }
    torch.save(checkpoint, path)


def build_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """根据配置构建模型并移动到目标设备。"""
    if config.model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None
        model = convnext_tiny(weights=weights)
        # ConvNeXt 的分类头: classifier[2] 是最终的线性层
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, config.num_classes
        )
    else:  # resnet18
        weights = ResNet18_Weights.DEFAULT if config.pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)

    # channels_last 内存格式: NHWC 布局让卷积更好利用 Tensor Core
    memory_format = (
        torch.channels_last if config.use_channels_last else torch.contiguous_format
    )
    model = model.to(device, memory_format=memory_format)

    # torch.compile: 图编译优化 (算子融合、内存优化)
    if config.use_compile:
        model = torch.compile(model)

    return model


def train_model(config: TrainingConfig) -> dict[str, Any]:
    """主训练流程编排。"""
    # 1. 环境准备
    set_seed(config.seed)
    device = select_device(config)

    # 2. 目录与日志准备
    run_dir = (
        config.output_root / config.experiment_name / time.strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir)

    logger.info(f"开始实验: {config.experiment_name}")
    logger.info(f"计算设备: {device} | 模型: {config.model_name}")

    # 保存当前的配置信息
    with open(run_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    # 3. 构建数据加载器
    logger.info("构建数据加载器...")
    train_loader, val_loader = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
    )

    # 4. 构建模型
    logger.info(f"构建 {config.model_name} 模型...")
    model = build_model(config, device)
    if config.use_channels_last:
        logger.info("channels_last 内存格式已启用")
    if config.use_compile:
        logger.info("torch.compile 图编译已启用")

    # 5. 设置优化器与损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    logger.info(f"标签平滑: {config.label_smoothing}")

    # 6. MixUp + CutMix 数据混合增强
    mix_transform = None
    if config.use_mixup:
        mixup = v2.MixUp(alpha=config.mixup_alpha, num_classes=config.num_classes)
        cutmix = v2.CutMix(alpha=config.cutmix_alpha, num_classes=config.num_classes)
        mix_transform = v2.RandomChoice([mixup, cutmix])
        logger.info(
            f"MixUp (α={config.mixup_alpha}) + CutMix (α={config.cutmix_alpha}) 已启用"
        )

    # 7. EMA 指数移动平均
    ema_model = None
    if config.use_ema:
        ema_model = AveragedModel(
            model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay)
        )
        logger.info(f"EMA 已启用 (衰减率: {config.ema_decay})")

    # 8. 配置自动混合精度 (AMP)
    use_amp = config.enable_amp and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    if use_amp:
        logger.info("自动混合精度 (AMP) 已启用")

    # 7. 配置学习率调度器 (Warmup + Cosine Annealing)
    total_steps = len(train_loader) * config.num_epochs
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - config.warmup_steps, 1)
    )
    scheduler = SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )
    logger.info(f"学习率调度: Warmup {config.warmup_steps} steps + Cosine Annealing")

    # 指标记录器
    train_tracker = MetricTracker()
    val_tracker = MetricTracker()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # 主训练循环
    memory_format = (
        torch.channels_last if config.use_channels_last else torch.contiguous_format
    )
    best_acc = 0.0
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.num_epochs} ---")

        # --- 训练阶段 ---
        model.train()
        train_tracker.reset()
        epoch_start_time = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, memory_format=memory_format, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # MixUp / CutMix 数据混合 (在前向传播前应用)
            if mix_transform is not None:
                images, targets = mix_transform(images, targets)

            optimizer.zero_grad(set_to_none=True)

            # 使用 autocast 进行前向传播
            with autocast(device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪 (防止梯度爆炸)
            if config.gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            # 更新权重与学习率
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # 更新 EMA 模型参数
            if ema_model is not None:
                ema_model.update_parameters(model)

            # 更新指标 (MixUp 下 targets 是软标签，用 argmax 还原)
            hard_targets = targets.argmax(dim=1) if targets.ndim > 1 else targets
            train_tracker.update(loss.item(), outputs.detach(), hard_targets)

            if step % config.log_every_n_steps == 0:
                logger.info(
                    f"Epoch [{epoch}/{config.num_epochs}], Step [{step}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, Acc: {train_tracker.accuracy:.2f}%"
                )

        # --- 验证阶段 (使用 EMA 模型进行推理，如果可用) ---
        eval_model = ema_model if ema_model is not None else model
        eval_model.eval()
        val_tracker.reset()

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(
                    device, memory_format=memory_format, non_blocking=True
                )
                targets = targets.to(device, non_blocking=True)

                with autocast(device.type, enabled=use_amp):
                    outputs = eval_model(images)
                    loss = criterion(outputs, targets)

                val_tracker.update(loss.item(), outputs, targets)

        # 8. Epoch 总结
        epoch_duration = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch} 总结: 训练 Loss: {train_tracker.avg_loss:.4f}, 训练 Acc: {train_tracker.accuracy:.2f}% | "
            f"验证 Loss: {val_tracker.avg_loss:.4f}, 验证 Acc: {val_tracker.accuracy:.2f}% | 耗时: {epoch_duration:.1f}s"
        )

        # 记录历史记录
        history["train_loss"].append(train_tracker.avg_loss)
        history["train_acc"].append(train_tracker.accuracy)
        history["val_loss"].append(val_tracker.avg_loss)
        history["val_acc"].append(val_tracker.accuracy)

        # 9. 保存最佳模型与最新模型
        is_best = val_tracker.accuracy > best_acc
        if is_best:
            best_acc = val_tracker.accuracy
            save_checkpoint(
                run_dir / "best_model.pth",
                epoch,
                model,
                optimizer,
                scaler,
                best_acc,
                config,
            )
            logger.info(f"保存新的最佳模型! (准确率: {best_acc:.2f}%)")

        save_checkpoint(
            run_dir / "latest_model.pth",
            epoch,
            model,
            optimizer,
            scaler,
            val_tracker.accuracy,
            config,
        )

    logger.info("训练完成。")
    return history


def main() -> None:
    """命令行入口函数。"""
    try:
        config = parse_args()
        train_model(config)
    except KeyboardInterrupt:
        print("\n[!] 用户中断了训练过程。", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
