from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from tiny_imagenet_trainer.config import TrainingConfig


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    state: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "config": config.to_dict(),
            "state": state,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restore_warnings: list[str] = []
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:
            restore_warnings.append(f"scheduler state was not restored: {exc}")
    if scaler and checkpoint.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as exc:
            restore_warnings.append(f"amp scaler state was not restored: {exc}")

    checkpoint["_restore_warnings"] = restore_warnings
    return checkpoint


def cleanup_old_epoch_checkpoints(checkpoints_dir: Path, keep_last_n: int) -> None:
    epoch_checkpoints = sorted(checkpoints_dir.glob("epoch_*.pt"))
    while len(epoch_checkpoints) > keep_last_n:
        oldest = epoch_checkpoints.pop(0)
        oldest.unlink(missing_ok=True)
