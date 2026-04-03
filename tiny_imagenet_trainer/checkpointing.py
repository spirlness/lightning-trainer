"""Checkpoint saving and loading utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tiny_imagenet_trainer.config import TrainingConfig


@dataclass
class Checkpoint:
    """Training checkpoint data."""
    epoch: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any] | None
    config: dict[str, Any]
    metrics: dict[str, Any]

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "Checkpoint":
        """Load checkpoint from disk."""
        data = torch.load(path, map_location=device)
        return cls(**data)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """Save a training checkpoint.

    Args:
        path: Destination file path
        model: Model to save
        optimizer: Optimizer to save
        scheduler: LR scheduler to save
        scaler: GradScaler for AMP to save
        config: Training configuration
        epoch: Current epoch number
        metrics: Current metrics dictionary
    """
    checkpoint = Checkpoint(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        scaler_state_dict=scaler.state_dict() if scaler.is_enabled() else None,
        config=config.to_dict(),
        metrics=metrics,
    )
    checkpoint.save(path)
