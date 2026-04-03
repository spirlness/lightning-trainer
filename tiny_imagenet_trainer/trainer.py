"""Optimized trainer with clean abstractions inspired by timm & transformers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny_imagenet_trainer.checkpointing import save_checkpoint
from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.environment import write_json


@dataclass
class Metrics:
    """Accumulate metrics over batches."""
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0
    count: int = 0

    def update(self, loss: float, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss_sum += loss
        self.correct += (predictions == labels).sum().item()
        self.total += labels.size(0)
        self.count += 1

    @property
    def loss(self) -> float:
        return self.loss_sum / max(self.count, 1)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def as_dict(self) -> dict[str, float]:
        return {"loss": self.loss, "accuracy": self.accuracy}


@dataclass
class TrainState:
    """Track training state."""
    step: int = 0
    best_val_loss: float | None = None
    history: list[dict] = field(default_factory=list)

    def is_best(self, val_loss: float) -> bool:
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False


class Trainer:
    """Simplified trainer with single-responsibility components."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_paths: RunPaths,
        device: str | torch.device,
        logger,
    ) -> None:
        self.cfg = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.paths = run_paths
        self.device = torch.device(device)
        self.logger = logger

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = len(train_loader) * config.num_epochs
        self.sched = _build_scheduler(self.opt, config, total_steps)

        # AMP setup
        amp = _resolve_amp(config, self.device)
        self.amp_enabled = amp.enabled
        self.amp_dtype = amp.dtype
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=amp.scaler)

        # State
        self.state = TrainState()

        self._log_info()

    def _log_info(self) -> None:
        self.logger.info("Using device: %s", self.device)
        self.logger.info(
            "AMP enabled: %s%s",
            self.amp_enabled,
            f" ({self.amp_dtype})" if self.amp_enabled else "",
        )

    def train(self) -> list[dict[str, Any]]:
        self.logger.info("Starting training for %s epochs", self.cfg.num_epochs)

        for epoch in range(self.cfg.num_epochs):
            self._run_epoch(epoch)

        return self.state.history

    def _run_epoch(self, epoch: int) -> None:
        start = time.perf_counter()

        train_m = self._train_epoch(epoch)
        val_m = self.evaluate()

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "epoch_seconds": round(time.perf_counter() - start, 2),
        }

        self.state.history.append(metrics)
        write_json(self.paths.history_file, self.state.history)

        self.logger.info(
            "Epoch %s summary | train_loss=%.4f | train_acc=%.2f%% | val_loss=%.4f | val_acc=%.2f%% | time=%.2fs",
            metrics["epoch"],
            metrics["train_loss"],
            metrics["train_accuracy"] * 100,
            metrics["val_loss"],
            metrics["val_accuracy"] * 100,
            metrics["epoch_seconds"],
        )

        self._save_checkpoint("last", metrics)
        if self.state.is_best(metrics["val_loss"]):
            self._save_checkpoint("best", metrics)

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        m = Metrics()

        with torch.no_grad():
            for batch in self.val_loader:
                loss, preds, labels = self._forward(batch)
                m.update(loss.item(), preds, labels)

        self.model.train()
        return m.as_dict()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        m = Metrics()

        for batch in self.train_loader:
            loss, preds, labels = self._forward(batch)
            gn = self._backward(loss)

            m.update(loss.item(), preds, labels)
            self.state.step += 1

            if self.state.step % self.cfg.log_every_n_steps == 0:
                self._log_step(epoch, loss, preds, labels, gn)

        return m.as_dict()

    def _forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["pixel_values"].to(self.device, non_blocking=True)
        y = batch["label"].to(self.device, non_blocking=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            logits = self.model(x)
            loss = self.criterion(logits, y)

        return loss, logits.argmax(dim=1), y

    def _backward(self, loss: torch.Tensor) -> float | None:
        self.opt.zero_grad(set_to_none=True)

        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
        else:
            loss.backward()

        gn = None
        if self.cfg.gradient_clip_norm is not None:
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_norm)

        if self.scaler.is_enabled():
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

        self.sched.step()
        return float(gn) if gn is not None else None

    def _log_step(
        self,
        epoch: int,
        loss: torch.Tensor,
        preds: torch.Tensor,
        labels: torch.Tensor,
        grad_norm: float | None,
    ) -> None:
        self.logger.info(
            "Epoch %s | step %s | loss=%.4f | acc=%.2f%% | grad_norm=%s | lr=%.2e",
            epoch + 1,
            self.state.step,
            loss.item(),
            ((preds == labels).float().mean().item() * 100),
            _fmt(grad_norm),
            self.opt.param_groups[0]["lr"],
        )

    def _save_checkpoint(self, name: str, metrics: dict[str, Any]) -> None:
        save_checkpoint(
            self.paths.checkpoints_dir / f"{name}.pt",
            model=self.model,
            optimizer=self.opt,
            scheduler=self.sched,
            scaler=self.scaler,
            config=self.cfg,
            epoch=metrics["epoch"],
            metrics=metrics,
        )


@dataclass(frozen=True)
class AMPConfig:
    """AMP configuration."""
    enabled: bool
    dtype: torch.dtype | None
    scaler: bool


def _resolve_amp(config: TrainingConfig, device: torch.device) -> AMPConfig:
    """Resolve Automatic Mixed Precision configuration."""
    if not config.enable_amp or device.type != "cuda":
        return AMPConfig(enabled=False, dtype=None, scaler=False)
    if torch.cuda.is_bf16_supported():
        return AMPConfig(enabled=True, dtype=torch.bfloat16, scaler=False)
    return AMPConfig(enabled=True, dtype=torch.float16, scaler=True)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build learning rate scheduler with warmup and cosine annealing."""
    if total_steps <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    warmup_steps = min(config.warmup_steps, total_steps - 1)
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Warmup + Cosine annealing
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1)),
        ],
        milestones=[warmup_steps],
    )


def _fmt(value: float | None) -> str:
    """Format optional value."""
    return "n/a" if value is None else f"{value:.4f}"
