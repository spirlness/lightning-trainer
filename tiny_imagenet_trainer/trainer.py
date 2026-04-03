from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny_imagenet_trainer.checkpointing import save_checkpoint
from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.environment import write_json


class Trainer:
    """Own the core training loop, validation, and checkpointing."""

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
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_paths = run_paths
        self.device = torch.device(device)
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = _build_scheduler(self.optimizer, config, len(train_loader) * config.num_epochs)
        self.autocast_enabled, self.autocast_dtype, scaler_enabled = _resolve_precision(
            config,
            self.device,
        )
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=scaler_enabled)

        self.best_val_loss: float | None = None
        self.global_step = 0
        self.history: list[dict[str, Any]] = []

        self.logger.info("Using device: %s", self.device)
        self.logger.info(
            "AMP enabled: %s%s",
            self.autocast_enabled,
            f" ({self.autocast_dtype})" if self.autocast_enabled else "",
        )

    def train(self) -> list[dict[str, Any]]:
        self.logger.info("Starting training for %s epochs", self.config.num_epochs)

        for epoch_index in range(self.config.num_epochs):
            start_time = time.perf_counter()
            train_metrics = self._train_one_epoch(epoch_index)
            val_metrics = self.evaluate()

            epoch_metrics = {
                "epoch": epoch_index + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "epoch_seconds": round(time.perf_counter() - start_time, 2),
            }

            self.history.append(epoch_metrics)
            write_json(self.run_paths.history_file, self.history)
            self.logger.info(
                "Epoch %s summary | train_loss=%.4f | train_acc=%.2f%% | val_loss=%.4f | val_acc=%.2f%% | time=%.2fs",
                epoch_metrics["epoch"],
                epoch_metrics["train_loss"],
                epoch_metrics["train_accuracy"] * 100,
                epoch_metrics["val_loss"],
                epoch_metrics["val_accuracy"] * 100,
                epoch_metrics["epoch_seconds"],
            )

            self._save_last_checkpoint(epoch_metrics)
            if self.best_val_loss is None or epoch_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = epoch_metrics["val_loss"]
                self._save_best_checkpoint(epoch_metrics)

        return self.history

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        batches_processed = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.autocast_dtype,
                    enabled=self.autocast_enabled,
                ):
                    logits = self.model(inputs)
                    loss = self.criterion(logits, labels)

                loss_sum += loss.item()
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                batches_processed += 1

        self.model.train()
        return {
            "loss": loss_sum / max(batches_processed, 1),
            "accuracy": correct / max(total, 1),
        }

    def _train_one_epoch(self, epoch_index: int) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        batches_processed = 0

        for batch in self.train_loader:
            inputs = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_enabled,
            ):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)

            grad_norm = self._backward_and_step(loss)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item()
            batches_processed += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                self.logger.info(
                    "Epoch %s | step %s | loss=%.4f | acc=%.2f%% | grad_norm=%s | lr=%.2e",
                    epoch_index + 1,
                    self.global_step,
                    loss.item(),
                    ((predictions == labels).float().mean().item() * 100),
                    _format_optional_metric(grad_norm),
                    self.optimizer.param_groups[0]["lr"],
                )

        return {
            "loss": loss_sum / max(batches_processed, 1),
            "accuracy": correct / max(total, 1),
        }

    def _backward_and_step(self, loss: torch.Tensor) -> float | None:
        grad_norm = None
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()

        self.scheduler.step()
        return float(grad_norm) if grad_norm is not None else None

    def _save_last_checkpoint(self, epoch_metrics: dict[str, Any]) -> None:
        save_checkpoint(
            self.run_paths.checkpoints_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            epoch=epoch_metrics["epoch"],
            metrics=epoch_metrics,
        )

    def _save_best_checkpoint(self, epoch_metrics: dict[str, Any]) -> None:
        save_checkpoint(
            self.run_paths.checkpoints_dir / "best.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            epoch=epoch_metrics["epoch"],
            metrics=epoch_metrics,
        )


def _resolve_precision(
    config: TrainingConfig,
    device: torch.device,
) -> tuple[bool, torch.dtype | None, bool]:
    if not config.enable_amp or device.type != "cuda":
        return False, None, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False
    return True, torch.float16, True


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps <= 1:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    warmup_steps = min(config.warmup_steps, total_steps - 1)
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_steps - warmup_steps, 1),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
