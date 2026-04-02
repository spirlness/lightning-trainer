from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from tiny_imagenet_trainer.checkpointing import (
    cleanup_old_epoch_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from tiny_imagenet_trainer.config import RunPaths, TrainingConfig
from tiny_imagenet_trainer.environment import write_json


@dataclass(slots=True)
class TrainerState:
    completed_epochs: int = 0
    global_step: int = 0
    best_val_loss: float | None = None
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_epochs": self.completed_epochs,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainerState":
        return cls(
            completed_epochs=payload.get("completed_epochs", 0),
            global_step=payload.get("global_step", 0),
            best_val_loss=payload.get("best_val_loss"),
            epochs_without_improvement=payload.get("epochs_without_improvement", 0),
            history=list(payload.get("history", [])),
        )


class Trainer:
    """Own the training loop, evaluation, and checkpoint lifecycle."""

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        run_paths: RunPaths,
        device: str | torch.device,
        logger,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_paths = run_paths
        self.device = torch.device(device)
        self.logger = logger

        self.criterion = nn.CrossEntropyLoss()
        self.raw_model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.raw_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        total_steps = _estimate_total_steps(config, train_loader)
        self.scheduler = _build_scheduler(self.optimizer, config, total_steps)
        self.autocast_enabled, self.autocast_dtype, scaler_enabled = _resolve_precision(
            config, self.device
        )
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=scaler_enabled)
        self.state = TrainerState()

        if config.resume_from:
            self._resume(config.resume_from)

        self.model = self.raw_model
        if config.compile_model:
            try:
                self.model = torch.compile(self.raw_model)
                self.logger.info("torch.compile enabled")
            except Exception as exc:
                self.logger.warning("torch.compile failed, falling back to eager mode: %s", exc)

        self.logger.info("Using device: %s", self.device)
        self.logger.info(
            "AMP enabled: %s%s",
            self.autocast_enabled,
            f" ({self.autocast_dtype})" if self.autocast_enabled else "",
        )

    def train(self) -> list[dict[str, Any]]:
        self.logger.info("Starting training for %s epochs", self.config.num_epochs)

        for epoch_index in range(self.state.completed_epochs, self.config.num_epochs):
            start_time = time.perf_counter()
            train_metrics = self._train_one_epoch(epoch_index)
            val_metrics = self.evaluate() if self.val_loader is not None else {}

            epoch_metrics = {
                "epoch": epoch_index + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics.get("loss"),
                "val_accuracy": val_metrics.get("accuracy"),
                "epoch_seconds": round(time.perf_counter() - start_time, 2),
            }

            self.state.completed_epochs = epoch_index + 1
            self.state.history.append(epoch_metrics)
            write_json(self.run_paths.history_file, self.state.history)

            self.logger.info(
                "Epoch %s summary | train_loss=%.4f | train_acc=%.2f%% | val_loss=%s | val_acc=%s | time=%.2fs",
                epoch_metrics["epoch"],
                epoch_metrics["train_loss"],
                epoch_metrics["train_accuracy"] * 100,
                _format_optional_metric(epoch_metrics["val_loss"]),
                _format_optional_metric(epoch_metrics["val_accuracy"], percentage=True),
                epoch_metrics["epoch_seconds"],
            )

            self._save_epoch_checkpoint(epoch_metrics)
            should_stop = self._update_early_stopping(epoch_metrics)
            self._save_last_checkpoint()

            if should_stop:
                self.logger.info(
                    "Early stopping triggered after %s consecutive non-improving epochs.",
                    self.state.epochs_without_improvement,
                )
                break

        return self.state.history

    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        batches_processed = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(self.val_loader or (), start=1):
                if self.config.max_val_batches and batch_index > self.config.max_val_batches:
                    break

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
        if batches_processed == 0:
            return {"loss": 0.0, "accuracy": 0.0}

        return {
            "loss": loss_sum / batches_processed,
            "accuracy": correct / max(total, 1),
        }

    def _resume(self, checkpoint_path: Path) -> None:
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        for warning in checkpoint.get("_restore_warnings", []):
            self.logger.warning("Checkpoint restore warning: %s", warning)
        self.state = TrainerState.from_dict(checkpoint.get("state", {}))

    def _train_one_epoch(self, epoch_index: int) -> dict[str, float]:
        self.model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        batches_processed = 0

        for batch_index, batch in enumerate(self.train_loader, start=1):
            if self.config.max_train_batches and batch_index > self.config.max_train_batches:
                break

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
            self.state.global_step += 1

            if self.state.global_step % self.config.log_every_n_steps == 0:
                self.logger.info(
                    "Epoch %s | step %s | loss=%.4f | acc=%.2f%% | grad_norm=%s | lr=%.2e",
                    epoch_index + 1,
                    self.state.global_step,
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
                    self.raw_model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.gradient_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.raw_model.parameters(),
                    self.config.gradient_clip_norm,
                )
            self.optimizer.step()

        self.scheduler.step()
        if grad_norm is None:
            return None
        return float(grad_norm)

    def _save_epoch_checkpoint(self, epoch_metrics: dict[str, Any]) -> None:
        if self.state.completed_epochs % self.config.checkpoint_every_n_epochs != 0:
            return

        checkpoint_path = self.run_paths.checkpoints_dir / f"epoch_{self.state.completed_epochs:03d}.pt"
        save_checkpoint(
            checkpoint_path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            state=self.state.to_dict(),
            metrics=epoch_metrics,
        )
        cleanup_old_epoch_checkpoints(
            self.run_paths.checkpoints_dir,
            self.config.keep_last_n_checkpoints,
        )

    def _save_last_checkpoint(self) -> None:
        save_checkpoint(
            self.run_paths.checkpoints_dir / "last.pt",
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            state=self.state.to_dict(),
        )

    def _update_early_stopping(self, epoch_metrics: dict[str, Any]) -> bool:
        val_loss = epoch_metrics.get("val_loss")
        if val_loss is None:
            return False

        if self.state.best_val_loss is None or val_loss < self.state.best_val_loss:
            self.state.best_val_loss = val_loss
            self.state.epochs_without_improvement = 0
            save_checkpoint(
                self.run_paths.checkpoints_dir / "best.pt",
                model=self.raw_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                config=self.config,
                state=self.state.to_dict(),
                metrics=epoch_metrics,
            )
            return False

        self.state.epochs_without_improvement += 1
        return self.state.epochs_without_improvement >= self.config.early_stopping_patience


def _estimate_total_steps(config: TrainingConfig, train_loader: DataLoader) -> int:
    steps_per_epoch = len(train_loader)
    if config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_train_batches)
    return max(steps_per_epoch * config.num_epochs, 1)


def _resolve_precision(
    config: TrainingConfig, device: torch.device
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


def _format_optional_metric(value: float | None, percentage: bool = False) -> str:
    if value is None:
        return "n/a"
    if percentage:
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"
