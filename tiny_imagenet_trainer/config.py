from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_worker_count() -> int:
    return min(2, os.cpu_count() or 1)


@dataclass(slots=True)
class RunPaths:
    """Filesystem layout for a single training run."""

    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    log_file: Path
    config_file: Path
    history_file: Path

    @classmethod
    def from_root(cls, run_dir: Path) -> "RunPaths":
        run_dir = Path(run_dir)
        checkpoints_dir = run_dir / "checkpoints"
        logs_dir = run_dir / "logs"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            log_file=logs_dir / "train.log",
            config_file=run_dir / "config.json",
            history_file=run_dir / "history.json",
        )


@dataclass(slots=True)
class TrainingConfig:
    """Minimal training configuration for local Tiny-ImageNet runs."""

    experiment_name: str = "tiny-imagenet-resnet18"
    output_root: Path = Path("outputs")
    data_dir: Path = Path("data") / "tiny_imagenet_local"

    num_classes: int = 200
    image_size: int = 224
    pretrained: bool = True
    enable_amp: bool = True
    device: str = "auto"

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 10
    warmup_steps: int = 50
    gradient_clip_norm: float | None = 1.0
    num_workers: int = field(default_factory=_default_worker_count)
    log_every_n_steps: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.data_dir = Path(self.data_dir)

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def build_run_paths(self) -> RunPaths:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.experiment_name}-{timestamp}"
        return RunPaths.from_root(self.output_root / run_name)
