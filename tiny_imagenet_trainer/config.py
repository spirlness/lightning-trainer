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
    """Top-level training configuration."""

    experiment_name: str = "tiny-imagenet-resnet18"
    output_root: Path = Path("outputs")

    dataset_source: str = "local"
    dataset_name: str = "zh-plus/tiny-imagenet"
    dataset_cache_dir: Path = Path("data") / "hf_cache"
    local_dataset_dir: Path = Path("data") / "tiny_imagenet_local"
    train_split: str = "train"
    val_split: str = "valid"
    validation_split_ratio: float = 0.1
    force_redownload: bool = False

    model_name: str = "resnet18"
    num_classes: int = 200
    image_size: int = 224
    pretrained: bool = True
    compile_model: bool = True
    enable_amp: bool = True
    device: str = "auto"

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    num_epochs: int = 10
    warmup_steps: int = 50
    gradient_clip_norm: float | None = 1.0
    num_workers: int = field(default_factory=_default_worker_count)
    max_train_batches: int | None = None
    max_val_batches: int | None = 50
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    early_stopping_patience: int = 3
    seed: int = 42
    resume_from: Path | None = None

    mock_train_samples: int = 256
    mock_val_samples: int = 64

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.dataset_cache_dir = Path(self.dataset_cache_dir)
        self.local_dataset_dir = Path(self.local_dataset_dir)
        self.resume_from = Path(self.resume_from) if self.resume_from else None

        if self.dataset_source not in {"local", "huggingface", "mock"}:
            raise ValueError("dataset_source must be 'local', 'huggingface', or 'mock'")
        if self.model_name != "resnet18":
            raise ValueError("Only model_name='resnet18' is currently supported")
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
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be positive")
        if self.keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be positive")
        if not 0.0 < self.validation_split_ratio < 1.0:
            raise ValueError("validation_split_ratio must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in payload.items():
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload

    def build_run_paths(self) -> RunPaths:
        if self.resume_from:
            return RunPaths.from_root(self.resume_from.resolve().parents[1])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.experiment_name}-{timestamp}"
        return RunPaths.from_root(self.output_root / run_name)
