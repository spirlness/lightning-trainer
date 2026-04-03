"""Training configuration with validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


def _default_workers() -> int:
    """Default number of data loading workers."""
    import os
    return min(2, os.cpu_count() or 1)


@dataclass(slots=True)
class RunPaths:
    """Filesystem layout for a training run."""
    run_dir: Path
    checkpoints_dir: Path
    logs_dir: Path
    log_file: Path
    config_file: Path
    history_file: Path

    @classmethod
    def from_root(cls, run_dir: Path) -> "RunPaths":
        """Create RunPaths and ensure directories exist."""
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
    """Training configuration with validation."""

    # Experiment
    experiment_name: str = field(
        default="tiny-imagenet-resnet18",
        metadata={
            "group": "Experiment",
            "help": "Name for this experiment",
        },
    )
    output_root: Path = field(
        default=Path("outputs"),
        metadata={
            "group": "Experiment",
            "help": "Root directory for all experiment outputs",
        },
    )

    # Data
    data_dir: Path = field(
        default=Path("data") / "tiny_imagenet_local",
        metadata={
            "group": "Data",
            "help": "Path to dataset directory with train/ and val/ subdirectories",
        },
    )
    num_classes: int = field(
        default=200,
        metadata={
            "group": "Data",
            "help": "Number of classes in the dataset",
            "cli_min": 1,
            "cli_min_inclusive": True,
        },
    )
    image_size: int = field(
        default=224,
        metadata={
            "group": "Data",
            "help": "Input image size (square)",
            "cli_min": 1,
            "cli_min_inclusive": True,
        },
    )

    # Model
    pretrained: bool = field(
        default=True,
        metadata={
            "group": "Model",
            "help": "Use pretrained ImageNet weights",
        },
    )

    # Training
    learning_rate: float = field(
        default=1e-4,
        metadata={
            "group": "Training",
            "help": "Base learning rate",
            "cli_min": 0.0,
            "cli_min_inclusive": False,
        },
    )
    weight_decay: float = field(
        default=1e-4,
        metadata={
            "group": "Training",
            "help": "Weight decay (L2 regularization)",
            "cli_min": 0.0,
            "cli_min_inclusive": True,
        },
    )
    batch_size: int = field(
        default=128,
        metadata={
            "group": "Training",
            "help": "Batch size for training and validation",
            "cli_min": 1,
            "cli_min_inclusive": True,
        },
    )
    num_epochs: int = field(
        default=10,
        metadata={
            "group": "Training",
            "help": "Number of training epochs",
            "cli_min": 1,
            "cli_min_inclusive": True,
        },
    )
    warmup_steps: int = field(
        default=50,
        metadata={
            "group": "Training",
            "help": "Number of warmup steps for learning rate",
            "cli_min": 0,
            "cli_min_inclusive": True,
        },
    )
    gradient_clip_norm: float | None = field(
        default=1.0,
        metadata={
            "group": "Training",
            "help": "Gradient clipping norm (None to disable)",
            "cli_allow_none": True,
            "cli_min": 0.0,
            "cli_min_inclusive": False,
        },
    )

    # Data loading
    num_workers: int = field(
        default_factory=_default_workers,
        metadata={
            "group": "Data Loading",
            "help": "Number of data loading workers",
            "cli_min": 0,
            "cli_min_inclusive": True,
        },
    )

    # Logging
    log_every_n_steps: int = field(
        default=10,
        metadata={
            "group": "Logging",
            "help": "Log training metrics every N steps",
            "cli_min": 1,
            "cli_min_inclusive": True,
        },
    )

    # Reproducibility
    seed: int = field(
        default=42,
        metadata={
            "group": "Reproducibility",
            "help": "Random seed for reproducibility",
        },
    )

    # Hardware
    device: str = field(
        default="auto",
        metadata={
            "group": "Hardware",
            "help": "Device to use for training",
            "choices": ["auto", "cpu", "cuda"],
        },
    )
    enable_amp: bool = field(
        default=True,
        metadata={
            "group": "Hardware",
            "help": "Enable Automatic Mixed Precision",
        },
    )

    def __post_init__(self) -> None:
        """Validate configuration and convert paths."""
        self.output_root = Path(self.output_root)
        self.data_dir = Path(self.data_dir)

        # Validation rules: (condition, error_message)
        rules = [
            (self.device in {"auto", "cpu", "cuda"}, f"Invalid device: {self.device}"),
            (self.num_classes > 0, "num_classes must be positive"),
            (self.image_size > 0, "image_size must be positive"),
            (self.batch_size > 0, "batch_size must be positive"),
            (self.num_epochs > 0, "num_epochs must be positive"),
            (self.warmup_steps >= 0, "warmup_steps must be non-negative"),
            (self.num_workers >= 0, "num_workers must be non-negative"),
            (self.log_every_n_steps > 0, "log_every_n_steps must be positive"),
            (self.learning_rate > 0, "learning_rate must be positive"),
            (self.weight_decay >= 0, "weight_decay must be non-negative"),
        ]

        for condition, message in rules:
            if not condition:
                raise ValueError(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with path serialization."""
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result

    def build_run_paths(self) -> RunPaths:
        """Create run paths with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.experiment_name}-{timestamp}"
        return RunPaths.from_root(self.output_root / run_name)
