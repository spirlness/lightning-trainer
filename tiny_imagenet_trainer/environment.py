"""Environment setup utilities."""
from __future__ import annotations

import json
import logging
import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, TypedDict

import torch

from tiny_imagenet_trainer.config import RunPaths, TrainingConfig

if TYPE_CHECKING:
    from types import FrameType


class LogHandlerConfig(TypedDict, total=False):
    """Configuration for a single log handler."""

    level: int
    formatter: str
    filename: str | None
    stream: Any | None


class LoggerConfig(TypedDict, total=False):
    """Structured configuration for logger setup."""

    version: int
    disable_existing_loggers: bool
    formatters: dict[str, dict[str, str]]
    handlers: dict[str, dict[str, Any]]
    loggers: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Immutable device information container."""

    device: torch.device
    name: str
    supports_amp: bool


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility using standard library.

    Seeds Python, torch, and numpy (if available) for deterministic results.

    Args:
        seed: Random seed value
    """
    # Python standard library random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic (at some performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Seed numpy if available (optional dependency)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


@lru_cache(maxsize=1)
def _get_cuda_info() -> tuple[bool, str | None]:
    """Cache CUDA availability check to avoid repeated system calls.

    Returns:
        Tuple of (is_available, device_name)
    """
    if not torch.cuda.is_available():
        return False, None
    try:
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    except RuntimeError:
        return False, None


def select_device(preference: str) -> DeviceInfo:
    """Select computation device based on preference and availability.

    Args:
        preference: One of "auto", "cpu", "cuda"

    Returns:
        DeviceInfo containing device, name, and AMP support info

    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    if preference == "cpu":
        return DeviceInfo(
            device=torch.device("cpu"), name="cpu", supports_amp=False
        )

    if preference == "cuda":
        is_available, device_name = _get_cuda_info()
        if not is_available:
            raise RuntimeError(
                "CUDA requested but not available on this machine"
            )
        return DeviceInfo(
            device=torch.device("cuda"),
            name=device_name or "cuda:0",
            supports_amp=True,
        )

    # Auto: use CUDA if available, else CPU
    is_available, device_name = _get_cuda_info()
    if is_available:
        return DeviceInfo(
            device=torch.device("cuda"),
            name=device_name or "cuda:0",
            supports_amp=True,
        )
    return DeviceInfo(
        device=torch.device("cpu"), name="cpu", supports_amp=False
    )


def configure_logger(
    log_file: Path,
    level: int = logging.INFO,
    name: str = "tiny_imagenet_trainer",
) -> logging.Logger:
    """Configure and return a logger with file and console handlers.

    Args:
        log_file: Path to log file
        level: Logging level
        name: Logger name

    Returns:
        Configured logger instance
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(level)
    logger.addHandler(ch)

    return logger


# Backwards compatibility alias
setup_logger = configure_logger


def write_json(path: Path, data: object) -> None:
    """Write data to JSON file with consistent formatting.

    Args:
        path: Target file path
        data: Data to serialize
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def prepare_run(
    config: TrainingConfig,
) -> tuple[RunPaths, logging.Logger]:
    """Initialize training run: create directories and logger.

    Args:
        config: Training configuration

    Returns:
        Tuple of (run_paths, logger)
    """
    run_paths = config.build_run_paths()
    logger = setup_logger(run_paths.log_file)
    write_json(run_paths.config_file, config.to_dict())
    return run_paths, logger
