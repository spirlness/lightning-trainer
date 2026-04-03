from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch

from tiny_imagenet_trainer.config import RunPaths, TrainingConfig


def seed_everything(seed: int) -> None:
    """Seed Python, torch, and numpy when available."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import numpy as np
    except ImportError:
        return

    np.random.seed(seed)


def select_device(preferred: str) -> torch.device:
    """Resolve the training device from a simple user-facing string."""
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_logger(log_file: Path) -> logging.Logger:
    """Create a concise project logger with file + console output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("tiny_imagenet_trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def write_json(path: Path, payload: object) -> None:
    """Write JSON with stable formatting for configs and history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare_run(config: TrainingConfig) -> tuple[RunPaths, logging.Logger]:
    """Create a run directory and ready-to-use logger."""
    run_paths = config.build_run_paths()
    logger = configure_logger(run_paths.log_file)
    write_json(run_paths.config_file, config.to_dict())
    return run_paths, logger
