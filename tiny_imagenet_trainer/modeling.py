from __future__ import annotations

import logging

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from tiny_imagenet_trainer.config import TrainingConfig


def build_model(config: TrainingConfig, logger: logging.Logger | None = None) -> nn.Module:
    """Build the ResNet18 backbone for the training run."""
    weights = ResNet18_Weights.DEFAULT if config.pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if not config.pretrained:
            raise
        if logger:
            logger.warning(
                "Failed to load pretrained weights (%s). Falling back to random initialization.",
                exc,
            )
        model = resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    return model
