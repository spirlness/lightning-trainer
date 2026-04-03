"""Model architecture definition."""
from __future__ import annotations

import logging

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from tiny_imagenet_trainer.config import TrainingConfig


def build_model(
    config: TrainingConfig,
    logger: logging.Logger | None = None,
) -> nn.Module:
    """Build ResNet18 model for image classification.

    Creates a ResNet18 backbone with a custom classification head
    matching the number of classes in the dataset.

    Args:
        config: Training configuration with model parameters
        logger: Optional logger for warnings

    Returns:
        Configured ResNet18 model

    Raises:
        RuntimeError: If pretrained weights fail to load and pretrained=True
    """
    # Determine which weights to load
    weights: ResNet18_Weights | None = (
        ResNet18_Weights.DEFAULT if config.pretrained else None
    )

    # Load model with appropriate weights
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        # If pretrained loading fails, fall back to random init
        if not config.pretrained:
            raise RuntimeError(f"Failed to create ResNet18 model: {exc}") from exc

        if logger:
            logger.warning(
                "Failed to load pretrained weights (%s). "
                "Falling back to random initialization.",
                exc,
            )
        model = resnet18(weights=None)

    # Replace final FC layer to match number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, config.num_classes)

    return model
