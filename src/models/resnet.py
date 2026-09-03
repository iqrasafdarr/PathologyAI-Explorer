"""
ResNet50 model for binary histopathology image classification.

This module provides a configurable ResNet50 transfer-learning model
for the PCam computational pathology dataset.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(
    num_classes: int = 1,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Build a ResNet50 model for binary or multi-class classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes. Use 1 for binary classification
        with BCEWithLogitsLoss.
    pretrained : bool
        Whether to initialize the backbone with ImageNet weights.
    freeze_backbone : bool
        Whether to freeze convolutional backbone parameters initially.

    Returns
    -------
    nn.Module
        Configured ResNet50 model.
    """

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

    model = resnet50(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters.

    Returns
    -------
    Tuple[int, int]
        Total parameters and trainable parameters.
    """

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


if __name__ == "__main__":
    print("Testing ResNet50 model...")

    model = build_resnet50(
        num_classes=1,
        pretrained=False,
        freeze_backbone=False,
    )

    dummy_input = torch.randn(2, 3, 96, 96)

    with torch.no_grad():
        output = model(dummy_input)

    total, trainable = count_trainable_parameters(model)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("ResNet50 test passed successfully.")