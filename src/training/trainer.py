"""
Training and validation utilities for binary image classification.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import (
    binary_confusion_counts,
    precision_recall_f1_from_counts,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train a model for one epoch.

    Returns averaged loss and binary classification metrics.
    """

    model.train()

    running_loss = 0.0
    total_samples = 0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)

        targets = targets.float().to(device, non_blocking=True)
        targets = targets.view(-1, 1)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, targets)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        counts = binary_confusion_counts(logits.detach(), targets)

        total_tp += counts["tp"]
        total_tn += counts["tn"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

    average_loss = (
        running_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    metrics = precision_recall_f1_from_counts(
        tp=total_tp,
        tn=total_tn,
        fp=total_fp,
        fn=total_fn,
    )

    metrics["loss"] = average_loss

    return metrics


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a model for one validation epoch.

    Returns averaged loss and binary classification metrics.
    """

    model.eval()

    running_loss = 0.0
    total_samples = 0

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)

        targets = targets.float().to(device, non_blocking=True)
        targets = targets.view(-1, 1)

        logits = model(images)

        loss = criterion(logits, targets)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        counts = binary_confusion_counts(logits, targets)

        total_tp += counts["tp"]
        total_tn += counts["tn"]
        total_fp += counts["fp"]
        total_fn += counts["fn"]

    average_loss = (
        running_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    metrics = precision_recall_f1_from_counts(
        tp=total_tp,
        tn=total_tn,
        fp=total_fp,
        fn=total_fn,
    )

    metrics["loss"] = average_loss

    return metrics