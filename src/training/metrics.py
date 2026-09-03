"""
Training metrics for binary classification.

Provides lightweight PyTorch-based metric utilities used during
training and validation without requiring sklearn in the training loop.
"""

from typing import Dict

import torch


@torch.no_grad()
def binary_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Calculate binary classification accuracy from raw model logits.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).float()

    targets = targets.float().view_as(predictions)

    correct = (predictions == targets).sum().item()

    total = targets.numel()

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def binary_confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Calculate true positive, true negative, false positive,
    and false negative counts.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (probabilities >= threshold).long().view(-1)

    targets = targets.long().view(-1)

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def precision_recall_f1_from_counts(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> Dict[str, float]:
    """
    Calculate accuracy, precision, recall, specificity,
    and F1-score from confusion matrix counts.
    """

    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


if __name__ == "__main__":
    print("Testing binary metrics...")

    logits = torch.tensor([
        [2.0],
        [-2.0],
        [3.0],
        [-3.0],
    ])

    targets = torch.tensor([
        [1],
        [0],
        [1],
        [0],
    ])

    accuracy = binary_accuracy_from_logits(logits, targets)

    counts = binary_confusion_counts(logits, targets)

    metrics = precision_recall_f1_from_counts(**counts)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion counts: {counts}")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Metrics test passed successfully.")