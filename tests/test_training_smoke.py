"""
Synthetic end-to-end smoke test for the training pipeline.

This test verifies that the dataset, DataLoader, model,
training loop, validation loop, loss function, optimizer,
and metrics work together correctly.
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import build_resnet50
from src.training.trainer import train_one_epoch, validate_one_epoch


def main():
    print("Starting end-to-end training smoke test...")

    torch.manual_seed(42)

    # Small synthetic PCam-shaped dataset.
    images = torch.randn(8, 3, 96, 96)

    # Binary labels.
    targets = torch.randint(
        low=0,
        high=2,
        size=(8,),
    )

    dataset = TensorDataset(images, targets)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    # Use ResNet50 without downloading pretrained weights.
    model = build_resnet50(
        num_classes=1,
        pretrained=False,
        freeze_backbone=False,
    )

    device = torch.device("cpu")

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    print("Running one training epoch...")

    train_metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print("Running validation...")

    validation_metrics = validate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    print("\n=== Training Metrics ===")

    for name, value in train_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n=== Validation Metrics ===")

    for name, value in validation_metrics.items():
        print(f"{name}: {value:.4f}")

    required_metrics = {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
    }

    assert required_metrics.issubset(train_metrics.keys())
    assert required_metrics.issubset(validation_metrics.keys())

    print("\nEnd-to-end smoke test passed successfully.")


if __name__ == "__main__":
    main()