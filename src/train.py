import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import build_dataloaders
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.resnet import build_resnet50
from src.training.checkpointing import save_checkpoint
from src.training.trainer import train_one_epoch, validate_one_epoch
from src.utils import get_device, load_config, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 on PCam.")

    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device", type=str)
    parser.add_argument("--smoke-test", action="store_true")

    return parser.parse_args()


def apply_overrides(config, args):
    config = deepcopy(config)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["dataloader"]["batch_size"] = args.batch_size

    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay

    if args.device is not None:
        config["device"]["prefer"] = args.device

    return config


def build_smoke_loaders(batch_size=2):
    torch.manual_seed(42)

    train_images = torch.randn(4, 3, 96, 96)
    train_labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

    valid_images = torch.randn(2, 3, 96, 96)
    valid_labels = torch.tensor([0, 1], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        TensorDataset(valid_images, valid_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader


def run_smoke_test(config):
    print("Starting training CLI smoke test...")

    device = torch.device("cpu")

    train_loader, valid_loader = build_smoke_loaders()

    model = build_resnet50(
        num_classes=1,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    print("Running one training epoch...")

    train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    print("Running validation...")

    valid_metrics = validate_one_epoch(
        model,
        valid_loader,
        criterion,
        device,
    )

    checkpoint_path = Path("results") / "checkpoints" / "smoke_test.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics=valid_metrics,
        best_metric=valid_metrics["f1"],
        config=config,
    )

    print("\n=== Training Metrics ===")
    for key, value in train_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\n=== Validation Metrics ===")
    for key, value in valid_metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nCheckpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise RuntimeError("Smoke-test checkpoint was not created.")

    print("\nTraining CLI smoke test passed successfully.")


def run_training(config):
    seed_everything(config["seed"])

    device = get_device(config["device"]["prefer"])
    print(f"Using device: {device}")

    train_transform = get_train_transforms(
        image_size=config["data"]["image_size"]
    )

    eval_transform = get_eval_transforms(
        image_size=config["data"]["image_size"]
    )

    train_loader, valid_loader, test_loader = build_dataloaders(
        config,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    model = build_resnet50(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        freeze_backbone=config["model"]["freeze_backbone"],
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]
    best_f1 = float("-inf")

    checkpoint_path = (
        Path(config["training"]["checkpoint_dir"])
        / "best_model.pt"
    )

    print(f"Training for {epochs} epoch(s)...")
    print(f"Best checkpoint: {checkpoint_path}")

    for epoch in range(1, epochs + 1):

        print(f"\nEpoch {epoch}/{epochs}")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        valid_metrics = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )

        print(
            f"Train | "
            f"loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"f1={train_metrics['f1']:.4f}"
        )

        print(
            f"Valid | "
            f"loss={valid_metrics['loss']:.4f} "
            f"acc={valid_metrics['accuracy']:.4f} "
            f"f1={valid_metrics['f1']:.4f}"
        )

        current_f1 = valid_metrics["f1"]

        if current_f1 > best_f1:

            best_f1 = current_f1

            save_checkpoint(
                path=str(checkpoint_path),
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=valid_metrics,
                best_metric=best_f1,
                config=config,
            )

            print(f"Saved new best checkpoint: {checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation F1: {best_f1:.4f}")


def main():
    args = parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args)

    seed_everything(config["seed"])

    if args.smoke_test:
        run_smoke_test(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
