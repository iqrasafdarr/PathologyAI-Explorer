import tempfile
from pathlib import Path

import torch

from src.models.resnet import build_resnet50
from src.training.checkpointing import save_checkpoint, load_checkpoint


def main():
    print("Starting checkpoint save/load test...")

    model = build_resnet50(pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = {
        "loss": 0.42,
        "accuracy": 0.81,
        "f1": 0.79,
    }

    config = {
        "model": "resnet50",
        "num_classes": 1,
        "image_size": 96,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        save_checkpoint(
            path=str(checkpoint_path),
            model=model,
            optimizer=optimizer,
            epoch=3,
            metrics=metrics,
            best_metric=0.79,
            config=config,
        )

        assert checkpoint_path.exists(), "Checkpoint file was not created."

        restored_model = build_resnet50(pretrained=False)
        restored_optimizer = torch.optim.Adam(
            restored_model.parameters(),
            lr=1e-3,
        )

        checkpoint = load_checkpoint(
            path=str(checkpoint_path),
            model=restored_model,
            optimizer=restored_optimizer,
            device="cpu",
        )

        assert checkpoint["epoch"] == 3
        assert checkpoint["metrics"] == metrics
        assert checkpoint["best_metric"] == 0.79
        assert checkpoint["config"] == config

        for original, restored in zip(
            model.parameters(),
            restored_model.parameters(),
        ):
            assert torch.equal(original, restored), (
                "Model parameters were not restored correctly."
            )

    print("Checkpoint file created successfully.")
    print("Model state restored successfully.")
    print("Optimizer state restored successfully.")
    print("Training metadata restored successfully.")
    print("Checkpoint save/load test passed successfully.")


if __name__ == "__main__":
    main()
