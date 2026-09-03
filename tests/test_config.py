from src.training.config import TrainingConfig


def main():
    print("Starting training configuration test...")

    config = TrainingConfig()

    assert config.model_name == "resnet50"
    assert config.num_classes == 1
    assert config.image_size == 96
    assert config.batch_size == 32
    assert config.epochs == 10
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-4
    assert config.seed == 42
    assert config.best_metric == "f1"

    assert config.resolved_device() in {"cpu", "cuda"}

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["model_name"] == "resnet50"
    assert config_dict["num_classes"] == 1

    checkpoint_path = config.checkpoint_path()

    assert checkpoint_path.name == "best_model.pt"
    assert str(checkpoint_path).endswith(
        "outputs\\checkpoints\\best_model.pt"
    ) or str(checkpoint_path).endswith(
        "outputs/checkpoints/best_model.pt"
    )

    print("Model configuration validated.")
    print("Optimization configuration validated.")
    print("Reproducibility configuration validated.")
    print("Device configuration validated.")
    print("Checkpoint configuration validated.")
    print("Training configuration test passed successfully.")


if __name__ == "__main__":
    main()
