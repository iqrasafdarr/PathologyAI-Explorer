from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingConfig:
    """Central configuration for ResNet50 PCam training."""

    # Model
    model_name: str = "resnet50"
    num_classes: int = 1
    pretrained: bool = True
    freeze_backbone: bool = False

    # Data
    image_size: int = 96
    batch_size: int = 32
    num_workers: int = 0

    # Optimization
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # Reproducibility
    seed: int = 42

    # Runtime
    device: str = "auto"

    # Checkpointing
    checkpoint_dir: str = "outputs/checkpoints"
    best_metric: str = "f1"

    def resolved_device(self) -> str:
        """Resolve automatic device selection."""
        if self.device != "auto":
            return self.device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as a serializable dictionary."""
        return asdict(self)

    def checkpoint_path(self) -> Path:
        """Return the path for the best model checkpoint."""
        return Path(self.checkpoint_dir) / "best_model.pt"
