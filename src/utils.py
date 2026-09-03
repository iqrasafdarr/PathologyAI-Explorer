"""
Shared utilities: config loading, reproducible seeding, and device selection.

Not part of the originally approved file list, but needed as small glue code
so that seeding and config parsing aren't duplicated across scripts/notebooks.
Kept intentionally minimal — see docs/limitations.md / the Phase 1 summary
for why this file was added.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path = "configs/experiment.yaml") -> dict[str, Any]:
    """Load the experiment YAML config into a plain dict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path.resolve()}. "
            f"Run this from the repository root, or pass an explicit path."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch (CPU and CUDA) for reproducibility.

    Note on limitations: this makes runs reproducible in the common case, but
    full bit-for-bit determinism on GPU is not guaranteed. Some cuDNN
    convolution algorithms are non-deterministic by default for performance
    reasons. If exact determinism is required, additionally set
    `torch.use_deterministic_algorithms(True)` and
    `torch.backends.cudnn.benchmark = False` — not enabled by default here
    because it can noticeably slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    """Return a torch.device, falling back to CPU if the preferred device is unavailable."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU.")
    return torch.device("cpu")
