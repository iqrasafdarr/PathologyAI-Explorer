"""
Visualization utilities.

Phase 1 scope: a sample-patch grid, used by the data pipeline sanity check
to visually confirm images and labels are loading correctly.

Phase 2 will add: training/validation curve plots, confusion matrix
plotting, and a misclassified-patch gallery for failure analysis — these
depend on a trained model and are out of scope until then.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_sample_grid(
    dataset,
    out_path: str | Path,
    n_samples: int = 16,
    seed: int = 42,
) -> None:
    """
    Save a grid of sample patches with their labels for visual inspection.

    Reads directly from the dataset's underlying HDF5 arrays (bypassing any
    normalization transform) so the saved images look like real patches
    rather than normalized tensors with negative pixel values.
    """
    rng = random.Random(seed)
    n_samples = min(n_samples, len(dataset))
    indices = rng.sample(range(len(dataset)), n_samples)

    dataset._ensure_open()  # PCamDataset lazy-open, see src/data/dataset.py
    ncols = 4
    nrows = (n_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, indices):
        image = dataset._x_file["x"][idx]
        label = int(np.array(dataset._y_file["y"][idx]).squeeze())
        ax.imshow(image)
        ax.set_title(f"label={label}", fontsize=10)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.suptitle("PCam sample patches (0 = no tumor tissue, 1 = tumor tissue present)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
