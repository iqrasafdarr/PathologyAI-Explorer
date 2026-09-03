"""
PyTorch Dataset for PatchCamelyon (PCam), plus a DataLoader builder and a
standalone data-pipeline sanity check.

HDF5 + multiprocessing note:
h5py file handles are not safely shared across process boundaries (this
matters on both Linux fork-based and Windows spawn-based multiprocessing).
The fix used here is lazy, per-worker opening: each PCamDataset instance
does NOT open its HDF5 files in __init__. Instead, it opens them the first
time __getitem__ is called *within whichever process/thread is using it*,
and caches the handle on `self`. When DataLoader forks/spawns worker
processes, each worker gets its own copy of the (not-yet-opened) dataset
object, so each worker ends up with its own independent file handle instead
of sharing one across processes. This works correctly with num_workers > 0
on Linux, Windows, and in Colab.

PCam label shape quirk: the official *_y.h5 files store labels with shape
(N, 1, 1, 1), not (N,) — this is squeezed to a plain scalar per item here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import get_device, load_config, seed_everything


class PCamDataset(Dataset):
    """
    PatchCamelyon dataset backed by a pair of HDF5 files (images, labels).

    Args:
        x_path: path to the *_x.h5 file (images, shape (N, 96, 96, 3), uint8).
        y_path: path to the *_y.h5 file (labels, shape (N, 1, 1, 1), uint8/int).
        transform: optional callable applied to each image (expects a HxWxC
            uint8 numpy array, as produced by torchvision's ToPILImage step
            in src/data/transforms.py).
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        if not self.x_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {self.x_path}. "
                f"Run scripts/download_data.py first."
            )
        if not self.y_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {self.y_path}. "
                f"Run scripts/download_data.py first."
            )

        # Read only the length up front (cheap, metadata-only); the actual
        # per-worker file handles are opened lazily — see module docstring.
        with h5py.File(self.x_path, "r") as f:
            self._length = f["x"].shape[0]

        self._x_file: Optional[h5py.File] = None
        self._y_file: Optional[h5py.File] = None

    def _ensure_open(self) -> None:
        """Open HDF5 handles lazily, once per worker process."""
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
        if self._y_file is None:
            self._y_file = h5py.File(self.y_path, "r")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        image = self._x_file["x"][idx]  # (96, 96, 3) uint8
        label = int(np.array(self._y_file["y"][idx]).squeeze())  # scalar 0/1

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, label

    def __del__(self) -> None:
        # Best-effort cleanup; safe even if handles were never opened.
        for handle in (self._x_file, self._y_file):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass


def build_datasets(config: dict, train_transform=None, eval_transform=None):
    """Construct train/valid/test PCamDataset objects from the config's data paths."""
    data_cfg = config["data"]
    root = Path(data_cfg["root_dir"])

    train_ds = PCamDataset(
        root / data_cfg["train_x"], root / data_cfg["train_y"], transform=train_transform
    )
    valid_ds = PCamDataset(
        root / data_cfg["valid_x"], root / data_cfg["valid_y"], transform=eval_transform
    )
    test_ds = PCamDataset(
        root / data_cfg["test_x"], root / data_cfg["test_y"], transform=eval_transform
    )
    return train_ds, valid_ds, test_ds


def build_dataloaders(config: dict, train_transform=None, eval_transform=None):
    """Construct train/valid/test DataLoaders from the config's dataloader settings."""
    train_ds, valid_ds, test_ds = build_datasets(config, train_transform, eval_transform)
    dl_cfg = config["dataloader"]
    device_is_cuda = get_device(config["device"]["prefer"]).type == "cuda"

    common_kwargs = dict(
        batch_size=dl_cfg["batch_size"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"] and device_is_cuda,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)
    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# Sanity check (Phase 1 deliverable): verify the full pipeline without
# training anything. Reports dataset sizes, class balance, tensor shapes,
# and saves a sample visualization grid.
# ---------------------------------------------------------------------------


def run_sanity_check(config_path: str = "configs/experiment.yaml") -> None:
    from src.data.transforms import get_eval_transforms, get_train_transforms
    from src.visualization.visualize import save_sample_grid

    config = load_config(config_path)
    seed_everything(config["seed"])
    device = get_device(config["device"]["prefer"])
    print(f"Device: {device}\n")

    image_size = config["data"]["image_size"]
    train_ds, valid_ds, test_ds = build_datasets(
        config,
        train_transform=get_train_transforms(image_size),
        eval_transform=get_eval_transforms(image_size),
    )

    print("=== Dataset sizes ===")
    print(f"  Train: {len(train_ds):,}")
    print(f"  Valid: {len(valid_ds):,}")
    print(f"  Test:  {len(test_ds):,}")

    print("\n=== Class distribution (from a 2,000-sample scan of each split) ===")
    for name, ds in [("Train", train_ds), ("Valid", valid_ds), ("Test", test_ds)]:
        sample_n = min(2000, len(ds))
        # Use the untransformed label reader directly for speed.
        ds._ensure_open()
        labels = np.array(ds._y_file["y"][:sample_n]).squeeze()
        n_pos = int(labels.sum())
        print(f"  {name}: {n_pos}/{sample_n} positive ({100 * n_pos / sample_n:.1f}%)")

    print("\n=== Number of classes ===")
    print("  2 (binary: metastatic tissue present / absent)")

    print("\n=== Tensor shape check (single training example) ===")
    image, label = train_ds[0]
    print(f"  Image tensor shape: {tuple(image.shape)}  dtype: {image.dtype}")
    print(f"  Label: {label} (type: {type(label).__name__})")

    print("\n=== Raw image shape check (before transform) ===")
    raw_ds = PCamDataset(
        Path(config["data"]["root_dir"]) / config["data"]["train_x"],
        Path(config["data"]["root_dir"]) / config["data"]["train_y"],
        transform=None,
    )
    raw_image, raw_label = raw_ds[0]
    print(f"  Raw image tensor shape: {tuple(raw_image.shape)} (from a 96x96x3 uint8 array)")

    print("\n=== DataLoader check ===")
    train_loader, valid_loader, test_loader = build_dataloaders(
        config,
        train_transform=get_train_transforms(image_size),
        eval_transform=get_eval_transforms(image_size),
    )
    batch_images, batch_labels = next(iter(train_loader))
    print(f"  Batch image shape: {tuple(batch_images.shape)}")
    print(f"  Batch label shape: {tuple(batch_labels.shape)}")
    print(f"  Example labels in batch: {batch_labels[:10].tolist()}")

    print("\n=== Sample visualization ===")
    fig_dir = Path(config["results"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_path = fig_dir / "sample_patches.png"
    save_sample_grid(train_ds, out_path, n_samples=16, seed=config["seed"])
    print(f"  Saved sample grid to: {out_path}")

    print("\nSanity check complete — data pipeline is working.")
    print("Status: Data pipeline implemented — model training pending.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCam dataset utilities.")
    parser.add_argument(
        "--sanity-check", action="store_true", help="Run the data pipeline sanity check."
    )
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to the experiment config."
    )
    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check(args.config)
    else:
        parser.print_help()
