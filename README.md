# PathologyAI Explorer

**Preliminary Exploration of Histopathology Patch Classification and
Attention-Based Multiple Instance Learning**

## Status: Data pipeline implemented — model training pending.

This repository is a work in progress. See "Project Progress" below for
exactly what has and hasn't been done.

## Overview

This repository documents a preliminary, undergraduate-level exploration of
computational pathology — specifically, histopathology patch classification
and the conceptual link between patch-level predictions and whole-slide
image (WSI) analysis via Multiple Instance Learning (MIL).

This is **not** a production-grade or clinical-grade pathology system, and
it does not implement full whole-slide image classification. It is an
honest, scoped research exploration intended to demonstrate the ability to
learn and reason about a new subdomain (computational pathology) using
transfer learning skills developed on other medical imaging tasks.

## Key Concepts

- **Histopathology**: microscopic examination of tissue for disease
  diagnosis, here specifically metastatic breast cancer in lymph node
  sections.
- **Whole-Slide Imaging (WSI)**: gigapixel-scale digitized tissue slides,
  far too large to feed directly into a standard CNN.
- **Patch classification**: the approach used here — small, fixed-size
  crops of a slide are classified individually, which is exactly what the
  PatchCamelyon (PCam) dataset provides pre-extracted.
- **Multiple Instance Learning (MIL)**: the standard approach for going from
  many patch-level predictions to a single slide-level prediction, without
  requiring patch-level ground truth for every patch on a slide.
- **Attention-based aggregation**: a learned weighting scheme (as in CLAM)
  for combining patch-level evidence into a slide-level decision, so the
  most diagnostically relevant patches contribute the most.

Full documentation of the WSI/MIL relationship and the CLAM exploration is
being developed in `docs/` (see Project Progress below).

## Experimental Pipeline (what is actually implemented so far)

**Phase 1 (current):**
- PCam dataset download, checksum verification, and decompression
  (`scripts/download_data.py`)
- PyTorch `Dataset`/`DataLoader` pipeline for the official PCam HDF5 files,
  safe for multi-worker loading on Linux, Windows, and Colab
  (`src/data/dataset.py`)
- Train/eval transform pipelines with ImageNet normalization and
  histopathology-appropriate mild augmentation (`src/data/transforms.py`)
- A full data-pipeline sanity check: dataset sizes, class balance, tensor
  shapes, and a saved sample-patch visualization grid

**Not yet implemented (planned for later phases):**
- ResNet50 transfer-learning training and checkpointing
- Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- Training/validation curves and failure-case analysis
- The WSI/MIL conceptual documentation and diagram (A2)
- The scoped CLAM architecture exploration (A3)

## Results

No experiments have been run yet. **No accuracy, F1, or other performance
numbers exist in this repository**, and none will be added until they come
from an actual executed training run.

## CLAM Exploration

Not started. Will be added as a clearly separate, explicitly scoped
exploration once the core PCam classification pipeline (A1) is complete —
see the approved project scope in `docs/` (to be added).

## Limitations

To be documented fully in `docs/limitations.md` once experiments exist.
Known limitations already worth noting at the pipeline stage:

- PCam provides pre-extracted 96×96 patches, not full WSIs — this project
  does not perform WSI-scale processing.
- Patch-level classification does not by itself constitute slide-level
  diagnosis; the relationship between the two is a documentation exercise
  here (A2), not an additional experiment.
- No stain normalization is applied in Phase 1 beyond the augmentation
  pipeline's mild color jitter; PCam's own patch-selection process already
  filters for tissue content (see `data/README.md`).

## Setup

```bash
pip install -r requirements.txt
```

### Download the dataset

```bash
python scripts/download_data.py
```

This downloads PCam's HDF5 files from the official Zenodo mirror
(~8 GB total), verifies checksums, and decompresses them into `data/`.
See `data/README.md` for source details, manual fallback links, and split/
leakage documentation.

### Run the data pipeline sanity check

```bash
python -m src.data.dataset --sanity-check
```

This does **not** train anything. It loads all three splits, reports
dataset sizes and class balance, checks tensor shapes end-to-end through a
real `DataLoader`, and saves a sample-patch grid to
`results/figures/sample_patches.png`.

## Citation

If referencing the dataset used here:

> B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation
> Equivariant CNNs for Digital Pathology." arXiv:1806.03962 (2018).

PCam is derived from the CAMELYON16 challenge dataset.

## License

Code in this repository is MIT licensed (see `LICENSE`). The PCam dataset
itself is separately CC0-licensed by its original authors — see
`data/README.md`.
