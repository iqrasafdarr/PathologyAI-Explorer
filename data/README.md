# Data Directory

This directory holds the PatchCamelyon (PCam) dataset files. **Nothing here is
committed to git** (see `.gitignore`) — the dataset is downloaded locally via
`scripts/download_data.py`.

## Dataset: PatchCamelyon (PCam)

- **Official homepage:** https://patchcamelyon.grand-challenge.org/
- **Official repository:** https://github.com/basveeling/pcam
- **Mirror (used by the downloader):** Zenodo, https://zenodo.org/records/2546921
- **Paper:** B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling,
  "Rotation Equivariant CNNs for Digital Pathology," arXiv:1806.03962 (2018).
- **License:** CC0 (public domain), following the license of the source
  CAMELYON16 dataset. This is separate from this repository's own MIT license
  (see root `LICENSE`).

PCam consists of 327,680 color patches (96×96 px) extracted from H&E-stained
lymph node histopathology scans in the CAMELYON16 challenge, each labeled for
presence/absence of metastatic tissue in the central 32×32 px region.

## Why Zenodo instead of Google Drive

The official README links to both a Google Drive folder and a Zenodo mirror
with matching MD5 checksums. `download_data.py` uses the **Zenodo URLs** by
default because they are direct HTTP downloads with no confirmation-page or
quota redirects, which makes scripted, resumable downloading reliable. The
Google Drive folder is documented below as a manual fallback.

## Files

| File | Content | Compressed size | MD5 (of the `.h5.gz`) |
|---|---|---|---|
| `camelyonpatch_level_2_split_train_x.h5.gz` | training images | 6.4 GB | `1571f514728f59376b705fc836ff4b63` |
| `camelyonpatch_level_2_split_train_y.h5.gz` | training labels | 21 KB | `35c2d7259d906cfc8143347bb8e05be7` |
| `camelyonpatch_level_2_split_valid_x.h5.gz` | validation images | 806 MB | `d5b63470df7cfa627aeec8b9dc0c066e` |
| `camelyonpatch_level_2_split_valid_y.h5.gz` | validation labels | 3.0 KB | `2b85f58b927af9964a4c15b8f7e8f179` |
| `camelyonpatch_level_2_split_test_x.h5.gz` | test images | 801 MB | `d8c2d60d490dbd479f8199bdfa0cf6ec` |
| `camelyonpatch_level_2_split_test_y.h5.gz` | test labels | 3.0 KB | `60a7035772fbdb7f34eb86d4420cf66a` |

`download_data.py` downloads these `.h5.gz` files, verifies each MD5 against
the table above, decompresses them to the plain `.h5` files listed in
`configs/experiment.yaml`, and leaves the originals in place (so re-running
the script is a no-op if files already exist and match).

> Note: the checksums above are the ones published in the official PCam
> README, cross-checked against the Zenodo record's own listed MD5s — both
> sources agree, which is the authoritative pairing used here.

## Official split structure and leakage

PCam ships with **fixed, pre-defined train/validation/test splits** — this
project uses them as-is rather than re-slicing the data:

- Train: 262,144 patches (2^18)
- Validation: 32,768 patches (2^15)
- Test: 32,768 patches (2^15)
- Each split is exactly 50/50 positive/negative.

Per the official documentation, **there is no whole-slide-image (WSI) overlap
between the splits** — i.e., patches from any single source slide appear in
only one of train/valid/test. This means the standard patch-level train/eval
protocol used here does not leak slide identity across splits, which is the
main leakage risk to check for in any WSI-derived patch dataset. The
`*_meta.csv` files (also downloaded) record which source slide each patch
came from, which is what enables this to be verified rather than assumed —
`scripts/download_data.py` fetches these alongside the image/label files.

## Manual fallback: Google Drive

If Zenodo is unreachable, all files are also available as a folder at:
https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB

Download manually, place the `.h5.gz` files in this `data/` directory, and
re-run `scripts/download_data.py` — it will detect the existing files,
verify their checksums, and skip re-downloading.
