"""
Download and verify the PatchCamelyon (PCam) dataset.

Source: official Zenodo mirror (https://zenodo.org/records/2546921), as
documented in data/README.md. Files are gzip-compressed HDF5; this script
downloads each `.h5.gz`, verifies its MD5 checksum against the official
values, decompresses it to the plain `.h5` file used by the rest of the
pipeline, and skips any step that has already completed successfully.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --data-dir data --splits train valid test
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_BASE = "https://zenodo.org/records/2546921/files"

# (filename, md5 of the .h5.gz, approximate size in bytes for progress bars)
PCAM_FILES: list["FileSpec"] = []


@dataclass(frozen=True)
class FileSpec:
    filename: str  # e.g. camelyonpatch_level_2_split_train_x.h5.gz
    md5: str
    approx_size: int  # bytes, for a sane default progress bar total


PCAM_FILES = [
    FileSpec("camelyonpatch_level_2_split_train_x.h5.gz", "1571f514728f59376b705fc836ff4b63", 6_400_000_000),
    FileSpec("camelyonpatch_level_2_split_train_y.h5.gz", "35c2d7259d906cfc8143347bb8e05be7", 21_400),
    FileSpec("camelyonpatch_level_2_split_valid_x.h5.gz", "d5b63470df7cfa627aeec8b9dc0c066e", 806_000_000),
    FileSpec("camelyonpatch_level_2_split_valid_y.h5.gz", "2b85f58b927af9964a4c15b8f7e8f179", 3_000),
    FileSpec("camelyonpatch_level_2_split_test_x.h5.gz", "d8c2d60d490dbd479f8199bdfa0cf6ec", 801_000_000),
    FileSpec("camelyonpatch_level_2_split_test_y.h5.gz", "60a7035772fbdb7f34eb86d4420cf66a", 3_000),
]

# Meta CSVs are small and useful for split/leakage documentation (see
# data/README.md); not required for the core pipeline but downloaded for
# completeness and reproducibility of any leakage checks.
PCAM_META_FILES = [
    "camelyonpatch_level_2_split_train_meta.csv",
    "camelyonpatch_level_2_split_valid_meta.csv",
    "camelyonpatch_level_2_split_test_meta.csv",
]

SPLIT_TO_FILES = {
    "train": ["camelyonpatch_level_2_split_train_x.h5.gz", "camelyonpatch_level_2_split_train_y.h5.gz"],
    "valid": ["camelyonpatch_level_2_split_valid_x.h5.gz", "camelyonpatch_level_2_split_valid_y.h5.gz"],
    "test": ["camelyonpatch_level_2_split_test_x.h5.gz", "camelyonpatch_level_2_split_test_y.h5.gz"],
}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the MD5 checksum of a file without loading it fully into RAM."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, expected_md5: str, approx_size: int) -> None:
    """
    Download `url` to `dest` with resume support. If `dest` already exists and
    matches `expected_md5`, does nothing. If it exists but is incomplete or
    corrupt, resumes (or restarts) as appropriate.
    """
    if dest.exists():
        print(f"  Found existing file: {dest.name} — verifying checksum...")
        if md5sum(dest) == expected_md5:
            print(f"  ✓ Checksum OK, skipping download of {dest.name}")
            return
        else:
            print(f"  ✗ Checksum mismatch for {dest.name}, re-downloading.")
            dest.unlink()

    headers = {}
    resume_pos = 0
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        resume_pos = tmp_path.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"

    try:
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if r.status_code not in (200, 206):
                raise RuntimeError(
                    f"Unexpected HTTP status {r.status_code} when downloading {url}"
                )
            total = int(r.headers.get("content-length", 0)) + resume_pos
            total = total if total > 0 else approx_size
            mode = "ab" if resume_pos else "wb"
            with open(tmp_path, mode) as f, tqdm(
                total=total,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                desc=f"  {dest.name}",
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except requests.RequestException as e:
        raise RuntimeError(
            f"Download failed for {url}\n"
            f"Reason: {e}\n"
            f"You can retry (the partial download will resume), or fetch this "
            f"file manually from the Google Drive mirror listed in data/README.md."
        ) from e

    print(f"  Verifying checksum for {dest.name}...")
    actual_md5 = md5sum(tmp_path)
    if actual_md5 != expected_md5:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch after download for {dest.name}.\n"
            f"Expected: {expected_md5}\nGot:      {actual_md5}\n"
            f"The partial/corrupt file was removed. Please re-run this script."
        )
    tmp_path.rename(dest)
    print(f"  ✓ Downloaded and verified {dest.name}")


def decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .h5.gz file to .h5, skipping if the output already exists."""
    if out_path.exists():
        print(f"  ✓ {out_path.name} already decompressed, skipping.")
        return
    print(f"  Decompressing {gz_path.name} -> {out_path.name} ...")
    tmp_out = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with gzip.open(gz_path, "rb") as f_in, open(tmp_out, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
    except OSError as e:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(
            f"Decompression failed for {gz_path.name}: {e}\n"
            f"The downloaded .gz file may be corrupt — delete it and re-run "
            f"this script to re-download."
        ) from e
    tmp_out.rename(out_path)
    print(f"  ✓ Decompressed {out_path.name}")


def download_meta_files(data_dir: Path) -> None:
    for fname in PCAM_META_FILES:
        dest = data_dir / fname
        if dest.exists():
            print(f"  ✓ {fname} already present, skipping.")
            continue
        url = f"{ZENODO_BASE}/{fname}?download=1"
        print(f"  Downloading {fname} ...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                tmp.rename(dest)
            print(f"  ✓ {fname}")
        except requests.RequestException as e:
            print(f"  ✗ Could not download {fname} ({e}). "
                  f"This file is optional metadata — continuing.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the PCam dataset.")
    parser.add_argument("--data-dir", default="data", help="Directory to store dataset files in.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Which splits to download (default: all three).",
    )
    parser.add_argument(
        "--skip-meta",
        action="store_true",
        help="Skip downloading the *_meta.csv slide-provenance files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    wanted_files = []
    for split in args.splits:
        wanted_files.extend(SPLIT_TO_FILES[split])
    specs = [f for f in PCAM_FILES if f.filename in wanted_files]

    print(f"PatchCamelyon download — splits: {args.splits}")
    print(f"Target directory: {data_dir.resolve()}\n")

    for spec in specs:
        gz_path = data_dir / spec.filename
        h5_name = spec.filename[: -len(".gz")]  # strip .gz
        h5_path = data_dir / h5_name

        print(f"[{spec.filename}]")
        if h5_path.exists():
            print(f"  ✓ {h5_name} already exists, skipping download+decompress.")
            continue

        url = f"{ZENODO_BASE}/{spec.filename}?download=1"
        download_with_resume(url, gz_path, spec.md5, spec.approx_size)
        decompress_gz(gz_path, h5_path)

    if not args.skip_meta:
        print("\n[metadata CSVs]")
        download_meta_files(data_dir)

    print("\nAll requested PCam files are present and verified in:", data_dir.resolve())
    print("Run the data sanity check next:")
    print("  python -m src.data.dataset --sanity-check")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
