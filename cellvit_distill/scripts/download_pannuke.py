#!/usr/bin/env python3
"""Download and prepare PanNuke dataset.

PanNuke is hosted on Warwick's servers. This script downloads all 3 folds
and organizes them into the expected directory structure.

Usage:
    python -m scripts.download_pannuke --output_dir /path/to/pannuke
"""

import argparse
import os
import zipfile
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Install dependencies: pip install requests tqdm")
    exit(1)


# PanNuke download URLs (from TIA Warwick)
PANNUKE_URLS = {
    "fold1": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_1.zip",
    "fold2": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_2.zip",
    "fold3": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke/fold_3.zip",
}


def download_file(url: str, dest: Path, chunk_size: int = 8192):
    """Download file with progress bar."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_and_organize(zip_path: Path, fold_name: str, output_dir: Path):
    """Extract zip and organize into fold directory."""
    fold_dir = output_dir / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(fold_dir)

    # PanNuke zips contain images.npy, masks.npy, types.npy
    # but sometimes in subdirectories - flatten if needed
    for npy_file in fold_dir.rglob("*.npy"):
        if npy_file.parent != fold_dir:
            target = fold_dir / npy_file.name
            if not target.exists():
                npy_file.rename(target)

    print(f"  {fold_name}: {list(fold_dir.glob('*.npy'))}")


def main():
    parser = argparse.ArgumentParser(description="Download PanNuke dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/corzent/caspian/thesis/datasets/pannuke",
        help="Output directory",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    for fold_idx in args.folds:
        fold_name = f"fold{fold_idx}"
        fold_dir = output_dir / fold_name

        # Check if already downloaded
        if (fold_dir / "images.npy").exists():
            print(f"{fold_name} already exists, skipping.")
            continue

        url = PANNUKE_URLS[fold_name]
        zip_path = tmp_dir / f"{fold_name}.zip"

        if not zip_path.exists():
            print(f"Downloading {fold_name}...")
            download_file(url, zip_path)

        extract_and_organize(zip_path, fold_name, output_dir)

    # Cleanup
    import shutil
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\nDone! Dataset structure:")
    for fold_idx in args.folds:
        fold_dir = output_dir / f"fold{fold_idx}"
        files = list(fold_dir.glob("*.npy"))
        print(f"  fold{fold_idx}/: {[f.name for f in files]}")


if __name__ == "__main__":
    main()
