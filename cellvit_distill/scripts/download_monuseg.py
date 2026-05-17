#!/usr/bin/env python3
"""Download and prepare MoNuSeg 2018 dataset for cross-dataset evaluation.

MoNuSeg is the MICCAI 2018 nuclei segmentation challenge dataset:
  - Train: 30 H&E images at 1000×1000 from 7 organs
  - Test:  14 images, additional held-out organs
  - Annotations: per-image XML with polygon coordinates for each nucleus
  - Single class (nuclei vs background) — no cell-type labels, unlike PanNuke

The original challenge data is gated behind grand-challenge.org registration.
A public mirror exists on Kaggle; the URL below points to a common archive
on the internet but may rot — verify and override with --url if needed.

Usage:
    # Default: pulls test+train archives, extracts to ./datasets/monuseg/
    python -m cellvit_distill.scripts.download_monuseg

    # Custom output dir + URL
    python -m cellvit_distill.scripts.download_monuseg \\
        --output_dir /data/monuseg \\
        --url-train https://... --url-test https://...

After running, structure should be:
    datasets/monuseg/
        train/
            images/  *.tif
            masks/   *.xml  (polygon annotations)
        test/
            images/  *.tif
            masks/   *.xml
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# Placeholder URLs — verify before use. The original challenge data is at
# https://monuseg.grand-challenge.org/Data/ (requires login). The Kaggle
# mirror at https://www.kaggle.com/datasets/tuanledinh/monuseg2018 needs
# the Kaggle CLI.
DEFAULT_URLS = {
    "train": os.environ.get("MONUSEG_TRAIN_URL",
        ""),  # left blank — user must provide via env or --url
    "test": os.environ.get("MONUSEG_TEST_URL", ""),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {dest.name}")
    urlretrieve(url, dest)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _organize(extracted_root: Path, target: Path) -> None:
    """Flatten typical MoNuSeg zip layout into images/ and masks/ subdirs."""
    images = target / "images"
    masks = target / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    for p in extracted_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in (".tif", ".tiff"):
            dest = images / p.name
            if not dest.exists():
                p.rename(dest)
        elif p.suffix.lower() == ".xml":
            dest = masks / p.name
            if not dest.exists():
                p.rename(dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=Path("./datasets/monuseg"))
    p.add_argument("--url-train", default=DEFAULT_URLS["train"])
    p.add_argument("--url-test", default=DEFAULT_URLS["test"])
    p.add_argument("--skip", choices=("none", "train", "test"), default="none",
                   help="Skip downloading a split (use if you placed it manually).")
    args = p.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = output_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    for split, url in (("train", args.url_train), ("test", args.url_test)):
        if args.skip == split:
            print(f"[{split}] skipped (--skip)")
            continue
        target = output_dir / split
        if (target / "images").exists() and any((target / "images").iterdir()):
            print(f"[{split}] already present at {target}, skipping download")
            continue
        if not url:
            print(f"[{split}] no URL provided; set MONUSEG_{split.upper()}_URL or "
                  f"--url-{split}, or place data manually at {target}/images, "
                  f"{target}/masks (see module docstring).", file=sys.stderr)
            continue
        zip_path = tmp / f"{split}.zip"
        if not zip_path.exists():
            print(f"[{split}] downloading {url}")
            _download(url, zip_path)
        print(f"[{split}] extracting to {target}")
        extract_dir = tmp / split
        extract_dir.mkdir(exist_ok=True)
        _extract_zip(zip_path, extract_dir)
        _organize(extract_dir, target)

    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)

    for split in ("train", "test"):
        d = output_dir / split / "images"
        n = len(list(d.glob("*.tif"))) + len(list(d.glob("*.tiff"))) if d.exists() else 0
        print(f"  {split}: {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
