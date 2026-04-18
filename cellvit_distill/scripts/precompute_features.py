#!/usr/bin/env python3
"""Pre-compute CellViT-SAM-H intermediate ViT-H tokens for feature distillation.

Runs the teacher in eval mode (fp16) on all PanNuke patches and saves the
deepest ViT-H token map (z4) per patch. For 256x256 input with patch_size=16
the tokens form a 16x16 grid of 1280-dim vectors — naturally spatially
aligned with the student's stage-3 feature map (16x16 grid, 256 channels).

Storage: 7901 patches * (1280*16*16 * 2 bytes fp16) = ~5 GB.

Usage:
    PYTHONPATH=vendor/CellViT:$PYTHONPATH uv run python \\
        -m cellvit_distill.scripts.precompute_features
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "CellViT"))

from cellvit_distill.scripts.precompute_soft_targets import (
    RawPanNukeDataset,
    load_teacher,
)


@torch.no_grad()
def extract_features(model, dataloader, output_dir: Path, device: torch.device):
    """Run teacher and save z4 tokens per image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for indices, images in tqdm(dataloader, desc="Computing ViT-H features"):
        images = images.to(device, dtype=torch.float16)
        outputs = model(images, retrieve_tokens=True)

        # tokens shape: (B, H, W, C) for SAM-H after internal reshape.
        # Check actual shape and permute to (B, C, H, W) if needed.
        tokens = outputs["tokens"]
        if tokens.ndim == 4 and tokens.shape[-1] == 1280:
            tokens = tokens.permute(0, 3, 1, 2).contiguous()
        # Keep fp16 on disk to halve storage.
        tokens = tokens.half().cpu().numpy()

        for i, global_idx in enumerate(indices):
            np.savez_compressed(
                output_dir / f"{global_idx.item()}.npz",
                features=tokens[i],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "datasets" / "pannuke"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "CellViT-SAM-H-x40.pth"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "datasets" / "pannuke" / "soft_features"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = RawPanNukeDataset(args.data_dir, args.folds)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print("Loading CellViT-SAM-H teacher...")
    model = load_teacher(args.checkpoint, device)

    output_dir = Path(args.output_dir)
    extract_features(model, dataloader, output_dir, device)
    print(f"\nFeatures saved: {len(list(output_dir.glob('*.npz')))} files in {output_dir}")


if __name__ == "__main__":
    main()
