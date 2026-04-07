#!/usr/bin/env python3
"""Pre-compute soft targets from CellViT-SAM-H teacher model.

Runs the teacher in eval mode (fp16) on all PanNuke patches and saves
raw logits to disk as .npz files. This allows training the student
without keeping the teacher in GPU memory.

Usage:
    PYTHONPATH=vendor/CellViT:$PYTHONPATH uv run python -m cellvit_distill.scripts.precompute_soft_targets

Requires ~6-8 GB VRAM with fp16 on RTX 5060 Ti.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add vendor CellViT to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "CellViT"))


class RawPanNukeDataset(Dataset):
    """Minimal dataset that just loads images from PanNuke .npy files."""

    def __init__(self, data_dir: str, folds: list):
        self._mmaps = []
        self._cumulative = [0]
        for fold_idx in folds:
            fold_dir = Path(data_dir) / f"fold{fold_idx}"
            imgs = np.load(fold_dir / "images.npy", mmap_mode="r")
            self._mmaps.append(imgs)
            self._cumulative.append(self._cumulative[-1] + len(imgs))
        self._total = self._cumulative[-1]
        print(f"Loaded {self._total} patches from folds {folds} (mmap)")

    def __len__(self):
        return self._total

    def _resolve(self, idx):
        for i, mmap in enumerate(self._mmaps):
            if idx < self._cumulative[i + 1]:
                return np.array(mmap[idx - self._cumulative[i]])
        raise IndexError(idx)

    def __getitem__(self, idx):
        img = self._resolve(idx)  # (256, 256, 3)
        # Handle float64 PanNuke data
        if img.dtype == np.float64 or img.dtype == np.float32:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        # Normalize to [0, 1] and CHW
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return idx, img


def load_teacher(checkpoint_path: str, device: torch.device):
    """Load CellViT-SAM-H from checkpoint."""
    from models.segmentation.cell_segmentation.cellvit import CellViTSAM

    model = CellViTSAM(
        model_path="dummy",
        num_nuclei_classes=6,
        num_tissue_classes=19,
        vit_structure="SAM-H",
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).half()  # fp16
    model.eval()
    return model


@torch.no_grad()
def extract_soft_targets(model, dataloader, output_dir: Path, device: torch.device):
    """Run teacher model on all patches and save soft targets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for indices, images in tqdm(dataloader, desc="Computing soft targets"):
        images = images.to(device, dtype=torch.float16)

        outputs = model(images)

        binary_logits = outputs["nuclei_binary_map"].float().cpu().numpy()
        hv_preds = outputs["hv_map"].float().cpu().numpy()
        type_logits = outputs["nuclei_type_map"].float().cpu().numpy()

        for i, global_idx in enumerate(indices):
            np.savez_compressed(
                output_dir / f"{global_idx.item()}.npz",
                binary=binary_logits[i],
                hv=hv_preds[i],
                type=type_logits[i],
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "datasets" / "pannuke"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "CellViT-SAM-H-x40.pth"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "datasets" / "pannuke" / "soft_targets"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = RawPanNukeDataset(args.data_dir, args.folds)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print("Loading CellViT-SAM-H teacher...")
    model = load_teacher(args.checkpoint, device)
    print(f"Teacher: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params (fp16)")

    output_dir = Path(args.output_dir)
    extract_soft_targets(model, dataloader, output_dir, device)
    print(f"\nSoft targets saved: {len(list(output_dir.glob('*.npz')))} files in {output_dir}")


if __name__ == "__main__":
    main()
