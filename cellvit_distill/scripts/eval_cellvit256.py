#!/usr/bin/env python3
"""Evaluate CellViT-256 on PanNuke test fold (Experiment 3).

Reference model for comparison: CellViT-256 (ViT-256 encoder, ~46M params).
Runs inference on fold 3, post-processes, and computes PQ/mPQ/F1 metrics.

Usage:
    PYTHONPATH=vendor/CellViT:$PYTHONPATH uv run python -m cellvit_distill.scripts.eval_cellvit256
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

from cellvit_distill.data.pannuke import PanNukeDataset, get_val_transform
from cellvit_distill.utils.postprocess import post_process_predictions
from cellvit_distill.utils.metrics import compute_all_metrics


def load_cellvit256(checkpoint_path: str, device: torch.device):
    """Load CellViT-256 model from checkpoint."""
    from models.segmentation.cell_segmentation.cellvit import CellViT256

    model = CellViT256(
        model256_path="dummy",
        num_nuclei_classes=6,
        num_tissue_classes=19,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).half()
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"CellViT-256: {num_params / 1e6:.1f}M params (fp16)")
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=5):
    """Run model on dataset and compute metrics."""
    all_pred_instances = []
    all_gt_instances = []
    all_pred_types = []
    all_gt_types = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device, dtype=torch.float16)
        outputs = model(images)

        binary_pred = torch.softmax(outputs["nuclei_binary_map"].float(), dim=1)[:, 1]
        hv_pred = outputs["hv_map"].float()
        type_pred = torch.softmax(outputs["nuclei_type_map"].float(), dim=1)

        for i in range(images.shape[0]):
            pred_inst, pred_type = post_process_predictions(
                binary_pred[i].cpu().numpy(),
                hv_pred[i].cpu().numpy(),
                type_pred[i].cpu().numpy(),
            )
            gt_inst = batch["instance_map"][i].numpy()
            gt_type = batch["type_map"][i].argmax(dim=0).numpy()

            all_pred_instances.append(pred_inst)
            all_gt_instances.append(gt_inst)
            all_pred_types.append(pred_type)
            all_gt_types.append(gt_type)

    metrics = compute_all_metrics(
        all_pred_instances, all_gt_instances,
        all_pred_types, all_gt_types,
        num_classes=num_classes,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "datasets" / "pannuke"))
    parser.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "CellViT-256-x20.pth"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_fold", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    dataset = PanNukeDataset(
        data_dir=args.data_dir,
        folds=[args.test_fold],
        transform=get_val_transform(),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # Load model
    model = load_cellvit256(args.checkpoint, device)

    # Evaluate
    metrics = evaluate(model, dataloader, device)

    print("\n=== CellViT-256 Results (fold {}) ===".format(args.test_fold))
    print(f"  bPQ:  {metrics['bPQ']:.4f}")
    print(f"  mPQ:  {metrics['mPQ']:.4f}")
    print(f"  F1:   {metrics['F1_detection']:.4f}")
    for c in range(1, 6):
        key = f"PQ_class_{c}"
        if key in metrics:
            print(f"  PQ class {c}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
