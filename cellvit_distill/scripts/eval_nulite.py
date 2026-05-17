#!/usr/bin/env python3
"""Evaluate the released NuLite-T checkpoint on PanNuke under our protocol.

NuLite-T (Tommasino et al., 2024) is the concurrent peer at the same
parameter budget (12.05M, FastViT-S12). The released weights at
https://zenodo.org/records/13272655 were trained on the FULL PanNuke
dataset (all 3 folds), so evaluating them on any single fold here is
**not a held-out test** — there is data leakage between train and eval.
The number this script produces is therefore an *upper bound* on the
fair comparison, useful as a roof against which to position our own
3-fold CV numbers.

Differences vs our eval_student.py:
- Constructs NuLite model class from vendor/NuLite/models/nulite.py
- Uses NuLite's input normalization mean=std=[0.5, 0.5, 0.5] (NOT ImageNet)
- Reuses our metrics + parallel post-processing for apples-to-apples
- Output naming: nuclei_binary_map / nuclei_type_map / hv_map (NuLite) is
  mapped to our binary / type_map / hv_map convention before post-proc.

Usage:
    PYTHONPATH=vendor/NuLite:vendor/CellViT:$PYTHONPATH \
        python -m cellvit_distill.scripts.eval_nulite \
        --checkpoint vendor/nulite_checkpoints/NuLite-T-Weights.pth \
        --test_fold 1 --tta
"""

import argparse
import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure NuLite is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "NuLite"))

from cellvit_distill.data.pannuke import PanNukeDataset
from cellvit_distill.utils.postprocess import (
    make_postprocess_pool,
    post_process_batch_parallel,
)
from cellvit_distill.utils.metrics import compute_all_metrics


def get_nulite_val_transform() -> A.Compose:
    """NuLite uses mean=std=[0.5, 0.5, 0.5] normalization (not ImageNet)."""
    return A.Compose([
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


def _tta_predict_nulite(model, images):
    """8-way TTA wrapper for NuLite forward (returns dict with same keys)."""
    accum_bin = 0.0
    accum_hv = 0.0
    accum_type = 0.0
    n = 0

    def forward_once(x):
        with torch.amp.autocast("cuda"):
            out = model(x)
        return (
            torch.softmax(out["nuclei_binary_map"].float(), dim=1)[:, 1],
            out["hv_map"].float(),
            torch.softmax(out["nuclei_type_map"].float(), dim=1),
        )

    for do_rot in (False, True):
        x = torch.rot90(images, k=1, dims=(-2, -1)) if do_rot else images
        for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
            xi = x
            if hflip:
                xi = torch.flip(xi, dims=(-1,))
            if vflip:
                xi = torch.flip(xi, dims=(-2,))

            b, hv, t = forward_once(xi)

            if vflip:
                b = torch.flip(b, dims=(-2,))
                hv = torch.flip(hv, dims=(-2,))
                hv[:, 1] = -hv[:, 1]
                t = torch.flip(t, dims=(-2,))
            if hflip:
                b = torch.flip(b, dims=(-1,))
                hv = torch.flip(hv, dims=(-1,))
                hv[:, 0] = -hv[:, 0]
                t = torch.flip(t, dims=(-1,))
            if do_rot:
                b = torch.rot90(b, k=-1, dims=(-2, -1))
                hv = torch.rot90(hv, k=-1, dims=(-2, -1))
                h_rot, v_rot = hv[:, 0].clone(), hv[:, 1].clone()
                hv[:, 0] = -v_rot
                hv[:, 1] = h_rot
                t = torch.rot90(t, k=-1, dims=(-2, -1))

            accum_bin = accum_bin + b
            accum_hv = accum_hv + hv
            accum_type = accum_type + t
            n += 1

    return accum_bin / n, accum_hv / n, accum_type / n


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=5, tta=False, n_workers_post=32):
    model.eval()
    all_pred_instances = []
    all_gt_instances = []
    all_pred_types = []
    all_gt_types = []

    pool = make_postprocess_pool(n_workers_post) if n_workers_post > 0 else None
    try:
        for batch in tqdm(dataloader, desc="Evaluating NuLite" + (" (TTA)" if tta else "")):
            images = batch["image"].to(device)

            if tta:
                binary_pred, hv_pred, type_pred = _tta_predict_nulite(model, images)
            else:
                with torch.amp.autocast("cuda"):
                    out = model(images)
                binary_pred = torch.softmax(out["nuclei_binary_map"].float(), dim=1)[:, 1]
                hv_pred = out["hv_map"].float()
                type_pred = torch.softmax(out["nuclei_type_map"].float(), dim=1)

            binary_np = binary_pred.cpu().numpy()
            hv_np = hv_pred.cpu().numpy()
            type_np = type_pred.cpu().numpy()

            results = post_process_batch_parallel(binary_np, hv_np, type_np, pool)

            for i, (pred_inst, pred_type) in enumerate(results):
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
            pool=pool,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="vendor/nulite_checkpoints/NuLite-T-Weights.pth")
    parser.add_argument("--data_dir", type=str,
                        default="/workspace/cellvit-distill/datasets/pannuke")
    parser.add_argument("--test_fold", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n_workers_post", type=int, default=32)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--output_dir", type=str,
                        default="logs/nulite_eval",
                        help="Where to save eval_fold{N}{,_tta}.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Construct NuLite-T and load weights
    from models.nulite import NuLite
    model = NuLite(num_nuclei_classes=6, num_tissue_classes=19, vit_structure="fastvit_s12")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    res = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"NuLite-T: {n_params/1e6:.2f}M params")
    print(f"Eval fold: {args.test_fold} (NOTE: data leakage — released ckpt trained on all 3 folds)")

    # Dataset with NuLite normalization
    dataset = PanNukeDataset(
        data_dir=args.data_dir,
        folds=[args.test_fold],
        transform=get_nulite_val_transform(),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    metrics = evaluate(model, dataloader, device,
                       num_classes=5, tta=args.tta,
                       n_workers_post=args.n_workers_post)

    tta_tag = " [TTA]" if args.tta else ""
    print(f"\n=== NuLite-T released checkpoint on fold {args.test_fold}{tta_tag} ===")
    print(f"  bPQ:  {metrics['bPQ']:.4f}")
    print(f"  mPQ:  {metrics['mPQ']:.4f}")
    print(f"  F1:   {metrics['F1_detection']:.4f}")
    class_names = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    for c, name in enumerate(class_names, start=1):
        key = f"PQ_class_{c}"
        if key in metrics:
            print(f"  PQ {name}: {metrics[key]:.4f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tta_suffix = "_tta" if args.tta else ""
    out_path = out_dir / f"nulite_t_fold{args.test_fold}{tta_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
