#!/usr/bin/env python3
"""Evaluate the CellViT-SAM-H teacher on PanNuke fold 3 using pre-computed soft targets.

Since the teacher needs ≥24 GB VRAM for direct inference, we reuse the
pre-computed raw logits stored in soft_targets/. We load those for fold 3
indices (offset by fold1 + fold2 sizes), run the same post-processing +
metric pipeline as eval_student.py, and report mPQ under the fixed protocol.

Usage:
    python -m cellvit_distill.scripts.eval_teacher_cached
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cellvit_distill.data.pannuke import PanNukeDataset, get_val_transform
from cellvit_distill.utils.postprocess import post_process_predictions
from cellvit_distill.utils.metrics import compute_all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/corzent/caspian/thesis/datasets/pannuke")
    parser.add_argument("--soft_targets_dir", default="/home/corzent/caspian/thesis/datasets/pannuke/soft_targets")
    parser.add_argument("--test_fold", type=int, default=3)
    parser.add_argument("--train_folds", type=int, nargs="+", default=[1, 2],
                        help="Folds used as training in precompute order — offset for global idx")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    st_dir = Path(args.soft_targets_dir)

    # Precompute ran folds [1,2,3] in that order → global idx offset for fold N
    # is the sum of sizes of folds [1..N-1].
    offsets = [0]
    for f in [1, 2, 3]:
        n = len(np.load(data_dir / f"fold{f}" / "images.npy", mmap_mode="r"))
        offsets.append(offsets[-1] + n)
    test_offset = offsets[args.test_fold - 1]
    test_n = offsets[args.test_fold] - test_offset
    print(f"Fold {args.test_fold}: {test_n} patches, global idx offset = {test_offset}")

    # Dataset for GT masks (we only need GT types and instance maps)
    ds = PanNukeDataset(
        data_dir=args.data_dir,
        folds=[args.test_fold],
        transform=get_val_transform(),
    )
    assert len(ds) == test_n, f"Dataset size mismatch: {len(ds)} vs {test_n}"

    all_pred_inst, all_gt_inst, all_pred_type, all_gt_type = [], [], [], []

    for local_idx in tqdm(range(test_n), desc="Evaluating teacher"):
        global_idx = test_offset + local_idx
        st_path = st_dir / f"{global_idx}.npz"
        if not st_path.exists():
            raise FileNotFoundError(f"Missing soft target: {st_path}")

        with np.load(st_path) as st:
            binary_logits = st["binary"].astype(np.float32)  # (2, 256, 256)
            hv_pred = st["hv"].astype(np.float32)            # (2, 256, 256)
            type_logits = st["type"].astype(np.float32)      # (6, 256, 256)

        # Apply softmax
        binary_probs = np.exp(binary_logits) / np.exp(binary_logits).sum(axis=0, keepdims=True)
        type_probs = np.exp(type_logits) / np.exp(type_logits).sum(axis=0, keepdims=True)

        pred_inst, pred_type = post_process_predictions(
            binary_probs[1],  # nucleus channel
            hv_pred,
            type_probs,
        )

        sample = ds[local_idx]
        gt_inst = sample["instance_map"].numpy()
        gt_type = sample["type_map"].argmax(dim=0).numpy()

        all_pred_inst.append(pred_inst)
        all_gt_inst.append(gt_inst)
        all_pred_type.append(pred_type)
        all_gt_type.append(gt_type)

    metrics = compute_all_metrics(
        all_pred_inst, all_gt_inst, all_pred_type, all_gt_type, num_classes=5,
    )

    print(f"\n=== CellViT-SAM-H teacher (from cached logits) on fold {args.test_fold} ===")
    print(f"  bPQ: {metrics['bPQ']:.4f}")
    print(f"  mPQ: {metrics['mPQ']:.4f}")
    print(f"  F1:  {metrics['F1_detection']:.4f}")
    for c, name in enumerate(["neoplastic", "inflammatory", "connective", "dead", "epithelial"], start=1):
        key = f"PQ_class_{c}"
        if key in metrics:
            print(f"  PQ {name}: {metrics[key]:.4f}")


if __name__ == "__main__":
    main()
