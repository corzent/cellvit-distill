#!/usr/bin/env python3
"""Evaluate trained student model on PanNuke test fold.

Loads the best checkpoint from a training run, runs post-processing,
and computes full metrics (bPQ, mPQ, F1).

Usage:
    uv run python -m cellvit_distill.scripts.eval_student \
        --run_dir cellvit_distill/runs/baseline_convnext_tiny_YYYYMMDD_HHMMSS
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellvit_distill.data.pannuke import PanNukeDataset, get_val_transform
from cellvit_distill.models.student import build_student
from cellvit_distill.utils.postprocess import post_process_predictions
from cellvit_distill.utils.metrics import compute_all_metrics


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes=5):
    """Run student on dataset and compute metrics."""
    model.eval()
    all_pred_instances = []
    all_gt_instances = []
    all_pred_types = []
    all_gt_types = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)

        with torch.amp.autocast("cuda"):
            outputs = model(images)

        binary_pred = torch.softmax(outputs["binary"], dim=1)[:, 1]
        hv_pred = outputs["hv_map"]
        type_pred = torch.softmax(outputs["type_map"], dim=1)

        for i in range(images.shape[0]):
            pred_inst, pred_type = post_process_predictions(
                binary_pred[i].float().cpu().numpy(),
                hv_pred[i].float().cpu().numpy(),
                type_pred[i].float().cpu().numpy(),
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
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth",
                        help="Checkpoint filename (default: best_model.pth)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_fold", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_dir = Path(args.run_dir)

    # Load config from run
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Load model
    model = build_student(cfg).to(device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    info = model.count_parameters()
    print(f"Student: {cfg['student']['encoder']} ({info['total_M']:.1f}M params)")
    print(f"Checkpoint: epoch {ckpt['epoch']}")

    # Load test data
    dataset = PanNukeDataset(
        data_dir=cfg["data"]["data_dir"],
        folds=[args.test_fold],
        transform=get_val_transform(),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Evaluate
    metrics = evaluate(model, dataloader, device,
                       num_classes=cfg["data"]["num_classes"] - 1)

    experiment = "distill" if cfg["training"]["distillation"]["enabled"] else "baseline"
    print(f"\n=== {experiment} / {cfg['student']['encoder']} (fold {args.test_fold}) ===")
    print(f"  bPQ:  {metrics['bPQ']:.4f}")
    print(f"  mPQ:  {metrics['mPQ']:.4f}")
    print(f"  F1:   {metrics['F1_detection']:.4f}")
    for c in range(1, cfg["data"]["num_classes"]):
        key = f"PQ_class_{c}"
        if key in metrics:
            name = PanNukeDataset.CLASS_NAMES[c]
            print(f"  PQ {name}: {metrics[key]:.4f}")

    # Save results
    import json
    results_path = run_dir / f"eval_fold{args.test_fold}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
