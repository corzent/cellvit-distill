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
from cellvit_distill.utils.postprocess import (
    post_process_batch_parallel,
    post_process_predictions,
    make_postprocess_pool,
)
from cellvit_distill.utils.metrics import compute_all_metrics


def _tta_predict(model, images: torch.Tensor):
    """8-way TTA: {identity, hflip, vflip, hvflip} x {rot0, rot90}.

    HV maps need sign correction under reflections and swapping under 90° rotation:
      hflip: h <- -h
      vflip: v <- -v
      rot90 (CCW): (h, v) -> (v, -h)
    Returns averaged binary_prob (B, H, W), hv (B, 2, H, W), type_prob (B, C, H, W).
    """
    accum_bin = 0.0
    accum_hv = 0.0
    accum_type = 0.0
    n = 0

    def forward_once(x):
        with torch.amp.autocast("cuda"):
            out = model(x)
        return (
            torch.softmax(out["binary"].float(), dim=1)[:, 1],
            out["hv_map"].float(),
            torch.softmax(out["type_map"].float(), dim=1),
        )

    # Identity + 3 flips + 90° rotation of each (8 total)
    for do_rot in (False, True):
        x = torch.rot90(images, k=1, dims=(-2, -1)) if do_rot else images
        for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
            xi = x
            if hflip:
                xi = torch.flip(xi, dims=(-1,))
            if vflip:
                xi = torch.flip(xi, dims=(-2,))

            b, hv, t = forward_once(xi)

            # Undo spatial transforms, inverting in reverse order
            if vflip:
                b = torch.flip(b, dims=(-2,))
                hv = torch.flip(hv, dims=(-2,))
                hv[:, 1] = -hv[:, 1]  # vertical component flips sign
                t = torch.flip(t, dims=(-2,))
            if hflip:
                b = torch.flip(b, dims=(-1,))
                hv = torch.flip(hv, dims=(-1,))
                hv[:, 0] = -hv[:, 0]  # horizontal component flips sign
                t = torch.flip(t, dims=(-1,))
            if do_rot:
                # rot90(k=1) was CCW; undo with rot90(k=-1)
                b = torch.rot90(b, k=-1, dims=(-2, -1))
                hv = torch.rot90(hv, k=-1, dims=(-2, -1))
                # Under CCW rot90, (h, v) mapped to (v, -h); inverse: (h, v) = (-v_rot, h_rot)
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
    """Run student on dataset and compute metrics.

    n_workers_post: size of multiprocessing pool for per-image post-processing
        (HV-watershed + Hungarian matching). 0 disables parallelism.
    """
    model.eval()
    all_pred_instances = []
    all_gt_instances = []
    all_pred_types = []
    all_gt_types = []

    pool = make_postprocess_pool(n_workers_post) if n_workers_post > 0 else None
    try:
        for batch in tqdm(dataloader, desc="Evaluating" + (" (TTA)" if tta else "")):
            images = batch["image"].to(device)

            if tta:
                binary_pred, hv_pred, type_pred = _tta_predict(model, images)
            else:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                binary_pred = torch.softmax(outputs["binary"].float(), dim=1)[:, 1]
                hv_pred = outputs["hv_map"].float()
                type_pred = torch.softmax(outputs["type_map"].float(), dim=1)

            binary_np = binary_pred.cpu().numpy()
            hv_np = hv_pred.cpu().numpy()
            type_np = type_pred.cpu().numpy()

            if pool is not None:
                results = post_process_batch_parallel(binary_np, hv_np, type_np, pool)
            else:
                results = [
                    post_process_predictions(binary_np[i], hv_np[i], type_np[i])
                    for i in range(images.shape[0])
                ]

            for i, (pred_inst, pred_type) in enumerate(results):
                gt_inst = batch["instance_map"][i].numpy()
                gt_type = batch["type_map"][i].argmax(dim=0).numpy()

                all_pred_instances.append(pred_inst)
                all_gt_instances.append(gt_inst)
                all_pred_types.append(pred_type)
                all_gt_types.append(gt_type)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

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
    parser.add_argument("--tta", action="store_true",
                        help="Enable 8-way test-time augmentation (flips + 90° rotation)")
    parser.add_argument("--n_workers_post", type=int, default=32,
                        help="Worker processes for parallel post-processing (0 = serial)")
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
                       num_classes=cfg["data"]["num_classes"] - 1,
                       tta=args.tta,
                       n_workers_post=args.n_workers_post)

    experiment = "distill" if cfg["training"]["distillation"]["enabled"] else "baseline"
    tta_tag = " [TTA]" if args.tta else ""
    print(f"\n=== {experiment} / {cfg['student']['encoder']} (fold {args.test_fold}){tta_tag} ===")
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
    tta_suffix = "_tta" if args.tta else ""
    results_path = run_dir / f"eval_fold{args.test_fold}{tta_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
