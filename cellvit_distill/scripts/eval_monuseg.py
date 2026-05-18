#!/usr/bin/env python3
"""Zero-shot cross-dataset evaluation on MoNuSeg.

Loads a student trained on PanNuke (--run_dir with best_model.pth +
config.yaml), runs full-image inference via overlap-averaged 256×256
patches, post-processes with HV-watershed, and computes binary panoptic
quality + F1-detection against MoNuSeg GT instance masks.

Only binary (nuclei vs background) metrics are reported — MoNuSeg has no
cell-type labels, so the type head is ignored.

Output: JSON with {bPQ, F1_detection, per_image_bpq, per_image_f1} and a
.npz with per-image arrays for use by scripts/stat_test.py.

Usage:
    python -m cellvit_distill.scripts.eval_monuseg \\
        --run_dir cellvit_distill/runs/distill_fastvit_s12_... \\
        [--split test] [--tta] [--stride 128]

The first invocation downloads RationAI/MoNuSeg (~96 MB) into HF cache.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from cellvit_distill.data.monuseg import MoNuSegDataset, _tile_indices
from cellvit_distill.models.student import build_student
from cellvit_distill.utils.postprocess import (
    post_process_predictions,
    make_postprocess_pool,
)
from cellvit_distill.utils.metrics import panoptic_quality


def _forward_patches(model, patches: torch.Tensor, device, tta: bool):
    """Run model on a (B, 3, 256, 256) uint8 patch batch.

    Returns binary_prob (B, H, W), hv (B, 2, H, W). type_prob is computed
    but not returned — type head is meaningless on MoNuSeg.
    """
    # Normalize to [0, 1] same as PanNukeDataset.get_val_transform.
    x = patches.float().permute(0, 3, 1, 2) / 255.0
    x = x.to(device)
    if tta:
        # Simple 4-way flip TTA (no rotation — keeps HV-map sign correction simple)
        accum_b, accum_hv, n = 0.0, 0.0, 0
        for hflip, vflip in [(False, False), (True, False), (False, True), (True, True)]:
            xi = x
            if hflip:
                xi = torch.flip(xi, dims=(-1,))
            if vflip:
                xi = torch.flip(xi, dims=(-2,))
            with torch.amp.autocast("cuda"):
                out = model(xi)
            b = torch.softmax(out["binary"].float(), dim=1)[:, 1]
            hv = out["hv_map"].float()
            if vflip:
                b = torch.flip(b, dims=(-2,))
                hv = torch.flip(hv, dims=(-2,))
                hv[:, 1] = -hv[:, 1]
            if hflip:
                b = torch.flip(b, dims=(-1,))
                hv = torch.flip(hv, dims=(-1,))
                hv[:, 0] = -hv[:, 0]
            accum_b = accum_b + b
            accum_hv = accum_hv + hv
            n += 1
        return accum_b / n, accum_hv / n
    with torch.amp.autocast("cuda"):
        out = model(x)
    b = torch.softmax(out["binary"].float(), dim=1)[:, 1]
    hv = out["hv_map"].float()
    return b, hv


def _stitch_image(model, ds: MoNuSegDataset, img_idx: int, device,
                  tta: bool, batch_size: int = 16) -> tuple:
    """Run full-image inference via overlap-averaged patches.

    Returns (binary_prob, hv_map, gt_instance) all at full image resolution.
    """
    img, gt, _patient, _tissue = ds._load(img_idx)
    H, W = img.shape[:2]
    ps = ds.patch_size
    stride = ds.stride
    coords = _tile_indices(H, W, ps, stride)

    accum_b = np.zeros((H, W), dtype=np.float32)
    accum_hv = np.zeros((2, H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    # Cosine window for soft blending at patch borders
    win = np.outer(
        0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ps)),
        0.5 - 0.5 * np.cos(np.linspace(0, np.pi, ps)),
    ).astype(np.float32) + 1e-3  # ensure positive

    # Batch through patches
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i+batch_size]
        patches = np.stack([img[y:y+ps, x:x+ps] for y, x in chunk])
        patches_t = torch.from_numpy(patches)
        b, hv = _forward_patches(model, patches_t, device, tta=tta)
        b = b.cpu().numpy()
        hv = hv.cpu().numpy()
        for j, (y, x) in enumerate(chunk):
            accum_b[y:y+ps, x:x+ps] += b[j] * win
            accum_hv[:, y:y+ps, x:x+ps] += hv[j] * win
            weight[y:y+ps, x:x+ps] += win

    binary_prob = accum_b / weight
    hv_map = accum_hv / weight
    return binary_prob, hv_map, gt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, default="best_model.pth")
    p.add_argument("--split", default="test", choices=("train", "test"),
                   help="HF dataset split (default: test, n=14 images)")
    p.add_argument("--cache_dir", default=None,
                   help="HF cache dir; defaults to $HF_HOME or ~/.cache/huggingface")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--tta", action="store_true",
                   help="4-way flip TTA per patch (skip rotation — HV invariance simpler)")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_name", type=str, default=None,
                   help="Eval output filename suffix (default: monuseg or monuseg_tta)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    run_dir = args.run_dir.resolve()
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_student(cfg).to(device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    info = model.count_parameters()
    print(f"Student: {cfg['student']['encoder']} ({info['total_M']:.1f}M params)")
    print(f"Checkpoint: epoch {ckpt['epoch']}")

    ds = MoNuSegDataset(split=args.split, patch_size=args.patch_size,
                        stride=args.stride, cache_dir=args.cache_dir,
                        load_masks=True)
    print(f"MoNuSeg ({args.split}): {ds.num_images} images, {len(ds)} patches")

    per_image_bpq = []
    per_image_f1 = []
    image_names = []
    pool = make_postprocess_pool(4)

    try:
        with torch.no_grad():
            for img_idx in tqdm(range(ds.num_images), desc="MoNuSeg"):
                binary_prob, hv_map, gt = _stitch_image(
                    model, ds, img_idx, device, tta=args.tta, batch_size=args.batch_size
                )
                # Post-process at full image scale — dummy type prob (all background).
                type_prob = np.zeros((cfg["data"]["num_classes"],) + binary_prob.shape, dtype=np.float32)
                type_prob[0] = 1.0
                pred_inst, _ = post_process_predictions(binary_prob, hv_map, type_prob)

                # Binary panoptic quality
                r = panoptic_quality(pred_inst, gt)
                per_image_bpq.append(r["PQ"])
                per_image_f1.append(r["DQ"])
                image_names.append(ds.image_names()[img_idx])
    finally:
        pool.close()
        pool.join()

    bpq = float(np.mean(per_image_bpq)) if per_image_bpq else 0.0
    f1 = float(np.mean(per_image_f1)) if per_image_f1 else 0.0
    print(f"\nMoNuSeg (zero-shot, n={len(per_image_bpq)} images, "
          f"{'TTA' if args.tta else 'plain'}):")
    print(f"  bPQ:  {bpq:.4f}")
    print(f"  F1:   {f1:.4f}")

    suffix = args.output_name or ("monuseg_tta" if args.tta else "monuseg")
    out_json = run_dir / f"eval_{suffix}.json"
    with open(out_json, "w") as fjs:
        json.dump({
            "bPQ": bpq, "F1_detection": f1,
            "n_images": len(per_image_bpq),
            "tta": args.tta,
            "stride": args.stride,
        }, fjs, indent=2)
    npz_path = run_dir / f"eval_{suffix}_per_image.npz"
    np.savez_compressed(
        npz_path,
        bpq=np.asarray(per_image_bpq, dtype=np.float64),
        f1=np.asarray(per_image_f1, dtype=np.float64),
        names=np.asarray(image_names),
    )
    print(f"  json: {out_json}")
    print(f"  npz : {npz_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
