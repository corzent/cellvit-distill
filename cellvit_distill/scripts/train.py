#!/usr/bin/env python3
"""Main training script for CellViT distillation experiments.

Experiment 1: Student from scratch (distillation.enabled = false)
Experiment 2: Student with distillation (distillation.enabled = true)

Usage:
    # Experiment 1: Student from scratch
    python -m scripts.train --config configs/default.yaml

    # Experiment 2: With distillation
    python -m scripts.train --config configs/default.yaml \
        --override training.distillation.enabled=true

    # Different encoder
    python -m scripts.train --config configs/default.yaml \
        --override student.encoder=resnet50
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cellvit_distill.data.pannuke import PanNukeDataset, get_train_transform, get_val_transform
from cellvit_distill.models.student import build_student
from cellvit_distill.utils.losses import CombinedLoss
from cellvit_distill.utils.postprocess import post_process_predictions
from cellvit_distill.utils.metrics import compute_all_metrics


def compute_sample_weights(dataset: PanNukeDataset, num_classes: int = 5) -> np.ndarray:
    """Inverse-frequency sample weights for class-balanced sampling.

    For each patch, weight = max inverse-frequency over classes present.
    Patches containing rare classes (e.g., Dead) get higher sampling weight,
    encouraging every epoch to see more examples of under-represented classes.
    """
    # Count per-class nucleus presence across dataset.
    class_presence = np.zeros(num_classes, dtype=np.int64)
    per_patch_classes = []
    print("Computing class-balanced sample weights...")
    for idx in range(len(dataset)):
        _, _, masks_mmap, _ = dataset._resolve_index(idx)
        local_idx = idx - dataset._cumulative_sizes[
            next(i for i in range(len(dataset._cumulative_sizes) - 1)
                 if idx < dataset._cumulative_sizes[i + 1])
        ]
        masks = masks_mmap[local_idx]
        present = np.zeros(num_classes, dtype=bool)
        for c in range(num_classes):
            if np.any(masks[:, :, c] > 0):
                present[c] = True
                class_presence[c] += 1
        per_patch_classes.append(present)

    # Inverse frequency per class
    total = len(dataset)
    inv_freq = total / (class_presence + 1)  # +1 to avoid div by zero

    # Per-patch weight = max inverse frequency over classes present; 1.0 for empty
    weights = np.ones(total, dtype=np.float64)
    for i, present in enumerate(per_patch_classes):
        if present.any():
            weights[i] = float(inv_freq[present].max())

    print(f"Class presence: {class_presence.tolist()}")
    print(f"Sample weight stats: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
    return weights


def load_config(config_path: str, overrides: list = None) -> dict:
    """Load YAML config with optional command-line overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            # Type inference
            try:
                value = yaml.safe_load(value)
            except Exception:
                pass
            d[keys[-1]] = value

    return cfg


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> dict:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        # Move to device
        images = batch["image"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)

        scaler.scale(losses["total"]).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses
        for k, v in losses.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0.0
            epoch_losses[k] += v.item()
        num_batches += 1

        if (batch_idx + 1) % 50 == 0:
            avg_total = epoch_losses["total"] / num_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] loss: {avg_total:.4f}")

    return {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CombinedLoss,
    device: torch.device,
    num_classes: int = 5,
) -> dict:
    """Validate: compute loss + PQ/F1 metrics."""
    model.eval()
    epoch_losses = {}
    num_batches = 0

    all_pred_instances = []
    all_gt_instances = []
    all_pred_types = []
    all_gt_types = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k != "image"}

        with autocast("cuda"):
            outputs = model(images)
            losses = criterion(outputs, targets)

        for k, v in losses.items():
            if k not in epoch_losses:
                epoch_losses[k] = 0.0
            epoch_losses[k] += v.item()
        num_batches += 1

        # Post-process predictions for metric computation (cast to fp32 for scipy)
        binary_pred = torch.softmax(outputs["binary"].float(), dim=1)[:, 1]  # nucleus prob
        hv_pred = outputs["hv_map"].float()
        type_pred = torch.softmax(outputs["type_map"].float(), dim=1)

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

    avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}

    # Compute metrics
    metrics = compute_all_metrics(
        all_pred_instances, all_gt_instances,
        all_pred_types, all_gt_types,
        num_classes=num_classes,
    )

    return {**avg_losses, **metrics}


def main():
    parser = argparse.ArgumentParser(description="Train student model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup ---
    dist_cfg = cfg["training"]["distillation"]
    experiment_name = "distill" if dist_cfg["enabled"] else "baseline"
    encoder_name = cfg["student"]["encoder"]
    run_name = f"{experiment_name}_{encoder_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(cfg["logging"]["output_dir"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"Run: {run_name}")
    print(f"Output: {run_dir}")

    # --- Data ---
    soft_dir = cfg["data"]["soft_targets_dir"] if dist_cfg["enabled"] else None

    # Feature distillation: load precomputed teacher dense features.
    feat_distill_enabled = cfg["training"].get("feature_distillation", {}).get("enabled", False)
    soft_feat_dir = cfg["data"].get("soft_features_dir") if feat_distill_enabled else None

    strong_aug = cfg.get("augmentation", {}).get("strong", False)
    # When feature distill is on, spatial aug is handled manually in dataset
    # (to sync image aug with low-resolution feature aug).
    skip_spatial = feat_distill_enabled and soft_feat_dir is not None
    train_dataset = PanNukeDataset(
        data_dir=cfg["data"]["data_dir"],
        folds=cfg["data"]["train_folds"],
        transform=get_train_transform(cfg["data"]["patch_size"], strong=strong_aug, skip_spatial=skip_spatial),
        soft_targets_dir=soft_dir,
        soft_features_dir=soft_feat_dir,
    )
    val_dataset = PanNukeDataset(
        data_dir=cfg["data"]["data_dir"],
        folds=[cfg["data"]["val_fold"]],
        transform=get_val_transform(),
    )

    # Class-balanced sampling (optional; NuLite recipe)
    if cfg["training"].get("class_balanced_sampling", False):
        weights = compute_sample_weights(train_dataset, num_classes=5)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            sampler=sampler,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
    # val_loader uses a smaller batch_size than train: per-image post-processing
    # (HV-watershed + linear_sum_assignment) is CPU-bound and serial inside the
    # loop, so big batches stall the GPU waiting for it. Empirically batch 8
    # matches eval_student.py speed (~3 min for 2656 patches), vs 11 min at
    # batch 64. persistent_workers avoids respawning the 16 workers on every
    # validate() call inside the epoch loop.
    val_batch_size = cfg["training"].get("val_batch_size", 8)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["data"]["num_workers"] > 0,
    )

    # --- Model ---
    model = build_student(cfg).to(device)
    param_info = model.count_parameters()
    print(f"Student model: {encoder_name}")
    print(f"  Encoder: {param_info['encoder'] / 1e6:.1f}M")
    print(f"  Decoder: {param_info['decoder'] / 1e6:.1f}M")
    print(f"  Heads:   {param_info['heads'] / 1e6:.1f}M")
    print(f"  Total:   {param_info['total_M']:.1f}M")

    # --- Loss ---
    fd_cfg = cfg["training"].get("feature_distillation", {})
    criterion = CombinedLoss(
        loss_weights=cfg["training"]["loss_weights"],
        alpha=dist_cfg["alpha"],
        temperature=dist_cfg["temperature"],
        head_weights=dist_cfg.get("head_weights"),
        distillation_enabled=dist_cfg["enabled"],
        use_focal=cfg["training"].get("use_focal", False),
        focal_gamma=cfg["training"].get("focal_gamma", 2.0),
        type_class_weights=cfg["training"].get("type_class_weights"),
        use_ftl_binary=cfg["training"].get("use_ftl_binary", False),
        tissue_aux=cfg["student"].get("tissue_aux", False),
        feature_distill_enabled=fd_cfg.get("enabled", False),
        beta=fd_cfg.get("beta", 1.0),
        feature_loss_use_cosine=fd_cfg.get("use_cosine", True),
        feature_loss_cosine_weight=fd_cfg.get("cosine_weight", 0.5),
    )

    # --- Optimizer & Scheduler ---
    betas = tuple(cfg["training"].get("adamw_betas", [0.9, 0.999]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=betas,
    )

    num_epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=1e-6,
    )

    # Linear warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, scheduler],
        milestones=[warmup_epochs],
    )

    scaler = GradScaler("cuda")

    # --- Training Loop ---
    best_mpq = 0.0
    epochs_without_improvement = 0
    val_every = cfg["training"].get("val_every_n_epochs", 5)
    early_stop_patience = cfg["training"].get("early_stop_patience", 0)  # 0 disables

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, cfg["training"]["grad_clip"],
        )

        combined_scheduler.step()

        # Validation frequency (per config; default every 5 epochs)
        val_results = {}
        if epoch % val_every == 0 or epoch == num_epochs:
            val_results = validate(
                model, val_loader, criterion, device,
                num_classes=cfg["data"]["num_classes"] - 1,
            )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Logging
        log_str = (
            f"Epoch {epoch}/{num_epochs} ({elapsed:.0f}s) lr={lr:.2e} "
            f"train_loss={train_losses['total']:.4f}"
        )
        if val_results:
            log_str += (
                f" val_loss={val_results.get('total', 0):.4f}"
                f" mPQ={val_results.get('mPQ', 0):.4f}"
                f" bPQ={val_results.get('bPQ', 0):.4f}"
                f" F1={val_results.get('F1_detection', 0):.4f}"
            )
        print(log_str)

        # Build checkpoint dict (needed for both periodic saves and best model)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": combined_scheduler.state_dict(),
            "train_losses": train_losses,
            "val_results": val_results,
            "config": cfg,
        }

        # Save checkpoint periodically
        if epoch % cfg["logging"]["save_checkpoint_every"] == 0 or epoch == num_epochs:
            torch.save(ckpt, run_dir / f"checkpoint_epoch{epoch}.pth")

        # Save best + track early stopping
        if val_results:
            if val_results.get("mPQ", 0) > best_mpq:
                best_mpq = val_results["mPQ"]
                torch.save(ckpt, run_dir / "best_model.pth")
                print(f"  New best mPQ: {best_mpq:.4f}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += val_every
                if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                    print(f"\nEarly stopping: {epochs_without_improvement} epochs without mPQ improvement "
                          f"(patience={early_stop_patience})")
                    break

    print(f"\nTraining complete. Best mPQ: {best_mpq:.4f}")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
