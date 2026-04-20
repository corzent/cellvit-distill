#!/usr/bin/env python3
"""Generate all figures for the thesis document.

Outputs to ../thesis_figures/:
  01_dataset_examples.png    — 3×4 grid of PanNuke patches with GT overlay
  02_class_distribution.png  — bar chart of nuclei class counts
  03_tissue_distribution.png — bar chart of tissue type counts
  04_training_curves.png     — train/val loss + mPQ over epochs for baseline/distill/feature
  05_results_comparison.png  — grouped bar chart teacher vs students per mPQ/bPQ/F1
  06_per_class_pq.png        — per-class PQ comparison
  07_predictions.png         — 3×3 grid of GT vs teacher vs student predictions
"""

import sys
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as fm

# Consistent academic style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Class palette
CLASS_NAMES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]
CLASS_COLORS = [
    (0, 0, 0, 0),            # background — transparent
    (1.0, 0.25, 0.25, 0.6),  # neoplastic — red
    (0.25, 0.6, 1.0, 0.6),   # inflammatory — blue
    (0.95, 0.75, 0.1, 0.6),  # connective — yellow
    (0.4, 0.4, 0.4, 0.75),   # dead — gray
    (0.3, 0.85, 0.3, 0.6),   # epithelial — green
]
CLASS_COLORS_SOLID = [tuple(c[:3]) for c in CLASS_COLORS]

# Thesis palette (match presentation)
PALETTE = {
    "ink": "#1A1A1A",
    "muted": "#6B6B6B",
    "accent": "#B85042",
    "green": "#2D6A4F",
    "soft": "#F0ECE8",
    "rule": "#D9D9D9",
}

DATA_DIR = Path("/home/corzent/caspian/thesis/datasets/pannuke")
OUT_DIR = Path("/home/corzent/caspian/thesis/thesis_figures")
LOG_DIR = Path("/home/corzent/caspian/thesis/.claude/worktrees/happy-tesla/logs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------
def load_fold(fold_idx):
    """Load images and class-channel masks for a fold."""
    d = DATA_DIR / f"fold{fold_idx}"
    imgs = np.load(d / "images.npy", mmap_mode="r")
    masks = np.load(d / "masks.npy", mmap_mode="r")
    types = np.load(d / "types.npy", allow_pickle=True)
    return imgs, masks, types


def build_class_map(masks_patch):
    """Convert PanNuke (H,W,6) instance masks to (H,W) class label map."""
    h, w = masks_patch.shape[:2]
    out = np.zeros((h, w), dtype=np.int32)
    for c in range(5):  # 5 foreground classes
        out[masks_patch[:, :, c] > 0] = c + 1
    return out


def overlay_mask(image_uint8, class_map):
    """Return RGB overlay with class-colored transparent mask over image."""
    rgb = image_uint8.astype(np.float32) / 255.0
    overlay = rgb.copy()
    for c in range(1, 6):
        mask = class_map == c
        if mask.any():
            color = np.array(CLASS_COLORS_SOLID[c])
            alpha = 0.45
            overlay[mask] = overlay[mask] * (1 - alpha) + color[None, :] * alpha
    return np.clip(overlay, 0, 1)


def instance_outlines(masks_patch):
    """(H,W) boolean mask of instance outlines across all 5 classes."""
    h, w = masks_patch.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    for c in range(5):
        ch = masks_patch[:, :, c]
        uniq = np.unique(ch)
        for uid in uniq:
            if uid == 0:
                continue
            m = (ch == uid)
            # Simple outline via erosion
            em = np.zeros_like(m)
            em[1:-1, 1:-1] = (
                m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] &
                m[1:-1, :-2] & m[1:-1, 2:]
            )
            out |= (m & ~em)
    return out


# ---------- 1. Dataset examples ----------
def fig_dataset_examples():
    print("  [1/7] dataset examples")
    # Load fold 1 for diversity
    imgs, masks, types = load_fold(1)

    # Pick 12 diverse patches — try to cover multiple tissues
    tissue_names = list(set(str(t) for t in types))[:12]
    indices = []
    picked_tissues = []
    for tn in tissue_names:
        where = np.where([str(t) == tn for t in types])[0]
        if len(where) > 0:
            indices.append(int(where[0]))
            picked_tissues.append(tn)
        if len(indices) >= 12:
            break

    # Fill with random if fewer than 12 tissues
    while len(indices) < 12:
        extra = np.random.RandomState(42).randint(0, len(imgs))
        if extra not in indices:
            indices.append(extra)
            picked_tissues.append(str(types[extra]))

    fig, axes = plt.subplots(3, 4, figsize=(14, 10.5), constrained_layout=True)
    for i, (idx, ax, tissue) in enumerate(zip(indices[:12], axes.flat, picked_tissues[:12])):
        img = np.array(imgs[idx])
        msk = np.array(masks[idx])
        cmap = build_class_map(msk)
        overlay = overlay_mask(img, cmap)
        # Thin black outlines around instances for clarity
        outlines = instance_outlines(msk)
        overlay[outlines] = [0.05, 0.05, 0.05]

        ax.imshow(overlay)
        ax.set_title(f"{tissue}  ·  fold 1 # {idx}", fontsize=10, color=PALETTE["muted"])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["rule"])

    # Legend across all
    legend_elems = [Patch(facecolor=CLASS_COLORS_SOLID[i], edgecolor="none",
                          label=CLASS_NAMES[i])
                    for i in range(1, 6)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Примеры патчей PanNuke с разметкой ядер",
                 fontsize=15, fontweight="bold", color=PALETTE["ink"])
    plt.savefig(OUT_DIR / "01_dataset_examples.png")
    plt.close(fig)


# ---------- 2. Class distribution ----------
def fig_class_distribution():
    print("  [2/7] class distribution")
    counts = np.zeros(5, dtype=np.int64)
    for fold in [1, 2, 3]:
        _, masks, _ = load_fold(fold)
        for idx in range(len(masks)):
            m = np.array(masks[idx])
            for c in range(5):
                uniq = np.unique(m[:, :, c])
                counts[c] += (uniq != 0).sum()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [CLASS_NAMES[i + 1] for i in range(5)],
        counts,
        color=[CLASS_COLORS_SOLID[i + 1] for i in range(5)],
        edgecolor=PALETTE["ink"], linewidth=0.7,
    )
    total = counts.sum()
    for bar, c in zip(bars, counts):
        pct = 100 * c / total
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c:,}\n{pct:.1f}%",
                ha="center", va="bottom", fontsize=10, color=PALETTE["ink"])
    ax.set_ylabel("Число размеченных ядер")
    ax.set_title("Распределение классов в PanNuke (все 3 фолда)")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.savefig(OUT_DIR / "02_class_distribution.png")
    plt.close(fig)


# ---------- 3. Tissue distribution ----------
def fig_tissue_distribution():
    print("  [3/7] tissue distribution")
    from collections import Counter
    c = Counter()
    for fold in [1, 2, 3]:
        _, _, types = load_fold(fold)
        for t in types:
            c[str(t)] += 1

    labels, counts = zip(*sorted(c.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, counts, color=PALETTE["accent"],
                  edgecolor=PALETTE["ink"], linewidth=0.5)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                ha="center", va="bottom", fontsize=9, color=PALETTE["ink"])
    ax.set_ylabel("Число патчей")
    ax.set_title("Распределение типов тканей в PanNuke")
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.savefig(OUT_DIR / "03_tissue_distribution.png")
    plt.close(fig)


# ---------- 4. Training curves ----------
def parse_training_log(log_path):
    """Parse train_loss, val_loss, mPQ from our training log format.

    Returns dict: epochs, train_loss, val_loss, mPQ
    """
    epochs, train_losses, val_losses, mpqs = [], [], [], []
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\([\d.]+s?\)\s+lr=[\d.e+-]+\s+train_loss=([\d.]+)"
        r"(?:\s+val_loss=([\d.]+)\s+mPQ=([\d.]+))?"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            e = int(m.group(1))
            tl = float(m.group(2))
            vl = float(m.group(3)) if m.group(3) else np.nan
            mp = float(m.group(4)) if m.group(4) else np.nan
            epochs.append(e)
            train_losses.append(tl)
            val_losses.append(vl)
            mpqs.append(mp)
    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(train_losses),
        "val_loss": np.array(val_losses),
        "mPQ": np.array(mpqs),
    }


def fig_training_curves():
    print("  [4/7] training curves")
    runs = [
        ("baseline",          LOG_DIR / "fastvit_v2_baseline.log",     PALETTE["muted"]),
        ("response KD",       LOG_DIR / "fastvit_v2_distill.log",      PALETTE["accent"]),
        ("feature KD (β=1)",  LOG_DIR / "fastvit_feature_distill.log", PALETTE["green"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    for name, log, color in runs:
        if not log.exists():
            print(f"    skip {name}: {log} not found")
            continue
        d = parse_training_log(log)
        if len(d["train_loss"]) == 0:
            continue

        ep = d["epochs"]
        tl = d["train_loss"]
        vl = d["val_loss"]
        mp = d["mPQ"]

        # Detect divergence: truncate when either train_loss jumps > 2× prior
        # OR val mPQ collapses below half of the peak reached so far.
        # Handles fp16 divergence on baseline (epoch 48).
        cut = len(tl)
        peak_mp_so_far = 0.0
        for i in range(len(tl)):
            if not np.isnan(mp[i]):
                peak_mp_so_far = max(peak_mp_so_far, mp[i])
            loss_jump = i > 0 and tl[i] > 2.0 * tl[i - 1] and tl[i] > 5.0
            mp_collapse = (not np.isnan(mp[i])) and peak_mp_so_far > 0.3 and mp[i] < 0.5 * peak_mp_so_far
            if loss_jump or mp_collapse:
                cut = i
                reason = "loss spike" if loss_jump else "mPQ collapse"
                print(f"    {name}: truncating at epoch {ep[i]} ({reason})")
                break
        ep, tl, vl, mp = ep[:cut], tl[:cut], vl[:cut], mp[:cut]

        # Train loss (solid, filled alpha)
        ax1.plot(ep, tl, "-", color=color, label=f"{name} · train",
                 linewidth=2.0, alpha=0.9)
        valid = ~np.isnan(vl)
        ax1.plot(ep[valid], vl[valid], "--", color=color, label=f"{name} · val",
                 linewidth=1.6, alpha=0.55)

        valid_mp = ~np.isnan(mp)
        ax2.plot(ep[valid_mp], mp[valid_mp], "-o", color=color, label=name,
                 linewidth=2.0, markersize=3.0, alpha=0.9)

    ax1.set_xlabel("Эпоха"); ax1.set_ylabel("Loss")
    ax1.set_title("Кривые обучения: train и val loss")
    # Log scale y if ranges span orders of magnitude (distill loss ~15 vs baseline ~4)
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=1)
    ax1.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4)

    ax2.set_xlabel("Эпоха"); ax2.set_ylabel("mPQ  (val, fold 3)")
    ax2.set_title("Динамика mPQ по эпохам")
    ax2.set_ylim(0, 0.65)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)
    # Horizontal line at teacher mPQ for reference
    ax2.axhline(0.592, color=PALETTE["ink"], linestyle=":", linewidth=1.2, alpha=0.6)
    ax2.text(2, 0.605, "teacher  = 0.592",
             fontsize=10, color=PALETTE["ink"], alpha=0.75, style="italic",
             fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_training_curves.png")
    plt.close(fig)


# ---------- 5. Results comparison ----------
def fig_results_comparison():
    print("  [5/7] results comparison")
    # Final numbers from all our evals
    models = [
        "CellViT-SAM-H\n(teacher, 630M)",
        "CellViT-256\n(x20, 46.8M)",
        "ConvNeXt-Tiny\n(31.9M)",
        "FastViT baseline\n(11.5M)",
        "FastViT + response KD\n(11.5M)",
        "FastViT + feature KD\n(11.8M)",
        "★ FastViT + KD + TTA\n(11.5M)",
    ]
    mpq = [0.592, 0.317, 0.468, 0.456, 0.467, 0.461, 0.472]
    bpq = [0.664, 0.471, 0.591, 0.578, 0.598, 0.591, 0.604]
    f1  = [0.784, 0.598, 0.719, 0.706, 0.724, 0.720, 0.729]

    x = np.arange(len(models))
    width = 0.27

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - width, mpq, width, label="mPQ",
                color=PALETTE["accent"], edgecolor=PALETTE["ink"], linewidth=0.5)
    b2 = ax.bar(x,         bpq, width, label="bPQ",
                color=PALETTE["green"],  edgecolor=PALETTE["ink"], linewidth=0.5)
    b3 = ax.bar(x + width, f1,  width, label="F1-детекции",
                color=PALETTE["muted"],  edgecolor=PALETTE["ink"], linewidth=0.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8.5, color=PALETTE["ink"])

    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9.5)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение моделей по трём метрикам (PanNuke fold 3)")
    ax.set_ylim(0, 0.92)
    ax.legend(loc="upper right", fontsize=11)
    # Highlight the best student row
    ax.axvspan(len(models) - 1.4, len(models) - 0.6, alpha=0.08, color=PALETTE["accent"], zorder=0)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_results_comparison.png")
    plt.close(fig)


# ---------- 6. Per-class PQ ----------
def fig_per_class_pq():
    print("  [6/7] per-class PQ")
    classes = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    teacher = [0.668, 0.576, 0.516, 0.443, 0.670]
    baseline = [0.530, 0.449, 0.385, 0.158, 0.531]
    distill  = [0.550, 0.446, 0.384, 0.104, 0.544]
    feature  = [0.553, 0.428, 0.381, 0.137, 0.545]

    x = np.arange(len(classes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(x - 1.5 * w, teacher,  w, label="Teacher (CellViT-SAM-H)",
           color=PALETTE["ink"], edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.bar(x - 0.5 * w, baseline, w, label="FastViT baseline",
           color=PALETTE["muted"], edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.bar(x + 0.5 * w, distill,  w, label="FastViT + response KD",
           color=PALETTE["accent"], edgecolor=PALETTE["ink"], linewidth=0.5)
    ax.bar(x + 1.5 * w, feature,  w, label="FastViT + feature KD",
           color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.5)

    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_ylabel("Panoptic Quality (PQ)")
    ax.set_title("PQ по классам клеточных ядер (fold 3)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 0.85)

    # annotate dead column with arrow noting gap
    ax.annotate("Большой gap на редком классе (<2% патчей)",
                xy=(3, 0.44), xytext=(3.0, 0.75),
                fontsize=9.5, color=PALETTE["muted"], style="italic",
                ha="center",
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=0.8))

    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_per_class_pq.png")
    plt.close(fig)


# ---------- 7. Prediction examples ----------
def fig_predictions():
    print("  [7/7] predictions (placeholder from GT for now)")
    # Show 3 patches with their GT masks from fold 3 — placeholder.
    # Later can add actual model predictions via post_process_predictions.
    imgs, masks, types = load_fold(3)
    rng = np.random.RandomState(7)
    indices = rng.choice(len(imgs), size=3, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11), constrained_layout=True)
    col_titles = ["Оригинал (H&E)", "GT маска (классы)", "GT outlines"]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", color=PALETTE["ink"])

    for i, idx in enumerate(indices):
        img_raw = np.array(imgs[idx])
        # PanNuke stores images as float64 in range [0, 255] — normalise to uint8.
        if img_raw.dtype != np.uint8:
            img_uint8 = np.clip(img_raw, 0, 255).astype(np.uint8)
        else:
            img_uint8 = img_raw
        msk = np.array(masks[idx])
        cmap = build_class_map(msk)
        overlay = overlay_mask(img_uint8, cmap)
        outlines = instance_outlines(msk)
        outline_img = img_uint8.astype(np.float32) / 255.0
        outline_img[outlines] = [1.0, 0.1, 0.1]  # red outlines

        axes[i, 0].imshow(img_uint8)
        axes[i, 0].set_ylabel(f"{types[idx]}\npatch #{idx}",
                              fontsize=10, color=PALETTE["muted"])
        axes[i, 1].imshow(overlay)
        axes[i, 2].imshow(outline_img)

        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for spine in axes[i, j].spines.values():
                spine.set_edgecolor(PALETTE["rule"])

    legend_elems = [Patch(facecolor=CLASS_COLORS_SOLID[i], edgecolor="none",
                          label=CLASS_NAMES[i])
                    for i in range(1, 6)]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Примеры из тестового фолда PanNuke с разметкой",
                 fontsize=14, fontweight="bold", color=PALETTE["ink"])
    plt.savefig(OUT_DIR / "07_predictions_gt.png")
    plt.close(fig)


# ---------- main ----------
if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    fig_dataset_examples()
    fig_class_distribution()
    fig_tissue_distribution()
    fig_training_curves()
    fig_results_comparison()
    fig_per_class_pq()
    fig_predictions()
    print("Done")
    # List outputs
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
