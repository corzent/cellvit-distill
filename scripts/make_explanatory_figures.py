#!/usr/bin/env python3
"""Generate explanatory diagrams for the thesis (no real data — concept figures).

Outputs into ../thesis_figures/:
  08_pipeline_diagram.png      — end-to-end data + loss pipeline
  09_kd_variants.png           — KL vs DKD vs UFD-KD information flow
  10_per_tissue_class_heatmap.png — 19 tissues × 5 classes nucleus density
  11_decoder_comparison.png    — HoVer-Net FPN vs Mamba PVM decoder
  12_dkd_decomposition.png     — per-pixel KL split into TCKD + NCKD
  13_alpha_sensitivity.png     — alpha bisection finding (KD instability)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

PALETTE = {
    "ink": "#1A1A1A", "muted": "#6B6B6B", "accent": "#B85042",
    "green": "#2D6A4F", "soft": "#F0ECE8", "rule": "#D9D9D9",
    "navy": "#2D2A6E", "gold": "#C5A05B", "softgreen": "#E6F0E8",
    "softred": "#FCEAE7", "softnavy": "#EEF0FA",
}

DATA_DIR = Path("/home/corzent/caspian/thesis/datasets/pannuke")
OUT_DIR = Path("/home/corzent/caspian/thesis/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, label, color=None, edge=None, fontsize=11, bold=False, lw=1.2):
    color = color or PALETTE["soft"]
    edge = edge or PALETTE["rule"]
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fontsize, fontweight=weight,
            color=PALETTE["ink"])


def _arrow(ax, x1, y1, x2, y2, color=None, label=None, lw=1.5, style="->"):
    color = color or PALETTE["ink"]
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color, lw=lw,
                          mutation_scale=14)
    ax.add_patch(arr)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label,
                ha="center", va="bottom", fontsize=9, color=color,
                style="italic")


# ============================================================
# 08. Pipeline diagram — clean layout: offline top, online bottom,
# cache flows down via a single vertical bridge.
# ============================================================
def fig_pipeline():
    print("  [08] pipeline diagram")
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8); ax.axis("off")

    ax.text(8, 7.6, "Сквозной пайплайн обучения с дистилляцией знаний",
            ha="center", fontsize=15, fontweight="bold", color=PALETTE["ink"])

    # --- OFFLINE row (y ≈ 6) ---
    ax.text(0.3, 6.85, "ОФФЛАЙН  ·  один раз",
            fontsize=11, fontweight="bold", color=PALETTE["muted"], style="italic")

    _box(ax, 0.6, 5.7, 2.6, 0.9, "PanNuke\n7901 патч 256×256",
         color=PALETTE["softnavy"], edge=PALETTE["navy"])
    _arrow(ax, 3.2, 6.15, 4.2, 6.15, lw=1.6)
    _box(ax, 4.2, 5.7, 3.4, 0.9, "Teacher\nCellViT-SAM-H (630M)",
         color=PALETTE["softred"], edge=PALETTE["accent"], bold=True, lw=1.8)
    _arrow(ax, 7.6, 6.15, 8.6, 6.15, lw=1.6)
    _box(ax, 8.6, 5.7, 3.0, 0.9, "Soft targets\nна диск (fp16)",
         color=PALETTE["softnavy"], edge=PALETTE["navy"], bold=True)

    # Separator
    ax.plot([0.3, 15.7], [5.0, 5.0], color=PALETTE["rule"],
            linestyle="--", linewidth=0.9, zorder=0)

    # --- ONLINE row (y ≈ 3) ---
    ax.text(0.3, 4.55, "ОНЛАЙН  ·  каждая итерация",
            fontsize=11, fontweight="bold", color=PALETTE["muted"], style="italic")

    # Inputs
    _box(ax, 0.6, 2.9, 1.7, 0.9, "Батч\nпатчей",
         color=PALETTE["softnavy"], edge=PALETTE["navy"])
    _arrow(ax, 2.3, 3.35, 3.1, 3.35, lw=1.6)
    _box(ax, 3.1, 2.9, 2.2, 0.9, "Augmentations\n(spatial + color)",
         color=PALETTE["soft"], edge=PALETTE["rule"])
    _arrow(ax, 5.3, 3.35, 6.1, 3.35, lw=1.6)
    _box(ax, 6.1, 2.55, 2.8, 1.6, "Student\nFastViT-S12 (11.5M)\n+ decoder",
         color=PALETTE["softgreen"], edge=PALETTE["green"], bold=True, lw=1.8)

    # Soft targets bridge down (right-side of disk → into online section)
    _arrow(ax, 10.1, 5.7, 10.1, 4.4, lw=1.4, color=PALETTE["accent"], style="->")
    ax.text(10.25, 5.05, " load + aug-sync",
            fontsize=10, color=PALETTE["accent"], style="italic", va="center")
    _box(ax, 8.9, 3.6, 2.6, 0.8, "Soft targets\n(spatially synced)",
         color=PALETTE["softred"], edge=PALETTE["accent"], bold=True)

    # 3 heads to the right of student
    _arrow(ax, 8.9, 3.7, 10.0, 3.7, lw=1.2)
    # Three head boxes stacked vertically
    head_x = 11.7; head_w = 1.4; head_h = 0.55
    for i, (name, ypos) in enumerate([("binary", 3.95), ("HV map", 3.2), ("type (6)", 2.45)]):
        _box(ax, head_x, ypos, head_w, head_h, name,
             color=PALETTE["soft"], edge=PALETTE["rule"], fontsize=10)
        _arrow(ax, 8.9, 3.35, head_x, ypos + head_h / 2, lw=0.7, color=PALETTE["muted"])

    # Loss block
    _box(ax, 13.5, 2.7, 2.3, 1.5,
         "Loss\nL = L_GT\n+  α·L_KD",
         color=PALETTE["soft"], edge=PALETTE["ink"], bold=True, lw=1.8)
    # Arrows from heads into loss
    for ypos in (4.22, 3.47, 2.72):
        _arrow(ax, head_x + head_w, ypos, 13.5, 3.45, lw=0.7, color=PALETTE["muted"])
    # Soft targets into loss (clean vertical-then-right path approx)
    _arrow(ax, 11.5, 4.0, 13.5, 3.7, lw=1.2, color=PALETTE["accent"], style="->")

    # Backward arrow (separate row, bottom)
    _arrow(ax, 14.6, 2.5, 7.5, 2.2, lw=2.0, color=PALETTE["green"], style="->")
    ax.text(11.0, 2.05, "backward (gradients to student)",
            ha="center", fontsize=10, color=PALETTE["green"], style="italic")

    # Legend (right side)
    legend_items = [
        ("forward (data)", PALETTE["ink"]),
        ("soft target / KD signal", PALETTE["accent"]),
        ("backward (gradient)", PALETTE["green"]),
    ]
    handles = [Line2D([0], [0], color=c, lw=2.5, label=name) for name, c in legend_items]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.02),
              fontsize=10, framealpha=0.97, frameon=True)

    # Caption note bottom
    ax.text(8, 0.5,
            "Teacher один раз прогоняется по датасету в fp16, его логиты кешируются на диск (≤ 10 ГБ).\n"
            "При обучении ученика мягкие цели подгружаются и проходят через те же пространственные аугментации, что и изображение.",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    plt.savefig(OUT_DIR / "08_pipeline_diagram.png")
    plt.close(fig)


# ============================================================
# 09. KD variants comparison
# ============================================================
def fig_kd_variants():
    print("  [09] KD variants comparison")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    titles = ["Vanilla KL", "Decoupled KD (DKD)", "Frequency-Decoupled KD (UFD)"]
    colors = [PALETTE["navy"], PALETTE["accent"], PALETTE["gold"]]

    for ax, title, color in zip(axes, titles, colors):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
        ax.text(5, 6.5, title, ha="center", fontsize=13, fontweight="bold", color=color)

    # ---- KL ----
    ax = axes[0]
    _box(ax, 0.5, 5.2, 4, 0.6, "logits_student", color=PALETTE["soft"])
    _box(ax, 5.5, 5.2, 4, 0.6, "logits_teacher", color=PALETTE["soft"])
    ax.annotate("", xy=(5, 4.5), xytext=(2.5, 5.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
    ax.annotate("", xy=(5, 4.5), xytext=(7.5, 5.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
    _box(ax, 3.0, 3.5, 4, 0.8,
         "softmax(·/T)", color=PALETTE["softnavy"], edge=PALETTE["navy"])
    ax.annotate("", xy=(5, 2.5), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle="->", color=PALETTE["navy"]))
    _box(ax, 3.0, 1.5, 4, 1.0,
         "KL(p_s || p_t)\nуниформно по всем классам",
         color=PALETTE["softnavy"], edge=PALETTE["navy"], bold=True)
    ax.text(5, 0.6, "→ редкие классы тонут\nв доминирующем background",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    # ---- DKD ----
    ax = axes[1]
    _box(ax, 0.5, 5.2, 4, 0.6, "logits_student", color=PALETTE["soft"])
    _box(ax, 5.5, 5.2, 4, 0.6, "logits_teacher", color=PALETTE["soft"])
    ax.annotate("", xy=(5, 4.5), xytext=(2.5, 5.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
    ax.annotate("", xy=(5, 4.5), xytext=(7.5, 5.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
    _box(ax, 0.5, 3.0, 4, 1.3,
         "TCKD\nKL по target-классу\n(уверенность)",
         color=PALETTE["softred"], edge=PALETTE["accent"], bold=True)
    _box(ax, 5.5, 3.0, 4, 1.3,
         "NCKD\nKL по не-target классам\n(тёмное знание)",
         color=PALETTE["softred"], edge=PALETTE["accent"], bold=True)
    ax.annotate("", xy=(4.5, 1.7), xytext=(2.5, 3.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))
    ax.annotate("", xy=(5.5, 1.7), xytext=(7.5, 3.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))
    _box(ax, 3.0, 1.0, 4, 0.7,
         "α·TCKD + β·NCKD,  β=8",
         color=PALETTE["softnavy"], edge=PALETTE["navy"], bold=True)
    ax.text(5, 0.2, "→ NCKD амплифицирует\nсигнал на редких классах",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    # ---- UFD ----
    ax = axes[2]
    _box(ax, 0.5, 5.2, 4, 0.6, "logits_student", color=PALETTE["soft"])
    _box(ax, 5.5, 5.2, 4, 0.6, "logits_teacher", color=PALETTE["soft"])
    ax.annotate("", xy=(5, 4.6), xytext=(5, 5.2),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
    _box(ax, 3.0, 4.0, 4, 0.6, "DCT-II (H×W)",
         color=PALETTE["soft"], edge=PALETTE["gold"])
    ax.annotate("", xy=(2.5, 3.0), xytext=(4, 4.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["gold"]))
    ax.annotate("", xy=(7.5, 3.0), xytext=(6, 4.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["gold"]))
    _box(ax, 0.5, 2.2, 4, 0.8, "LF (32×32)\nглобальная структура",
         color="#FFF8E1", edge=PALETTE["gold"])
    _box(ax, 5.5, 2.2, 4, 0.8, "HF (остальное)\nграницы, мелкое",
         color="#FFF8E1", edge=PALETTE["gold"])
    _box(ax, 3.0, 0.9, 4, 0.7,
         "w_lf·MSE_lf + w_hf·MSE_hf",
         color="#FFF8E1", edge=PALETTE["gold"], bold=True)
    ax.text(5, 0.1, "→ negative result:\nLF доминирует HF в ×600",
            ha="center", fontsize=10, color=PALETTE["accent"],
            style="italic", fontweight="bold")

    plt.suptitle("Три варианта response-based дистилляции",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_kd_variants.png")
    plt.close(fig)


# ============================================================
# 10. Per-tissue × per-class heatmap
# ============================================================
def fig_tissue_class_heatmap():
    print("  [10] per-tissue × per-class heatmap")
    CLASS_NAMES_EN = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    counts = {}

    for fold in [1, 2, 3]:
        d = DATA_DIR / f"fold{fold}"
        imgs = np.load(d / "images.npy", mmap_mode="r")
        msks = np.load(d / "masks.npy", mmap_mode="r")
        types = np.load(d / "types.npy", allow_pickle=True)

        for idx in range(len(imgs)):
            tissue = str(types[idx])
            if tissue not in counts:
                counts[tissue] = np.zeros(5, dtype=np.int64)
            m = msks[idx]
            for c in range(5):
                uniq = np.unique(m[:, :, c])
                counts[tissue][c] += (uniq != 0).sum()

    tissue_names = sorted(counts.keys(), key=lambda t: -counts[t].sum())
    mat = np.array([counts[t] for t in tissue_names])
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    pct = 100 * mat / row_sum  # row-normalised percentages

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pct, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(CLASS_NAMES_EN, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(tissue_names)))
    ax.set_yticklabels(tissue_names)

    for i in range(len(tissue_names)):
        for j in range(5):
            txt_color = "white" if pct[i, j] > 50 else PALETTE["ink"]
            ax.text(j, i, f"{pct[i, j]:.0f}",
                    ha="center", va="center", fontsize=10, color=txt_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("% ядер класса в ткани", rotation=270, labelpad=18)

    # Side bar: absolute counts
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(np.arange(len(tissue_names)))
    ax2.set_yticklabels([f"{c:>6,}".replace(",", " ") for c in mat.sum(axis=1)],
                        fontsize=9, color=PALETTE["muted"])
    ax2.set_ylabel("Всего ядер", color=PALETTE["muted"], labelpad=10)
    ax2.tick_params(axis="y", length=0)

    ax.set_title("Распределение классов ядер по тканям PanNuke (% по строке)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_per_tissue_class_heatmap.png")
    plt.close(fig)


# ============================================================
# 11. Decoder comparison
# ============================================================
def fig_decoder_comparison():
    print("  [11] decoder comparison")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    # HoVer-Net FPN
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(5, 6.6, "Декодер HoVer-Net (FPN)",
            ha="center", fontsize=13, fontweight="bold", color=PALETTE["navy"])

    stages = [
        ("Stage 4: 8×8 × 512",  5.0, "softnavy"),
        ("Stage 3: 16×16 × 256", 4.2, "softnavy"),
        ("Stage 2: 32×32 × 128", 3.4, "softnavy"),
        ("Stage 1: 64×64 × 64",  2.6, "softnavy"),
    ]
    for label, y, c in stages:
        _box(ax, 0.5, y, 3.5, 0.5, label, color=PALETTE[c], edge=PALETTE["navy"])

    for label, y, _ in stages[1:]:
        _arrow(ax, 4.0, y + 0.25, 6.0, y + 0.25, color=PALETTE["muted"], lw=1.0)

    fpn_y = [4.7, 3.9, 3.1, 2.3]
    fpn_names = ["upsample + concat", "upsample + concat", "upsample + concat", "+ conv-conv"]
    for y, name in zip(fpn_y, fpn_names):
        _box(ax, 6.0, y, 3.5, 0.5, name, color=PALETTE["soft"])

    for i in range(3):
        ax.annotate("", xy=(7.75, fpn_y[i + 1] + 0.5),
                    xytext=(7.75, fpn_y[i]),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["navy"]))

    _box(ax, 6.0, 1.0, 3.5, 0.7, "32-ch feature → 3 головы",
         color=PALETTE["softgreen"], edge=PALETTE["green"], bold=True)
    ax.annotate("", xy=(7.75, 1.7), xytext=(7.75, 2.3),
                arrowprops=dict(arrowstyle="->", color=PALETTE["green"]))

    ax.text(5, 0.4, "Свёртки 3×3, билинейный upsample. ~3.1M параметров.",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    # Mamba PVM
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(5, 6.6, "Декодер Mamba (PVM, UltraLight VM-UNet)",
            ha="center", fontsize=13, fontweight="bold", color=PALETTE["accent"])

    for label, y, c in stages:
        _box(ax, 0.5, y, 3.5, 0.5, label, color=PALETTE[c], edge=PALETTE["accent"])

    for label, y, _ in stages[1:]:
        _arrow(ax, 4.0, y + 0.25, 6.0, y + 0.25, color=PALETTE["muted"], lw=1.0)

    # PVM block detail
    pvm_y = [4.5, 3.7, 2.9, 2.1]
    for y in pvm_y:
        # 4 groups
        for gi in range(4):
            _box(ax, 6.0 + gi * 0.55, y, 0.5, 0.4, "M", color="#FFF1F0",
                 edge=PALETTE["accent"], fontsize=9, bold=True)
        ax.text(8.4, y + 0.2, "+ GN res", fontsize=9, color=PALETTE["muted"],
                va="center")

    for i in range(3):
        ax.annotate("", xy=(7.0, pvm_y[i + 1] + 0.4),
                    xytext=(7.0, pvm_y[i]),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    _box(ax, 6.0, 1.0, 3.5, 0.7, "32-ch feature → 3 головы",
         color=PALETTE["softgreen"], edge=PALETTE["green"], bold=True)
    ax.annotate("", xy=(7.5, 1.7), xytext=(7.5, 2.1),
                arrowprops=dict(arrowstyle="->", color=PALETTE["green"]))

    ax.text(5, 0.4,
            "Каналы делятся на 4 группы, каждая через свой Mamba (SSM),\n"
            "4-way cross-scan, конкатенация. Линейная сложность по токенам.",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    plt.suptitle("Сравнение декодеров: свёрточный (HoVer-Net) vs SSM (Mamba)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_decoder_comparison.png")
    plt.close(fig)


# ============================================================
# 12. DKD decomposition — pictorial
# ============================================================
def fig_dkd_decomposition():
    print("  [12] DKD decomposition")
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(6.5, 5.6, "Декомпозиция KL → TCKD + NCKD (Zhao et al., CVPR 2022)",
            ha="center", fontsize=14, fontweight="bold", color=PALETTE["ink"])

    # Teacher and student distributions illustrated as small bar charts
    classes = ["bg", "neopl.", "inflam.", "conn.", "dead", "epith."]
    np.random.seed(7)
    p_teacher = np.array([0.05, 0.78, 0.05, 0.05, 0.02, 0.05])
    p_student = np.array([0.06, 0.55, 0.10, 0.12, 0.04, 0.13])

    # Left: full KL
    x0 = 0.5; w = 3.5; y0 = 1.5; h = 2.7
    _box(ax, x0, y0, w, h, "", color=PALETTE["softnavy"], edge=PALETTE["navy"])
    ax.text(x0 + w / 2, y0 + h + 0.15, "Полная KL",
            ha="center", fontsize=12, fontweight="bold", color=PALETTE["navy"])
    # mini bar charts
    barx = np.arange(6) * (w * 0.85 / 6) + x0 + 0.4
    for i, (pt, ps) in enumerate(zip(p_teacher, p_student)):
        ax.bar(barx[i] - 0.13, pt, 0.22, color=PALETTE["accent"],
               bottom=y0 + 0.5, alpha=0.85)
        ax.bar(barx[i] + 0.13, ps, 0.22, color=PALETTE["navy"],
               bottom=y0 + 0.5, alpha=0.85)
    ax.text(x0 + w / 2, y0 + 0.25, "p_teacher (red)   p_student (blue)",
            ha="center", fontsize=8, color=PALETTE["muted"], style="italic")

    # Arrow + decomposition split
    ax.annotate("", xy=(5.6, 3.0), xytext=(4.2, 3.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["ink"], lw=2))
    ax.text(4.9, 3.25, "разделение", ha="center", fontsize=10, style="italic")

    # TCKD box
    _box(ax, 5.6, 3.2, 3.5, 1.6, "", color=PALETTE["softred"], edge=PALETTE["accent"])
    ax.text(7.35, 4.55, "TCKD",
            ha="center", fontsize=13, fontweight="bold", color=PALETTE["accent"])
    ax.text(7.35, 4.05,
            "бинарная KL\nна target-классе vs остальные\n(уверенность в правильном)",
            ha="center", fontsize=10, color=PALETTE["ink"])

    # NCKD box
    _box(ax, 5.6, 1.2, 3.5, 1.6, "", color=PALETTE["softred"], edge=PALETTE["accent"])
    ax.text(7.35, 2.55, "NCKD",
            ha="center", fontsize=13, fontweight="bold", color=PALETTE["accent"])
    ax.text(7.35, 2.05,
            "KL по не-target классам\nпосле renormalize (тёмное знание)",
            ha="center", fontsize=10, color=PALETTE["ink"])

    # Reweight box
    ax.annotate("", xy=(9.4, 3.6), xytext=(9.1, 4.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))
    ax.annotate("", xy=(9.4, 2.4), xytext=(9.1, 2.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))

    _box(ax, 9.4, 2.2, 3.3, 1.6, "", color="#FFF6E0", edge=PALETTE["gold"])
    ax.text(11.05, 3.45, "α·TCKD + β·NCKD",
            ha="center", fontsize=13, fontweight="bold", color=PALETTE["gold"])
    ax.text(11.05, 2.85, "α = 1,  β = 8",
            ha="center", fontsize=12, color=PALETTE["ink"])
    ax.text(11.05, 2.45,
            "NCKD усилен ×8 — лечит\nfailure mode на редком Dead",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    # Comparison note
    ax.text(6.5, 0.6,
            "Vanilla KL: NCKD автоматически подавляется фактором (1 − p_t(c)) → теряем сигнал на редких классах.\n"
            "DKD устраняет это подавление, делая non-target signal первоклассным.",
            ha="center", fontsize=10, color=PALETTE["muted"], style="italic")

    plt.savefig(OUT_DIR / "12_dkd_decomposition.png")
    plt.close(fig)


# ============================================================
# 13. Alpha sensitivity
# ============================================================
def fig_alpha_sensitivity():
    print("  [13] alpha sensitivity")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Synthetic illustration of the bisection finding
    epochs = np.arange(0, 60)
    rng = np.random.default_rng(42)

    def curve(peak, peak_ep, noise=0.005):
        c = peak * (1 - np.exp(-epochs / (peak_ep * 0.4)))
        c[peak_ep:] *= np.exp(-(epochs[peak_ep:] - peak_ep) * 0.02)
        c += rng.normal(0, noise, size=epochs.shape)
        return np.clip(c, 0, 0.55)

    baseline = curve(0.466, 40)
    alpha_005 = curve(0.470, 38)
    alpha_020 = np.where(epochs < 12, curve(0.30, 8) * 0.6, 0.29 + rng.normal(0, 0.012, size=epochs.shape))
    alpha_020 = np.clip(alpha_020, 0, 0.5)

    ax.plot(epochs, baseline, color=PALETTE["muted"], lw=1.8, label="baseline (без KD)")
    ax.plot(epochs, alpha_005, color=PALETTE["green"], lw=2.0, label="distill α = 0.05  ✓")
    ax.plot(epochs, alpha_020, color=PALETTE["accent"], lw=2.0, label="distill α = 0.20  ✗ нестабильно")

    ax.axhline(0.466, color=PALETTE["muted"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0.295, color=PALETTE["accent"], linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Val mPQ (fold 3)")
    ax.set_title("Чувствительность дистилляции к весу α при batch=16, bf16, CUDA 13")
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
    ax.legend(loc="lower right", fontsize=11)

    ax.text(50, 0.42, "ожидаемая\nрабочая зона",
            fontsize=10, color=PALETTE["muted"], style="italic", ha="center")
    ax.text(50, 0.22, "коллапс KD\nна больших α",
            fontsize=10, color=PALETTE["accent"], style="italic", ha="center")

    ax.text(2, 0.05,
            "Иллюстрация: реальные кривые сохранены в logs/. На старом RTX 5060 Ti α=0.2 работал;\n"
            "на 5090 + PyTorch 2.12 + CUDA 13 + batch 16 — нет. Требуется α ≈ 0.05.",
            fontsize=9, color=PALETTE["muted"], style="italic")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_alpha_sensitivity.png")
    plt.close(fig)


# ============================================================
if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    fig_pipeline()
    fig_kd_variants()
    fig_tissue_class_heatmap()
    fig_decoder_comparison()
    fig_dkd_decomposition()
    fig_alpha_sensitivity()
    print("Done")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
