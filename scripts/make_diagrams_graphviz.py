#!/usr/bin/env python3
"""Re-render the architectural diagrams via Graphviz.

Replaces matplotlib-drawn versions of figs 2.1, 2.2, 2.3, 2.4 with proper
box-and-arrow diagrams. Graphviz handles layout, orthogonal arrows, and
crisp rendering — no overlapping boxes, no diagonal arrow chaos.

Outputs to ../thesis_figures/:
  08_pipeline_diagram.png      — fig 2.1
  09_kd_variants.png           — fig 2.2
  11_decoder_comparison.png    — fig 2.4
  12_dkd_decomposition.png     — fig 2.3
"""

from pathlib import Path
import graphviz

OUT_DIR = Path("/home/corzent/caspian/thesis/thesis_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette consistent with matplotlib figures
NAVY = "#2D2A6E"
ACCENT = "#B85042"
GREEN = "#2D6A4F"
GOLD = "#C5A05B"
INK = "#1A1A1A"
MUTED = "#6B6B6B"
SOFT_NAVY = "#EEF0FA"
SOFT_RED = "#FCEAE7"
SOFT_GREEN = "#E6F0E8"
SOFT_GOLD = "#FFF8E1"
SOFT = "#F0ECE8"
WHITE = "#FFFFFF"

# Common attrs
COMMON_NODE = dict(
    shape="box", style="rounded,filled",
    fontname="DejaVu Sans", fontsize="11",
)


def render(g: graphviz.Digraph, name: str):
    out_path = OUT_DIR / name
    # Render as PNG via dot; remove .gv source after
    g.attr(dpi="180")
    g.format = "png"
    g.render(filename=str(out_path.with_suffix("")), cleanup=True)
    print(f"  {name}  saved")


# ============================================================
# Fig 2.1 — Pipeline
# ============================================================
def fig_pipeline():
    g = graphviz.Digraph("pipeline")
    g.attr(rankdir="TB", bgcolor="white",
           label="Сквозной пайплайн обучения с дистилляцией знаний",
           labelloc="t", fontname="DejaVu Sans Bold", fontsize="16",
           ranksep="0.6", nodesep="0.4", pad="0.4")

    # ----- OFFLINE row -----
    with g.subgraph(name="cluster_offline") as c:
        c.attr(label="1.  ОФФЛАЙН  ·  один раз",
               labelloc="t", fontsize="12", fontname="DejaVu Sans Bold",
               color=NAVY, style="rounded,dashed", margin="14")
        c.node("pannuke", "PanNuke\n7901 патч 256×256",
               fillcolor=SOFT_NAVY, color=NAVY, **COMMON_NODE)
        c.node("teacher", "Teacher\nCellViT-SAM-H (630M)",
               fillcolor=SOFT_RED, color=ACCENT,
               fontname="DejaVu Sans Bold", **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
        c.node("cache", "Soft targets\nдиск (fp16, ≤10 ГБ)",
               fillcolor=SOFT_NAVY, color=NAVY,
               fontname="DejaVu Sans Bold", **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
        c.edge("pannuke", "teacher", color=INK, penwidth="1.8")
        c.edge("teacher", "cache", color=INK, penwidth="1.8")

    # ----- ONLINE -----
    with g.subgraph(name="cluster_online") as c:
        c.attr(label="2.  ОНЛАЙН  ·  каждая итерация",
               labelloc="t", fontsize="12", fontname="DejaVu Sans Bold",
               color=GREEN, style="rounded,dashed", margin="14")

        c.node("batch", "Батч изображений\n+ GT-метки",
               fillcolor=SOFT_NAVY, color=NAVY, **COMMON_NODE)
        c.node("aug", "Synchronized augmentations\n(spatial + color, applied к обоим)",
               fillcolor=SOFT, color=INK,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
        c.node("student", "Student\nFastViT-S12 (11.5M)",
               fillcolor=SOFT_GREEN, color=GREEN,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})

        # 3 heads cluster
        with c.subgraph(name="cluster_heads") as h:
            h.attr(label="3 выходные головы", labelloc="t", fontsize="10",
                   color=MUTED, style="rounded,dotted", margin="8")
            for name, lbl in [("binary", "binary"), ("hv", "HV map"),
                              ("typeh", "type (6)")]:
                h.node(name, lbl, fillcolor=SOFT, color=MUTED, **COMMON_NODE)

        c.node("loss", "Loss\nL  =  L_GT  +  α · L_KD",
               fillcolor=SOFT, color=INK,
               fontname="DejaVu Sans Bold", fontsize="12",
               shape="box", style="rounded,filled", penwidth="2")

        # Forward edges
        c.edge("batch", "aug", color=INK, penwidth="1.6")
        c.edge("aug", "student", color=INK, penwidth="1.6")
        c.edge("student", "binary", color=INK, penwidth="1.2")
        c.edge("student", "hv", color=INK, penwidth="1.2")
        c.edge("student", "typeh", color=INK, penwidth="1.2")
        c.edge("binary", "loss", color=MUTED, penwidth="0.9")
        c.edge("hv", "loss", color=MUTED, penwidth="0.9")
        c.edge("typeh", "loss", color=MUTED, penwidth="0.9")

        # Backward
        c.edge("loss", "student",
               color=GREEN, penwidth="2.0", style="bold",
               label="backward (gradient)",
               fontname="DejaVu Sans Italic", fontsize="10", fontcolor=GREEN,
               constraint="false")

    # Cache → augmentations (soft target sync)
    g.edge("cache", "aug",
           color=ACCENT, penwidth="2.0", style="bold",
           label="soft targets\nспустить в аугментации",
           fontname="DejaVu Sans Italic", fontsize="10", fontcolor=ACCENT)
    # Cache → loss (KD signal directly)
    g.edge("cache", "loss",
           color=ACCENT, penwidth="1.5", style="dashed",
           label="L_KD signal",
           fontname="DejaVu Sans Italic", fontsize="10", fontcolor=ACCENT,
           constraint="false")

    render(g, "08_pipeline_diagram.png")


# ============================================================
# Fig 2.2 — KD variants comparison (3 columns)
# ============================================================
def fig_kd_variants():
    g = graphviz.Digraph("kd_variants")
    g.attr(rankdir="TB", bgcolor="white",
           label="Три варианта response-based дистилляции",
           labelloc="t", fontname="DejaVu Sans Bold", fontsize="16",
           ranksep="0.45", nodesep="0.35", pad="0.4")

    def panel(c, prefix, title, color, soft_color, content):
        """content: list of (id_suffix, label, special_color) for boxes top-to-bottom."""
        c.attr(label=title, labelloc="t", fontsize="13",
               fontname="DejaVu Sans Bold", color=color,
               fontcolor=color, style="rounded", margin="14")
        # Two logit inputs (top)
        c.node(f"{prefix}_s", "logits_student",
               fillcolor=SOFT, color=MUTED, **COMMON_NODE)
        c.node(f"{prefix}_t", "logits_teacher",
               fillcolor=SOFT, color=MUTED, **COMMON_NODE)
        # Force them on same rank
        with c.subgraph() as same:
            same.attr(rank="same")
            same.node(f"{prefix}_s")
            same.node(f"{prefix}_t")

        prev = [f"{prefix}_s", f"{prefix}_t"]
        for i, (sid, label, node_color, node_bg) in enumerate(content):
            full_id = f"{prefix}_{sid}"
            c.node(full_id, label, fillcolor=node_bg, color=node_color,
                   fontname="DejaVu Sans Bold",
                   **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
            for p in prev:
                c.edge(p, full_id, color=MUTED, penwidth="1.2",
                       arrowhead="normal", arrowsize="0.8")
            prev = [full_id]

    # ---- Vanilla KL ----
    with g.subgraph(name="cluster_kl") as c:
        panel(c, "kl", "Vanilla KL", NAVY, SOFT_NAVY, [
            ("softmax", "σ(·/T)\nsoftmax with temperature", NAVY, SOFT_NAVY),
            ("loss", "L_KL  =  T² · KL(p_t || p_s)\nуниформно по всем классам", NAVY, SOFT_NAVY),
        ])
        c.node("kl_note", "→ редкие классы тонут\nв доминирующем background",
               shape="plaintext", fontcolor=MUTED,
               fontname="DejaVu Sans Oblique", fontsize="10")
        c.edge("kl_loss", "kl_note", style="invis")

    # ---- Decoupled KD ----
    with g.subgraph(name="cluster_dkd") as c:
        panel(c, "dkd", "Decoupled KD  (DKD)", ACCENT, SOFT_RED, [
            ("split", "split  KL  →  TCKD  +  NCKD", ACCENT, SOFT_RED),
        ])
        # TCKD and NCKD branches
        c.node("dkd_tckd", "TCKD\nKL на target-классе",
               fillcolor="#FCEAE7", color=ACCENT, **COMMON_NODE)
        c.node("dkd_nckd", "NCKD\nKL по не-target классам",
               fillcolor="#FCEAE7", color=ACCENT, **COMMON_NODE)
        with c.subgraph() as same:
            same.attr(rank="same")
            same.node("dkd_tckd"); same.node("dkd_nckd")
        c.edge("dkd_split", "dkd_tckd", color=ACCENT, penwidth="1.2")
        c.edge("dkd_split", "dkd_nckd", color=ACCENT, penwidth="1.2")
        c.node("dkd_loss",
               "L_DKD  =  α · TCKD  +  β · NCKD\n(α = 1,    β = 8)",
               fillcolor=SOFT_NAVY, color=NAVY,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
        c.edge("dkd_tckd", "dkd_loss", color=ACCENT, penwidth="1.2")
        c.edge("dkd_nckd", "dkd_loss", color=ACCENT, penwidth="1.2")
        c.node("dkd_note", "→ NCKD амплифицирует\nсигнал на редких классах",
               shape="plaintext", fontcolor=MUTED,
               fontname="DejaVu Sans Oblique", fontsize="10")
        c.edge("dkd_loss", "dkd_note", style="invis")

    # ---- UFD ----
    with g.subgraph(name="cluster_ufd") as c:
        panel(c, "ufd", "Frequency-Decoupled KD  (UFD)", GOLD, SOFT_GOLD, [
            ("dct", "DCT-II по (H, W)", GOLD, SOFT_GOLD),
        ])
        c.node("ufd_lf", "LF (K×K)\nглобальная структура",
               fillcolor=SOFT_GOLD, color=GOLD, **COMMON_NODE)
        c.node("ufd_hf", "HF (остальное)\nграницы, мелкое",
               fillcolor=SOFT_GOLD, color=GOLD, **COMMON_NODE)
        with c.subgraph() as same:
            same.attr(rank="same")
            same.node("ufd_lf"); same.node("ufd_hf")
        c.edge("ufd_dct", "ufd_lf", color=GOLD, penwidth="1.2")
        c.edge("ufd_dct", "ufd_hf", color=GOLD, penwidth="1.2")
        c.node("ufd_loss",
               "L_UFD  =  w_LF · MSE_LF  +  w_HF · MSE_HF",
               fillcolor=SOFT_GOLD, color=GOLD,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})
        c.edge("ufd_lf", "ufd_loss", color=GOLD, penwidth="1.2")
        c.edge("ufd_hf", "ufd_loss", color=GOLD, penwidth="1.2")
        c.node("ufd_note", "→ negative result:\nLF доминирует HF в ×600",
               shape="plaintext", fontcolor=ACCENT,
               fontname="DejaVu Sans Bold Oblique", fontsize="10")
        c.edge("ufd_loss", "ufd_note", style="invis")

    render(g, "09_kd_variants.png")


# ============================================================
# Fig 2.3 — DKD decomposition
# ============================================================
def fig_dkd():
    g = graphviz.Digraph("dkd")
    g.attr(rankdir="LR", bgcolor="white",
           label="Декомпозиция KL  →  TCKD + NCKD  (Zhao et al., CVPR 2022)",
           labelloc="t", fontname="DejaVu Sans Bold", fontsize="15",
           ranksep="1.0", nodesep="0.5", pad="0.4")

    g.node("kl", "Полная KL\nKL(p_t || p_s)",
           fillcolor=SOFT_NAVY, color=NAVY,
           fontname="DejaVu Sans Bold", fontsize="12",
           shape="box", style="rounded,filled", penwidth="2",
           width="2.0", height="1.2")

    g.node("tckd",
           "TCKD\n(Target-Class)\n\n"
           "бинарная KL на\ntarget-классе vs остальные",
           fillcolor=SOFT_RED, color=ACCENT,
           fontname="DejaVu Sans Bold", fontsize="11",
           shape="box", style="rounded,filled", penwidth="1.5")

    g.node("nckd",
           "NCKD\n(Non-target Class)\n\n"
           "KL по не-target классам\nпосле renormalize",
           fillcolor=SOFT_RED, color=ACCENT,
           fontname="DejaVu Sans Bold", fontsize="11",
           shape="box", style="rounded,filled", penwidth="1.5")

    g.node("dkd",
           "L_DKD =\nα · TCKD + β · NCKD\n\nα = 1,   β = 8",
           fillcolor=SOFT_GOLD, color=GOLD,
           fontname="DejaVu Sans Bold", fontsize="11",
           shape="box", style="rounded,filled", penwidth="2",
           width="2.2", height="1.4")

    g.edge("kl", "tckd", label="разделение",
           fontname="DejaVu Sans Italic", fontsize="10", fontcolor=INK,
           color=INK, penwidth="1.5")
    g.edge("kl", "nckd", label="разделение",
           fontname="DejaVu Sans Italic", fontsize="10", fontcolor=INK,
           color=INK, penwidth="1.5")
    g.edge("tckd", "dkd", label="× α",
           fontname="DejaVu Sans Bold", fontsize="11", fontcolor=ACCENT,
           color=ACCENT, penwidth="1.5")
    g.edge("nckd", "dkd", label="× β",
           fontname="DejaVu Sans Bold", fontsize="11", fontcolor=ACCENT,
           color=ACCENT, penwidth="1.5")

    g.node("note",
           "Vanilla KL:  NCKD автоматически подавляется фактором (1 − p_t(c))\n"
           "→  теряем сигнал на редких классах.\n"
           "DKD устраняет это подавление, делая non-target signal первоклассным.",
           shape="plaintext", fontcolor=MUTED,
           fontname="DejaVu Sans Oblique", fontsize="10")
    # Push the note to its own row at the bottom (invisible edge)
    g.edge("dkd", "note", style="invis")

    render(g, "12_dkd_decomposition.png")


# ============================================================
# Fig 2.4 — Decoder comparison
# ============================================================
def fig_decoder_comparison():
    g = graphviz.Digraph("decoders")
    g.attr(rankdir="LR", bgcolor="white",
           label="Сравнение декодеров: свёрточный (HoVer-Net) vs SSM (Mamba)",
           labelloc="t", fontname="DejaVu Sans Bold", fontsize="15",
           ranksep="0.5", nodesep="0.4", pad="0.4", compound="true")

    # ----- Left: HoVer-Net FPN -----
    with g.subgraph(name="cluster_hover") as c:
        c.attr(label="Декодер HoVer-Net (FPN)",
               labelloc="t", fontsize="13", fontname="DejaVu Sans Bold",
               color=NAVY, fontcolor=NAVY, style="rounded", margin="14")

        # 4 encoder stages
        stages = [
            ("hv_s4", "Stage 4\n8×8 × 512"),
            ("hv_s3", "Stage 3\n16×16 × 256"),
            ("hv_s2", "Stage 2\n32×32 × 128"),
            ("hv_s1", "Stage 1\n64×64 × 64"),
        ]
        for sid, lbl in stages:
            c.node(sid, lbl, fillcolor=SOFT_NAVY, color=NAVY, **COMMON_NODE)

        # 4 decoder blocks
        decs = [
            ("hv_d4", "upsample + concat\n+ conv-conv"),
            ("hv_d3", "upsample + concat\n+ conv-conv"),
            ("hv_d2", "upsample + concat\n+ conv-conv"),
            ("hv_d1", "upsample + conv-conv\n→ 32-ch feature"),
        ]
        for sid, lbl in decs:
            c.node(sid, lbl, fillcolor=SOFT, color=NAVY, **COMMON_NODE)

        c.node("hv_heads", "3 головы\n(binary, HV, type)",
               fillcolor=SOFT_GREEN, color=GREEN,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})

        # Encoder vertical chain
        c.edge("hv_s4", "hv_d4", color=NAVY, penwidth="1.5", label="bottleneck",
               fontname="DejaVu Sans Italic", fontsize="9", fontcolor=MUTED)
        # Decoder chain (vertical) and skip connections from encoder
        c.edge("hv_d4", "hv_d3", color=NAVY, penwidth="1.5")
        c.edge("hv_d3", "hv_d2", color=NAVY, penwidth="1.5")
        c.edge("hv_d2", "hv_d1", color=NAVY, penwidth="1.5")
        c.edge("hv_s3", "hv_d3", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("hv_s2", "hv_d2", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("hv_s1", "hv_d1", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("hv_d1", "hv_heads", color=GREEN, penwidth="1.8")

    # ----- Right: Mamba PVM -----
    with g.subgraph(name="cluster_mamba") as c:
        c.attr(label="Декодер Mamba (PVM, UltraLight VM-UNet)",
               labelloc="t", fontsize="13", fontname="DejaVu Sans Bold",
               color=ACCENT, fontcolor=ACCENT, style="rounded", margin="14")

        for sid, lbl in [
            ("mb_s4", "Stage 4\n8×8 × 512"),
            ("mb_s3", "Stage 3\n16×16 × 256"),
            ("mb_s2", "Stage 2\n32×32 × 128"),
            ("mb_s1", "Stage 1\n64×64 × 64"),
        ]:
            c.node(sid, lbl, fillcolor=SOFT_NAVY, color=NAVY, **COMMON_NODE)

        for sid, lbl in [
            ("mb_d4", "PVM block\nM₁ ‖ M₂ ‖ M₃ ‖ M₄  + GN res"),
            ("mb_d3", "PVM block\nM₁ ‖ M₂ ‖ M₃ ‖ M₄  + GN res"),
            ("mb_d2", "PVM block\nM₁ ‖ M₂ ‖ M₃ ‖ M₄  + GN res"),
            ("mb_d1", "PVM block\n→ 32-ch feature"),
        ]:
            c.node(sid, lbl, fillcolor="#FFF1F0", color=ACCENT, **COMMON_NODE)

        c.node("mb_heads", "3 головы\n(binary, HV, type)",
               fillcolor=SOFT_GREEN, color=GREEN,
               fontname="DejaVu Sans Bold",
               **{k: v for k, v in COMMON_NODE.items() if k != "fontname"})

        c.edge("mb_s4", "mb_d4", color=ACCENT, penwidth="1.5", label="bottleneck",
               fontname="DejaVu Sans Italic", fontsize="9", fontcolor=MUTED)
        c.edge("mb_d4", "mb_d3", color=ACCENT, penwidth="1.5")
        c.edge("mb_d3", "mb_d2", color=ACCENT, penwidth="1.5")
        c.edge("mb_d2", "mb_d1", color=ACCENT, penwidth="1.5")
        c.edge("mb_s3", "mb_d3", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("mb_s2", "mb_d2", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("mb_s1", "mb_d1", color=MUTED, penwidth="1.0",
               style="dashed", label="skip", fontname="DejaVu Sans Italic",
               fontsize="9", fontcolor=MUTED)
        c.edge("mb_d1", "mb_heads", color=GREEN, penwidth="1.8")

    render(g, "11_decoder_comparison.png")


if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}")
    fig_pipeline()
    fig_kd_variants()
    fig_dkd()
    fig_decoder_comparison()
    print("Done")
