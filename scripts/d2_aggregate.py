#!/usr/bin/env python3
"""Aggregate D2 3-fold results: PanNuke mean±std + MoNuSeg mean±std + paired stats.

Reads:
  - logs/d2_critical/runs/{cond}_fold{1,2}_*/eval_fold{N}{,_tta}.json
  - For fold 3: reuse logs/{c3_sweep, c4_*, mamba_baseline}/runs/*/eval_fold3*.json
  - logs/monuseg_d2/runs/*/eval_monuseg{,_tta}.json (after followup)
  - per_image.npz files alongside each eval json (for paired Wilcoxon)

Writes a single markdown table to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# Map condition -> (per-fold run dirs)
# Fold 1/2 from d2_critical; fold 3 from prior C3/C4 runs.
CONDITIONS = {
    "mamba_cs4_no_kd": {
        1: "logs/d2_critical/runs/mamba_cs4_no_kd_fold1_baseline_fastvit_s12_20260518_125138",
        2: "logs/d2_critical/runs/mamba_cs4_no_kd_fold2_baseline_fastvit_s12_20260518_141218",
        3: "logs/c3_sweep/runs/cs4_d64_g4_baseline_fastvit_s12_20260518_015544",
    },
    "mamba_cs4_kl": {
        1: "logs/d2_critical/runs/mamba_cs4_kl_fold1_distill_fastvit_s12_20260518_152115",
        2: "logs/d2_critical/runs/mamba_cs4_kl_fold2_distill_fastvit_s12_20260518_165715",
        3: "logs/c4_mamba_kl/runs/distill_fastvit_s12_20260518_071225",
    },
    "mamba_cs4_dkd": {
        1: "logs/d2_critical/runs/mamba_cs4_dkd_fold1_distill_fastvit_s12_20260518_183536",
        2: "logs/d2_critical/runs/mamba_cs4_dkd_fold2_distill_fastvit_s12_20260518_200946",
        3: "logs/c4_mamba_dkd/runs/distill_fastvit_s12_20260518_085801",
    },
    "mamba_default": {
        1: "logs/d2_critical/runs/mamba_default_fold1_baseline_fastvit_s12_20260518_211001",
        2: "logs/d2_critical/runs/mamba_default_fold2_baseline_fastvit_s12_20260518_220954",
        3: "logs/mamba_baseline/runs/baseline_fastvit_s12_20260517_210340",
    },
}


def _load_json(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _load_per_image(p: Path) -> Optional[Dict[str, np.ndarray]]:
    if not p.exists():
        return None
    return dict(np.load(p))


def mean_std(values: List[float]) -> str:
    if not values or any(v is None for v in values):
        return "NA"
    if len(values) < 2:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def collect(condition: str, folds: Dict[int, str]):
    """Returns dict with PanNuke mPQ_TTA / bPQ_TTA / F1_TTA / per-class lists,
    plus MoNuSeg bPQ_TTA / F1_TTA, plus per-image arrays for stat tests."""
    out = {
        "mpq_tta": [], "bpq_tta": [], "f1_tta": [],
        "mpq_plain": [], "bpq_plain": [], "f1_plain": [],
        "monuseg_bpq_tta": [], "monuseg_f1_tta": [],
        "per_image_mpq_tta": [],     # list of arrays, one per fold
        "per_image_monuseg_tta": [],
        "per_class_tta": {1: [], 2: [], 3: [], 4: [], 5: []},
        "missing": [],
    }
    for fold, run_dir in folds.items():
        rd = Path(run_dir)
        # PanNuke TTA
        j_tta = _load_json(rd / f"eval_fold{fold}_tta.json")
        j_plain = _load_json(rd / f"eval_fold{fold}.json")
        if j_tta:
            out["mpq_tta"].append(j_tta["mPQ"])
            out["bpq_tta"].append(j_tta["bPQ"])
            out["f1_tta"].append(j_tta["F1_detection"])
            for c in range(1, 6):
                k = f"PQ_class_{c}"
                if k in j_tta:
                    out["per_class_tta"][c].append(j_tta[k])
        else:
            out["missing"].append(f"PanNuke TTA fold {fold}")
        if j_plain:
            out["mpq_plain"].append(j_plain["mPQ"])
            out["bpq_plain"].append(j_plain["bPQ"])
            out["f1_plain"].append(j_plain["F1_detection"])
        # per-image TTA
        npz = _load_per_image(rd / f"eval_fold{fold}_tta_per_image.npz")
        if npz is not None and "mpq" in npz:
            out["per_image_mpq_tta"].append(npz["mpq"])
        # MoNuSeg
        m_tta = _load_json(rd / "eval_monuseg_tta.json")
        if m_tta:
            out["monuseg_bpq_tta"].append(m_tta["bPQ"])
            out["monuseg_f1_tta"].append(m_tta["F1_detection"])
        # MoNuSeg per-image
        m_npz = _load_per_image(rd / "eval_monuseg_tta_per_image.npz")
        if m_npz is not None and "bpq" in m_npz:
            out["per_image_monuseg_tta"].append(m_npz["bpq"])
    return out


def paired_wilcoxon_concat(a: List[np.ndarray], b: List[np.ndarray]) -> Optional[dict]:
    """Concat per-image arrays across folds, run paired Wilcoxon + bootstrap CI."""
    if not a or not b or len(a) != len(b):
        return None
    A = np.concatenate(a)
    B = np.concatenate(b)
    if len(A) != len(B):
        return None
    mask = ~(np.isnan(A) | np.isnan(B))
    A, B = A[mask], B[mask]
    if len(A) < 2:
        return None
    diff = B - A
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    try:
        w_stat, w_p = stats.wilcoxon(B, A, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        w_p = float("nan")
    return {
        "n": len(A),
        "mean_delta": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "wilcoxon_p": float(w_p),
    }


def main() -> int:
    print()
    print("=" * 78)
    print("D2 critical 3-fold aggregate (Mamba conditions)")
    print("=" * 78)
    print()

    data = {cond: collect(cond, folds) for cond, folds in CONDITIONS.items()}

    # ---- PanNuke 3-fold mean ± std table ----
    print("## PanNuke 3-fold mean ± std (TTA)")
    print()
    print("| Condition | n | mPQ (TTA) | bPQ (TTA) | F1 (TTA) |")
    print("|---|---|---|---|---|")
    for cond, d in data.items():
        n = len(d["mpq_tta"])
        print(f"| {cond} | {n} | {mean_std(d['mpq_tta'])} | {mean_std(d['bpq_tta'])} | {mean_std(d['f1_tta'])} |")
    print()

    # ---- Per-class PQ 3-fold ----
    print("## Per-class PQ (TTA, 3-fold mean ± std)")
    print()
    cls_names = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective", 4: "Dead", 5: "Epithelial"}
    header = "| Condition | " + " | ".join(cls_names[c] for c in range(1, 6)) + " |"
    print(header)
    print("|---" * 6 + "|")
    for cond, d in data.items():
        row = f"| {cond} | " + " | ".join(mean_std(d["per_class_tta"][c]) for c in range(1, 6)) + " |"
        print(row)
    print()

    # ---- MoNuSeg 3-fold ----
    print("## MoNuSeg zero-shot 3-fold mean ± std (TTA, n=14 images per fold)")
    print()
    print("| Condition | n_folds | bPQ (TTA) | F1 (TTA) |")
    print("|---|---|---|---|")
    for cond, d in data.items():
        n = len(d["monuseg_bpq_tta"])
        print(f"| {cond} | {n} | {mean_std(d['monuseg_bpq_tta'])} | {mean_std(d['monuseg_f1_tta'])} |")
    print()

    # ---- Paired Wilcoxon / bootstrap CI between conditions ----
    print("## Paired stat tests on per-image PQ (concat across folds)")
    print()
    print("### PanNuke mPQ (TTA)")
    print()
    print("| A | B | n | Δ (B − A) | 95% CI | Wilcoxon p |")
    print("|---|---|---|---|---|---|")
    comparisons = [
        ("mamba_cs4_no_kd", "mamba_cs4_kl"),
        ("mamba_cs4_no_kd", "mamba_cs4_dkd"),
        ("mamba_cs4_no_kd", "mamba_default"),
        ("mamba_cs4_kl",    "mamba_cs4_dkd"),
    ]
    for a, b in comparisons:
        r = paired_wilcoxon_concat(
            data[a]["per_image_mpq_tta"],
            data[b]["per_image_mpq_tta"],
        )
        if r is None:
            print(f"| {a} | {b} | NA | NA | NA | NA |")
        else:
            print(f"| {a} | {b} | {r['n']} | {r['mean_delta']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4g} |")
    print()
    print("### MoNuSeg bPQ (TTA)")
    print()
    print("| A | B | n | Δ (B − A) | 95% CI | Wilcoxon p |")
    print("|---|---|---|---|---|---|")
    for a, b in comparisons:
        r = paired_wilcoxon_concat(
            data[a]["per_image_monuseg_tta"],
            data[b]["per_image_monuseg_tta"],
        )
        if r is None:
            print(f"| {a} | {b} | NA | NA | NA | NA |")
        else:
            print(f"| {a} | {b} | {r['n']} | {r['mean_delta']:+.4f} | "
                  f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['wilcoxon_p']:.4g} |")
    print()

    # ---- Missing data warnings ----
    for cond, d in data.items():
        if d["missing"]:
            print(f"# WARN {cond}: missing {d['missing']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
