#!/usr/bin/env python3
"""Aggregate 3-fold CV results into mean ± std tables.

Reads a manifest of `condition\tfold\trun_dir` rows (one per training run)
and the corresponding eval_fold{N}.json / eval_fold{N}_tta.json files inside
each run_dir, then prints a markdown summary.

Usage:
    python -m cellvit_distill.scripts.aggregate_3fold \
        --manifest logs/3fold_cv/runs.manifest
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


METRIC_ORDER = ["mPQ", "bPQ", "F1_detection",
                "PQ_class_1", "PQ_class_2", "PQ_class_3",
                "PQ_class_4", "PQ_class_5"]

CLASS_NAMES = {
    "PQ_class_1": "Neoplastic",
    "PQ_class_2": "Inflammatory",
    "PQ_class_3": "Connective",
    "PQ_class_4": "Dead",
    "PQ_class_5": "Epithelial",
}


def load_manifest(path: Path) -> List[Tuple[str, int, Path]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"Bad manifest row: {line!r}")
        condition, fold_s, run_dir = parts
        rows.append((condition, int(fold_s), Path(run_dir)))
    return rows


def load_eval(run_dir: Path, fold: int, tta: bool) -> Dict[str, float] | None:
    suffix = "_tta" if tta else ""
    p = run_dir / f"eval_fold{fold}{suffix}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)  # sample std
    return mean, math.sqrt(var)


def fmt(mean: float, std: float, digits: int = 4) -> str:
    if math.isnan(mean):
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize(rows: List[Tuple[str, int, Path]], tta: bool) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Returns {condition: {metric: (mean, std)}}."""
    bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    missing: List[Tuple[str, int]] = []

    for condition, fold, run_dir in rows:
        metrics = load_eval(run_dir, fold, tta)
        if metrics is None:
            missing.append((condition, fold))
            continue
        for k in METRIC_ORDER:
            if k in metrics:
                bucket[condition][k].append(float(metrics[k]))

    if missing:
        kind = "TTA" if tta else "no-TTA"
        print(f"\n[warn] missing {kind} evals for: " +
              ", ".join(f"{c}/fold{f}" for c, f in missing))

    return {
        cond: {m: mean_std(vals) for m, vals in metrics.items()}
        for cond, metrics in bucket.items()
    }


def print_table(title: str, summary: Dict[str, Dict[str, Tuple[float, float]]]):
    print(f"\n## {title}\n")
    if not summary:
        print("_(no data)_")
        return

    conditions = list(summary.keys())
    metric_cols = [m for m in METRIC_ORDER
                   if any(m in summary[c] for c in conditions)]

    header = ["Condition"] + [
        CLASS_NAMES.get(m, m) for m in metric_cols
    ]
    sep = ["---"] * len(header)
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for cond in conditions:
        row = [cond]
        for m in metric_cols:
            if m in summary[cond]:
                mean, std = summary[cond][m]
                row.append(fmt(mean, std))
            else:
                row.append("—")
        print("| " + " | ".join(row) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to runs.manifest (condition\\tfold\\trun_dir per line)")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    print(f"Manifest: {args.manifest} ({len(rows)} runs)")

    summary_notta = summarize(rows, tta=False)
    summary_tta = summarize(rows, tta=True)

    print_table("3-fold CV (no TTA), mean ± std (sample)", summary_notta)
    print_table("3-fold CV (8-way TTA), mean ± std (sample)", summary_tta)


if __name__ == "__main__":
    main()
