#!/usr/bin/env python3
"""Statistical comparison between two trained conditions on PanNuke.

Two modes:

  per-image  (preferred — high statistical power)
      Compares two .npz files produced by eval_student.py --save-per-image-…
      (eval_fold{N}[_tta]_per_image.npz). Uses paired Wilcoxon, paired
      t-test, and percentile bootstrap on per-image PQ deltas.

  across-folds  (low power — only 3 paired observations on PanNuke 3-fold CV)
      Compares two lists of per-fold scalars passed via --a-vals / --b-vals.
      Use for headline numbers when per-image arrays were not saved.

Usage:
    # per-image (one fold)
    python -m cellvit_distill.scripts.stat_test per-image \\
        --a path/to/baseline/eval_fold3_per_image.npz \\
        --b path/to/distill/eval_fold3_per_image.npz \\
        --metric mpq

    # per-image (concat across multiple folds)
    python -m cellvit_distill.scripts.stat_test per-image \\
        --a baseline_fold{1,2,3}/eval_fold*_per_image.npz \\
        --b distill_fold{1,2,3}/eval_fold*_per_image.npz \\
        --metric mpq

    # across-folds
    python -m cellvit_distill.scripts.stat_test across-folds \\
        --a-vals 0.4668 0.4627 0.4668 \\
        --b-vals 0.4723 0.4677 0.4695 \\
        --label-a baseline --label-b distill_resp
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats


METRIC_ALIASES = {
    "mpq": "mpq",
    "bpq": "bpq",
    "f1": "f1",
    "f1_detection": "f1",
}


def _load_per_image(paths, metric_key: str) -> np.ndarray:
    """Concatenate per-image metric arrays across one or more npz files."""
    out = []
    for p in paths:
        data = np.load(p)
        if metric_key not in data.files:
            raise KeyError(f"{p}: key {metric_key!r} not in {list(data.files)}")
        out.append(data[metric_key])
    return np.concatenate(out)


def _bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_iter: int = 10_000,
                       alpha: float = 0.05, seed: int = 42) -> tuple:
    """Paired percentile bootstrap CI on mean(b - a). NaN-safe."""
    diffs = b - a
    mask = ~np.isnan(diffs)
    diffs = diffs[mask]
    if len(diffs) == 0:
        return 0.0, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_iter, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(diffs.mean()), (float(lo), float(hi))


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized effect size for paired diff: mean(diff) / std(diff)."""
    d = b - a
    d = d[~np.isnan(d)]
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def _print_report(a: np.ndarray, b: np.ndarray, metric: str,
                  label_a: str, label_b: str) -> int:
    if len(a) != len(b):
        print(f"ERROR: length mismatch a={len(a)} vs b={len(b)} (paired tests require equal n)",
              file=sys.stderr)
        return 1

    # Strip rows where either is NaN (paired NaN handling).
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        print(f"ERROR: only {n} paired observations after NaN filter — cannot test",
              file=sys.stderr)
        return 1

    mean_a, mean_b = a.mean(), b.mean()
    mean_d, (lo, hi) = _bootstrap_ci_diff(a, b)
    d_cohen = _cohens_d_paired(a, b)

    # Paired t-test (assumes diff normality; weak for small n).
    t_stat, t_p = stats.ttest_rel(a, b, nan_policy="omit")

    # Wilcoxon (no normality assumption; preferred for n<30).
    try:
        w_stat, w_p = stats.wilcoxon(b, a, alternative="two-sided", zero_method="wilcox")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        wilcox_err = str(e)
    else:
        wilcox_err = None

    print(f"{'=' * 64}")
    print(f"Paired comparison: {label_a}  vs  {label_b}  (metric: {metric})")
    print(f"{'=' * 64}")
    print(f"n (paired):       {n}")
    print(f"mean {label_a}:    {mean_a:.4f}")
    print(f"mean {label_b}:    {mean_b:.4f}")
    print(f"Δ (b − a):        {mean_d:+.4f}")
    print(f"95% CI on Δ:      [{lo:+.4f}, {hi:+.4f}] (10k percentile bootstrap)")
    print(f"Cohen's d (paired): {d_cohen:+.3f}")
    print(f"Paired t-test:    t={t_stat:.3f},  p={t_p:.4g}")
    if wilcox_err:
        print(f"Wilcoxon:         FAILED ({wilcox_err})")
    else:
        print(f"Wilcoxon:         W={w_stat:.3f},  p={w_p:.4g}")
    print()

    # Verdict: significant if CI excludes 0 AND smaller of two p-values < 0.05.
    ci_excl_0 = (lo > 0 or hi < 0)
    p_min = min(t_p, w_p) if not np.isnan(w_p) else t_p
    sig = ci_excl_0 and p_min < 0.05
    direction = ("higher" if mean_d > 0 else "lower") if sig else "indistinguishable from"
    print(f"Verdict: {label_b} is {direction} {label_a} on {metric} "
          f"(α=0.05, n={n}, |d|={abs(d_cohen):.2f})")
    return 0


def cmd_per_image(args) -> int:
    metric_key = METRIC_ALIASES.get(args.metric.lower())
    if metric_key is None:
        print(f"ERROR: unknown metric {args.metric!r}; valid: {sorted(set(METRIC_ALIASES.values()))}",
              file=sys.stderr)
        return 2

    a = _load_per_image(args.a, metric_key)
    b = _load_per_image(args.b, metric_key)
    return _print_report(a, b, metric_key, args.label_a, args.label_b)


def cmd_across_folds(args) -> int:
    a = np.asarray(args.a_vals, dtype=np.float64)
    b = np.asarray(args.b_vals, dtype=np.float64)
    if len(a) != len(b):
        print(f"ERROR: a-vals len {len(a)} != b-vals len {len(b)}", file=sys.stderr)
        return 2
    print("Note: across-folds mode has very low statistical power (n=fold count).",
          "Per-image mode is strongly preferred when arrays are available.\n")
    return _print_report(a, b, args.metric, args.label_a, args.label_b)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("per-image", help="Paired test on per-image PQ arrays")
    pi.add_argument("--a", nargs="+", type=Path, required=True,
                    help="One or more per-image npz files for condition A")
    pi.add_argument("--b", nargs="+", type=Path, required=True,
                    help="One or more per-image npz files for condition B (paired with A)")
    pi.add_argument("--metric", default="mpq", help="mpq | bpq | f1")
    pi.add_argument("--label-a", default="A")
    pi.add_argument("--label-b", default="B")
    pi.set_defaults(func=cmd_per_image)

    af = sub.add_parser("across-folds", help="Paired test on per-fold scalars (low power)")
    af.add_argument("--a-vals", nargs="+", type=float, required=True)
    af.add_argument("--b-vals", nargs="+", type=float, required=True)
    af.add_argument("--metric", default="mPQ")
    af.add_argument("--label-a", default="A")
    af.add_argument("--label-b", default="B")
    af.set_defaults(func=cmd_across_folds)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
