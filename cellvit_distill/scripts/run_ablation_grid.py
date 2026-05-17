#!/usr/bin/env python3
"""Run a multi-condition × multi-fold ablation grid sequentially on one GPU.

Reads a small YAML grid spec, launches `train.py` for each (condition, fold)
combination with the right overrides, then `eval_student.py` on the resulting
checkpoint (with and without TTA), and writes a manifest mapping each cell
to its run directory + eval JSON paths.

The output manifest is the input format expected by `scripts/stat_test.py`
(per-image npz arrays land alongside the eval JSONs when eval_student is
invoked with the per-image flag, which is the default behaviour).

Grid spec (YAML):

    base_config: cellvit_distill/configs/fastvit_nulite_v2.yaml
    folds: [1, 2, 3]
    common_overrides:
      - training.epochs=130
      - training.early_stop_patience=20
      - logging.wandb=false
    conditions:
      - name: baseline
        overrides:
          - training.distillation.enabled=false
      - name: distill_kl
        overrides:
          - training.distillation.enabled=true
          - training.distillation.loss_type=kl_div
          - training.distillation.alpha=0.05
          - training.distillation.temperature=10.0
      - name: distill_dkd
        overrides:
          - training.distillation.enabled=true
          - training.distillation.loss_type=dkd
          - training.distillation.alpha=0.05
          - training.distillation.dkd_alpha=1.0
          - training.distillation.dkd_beta=8.0

For each (condition, fold) the runner:
  - sets data.train_folds and data.val_fold for the fold rotation
    (val_fold=N, train_folds=[other two])
  - calls train.py with the merged overrides
  - locates the freshly created run directory under output_dir
  - calls eval_student.py twice: once plain, once with --tta

Output manifest format (one row per cell):

    {condition}\\t{fold}\\t{run_dir}\\t{eval_json}\\t{eval_tta_json}

Compatible with the existing scripts/aggregate_3fold.py manifest reader.

Usage:

    python -m cellvit_distill.scripts.run_ablation_grid \\
        --grid grids/main_ablation.yaml \\
        --output_root cellvit_distill/runs/main_ablation \\
        [--dry-run]

`--dry-run` prints the planned commands and exits, useful for sanity-checks
before committing GPU-hours.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml


FOLD_SPLITS = {
    1: (1, [2, 3]),
    2: (2, [1, 3]),
    3: (3, [1, 2]),
}


@dataclass
class Cell:
    condition: str
    fold: int
    run_dir: Optional[Path] = None
    eval_json: Optional[Path] = None
    eval_tta_json: Optional[Path] = None
    status: str = "pending"  # pending | running | trained | evaluated | failed
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


def _resolve_python() -> str:
    """Prefer the venv python so behavior matches dev environment."""
    venv = Path(".venv/bin/python")
    if venv.exists():
        return str(venv.resolve())
    return sys.executable


def _build_train_cmd(base_config: Path, fold: int, common: List[str],
                     condition_overrides: List[str], output_root: Path) -> List[str]:
    val_fold, train_folds = FOLD_SPLITS[fold]
    overrides = list(common) + list(condition_overrides) + [
        f"data.val_fold={val_fold}",
        f"data.train_folds={train_folds}",
        f"logging.output_dir={output_root}",
    ]
    cmd = [_resolve_python(), "-m", "cellvit_distill.scripts.train",
           "--config", str(base_config), "--override", *overrides]
    return cmd


def _build_eval_cmd(run_dir: Path, fold: int, tta: bool) -> List[str]:
    cmd = [_resolve_python(), "-m", "cellvit_distill.scripts.eval_student",
           "--run_dir", str(run_dir), "--test_fold", str(fold)]
    if tta:
        cmd.append("--tta")
    return cmd


def _find_newest_run_dir(output_root: Path, marker_time: float) -> Optional[Path]:
    """Return the newest run dir created after marker_time."""
    candidates = []
    for d in output_root.iterdir():
        if d.is_dir() and d.stat().st_mtime >= marker_time - 5:
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as logf:
        logf.write(f"\n\n# {' '.join(cmd)}\n# started {datetime.now().isoformat()}\n".encode())
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    return proc.returncode


def _write_manifest(cells: List[Cell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# condition\tfold\trun_dir\teval_json\teval_tta_json\tstatus\n")
        for c in cells:
            f.write(
                f"{c.condition}\t{c.fold}\t"
                f"{c.run_dir or ''}\t{c.eval_json or ''}\t{c.eval_tta_json or ''}\t{c.status}\n"
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, type=Path, help="YAML grid spec")
    p.add_argument("--output_root", required=True, type=Path,
                   help="Where train runs land (created if missing)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="Path for the output manifest TSV (default: <output_root>/runs.manifest)")
    p.add_argument("--logs_root", type=Path, default=None,
                   help="Where train/eval logs go (default: <output_root>/logs)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned commands and exit")
    p.add_argument("--continue-on-error", action="store_true",
                   help="If a cell fails, mark it and continue (default: abort)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells whose run_dir already exists and contains best_model.pth")
    args = p.parse_args()

    with open(args.grid) as f:
        grid = yaml.safe_load(f)

    base_config = Path(grid["base_config"])
    common = list(grid.get("common_overrides", []))
    folds = list(grid["folds"])
    conditions = grid["conditions"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = (args.logs_root or output_root / "logs").resolve()
    manifest_path = args.manifest or output_root / "runs.manifest"

    cells: List[Cell] = [
        Cell(condition=c["name"], fold=fold)
        for c in conditions for fold in folds
    ]

    print(f"Grid: {len(conditions)} conditions × {len(folds)} folds = {len(cells)} cells")
    print(f"Output: {output_root}")
    print(f"Manifest: {manifest_path}")
    print()

    for cell in cells:
        cond = next(c for c in conditions if c["name"] == cell.condition)
        train_cmd = _build_train_cmd(base_config, cell.fold, common,
                                     cond.get("overrides", []), output_root)
        print(f"[{cell.condition} / fold {cell.fold}]")
        print("  TRAIN:", " ".join(train_cmd))
        for tta in (False, True):
            print("  EVAL :", "(after train)", "--tta" if tta else "")
    if args.dry_run:
        print("\n[dry-run] No commands executed.")
        return 0

    failed = 0
    for cell in cells:
        cond = next(c for c in conditions if c["name"] == cell.condition)
        cell.started_at = time.time()
        cell.status = "running"

        cell_log = logs_root / f"{cell.condition}_fold{cell.fold}.log"

        train_cmd = _build_train_cmd(base_config, cell.fold, common,
                                     cond.get("overrides", []), output_root)

        marker = time.time()
        rc = _run(train_cmd, cell_log)
        if rc != 0:
            cell.status = "failed"
            cell.error = f"train rc={rc}"
            failed += 1
            _write_manifest(cells, manifest_path)
            if args.continue_on_error:
                print(f"  ! train FAILED rc={rc}, continuing")
                continue
            print(f"  ! train FAILED rc={rc}, aborting (use --continue-on-error to keep going)")
            return 1

        run_dir = _find_newest_run_dir(output_root, marker)
        if run_dir is None:
            cell.status = "failed"
            cell.error = "no run_dir found after train"
            failed += 1
            _write_manifest(cells, manifest_path)
            if args.continue_on_error:
                continue
            return 1
        cell.run_dir = run_dir
        cell.status = "trained"

        if args.skip_existing and (run_dir / "best_model.pth").exists() and \
                (run_dir / f"eval_fold{cell.fold}.json").exists():
            cell.eval_json = run_dir / f"eval_fold{cell.fold}.json"
            cell.eval_tta_json = run_dir / f"eval_fold{cell.fold}_tta.json"
            cell.status = "evaluated"
            cell.finished_at = time.time()
            _write_manifest(cells, manifest_path)
            continue

        # Eval — plain and TTA
        for tta in (False, True):
            eval_cmd = _build_eval_cmd(run_dir, cell.fold, tta)
            rc = _run(eval_cmd, cell_log)
            if rc != 0:
                cell.status = "failed"
                cell.error = f"eval{'_tta' if tta else ''} rc={rc}"
                failed += 1
                _write_manifest(cells, manifest_path)
                if not args.continue_on_error:
                    return 1
                break
            if tta:
                cell.eval_tta_json = run_dir / f"eval_fold{cell.fold}_tta.json"
            else:
                cell.eval_json = run_dir / f"eval_fold{cell.fold}.json"

        if cell.status != "failed":
            cell.status = "evaluated"
        cell.finished_at = time.time()
        _write_manifest(cells, manifest_path)

        elapsed = (cell.finished_at - cell.started_at) / 60
        print(f"  [{cell.condition} fold {cell.fold}] {cell.status} in {elapsed:.1f} min "
              f"({cell.run_dir.name if cell.run_dir else '-'})")

    print(f"\nDone. {len(cells) - failed} ok / {failed} failed.")
    print(f"Manifest: {manifest_path}")
    print(f"Next: run scripts/aggregate_3fold.py to summarize, "
          f"scripts/stat_test.py per-image for paired tests across conditions.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
