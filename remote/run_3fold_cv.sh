#!/bin/bash
# 3-fold cross-validation on RTX 5090.
# For each PanNuke fold split: trains baseline and response-KD, then evaluates
# each with and without 8-way TTA on the held-out fold.
# Writes a manifest of (condition, fold, run_dir) for the aggregator.
# Idempotent: re-running skips (condition, fold) pairs already in the manifest.
#
# Estimated time on RTX 5090 with optimizations below: ~3-5 h total
# (6 train runs + 12 evals). Hard to predict precisely until first run gives
# real per-epoch tempo at batch 32 + 16 workers + val_every 5; original
# config defaults would put this at 25-30 h, so we MUST tune.
set -e
set -o pipefail

cd /workspace/cellvit-distill
source .venv/bin/activate
export PYTHONPATH="$(pwd)/vendor/CellViT:$PYTHONPATH"

LOG_DIR=logs/3fold_cv
mkdir -p "$LOG_DIR"
MANIFEST="$LOG_DIR/runs.manifest"
touch "$MANIFEST"  # do not truncate — supports resume

BASELINE_CFG=cellvit_distill/configs/fastvit_nulite_v2.yaml

# Paths inside this container. Configs hard-code laptop paths; we override here.
DATA_DIR=/workspace/cellvit-distill/datasets/pannuke
SOFT_TARGETS_DIR=/workspace/cellvit-distill/datasets/pannuke/soft_targets
TEACHER_CKPT=/workspace/cellvit-distill/checkpoints/CellViT-SAM-H-x40.pth
OUTPUT_DIR=/workspace/cellvit-distill/cellvit_distill/runs

# 5090 has 32 GB VRAM; student is ~12M params. batch 8 (config default) uses
# only ~3-4 GB and severely underutilizes the GPU. batch 32 brings VRAM
# usage to ~12-16 GB and cuts wall-clock by ~4×. lr unchanged: AdamW with
# 10-epoch warmup absorbs the 4× batch increase without retuning.
BATCH_SIZE=32

# vast.ai 5090 instance has 32 vCPU; config default num_workers=4 leaves
# the data loader CPU-bound. Smoke test showed ~5 patches/sec on batch 8,
# meaning GPU was ~90% idle waiting on data. Bumping to 16 workers.
NUM_WORKERS=16

# Validate every 5 epochs instead of every epoch. validate() runs the
# expensive HoVer-watershed + linear_sum_assignment per image on CPU, so
# val takes ~1/3 the time of train per epoch. early_stop_patience=20 in
# epoch-units still works correctly (4 stagnant validations trigger stop).
VAL_EVERY=5

# Run one training + eval pass.
# Args: condition_label, hold_out_fold, train_folds_yaml, config_path, extra_overrides...
run_one() {
    local condition="$1"; shift
    local hold_out="$1"; shift
    local train_folds="$1"; shift
    local config="$1"; shift

    if grep -q "^${condition}	${hold_out}	" "$MANIFEST" 2>/dev/null; then
        echo "[skip] ${condition}/fold${hold_out} already in manifest"
        return
    fi

    local log="$LOG_DIR/${condition}_fold${hold_out}.log"

    echo "============================================"
    echo "[${condition}] hold out ${hold_out}, train on ${train_folds}"
    echo "  log: $log"
    echo "============================================"

    python -m cellvit_distill.scripts.train \
        --config "$config" \
        --override \
            "data.data_dir=${DATA_DIR}" \
            "data.soft_targets_dir=${SOFT_TARGETS_DIR}" \
            "teacher.checkpoint=${TEACHER_CKPT}" \
            "logging.output_dir=${OUTPUT_DIR}" \
            "data.train_folds=${train_folds}" \
            "data.val_fold=${hold_out}" \
            "training.batch_size=${BATCH_SIZE}" \
            "data.num_workers=${NUM_WORKERS}" \
            "training.val_every_n_epochs=${VAL_EVERY}" \
            "$@" \
        2>&1 | tee "$log"

    # Extract run_dir from train.py's `Output: <path>` line.
    # `|| true` so set -e doesn't kill us before our own error message runs.
    local run_dir
    run_dir=$(grep -m1 '^Output: ' "$log" | sed 's/^Output: //' || true)
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "ERROR: could not locate run_dir for ${condition}/fold${hold_out}" >&2
        exit 1
    fi
    echo "  run_dir: $run_dir"

    # Eval on held-out fold, both with and without TTA.
    python -m cellvit_distill.scripts.eval_student \
        --run_dir "$run_dir" --test_fold "$hold_out" \
        2>&1 | tee -a "$log"
    python -m cellvit_distill.scripts.eval_student \
        --run_dir "$run_dir" --test_fold "$hold_out" --tta \
        2>&1 | tee -a "$log"

    printf '%s\t%s\t%s\n' "$condition" "$hold_out" "$run_dir" >> "$MANIFEST"
}

for HOLD_OUT in 1 2 3; do
    case $HOLD_OUT in
        1) TRAIN="[2, 3]" ;;
        2) TRAIN="[1, 3]" ;;
        3) TRAIN="[1, 2]" ;;
    esac

    run_one baseline      "$HOLD_OUT" "$TRAIN" "$BASELINE_CFG"
    run_one distill_resp  "$HOLD_OUT" "$TRAIN" "$BASELINE_CFG" "training.distillation.enabled=true"
done

echo "============================================"
echo "Aggregating 3-fold results..."
echo "============================================"
python -m cellvit_distill.scripts.aggregate_3fold --manifest "$MANIFEST" \
    | tee "$LOG_DIR/summary.md"
