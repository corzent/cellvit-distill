#!/bin/bash
# 3-fold cross-validation on RTX 5090
# Runs baseline + response KD for each of 3 PanNuke fold splits.
# Estimated time on 5090: ~3-4 hours total (parallel batches × fold).
set -e

cd /workspace/cellvit-distill
source .venv/bin/activate
export PYTHONPATH="$(pwd)/vendor/CellViT:$PYTHONPATH"

LOG_DIR=logs/3fold_cv
mkdir -p $LOG_DIR

# Each split: train on 2 folds, test on 1
# Run baseline + distill pair for each split
for HOLD_OUT in 1 2 3; do
    case $HOLD_OUT in
        1) TRAIN="[2, 3]" ;;
        2) TRAIN="[1, 3]" ;;
        3) TRAIN="[1, 2]" ;;
    esac

    echo "============================================"
    echo "Split: hold out fold $HOLD_OUT, train on $TRAIN"
    echo "============================================"

    # Baseline (no distill)
    python -m cellvit_distill.scripts.train \
        --config cellvit_distill/configs/fastvit_nulite_v2.yaml \
        --override "data.train_folds=$TRAIN" "data.val_fold=$HOLD_OUT" \
        > $LOG_DIR/baseline_fold${HOLD_OUT}.log 2>&1

    # Distill (response KD)
    python -m cellvit_distill.scripts.train \
        --config cellvit_distill/configs/fastvit_nulite_v2.yaml \
        --override "data.train_folds=$TRAIN" "data.val_fold=$HOLD_OUT" \
        "training.distillation.enabled=true" \
        > $LOG_DIR/distill_fold${HOLD_OUT}.log 2>&1
done

# After training, aggregate results
echo "============================================"
echo "Aggregating 3-fold results..."
echo "============================================"
python -m cellvit_distill.scripts.aggregate_3fold > $LOG_DIR/summary.txt
cat $LOG_DIR/summary.txt
