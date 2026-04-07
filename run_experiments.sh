#!/bin/bash
# Run all three experiments sequentially
# Usage: bash run_experiments.sh

set -e
cd "$(dirname "$0")"

export PYTHONPATH="vendor/CellViT:$PYTHONPATH"

echo "=========================================="
echo "Experiment 1: Baseline (no distillation)"
echo "=========================================="
.venv/bin/python -m cellvit_distill.scripts.train \
    --config cellvit_distill/configs/default.yaml

echo ""
echo "=========================================="
echo "Experiment 2: With distillation"
echo "=========================================="
.venv/bin/python -m cellvit_distill.scripts.train \
    --config cellvit_distill/configs/default.yaml \
    --override training.distillation.enabled=true

echo ""
echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
