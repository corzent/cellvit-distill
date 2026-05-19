#!/usr/bin/env bash
# Phase E follow-up: MoNuSeg eval of the 8 D2 critical checkpoints.
# Combined with the existing fold-3 MoNuSeg results from logs/monuseg_sweep/,
# yields a 3-fold cross-dataset table for each of the 4 Mamba conditions.

set -e
cd "$(dirname "$0")/.."

OUT="$PWD/logs/monuseg_d2"
mkdir -p "$OUT"
export PYTHONPATH="vendor/CellViT:vendor/NuLite:."
export CELLVIT_DATA_DIR="$PWD/datasets/pannuke"
export HF_HOME="$PWD/.hf_home"

SUMMARY="$OUT/summary.tsv"
echo -e "condition\tfold\trun_dir\tbPQ_plain\tF1_plain\tbPQ_tta\tF1_tta" > "$SUMMARY"

for run_dir in logs/d2_critical/runs/*/; do
  run_dir="${run_dir%/}"
  cell_name=$(basename "$run_dir" | sed -E 's/_(baseline|distill)_fastvit_s12_[0-9_]+$//')
  condition=$(echo "$cell_name" | sed -E 's/_fold[0-9]+$//')
  fold=$(echo "$cell_name" | grep -oE 'fold[0-9]+$' | grep -oE '[0-9]+')
  log="$OUT/${cell_name}_monuseg.log"
  echo "=== $cell_name ($run_dir) ==="

  .venv/bin/python -m cellvit_distill.scripts.eval_monuseg \
    --run_dir "$run_dir" --split test --batch_size 16 \
    > "$log" 2>&1 || echo "PLAIN FAILED $cell_name"
  .venv/bin/python -m cellvit_distill.scripts.eval_monuseg \
    --run_dir "$run_dir" --split test --batch_size 16 --tta \
    >> "$log" 2>&1 || echo "TTA FAILED $cell_name"

  bpq_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")
  bpq_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg_tta.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg_tta.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")

  echo -e "${condition}\t${fold}\t${run_dir}\t${bpq_p}\t${f1_p}\t${bpq_t}\t${f1_t}" >> "$SUMMARY"
  echo "  -> bPQ_plain=${bpq_p} bPQ_tta=${bpq_t}"
done

echo
echo "=============== MONUSEG D2 FOLLOWUP DONE ==============="
column -t -s $'\t' "$SUMMARY"
