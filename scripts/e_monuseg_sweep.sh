#!/usr/bin/env bash
# Phase E: MoNuSeg zero-shot cross-dataset eval over today's checkpoints.
#
# 5 models × {plain, TTA} = 10 eval runs, ~6 min each = ~60 min total.
# Results written next to each run_dir as eval_monuseg{,_tta}.json /
# eval_monuseg{,_tta}_per_image.npz. Summary TSV at the end.

set -e
cd "$(dirname "$0")/.."

export PYTHONPATH="vendor/CellViT:vendor/NuLite:."
export CELLVIT_DATA_DIR="$PWD/datasets/pannuke"
export HF_HOME="$PWD/.hf_home"

# label | run_dir
RUNS=(
  "hover_ufd     | logs/ufd_final/runs/distill_fastvit_s12_20260517_191745"
  "mamba_default | logs/mamba_baseline/runs/baseline_fastvit_s12_20260517_210340"
  "mamba_cs4_d64 | logs/c3_sweep/runs/cs4_d64_g4_baseline_fastvit_s12_20260518_015544"
  "mamba_cs4_kl  | logs/c4_mamba_kl/runs/distill_fastvit_s12_20260518_071225"
  "mamba_cs4_dkd | logs/c4_mamba_dkd/runs/distill_fastvit_s12_20260518_085801"
)

OUT="$PWD/logs/monuseg_sweep"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
echo -e "label\trun_dir\tbPQ_plain\tF1_plain\tbPQ_tta\tF1_tta" > "$SUMMARY"

for entry in "${RUNS[@]}"; do
  label=$(echo "$entry" | cut -d'|' -f1 | xargs)
  run_dir=$(echo "$entry" | cut -d'|' -f2 | xargs)
  if [ ! -f "$run_dir/best_model.pth" ]; then
    echo "SKIP $label: $run_dir/best_model.pth not found"; continue
  fi
  log="$OUT/${label}.log"
  echo "=== $label ($run_dir) ==="

  .venv/bin/python -m cellvit_distill.scripts.eval_monuseg \
    --run_dir "$run_dir" --split test --batch_size 16 \
    > "$log" 2>&1 || echo "PLAIN eval FAILED $label"
  .venv/bin/python -m cellvit_distill.scripts.eval_monuseg \
    --run_dir "$run_dir" --split test --batch_size 16 --tta \
    >> "$log" 2>&1 || echo "TTA eval FAILED $label"

  # Parse JSONs
  bpq_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")
  bpq_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg_tta.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_monuseg_tta.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")

  echo -e "${label}\t${run_dir}\t${bpq_p}\t${f1_p}\t${bpq_t}\t${f1_t}" >> "$SUMMARY"
  echo "  -> bPQ_plain=${bpq_p}  bPQ_tta=${bpq_t}"
done

echo
echo "=============== MONUSEG SWEEP DONE ==============="
column -t -s $'\t' "$SUMMARY"
