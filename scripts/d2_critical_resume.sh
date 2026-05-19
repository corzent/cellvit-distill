#!/usr/bin/env bash
# Phase D2 critical RESUME: 7 remaining cells after the branch-switch incident.
# Cell 1 (mamba_cs4_no_kd_fold1) already trained + re-eval'd manually.

set -e
cd "$(dirname "$0")/.."

OUT_ROOT="$PWD/logs/d2_critical"
mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/logs"

export PYTHONPATH="vendor/CellViT:vendor/NuLite:."
export CELLVIT_DATA_DIR="$PWD/datasets/pannuke"

# Pre-populate summary with cell 1's already-evaluated result.
SUMMARY="$OUT_ROOT/summary.tsv"
if ! grep -q "^mamba_cs4_no_kd\s*1" "$SUMMARY" 2>/dev/null; then
  # Header line (rewrite to be safe)
  echo -e "condition\tfold\tmPQ_plain\tmPQ_tta\tbPQ_plain\tbPQ_tta\tF1_plain\tF1_tta\trun_dir" > "$SUMMARY"
  # Cell 1 numbers from manual re-eval
  cell1_dir="$OUT_ROOT/runs/mamba_cs4_no_kd_fold1_baseline_fastvit_s12_20260518_125138"
  echo -e "mamba_cs4_no_kd\t1\t0.4660\t0.4759\t0.5881\t0.5983\t0.7184\t0.7266\t${cell1_dir}" >> "$SUMMARY"
fi

# Remaining 7 cells.
# Each entry: name | extra_overrides | fold
CELLS=(
  "mamba_cs4_no_kd | student.decoder_type=mamba student.mamba_scan_pattern=cross_scan_4way student.mamba_d_state=64 training.distillation.enabled=false | 2"
  "mamba_cs4_kl    | student.decoder_type=mamba student.mamba_scan_pattern=cross_scan_4way student.mamba_d_state=64 training.distillation.enabled=true training.distillation.loss_type=kl_div training.distillation.alpha=0.05 training.distillation.temperature=10.0 | 1"
  "mamba_cs4_kl    | student.decoder_type=mamba student.mamba_scan_pattern=cross_scan_4way student.mamba_d_state=64 training.distillation.enabled=true training.distillation.loss_type=kl_div training.distillation.alpha=0.05 training.distillation.temperature=10.0 | 2"
  "mamba_cs4_dkd   | student.decoder_type=mamba student.mamba_scan_pattern=cross_scan_4way student.mamba_d_state=64 training.distillation.enabled=true training.distillation.loss_type=dkd training.distillation.alpha=0.05 training.distillation.temperature=10.0 training.distillation.dkd_alpha=1.0 training.distillation.dkd_beta=8.0 | 1"
  "mamba_cs4_dkd   | student.decoder_type=mamba student.mamba_scan_pattern=cross_scan_4way student.mamba_d_state=64 training.distillation.enabled=true training.distillation.loss_type=dkd training.distillation.alpha=0.05 training.distillation.temperature=10.0 training.distillation.dkd_alpha=1.0 training.distillation.dkd_beta=8.0 | 2"
  "mamba_default   | student.decoder_type=mamba student.mamba_scan_pattern=bidirectional student.mamba_d_state=16 training.distillation.enabled=false | 1"
  "mamba_default   | student.decoder_type=mamba student.mamba_scan_pattern=bidirectional student.mamba_d_state=16 training.distillation.enabled=false | 2"
)

declare -A FOLD_TRAIN=( [1]="[2, 3]" [2]="[1, 3]" )

for entry in "${CELLS[@]}"; do
  name=$(echo "$entry" | cut -d'|' -f1 | xargs)
  overrides=$(echo "$entry" | cut -d'|' -f2 | sed 's/^ *//;s/ *$//')
  fold=$(echo "$entry" | cut -d'|' -f3 | xargs)
  val_fold=$fold
  train_folds=${FOLD_TRAIN[$fold]}
  cell="${name}_fold${fold}"
  log="$OUT_ROOT/logs/${cell}.log"
  echo "==============================================="
  echo "Cell: $cell  (val=fold${val_fold}, train=$train_folds)"
  echo "==============================================="

  .venv/bin/python -m cellvit_distill.scripts.train \
    --config cellvit_distill/configs/fastvit_nulite_v2.yaml \
    --override \
      data.data_dir=$PWD/datasets/pannuke \
      data.soft_targets_dir=$PWD/datasets/pannuke/soft_targets \
      data.num_workers=16 \
      data.n_workers_post=24 \
      data.val_fold=$val_fold \
      data.train_folds="$train_folds" \
      student.mamba_version=v1 \
      student.mamba_groups=4 \
      training.amp_dtype=bf16 \
      training.batch_size=64 \
      training.val_batch_size=16 \
      training.epochs=60 \
      training.warmup_epochs=5 \
      training.val_every_n_epochs=2 \
      training.early_stop_patience=12 \
      training.class_balanced_sampling=true \
      logging.wandb=false \
      logging.save_checkpoint_every=999 \
      logging.output_dir=$OUT_ROOT/runs \
      $overrides \
    > "$log" 2>&1 \
    && echo "TRAIN OK" >> "$log" \
    || { echo "TRAIN FAILED $cell"; continue; }

  run_dir=$(ls -dt $OUT_ROOT/runs/*fastvit_s12_* 2>/dev/null | grep -v "mamba_cs4_no_kd_fold1" | grep -v "_fold[12]_" | head -1)
  if [ -z "$run_dir" ] || [ ! -f "$run_dir/best_model.pth" ]; then
    echo "no run_dir or checkpoint for $cell"; continue
  fi
  mv "$run_dir" "$OUT_ROOT/runs/${cell}_$(basename $run_dir)"
  run_dir="$OUT_ROOT/runs/${cell}_$(basename $run_dir)"

  .venv/bin/python -m cellvit_distill.scripts.eval_student \
    --run_dir "$run_dir" --test_fold $val_fold --batch_size 16 --n_workers_post 16 \
    >> "$log" 2>&1 || echo "EVAL plain FAILED $cell"
  .venv/bin/python -m cellvit_distill.scripts.eval_student \
    --run_dir "$run_dir" --test_fold $val_fold --batch_size 16 --n_workers_post 16 --tta \
    >> "$log" 2>&1 || echo "EVAL TTA FAILED $cell"

  mpq_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}.json')); print(f\"{d['mPQ']:.4f}\")" 2>/dev/null || echo "NA")
  mpq_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}_tta.json')); print(f\"{d['mPQ']:.4f}\")" 2>/dev/null || echo "NA")
  bpq_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  bpq_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}_tta.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_p=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")
  f1_t=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold${val_fold}_tta.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")

  echo -e "${name}\t${fold}\t${mpq_p}\t${mpq_t}\t${bpq_p}\t${bpq_t}\t${f1_p}\t${f1_t}\t${run_dir}" >> "$SUMMARY"
  echo "  -> mPQ_plain=${mpq_p} mPQ_tta=${mpq_t}"
done

echo
echo "=============== D2 CRITICAL RESUME DONE ==============="
column -t -s $'\t' "$SUMMARY"
