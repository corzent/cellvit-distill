#!/usr/bin/env bash
# Phase C3: Mamba arch sweep on fold 3, 40 epochs each, no KD.
#
# 6 configs vary scan_pattern × d_state × groups. Baseline (bidirectional,
# d_state=16, groups=4) already done in logs/mamba_baseline/, ref mPQ_TTA=0.4650.
# Each config trains then evals (plain + TTA), writes per-image npz for
# downstream stat_test comparisons.
#
# Skips Mamba2 entirely — current decoder snaps per-group channels down to 8
# at the smallest decoder stage, below Mamba2's default headdim=64.

set -e
cd "$(dirname "$0")/.."

OUT_ROOT="$PWD/logs/c3_sweep"
mkdir -p "$OUT_ROOT/runs" "$OUT_ROOT/logs"

# Each config: name | scan_pattern | d_state | groups
CONFIGS=(
  "bi_d32_g4    | bidirectional   | 32 | 4"
  "bi_d64_g4    | bidirectional   | 64 | 4"
  "cs4_d16_g4   | cross_scan_4way | 16 | 4"
  "cs4_d32_g4   | cross_scan_4way | 32 | 4"
  "cs4_d64_g4   | cross_scan_4way | 64 | 4"
  "bi_d32_g2    | bidirectional   | 32 | 2"
)

export PYTHONPATH="vendor/CellViT:vendor/NuLite:."
export CELLVIT_DATA_DIR="$PWD/datasets/pannuke"

SUMMARY="$OUT_ROOT/summary.tsv"
echo -e "config\tscan\td_state\tgroups\tparams_M\tmPQ\tmPQ_tta\tbPQ_tta\tF1_tta\trun_dir" > "$SUMMARY"

for entry in "${CONFIGS[@]}"; do
  name=$(echo "$entry" | cut -d'|' -f1 | xargs)
  scan=$(echo "$entry" | cut -d'|' -f2 | xargs)
  dstate=$(echo "$entry" | cut -d'|' -f3 | xargs)
  groups=$(echo "$entry" | cut -d'|' -f4 | xargs)
  log="$OUT_ROOT/logs/${name}.log"

  echo "==============================================="
  echo "Config: $name  scan=$scan d_state=$dstate groups=$groups"
  echo "==============================================="

  .venv/bin/python -m cellvit_distill.scripts.train \
    --config cellvit_distill/configs/fastvit_nulite_v2.yaml \
    --override \
      data.data_dir=$PWD/datasets/pannuke \
      data.num_workers=16 \
      data.n_workers_post=24 \
      student.decoder_type=mamba \
      student.mamba_scan_pattern=$scan \
      student.mamba_d_state=$dstate \
      student.mamba_groups=$groups \
      student.mamba_version=v1 \
      training.amp_dtype=bf16 \
      training.batch_size=64 \
      training.val_batch_size=16 \
      training.epochs=40 \
      training.warmup_epochs=5 \
      training.val_every_n_epochs=2 \
      training.early_stop_patience=12 \
      training.class_balanced_sampling=true \
      training.distillation.enabled=false \
      logging.wandb=false \
      logging.save_checkpoint_every=999 \
      logging.output_dir=$OUT_ROOT/runs \
    > "$log" 2>&1 \
    && echo "TRAIN OK" >> "$log" \
    || { echo "TRAIN FAILED for $name"; continue; }

  # Find newest run dir (the one we just created)
  run_dir=$(ls -dt $OUT_ROOT/runs/baseline_fastvit_s12_* 2>/dev/null | head -1)
  if [ -z "$run_dir" ]; then
    echo "no run_dir for $name"; continue
  fi
  # Rename to include the sweep name so it's easy to find later
  mv "$run_dir" "$OUT_ROOT/runs/${name}_$(basename $run_dir)"
  run_dir="$OUT_ROOT/runs/${name}_$(basename $run_dir)"

  # Parse params from log
  params_M=$(grep -E "^Total: " "$log" | head -1 | awk '{print $2}' | tr -d 'M')

  # Eval plain
  .venv/bin/python -m cellvit_distill.scripts.eval_student \
    --run_dir "$run_dir" --test_fold 3 --batch_size 16 --n_workers_post 16 \
    >> "$log" 2>&1 || echo "EVAL plain FAILED for $name"

  # Eval TTA
  .venv/bin/python -m cellvit_distill.scripts.eval_student \
    --run_dir "$run_dir" --test_fold 3 --batch_size 16 --n_workers_post 16 --tta \
    >> "$log" 2>&1 || echo "EVAL TTA FAILED for $name"

  # Extract numbers
  mpq=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold3.json')); print(f\"{d['mPQ']:.4f}\")" 2>/dev/null || echo "NA")
  mpq_tta=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold3_tta.json')); print(f\"{d['mPQ']:.4f}\")" 2>/dev/null || echo "NA")
  bpq_tta=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold3_tta.json')); print(f\"{d['bPQ']:.4f}\")" 2>/dev/null || echo "NA")
  f1_tta=$(.venv/bin/python -c "import json; d=json.load(open('$run_dir/eval_fold3_tta.json')); print(f\"{d['F1_detection']:.4f}\")" 2>/dev/null || echo "NA")

  echo -e "${name}\t${scan}\t${dstate}\t${groups}\t${params_M}\t${mpq}\t${mpq_tta}\t${bpq_tta}\t${f1_tta}\t${run_dir}" >> "$SUMMARY"
  echo "  -> mPQ=${mpq} mPQ_TTA=${mpq_tta}  saved to $SUMMARY"
done

echo
echo "=============== SWEEP DONE ==============="
column -t -s $'\t' "$SUMMARY"
