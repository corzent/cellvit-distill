# cellvit-distill

Knowledge distillation from CellViT-SAM-H into a lightweight ConvNeXt-Tiny student for cell nuclei segmentation and classification on PanNuke.

Part of a bachelor thesis on compressing heavy pathology foundation models for deployment on consumer GPUs (16 GB VRAM).

## Overview

- **Teacher:** CellViT-SAM-H (630M params, ~24 GB VRAM at inference)
- **Student:** ConvNeXt-Tiny encoder + HoVer-Net style decoder with 3 heads (binary, HV, type) — ~32M params, ~2 GB VRAM
- **Dataset:** PanNuke (7901 patches, 5 cell classes, 19 tissue types)
- **Distillation:** response-based, KL divergence on binary + type heads with temperature-scaled logits. HV head trained from GT only.

## Layout

```
cellvit_distill/
├── configs/default.yaml          # All hyperparameters
├── data/pannuke.py               # Dataset + augmentations (spatial-aligned soft targets)
├── models/student.py             # ConvNeXt-Tiny + HoVer-Net decoder
├── utils/
│   ├── losses.py                 # GT loss + KL distillation
│   ├── metrics.py                # PQ, mPQ, bPQ, F1-detection
│   └── postprocess.py            # HV-watershed instance extraction
└── scripts/
    ├── precompute_soft_targets.py  # Run teacher once, cache logits to disk
    ├── train.py                    # Train student (baseline or distill)
    ├── eval_student.py             # Evaluate a checkpoint on a fold
    └── eval_cellvit256.py          # Reference: evaluate CellViT-256 for comparison
```

## Setup

```bash
# Python 3.11+, CUDA GPU with 8+ GB VRAM (tested on RTX 5060 Ti 16GB)
uv sync

# Vendored CellViT needed for the teacher model
export PYTHONPATH="vendor/CellViT:$PYTHONPATH"
```

Expected data layout under `datasets/pannuke/`:

```
fold1/images.npy   # (N, 256, 256, 3) uint8
fold1/masks.npy    # (N, 256, 256, 6) instance IDs per class
fold1/types.npy    # (N,) tissue type labels
fold2/...
fold3/...
```

Teacher checkpoint: `checkpoints/CellViT-SAM-H-x40.pth`.

## Running experiments

```bash
# 1. One-time: cache teacher soft targets to disk (~6-8 GB VRAM in fp16)
python -m cellvit_distill.scripts.precompute_soft_targets

# 2. Baseline — student trained on GT only
python -m cellvit_distill.scripts.train --config cellvit_distill/configs/default.yaml

# 3. Distillation — student trained on GT + teacher logits
python -m cellvit_distill.scripts.train \
    --config cellvit_distill/configs/default.yaml \
    --override training.distillation.enabled=true

# Or both sequentially
bash run_experiments.sh
```

Checkpoints and logs land in `cellvit_distill/runs/<experiment>_<encoder>_<timestamp>/`.

## Evaluation

```bash
python -m cellvit_distill.scripts.eval_student \
    --run_dir cellvit_distill/runs/<run_name> \
    --test_fold 3
```

Reports bPQ, mPQ (with per-class breakdown), and F1-detection.

## Key implementation notes

- **Spatial alignment of soft targets.** Teacher logits are precomputed on unaugmented images, so they must pass through the same spatial transforms (flips, rotations, elastic) as the student's input. Binary and type channels are included in the Albumentations `masks` pipeline; HV distillation is disabled because distance values require sign correction under reflections, and GT HV maps are already exact.
- **Class imbalance.** Dead nuclei make up ~1.5% of PanNuke annotations. The type head uses focal loss (γ=2.0) with inverse-frequency class weights to compensate.
- **Memory.** Teacher inference requires ≥24 GB VRAM, so soft targets are cached offline and only the 32M-param student is in GPU memory at training time.
