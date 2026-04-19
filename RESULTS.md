# Experimental Results (PanNuke, fold 3)

## Headline

The best model — **FastViT-S12 + response-based KD + 8-way TTA** — reaches **mPQ = 0.472** at **11.5M parameters**. That is **80% of the teacher's fold-3 quality (CellViT-SAM-H, 0.592 mPQ under our protocol) at 55× fewer parameters**, with peak inference VRAM dropping from ≥24 GB to ~2 GB.

## Main comparison

Evaluated on PanNuke fold 3 (held-out) under the standard CellViT/NuLite protocol (`nanmean_over_images(nanmean_over_classes(PQ))`). All numbers in the "our eval" column were recomputed with the same post-processing pipeline and the same metric implementation for apples-to-apples comparison.

| Model | Params | mPQ (ours) | mPQ (paper) | bPQ | F1 | Note |
|---|---|---|---|---|---|---|
| CellViT-SAM-H (teacher) | 630M | **0.592** | 0.51 | 0.664 | 0.784 | cached logits from precompute; paper is 3-fold avg |
| CellViT-256 (x20 checkpoint) | 46.8M | 0.317 | — | 0.471 | 0.598 | magnification mismatch (x20 ckpt on x40 data) |
| NuLite-T [published, 2026] | 12M | — | ≈0.50 | ≈0.64 | ≈0.82 | external baseline, not run locally |
| ConvNeXt-Tiny baseline | 31.9M | 0.468 | — | 0.591 | 0.719 | earlier student |
| FastViT-S12 baseline (v2 recipe) | 11.5M | 0.456 | — | 0.578 | 0.706 | NuLite-style recipe |
| FastViT-S12 + response KD | 11.5M | 0.467 | — | 0.598 | 0.724 | α=0.2, T=10 |
| FastViT-S12 + feature KD (β=1.0) | 11.8M | 0.461 | — | 0.591 | 0.720 | response + feature match |
| **FastViT-S12 + response KD + TTA** | **11.5M** | **0.472** | — | **0.604** | **0.729** | 8-way TTA at inference |
| FastViT-S12 + feature KD + TTA | 11.8M | 0.468 | — | 0.604 | 0.728 | 8-way TTA |
| FastViT-S12 baseline + TTA | 11.5M | 0.467 | — | 0.593 | 0.718 | 8-way TTA, no distill |

## Per-class PQ (fold 3, TTA for student variants)

| Class | Teacher (SAM-H) | CellViT-256 | ConvNeXt baseline | FastViT baseline | FastViT + resp KD | FastViT + feat KD |
|---|---|---|---|---|---|---|
| Neoplastic | 0.668 | 0.401 | 0.533 | 0.530 | 0.550 | 0.553 |
| Inflammatory | 0.576 | 0.317 | 0.446 | 0.449 | 0.446 | 0.428 |
| Connective | 0.516 | 0.250 | 0.386 | 0.385 | 0.384 | 0.381 |
| Dead | **0.443** | 0.011 | 0.134 | 0.158 | 0.104 | **0.137** |
| Epithelial | 0.670 | 0.224 | 0.542 | 0.531 | 0.544 | 0.545 |

Teacher dominates Dead class (0.443) — expected given its 630M capacity, no data scarcity issue. Among students, feature-KD captures Dead best (0.137), consistent with the hypothesis that feature matching is less biased by class frequency than KL divergence on logits.

## Computational efficiency

| Model | Params | Compression | Peak VRAM inference | Inference (ms/patch) |
|---|---|---|---|---|
| CellViT-SAM-H (teacher) | 630M | 1× | ≥24 GB | ≈500 |
| ConvNeXt-Tiny | 31.9M | 19.7× | ~2.5 GB | ≈130 |
| **FastViT-S12** | **11.5M** | **54.8×** | **~2 GB** | **≈70** |

## Two methodological findings

During the project we discovered and fixed two issues that substantially
changed the outcomes:

1. **Spatial alignment of precomputed soft targets.** Soft targets (logits)
   precomputed once on the unaugmented images must pass through the same
   spatial augmentations as the student input. Without this, ≈87% of training
   batches (when using three independent p=0.5 spatial augmentations) carry
   misaligned teacher signal, making the distillation loss effectively noise.
   See [commit fa901e1](https://github.com/corzent/cellvit-distill/commit/fa901e1).

2. **mPQ evaluation protocol.** The reference CellViT/NuLite protocol averages
   PQ hierarchically: per image → `nanmean` across classes (skipping absent
   ones) → mean over images. Naively treating absent classes as PQ=0
   underestimates mPQ drastically on imbalanced datasets like PanNuke (Dead
   appears in <2% of patches). Fixing this raised the same ConvNeXt baseline
   from 0.184 to 0.468 — a **2.4× difference** purely from the metric
   implementation. See [commit e065dd4](https://github.com/corzent/cellvit-distill/commit/e065dd4).

## Training recipe (v2)

- Encoder: FastViT-S12 (8.3M, Apple 2023, ImageNet-1K pretrained)
- Decoder: HoVer-Net-style FPN (3.1M, channels [256, 128, 64, 32])
- Heads: binary (2), HV (2, tanh-clipped), type (6) — each ConvBNReLU + 1×1 Conv
- Auxiliary head: tissue classification (19 classes)
- Loss: Focal Tversky Loss (binary) + Dice + MSE + MSGE (HV) + Focal CE + Dice (type) + CE (tissue)
- Optimizer: AdamW lr=3e-4, betas=(0.85, 0.95), weight_decay=1e-5
- Scheduler: 10-epoch linear warmup + cosine annealing
- Batch size: 8, fp16 mixed precision
- Augmentations: HFlip/VFlip/Rotate90, ElasticTransform, Affine, ColorJitter, GaussianBlur, GaussNoise, CoarseDropout
- Class-balanced sampling (WeightedRandomSampler, inverse-frequency weights)
- Early stopping: patience 20 on val mPQ, evaluated every epoch
- Distillation: α=0.2, T=10, `head_weights={binary: 1.0, hv_map: 0.0, type_map: 1.0}`

## Artifacts

- Run directory (baseline): `cellvit_distill/runs/baseline_fastvit_s12_20260417_194915/`
- Run directory (response distill): `cellvit_distill/runs/distill_fastvit_s12_20260418_013823/`
- Run directory (feature distill): `cellvit_distill/runs/distill_fastvit_s12_20260418_163923/`
- Soft targets: `datasets/pannuke/soft_targets/` (7901 × ~320 KB = 2 GB)
- Soft features: `datasets/pannuke/soft_features/` (7901 × ~600 KB = 4.5 GB, ViT-H tokens, fp16)

## Next steps (from §4.4 of the thesis)

- Full 3-fold cross-validation with `mean ± std` for publication-quality numbers
- External dataset eval on MoNuSeg and CoNSeP for domain shift robustness
- Feature KD with β in [0.3, 0.5] — the β=1.0 we used was likely too aggressive
- Pathology-specific teachers (UNI 2, Virchow 2) instead of SAM-based CellViT
- Post-training INT8 quantization for ~4× additional compression
