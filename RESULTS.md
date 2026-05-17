# Experimental Results (PanNuke, 3-fold CV)

## Headline

The best model — **FastViT-S12 + response-based KD + 8-way TTA** — reaches
**mPQ = 0.4698 ± 0.0023** on PanNuke 3-fold cross-validation at **11.5M
parameters**. That is **79% of the teacher's fold-3 quality (CellViT-SAM-H,
0.592 mPQ under our protocol) at 55× fewer parameters**, with peak inference
VRAM dropping from ≥24 GB to ~2 GB.

Distillation is consistent across all 3 folds: per-fold KD − baseline mPQ
deltas are +0.0055, +0.0050, +0.0027 — small but systematically positive,
with tight cross-fold variance (std ±0.0023–0.0024) supporting that the
improvement is real rather than fold-noise.

## Main comparison

Evaluated on PanNuke under the standard CellViT/NuLite protocol
(`nanmean_over_images(nanmean_over_classes(PQ))`). All "ours" numbers are
3-fold mean ± std (sample), 8-way TTA. Reference baselines (teacher,
CellViT-256, NuLite, LKCell, KongNet, HoVer-NeXt) are from their papers
unless re-evaluated locally — `eval (ours)` column indicates which.

| Model | Params | mPQ | bPQ | F1 | Note |
|---|---|---|---|---|---|
| CellViT-SAM-H (teacher) | 630M | 0.592 (our eval, fold 3) | 0.664 | 0.784 | cached logits from precompute; paper reports 0.51 (3-fold avg) |
| CellViT-256 (x20 ckpt) | 46.8M | 0.317 (our eval, fold 3) | 0.471 | 0.598 | magnification mismatch (x20 ckpt on x40 data) — not a fair comparison |
| NuLite-T [Tommasino 2024] | 12M | ≈0.47–0.49 (paper) | ≈0.66 (paper) | ≈0.83 (paper) | concurrent work, same encoder, no KD |
| NuLite-H [Tommasino 2024] | 34M | 0.496 (paper) | 0.677 (paper) | 0.83 (paper) | bigger NuLite variant |
| LKCell-L [arXiv 2407.18054] | 163M | 0.508 (paper) | 0.685 (paper) | — | larger model with UniRepLKNet backbone |
| KongNet-Det [arXiv 2510.23559] | EfficientNetV2-L | F1 0.674 (paper) | — | 0.674 | per-class decoders, no watershed |
| HoVer-NeXt [Baumann 2024] | (n/a) | 0.477 mPQ_tiss (paper) | — | — | 2-decoder fast variant |
| **FastViT-S12 baseline** | **11.5M** | **0.4655 ± 0.0024** | 0.5900 ± 0.0062 | 0.7162 ± 0.0054 | NuLite-style recipe, no KD |
| **FastViT-S12 + response KD** | **11.5M** | **0.4698 ± 0.0023** | **0.5988 ± 0.0062** | **0.7251 ± 0.0072** | α=0.05, T=10, 8-way TTA |

Per-fold breakdown (TTA):

| Fold | baseline | + response KD | Δ |
|---|---|---|---|
| 1 | 0.4668 | 0.4723 | +0.0055 |
| 2 | 0.4627 | 0.4677 | +0.0050 |
| 3 | 0.4668 | 0.4695 | +0.0027 |

Without TTA: baseline 0.4534 ± 0.0046, distill 0.4601 ± 0.0030 (+0.0067).
TTA contributes +0.012 to baseline and +0.010 to distill — large enough to
matter, small enough to not change the KD direction.

## Per-class PQ (3-fold mean ± std, TTA)

| Class | FastViT baseline | FastViT + response KD | Δ |
|---|---|---|---|
| Neoplastic | 0.5286 ± 0.0150 | 0.5352 ± 0.0163 | +0.0066 |
| Inflammatory | 0.4440 ± 0.0164 | 0.4470 ± 0.0162 | +0.0030 |
| Connective | 0.3887 ± 0.0040 | 0.3900 ± 0.0077 | +0.0013 |
| **Dead** | 0.1680 ± 0.0436 | 0.1635 ± 0.0362 | −0.0045 |
| Epithelial | 0.5215 ± 0.0054 | 0.5373 ± 0.0150 | +0.0158 |

KD helps Neoplastic and Epithelial most (well-represented classes where
teacher is confident). Dead is approximately neutral and remains the weak
point — consistent with the response-KD literature: KL divergence on
classification logits transfers little information about rare classes
because the teacher's softmax there is dominated by the majority class
mode. **Future work:** feature-based KD (FitNet / FRSKD style), focal-KD,
or decoupled-KD (DKD, Zhao et al. CVPR 2022) which separates the
target-class vs non-target-class KL components.

## Computational efficiency

| Model | Params | Compression | Peak VRAM inference | Inference (ms/patch) |
|---|---|---|---|---|
| CellViT-SAM-H (teacher) | 630M | 1× | ≥24 GB | ≈500 |
| ConvNeXt-Tiny | 31.9M | 19.7× | ~2.5 GB | ≈130 |
| **FastViT-S12** | **11.5M** | **54.8×** | **~2 GB** | **≈70** |

Numbers measured on RTX 5090. Latency for the student is dominated by the
HoVer-watershed post-processing (CPU-bound), not the GPU forward pass.

## Methodological findings

Three issues discovered and fixed during the project that substantially
changed the reported numbers:

1. **Spatial alignment of precomputed soft targets.** Soft targets
   (logits) precomputed once on the unaugmented images must pass through
   the same spatial augmentations as the student input. Without this,
   ≈87% of training batches (when using three independent p=0.5 spatial
   augmentations) carry misaligned teacher signal, making the
   distillation loss effectively noise.
   See [commit fa901e1](https://github.com/corzent/cellvit-distill/commit/fa901e1).

2. **mPQ evaluation protocol.** The reference CellViT/NuLite protocol
   averages PQ hierarchically: per image → `nanmean` across classes
   (skipping absent ones) → mean over images. Naively treating absent
   classes as PQ=0 underestimates mPQ drastically on imbalanced datasets
   like PanNuke (Dead appears in <2% of patches). Fixing this raised the
   same ConvNeXt baseline from 0.184 to 0.468 — a **2.4× difference**
   purely from the metric implementation.
   See [commit e065dd4](https://github.com/corzent/cellvit-distill/commit/e065dd4).

3. **KD α hyperparameter is environment-sensitive.** The original v1
   pilot (single fold 3, PyTorch 2.5 era) used α=0.2 successfully. On
   the rented RTX 5090 with PyTorch 2.12 + CUDA 13 + batch_size=16 (vs
   pilot batch=8), the same α=0.2 destabilized distillation:
   distill mPQ dropped to ~0.29 while baseline reached ~0.46. Bisection
   confirmed α is the culprit (not batch alone, not parallel
   post-processing pool, not data quality); α=0.05 restored stable
   training. The KD gradient is more coherent with larger batches /
   different numerical paths, so α needs to scale down proportionally
   when changing those.
   See `EXPERIMENT_LOG.md` 2026-05-17 entry for full bisection.

## Training recipe (final, 3-fold CV)

- Encoder: FastViT-S12 (8.3M, Apple 2023, ImageNet-1K pretrained)
- Decoder: HoVer-Net-style FPN (3.1M, channels [256, 128, 64, 32])
- Heads: binary (2), HV (2, tanh-clipped), type (6) — each ConvBNReLU + 1×1 Conv
- Auxiliary head: tissue classification (19 PanNuke types)
- Loss: Focal Tversky (binary) + Dice + MSE + MSGE (HV) + Focal CE + Dice (type) + CE (tissue)
- Optimizer: AdamW lr=3e-4, betas=(0.85, 0.95), weight_decay=1e-5
- Scheduler: 10-epoch linear warmup + cosine annealing (max 130 epochs)
- Batch size: 16 (val 8), fp16 mixed precision
- Augmentations: HFlip/VFlip/Rotate90, ElasticTransform, Affine, ColorJitter, GaussianBlur, CoarseDropout
- Class-balanced sampling (WeightedRandomSampler, inverse-frequency weights)
- Early stopping: patience 20 on val mPQ, validated every 5 epochs
- **Distillation: α=0.05, T=10**, `head_weights={binary: 1.0, hv_map: 0.0, type_map: 1.0}`

## Artifacts (3-fold CV, 2026-05-17)

- 6 run directories: `cellvit_distill/runs/{baseline,distill}_fastvit_s12_20260517_*/`
  (best_model.pth + checkpoints + config.yaml + eval_fold{N}{,_tta}.json each)
- Manifest: `logs/3fold_cv/runs.manifest` (6 lines: condition, fold, run_dir)
- Aggregated summary: `logs/3fold_cv/summary.md`
- Per-run training logs: `logs/3fold_cv/{baseline,distill_resp}_fold{1,2,3}.log`
- Soft targets: `datasets/pannuke/soft_targets/` (7901 × ~1.2 MB = 9.3 GB, fp32 raw logits)
- Soft features (for future feature-KD work): not precomputed in this CV

## Next steps

- ~~Full 3-fold cross-validation with `mean ± std`~~ — **done** (this report)
- External dataset eval on MoNuSeg and CoNSeP for domain-shift robustness
- Feature KD with β in [0.3, 0.5] — original β=1.0 was likely too aggressive
- Decoupled-KD (DKD) or focal-KD targeting Dead-class transfer
- Architecture ablation: separate type-head mini-decoder, ASPP block before
  heads, larger final decoder stage (current 32 ch is a likely bottleneck
  for rare classes)
- Pathology-specific teachers (UNI 2, Virchow 2) instead of SAM-based CellViT
- Post-training INT8 quantization for ~4× additional compression
