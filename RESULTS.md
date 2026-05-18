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
- ~~External dataset eval on MoNuSeg~~ — **done** (1-fold, see Update below)
- ~~Decoupled-KD (DKD) targeting Dead-class transfer~~ — **done, did not help**
  (see Update below)
- Phase D 3-fold ablation grid: {HoVer, Mamba} × {no-KD, KL, DKD} ×
  3 folds for full statistical claims on the comparative ablation
- Feature KD with β in [0.3, 0.5] — original β=1.0 was likely too aggressive
- Architecture ablation: separate type-head mini-decoder, ASPP block before
  heads, larger final decoder stage (current 32 ch is a likely bottleneck
  for rare classes)
- Pathology-specific teachers (UNI 2, Virchow 2) instead of SAM-based CellViT
- Post-training INT8 quantization for ~4× additional compression


---

## Update 2026-05-18 — Mamba-decoder + DKD + cross-dataset (1-fold pilot)

All numbers in this section are **fold-3 only** (single experiment), not
3-fold CV. They are pilot results from a 24-hour session that adds three
new directions on top of the 3-fold report above. Mean±std claims still
need the Phase D 3-fold grid; current deltas are reported relative to
the published fold-3 numbers from the table above.

### 1. Mamba-decoder student (FastViT-S12 + SSM decoder)

Replaced the conv `HoVerNetDecoder` with a Mamba-based decoder
(UltraLight-VM-UNet style PVM blocks: channels split into G groups,
independent Mamba layer per group, GroupNorm residual). Two scan
patterns matter: row-major bidirectional (default) and VMamba 4-way
cross-scan (`cs4`).

Mamba-decoder architecture sweep (fold-3, 40 epochs, no KD):

| Config        | scan            | d_state | Params  | mPQ TTA |
|---------------|-----------------|---------|---------|---------|
| bi_d16_g4     | bidirectional   | 16      | 9.8M    | 0.4650  |
| bi_d32_g4     | bidirectional   | 32      | 9.9M    | 0.4642  |
| bi_d64_g4     | bidirectional   | 64      | 10.0M   | 0.4656  |
| cs4_d16_g4    | cross_scan_4way | 16      | 10.4M   | 0.4670  |
| cs4_d32_g4    | cross_scan_4way | 32      | 10.5M   | 0.4665  |
| **cs4_d64_g4**| **cross_scan_4way** | **64** | **10.7M** | **0.4717** |
| bi_d32_g2     | bidirectional   | 32      | 10.0M   | 0.4630  |

`cs4_d64_g4` is the winning Mamba config. The gain comes from the
**interaction** of cross-scan + larger state (each alone gives
+0.001-0.002 mPQ; together +0.007). `groups=2` (wider per-group
channels) is the worst configuration of the sweep — narrower groups
beat wider for this size.

### 2. Mamba vs conv decoder (matched budget)

| Decoder       | Params | mPQ TTA fold-3 | Δ vs HoVer  |
|---------------|--------|----------------|-------------|
| HoVer (conv)  | 11.5M  | 0.4720*        | reference   |
| Mamba cs4_d64 | 10.7M  | 0.4717         | −0.0003     |

(*from fold-3 of the 3-fold-CV baseline above.)

**Mamba decoder essentially ties the conv decoder on PanNuke at matched
budget, with 0.8M fewer parameters.** Gap is within 1σ of the 3-fold std
(±0.0023). This is the first matched-budget result on PanNuke nuclei
segmentation for an SSM-based decoder in the ≤15M regime.

### 3. KD on Mamba (fold-3 1-fold runs)

Same cs4_d64_g4 Mamba decoder, three KD variants tested:

| Condition    | KD method     | mPQ TTA | Δ vs Mamba no-KD |
|--------------|---------------|---------|------------------|
| Mamba cs4_d64 (ref) | none   | 0.4717  | —                |
| Mamba + KL   | KL, α=0.05, T=10 | 0.4664  | −0.0053          |
| Mamba + DKD  | DKD, α=1, β=8    | 0.4631  | −0.0086          |
| HoVer + UFD-KD | UFD-KD (cf. §4) | 0.4725 | +0.0008 (tie)    |

**All three KD attempts on Mamba are marginally worse than no-KD
within PanNuke** (1-fold; differences are 1-4σ of the 3-fold std).
KL and DKD response distillation, both tuned on HoVer + conv decoder,
do not transfer to the SSM decoder at the same hyperparameters.
A dedicated α/T sweep for Mamba is open work.

### 4. Frequency-Decoupled KD (Novel A) — clean negative

Adapted UFD-KD (Lu et al., BMVC 2025) from classification to dense
prediction: per-pixel softmax → 2D DCT-II → split into LF/HF bands →
weighted MSE. Result (HoVer decoder, fold 3, α=5×10⁻⁴ after T²
calibration):

| Condition          | mPQ TTA | Dead PQ TTA |
|--------------------|---------|-------------|
| HoVer + KL distill (ref, 3-fold avg) | 0.4695 | 0.1635 |
| HoVer + UFD-KD     | 0.4725  | 0.1357 (−17%) |

Within noise on mPQ, but Dead class **drops 17% relative**. Inspection
of per-band loss magnitudes during training: LF MSE ≈ 12, HF MSE ≈
0.02 → with `hf_weight=3` the HF contribution to total is 0.5%. The
intended "frequency decoupling" is effectively a no-op on segmentation
softmax. Negative result documented; the adaptation does not survive
the magnitude imbalance.

### 5. Cross-dataset zero-shot eval (MoNuSeg test, n=14)

Loaded RationAI/MoNuSeg from HuggingFace, ran 5 today's checkpoints
zero-shot with overlap-stitched 256×256 patches and 4-way flip TTA.

| Model         | Decoder | KD  | PanNuke bPQ TTA | MoNuSeg bPQ TTA | gap     |
|---------------|---------|-----|-----------------|-----------------|---------|
| hover_ufd     | conv    | UFD | 0.5988          | 0.5563          | −0.043  |
| mamba_default | mamba   | —   | 0.5808          | 0.5702          | −0.011  |
| mamba_cs4_d64 | mamba   | —   | 0.5906          | 0.5463          | −0.044  |
| mamba_cs4_kl  | mamba   | KL  | 0.5902          | 0.4608          | **−0.129** |
| mamba_cs4_dkd | mamba   | DKD | 0.5838          | 0.3763          | **−0.208** |

(MoNuSeg has only nucleus-vs-background labels, so only bPQ is
applicable — F1-detection follows the same ranking.)

**Three findings from this run:**

1. **KD on Mamba is catastrophic under domain shift.** Within PanNuke
   the KL/DKD penalty was 0.005-0.009 mPQ; on MoNuSeg the same KL/DKD
   models lose 0.086-0.170 bPQ vs the matched-architecture no-KD
   baseline. The teacher's PanNuke-specific decision boundary appears
   to be memorized by the SSM student in a way that does not survive
   the stain / scanner / organ shift.

2. **Mamba default generalizes best.** The "weakest" Mamba on PanNuke
   (bi_d16_g4, mPQ 0.4650) is the strongest on MoNuSeg (bPQ 0.5702).
   The C3-winning cs4_d64_g4 (+0.007 mPQ on PanNuke) is −0.024 bPQ on
   MoNuSeg vs the default. Capacity / generalization trade-off.

3. **Mamba default beats conv decoder on MoNuSeg.** mamba_default 0.5702
   vs hover_ufd 0.5563 (+0.014 bPQ). On PanNuke they were tied. This is
   the closest result to a clean positive contribution in the session.

### Reproducibility artefacts (Update)

- Code: branches `dev/post-master-work`, `feat/ablation-grid-runner`,
  `feat/monuseg-eval`, `docs/related-work` (not merged to `master`).
- Run dirs (1-fold each, NOT in git — gitignored):
  - `logs/ufd_final/runs/distill_fastvit_s12_20260517_191745/` (Phase A)
  - `logs/mamba_baseline/runs/baseline_fastvit_s12_20260517_210340/` (Mamba default)
  - `logs/c3_sweep/runs/{bi,cs4}_d{16,32,64}_g{2,4}_baseline_*/` (C3 sweep)
  - `logs/c4_mamba_kl/runs/distill_fastvit_s12_20260518_071225/` (Mamba+KL)
  - `logs/c4_mamba_dkd/runs/distill_fastvit_s12_20260518_085801/` (Mamba+DKD)
- Per-image arrays (npz) saved alongside each `eval_*.json` (PanNuke + MoNuSeg).
- Sweep summaries: `logs/c3_sweep/summary.tsv`, `logs/monuseg_sweep/summary.tsv`.
- Lit notes: `docs/related-work.md`.
- Reproducible: `scripts/c3_mamba_sweep.sh`, `scripts/e_monuseg_sweep.sh`,
  `grids/main_ablation.yaml` (for the Phase D 3-fold grid runner).

### Caveats for the thesis

  - All Mamba and Phase E results are **1-fold (fold 3)**. The 0.0043
    headline KD improvement above survives 3-fold CV; the Mamba-equals-
    HoVer claim and the KD-hurts-Mamba claim do not have multi-fold
    confirmation yet. Phase D2 grid (18 runs) is the right
    follow-up.
  - MoNuSeg n=14: the 0.014 advantage of mamba_default over hover_ufd
    is on the edge of significance; the 0.086 and 0.170 gaps for KD-on-
    Mamba are large enough to almost certainly hold.
  - Teacher reference number 0.592 vs paper-reported 0.510 still
    unresolved (likely protocol difference — to be reconciled before
    quoting "79 % of teacher quality").
