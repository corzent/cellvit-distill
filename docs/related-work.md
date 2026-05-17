# Related Work

Working notes for the thesis literature review. Each entry: paper, key
relationship to our work, what number we should cite, and what reviewer
question it answers (or raises).

Last updated: 2026-05-17.

## Direct neighbours — nuclei segmentation on PanNuke

### CellViT-SAM-H (Hörst et al., MIA 2024)
- arXiv 2306.15350 · github.com/TIO-IKIM/CellViT
- **Teacher in this work.** ViT-H SAM encoder + HoVer-Net heads, 630M params.
- Reported PanNuke mPQ: 0.51 (3-fold avg, their protocol). Our re-eval on
  fold 3 with the standard hierarchical mPQ: 0.592. The gap is unresolved
  and is exactly what Phase B4 needs to nail down — almost certainly a
  protocol difference (per-image vs flat averaging), but we have to
  document it before claiming "79% of teacher quality".

### NuLite-T / NuLite-H (Tommasino et al., 2024)
- arXiv 2408.01797 · Biomed. Signal Process. Control 2025
- **Closest concurrent work.** Same encoder (FastViT-S12), same param scale
  (12M), same dataset, single decoder with 3 HoVer heads. NO knowledge
  distillation — pure GT training with tissue-aware sampling.
- Numbers: mPQ ≈ 0.47–0.49 (T, 12M), 0.496 (H, 34M).
- **Defense:** "Why is this thesis needed if NuLite already exists?"
  Answer: NuLite establishes the architecture; we add the comparative-KD
  ablation that NuLite does not run. The story is the ablation grid, not
  the architecture.

### CP-Mamba (Zhang et al., AAAI 2025)
- arXiv 2503.10422
- **First Mamba on PanNuke — published.** Encoder-decoder Mamba with
  category-prompt supervision. Reports **mPQ 0.6149 on PanNuke** — above
  our teacher under our protocol.
- Implications for our framing:
  - "First Mamba on PanNuke" is gone.
  - "First KD-into-Mamba-decoder for nuclei seg" is still defensible —
    they use prompt supervision, not distillation.
  - CP-Mamba is cited, not reproduced (per user decision 2026-05-17).
    Use their published number as a published upper bound; honest
    discussion section notes they are heavier than our ≤15M budget.

### HoVer-UNet (Tommasino et al., 2023)
- arXiv 2311.12553
- **Only prior KD-on-nuclei-seg work.** ConvNeXt encoder + HoVer-Net
  decoder + response KD from CellViT-SAM-H.
- Direct **KD-into-conv baseline** for our comparative ablation. Our
  `distill_kl_hover` condition is essentially a re-implementation of
  this — we expect to match their headline numbers.
- Numbers from their paper: mPQ ≈ 0.49 (PanNuke 3-fold).

## Architecture references

### UltraLight VM-UNet (Wu et al., ISBI 2024)
- arXiv 2403.20035 · github.com/wurenkai/UltraLight-VM-UNet
- **Architecture template for our Mamba decoder.** Their PVM block
  (Parallel Vision Mamba, channels split into G groups with independent
  Mamba per group + GroupNorm residual) is what `cellvit_distill/models/
  mamba_decoder.py` adapts. They get 0.049M-param U-Net on skin lesion
  datasets; we scale up groups + channel widths for the dense
  classification task.
- Caveat: their repo pins mamba_ssm==1.0.1 / torch==1.13. Port needed
  for PyTorch 2.12 + CUDA 13 (our stack). Phase C0 work.

### LightM-UNet (arXiv 2403.05246)
- github.com/MrBlankness/LightM-UNet
- Alternative Mamba seg template (1.09M, Residual Vision Mamba Layer).
  Reviewed and not adopted — UltraLight's PVM split is closer to our
  needs.

### Swin-UMamba (arXiv 2402.03302)
- Mamba-decoder variant with reduced compute. Considered but their
  encoder is Swin which doesn't fit our FastViT-fixed setup.

### VMamba (arXiv 2401.10166)
- Source of the 4-way cross-scan (SS2D) pattern we implemented in
  `MambaBlock.scan_pattern="cross_scan_4way"`. ImageNet ablations show
  cross-scan beats bidirectional for classification; the same isn't
  established for biomedical seg, so we ablate.

## KD methods we benchmark

### KL distillation (Hinton et al., NeurIPS 2015 workshop)
- Standard `loss_type=kl_div` in our DistillationLoss. T-scaled softmax
  KL, averaged over spatial positions. Our current published recipe
  uses α=0.05, T=10 (after the bisection documented in RESULTS.md).

### Decoupled KD (Zhao et al., CVPR 2022)
- arXiv 2203.08679 · github.com/megvii-research/mdistiller
- **Implemented (commit 32bb996 on dev/post-master-work).** Splits
  per-pixel KL into TCKD (target-class binary KL) and NCKD
  (renormalized non-target KL). Reweights NCKD independently — paper
  default α=1, β=8.
- Motivation for us: the Dead class on PanNuke is the failure mode of
  vanilla KL (KD currently *hurts* Dead by -0.0045 mPQ). DKD's β=8
  on NCKD specifically amplifies the rare-class portion of the dark
  knowledge.
- For dense prediction we use teacher's argmax as per-pixel target,
  keeping the DistillationLoss signature unchanged.

### Frequency-Decoupled KD (Lu et al., BMVC 2025) — abandoned hypothesis
- arXiv ID TBD (BMVC 2025)
- Adapted classification UFD-KD to dense prediction (commit 298ee6a +
  8f72b7e). Smoke + 1-fold (Phase A in flight 2026-05-17) confirmed:
  the loss runs end-to-end but the LF band dominates the HF band by
  ~600× in magnitude on segmentation softmax, so the intended
  "frequency decoupling" is effectively a no-op. Cited in thesis as
  a negative result.

## Distant comparisons (cited for SOTA landscape, not direct baselines)

| Model | Params | mPQ (PanNuke) | Year | Note |
|---|---|---|---|---|
| LKCell-L | 163M | 0.508 | 2024 | UniRepLKNet backbone — too heavy for our budget |
| KongNet-Det | EffNetV2-L | F1 0.674 | 2025 | per-class decoders, no watershed |
| HoVer-NeXt | n/a | 0.477 (tissue mPQ) | 2024 | 2-decoder fast variant |
| CellViT++ | foundation+head | n/a | 2025 | frozen FM + lightweight classifier — paradigm shift, mentioned in Future Work |

## KD into Mamba — the genuinely novel slice

- **MCCL4PASS** (Expert Syst. Appl. 2025) — Mamba+CNN dual-student KD
  for OUTDOOR panoramic segmentation. Not applicable directly.
- **KD-Mamba** (2025) — KD into Mamba for trajectory prediction. Wrong
  task family.
- **MambaLiteSR** (2025) — KD into Mamba for super-resolution. Wrong
  task family.
- **mmMamba** (arXiv 2502.13145) — quadratic→linear distillation into
  Mamba LLM. Wrong modality.
- **No prior work** distills a transformer/conv teacher into a
  Mamba-decoder student for nuclei (or any other) instance segmentation
  on histopathology. This is the defensible novelty of our thesis as
  pivoted on 2026-05-17.

## Reviewer questions we should be ready for

1. **"Why not just use CP-Mamba?"** — heavier (params not specified in
   abstract but architecture suggests well above 15M), trained without
   KD (we add that signal). Honest answer: at fixed budget ≤15M, CP-Mamba
   is not in play; at unlimited budget, CP-Mamba wins.
2. **"Why not just use NuLite-T?"** — see "Direct neighbours" above.
3. **"Why does KD hurt the Dead class?"** — known limitation of vanilla
   KL when teacher's softmax is heavily concentrated. DKD with β=8 is
   our proposed fix — we test in Phase D.
4. **"Show statistical significance, not just std over 3 folds."** —
   per-image PQ npz files + scripts/stat_test.py exist (commit 7005929
   on dev/post-master-work). We will report paired Wilcoxon and
   bootstrap CI for every headline comparison.
5. **"Domain shift?"** — MoNuSeg eval pipeline ready (feat/monuseg-eval
   branch). Phase E results to be reported alongside PanNuke headline.
6. **"What about the methodological fixes you found?"** — spatial
   alignment of soft targets + correct hierarchical mPQ protocol are
   the two findings in RESULTS.md and are positioned as a secondary
   contribution.
