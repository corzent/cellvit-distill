"""Loss functions for CellViT distillation.

Ground truth losses (HoVer-Net style):
    - Binary: CE + Dice (or Focal + Dice)
    - HV maps: MSE + MSGE (mean squared gradient error)
    - Type: CE + Dice (with class weights and optional focal loss)

Distillation losses:
    - KL divergence on softened logits (binary, type heads)         [loss_type="kl_div"]
    - Frequency-Decoupled KD on softmax maps (binary, type)         [loss_type="ufd_kd"]
    - MSE on HV map regression outputs (always)
    - Feature matching on intermediate teacher tokens (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) softmax probabilities
            target: (B, C, H, W) one-hot encoded
        """
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        intersection = (pred * target).sum(dim=2)
        cardinality = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for dense classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples and focuses on hard ones. Particularly useful
    for class-imbalanced segmentation (e.g., Dead class = 1.5% of nuclei).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter. gamma=0 reduces to CE.
            alpha: Per-class weights tensor of shape (C,). If None, uniform.
            reduction: 'mean' or 'sum'.
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W) raw logits
            targets: (B, H, W) class indices
        """
        B, C, H, W = logits.shape
        # (B, C, H, W) -> (B*H*W, C)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        log_p = F.log_softmax(logits_flat, dim=1)
        p = log_p.exp()

        # Gather probabilities at target class
        log_pt = log_p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets_flat)
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss (Abraham & Khan 2019).

    Generalizes Dice: Tversky index = TP / (TP + alpha*FN + beta*FP).
    Setting alpha > beta penalizes false negatives more (useful for small
    nuclei that are easily missed). Focal exponent gamma < 1 focuses on
    hard examples.

    FTL = (1 - Tversky)^gamma, summed over foreground classes.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) softmax probabilities
            target: (B, C, H, W) one-hot encoded
        """
        pred = pred.contiguous().view(pred.shape[0], pred.shape[1], -1)
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)

        tp = (pred * target).sum(dim=2)
        fn = (target * (1 - pred)).sum(dim=2)
        fp = ((1 - target) * pred).sum(dim=2)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        # Skip background (class 0) and focus on foreground classes.
        if tversky.shape[1] > 1:
            tversky = tversky[:, 1:]
        return ((1 - tversky) ** self.gamma).mean()


class MSGELoss(nn.Module):
    """Mean Squared Gradient Error for HV maps.

    Computes MSE on Sobel gradients of predicted vs target HV maps.
    This encourages sharp boundaries between nuclei.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels for gradient computation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        # Shape: (1, 1, 3, 3) for depthwise conv
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def _gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradients for each channel independently."""
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        # Cast Sobel kernels to match input device and dtype (fp16 under autocast)
        sx = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sy = self.sobel_y.to(device=x.device, dtype=x.dtype)
        gx = F.conv2d(x, sx, padding=1)
        gy = F.conv2d(x, sy, padding=1)
        return torch.cat([gx, gy], dim=1).reshape(B, C * 2, H, W)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 2, H, W) predicted HV maps
            target: (B, 2, H, W) ground truth HV maps
            mask: (B, 1, H, W) binary mask (1 where nuclei exist)
        """
        pred_grad = self._gradient(pred)
        target_grad = self._gradient(target)

        # Only compute loss within nuclear regions
        mask = mask.expand_as(pred_grad)
        diff = (pred_grad - target_grad) ** 2
        diff = diff * mask

        # Mean over nuclear pixels
        num_pixels = mask.sum().clamp(min=1)
        return diff.sum() / num_pixels


class CellSegLoss(nn.Module):
    """Combined ground truth loss for cell segmentation (HoVer-Net style).

    Supports optional focal loss and per-class weights for type classification,
    optional Focal Tversky Loss for binary head (NuLite recipe), and an
    optional tissue classification auxiliary task to encourage richer encoder
    representations.
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        type_class_weights: Optional[List[float]] = None,
        use_ftl_binary: bool = False,
        tissue_aux: bool = False,
    ):
        super().__init__()
        self.weights = loss_weights
        self.dice = DiceLoss()
        self.mse = nn.MSELoss()
        self.msge = MSGELoss()
        self.ftl = FocalTverskyLoss() if use_ftl_binary else None
        self.use_ftl_binary = use_ftl_binary
        self.tissue_aux = tissue_aux

        # Binary head loss
        self.binary_ce = nn.CrossEntropyLoss()

        # Type head loss: focal + class weights or plain CE + class weights
        type_w = torch.tensor(type_class_weights, dtype=torch.float32) if type_class_weights else None
        if use_focal:
            self.type_cls_loss = FocalLoss(gamma=focal_gamma, alpha=type_w)
        else:
            self.type_cls_loss = nn.CrossEntropyLoss(
                weight=type_w if type_w is not None else None,
            )

        # Tissue classification aux head — 19 PanNuke tissue types
        self.tissue_ce = nn.CrossEntropyLoss() if tissue_aux else None

        self.use_focal = use_focal

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Model outputs (raw logits)
                - binary: (B, 2, H, W)
                - hv_map: (B, 2, H, W)
                - type_map: (B, C, H, W)
            target: Ground truth
                - binary_map: (B, 2, H, W) one-hot
                - hv_map: (B, 2, H, W)
                - type_map: (B, C, H, W) one-hot

        Returns:
            Dict with individual losses and total
        """
        losses = {}

        # --- Binary head ---
        binary_pred = pred["binary"]
        binary_target = target["binary_map"]
        binary_target_class = binary_target.argmax(dim=1)  # (B, H, W) class indices
        binary_probs = F.softmax(binary_pred, dim=1)

        if self.use_ftl_binary:
            # NuLite recipe: FTL + Dice, more sensitive to false negatives
            losses["binary_ftl"] = self.ftl(binary_probs, binary_target) * self.weights["binary_ce"]
            losses["binary_dice"] = self.dice(binary_probs, binary_target) * self.weights["binary_dice"]
        else:
            losses["binary_ce"] = self.binary_ce(binary_pred, binary_target_class) * self.weights["binary_ce"]
            losses["binary_dice"] = self.dice(binary_probs, binary_target) * self.weights["binary_dice"]

        # --- HV map head ---
        hv_pred = pred["hv_map"]
        hv_target = target["hv_map"]
        nucleus_mask = binary_target[:, 1:2, :, :]  # (B, 1, H, W)

        # MSE only within nuclear regions
        hv_diff = (hv_pred - hv_target) ** 2 * nucleus_mask.expand_as(hv_pred)
        num_nuc_pixels = nucleus_mask.sum().clamp(min=1)
        losses["hv_mse"] = (hv_diff.sum() / num_nuc_pixels) * self.weights["hv_mse"]

        # Gradient error
        losses["hv_msge"] = self.msge(hv_pred, hv_target, nucleus_mask) * self.weights["hv_msge"]

        # --- Type head ---
        type_pred = pred["type_map"]
        type_target = target["type_map"]
        type_target_class = type_target.argmax(dim=1)

        if self.use_focal:
            losses["type_focal"] = self.type_cls_loss(type_pred, type_target_class) * self.weights["type_ce"]
        else:
            losses["type_ce"] = self.type_cls_loss(type_pred, type_target_class) * self.weights["type_ce"]
        losses["type_dice"] = self.dice(
            F.softmax(type_pred, dim=1), type_target
        ) * self.weights["type_dice"]

        # --- Tissue classification aux task (optional) ---
        if self.tissue_aux and "tissue_logits" in pred and "tissue_label" in target:
            losses["tissue_ce"] = self.tissue_ce(
                pred["tissue_logits"], target["tissue_label"]
            ) * self.weights.get("tissue_ce", 0.1)

        losses["total"] = sum(v for k, v in losses.items() if k != "total")
        return losses


class FeatureMatchLoss(nn.Module):
    """Feature-based distillation loss (FitNets-style).

    Matches intermediate student features to teacher ViT-H tokens via a 1x1
    projection layer that lives inside the student. Combines MSE and
    (optionally) cosine similarity — cosine captures directional alignment
    while MSE constrains magnitude.

    The projection is expected to be applied before calling this loss. Inputs
    should already be at matching spatial resolution (16x16 for SAM-H tokens
    at 256x256 input, matches FastViT-S12 stage 2 / ConvNeXt-Tiny stage 3).
    """

    def __init__(self, use_cosine: bool = True, cosine_weight: float = 0.5):
        super().__init__()
        self.use_cosine = use_cosine
        self.cosine_weight = cosine_weight

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            student_feat: (B, C, H, W) after projection to match teacher
            teacher_feat: (B, C, H, W) pre-computed teacher tokens
        """
        student_feat = student_feat.float()
        teacher_feat = teacher_feat.float()

        # Spatial size mismatch fallback (shouldn't happen if aligned).
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            student_feat = F.interpolate(
                student_feat, size=teacher_feat.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        mse = F.mse_loss(student_feat, teacher_feat)

        if not self.use_cosine:
            return mse

        # Cosine similarity per pixel, averaged.
        s_norm = F.normalize(student_feat, dim=1)
        t_norm = F.normalize(teacher_feat, dim=1)
        cos_sim = (s_norm * t_norm).sum(dim=1)  # (B, H, W)
        cos_loss = 1.0 - cos_sim.mean()

        return mse + self.cosine_weight * cos_loss


class FrequencyDecoupledDistillLoss(nn.Module):
    """Frequency-decoupled KD for dense predictions (adapted from UFD-KD, BMVC 2025).

    Standard per-pixel KL on segmentation logits averages over all pixels,
    diluting rare/small-object signal (e.g. Dead nuclei occupy <2% of pixels
    so their KL contribution is dwarfed by background). High-frequency
    components of the softmax map carry most of the rare-class and boundary
    information; low-frequency components carry the global class layout.

    Pipeline:
      1. Convert student/teacher logits to softmax with temperature T.
      2. Apply 2D DCT-II across the (H, W) spatial axes for each (B, C).
      3. Split DCT coefficients into LF block (top-left K_lf × K_lf) and HF
         residual (everything else).
      4. Weighted MSE in DCT domain:
         L = w_lf * MSE(s_lf, t_lf) + w_hf * MSE(s_hf, t_hf).
      5. Scale by T² (KD convention) and return a scalar.

    Args:
        temperature: KD softmax temperature.
        lf_size: Side length of the LF block to keep (e.g. 32 for a 256x256
            map keeps DC + the 32x32 lowest-frequency components). Clamped
            to min(H, W) at runtime if a head's spatial size is smaller.
        lf_weight: Multiplier on the LF-band MSE.
        hf_weight: Multiplier on the HF-band MSE. For rare-class focus set
            hf_weight > lf_weight (HF carries small-object information).
        debug_steps: If >0, print raw (pre-T²) lf/hf magnitudes for that many
            forward calls. Helps calibrate alpha vs the kl_div baseline.

    Debug attributes (populated after each forward):
        last_lf_raw, last_hf_raw — Python floats, pre-T² band losses.
        last_total_raw           — w_lf*lf + w_hf*hf, pre-T².
    """

    def __init__(
        self,
        temperature: float = 10.0,
        lf_size: int = 32,
        lf_weight: float = 1.0,
        hf_weight: float = 3.0,
        debug_steps: int = 0,
    ):
        super().__init__()
        self.temperature = temperature
        self.lf_size = lf_size
        self.lf_weight = lf_weight
        self.hf_weight = hf_weight
        # Cache keyed by (N, device, dtype) — survives multi-GPU / fp32-only paths.
        self._dct_cache: Dict[tuple, torch.Tensor] = {}
        self._debug_steps_remaining = debug_steps
        self.last_lf_raw: float = 0.0
        self.last_hf_raw: float = 0.0
        self.last_total_raw: float = 0.0

    def _dct_matrix(self, N: int, device, dtype) -> torch.Tensor:
        """Orthonormal DCT-II matrix of side N."""
        key = (N, device, dtype)
        cached = self._dct_cache.get(key)
        if cached is not None:
            return cached
        n = torch.arange(N, device=device, dtype=dtype)
        k = n.view(-1, 1)
        D = torch.cos(torch.pi * (2 * n + 1) * k / (2 * N)) * (2.0 / N) ** 0.5
        D[0] = D[0] / (2 ** 0.5)
        self._dct_cache[key] = D
        return D

    def _dct2(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT-II along the last two dims; preserves leading dims.

        Implemented as Dh @ x @ Dw^T (matrix DCT). For 256x256 maps with B*C
        in the leading dims this is a few batched matmuls — fast vs FFT-DCT.
        """
        H, W = x.shape[-2:]
        Dh = self._dct_matrix(H, x.device, x.dtype)
        Dw = self._dct_matrix(W, x.device, x.dtype)
        x_t = torch.einsum("ij,...jk->...ik", Dh, x)
        x_t = torch.einsum("...ik,jk->...ij", x_t, Dw)
        return x_t

    @torch.amp.autocast("cuda", enabled=False)
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Returns scalar loss (already scaled by T²)."""
        T = self.temperature
        # fp32 promotion for DCT precision (softmax then DCT in fp16 underflows).
        s_prob = F.softmax(student_logits.float() / T, dim=1)
        t_prob = F.softmax(teacher_logits.float() / T, dim=1)

        H, W = s_prob.shape[-2:]
        K = min(self.lf_size, H, W)

        s_dct = self._dct2(s_prob)
        t_dct = self._dct2(t_prob)

        s_lf = s_dct[..., :K, :K]
        t_lf = t_dct[..., :K, :K]
        lf_loss = F.mse_loss(s_lf, t_lf)

        # HF residual = full - LF block. Compute via squared-norm difference.
        full_sq = (s_dct - t_dct).pow(2)
        lf_sq = (s_lf - t_lf).pow(2)
        total_elems = full_sq.numel()
        lf_elems = lf_sq.numel()
        hf_elems = max(total_elems - lf_elems, 1)
        hf_loss = (full_sq.sum() - lf_sq.sum()) / hf_elems

        weighted = self.lf_weight * lf_loss + self.hf_weight * hf_loss

        # Stash raw magnitudes for outside-loop calibration logging.
        self.last_lf_raw = lf_loss.detach().item()
        self.last_hf_raw = hf_loss.detach().item()
        self.last_total_raw = weighted.detach().item()
        if self._debug_steps_remaining > 0:
            print(
                f"  [ufd_kd debug] H={H} W={W} K={K} "
                f"lf_raw={self.last_lf_raw:.3e} hf_raw={self.last_hf_raw:.3e} "
                f"weighted={self.last_total_raw:.3e} "
                f"final(×T²={T**2:.0f})={(self.last_total_raw * T**2):.3e}"
            )
            self._debug_steps_remaining -= 1

        return weighted * (T ** 2)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss from teacher soft targets.

    For classification heads (binary, type): KL divergence on temperature-scaled logits
        (loss_type="kl_div") or Frequency-Decoupled KD (loss_type="ufd_kd").
    For regression head (HV maps): MSE between teacher and student predictions.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        head_weights: Dict[str, float] = None,
        loss_type: str = "kl_div",
        ufd_lf_size: int = 32,
        ufd_lf_weight: float = 1.0,
        ufd_hf_weight: float = 3.0,
        ufd_debug_steps: int = 0,
    ):
        super().__init__()
        self.temperature = temperature
        self.head_weights = head_weights or {"binary": 1.0, "hv_map": 1.0, "type_map": 1.0}
        self.loss_type = loss_type
        if loss_type == "ufd_kd":
            self.fdkd = FrequencyDecoupledDistillLoss(
                temperature=temperature,
                lf_size=ufd_lf_size,
                lf_weight=ufd_lf_weight,
                hf_weight=ufd_hf_weight,
                debug_steps=ufd_debug_steps,
            )
        elif loss_type != "kl_div":
            raise ValueError(
                f"Unknown distillation loss_type={loss_type!r}; "
                "expected 'kl_div' or 'ufd_kd'"
            )

    @torch.amp.autocast("cuda", enabled=False)
    def _spatial_kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence averaged over all spatial positions.

        Computed in fp32 to avoid overflow in large spatial reductions.
        Standard KLDivLoss(batchmean) only divides by batch size, not H×W.
        For dense prediction we need per-pixel mean.
        """
        T = self.temperature
        # Force fp32 for numerical stability
        student_logits = student_logits.float()
        teacher_logits = teacher_logits.float()
        B, C, H, W = student_logits.shape
        # (B, C, H, W) -> (B*H*W, C)
        s = F.log_softmax(student_logits.permute(0, 2, 3, 1).reshape(-1, C) / T, dim=1)
        t = F.softmax(teacher_logits.permute(0, 2, 3, 1).reshape(-1, C) / T, dim=1)
        # Mean over all pixels and batch, scale by T²
        kl = F.kl_div(s, t, reduction="batchmean") * (T ** 2)
        # batchmean divides by B*H*W here since first dim is B*H*W
        return kl

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        soft_targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Student model outputs (raw logits)
            soft_targets: Pre-computed teacher outputs (raw logits)

        Returns:
            Dict with individual distillation losses and total
        """
        losses = {}

        # Classification heads (binary, type): KL or UFD-KD per loss_type.
        def _classifier_distill(student, teacher):
            if self.loss_type == "ufd_kd":
                return self.fdkd(student, teacher)
            return self._spatial_kl(student, teacher)

        losses["distill_binary"] = (
            _classifier_distill(pred["binary"], soft_targets["soft_binary"])
            * self.head_weights["binary"]
        )

        # HV map head: MSE (regression, no temperature needed, fp32 for stability)
        hv_pred = pred["hv_map"].float()
        hv_teacher = soft_targets["soft_hv"].float()
        nucleus_mask = (soft_targets["soft_binary"].argmax(dim=1, keepdim=True) == 1).float()
        hv_diff = (hv_pred - hv_teacher) ** 2 * nucleus_mask.expand_as(hv_pred)
        num_pixels = nucleus_mask.sum().clamp(min=1)
        losses["distill_hv"] = (hv_diff.sum() / num_pixels) * self.head_weights["hv_map"]

        # Type head: KL or UFD-KD per loss_type.
        losses["distill_type"] = (
            _classifier_distill(pred["type_map"], soft_targets["soft_type"])
            * self.head_weights["type_map"]
        )

        losses["distill_total"] = sum(v for k, v in losses.items() if k != "distill_total")
        return losses


class CombinedLoss(nn.Module):
    """Combined GT + Distillation loss.

    L = (1 - alpha) * L_gt + alpha * L_distill_response + beta * L_distill_feature

    - alpha: weight of response-based (logits) distillation vs GT
    - beta: weight of feature-based (intermediate representations) distillation
    - When beta > 0 and model provides feat_proj + target has soft_feat,
      feature distillation is added as an additional term.
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        alpha: float = 0.5,
        temperature: float = 4.0,
        head_weights: Dict[str, float] = None,
        distillation_enabled: bool = False,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        type_class_weights: Optional[List[float]] = None,
        use_ftl_binary: bool = False,
        tissue_aux: bool = False,
        feature_distill_enabled: bool = False,
        beta: float = 1.0,
        feature_loss_use_cosine: bool = True,
        feature_loss_cosine_weight: float = 0.5,
        distill_loss_type: str = "kl_div",
        ufd_lf_size: int = 32,
        ufd_lf_weight: float = 1.0,
        ufd_hf_weight: float = 3.0,
        ufd_debug_steps: int = 0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.distillation_enabled = distillation_enabled
        self.feature_distill_enabled = feature_distill_enabled
        self.gt_loss = CellSegLoss(
            loss_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            type_class_weights=type_class_weights,
            use_ftl_binary=use_ftl_binary,
            tissue_aux=tissue_aux,
        )

        if distillation_enabled:
            self.distill_loss = DistillationLoss(
                temperature,
                head_weights,
                loss_type=distill_loss_type,
                ufd_lf_size=ufd_lf_size,
                ufd_lf_weight=ufd_lf_weight,
                ufd_hf_weight=ufd_hf_weight,
                ufd_debug_steps=ufd_debug_steps,
            )

        if feature_distill_enabled:
            self.feature_match_loss = FeatureMatchLoss(
                use_cosine=feature_loss_use_cosine,
                cosine_weight=feature_loss_cosine_weight,
            )

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Student outputs (may include "feat_proj" if feature distill on)
            target: GT maps + (optionally) soft_binary/soft_hv/soft_type + soft_feat
        """
        gt_losses = self.gt_loss(pred, target)

        # Start with GT losses
        all_losses = dict(gt_losses)
        total = gt_losses["total"]

        # Response-based distillation (logit matching)
        if self.distillation_enabled and "soft_binary" in target:
            distill_losses = self.distill_loss(pred, target)
            total = (1 - self.alpha) * total + self.alpha * distill_losses["distill_total"]
            all_losses.update(distill_losses)

        # Feature-based distillation (intermediate representation matching)
        if (
            self.feature_distill_enabled
            and "feat_proj" in pred
            and "soft_feat" in target
        ):
            feat_loss = self.feature_match_loss(pred["feat_proj"], target["soft_feat"])
            all_losses["distill_feature"] = feat_loss * self.beta
            total = total + self.beta * feat_loss

        all_losses["total"] = total
        return all_losses
