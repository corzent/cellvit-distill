"""Loss functions for CellViT distillation.

Ground truth losses (HoVer-Net style):
    - Binary: CE + Dice (or Focal + Dice)
    - HV maps: MSE + MSGE (mean squared gradient error)
    - Type: CE + Dice (with class weights and optional focal loss)

Distillation losses:
    - KL divergence on softened logits (binary, type heads)
    - MSE on HV map regression outputs
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


class DistillationLoss(nn.Module):
    """Knowledge distillation loss from teacher soft targets.

    For classification heads (binary, type): KL divergence on temperature-scaled logits
    For regression head (HV maps): MSE between teacher and student predictions
    """

    def __init__(self, temperature: float = 4.0, head_weights: Dict[str, float] = None):
        super().__init__()
        self.temperature = temperature
        self.head_weights = head_weights or {"binary": 1.0, "hv_map": 1.0, "type_map": 1.0}

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

        # Binary head: KL divergence (per-pixel mean)
        losses["distill_binary"] = (
            self._spatial_kl(pred["binary"], soft_targets["soft_binary"])
            * self.head_weights["binary"]
        )

        # HV map head: MSE (regression, no temperature needed, fp32 for stability)
        hv_pred = pred["hv_map"].float()
        hv_teacher = soft_targets["soft_hv"].float()
        nucleus_mask = (soft_targets["soft_binary"].argmax(dim=1, keepdim=True) == 1).float()
        hv_diff = (hv_pred - hv_teacher) ** 2 * nucleus_mask.expand_as(hv_pred)
        num_pixels = nucleus_mask.sum().clamp(min=1)
        losses["distill_hv"] = (hv_diff.sum() / num_pixels) * self.head_weights["hv_map"]

        # Type head: KL divergence (per-pixel mean)
        losses["distill_type"] = (
            self._spatial_kl(pred["type_map"], soft_targets["soft_type"])
            * self.head_weights["type_map"]
        )

        losses["distill_total"] = sum(v for k, v in losses.items() if k != "distill_total")
        return losses


class CombinedLoss(nn.Module):
    """Combined GT + Distillation loss: L = (1 - alpha) * L_gt + alpha * L_distill"""

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
    ):
        super().__init__()
        self.alpha = alpha
        self.distillation_enabled = distillation_enabled
        self.gt_loss = CellSegLoss(
            loss_weights,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            type_class_weights=type_class_weights,
            use_ftl_binary=use_ftl_binary,
            tissue_aux=tissue_aux,
        )

        if distillation_enabled:
            self.distill_loss = DistillationLoss(temperature, head_weights)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Student outputs
            target: Dict containing both GT maps and (optionally) soft targets
        """
        gt_losses = self.gt_loss(pred, target)

        if self.distillation_enabled and "soft_binary" in target:
            distill_losses = self.distill_loss(pred, target)
            total = (1 - self.alpha) * gt_losses["total"] + self.alpha * distill_losses["distill_total"]

            all_losses = {**gt_losses, **distill_losses}
            all_losses["total"] = total
            return all_losses

        return gt_losses
