"""Mamba (SSM) based decoder for nuclei segmentation, draft for Phase C.

Design: FastViT-S12 (or any timm) encoder feeds a 4-stage Mamba decoder.
Each decoder stage applies a Parallel Vision Mamba (PVM) block in the
UltraLight VM-UNet style — channels split into G groups, each group runs
through an independent Mamba layer, then concatenated. Cheap vs running
one Mamba on the full channel count.

Status: STRUCTURE FROZEN. The Mamba kernel itself is loaded lazily — when
mamba_ssm is unavailable (env still being built), `_MambaPlaceholder` is
used which is a LayerNorm+Conv1d stand-in so the shapes / param count /
forward pass can be exercised CPU-only. Once env is up, set
`use_real_mamba=True` in MambaDecoder ctor to switch.

Reference: UltraLight VM-UNet (Wu et al., 2024, github.com/wurenkai/UltraLight-VM-UNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ----------------------------------------------------------------------
# Mamba block — real or placeholder
# ----------------------------------------------------------------------


def _try_import_mamba():
    """Return (Mamba_cls, available_bool). None if mamba_ssm is missing."""
    try:
        from mamba_ssm import Mamba  # type: ignore
        return Mamba, True
    except Exception:
        return None, False


class _MambaPlaceholder(nn.Module):
    """CPU-friendly stand-in with similar param scale to real Mamba.

    Roughly: 2 × C² + 4 × C × d_state weights vs Mamba's ~ same. Lets us
    forward-test shapes & count params before mamba_ssm is installed.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        inner = d_model * expand
        self.in_proj = nn.Linear(d_model, inner * 2, bias=False)
        self.conv = nn.Conv1d(inner, inner, kernel_size=d_conv, padding=d_conv - 1, groups=inner)
        self.x_proj = nn.Linear(inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, inner)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.d_state = d_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D). Returns: (B, L, D)."""
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        # Causal conv along sequence
        x = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        x = F.silu(x)
        # Fake selective-scan: project, average, broadcast
        bc = self.x_proj(x)
        b, c = bc.chunk(2, dim=-1)
        dt = self.dt_proj(b.mean(dim=1, keepdim=True))
        x = x + dt
        x = x * F.silu(gate)
        return residual + self.out_proj(x)


class MambaBlock(nn.Module):
    """Wrapper that runs Mamba on a 2D feature map.

    Optional bidirectional scan (forward + reverse, averaged). Bidirectional
    is required for non-causal vision tasks where future tokens carry signal
    for past ones (BMVC 2024 vision-Mamba ablations confirm).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, bidirectional: bool = True,
                 use_real_mamba: bool = True):
        super().__init__()
        Mamba, real_available = _try_import_mamba()
        if use_real_mamba and not real_available:
            # Silent fallback so dev can iterate on architecture before env ready
            cls = _MambaPlaceholder
            self._using_real = False
        else:
            cls = Mamba if use_real_mamba else _MambaPlaceholder
            self._using_real = use_real_mamba and real_available
        self.fwd = cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bwd = cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) if bidirectional else None
        self.bidirectional = bidirectional

    @property
    def using_real_mamba(self) -> bool:
        return self._using_real

    def _scan(self, x_seq: torch.Tensor) -> torch.Tensor:
        """x_seq: (B, L, D)."""
        y = self.fwd(x_seq)
        if self.bwd is not None:
            y_rev = self.bwd(x_seq.flip(dims=[1])).flip(dims=[1])
            y = (y + y_rev) * 0.5
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Returns: (B, C, H, W)."""
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        y = self._scan(x_seq)
        return y.transpose(1, 2).reshape(B, C, H, W)


class PVMBlock(nn.Module):
    """Parallel Vision Mamba (UltraLight VM-UNet style).

    Splits channels into G groups, runs an independent (small) MambaBlock
    on each, concatenates. Halves the inner SSM compute vs running one
    full-channel block and tends to be more stable in mixed precision.
    """

    def __init__(self, channels: int, groups: int = 4, d_state: int = 16,
                 bidirectional: bool = True, use_real_mamba: bool = True):
        super().__init__()
        assert channels % groups == 0, f"channels {channels} not divisible by groups {groups}"
        per = channels // groups
        self.groups = groups
        self.per = per
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=per, d_state=d_state, bidirectional=bidirectional,
                       use_real_mamba=use_real_mamba)
            for _ in range(groups)
        ])
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    @property
    def using_real_mamba(self) -> bool:
        return self.blocks[0].using_real_mamba

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.groups, dim=1)
        out = torch.cat([blk(c) for blk, c in zip(self.blocks, chunks)], dim=1)
        return self.norm(out + x)  # residual + group-norm stabilization


# ----------------------------------------------------------------------
# Decoder
# ----------------------------------------------------------------------


class MambaDecoderBlock(nn.Module):
    """Upsample + concat-skip + PVM + projection conv.

    Mirrors DecoderBlock from student.py so the interface is identical:
    forward(x, skip) -> upsampled, mamba-mixed feature at out_ch channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 groups: int = 4, d_state: int = 16,
                 bidirectional: bool = True, use_real_mamba: bool = True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        merged = in_ch + skip_ch
        self.proj_in = nn.Conv2d(merged, out_ch, kernel_size=1, bias=False)
        self.pvm = PVMBlock(out_ch, groups=groups, d_state=d_state,
                            bidirectional=bidirectional, use_real_mamba=use_real_mamba)
        self.proj_out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    @property
    def using_real_mamba(self) -> bool:
        return self.pvm.using_real_mamba

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.proj_in(x)
        x = self.pvm(x)
        return self.proj_out(x)


class MambaDecoder(nn.Module):
    """Drop-in replacement for HoVerNetDecoder using Mamba blocks.

    Same input contract: list of 4 encoder features at stages [s1, s2, s3, s4]
    from shallow to deep. Same output contract: a single feature map of
    decoder_channels[-1] channels at near-input resolution.

    Params for FastViT-S12 ([64, 128, 256, 512]) + decoder_channels
    [256, 128, 64, 32]:
        ~2.5-4.5M depending on groups & d_state. Total student stays ~13M.
    """

    def __init__(self, encoder_channels: List[int], decoder_channels: List[int],
                 groups: int = 4, d_state: int = 16,
                 bidirectional: bool = True, use_real_mamba: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for i, out_ch in enumerate(decoder_channels):
            # Snap out_ch to divisible-by-groups so PVMBlock is happy.
            if out_ch % groups != 0:
                out_ch = out_ch - (out_ch % groups) or groups
            skip_idx = len(encoder_channels) - 2 - i
            skip_ch = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            self.blocks.append(MambaDecoderBlock(
                in_ch=in_ch, skip_ch=skip_ch, out_ch=out_ch,
                groups=groups, d_state=d_state,
                bidirectional=bidirectional, use_real_mamba=use_real_mamba,
            ))
            in_ch = out_ch
        self.final_channels = in_ch

    @property
    def using_real_mamba(self) -> bool:
        return self.blocks[0].using_real_mamba

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for i, block in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx] if skip_idx >= 0 else None
            x = block(x, skip)
        return x


# ----------------------------------------------------------------------
# Smoke test (CPU-only with placeholder, GPU with real Mamba)
# ----------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("MambaDecoder smoke")
    print("=" * 60)

    enc_chs = [64, 128, 256, 512]   # FastViT-S12
    dec_chs = [256, 128, 64, 32]    # current student decoder widths
    dec = MambaDecoder(enc_chs, dec_chs, groups=4, d_state=16,
                       bidirectional=True, use_real_mamba=True)
    print(f"using_real_mamba: {dec.using_real_mamba}")
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"decoder params:   {n_params/1e6:.2f}M  (target ≤5M)")

    # Fake encoder features at 256x256 input (FastViT downsamples 4×, 8×, 16×, 32×)
    feats = [
        torch.randn(2, 64,  64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512,  8,  8),
    ]
    y = dec(feats)
    print(f"output shape:     {tuple(y.shape)}  (expect [2, {dec_chs[-1]}, 128, 128])")
    assert y.shape[1] == dec_chs[-1], "channel mismatch"
    print("forward ok.")
    sys.exit(0)
