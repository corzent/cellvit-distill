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


def _try_import_mamba(version: str = "v1"):
    """Return (Mamba_cls, available_bool).

    version "v1" → Mamba1 (mamba_ssm.Mamba)
    version "v2" → Mamba2 (mamba_ssm.Mamba2). More stable in bf16 per
                   state-spaces/mamba issue tracker, preferred for vision.
    None if the requested class is missing or mamba_ssm isn't installed.
    """
    try:
        if version == "v2":
            from mamba_ssm import Mamba2  # type: ignore
            return Mamba2, True
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


_SCAN_PATTERNS = ("fwd", "bidirectional", "cross_scan_4way")


class MambaBlock(nn.Module):
    """Wrapper that runs Mamba on a 2D feature map with a selectable scan pattern.

    scan_pattern:
        "fwd"             — single forward row-major scan.
        "bidirectional"   — forward + reverse row-major, averaged.
        "cross_scan_4way" — VMamba SS2D style: row-fwd + row-rev + col-fwd +
                            col-rev, averaged. Best ImageNet results in
                            VMamba paper; +2 Mamba layers worth of compute.

    mamba_version:
        "v1" — mamba_ssm.Mamba (older, more reports of bf16 instability).
        "v2" — mamba_ssm.Mamba2 (newer, preferred for vision).

    use_real_mamba=False forces the CPU-friendly placeholder regardless of
    whether mamba_ssm is installed — useful for shape tests on CPU machines.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, scan_pattern: str = "bidirectional",
                 mamba_version: str = "v1", use_real_mamba: bool = True,
                 # Legacy kwarg: kept for tests that still pass bidirectional=...
                 bidirectional: Optional[bool] = None):
        super().__init__()
        if bidirectional is not None:
            scan_pattern = "bidirectional" if bidirectional else "fwd"
        if scan_pattern not in _SCAN_PATTERNS:
            raise ValueError(f"scan_pattern={scan_pattern!r} not in {_SCAN_PATTERNS}")
        self.scan_pattern = scan_pattern
        self.mamba_version = mamba_version

        Mamba, real_available = _try_import_mamba(mamba_version)
        if use_real_mamba and not real_available:
            cls = _MambaPlaceholder
            self._using_real = False
        else:
            cls = Mamba if use_real_mamba else _MambaPlaceholder
            self._using_real = use_real_mamba and real_available

        # One submodule per scan direction. For "fwd" → 1, "bidirectional" → 2,
        # "cross_scan_4way" → 4. Each owns its own selective-scan parameters.
        n_scans = {"fwd": 1, "bidirectional": 2, "cross_scan_4way": 4}[scan_pattern]
        self.scans = nn.ModuleList([
            cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_scans)
        ])

    @property
    def using_real_mamba(self) -> bool:
        return self._using_real

    def _flatten_pattern(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Reorder (B, C, H, W) to (B, L, C) for scan #idx, returning the flat seq."""
        if self.scan_pattern == "fwd":
            return x.flatten(2).transpose(1, 2)            # row-major fwd
        if self.scan_pattern == "bidirectional":
            seq = x.flatten(2).transpose(1, 2)
            return seq if idx == 0 else seq.flip(dims=[1])
        # cross_scan_4way: 0=row-fwd, 1=row-rev, 2=col-fwd, 3=col-rev
        if idx in (0, 1):
            seq = x.flatten(2).transpose(1, 2)             # row-major
        else:
            seq = x.transpose(-1, -2).flatten(2).transpose(1, 2)  # column-major
        if idx in (1, 3):
            seq = seq.flip(dims=[1])
        return seq

    def _unflatten_pattern(self, seq: torch.Tensor, idx: int, H: int, W: int) -> torch.Tensor:
        """Invert _flatten_pattern, returning (B, C, H, W)."""
        if self.scan_pattern == "bidirectional" and idx == 1:
            seq = seq.flip(dims=[1])
        if self.scan_pattern == "cross_scan_4way" and idx in (1, 3):
            seq = seq.flip(dims=[1])
        # idx 0/1 (row-major) -> reshape direct; idx 2/3 (col-major) -> reshape transposed
        B = seq.shape[0]
        C = seq.shape[2]
        if self.scan_pattern == "cross_scan_4way" and idx in (2, 3):
            # seq came from x.transpose(-1, -2) with shape (B, W*H, C). Reshape to
            # (B, C, W, H) then transpose to (B, C, H, W).
            return seq.transpose(1, 2).reshape(B, C, W, H).transpose(-1, -2)
        return seq.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W). Returns: (B, C, H, W)."""
        B, C, H, W = x.shape
        outs = []
        for idx, scan in enumerate(self.scans):
            seq = self._flatten_pattern(x, idx)
            y = scan(seq)
            outs.append(self._unflatten_pattern(y, idx, H, W))
        if len(outs) == 1:
            return outs[0]
        # Stack along new dim and average — same as sum / n but cheaper to write
        return torch.stack(outs, dim=0).mean(dim=0)


class PVMBlock(nn.Module):
    """Parallel Vision Mamba (UltraLight VM-UNet style).

    Splits channels into G groups, runs an independent (small) MambaBlock
    on each, concatenates. Halves the inner SSM compute vs running one
    full-channel block and tends to be more stable in mixed precision.
    """

    def __init__(self, channels: int, groups: int = 4, d_state: int = 16,
                 scan_pattern: str = "bidirectional", mamba_version: str = "v1",
                 use_real_mamba: bool = True,
                 bidirectional: Optional[bool] = None):
        super().__init__()
        assert channels % groups == 0, f"channels {channels} not divisible by groups {groups}"
        per = channels // groups
        self.groups = groups
        self.per = per
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=per, d_state=d_state, scan_pattern=scan_pattern,
                       mamba_version=mamba_version, use_real_mamba=use_real_mamba,
                       bidirectional=bidirectional)
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
                 scan_pattern: str = "bidirectional", mamba_version: str = "v1",
                 use_real_mamba: bool = True,
                 bidirectional: Optional[bool] = None):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        merged = in_ch + skip_ch
        self.proj_in = nn.Conv2d(merged, out_ch, kernel_size=1, bias=False)
        self.pvm = PVMBlock(out_ch, groups=groups, d_state=d_state,
                            scan_pattern=scan_pattern, mamba_version=mamba_version,
                            use_real_mamba=use_real_mamba,
                            bidirectional=bidirectional)
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
                 scan_pattern: str = "bidirectional", mamba_version: str = "v1",
                 use_real_mamba: bool = True,
                 bidirectional: Optional[bool] = None):
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
                scan_pattern=scan_pattern, mamba_version=mamba_version,
                use_real_mamba=use_real_mamba,
                bidirectional=bidirectional,
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
    print("MambaDecoder smoke (all scan patterns × both Mamba versions)")
    print("=" * 60)

    enc_chs = [64, 128, 256, 512]   # FastViT-S12
    dec_chs = [256, 128, 64, 32]    # current student decoder widths
    feats = [
        torch.randn(2, 64,  64, 64),
        torch.randn(2, 128, 32, 32),
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 512,  8,  8),
    ]

    rows = []
    for scan in _SCAN_PATTERNS:
        for ver in ("v1", "v2"):
            dec = MambaDecoder(enc_chs, dec_chs, groups=4, d_state=16,
                               scan_pattern=scan, mamba_version=ver,
                               use_real_mamba=True)
            n_params = sum(p.numel() for p in dec.parameters())
            y = dec(feats)
            real = "real" if dec.using_real_mamba else "stub"
            ok = y.shape == (2, dec_chs[-1], 128, 128)
            rows.append((scan, ver, real, n_params / 1e6, tuple(y.shape), ok))

    print(f"{'scan':22} {'ver':4} {'kernel':6} {'params(M)':>10} {'out_shape':>22} ok")
    print("-" * 70)
    for r in rows:
        scan, ver, real, mp, sh, ok = r
        print(f"{scan:22} {ver:4} {real:6} {mp:>10.2f} {str(sh):>22} {'✓' if ok else '✗'}")

    sys.exit(0 if all(r[-1] for r in rows) else 1)
