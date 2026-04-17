"""Student model: lightweight encoder + HoVer-Net style decoder with 3 heads.

Architecture:
    Encoder: ConvNeXt-Tiny (or ResNet50, EfficientNet-B0) from timm
    Decoder: FPN-style progressive upsampling with skip connections
    Heads:
        - binary: 2-ch (background, nucleus)
        - hv_map: 2-ch (horizontal, vertical distance maps)
        - type_map: N-ch (cell type classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single decoder block: upsample + concat skip + 2x ConvBNReLU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle size mismatch from odd dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class HoVerNetDecoder(nn.Module):
    """FPN-style decoder that produces features for 3 heads."""

    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]):
        """
        Args:
            encoder_channels: Feature channels at each encoder stage [stage1, ..., stage4]
                              (from shallow to deep, e.g. [96, 192, 384, 768] for ConvNeXt-Tiny)
            decoder_channels: Output channels at each decoder stage [256, 128, 64, 32]
        """
        super().__init__()
        self.blocks = nn.ModuleList()

        # Bottleneck (deepest encoder features -> first decoder level)
        in_ch = encoder_channels[-1]

        for i, out_ch in enumerate(decoder_channels):
            # Skip connection comes from encoder (reversed order: deep -> shallow)
            skip_idx = len(encoder_channels) - 2 - i
            skip_ch = encoder_channels[skip_idx] if skip_idx >= 0 else 0
            self.blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch

        self.final_channels = decoder_channels[-1]

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Encoder feature maps [stage1, stage2, stage3, stage4]

        Returns:
            Decoded feature map at original resolution
        """
        x = features[-1]  # deepest features

        for i, block in enumerate(self.blocks):
            skip_idx = len(features) - 2 - i
            skip = features[skip_idx] if skip_idx >= 0 else None
            x = block(x, skip)

        return x


class StudentCellViT(nn.Module):
    """Student model for CellViT distillation.

    Uses a lightweight pretrained encoder (e.g., ConvNeXt-Tiny ~28M params)
    with a HoVer-Net style decoder and three output heads.
    """

    # Encoder configs: name -> (timm model name, feature channels per stage)
    # Channels are auto-detected via a dummy forward pass; values here are
    # for reference only (used when timm is unavailable).
    ENCODER_CONFIGS = {
        "convnext_tiny": ("convnext_tiny.fb_in22k", [96, 192, 384, 768]),
        "fastvit_s12": ("fastvit_s12.apple_in1k", [64, 128, 256, 512]),
        "fastvit_sa12": ("fastvit_sa12.apple_in1k", [64, 128, 256, 512]),
        "fastvit_sa24": ("fastvit_sa24.apple_in1k", [64, 128, 256, 512]),
        "resnet50": ("resnet50.a1_in1k", [256, 512, 1024, 2048]),
        "efficientnet_b0": ("efficientnet_b0.ra_in1k", [24, 40, 112, 320]),
        "mobilenetv3": ("mobilenetv3_large_100.ra_in1k", [24, 40, 112, 960]),
    }

    def __init__(
        self,
        encoder_name: str = "convnext_tiny",
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        num_classes: int = 6,
        tissue_aux: bool = False,
        num_tissue_classes: int = 19,
        hv_tanh: bool = False,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.num_classes = num_classes
        self.tissue_aux = tissue_aux

        # Get encoder config
        if encoder_name not in self.ENCODER_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder_name}. Choose from {list(self.ENCODER_CONFIGS.keys())}")

        timm_name, encoder_channels = self.ENCODER_CONFIGS[encoder_name]

        # Create encoder with feature extraction at 4 stages
        self.encoder = timm.create_model(
            timm_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Get actual channel counts (may differ from config for some models)
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            dummy_features = self.encoder(dummy_input)
            encoder_channels = [f.shape[1] for f in dummy_features]

        # Shared decoder
        self.decoder = HoVerNetDecoder(encoder_channels, decoder_channels)

        feat_ch = self.decoder.final_channels

        # --- Output Heads ---
        # Binary head: nucleus vs background
        self.binary_head = nn.Sequential(
            ConvBNReLU(feat_ch, feat_ch),
            nn.Conv2d(feat_ch, 2, 1),
        )

        # HV map head: horizontal + vertical distance maps.
        # Optional tanh activation enforces the target range [-1, 1] exactly
        # for cleaner gradients at the boundary (config: student.hv_tanh).
        hv_layers = [ConvBNReLU(feat_ch, feat_ch), nn.Conv2d(feat_ch, 2, 1)]
        if hv_tanh:
            hv_layers.append(nn.Tanh())
        self.hv_head = nn.Sequential(*hv_layers)

        # Type classification head
        self.type_head = nn.Sequential(
            ConvBNReLU(feat_ch, feat_ch),
            nn.Conv2d(feat_ch, num_classes, 1),
        )

        # Tissue classification aux head — global AvgPool over deepest encoder
        # features + linear classifier. Encourages the encoder to learn
        # tissue-aware representations (NuLite recipe).
        if tissue_aux:
            bottleneck_ch = encoder_channels[-1]
            self.tissue_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bottleneck_ch, num_tissue_classes),
            )
        else:
            self.tissue_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image (B, 3, 256, 256)

        Returns:
            Dictionary with raw logits/predictions:
                binary: (B, 2, 256, 256) - logits
                hv_map: (B, 2, 256, 256) - raw predictions (no activation)
                type_map: (B, 6, 256, 256) - logits
        """
        # Handle resolution: encoder may downsample more than decoder upsamples
        input_size = x.shape[2:]

        features = self.encoder(x)
        decoded = self.decoder(features)

        # Ensure output matches input resolution
        if decoded.shape[2:] != input_size:
            decoded = F.interpolate(decoded, size=input_size, mode="bilinear", align_corners=False)

        out = {
            "binary": self.binary_head(decoded),
            "hv_map": self.hv_head(decoded),
            "type_map": self.type_head(decoded),
        }
        if self.tissue_head is not None:
            out["tissue_logits"] = self.tissue_head(features[-1])
        return out

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        head_params = sum(
            p.numel()
            for module in [self.binary_head, self.hv_head, self.type_head]
            for p in module.parameters()
            if p.requires_grad
        )
        total = encoder_params + decoder_params + head_params
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "heads": head_params,
            "total": total,
            "total_M": total / 1e6,
        }


def build_student(cfg: dict) -> StudentCellViT:
    """Build student model from config dict."""
    student_cfg = cfg["student"]
    return StudentCellViT(
        encoder_name=student_cfg["encoder"],
        pretrained=student_cfg["encoder_pretrained"],
        decoder_channels=student_cfg["decoder_channels"],
        num_classes=student_cfg["heads"]["type_map"],
        tissue_aux=student_cfg.get("tissue_aux", False),
        num_tissue_classes=student_cfg.get("num_tissue_classes", 19),
        hv_tanh=student_cfg.get("hv_tanh", False),
    )
