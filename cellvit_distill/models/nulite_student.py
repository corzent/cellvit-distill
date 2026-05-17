"""Wrapper that fits NuLite model class into our train.py interface.

Provides a thin adapter around vendor/NuLite/models/nulite.NuLite so that
our existing train/eval/distill code can use NuLite as the student
without changes to the training loop.

Differences from native NuLite:
- Output dict is renamed to our convention: nuclei_binary_map -> binary,
  nuclei_type_map -> type_map, tissue_types -> tissue_logits (hv_map
  stays the same).
- Adds count_parameters() method matching StudentCellViT's interface.
- Forwards through NuLite's _init_<variant> path based on encoder_name
  ("nulite_fastvit_s12", "nulite_fastvit_sa24", etc.).

Used to answer the architecture-vs-recipe question: train NuLite on our
3-fold protocol and compare to our FastViT-S12 + HoVer-Net decoder
baseline.
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

# Add NuLite repo to sys.path so its `models.nulite` import works.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "NuLite"))


_NULITE_VARIANT_MAP = {
    "nulite_fastvit_s12":  "fastvit_s12",   # NuLite-T (12M)
    "nulite_fastvit_sa24": "fastvit_sa24",  # NuLite-M (24M)
    "nulite_fastvit_sa36": "fastvit_sa36",  # NuLite-H (34M)
    "nulite_fastvit_t8":   "fastvit_t8",
    "nulite_fastvit_t12":  "fastvit_t12",
    "nulite_fastvit_ma36": "fastvit_ma36",
}


class NuLiteStudent(nn.Module):
    """Wrap vendor/NuLite/models/nulite.NuLite into our student interface.

    Output dict keys are renamed to match our convention:
      nuclei_binary_map -> binary
      nuclei_type_map   -> type_map
      hv_map            -> hv_map  (unchanged)
      tissue_types      -> tissue_logits  (when tissue_aux=True)
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes: int = 6,
        num_tissue_classes: int = 19,
        drop_rate: float = 0.0,
        tissue_aux: bool = True,
    ):
        super().__init__()
        if encoder_name not in _NULITE_VARIANT_MAP:
            raise ValueError(
                f"Unknown NuLite encoder {encoder_name!r}; "
                f"choose from {list(_NULITE_VARIANT_MAP.keys())}"
            )
        nulite_variant = _NULITE_VARIANT_MAP[encoder_name]

        from models.nulite import NuLite  # noqa: E402 — sys.path injected above
        self.model = NuLite(
            num_nuclei_classes=num_classes,
            num_tissue_classes=num_tissue_classes,
            vit_structure=nulite_variant,
            drop_rate=drop_rate,
        )
        self.encoder_name = encoder_name
        self.tissue_aux = tissue_aux

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.model(x)
        out = {
            "binary":   raw["nuclei_binary_map"],
            "hv_map":   raw["hv_map"],
            "type_map": raw["nuclei_type_map"],
        }
        if self.tissue_aux and "tissue_types" in raw:
            out["tissue_logits"] = raw["tissue_types"]
        return out

    def count_parameters(self) -> Dict[str, int]:
        # NuLite's internal modules: encoder, classifier_head, decoder,
        # decoder0, np_head, hv_head, tp_head. Group encoder vs the rest
        # (decoders + heads) to mirror StudentCellViT's accounting.
        encoder = sum(
            p.numel() for p in self.model.encoder.parameters() if p.requires_grad
        )
        decoder = sum(
            p.numel() for n, p in self.model.named_parameters()
            if p.requires_grad
            and not n.startswith("encoder.")
            and not n.startswith(("np_head.", "hv_head.", "tp_head.", "classifier_head."))
        )
        heads = sum(
            p.numel() for n, p in self.model.named_parameters()
            if p.requires_grad
            and n.startswith(("np_head.", "hv_head.", "tp_head.", "classifier_head."))
        )
        total = encoder + decoder + heads
        return {
            "encoder": encoder,
            "decoder": decoder,
            "heads":   heads,
            "total":   total,
            "total_M": total / 1e6,
        }
