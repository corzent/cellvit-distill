"""MoNuSeg 2018 dataset adapter — backed by HuggingFace `RationAI/MoNuSeg`.

Why HF: official challenge data is gated behind grand-challenge.org login
and public mirrors rot. The RationAI mirror is CC-BY-NC-SA-4.0, ~96 MB,
already pre-rasterized to per-nucleus binary masks (so no XML polygon
parsing). One-time download lands in ~/.cache/huggingface (or $HF_HOME).

Schema (from the dataset card):
    patient    : str (TCGA case id)
    image      : RGB PIL Image, 1000×1000
    instances  : list[1-bit PIL Image], one per nucleus
    tissue     : class label (0=Unknown, 1=Breast, ..., 7=Stomach)

We compose `instances` into a single (H, W) int32 instance map where
0=background and 1..N=nucleus IDs. Then tile each 1000×1000 image into
overlapping 256×256 patches for our 256-trained student.

Usage:
    from cellvit_distill.data.monuseg import MoNuSegDataset
    ds = MoNuSegDataset(split="test", patch_size=256, stride=128)
    item = ds[0]  # {'image': uint8 (256,256,3),
                  #  'instance_map': int32 (256,256),
                  #  'source_image': 'TCGA-...', 'patch_xy': (y, x),
                  #  'tissue': int}
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


_HF_REPO = "RationAI/MoNuSeg"


def _tile_indices(H: int, W: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """Top-left corners (y, x) covering H×W; last row/col clamped to H-patch."""
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs]


def _compose_instances(binary_masks) -> np.ndarray:
    """Stack list of binary masks into (H, W) int32 instance map.

    Each input mask is mode-1 PIL Image. Pixels equal to 1 in the i-th mask
    are assigned instance ID (i+1) in the output. Background = 0.

    If two masks overlap (rare in MoNuSeg annotations), the later mask wins
    — same convention as the standard MoNuSeg eval scripts use.
    """
    if not binary_masks:
        return np.zeros((1000, 1000), dtype=np.int32)
    first = np.array(binary_masks[0], dtype=bool)
    H, W = first.shape
    out = np.zeros((H, W), dtype=np.int32)
    for i, m in enumerate(binary_masks, start=1):
        arr = np.array(m, dtype=bool)
        out[arr] = i
    return out


class MoNuSegDataset(Dataset):
    """Patch-based MoNuSeg dataset, HF-backed.

    Args:
        split: "train" (37 images) | "test" (14 images).
        patch_size: side length of square patch (default 256).
        stride: tile stride; defaults to patch_size // 2.
        cache_dir: HF cache directory; defaults to env $HF_HOME or
            ~/.cache/huggingface.
        load_masks: if False, only image patches are returned.
        transform: optional Albumentations Compose; if None, no augmentation.
    """

    def __init__(
        self,
        split: str = "test",
        patch_size: int = 256,
        stride: Optional[int] = None,
        cache_dir: Optional[str] = None,
        load_masks: bool = True,
        transform=None,
    ):
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "datasets library not installed. Run: "
                "uv pip install datasets pillow"
            ) from e

        self.split = split
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.load_masks = load_masks
        self.transform = transform

        self._ds = load_dataset(_HF_REPO, split=split, cache_dir=cache_dir)

        # Pre-decode + pre-compose all instance maps once (small dataset).
        self._cache: List[Tuple[np.ndarray, Optional[np.ndarray], str, int]] = []
        for row in self._ds:
            img = np.array(row["image"].convert("RGB"), dtype=np.uint8)
            inst = _compose_instances(row["instances"]) if load_masks else None
            self._cache.append((img, inst, row.get("patient", ""),
                                int(row.get("tissue", 0))))

        # Pre-compute patch index: (image_idx, y, x)
        self._index: List[Tuple[int, int, int]] = []
        for img_idx, (img, _, _, _) in enumerate(self._cache):
            H, W = img.shape[:2]
            for y, x in _tile_indices(H, W, patch_size, self.stride):
                self._index.append((img_idx, y, x))

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_images(self) -> int:
        return len(self._cache)

    def _load(self, img_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], str, int]:
        """Returns (image_array, instance_map, patient_id, tissue_label)."""
        return self._cache[img_idx]

    def image_names(self) -> List[str]:
        return [t[2] for t in self._cache]

    def __getitem__(self, idx: int) -> Dict:
        img_idx, y, x = self._index[idx]
        img, inst, patient, tissue = self._cache[img_idx]
        ps = self.patch_size
        patch_img = img[y:y+ps, x:x+ps]
        out: Dict = {
            "image": patch_img,
            "source_image": patient,
            "patch_xy": (y, x),
            "tissue": tissue,
        }
        if inst is not None:
            out["instance_map"] = inst[y:y+ps, x:x+ps]
        if self.transform is not None:
            transformed = self.transform(**{k: v for k, v in out.items()
                                            if isinstance(v, np.ndarray)})
            out.update(transformed)
        return out
