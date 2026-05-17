"""MoNuSeg 2018 dataset adapter for cross-dataset (zero-shot) evaluation.

MoNuSeg has H&E images at 1000×1000 with polygon XML annotations and a
single nucleus class. Our PanNuke-trained student outputs 5-class type
predictions but only the binary head is meaningful here, so this adapter
returns (image_patch, gt_binary_instance_map) per item — `type_map` is
omitted because no GT type signal exists.

Tiling: 1000×1000 → overlapping 256×256 patches with stride 128 (50%
overlap, 8×8 = 64 patches per image with edge padding). This matches
the input resolution the student saw at train time and lets us reuse
HoVer-watershed post-processing per patch, then stitch by simple
overlap-averaging on the binary probability map and re-extracting
instances at full image scale.

Polygon XML format: ASAP-style ASA AperioImageScope annotations with
<Annotation>/<Region>/<Vertices>/<Vertex X= Y= /> entries. We rasterize
each polygon into a unique integer ID in a 1000×1000 instance mask.

Stain normalization: NOT applied by default. PanNuke and MoNuSeg use
similar H&E staining but MoNuSeg has wider color variance (multiple
organs, multiple scanners). For honest domain-shift numbers, leave
normalization off; for an upper-bound number, add Macenko or Reinhard.

Usage:
    from cellvit_distill.data.monuseg import MoNuSegDataset
    ds = MoNuSegDataset("./datasets/monuseg/test", patch_size=256, stride=128)
    item = ds[0]  # {'image': uint8 (256,256,3), 'instance_map': int32 (256,256),
                  #  'source_image': 'TCGA-...', 'patch_xy': (y, x)}
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def _parse_xml_polygons(xml_path: Path) -> List[List[Tuple[float, float]]]:
    """Return list of polygons; each polygon is a list of (x, y) vertices.

    Supports the standard ASAP / AperioImageScope schema MoNuSeg ships with.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons: List[List[Tuple[float, float]]] = []
    # Two layouts seen in the wild: <Annotations><Annotation><Regions><Region>
    # ... and a flatter <Annotation><Regions><Region>; iterate over Regions
    # wherever found.
    for region in root.iter("Region"):
        verts = []
        for v in region.iter("Vertex"):
            try:
                x = float(v.get("X"))
                y = float(v.get("Y"))
            except (TypeError, ValueError):
                continue
            verts.append((x, y))
        if len(verts) >= 3:
            polygons.append(verts)
    return polygons


def _rasterize_polygons(polygons: List[List[Tuple[float, float]]],
                         size: Tuple[int, int]) -> np.ndarray:
    """Rasterize each polygon into a unique integer ID. size=(H, W).

    Returns (H, W) int32 mask with 0 = background and 1..N = instance IDs.
    """
    H, W = size
    mask = np.zeros((H, W), dtype=np.int32)
    img = Image.new("I", (W, H), 0)  # 32-bit integer mode
    draw = ImageDraw.Draw(img)
    for i, poly in enumerate(polygons, start=1):
        if len(poly) < 3:
            continue
        draw.polygon([(x, y) for x, y in poly], outline=i, fill=i)
    return np.array(img, dtype=np.int32)


def _tile_indices(H: int, W: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """Top-left corners (y, x) covering H×W with given patch and stride.

    Ensures the last row/col fully covers the image by clamping to H-patch.
    """
    ys = list(range(0, max(H - patch, 0) + 1, stride))
    xs = list(range(0, max(W - patch, 0) + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)
    return [(y, x) for y in ys for x in xs]


class MoNuSegDataset(Dataset):
    """Patch-based MoNuSeg dataset.

    Args:
        data_dir: contains images/*.tif (or .tiff) and masks/*.xml
        patch_size: side length of square patch (default 256, matches student)
        stride: tile stride; defaults to patch_size // 2 = 128 (50% overlap)
        load_masks: if False, only return images (useful for inference-only)
        transform: optional Albumentations Compose; if None, no augmentation
    """

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = 256,
        stride: Optional[int] = None,
        load_masks: bool = True,
        transform=None,
    ):
        self.root = Path(data_dir)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Missing {self.images_dir}; run download_monuseg.py first")
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2
        self.load_masks = load_masks
        self.transform = transform

        # Index image files (.tif or .tiff)
        self._image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
        )
        if not self._image_files:
            raise RuntimeError(f"No .tif/.tiff under {self.images_dir}")

        # Pre-compute patch index: list of (image_idx, y, x)
        self._index: List[Tuple[int, int, int]] = []
        for img_idx, img_path in enumerate(self._image_files):
            with Image.open(img_path) as im:
                W, H = im.size
            for y, x in _tile_indices(H, W, patch_size, self.stride):
                self._index.append((img_idx, y, x))

        # Lazy-loaded cache: image_idx -> (image_array, instance_mask)
        self._cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def __len__(self) -> int:
        return len(self._index)

    @property
    def num_images(self) -> int:
        return len(self._image_files)

    def _load(self, img_idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if img_idx in self._cache:
            return self._cache[img_idx]
        img_path = self._image_files[img_idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = None
        if self.load_masks:
            xml_path = self.masks_dir / (img_path.stem + ".xml")
            if xml_path.exists():
                polygons = _parse_xml_polygons(xml_path)
                mask = _rasterize_polygons(polygons, img.shape[:2])
            else:
                mask = np.zeros(img.shape[:2], dtype=np.int32)
        # Cache modestly — full MoNuSeg test set fits in RAM (14 × 3 MB)
        self._cache[img_idx] = (img, mask)
        return img, mask

    def __getitem__(self, idx: int) -> Dict:
        img_idx, y, x = self._index[idx]
        img, mask = self._load(img_idx)
        ps = self.patch_size
        patch_img = img[y:y+ps, x:x+ps]
        out: Dict = {
            "image": patch_img,
            "source_image": self._image_files[img_idx].stem,
            "patch_xy": (y, x),
        }
        if mask is not None:
            out["instance_map"] = mask[y:y+ps, x:x+ps]
        if self.transform is not None:
            out = self.transform(**{k: v for k, v in out.items()
                                    if isinstance(v, np.ndarray)})
            out["source_image"] = self._image_files[img_idx].stem
            out["patch_xy"] = (y, x)
        return out
