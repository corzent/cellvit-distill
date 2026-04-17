"""PanNuke dataset loader for cell segmentation and classification.

PanNuke contains 7904 patches (256x256) from 19 tissue types with ~190K nuclei
annotated across 5 classes: neoplastic, inflammatory, connective, dead, epithelial.
Data is split into 3 folds for cross-validation.

Expected directory structure after download:
    pannuke/
    ├── fold1/
    │   ├── images.npy    # (N, 256, 256, 3) uint8
    │   ├── masks.npy     # (N, 256, 256, 6) - 5 classes + background, instance IDs
    │   └── types.npy     # (N,) tissue type labels
    ├── fold2/
    │   └── ...
    └── fold3/
        └── ...
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import center_of_mass


class PanNukeDataset(Dataset):
    """PanNuke dataset with HoVer-Net style targets.

    For each patch, generates:
        - image: (3, 256, 256) float32 normalized
        - binary_map: (2, 256, 256) nucleus vs background (one-hot)
        - hv_map: (2, 256, 256) horizontal and vertical distance maps
        - type_map: (6, 256, 256) per-pixel class probabilities (one-hot)
        - instance_map: (256, 256) integer instance IDs (for evaluation)
    """

    # PanNuke class names (index 0 = background)
    CLASS_NAMES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]
    NUM_CLASSES = 6

    def __init__(
        self,
        data_dir: str,
        folds: List[int],
        transform: Optional[A.Compose] = None,
        soft_targets_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Root directory containing fold1/, fold2/, fold3/
            folds: List of fold indices to load (1, 2, 3)
            transform: Albumentations transform pipeline
            soft_targets_dir: Directory with pre-computed teacher soft targets
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.soft_targets_dir = Path(soft_targets_dir) if soft_targets_dir else None

        # Load folds with memory-mapped access (avoids loading ~12GB per fold into RAM)
        self._mmap_images = []
        self._mmap_masks = []
        self._tissue_labels = []  # (N,) per fold, loaded eagerly (small)
        self._fold_sizes = []
        self._cumulative_sizes = [0]

        for fold_idx in folds:
            fold_dir = self.data_dir / f"fold{fold_idx}"
            imgs = np.load(fold_dir / "images.npy", mmap_mode="r")  # (N, 256, 256, 3)
            msks = np.load(fold_dir / "masks.npy", mmap_mode="r")    # (N, 256, 256, 6)
            self._mmap_images.append(imgs)
            self._mmap_masks.append(msks)
            types_path = fold_dir / "types.npy"
            if types_path.exists():
                self._tissue_labels.append(np.load(types_path, allow_pickle=True))
            else:
                self._tissue_labels.append(None)
            self._fold_sizes.append(len(imgs))
            self._cumulative_sizes.append(self._cumulative_sizes[-1] + len(imgs))

        # Build tissue name -> index mapping across folds (stable order)
        all_tissue_names = set()
        for labels in self._tissue_labels:
            if labels is not None:
                all_tissue_names.update(str(t) for t in labels)
        self.tissue_name_to_idx = {n: i for i, n in enumerate(sorted(all_tissue_names))}

        self._total_len = self._cumulative_sizes[-1]
        print(f"PanNuke: {self._total_len} patches from folds {folds} (memory-mapped)")

    def __len__(self) -> int:
        return self._total_len

    def _resolve_index(self, idx: int):
        """Map global index to (fold_local_idx, fold_mmap_images, fold_mmap_masks, fold_tissue_labels)."""
        for i, (imgs, msks, tl) in enumerate(
            zip(self._mmap_images, self._mmap_masks, self._tissue_labels)
        ):
            offset = self._cumulative_sizes[i]
            if idx < self._cumulative_sizes[i + 1]:
                local_idx = idx - offset
                return local_idx, imgs, msks, tl
        raise IndexError(f"Index {idx} out of range [0, {self._total_len})")

    def _generate_instance_map(self, masks: np.ndarray) -> np.ndarray:
        """Convert PanNuke class-wise instance masks to a single instance map.

        Args:
            masks: (256, 256, 6) where each channel contains instance IDs for that class

        Returns:
            instance_map: (256, 256) with unique instance IDs across all classes
        """
        instance_map = np.zeros((masks.shape[0], masks.shape[1]), dtype=np.int32)
        instance_id = 0

        # Channels 0-4 are cell classes, channel 5 is background
        for class_idx in range(5):
            class_mask = masks[:, :, class_idx]
            unique_ids = np.unique(class_mask)
            unique_ids = unique_ids[unique_ids > 0]  # skip background (0)

            for uid in unique_ids:
                instance_id += 1
                instance_map[class_mask == uid] = instance_id

        return instance_map

    def _generate_type_map(self, masks: np.ndarray) -> np.ndarray:
        """Generate per-pixel type map from PanNuke masks.

        Args:
            masks: (256, 256, 6) PanNuke format

        Returns:
            type_map: (6, 256, 256) one-hot encoded class map
        """
        h, w = masks.shape[:2]
        type_map = np.zeros((self.NUM_CLASSES, h, w), dtype=np.float32)

        # Background: everywhere that has no cell
        has_cell = np.zeros((h, w), dtype=bool)
        for class_idx in range(5):
            cell_mask = masks[:, :, class_idx] > 0
            has_cell |= cell_mask
            type_map[class_idx + 1] = cell_mask.astype(np.float32)  # classes 1-5

        type_map[0] = (~has_cell).astype(np.float32)  # background
        return type_map

    def _generate_hv_map(self, instance_map: np.ndarray) -> np.ndarray:
        """Generate horizontal and vertical distance maps (HoVer-Net style).

        For each pixel in a nucleus, compute normalized distance to the
        center of mass of that nucleus. Horizontal = x-direction, Vertical = y-direction.
        Values are in [-1, 1], with 0 at the center of mass and ±1 at the boundary.

        Args:
            instance_map: (256, 256) with unique instance IDs

        Returns:
            hv_map: (2, 256, 256) - channel 0 = horizontal, channel 1 = vertical
        """
        h, w = instance_map.shape
        hv_map = np.zeros((2, h, w), dtype=np.float32)

        instance_ids = np.unique(instance_map)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
            inst_mask = instance_map == inst_id
            ys, xs = np.where(inst_mask)

            if len(ys) == 0:
                continue

            # Center of mass
            cy, cx = center_of_mass(inst_mask)

            # Compute distances and normalize to [-1, 1]
            x_dist = xs - cx
            y_dist = ys - cy

            # Normalize by max distance in each direction
            x_max = np.max(np.abs(x_dist)) if np.max(np.abs(x_dist)) > 0 else 1
            y_max = np.max(np.abs(y_dist)) if np.max(np.abs(y_dist)) > 0 else 1

            hv_map[0, ys, xs] = x_dist / x_max  # horizontal
            hv_map[1, ys, xs] = y_dist / y_max  # vertical

        return hv_map

    def _generate_binary_map(self, instance_map: np.ndarray) -> np.ndarray:
        """Generate binary nucleus/background map.

        Returns:
            binary_map: (2, 256, 256) one-hot [background, nucleus]
        """
        nucleus = (instance_map > 0).astype(np.float32)
        background = 1.0 - nucleus
        return np.stack([background, nucleus], axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        local_idx, imgs_mmap, masks_mmap, tissue_labels = self._resolve_index(idx)
        image = np.array(imgs_mmap[local_idx])  # copy from mmap -> (256, 256, 3) uint8
        masks = np.array(masks_mmap[local_idx])  # copy from mmap -> (256, 256, 6)
        tissue_label_idx = (
            self.tissue_name_to_idx[str(tissue_labels[local_idx])]
            if tissue_labels is not None
            else -1
        )

        # Generate targets
        instance_map = self._generate_instance_map(masks)
        binary_map = self._generate_binary_map(instance_map)
        hv_map = self._generate_hv_map(instance_map)
        type_map = self._generate_type_map(masks)

        # Generate per-pixel type class label (needed for augmentation)
        type_class_map = type_map.argmax(axis=0).astype(np.int32)  # (H, W) class indices

        # Load soft targets BEFORE augmentation so they can be spatially transformed
        # together with the image. Without this, flips/rotations misalign soft targets
        # with the augmented image, turning distillation loss into noise.
        soft_data = None
        if self.soft_targets_dir is not None:
            st_path = self.soft_targets_dir / f"{idx}.npz"
            if st_path.exists():
                # Use context manager to avoid leaking file handles across workers.
                with np.load(st_path) as st:
                    soft_data = {
                        "binary": st["binary"].astype(np.float32, copy=True),
                        "hv": st["hv"].astype(np.float32, copy=True),
                        "type": st["type"].astype(np.float32, copy=True),
                    }

        # Apply augmentations
        if self.transform is not None:
            # Convert image to uint8 for albumentations if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            else:
                image_uint8 = image

            # Build masks list: GT masks + soft target channels (same spatial transform)
            masks_list = [instance_map.astype(np.int32), type_class_map]
            if soft_data is not None:
                # Append each soft target channel as a separate (H, W) mask.
                # Spatial transforms (flip, rotate90) are exact on float arrays.
                # Elastic/affine use nearest interpolation — acceptable for logits.
                soft_binary = soft_data["binary"]  # (2, H, W)
                soft_type = soft_data["type"]      # (6, H, W)
                for c in range(soft_binary.shape[0]):
                    masks_list.append(soft_binary[c].astype(np.float32))
                for c in range(soft_type.shape[0]):
                    masks_list.append(soft_type[c].astype(np.float32))
                # soft_hv excluded: flips require sign correction on distance values,
                # and GT HV maps (recomputed from instance_map) are already exact.

            # Transform image + all masks with same spatial ops
            transformed = self.transform(
                image=image_uint8,
                masks=masks_list,
            )
            image = transformed["image"]  # already a tensor from ToTensorV2
            inst_mask = transformed["masks"][0]
            type_mask = transformed["masks"][1]

            instance_map_np = inst_mask.numpy() if isinstance(inst_mask, torch.Tensor) else inst_mask
            type_class_np = type_mask.numpy() if isinstance(type_mask, torch.Tensor) else type_mask

            # Recombine soft target channels after augmentation
            if soft_data is not None:
                aug_masks = transformed["masks"]

                def _to_np(m):
                    return m.numpy() if isinstance(m, torch.Tensor) else m

                soft_data["binary"] = np.stack([_to_np(aug_masks[2 + c]) for c in range(2)])
                soft_data["type"] = np.stack([_to_np(aug_masks[4 + c]) for c in range(6)])
                # soft_data["hv"] stays unchanged — HV distillation skipped (see config)

            # Regenerate maps from spatially-transformed masks
            binary_map = self._generate_binary_map(instance_map_np)
            hv_map = self._generate_hv_map(instance_map_np)

            # Rebuild type_map one-hot from transformed class labels
            h, w = type_class_np.shape
            type_map = np.zeros((self.NUM_CLASSES, h, w), dtype=np.float32)
            for c in range(self.NUM_CLASSES):
                type_map[c] = (type_class_np == c).astype(np.float32)

            instance_map = instance_map_np
        else:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0

        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()

        sample = {
            "image": image,
            "binary_map": torch.from_numpy(binary_map).float(),
            "hv_map": torch.from_numpy(hv_map).float(),
            "type_map": torch.from_numpy(type_map).float(),
            "instance_map": torch.from_numpy(instance_map.astype(np.int64)).long(),
            "tissue_label": torch.tensor(tissue_label_idx, dtype=torch.long),
        }

        # Add spatially-aligned soft targets
        if soft_data is not None:
            sample["soft_binary"] = torch.from_numpy(soft_data["binary"]).float()
            sample["soft_hv"] = torch.from_numpy(soft_data["hv"]).float()
            sample["soft_type"] = torch.from_numpy(soft_data["type"]).float()

        return sample


def get_train_transform(patch_size: int = 256, strong: bool = False) -> A.Compose:
    """Training augmentations for PanNuke.

    Args:
        patch_size: Input patch resolution.
        strong: If True, adds elastic deformation, coarse dropout, and
                heavier color jitter — helps combat overfitting on small
                datasets like PanNuke (only ~5.2K training patches).
    """
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    if strong:
        transforms += [
            # Geometric: elastic deformation simulates tissue variability
            A.ElasticTransform(alpha=80, sigma=10, p=0.3),
            # Affine jitter (small scale + rotation)
            A.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                     scale=(0.9, 1.1), rotate=(-15, 15),
                     border_mode=0, p=0.4),
            # Stronger color jitter to simulate stain variation
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.08, p=0.6),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(p=0.2),
            # Coarse dropout: rectangular cutout regions (regularisation)
            A.CoarseDropout(num_holes_range=(1, 6),
                            hole_height_range=(8, 24),
                            hole_width_range=(8, 24),
                            fill=0, p=0.3),
        ]
    else:
        transforms += [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ]

    transforms += [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    return A.Compose(transforms)


def get_val_transform() -> A.Compose:
    """Validation/test transform (normalize only)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
