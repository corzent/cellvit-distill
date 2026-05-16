"""Post-processing: convert model outputs to instance maps via watershed on HV maps.

Pipeline (following HoVer-Net):
1. Threshold binary prediction to get nucleus mask
2. Compute Sobel gradients of HV maps
3. Apply watershed using gradient energy landscape
4. Assign cell type to each instance by majority vote from type_map
"""

import multiprocessing as mp
import multiprocessing.pool  # noqa: F401 — needed for mp.pool.Pool annotation
import os
from typing import List, Tuple

import numpy as np
from scipy.ndimage import label as scipy_label, binary_fill_holes
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def _compute_hv_gradient(hv_map: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude from horizontal/vertical maps.

    Args:
        hv_map: (2, H, W) horizontal and vertical distance maps

    Returns:
        gradient: (H, W) gradient magnitude (high at boundaries)
    """
    # Sobel-like gradient for each direction
    h_map = hv_map[0]  # horizontal
    v_map = hv_map[1]  # vertical

    # Gradient in x and y for horizontal map
    h_grad_x = np.gradient(h_map, axis=1)
    h_grad_y = np.gradient(h_map, axis=0)

    # Gradient in x and y for vertical map
    v_grad_x = np.gradient(v_map, axis=1)
    v_grad_y = np.gradient(v_map, axis=0)

    # Combined gradient magnitude
    gradient = np.sqrt(h_grad_x**2 + h_grad_y**2 + v_grad_x**2 + v_grad_y**2)

    return gradient


def post_process_predictions(
    binary_prob: np.ndarray,
    hv_map: np.ndarray,
    type_probs: np.ndarray,
    binary_threshold: float = 0.5,
    min_object_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw model outputs to instance segmentation + type map.

    Args:
        binary_prob: (H, W) nucleus probability [0, 1]
        hv_map: (2, H, W) horizontal/vertical distance maps
        type_probs: (C, H, W) class probabilities
        binary_threshold: Threshold for nucleus detection
        min_object_size: Minimum instance size in pixels

    Returns:
        instance_map: (H, W) integer instance IDs
        type_map: (H, W) per-pixel class labels (0 = background)
    """
    H, W = binary_prob.shape

    # Step 1: Binary mask
    binary_mask = binary_prob > binary_threshold
    binary_mask = binary_fill_holes(binary_mask)

    if binary_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.int32), np.zeros((H, W), dtype=np.int32)

    # Step 2: Compute energy landscape from HV gradients
    gradient = _compute_hv_gradient(hv_map)

    # Invert gradient for watershed (minima at nuclei centers)
    # Smooth to reduce noise
    from scipy.ndimage import gaussian_filter
    gradient_smooth = gaussian_filter(gradient, sigma=1.0)

    # Energy: high gradient = boundary, low = center
    energy = gradient_smooth.copy()
    energy[~binary_mask] = 0

    # Step 3: Find markers (local minima in energy = nuclei centers)
    # Use distance transform for more robust markers
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(binary_mask)

    # Local maxima in distance transform = nuclei centers
    from skimage.feature import peak_local_max
    coords = peak_local_max(
        distance,
        min_distance=3,
        threshold_abs=2,
        labels=binary_mask.astype(int),
    )

    if len(coords) == 0:
        # Fallback: connected components
        instance_map, _ = scipy_label(binary_mask)
    else:
        # Create marker image
        markers = np.zeros((H, W), dtype=np.int32)
        for i, (y, x) in enumerate(coords, start=1):
            markers[y, x] = i

        # Step 4: Watershed
        instance_map = watershed(
            gradient_smooth,
            markers=markers,
            mask=binary_mask,
        )

    # Remove small objects
    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        if (instance_map == inst_id).sum() < min_object_size:
            instance_map[instance_map == inst_id] = 0

    # Re-label consecutively
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    new_map = np.zeros_like(instance_map)
    for new_id, old_id in enumerate(unique_ids, start=1):
        new_map[instance_map == old_id] = new_id
    instance_map = new_map

    # Step 5: Assign cell types by majority vote per instance
    type_pred = type_probs.argmax(axis=0)  # (H, W) class labels
    type_map = np.zeros((H, W), dtype=np.int32)

    for inst_id in np.unique(instance_map):
        if inst_id == 0:
            continue
        mask = instance_map == inst_id
        inst_types = type_pred[mask]
        if len(inst_types) > 0:
            # Exclude background (class 0) from majority vote
            non_bg = inst_types[inst_types > 0]
            if len(non_bg) > 0:
                majority = np.bincount(non_bg).argmax()
            else:
                majority = np.bincount(inst_types).argmax()
            type_map[mask] = majority

    return instance_map, type_map


# ============================================================
# Parallel batch post-processing
# ============================================================

def _process_one(args):
    """Module-level worker for Pool.map — must be picklable."""
    binary_prob, hv_map, type_probs = args
    return post_process_predictions(binary_prob, hv_map, type_probs)


def _worker_init():
    """Pin each worker to a single OpenMP thread to avoid oversubscription
    with the per-worker scipy/skimage internals."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def make_postprocess_pool(n_workers: int) -> mp.pool.Pool:
    """Create a multiprocessing Pool for parallel post-processing.

    Uses the 'spawn' start method because the parent process has already
    initialized CUDA contexts; forking after CUDA init can hang or crash
    the children.
    """
    ctx = mp.get_context("spawn")
    return ctx.Pool(processes=n_workers, initializer=_worker_init)


def post_process_batch_parallel(
    binary_batch: np.ndarray,
    hv_batch: np.ndarray,
    type_batch: np.ndarray,
    pool: mp.pool.Pool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Run post_process_predictions in parallel across the batch dim.

    Args:
        binary_batch: (B, H, W) fp32 nucleus probabilities
        hv_batch: (B, 2, H, W) fp32 HV maps
        type_batch: (B, C, H, W) fp32 per-class probabilities
        pool: pool returned by make_postprocess_pool()

    Returns: list of B (instance_map, type_map) tuples, in input order.
    """
    B = binary_batch.shape[0]
    args = [(binary_batch[i], hv_batch[i], type_batch[i]) for i in range(B)]
    return pool.map(_process_one, args)
