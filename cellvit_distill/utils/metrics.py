"""Evaluation metrics for cell segmentation: PQ, mPQ, bPQ, F1-detection.

Based on PanNuke evaluation protocol.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, List


def _match_instances(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], List[float]]:
    """Match predicted instances to ground truth using IoU and Hungarian algorithm.

    Returns:
        matched_pairs: List of (gt_id, pred_id) tuples
        unmatched_gt: List of unmatched GT instance IDs
        unmatched_pred: List of unmatched predicted instance IDs
        matched_ious: IoU for each matched pair
    """
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids > 0]
    pred_ids = np.unique(pred_map)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return [], [], [], []
    if len(gt_ids) == 0:
        return [], [], list(pred_ids), []
    if len(pred_ids) == 0:
        return [], list(gt_ids), [], []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_ids), len(pred_ids)))

    for i, gt_id in enumerate(gt_ids):
        gt_mask = gt_map == gt_id
        for j, pred_id in enumerate(pred_ids):
            pred_mask = pred_map == pred_id
            intersection = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            if union > 0:
                iou_matrix[i, j] = intersection / union

    # Hungarian matching (maximize IoU = minimize negative IoU)
    cost_matrix = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    matched_ious = []
    matched_gt_set = set()
    matched_pred_set = set()

    for r, c in zip(row_ind, col_ind):
        if iou_matrix[r, c] >= iou_threshold:
            matched_pairs.append((gt_ids[r], pred_ids[c]))
            matched_ious.append(iou_matrix[r, c])
            matched_gt_set.add(gt_ids[r])
            matched_pred_set.add(pred_ids[c])

    unmatched_gt = [gid for gid in gt_ids if gid not in matched_gt_set]
    unmatched_pred = [pid for pid in pred_ids if pid not in matched_pred_set]

    return matched_pairs, unmatched_gt, unmatched_pred, matched_ious


def panoptic_quality(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Panoptic Quality = DQ × SQ.

    Args:
        pred_map: (H, W) predicted instance map
        gt_map: (H, W) ground truth instance map

    Returns:
        Dict with PQ, DQ, SQ values
    """
    matched, unmatched_gt, unmatched_pred, ious = _match_instances(
        pred_map, gt_map, iou_threshold
    )

    tp = len(matched)
    fp = len(unmatched_pred)
    fn = len(unmatched_gt)

    if tp == 0:
        return {"PQ": 0.0, "DQ": 0.0, "SQ": 0.0, "TP": 0, "FP": fp, "FN": fn}

    sq = np.mean(ious)
    dq = (2 * tp) / (2 * tp + fp + fn)
    pq = dq * sq

    return {"PQ": pq, "DQ": dq, "SQ": sq, "TP": tp, "FP": fp, "FN": fn}


def compute_binary_pq(
    pred_maps: List[np.ndarray],
    gt_maps: List[np.ndarray],
) -> float:
    """Compute binary Panoptic Quality (ignoring cell types)."""
    pqs = []
    for pred, gt in zip(pred_maps, gt_maps):
        result = panoptic_quality(pred, gt)
        pqs.append(result["PQ"])
    return np.mean(pqs) if pqs else 0.0


def compute_multi_class_pq(
    pred_maps: List[np.ndarray],
    gt_maps: List[np.ndarray],
    pred_type_maps: List[np.ndarray],
    gt_type_maps: List[np.ndarray],
    num_classes: int = 5,
) -> Dict[str, float]:
    """Compute multi-class Panoptic Quality (mPQ) using the CellViT/NuLite protocol.

    For each image, PQ is computed per cell class. When an image has no GT
    instances of a class, that class contributes nan (not 0) for that image.
    Image-level mPQ is np.nanmean over classes; dataset-level mPQ is np.nanmean
    of image-level mPQ. Per-class mPQ is np.nanmean over images.

    This matches:
      mPQ = nanmean_over_images( nanmean_over_classes( PQ[image, class] ) )

    The earlier implementation incorrectly treated absent classes as PQ=0,
    which drastically underestimates rare classes (e.g., Dead ~1% of images)
    and inflates the apparent gap to published numbers.
    """
    per_image_class_pq = []  # list of (num_classes,) arrays, nan where GT empty

    for pred_inst, gt_inst, pred_type, gt_type in zip(
        pred_maps, gt_maps, pred_type_maps, gt_type_maps
    ):
        image_pqs = np.full(num_classes, np.nan, dtype=np.float64)

        for cls_idx, cls in enumerate(range(1, num_classes + 1)):
            pred_cls = np.zeros_like(pred_inst)
            gt_cls = np.zeros_like(gt_inst)

            for inst_id in np.unique(pred_inst):
                if inst_id == 0:
                    continue
                mask = pred_inst == inst_id
                inst_type = pred_type[mask]
                if len(inst_type) > 0:
                    majority_class = np.bincount(
                        inst_type.astype(int), minlength=num_classes + 1
                    ).argmax()
                    if majority_class == cls:
                        pred_cls[mask] = inst_id

            for inst_id in np.unique(gt_inst):
                if inst_id == 0:
                    continue
                mask = gt_inst == inst_id
                inst_type = gt_type[mask]
                if len(inst_type) > 0:
                    majority_class = np.bincount(
                        inst_type.astype(int), minlength=num_classes + 1
                    ).argmax()
                    if majority_class == cls:
                        gt_cls[mask] = inst_id

            # Skip images where GT has no instances of this class (CellViT convention)
            if not np.any(gt_cls):
                continue

            result = panoptic_quality(pred_cls, gt_cls)
            image_pqs[cls_idx] = result["PQ"]

        per_image_class_pq.append(image_pqs)

    per_image_class_pq = np.array(per_image_class_pq)  # (N, num_classes)

    with np.errstate(invalid="ignore"):
        per_image_mpq = np.nanmean(per_image_class_pq, axis=1)  # (N,)
        mpq = np.nanmean(per_image_mpq)
        per_class_mpq = np.nanmean(per_image_class_pq, axis=0)  # (num_classes,)

    return {
        "mPQ": float(mpq) if not np.isnan(mpq) else 0.0,
        **{
            f"PQ_class_{c}": float(per_class_mpq[i]) if not np.isnan(per_class_mpq[i]) else 0.0
            for i, c in enumerate(range(1, num_classes + 1))
        },
    }


def f1_detection(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """Compute F1 detection score."""
    result = panoptic_quality(pred_map, gt_map, iou_threshold)
    return result["DQ"]  # DQ is equivalent to F1 for detection


def compute_all_metrics(
    pred_maps: List[np.ndarray],
    gt_maps: List[np.ndarray],
    pred_type_maps: List[np.ndarray],
    gt_type_maps: List[np.ndarray],
    num_classes: int = 5,
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    bpq = compute_binary_pq(pred_maps, gt_maps)
    mpq_results = compute_multi_class_pq(
        pred_maps, gt_maps, pred_type_maps, gt_type_maps, num_classes
    )

    f1_scores = []
    for pred, gt in zip(pred_maps, gt_maps):
        f1_scores.append(f1_detection(pred, gt))

    return {
        "bPQ": bpq,
        **mpq_results,
        "F1_detection": np.mean(f1_scores) if f1_scores else 0.0,
    }
