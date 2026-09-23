"""Assign stored detections to annotated objects."""

from __future__ import annotations

from typing import Any, Dict, List

from .geometry import iou, to_corners


def match_image(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Match predictions on a single image against ground-truth objects.

    Matching is class-specific and one-to-one: a ground-truth object can be
    claimed by at most one prediction.  Predictions are processed in descending
    score order, which mirrors the usual detector evaluation convention and
    prevents duplicate detections on the same object from inflating recall.

    Returns a list of result rows, one per prediction in the original input
    order, each with fields: category_id, score, matched, iou.  Additional
    diagnostic fields (prediction_index, gt_index, gt_id) identify the assigned
    pair when a match exists.
    """
    rows_by_prediction_index: Dict[int, Dict[str, Any]] = {}
    matched_gt_indices: set[int] = set()

    # Highest-confidence predictions get first chance to claim each object.
    ordered_predictions = sorted(
        enumerate(predictions),
        key=lambda item: (-float(item[1].get("score", 1.0)), item[0]),
    )

    for pred_index, pred in ordered_predictions:
        pbox = to_corners(pred["bbox"])
        pred_category = pred["category_id"]

        best_any_iou = 0.0
        best_unmatched_iou = 0.0
        best_unmatched_gt_index: int | None = None

        for gt_index, gt in enumerate(ground_truth):
            if gt["category_id"] != pred_category:
                continue
            gbox = to_corners(gt["bbox"])
            overlap = iou(pbox, gbox)
            if overlap > best_any_iou:
                best_any_iou = overlap
            if gt_index in matched_gt_indices:
                continue
            if overlap > best_unmatched_iou:
                best_unmatched_iou = overlap
                best_unmatched_gt_index = gt_index

        matched = (
            best_unmatched_gt_index is not None
            and best_unmatched_iou >= iou_threshold
        )
        gt_id = None
        gt_index_value = None
        reported_iou = best_any_iou
        if matched:
            matched_gt_indices.add(best_unmatched_gt_index)  # type: ignore[arg-type]
            gt = ground_truth[best_unmatched_gt_index]  # type: ignore[index]
            gt_id = gt.get("id")
            gt_index_value = best_unmatched_gt_index
            reported_iou = best_unmatched_iou

        rows_by_prediction_index[pred_index] = {
            "prediction_index": pred_index,
            "category_id": pred_category,
            "score": float(pred.get("score", 1.0)),
            "matched": bool(matched),
            "iou": float(reported_iou),
            "gt_index": gt_index_value,
            "gt_id": gt_id,
        }

    return [rows_by_prediction_index[i] for i in range(len(predictions))]


def count_ground_truth(ground_truth: List[Dict[str, Any]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for gt in ground_truth:
        counts[gt["category_id"]] = counts.get(gt["category_id"], 0) + 1
    return counts
