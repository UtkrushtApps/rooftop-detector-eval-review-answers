"""Class-level and aggregate metric computation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .io import group_by_image
from .matching import count_ground_truth, match_image


def evaluate_split(
    annotations: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Produce per-class precision/recall/F1 at a given operating threshold.

    The returned rows are intentionally per-class, not just pooled counts, so
    rare but safety-relevant classes remain visible in every report.
    """
    ann_records = annotations["annotations"]
    categories = annotations["categories"]

    gt_by_image = group_by_image(ann_records)
    pred_by_image = group_by_image(predictions)

    total_gt = count_ground_truth(ann_records)

    tp: Dict[int, int] = {c: 0 for c in categories}
    fp: Dict[int, int] = {c: 0 for c in categories}

    all_image_ids = set(gt_by_image) | set(pred_by_image)
    for image_id in sorted(all_image_ids):
        preds = [
            p
            for p in pred_by_image.get(image_id, [])
            if float(p.get("score", 1.0)) >= score_threshold
        ]
        gts = gt_by_image.get(image_id, [])
        rows = match_image(preds, gts, iou_threshold=iou_threshold)
        for row in rows:
            cid = row["category_id"]
            if row["matched"]:
                tp[cid] = tp.get(cid, 0) + 1
            else:
                fp[cid] = fp.get(cid, 0) + 1

    records = []
    for cid, name in categories.items():
        c_tp = tp.get(cid, 0)
        c_fp = fp.get(cid, 0)
        c_gt = total_gt.get(cid, 0)
        c_fn = max(0, c_gt - c_tp)
        precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        recall = c_tp / c_gt if c_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        records.append(
            {
                "category_id": cid,
                "category": name,
                "gt_count": c_gt,
                "tp": c_tp,
                "fp": c_fp,
                "fn": c_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    df = pd.DataFrame(records).sort_values("category_id").reset_index(drop=True)
    return df


def headline_score(per_class: pd.DataFrame) -> float:
    """Roll up per-class F1 into a balanced headline number.

    The headline is macro F1 over classes with ground-truth support.  This is
    deliberately not weighted by the number of objects: support-weighted F1 can
    make a common easy class dominate the summary and hide poor performance on a
    rare safety-relevant class such as ``safety_rail_gap``.
    """
    if per_class.empty:
        return 0.0

    supported = per_class[per_class["gt_count"] > 0]
    source = supported if not supported.empty else per_class
    f1 = source["f1"].to_numpy(dtype=float)
    if len(f1) == 0:
        return 0.0
    return float(np.mean(f1))


def support_weighted_f1(per_class: pd.DataFrame) -> float:
    """Return support-weighted F1 for diagnostics, not for the headline."""
    if per_class.empty:
        return 0.0
    weights = per_class["gt_count"].to_numpy(dtype=float)
    f1 = per_class["f1"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        return float(np.mean(f1)) if len(f1) else 0.0
    return float(np.average(f1, weights=weights))
