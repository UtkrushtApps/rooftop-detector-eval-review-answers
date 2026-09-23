"""Readable metric summaries and failure-analysis support."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .io import group_by_image
from .matching import match_image
from .metrics import evaluate_split, headline_score, support_weighted_f1


def build_summary(
    annotations: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    score_threshold: float,
    iou_threshold: float = 0.5,
    split_name: str | None = None,
) -> Dict[str, Any]:
    per_class = evaluate_split(
        annotations, predictions, score_threshold=score_threshold, iou_threshold=iou_threshold
    )
    supported = per_class[per_class["gt_count"] > 0]
    source = supported if not supported.empty else per_class
    return {
        "split_name": split_name,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "headline": headline_score(per_class),
        "headline_definition": "macro F1 over classes with ground-truth support",
        "weighted_f1_diagnostic": support_weighted_f1(per_class),
        "macro_precision": float(source["precision"].mean()) if not source.empty else 0.0,
        "macro_recall": float(source["recall"].mean()) if not source.empty else 0.0,
        "per_class": per_class,
    }


def format_summary(summary: Dict[str, Any]) -> str:
    lines = []
    title = "=== Rooftop Detector Evaluation ==="
    if summary.get("split_name"):
        title = f"=== Rooftop Detector Evaluation: {summary['split_name']} ==="
    lines.append(title)
    lines.append(f"operating score threshold: {summary['score_threshold']}")
    lines.append(f"iou threshold: {summary['iou_threshold']}")
    lines.append(f"headline score ({summary.get('headline_definition', 'macro F1')}): {summary['headline']:.4f}")
    lines.append(f"macro precision: {summary.get('macro_precision', 0.0):.4f}")
    lines.append(f"macro recall: {summary.get('macro_recall', 0.0):.4f}")
    lines.append(f"support-weighted F1 diagnostic: {summary.get('weighted_f1_diagnostic', 0.0):.4f}")
    lines.append("")
    lines.append("per-class:")
    per_class: pd.DataFrame = summary["per_class"]
    for _, row in per_class.iterrows():
        lines.append(
            f"  {row['category']:<18} gt={int(row['gt_count']):<4} "
            f"tp={int(row['tp']):<4} fp={int(row['fp']):<4} fn={int(row['fn']):<4} "
            f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f}"
        )
    return "\n".join(lines)


def collect_false_negatives(
    annotations: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    score_threshold: float,
    iou_threshold: float = 0.5,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Return ground-truth objects that no accepted detection matched."""
    ann_records = annotations["annotations"]
    categories = annotations["categories"]
    gt_by_image = group_by_image(ann_records)
    pred_by_image = group_by_image(predictions)

    misses: List[Dict[str, Any]] = []
    for image_id in sorted(gt_by_image):
        gts = gt_by_image[image_id]
        preds = [
            p
            for p in pred_by_image.get(image_id, [])
            if float(p.get("score", 1.0)) >= score_threshold
        ]
        rows = match_image(preds, gts, iou_threshold=iou_threshold)
        matched_gt_indices = {r["gt_index"] for r in rows if r["matched"] and r["gt_index"] is not None}
        for gt_index, gt in enumerate(gts):
            if gt_index in matched_gt_indices:
                continue
            misses.append(
                {
                    "image_id": image_id,
                    "annotation_id": gt.get("id"),
                    "category": categories.get(gt["category_id"], str(gt["category_id"])),
                    "bbox": gt["bbox"],
                }
            )
            if len(misses) >= limit:
                return misses
    return misses
