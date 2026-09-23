"""Confidence-threshold sweeps and operating-point selection."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .metrics import evaluate_split, headline_score, support_weighted_f1


def sweep_thresholds(
    annotations: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    thresholds: List[float] | None = None,
    iou_threshold: float = 0.5,
) -> pd.DataFrame:
    """Evaluate one split across a range of score thresholds.

    The ``headline`` column is balanced macro F1.  Additional columns make the
    precision/recall trade-off and the rare safety class visible during
    threshold selection.
    """
    if thresholds is None:
        thresholds = [round(0.1 * i, 2) for i in range(1, 10)]

    rows = []
    for t in thresholds:
        per_class = evaluate_split(
            annotations, predictions, score_threshold=t, iou_threshold=iou_threshold
        )
        supported = per_class[per_class["gt_count"] > 0]
        metric_source = supported if not supported.empty else per_class
        safety = per_class[per_class["category"] == "safety_rail_gap"]
        safety_row = safety.iloc[0] if not safety.empty else None
        rows.append(
            {
                "threshold": float(t),
                "headline": headline_score(per_class),
                "macro_precision": float(metric_source["precision"].mean()) if not metric_source.empty else 0.0,
                "macro_recall": float(metric_source["recall"].mean()) if not metric_source.empty else 0.0,
                "min_class_recall": float(metric_source["recall"].min()) if not metric_source.empty else 0.0,
                "weighted_f1_diagnostic": support_weighted_f1(per_class),
                "safety_rail_gap_f1": float(safety_row["f1"]) if safety_row is not None else 0.0,
                "safety_rail_gap_recall": float(safety_row["recall"]) if safety_row is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_operating_point(sweep: pd.DataFrame) -> float:
    """Pick an operating threshold from validation evidence.

    The caller is responsible for passing a validation-set sweep, not a test-set
    sweep.  We maximize balanced macro F1.  If several thresholds are tied up to
    floating-point tolerance, we prefer the one with better safety-class recall,
    then better macro recall, then the lower threshold; this avoids choosing an
    unnecessarily aggressive cutoff when the validation evidence does not show a
    real F1 gain.
    """
    if sweep.empty:
        return 0.5

    best_headline = float(sweep["headline"].max())
    candidates = sweep[sweep["headline"] >= best_headline - 1e-12].copy()

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if "safety_rail_gap_recall" in candidates.columns:
        sort_columns.append("safety_rail_gap_recall")
        ascending.append(False)
    if "macro_recall" in candidates.columns:
        sort_columns.append("macro_recall")
        ascending.append(False)
    sort_columns.append("threshold")
    ascending.append(True)

    best = candidates.sort_values(sort_columns, ascending=ascending).iloc[0]
    return float(best["threshold"])
