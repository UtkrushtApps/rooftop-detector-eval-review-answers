"""Behavioral tests for a correct evaluation pipeline."""

from pathlib import Path

import pytest

from roof_eval import io
from roof_eval.geometry import iou
from roof_eval.matching import match_image
from roof_eval.metrics import evaluate_split, headline_score

DATA = Path(__file__).resolve().parents[1] / "data"


def _load(split):
    return io.load_split(DATA / split)


def test_iou_identity_and_disjoint():
    box = [0.0, 0.0, 10.0, 10.0]
    assert iou(box, box) == pytest.approx(1.0)
    assert iou(box, [20.0, 20.0, 30.0, 30.0]) == pytest.approx(0.0)


def test_iou_partial_overlap_reasonable():
    a = [0.0, 0.0, 10.0, 10.0]
    b = [5.0, 0.0, 15.0, 10.0]
    val = iou(a, b)
    assert 0.0 < val < 1.0


def test_each_gt_matched_at_most_once():
    # Two overlapping predictions on the same object must not both count as TP.
    preds = [
        {"category_id": 1, "bbox": [100, 100, 200, 200], "score": 0.9},
        {"category_id": 1, "bbox": [102, 102, 198, 198], "score": 0.8},
    ]
    gts = [{"category_id": 1, "bbox": [100, 100, 200, 200]}]
    rows = match_image(preds, gts, iou_threshold=0.5)
    matched = [r for r in rows if r["matched"]]
    assert len(matched) <= 1


def test_metrics_have_all_classes():
    data = _load("test")
    df = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    assert set(df["category"]) == {"antenna_mount", "rru", "cable_tray", "safety_rail_gap"}


def test_recall_cannot_exceed_one():
    data = _load("test")
    df = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    assert (df["recall"] <= 1.0 + 1e-9).all()
    assert (df["precision"] <= 1.0 + 1e-9).all()


def test_headline_in_range():
    data = _load("test")
    df = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    h = headline_score(df)
    assert 0.0 <= h <= 1.0


def test_rare_class_visible_in_breakdown():
    data = _load("test")
    df = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    rare = df[df["category"] == "safety_rail_gap"]
    assert not rare.empty
    assert int(rare.iloc[0]["gt_count"]) > 0


def test_deterministic_output():
    data = _load("test")
    df1 = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    df2 = evaluate_split(data["annotations"], data["predictions"], score_threshold=0.5)
    assert df1.equals(df2)
