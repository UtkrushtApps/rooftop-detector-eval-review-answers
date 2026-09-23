"""Smoke tests: imports, fixture loading, CLI execution."""

from pathlib import Path

import pytest

from roof_eval import cli, io, metrics, reporting, thresholds

DATA = Path(__file__).resolve().parents[1] / "data"


def test_imports():
    assert hasattr(cli, "run")
    assert hasattr(metrics, "evaluate_split")
    assert hasattr(thresholds, "select_operating_point")


def test_fixtures_load():
    for split in ("validation", "test"):
        data = io.load_split(DATA / split)
        assert data["annotations"]["annotations"]
        assert data["predictions"]
        assert data["annotations"]["categories"]


def test_prediction_records_consistent():
    for split in ("validation", "test"):
        data = io.load_split(DATA / split)
        image_ids = set(data["annotations"]["images"].keys())
        cat_ids = set(data["annotations"]["categories"].keys())
        for p in data["predictions"]:
            assert p["image_id"] in image_ids
            assert p["category_id"] in cat_ids
            assert len(p["bbox"]) == 4
            assert 0.0 <= p["score"] <= 1.0


def test_cli_runs():
    out = cli.run(DATA)
    assert isinstance(out, str)
    assert "per-class" in out
