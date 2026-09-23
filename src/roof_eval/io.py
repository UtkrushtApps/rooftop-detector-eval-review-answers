"""Loading and light normalization of annotation and prediction records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_annotations(path: str | Path) -> Dict[str, Any]:
    """Load a COCO-style annotation file.

    Returns a dict with keys: images, categories, annotations.
    """
    data = _read_json(path)
    images = {img["id"]: img for img in data.get("images", [])}
    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    annotations: List[Dict[str, Any]] = data.get("annotations", [])
    return {
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "raw": data,
    }


def load_predictions(path: str | Path) -> List[Dict[str, Any]]:
    """Load precomputed detector outputs.

    Each record has: image_id, category_id, bbox, score.  The fixtures use the
    same COCO bbox convention as the annotations: [x, y, width, height].
    """
    data = _read_json(path)
    if isinstance(data, dict):
        return data.get("predictions", [])
    return data


def group_by_image(records: List[Dict[str, Any]], key: str = "image_id") -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(r[key], []).append(r)
    return grouped


def load_split(split_dir: str | Path) -> Dict[str, Any]:
    split_dir = Path(split_dir)
    ann = load_annotations(split_dir / "annotations.json")
    preds = load_predictions(split_dir / "predictions.json")
    return {"annotations": ann, "predictions": preds}
