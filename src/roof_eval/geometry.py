"""Bounding-box geometry helpers."""

from __future__ import annotations

from typing import Sequence


def to_corners(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a COCO bbox to corner coordinates.

    COCO annotations and COCO-style detection result files store boxes as
    ``[x, y, width, height]``.  The IoU helper below operates on
    ``(x1, y1, x2, y2)``, so all matching code should pass boxes through this
    conversion before computing overlap.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four numbers, got {bbox!r}")
    x, y, w, h = bbox
    x1 = float(x)
    y1 = float(y)
    x2 = x1 + max(0.0, float(w))
    y2 = y1 + max(0.0, float(h))
    return x1, y1, x2, y2


def area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union of two corner boxes (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    union = area(box_a) + area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union
