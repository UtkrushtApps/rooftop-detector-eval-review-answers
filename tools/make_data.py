"""Regenerate the task's stored detector outputs and annotations (deterministic).

Writes validation + test into the candidate repo layout and a hidden split for
grading. Boxes are COCO [x, y, width, height] in both files.

    python tools/make_data.py --starter ../rooftop-detector-eval-review --answers .
"""

import argparse
import json
import random
from pathlib import Path

CATEGORIES = [
    {"id": 1, "name": "antenna_mount"},
    {"id": 2, "name": "rru"},
    {"id": 3, "name": "cable_tray"},
    {"id": 4, "name": "safety_rail_gap"},
]
SIZES = [(1280, 960), (1920, 1080)]
CONDITIONS = ["clear", "clear", "clear", "overcast", "overcast", "glare", "dusk"]
INSTALL_TYPES = ["macro", "macro", "small_cell"]
# class id -> (objects per photo range, box side range px, base detection probability)
CLASS_SPEC = {
    1: ((1, 3), (140, 320), 0.95),
    2: ((0, 2), (80, 180), 0.92),
    3: ((0, 1), (200, 420), 0.88),
    4: ((0, 1), (28, 64), 0.72),
}
RARE_PROB = 0.14   # share of photos that contain a safety_rail_gap
EMPTY_PROB = 0.12  # share of photos with no annotated equipment at all


def jitter(box, rng, scale):
    x, y, w, h = box
    return [
        round(x + rng.gauss(0, scale * w), 1),
        round(y + rng.gauss(0, scale * h), 1),
        round(max(8, w + rng.gauss(0, scale * w)), 1),
        round(max(8, h + rng.gauss(0, scale * h)), 1),
    ]


def detect_prob(cid, condition, side):
    p = CLASS_SPEC[cid][2]
    if condition == "glare" and cid in (3, 4):
        p -= 0.35
    if condition == "dusk" and cid in (2, 4):
        p -= 0.25
    if cid == 4 and side < 40:
        p -= 0.15
    return max(0.05, p)


def make_split(name, n_images, site_prefix, id_offset, rng):
    images, anns, preds = [], [], []
    ann_id = id_offset * 10
    for i in range(n_images):
        img_id = id_offset + i + 1
        w, h = rng.choice(SIZES)
        cond = rng.choice(CONDITIONS)
        images.append({
            "id": img_id, "file_name": f"{name}_{i + 1:04d}.jpg", "width": w, "height": h,
            "site": f"{site_prefix}{rng.randint(1, 40):02d}", "condition": cond,
            "install_type": rng.choice(INSTALL_TYPES),
        })
        empty = rng.random() < EMPTY_PROB
        for cid, ((lo, hi), (smin, smax), _) in CLASS_SPEC.items():
            if empty:
                break
            count = (1 if rng.random() < RARE_PROB else 0) if cid == 4 else rng.randint(lo, hi)
            for _ in range(count):
                side = rng.uniform(smin, smax)
                bw, bh = side * rng.uniform(0.7, 1.3), side * rng.uniform(0.7, 1.3)
                box = [round(rng.uniform(0, w - bw), 1), round(rng.uniform(0, h - bh), 1),
                       round(bw, 1), round(bh, 1)]
                ann_id += 1
                anns.append({"id": ann_id, "image_id": img_id, "category_id": cid, "bbox": box,
                             "area": round(bw * bh, 1), "iscrowd": 0})
                if rng.random() < detect_prob(cid, cond, side):
                    loose = 0.12 if cid == 4 else 0.05
                    score = rng.uniform(0.28, 0.68) if cid == 4 else rng.uniform(0.55, 0.99)
                    preds.append({"image_id": img_id, "category_id": cid,
                                  "bbox": jitter(box, rng, loose), "score": round(score, 3)})
                    if cid != 4 and rng.random() < 0.22:  # detection NMS did not suppress
                        preds.append({"image_id": img_id, "category_id": cid,
                                      "bbox": jitter(box, rng, 0.06),
                                      "score": round(max(0.2, score - rng.uniform(0.05, 0.25)), 3)})
                elif cid == 3 and rng.random() < 0.5:  # tray mistaken for an rru
                    preds.append({"image_id": img_id, "category_id": 2,
                                  "bbox": jitter(box, rng, 0.05),
                                  "score": round(rng.uniform(0.35, 0.7), 3)})
        n_fp = rng.choice([0, 0, 1, 1, 2]) + (1 if empty else 0)
        for _ in range(n_fp):  # rooftop clutter: vents, ladders, reflections
            cid = rng.choice([1, 1, 2, 3, 4])
            side = rng.uniform(*CLASS_SPEC[cid][1])
            preds.append({"image_id": img_id, "category_id": cid,
                          "bbox": [round(rng.uniform(0, w - side), 1),
                                   round(rng.uniform(0, h - side), 1),
                                   round(side, 1), round(side * rng.uniform(0.7, 1.3), 1)],
                          "score": round(rng.uniform(0.15, 0.62), 3)})
    ann_doc = {"info": {"description": f"Rooftop installation {name} split", "bbox_format": "coco_xywh"},
               "categories": CATEGORIES, "images": images, "annotations": anns}
    pred_doc = {"info": {"model": "rooftop-det v2.3", "bbox_format": "coco_xywh"}, "predictions": preds}
    return ann_doc, pred_doc


def write(docs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations.json").write_text(json.dumps(docs[0], indent=1) + "\n")
    (out_dir / "predictions.json").write_text(json.dumps(docs[1], indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starter", required=True)
    ap.add_argument("--answers", required=True)
    args = ap.parse_args()
    rng = random.Random(20260923)
    val = make_split("val", 150, "S-1", 1000, rng)
    test = make_split("test", 150, "S-2", 2000, rng)
    hidden = make_split("hidden", 100, "S-3", 3000, rng)
    for root in (Path(args.starter), Path(args.answers)):
        write(val, root / "data" / "validation")
        write(test, root / "data" / "test")
    write(hidden, Path(args.answers) / "grading" / "hidden")


if __name__ == "__main__":
    main()
