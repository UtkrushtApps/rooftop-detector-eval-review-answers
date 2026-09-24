"""Hidden grader for rooftop-detector-eval-review.

Runs the candidate's evaluation code against the hidden split and a few
crafted cases. Prints one PASS/FAIL line per check and a final score.

    python grading/grade.py /path/to/candidate/repo
    python grading/grade.py /path/to/candidate/repo --write-expected   # reference repo only
"""

import argparse
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIDDEN = HERE / "hidden"
EXPECTED = HERE / "expected.json"
HIDDEN_THRESHOLD = 0.5


def load_candidate(repo):
    sys.path.insert(0, str(Path(repo) / "src"))
    names = ("io", "matching", "metrics", "thresholds", "cli")
    return {n: importlib.import_module(f"roof_eval.{n}") for n in names}


def hidden_counts(m):
    split = m["io"].load_split(HIDDEN)
    df = m["metrics"].evaluate_split(split["annotations"], split["predictions"],
                                     score_threshold=HIDDEN_THRESHOLD)
    per_class = {r["category"]: {"tp": int(r["tp"]), "fp": int(r["fp"])} for _, r in df.iterrows()}
    return df, per_class


def check_one_to_one(m):
    preds = [{"image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "score": 0.9},
             {"image_id": 1, "category_id": 1, "bbox": [102, 102, 96, 96], "score": 0.8}]
    gts = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100]}]
    rows = m["matching"].match_image(preds, gts, iou_threshold=0.5)
    return sum(bool(r["matched"]) for r in rows) == 1


def check_background_fp(m):
    annotations = {
        "images": {1: {"id": 1}, 2: {"id": 2}},
        "categories": {1: "antenna_mount"},
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50]}],
    }
    preds = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "score": 0.9},
             {"image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 50], "score": 0.9}]
    df = m["metrics"].evaluate_split(annotations, preds, score_threshold=0.5)
    return int(df.iloc[0]["fp"]) == 1


def check_hidden_counts(m, expected):
    _, per_class = hidden_counts(m)
    return all(per_class.get(k) == v for k, v in expected["per_class"].items())


def check_balanced_headline(m, expected):
    df, _ = hidden_counts(m)
    return abs(m["metrics"].headline_score(df) - expected["macro_f1"]) < 1e-6


def check_threshold_not_from_reported_split(m, repo):
    seen = []
    original = m["thresholds"].sweep_thresholds

    def recording_sweep(annotations, *args, **kwargs):
        seen.append(annotations["raw"]["info"]["description"])
        return original(annotations, *args, **kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(Path(repo) / "data" / "validation", root / "validation")
        shutil.copytree(HIDDEN, root / "test")
        m["thresholds"].sweep_thresholds = recording_sweep
        m["cli"].sweep_thresholds = recording_sweep
        try:
            m["cli"].run(root)
        finally:
            m["thresholds"].sweep_thresholds = original
            m["cli"].sweep_thresholds = original
    return bool(seen) and not any("hidden" in s for s in seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--write-expected", action="store_true")
    args = ap.parse_args()
    m = load_candidate(args.repo)

    if args.write_expected:
        df, per_class = hidden_counts(m)
        EXPECTED.write_text(json.dumps(
            {"threshold": HIDDEN_THRESHOLD, "per_class": per_class,
             "macro_f1": m["metrics"].headline_score(df)}, indent=2) + "\n")
        print(f"wrote {EXPECTED}")
        return

    expected = json.loads(EXPECTED.read_text())
    checks = [
        ("each annotated object is matched at most once", lambda: check_one_to_one(m)),
        ("detections on photos with no annotations count as false positives",
         lambda: check_background_fp(m)),
        ("per-class TP/FP on the hidden split match the reference",
         lambda: check_hidden_counts(m, expected)),
        ("headline treats every class equally (macro F1)", lambda: check_balanced_headline(m, expected)),
        ("threshold is never chosen on the reported (held-out) split",
         lambda: check_threshold_not_from_reported_split(m, args.repo)),
    ]
    passed = 0
    for name, fn in checks:
        try:
            ok = fn()
        except Exception as exc:  # a crash in candidate code is a failed check, not a grader crash
            ok = False
            name = f"{name} (error: {exc!r})"
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"score: {passed}/{len(checks)}")


if __name__ == "__main__":
    main()
