# Solution Steps

1. Boxes are COCO [x, y, width, height] in both files (see the info block of each JSON); geometry.to_corners already converts them correctly. Nothing to change, but a strong candidate verifies it by hand on a few photos.

2. Count detections on photos that have no annotations. The starter iterates only over photos with ground truth, so every false positive on an empty rooftop is silently dropped and precision is inflated. Iterate over every photo that has annotations or predictions.

3. Rewrite single-image matching to process predictions in descending confidence order and allow each ground-truth object to be matched at most once, class-by-class. Duplicate detections on the same object should become false positives after the first match.

4. Keep per-class accounting in evaluate_split, including TP, FP, FN, precision, recall, and F1 for every category in the annotation file. Iterate deterministically over image ids for reproducible output.

5. Change the headline metric from support-weighted F1 to balanced macro F1 over classes with ground-truth support, and keep support-weighted F1 only as a diagnostic so rare classes are not hidden.

6. Expand threshold sweeps to report macro precision/recall and safety-class recall/F1 in addition to the balanced headline, making the precision/recall trade-off auditable.

7. Select the operating threshold only from the validation sweep. Maximize validation macro F1, and in ties prefer better safety-class recall, then better macro recall, then the lower threshold.

8. Update the CLI so it sweeps validation, freezes the selected threshold, and then reports both validation-at-threshold and test-at-threshold summaries. Do not choose the threshold from test performance.

9. Fix false-negative collection to use matched ground-truth indices from the one-to-one matcher instead of category-only bookkeeping, which can be wrong when an image contains multiple objects of the same class.

10. Replace the placeholder failure analysis with a concise explanation of the corrected evaluation artifacts, validation-based operating point, remaining real detector failures, and field-review risks.


## Grading

The hidden split lives in grading/hidden (100 photos, sites S-3xx, never shipped to candidates). Run `python grading/grade.py <candidate repo>`. Reference solution 5/5, untouched starter 0/5. Regenerate all data with `python tools/make_data.py --starter <starter repo> --answers .` then `python grading/grade.py . --write-expected`.
