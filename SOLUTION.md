# Solution Steps

1. Confirm the data format: COCO annotations and detection results use bboxes as [x, y, width, height], while IoU operates on corner coordinates.

2. Fix bbox conversion in geometry.to_corners so every annotation and prediction is converted from COCO xywh into (x1, y1, x2, y2) before IoU matching.

3. Rewrite single-image matching to process predictions in descending confidence order and allow each ground-truth object to be matched at most once, class-by-class. Duplicate detections on the same object should become false positives after the first match.

4. Keep per-class accounting in evaluate_split, including TP, FP, FN, precision, recall, and F1 for every category in the annotation file. Iterate deterministically over image ids for reproducible output.

5. Change the headline metric from support-weighted F1 to balanced macro F1 over classes with ground-truth support, and keep support-weighted F1 only as a diagnostic so rare classes are not hidden.

6. Expand threshold sweeps to report macro precision/recall and safety-class recall/F1 in addition to the balanced headline, making the precision/recall trade-off auditable.

7. Select the operating threshold only from the validation sweep. Maximize validation macro F1, and in ties prefer better safety-class recall, then better macro recall, then the lower threshold.

8. Update the CLI so it sweeps validation, freezes the selected threshold, and then reports both validation-at-threshold and test-at-threshold summaries. Do not choose the threshold from test performance.

9. Fix false-negative collection to use matched ground-truth indices from the one-to-one matcher instead of category-only bookkeeping, which can be wrong when an image contains multiple objects of the same class.

10. Replace the placeholder failure analysis with a concise explanation of the corrected evaluation artifacts, validation-based operating point, remaining real detector failures, and field-review risks.

