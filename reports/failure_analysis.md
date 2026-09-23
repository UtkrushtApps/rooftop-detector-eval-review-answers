# Rooftop Detector Evaluation — Failure Analysis

## Trustworthy Results

The trustworthy headline is now the **test-set macro F1 at the threshold selected on validation**, with the full per-class table reported beside it. This differs from the earlier report in three important ways: duplicate detections can no longer claim the same ground-truth object, COCO boxes are converted consistently before IoU matching, and the headline is balanced across classes instead of being dominated by common objects. The support-weighted F1 is still printed only as a diagnostic, because it can hide poor performance on the rare safety-relevant `safety_rail_gap` class.

## Operating Point

The operating confidence threshold is chosen from the validation sweep, then frozen before evaluating the test split. Among validation ties, the selector prefers better `safety_rail_gap` recall and macro recall before choosing the lower threshold. This is a defensible operating point because it uses held-out validation evidence for the precision/recall trade-off and avoids tuning the threshold on test performance.

## Where the Detector Actually Fails

The remaining detector weaknesses are the per-class false negatives and false positives shown in the corrected test summary. In particular, any misses on `safety_rail_gap` should be treated as real review risk rather than an averaging artifact, because the corrected report keeps that class visible. Duplicate high-IoU boxes are now counted as false positives after the first match, so they represent a real non-maximum-suppression or duplicate-reporting issue rather than extra recall.

## Remaining Risks

Field reviewers should not rely on the detector as a sole safety gate. The stored evaluation covers only the provided validation and test fixtures, so performance may degrade on different roof materials, lighting, camera angles, occlusions, small objects, unusual antenna layouts, or sites whose annotation quality differs from these fixtures. The rare safety class remains the highest-risk class to monitor, and production rollout should include human review, drift checks, and periodic re-evaluation on newly collected field examples.
