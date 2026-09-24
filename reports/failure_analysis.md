# Rooftop Detector Evaluation: Failure Analysis

## Trustworthy Results

The original report said 0.986. On the test split, at a threshold chosen on validation, the honest headline (macro F1 across the four classes) is **0.704**: antenna_mount 0.897, rru 0.828, cable_tray 0.769, safety_rail_gap 0.323. Three things in the evaluation inflated the old number, and none of them was the detector getting better:

- Duplicate detections of the same object each counted as a hit, which is why recall read above 1.0 for three classes.
- Photos with no equipment were never evaluated, so the 6 false detections on 18 empty rooftops never reached precision.
- The headline weighted classes by object count, so 21 safety rail gaps barely moved a number dominated by 275 antenna mounts.

## Operating Point

The threshold was swept on validation only, then frozen before touching test. Macro F1 peaks at **0.5** (0.666), which is the point I report. The cost is visible in the same sweep: safety_rail_gap recall falls from 0.29 at 0.3 to 0.18 at 0.5 and to zero above 0.6. If the safety team would rather send more photos to a human than miss a rail gap, 0.3 is the defensible alternative (macro F1 0.632, rail-gap recall 0.29). That is a policy decision for them, and I would not make it silently.

## Where the Detector Actually Fails

- **Safety rail gaps are the real weakness.** 5 of 21 were found on test. None of the 4 at dusk were found, and only 1 of 4 under glare. The detections that do exist are low-confidence (roughly 0.3 to 0.7) and loosely placed, so many fall below the overlap needed to count.
- **Lighting drives the other misses.** RRUs drop to 16/23 at dusk, and cable trays to 4/7 under glare, against about 90% for both in clear conditions.
- **Clutter creates false alarms.** Vents, ladders and reflections produce confident detections on rooftops with no equipment at all.

## Remaining Risks

The detector should not be the only gate for safety rail gaps: at any usable threshold it misses most of them, and it is worst in the dusk and glare photos where a human reviewer also struggles. Rail gaps are rare, so these figures rest on about 20 examples per split and will move as more are labelled. Before field use I would collect more dusk and glare examples, re-evaluate per condition, and route every low-light install photo to manual review.
