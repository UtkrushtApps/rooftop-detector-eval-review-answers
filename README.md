## Task Overview

A telecom operator is deciding whether a rooftop equipment detector can take over part of the site-survey review, which today is done entirely by people looking at installation photos. The detector has already scored 300 field photos, and this repository turns its stored outputs and the survey team's annotations into the evaluation report that decision rests on. The report says the detector is close to perfect, yet the field reviewers who check real installs keep finding problems it did not catch. One of the equipment classes is a safety defect, so a report that overstates the detector is a safety problem, not a cosmetic one.

## Objectives

- Make the evaluation report reflect how the detector really performs.
- Represent every equipment class fairly in the results, however rare it is.
- Choose an operating threshold you could defend to the safety team.
- Explain where the detector fails in the field and what that means for reviewers.

## Helpful Tips

- Run the evaluation once and read the whole report before you change anything.
- Work a few photos through by hand and compare them with what the report says.
- Think about what correct means for a report that decides whether a safety check can be automated, not just whether the code runs without errors.
- Whatever you change should still be checkable by hand afterwards.

## How to Verify

- Every figure in the report should match what you get by working the same photos through by hand.
- The results for the rarest equipment class should be as easy to find and judge as those for the most common one.
- The chosen threshold should be one you would still pick if the reported photos were replaced with new ones.
- The failure analysis should let a reviewer tell real detector weaknesses apart from problems in the evaluation itself.
- The tests under `tests/` run with `python -m pytest` once the package is installed.
