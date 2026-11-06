# Why taught truth costs on chains: diagnosis on checkpoints

Status: **method written 2026-11-06 before the reads; results appended below.** A diagnostic on the
[multi-step reader v1](multistep-reader-v1-results.md) checkpoints (`pred-one`, `unit-one`, seeds 450–453). No training.
User direction: understand the cost before deciding taught truth's fate.

## The observation

On chains, the one-pass taught-truth reader trails the built-in truth unit by 0.12–0.44 balanced accuracy. Its ranking
is closer (AUROC 0.82–0.91 against 0.96–1.00), but it answers F for most held-out true facts.

## Candidate causes, and the read that separates them

| Cause | If it is the cause | Read |
|---|---|---|
| **A. The decision rule:** the 7-way argmax gives mass to values that are wrong for an individual (U, quantities) | Deciding between T and F alone recovers most of the gap | balanced accuracy with the rule "logit_T > logit_F"; the share of each predicted value on held-out true and false facts |
| **B. Calibration:** held-out true facts are ranked above false ones but fall on the F side of the boundary | A boundary fitted on *trained* statements only (no test labels) recovers most of the gap; held-out positives' margins sit between trained positives and negatives | a logistic fit of the T−F margin on trained rows (global and per relation), applied to held-out rows; margin quantiles for trained/held-out × true/false, for both readers |
| **C. Ranking (the representation):** held-out true facts are not well separated from false ones | Neither A nor B helps; AUROC stays below the unit's | AUROC per test, already measured; the calibrated balanced accuracy is bounded by it |
| **D. The value codes:** T and F codes too close to separate sharply | cos(T, F) high; the temperature low | the geometry of the 7 value codes and the learned temperature |

Tests read: L1 size, L1 ancestor, L2 named, P6 grandparent, P4 order, P3 inverse, P1 inheritance (`default`). The
trained/validation split is recomputed exactly as in the runner (same seed). Fits use trained rows only.

Not separable here, needs training: whether the extra statement groups the taught-truth reader also learns (kind,
anonymous, fill-the-tail, connectives, unknown colours) take capacity from T/F inference. That would need a `pred` run
trained on individual T/F only.
