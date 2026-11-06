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

## Results (2026-11-06)

Command: `.venv/bin/python -m experiments.taught_truth_cost_diagnosis` (0.9 s). Artifacts:
`reports/taught-truth-cost-diagnosis{,-runtime}.json`. Mean of 4 seeds; balanced accuracy.

| Test | Taught truth, 7-way argmax | T vs F only | Boundary fitted on trained rows (global / per relation) | Taught-truth AUROC | Unit | Unit AUROC |
|---|---|---|---|---|---|---|
| L1 size | 0.73 | 0.73 | 0.73 / 0.73 | 0.88 | 0.96 | 1.00 |
| L1 ancestor | 0.55 | 0.55 | 0.53 / 0.54 | 0.91 | 0.71 | 1.00 |
| L2 named | 0.78 | 0.78 | 0.77 / 0.78 | 0.86 | 0.90 | 0.96 |
| P6 grandparent | 0.53 | 0.53 | 0.53 / 0.53 | 0.91 | 0.97 | 1.00 |
| P4 order | 0.61 | 0.61 | 0.60 / 0.61 | 0.71 | 0.78 | 0.93 |
| P3 inverse | 0.68 | 0.68 | 0.68 / 0.68 | 0.80 | 0.74 | 0.86 |
| P1 inheritance | 0.84 | 0.85 | 0.85 / 0.85 | 0.93 | 0.88 | 0.96 |

**Where the held-out true facts land** (seed 450; median; the taught-truth margin is logit T − logit F, the unit's is
its logit):

| Test | Taught truth: trained true / held-out true / held-out false / trained false | Unit: same order |
|---|---|---|
| P6 grandparent | +9.9 / **−9.6** / −18.9 / −19.5 | +8.5 / **+3.6** / −10.4 / −19.2 |
| L1 ancestor | +9.5 / **−13.2** / −20.4 / −19.3 | +7.2 / −0.7 / −15.5 / −11.6 |
| L1 size | +10.6 / −0.4 / −21.1 / −11.5 | +8.3 / +7.2 / −14.2 / −8.0 |
| L2 named | +10.1 / +3.8 / −15.6 / −17.6 | +8.6 / +4.5 / −7.1 / −8.8 |

**Value codes:** cos(T, F) 0.37–0.46, temperature scale 19–20 (margins of ±20 are reachable). Other values are almost
never predicted for individual statements (at most 0.02 U).

### Reading

- **A. The decision rule is not the cause.** T against F alone gives the same accuracy as the 7-way argmax. The extra
  values take almost no answers.
- **B. Calibration is not the cause.** A boundary fitted on trained statements, global or per relation, changes nothing.
  Trained true and false statements are far apart (about +10 against −12 to −20), so any boundary between them sits
  near 0. The held-out true facts lie well on the F side of it.
- **C. The cause is how the reader places what it never saw.** Held-out true facts are ranked above held-out false ones
  (AUROC 0.80–0.93), but the taught-truth reader puts them deep on the F side (grandparent −9.6, ancestor −13.2). The
  built-in unit puts them near or above its boundary (+3.6, −0.7). Its ranking is also better (0.86–1.00).

  Novel true statements come out "mostly false" in the taught-truth reader, "mostly true" in the unit. That is a
  difference in generalization, not in reading out.
- **D. The value codes are not the cause.** T and F are moderately apart (cos ≈ 0.4) with a large scale; trained
  statements are separated sharply.

### What this does not settle (needs one training run)

Two differences between the readers remain:
1. the taught-truth reader also learns other statement groups (kind quantities, anonymous defaults, fill-the-tail,
   connectives, unknown colours);
2. it answers by a predicted code compared by cosine, instead of a scalar logit.

A `pred` trained on individual T/F statements only would separate them:
- if it matches the unit, the extra groups are the cost;
- if not, the readout is.

Both are design decisions the user made deliberately, so this is reported, not run.

## Follow-up run: taught-truth reader trained on T/F statements only (user-approved, 2026-11-06)

Command: `.venv/bin/python -m experiments.taught_truth_tf_only` (7.5 min wall, 30 CPU min, 4 workers). The same `pred`
reader, statements, splits, seeds 450–453 and 3,000 steps as `pred-one`, trained only on individual T/F statements
(the unit's diet). Artifacts: `reports/taught-truth-tf-only{,-runtime}.json`; checkpoints in
`runs/taught-truth-tf-only/` (not tracked). Training fit 1.00.

| Test | Taught truth, all statements | Taught truth, T/F only | Built-in unit | AUROC (all / T/F only / unit) | Held-out true margin, T/F only (median) |
|---|---|---|---|---|---|
| L1 size | 0.73 | **0.89** | 0.96 | 0.88 / 0.94 / 1.00 | +7.9 |
| L1 ancestor | 0.55 | 0.65 | 0.71 | 0.91 / 0.73 / 1.00 | −7.8 |
| L2 named | 0.78 | 0.76 | 0.90 | 0.86 / 0.83 / 0.96 | +3.1 |
| P6 grandparent | 0.53 | 0.66 | 0.97 | 0.91 / 0.89 / 1.00 | −2.1 |
| P4 order | 0.61 | 0.68 | 0.78 | 0.71 / 0.77 / 0.93 | −5.3 |
| P3 inverse | 0.68 | 0.73 | 0.74 | 0.80 / 0.79 / 0.86 | −3.2 |
| P1 inheritance | 0.84 | 0.74 | 0.88 | 0.93 / 0.88 / 0.96 | +2.7 |

### Reading: both causes contribute

- **The extra statement groups explain part of the cost.** Without them, held-out true facts move toward or past the
  boundary: grandparent margin −9.6 → −2.1, ancestor −13.2 → −7.8, size −0.4 → +7.9. Balanced accuracy rises on size
  (+0.16), grandparent (+0.13), ancestor (+0.10), order (+0.07) and inverse (+0.05).
- **The readout explains the rest.** Even on the unit's diet, the code-and-cosine reader trails the unit on every chain:
  grandparent 0.66 against 0.97, named 0.76 against 0.90. Its ranking is below the unit's everywhere (AUROC 0.73–0.94
  against 0.86–1.00). The ancestor ranking is lower than with all statements (0.73 against 0.91).
- **The extra groups also help some things.** Inheritance falls without them (0.84 → 0.74), and named relations slightly
  (0.78 → 0.76): kind statements and fill-the-tail rows carry information about classes and names.
- So the cost is not a single bug. Training on the other statement groups pushes unseen facts toward F. Answering by a
  code compared with value codes ranks unseen facts less well than a scalar truth unit. Neither alone closes the gap.
