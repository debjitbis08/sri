# Number construction v1: relational composition test on Part D checkpoints

Protocol: [number construction v1 spec, "Relational composition test on Part D
checkpoints"](number-construction-v1-spec.md) and its amendment after external review.
Motivation: question (1) of "Relational conflict without algebra" in the
[conceptual instinct notes](conceptual-instinct-notes.md). No training; frozen
end-of-training Part D checkpoints, seeds 100–102. **A pilot analysis.**

- **Command:** `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_composition pilot`
  (37 s, of which 27 s data preparation).
- **Artifacts:** `reports/number-construction-v1-composition-pilot-results.json.gz`, `-runtime.json`.
- **Order of events:** the original protocol was run twice before the external review
  arrived (the second run split off ties). The amendment changed the primary measure,
  strata, baselines and interventions; only the amended run is reported. Its design follows
  the review, not the numbers seen.

**Question.** A start item, a hidden action ("add k" or "remove k", k = 1–3), and a claim
that the result is more, or fewer, than a visible reference. Does applying the learned
successor before the comparison head improve the verdict over ignoring the action, while
keeping accuracy where the action does not change the verdict?

Each checkpoint: 283,500 decisions per action (25 start/reference kind pairs, 30 items per
count), of which 60,000 are critical (the action changes the truth).

Readings are the two varied arms, which have a trained comparison head. In the other three
arms the head is untrained and gives one constant answer (0.44 overall, 0.50 on critical
items), so the test needs it, as expected.

## Addition (primary)

D-varied+rehearsal, means over seeds (D-varied in brackets):

| Verdict | All items | Non-critical | Critical |
| --- | --- | --- | --- |
| Composed: Sᵏ(start), then comparison | **0.80** (0.68) | 0.87 (0.73) | 0.53 (0.49) |
| Comparison-only shortcut (ignores the action) | 0.75 (0.67) | 0.87 (0.75) | 0.30 (0.38) |
| Action-and-claim rule (ignores both quantities) | 0.61 | 0.50 | 1.00 |
| Shown result (a real item of the result's count; a reference, not a bound) | 0.82 (0.71) | 0.85 (0.74) | 0.70 (0.62) |

- **Applying the successor helps over ignoring the action** (0.80 against 0.75) and keeps
  non-critical accuracy (0.87 against 0.87). It nearly matches the shown result (0.82).
  The same holds for items involving withheld forms (0.80 against 0.74) and across senses
  (0.81 against 0.76).
- **But the cleanest crossings fail.** Critical items in the "neither" stratum (the
  reference strictly between start and result, k = 2, 3):

  | k | Composed | Shortcut | Shown result |
  | --- | --- | --- | --- |
  | 2 | 0.53 (seeds 0.41, 0.44, 0.72) | 0.28 | 0.71 |
  | 3 | 0.54 (seeds 0.35, 0.43, 0.84) | 0.20 | 0.78 |

  Composed is at chance on average, with one seed (102) succeeding and two failing. The
  full-set gain comes from the critical items where the reference equals the start
  (composed 0.71, shortcut 0.52).
- **Equality is a separate limitation, as the review predicted.** Where the result equals
  the reference, every verdict is near or below chance, the shown result included (0.48).
  The comparison head was never trained on equal pairs.
- **Action interventions** (on items where the intervened action's verdict differs from the
  truth): the specified action beats the reversed action clearly (composed 0.68, reversed
  0.21) but a wrong step count only weakly (0.69 against 0.59).
- **Does the transformed state keep its count?** Sᵏ(start) read against collection
  prototypes with the *same* head (chance 0.11): taps 0.71 at k = 1 and 0.65 at k = 3;
  collections 0.63 and 0.44; Chinese forms 0.55 and 0.39; Devanagari 0.54 and 0.44;
  English 0.41 and 0.34. Quantity meaning is mostly kept, though it degrades with steps.
  So the crossing failure is not only loss of meaning; the comparison head also misreads
  transformed states, or off-by-one drift falls on the wrong side of the reference. These
  two are not separated here.

## Removal (diagnostic: inverse-successor inference)

The successor matrix is well conditioned in every checkpoint (condition number 1.4–5.6),
so removal was evaluated everywhere. It is inference through the exact inverse of the
learned successor, not a learned removal operation.

| Verdict | All items | Critical, neither, k = 2 | Critical, neither, k = 3 |
| --- | --- | --- | --- |
| Composed | 0.81 (0.73) | 0.68 | 0.77 |
| Shortcut | 0.74 (0.66) | 0.29 | 0.22 |
| Shown result | 0.84 (0.72) | 0.72 | 0.81 |

- Strict crossings work in the backward direction (0.68–0.77, close to the shown result),
  although they fail forward. This asymmetry is **unexplained**. The forward direction is
  the one the successor was trained in.

## Reading

- **Primary question: partly yes.** Applying the learned successor before comparison
  improves hidden-result judgments over ignoring the action (0.80 against 0.75 for
  addition, 0.81 against 0.74 for removal) without losing non-critical accuracy.
- **Not yet established:** forward composition across a strict crossing. On the cleanest
  diagnostic, adding 2 or 3 and comparing with a reference in between is at chance on
  average and unstable across seeds.
- **Equality** is untrained in these checkpoints and caps every method on tie items.
- So the compositional competence that a fast single-pass detector would need to absorb is
  present only in part. Training comparisons on transformed states, multi-step succession
  or equality judgments would be the obvious remedies; none is chosen yet.

## Computation

37 s wall on 11 workers (27 s data preparation, 10 s analysis); 82 worker CPU seconds;
6 s per checkpoint; peak RSS of the largest worker 1.1 GiB. Cost per checkpoint: one
encoding pass over the evaluation items, then O(items × d²) for successor applications and
O(items × d) for comparisons, with d = 16. Selection costs: the two superseded runs
(about 37 s each).
