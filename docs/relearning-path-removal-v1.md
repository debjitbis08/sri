# Removing one loss path from the applied update: 20-update diagnostic

Status: **diagnostic, 2026-11-04, seeds 300–303. Not a contract and not a remedy.** It runs the intervention proposed in the [margin decomposition](relearning-margins-v1.md), as revised before the run.

Code: `experiments/relearning_path_removal.py`, with tables from `experiments/relearning_path_removal_summary.py`. Report: `reports/relearning-path-removal-v1.json.gz`.

**The question:** does selectively removing this path's contribution to the applied update protect the decisions attributed to it, while leaving useful acquisition intact?

## Arms

All arms start from the captured pre-relearning state and run 20 updates on identical batches and draws. The path is form rows' sequence-prediction gradient through the candidate collections.

| Arm | Change |
|---|---|
| `unchanged` | The original relearning. Its reads equal stage A's `unchanged` exactly (largest difference 0 over all measures at 0, 10 and 20). |
| `form-cand-stop` | Ordinary stop-gradient of the path: form rows' candidates detached in the sequence loss, with clipping and Adam applied to the resulting gradient. |
| `form-cand-removed` | The path removed from the applied update only. On this arm's own trajectory, the clip factor and Adam's second moment come from the full gradient, and the first moment receives the clipped full gradient minus the clipped path gradient. |

Every arm keeps the context rows' sequence gradients on both sides, form rows' prediction path and every other loss.

**Reads** at 0, 5, 10, 15 and 20:
- the interpreter;
- memory present, with actual retrieval;
- English matching;
- sequence prediction and composition, including the forms' own sequence prediction.

## Results

Means over seeds at updates 10 / 20; the value at update 0 is in parentheses. The expected values are the margin decomposition's approximate linear expectations.

| Read (at 0) | `unchanged` | `form-cand-stop` | `form-cand-removed` | Expected |
|---|---|---|---|---|
| Collections → taps, fresh (0.89) | 0.53 / 0.26 | 0.75 / 0.69 | 0.77 / 0.65 | 0.85 / 0.75 |
| Collections → taps, stored (0.94) | 0.54 / 0.26 | 0.84 / 0.77 | 0.86 / 0.74 | 0.92 / 0.80 |
| Taps → collections (0.99) | 0.59 / 0.50 | 0.81 / 0.87 | 0.83 / 0.88 | 0.91 / 0.91 |
| Memory A, symbol → taps (0.76) | 0.43 / 0.19 | 0.68 / 0.61 | 0.70 / 0.59 | 0.75 / 0.64 |
| Memory B, symbol → taps (0.81) | 0.45 / 0.20 | 0.72 / 0.65 | 0.74 / 0.61 | 0.80 / 0.69 |
| Memory A, taps → symbol (0.81) | 0.51 / 0.41 | 0.72 / 0.76 | 0.73 / 0.77 | — |
| Sequence, collections up 1 (0.92) | 0.71 / 0.52 | 0.89 / 0.80 | 0.90 / 0.81 | 0.89 / 0.92 |
| Sequence, collections down 1 (0.91) | 0.72 / 0.43 | 0.84 / 0.64 | 0.85 / 0.66 | 0.88 / 0.82 |
| Sequence, taps up 1 (0.94) | 0.78 / 0.64 | 0.91 / 0.86 | 0.91 / 0.87 | 0.94 / 0.95 |
| Composition, collections up 2 (0.71) | 0.46 / 0.44 | 0.69 / 0.53 | 0.70 / 0.56 | 0.67 / 0.74 |
| Composition, collections up 3 (0.62) | 0.38 / 0.32 | 0.62 / 0.45 | 0.62 / 0.49 | 0.59 / 0.69 |
| Composition, taps up 2 (0.76) | 0.80 / 0.53 | 0.82 / 0.64 | 0.82 / 0.66 | 0.76 / 0.74 |
| **English matching (0.15)** | 0.18 / **0.23** (0.12–0.32) | 0.17 / 0.21 | 0.18 / **0.20** (0.13–0.23) | — |
| **Sequence, English forms up 1 (0.16)** | 0.17 / 0.22 | 0.18 / 0.21 | 0.16 / 0.24 | — |
| Sequence, English forms down 1 (0.13) | 0.21 / 0.20 | 0.17 / 0.20 | 0.19 / 0.17 | — |
| Sequence, Chinese forms up 1 (0.06) | 0.14 / 0.26 | 0.14 / 0.33 | 0.13 / 0.34 | — |
| Sequence, Devanagari forms up 1 (0.59) | 0.61 / 0.53 | 0.56 / 0.57 | 0.57 / 0.54 | — |
| Collection spacing (26.7) | 14.6 / 8.9 | 23.5 / 22.9 | 23.8 / 23.1 | — |
| Same-count c–t distance / spacing (0.28) | 0.86 / 1.46 | 0.35 / 0.37 | 0.34 / 0.36 | — |

**Clip factor,** mean over seeds at updates 0, 10 and 19:
- `unchanged`: 0.012, 0.037, 0.085;
- `form-cand-stop`: 0.014, 0.021, 0.024;
- `form-cand-removed`: 0.012, 0.022, 0.027.

The removed path is 23–40% of the clipped full gradient's norm.

## What this shows

1. **Removing this path protects the decisions attributed to it, partly.**
   - Over 20 updates, collections → taps holds at 0.65 against 0.26, and memory-present symbol → taps at 0.59–0.61 against 0.19–0.20.
   - The trained taps → collections holds at 0.88 against 0.50, and the code space keeps its spacing and alignment.
   - Collection sequence prediction and composition are protected through update 10, then decline by 20. They remain above `unchanged`.
   - The protection is not complete: collections → taps still falls from 0.89 to 0.65.
2. **No clear acquisition penalty is observed within 20 updates.**
   - English matching at 20 is 0.20 against 0.23 (seed ranges 0.13–0.23 and 0.12–0.32).
   - English forms' sequence prediction is 0.24 against 0.22.
   - Chinese and Devanagari forms' sequence prediction are equal or better.
   - Twenty updates are too few to judge acquisition fully: English matching is still rising in every arm.
3. **The direction was correct, but protection was already below the numerical expectations at update 10, and further below at 20.**
   - At 10, the interpreter and memory reads are 0.05–0.08 below expectation, and sequence and composition are within 0.03 or above it.
   - At 20 the shortfall grows: 0.03–0.10 on the interpreter and memory reads, and 0.08–0.20 on sequence prediction and composition (largest on composition and descending prediction).
   - The linear subtraction does not capture how the remaining paths act on the changed trajectory. That was its stated limitation.
4. **The way the path is removed barely matters under these two rules.** The ordinary stop-gradient and the held-scale removal agree within 0.05 on every read at every point, the one exception being taps composition at update 15 (0.08), in seed means.
   - With the path removed, the two rules' clip factors are nearly the same (0.021–0.027 against 0.022–0.027 from update 10).
   - So the shortfall from expectation is not the difference between these two rules. This comparison does not test other step rules, such as replaying the original run's recorded denominators.

## Answer to the question

**Yes, partly, under this update rule.** Removing form rows' sequence-prediction contribution through its candidates protects about two-thirds of the interpreter's and memory's early loss at 20 updates:
- collections → taps loses 0.24 instead of 0.63;
- memory A loses 0.17 instead of 0.57.

It keeps the code space, with no clear acquisition penalty observed within 20 updates on English matching or form sequence prediction. English acquisition is still weak at this point, so long-term equivalence is not established. Collection composition is protected early but not through 20 updates.

This supports a causal role for the path in the early lost decisions. It is a diagnostic, not a remedy:
- the window is 20 updates;
- stage A's all-rows stop-gradient weakened composition at 200;
- the remaining loss is unexplained.

## What remains open

Follow-up: [attribution on the stop trajectory, updates 10–20](relearning-margins-stop-v1.md).

- **The remaining early loss on this arm's trajectory.** The decomposition attributes smaller shares to form rows' prediction path (the predictor, and taps composition), and to form matching through the comparator. Whether those, or effects of the changed trajectory, account for it would take the same margin decomposition rerun on this arm's trajectory.
- **Beyond 20 updates.** Composition already declines between 10 and 20.

## Computation

- **GPU** (GTX 1650), three workers, float32, deterministic algorithms.
- **Per seed:** capture 156–205 s; each arm's 20 updates with 5 read points, 4–5 s.
- **Totals:** 7 min 7 s wall, 970 s aggregate worker CPU (mostly the capture), peak RSS 3.7 GB.
- **Cost of the held-scale removal:** one extra gradient evaluation per update (the path's), plus a manual Adam step. O(P) time and storage for P parameters.
- **Selection and tuning:** none.
