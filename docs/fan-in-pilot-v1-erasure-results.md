# Fan-in pressure pilot v1, erasure, moving, sparsity and rate: results

Design, contrasts and predictions: [proposal, "Erasure, moving, sparsity and rate"](fan-in-pressure-proposal.md#erasure-moving-sparsity-and-rate-written-before-running),
written before the run. Follows the [reset results](fan-in-pilot-v1-reset-results.md) and
the review of them ([after-reset analysis](rigl-after-reset-analysis.md),
[literature notes](forget-relearn-iterated-learning-notes.md)). Run 2026-10-30, seeds
230–239 (paired with the earlier runs), D-varied+rehearsal. **A pilot.**

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot erasure` (79 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot erasure` (56 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-erasure-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-erasure-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-erasure/` (not tracked).
- **Exactness:** the rerun arms (`dense`, random, random rewire) reproduce the reset run
  exactly in all 30 runs.
- The runner was committed in `77c67ce`. The runtime file records `ddc3de1`, because
  another session committed documentation during the run, and `dirty: true`, because of
  that session's untracked files. The runner's hash is recorded.

## Summary

**Disrupting a sparse reader's connection is what matters, whether by zeroing it or by
moving it. Dense readers do not benefit.**

- **Erasure alone suffices.** Resetting connections to zero in place, without moving them,
  beats random wiring on taps → withheld Chinese (+0.26) and English comparison (+0.16).
  It is equivalent to random rewiring on English comparison and somewhat below it on
  taps → withheld Chinese (−0.13, inconclusive).
- **Moving alone also suffices, against my prediction.** Random moves that keep their weight
  beat random wiring on both measures (+0.44, +0.10). On taps → withheld Chinese they are
  the strongest arm (0.56). They are worse than rewire on English comparison (−0.08) and
  Chinese comparison (−0.23). They leave **no dead units**.
- **Sparsity is needed.** Dense readers with the same per-connection reset rate gain
  nothing on transfer (+0.07 inconclusive, +0.03 equivalent), yet lose taught Chinese at
  its own block (−0.24), the same cost the sparse arms pay.
- **The rate saturates.** Half the rate loses some of the gain (English comparison −0.07
  against rewire). Twice the rate adds nothing on the two measures (+0.02, +0.03,
  inconclusive). It gives the most consistent English comparison across seeds
  (0.73–0.79) and the best probe order on Chinese forms (0.87).
- **Count becomes readable from single coordinates of form codes.** A count map fitted on
  one coordinate of collection codes orders Chinese forms at 0.90 on average over all 16
  coordinates in the rewire arm, against 0.61 for random wiring and 0.59 for dense.

## Pre-declared readings

On taps → withheld Chinese forms and English comparison.

| Reading | Rule | Outcome |
| --- | --- | --- |
| Erasure alone suffices | In place better than random on both | **Yes:** +0.26 [+0.05, +0.46]; +0.16 [+0.11, +0.22] |
| Moving adds to erasure | In place worse than rewire on either | **No:** −0.13 [−0.35, +0.09] inc; −0.01 [−0.03, −0.00] equ |
| Moving alone suffices | Keep-weight better than random on both | **Yes:** +0.44 [+0.24, +0.64]; +0.10 [+0.07, +0.13] |
| Erasure is needed for moves | Keep-weight worse than rewire on either | **Yes, on English:** −0.08 [−0.14, −0.02]; taps → Chinese +0.06 [−0.20, +0.31] inc |
| Sparsity is needed | Dense in place not better than dense on either | **Yes:** +0.07 [−0.04, +0.18] inc; +0.03 [+0.01, +0.05] equ |
| Erasure also helps dense readers | Dense in place better than dense on both | **No** |
| More erasure helps | Double better than rewire on either | **No:** +0.02 [−0.15, +0.19] inc; +0.03 [−0.01, +0.06] inc |
| Less erasure loses | Half worse than rewire on either | **Yes, on English:** −0.07 [−0.09, −0.04]; taps → Chinese −0.10 [−0.36, +0.16] inc |
| An intermediate rate is best | Half and double both worse than rewire on either | **No** |

Devanagari comparison, reported alongside: in place and keep-weight are both better than
random (+0.08, +0.07) and inconclusive against rewire (−0.03, −0.05); dense in place is
inconclusive against dense; double is equivalent to rewire, half is inconclusive.

## Results

Means over ten seeds. Outcome against dense: paired 95% t-interval rule.

| Measure | Dense | Random | Rewire | In place | Move, keep weight | Dense in place | Rewire ½ | Rewire 2× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Transfer** | | | | | | | | |
| Chinese withheld, own block | 0.28 | 0.35 inc | 0.34 inc | 0.37 inc | 0.38 inc | 0.24 inc | 0.33 inc | 0.28 inc |
| Devanagari withheld, own block | 0.20 | 0.19 inc | 0.40 bet | 0.40 bet | 0.41 bet | 0.13 inc | 0.40 bet | 0.36 bet |
| taps → Chinese form, withheld | 0.13 | 0.12 inc | **0.51 bet** | 0.38 bet | **0.56 bet** | 0.20 inc | 0.41 bet | **0.53 bet** |
| taps → Devanagari form, withheld | 0.32 | 0.32 inc | **0.61 bet** | 0.43 inc | **0.64 bet** | 0.35 inc | 0.48 inc | **0.63 bet** |
| form → taps, Chinese withheld | 0.31 | 0.20 inc | 0.54 inc | 0.58 bet | **0.71 bet** | 0.41 inc | 0.58 bet | **0.69 bet** |
| Comparison, Chinese forms | 0.57 | 0.64 inc | **0.98 bet** | 0.87 bet | 0.75 bet | 0.64 inc | 0.90 bet | **0.97 bet** |
| Comparison, Devanagari forms | 0.64 | 0.68 inc | **0.80 bet** | 0.77 bet | 0.75 bet | 0.66 inc | 0.76 bet | **0.82 bet** |
| Comparison, English forms | 0.49 | 0.56 bet | **0.74 bet** | **0.72 bet** | 0.66 bet | 0.52 equ | 0.67 bet | **0.76 bet** |
| **Taught** | | | | | | | | |
| Chinese paired, own block | 0.64 | 0.62 inc | 0.45 wor | 0.46 wor | 0.47 wor | 0.40 wor | 0.45 wor | 0.43 wor |
| taps → collection | 0.64 | 0.68 inc | 0.61 inc | 0.66 inc | 0.62 inc | 0.63 inc | 0.63 inc | 0.60 inc |
| Comparison, collections | 0.97 | 0.97 equ | 0.96 equ | 0.97 equ | 0.97 equ | 0.97 equ | 0.97 equ | 0.96 equ |
| Succession pairwise, collections | 0.91 | 0.92 equ | 0.90 equ | 0.91 equ | 0.90 equ | 0.92 equ | 0.91 equ | 0.88 inc |
| **Other** | | | | | | | | |
| Chinese paired, end | 0.62 | 0.63 inc | 0.55 wor | 0.58 inc | 0.58 inc | 0.62 inc | 0.57 inc | 0.53 wor |
| English, end | 0.39 | 0.40 inc | 0.28 wor | 0.30 wor | 0.33 inc | 0.36 inc | 0.31 wor | 0.26 wor |
| Judgment revision: isolated, withheld | 0.23 | 0.30 inc | 0.41 bet | 0.43 bet | 0.43 bet | 0.25 inc | 0.40 bet | 0.42 bet |
| Judgment revision: revised, withheld | 0.59 | 0.61 inc | 0.57 inc | 0.62 inc | 0.60 inc | 0.61 inc | 0.59 inc | 0.57 inc |
| Next-judgment coherence AUROC, Chinese | 0.93 | 0.93 equ | 0.94 inc | 0.93 equ | 0.95 inc | 0.93 equ | 0.94 inc | 0.94 equ |
| Next-judgment coherence AUROC, English | 0.81 | 0.82 inc | 0.72 wor | 0.71 wor | 0.80 inc | 0.78 inc | 0.77 inc | 0.70 wor |
| Thai, 4 shots, after 200 updates | 0.19 | 0.18 equ | 0.17 inc | 0.18 equ | 0.17 equ | 0.16 inc | 0.17 equ | 0.16 inc |
| Probe order, Chinese forms | 0.57 | 0.53 inc | 0.77 bet | 0.63 inc | 0.67 inc | 0.58 inc | 0.66 inc | 0.87 bet |
| Probe order, Devanagari forms | 0.65 | 0.61 inc | 0.74 bet | 0.69 inc | 0.71 bet | 0.63 inc | 0.69 inc | 0.75 bet |
| Probe order, English forms | 0.50 | 0.51 equ | 0.68 bet | 0.63 bet | 0.64 bet | 0.51 inc | 0.61 bet | 0.71 bet |

Arm names in the artifacts: `dense`, `read-q16-untied`, `read-q16-untied-rewire`,
`-inplace`, `-rewirecarry`, `dense-inplace`, `read-q16-untied-rewire-half`, `-rewire-double`.

### Declared contrasts

Mean paired difference [95% t-interval], outcome.

| Measure | In place − random | In place − rewire | Keep − random | Keep − rewire | Dense in place − dense | ½ − rewire | 2× − rewire | ½ − random | 2× − random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taps → Chinese form, withheld | +0.26 [+0.05, +0.46] bet | −0.13 [−0.35, +0.09] inc | +0.44 [+0.24, +0.64] bet | +0.06 [−0.20, +0.31] inc | +0.07 [−0.04, +0.18] inc | −0.10 [−0.36, +0.16] inc | +0.02 [−0.15, +0.19] inc | +0.28 [+0.13, +0.44] bet | +0.41 [+0.21, +0.60] bet |
| Comparison, English forms | +0.16 [+0.11, +0.22] bet | −0.01 [−0.03, −0.00] equ | +0.10 [+0.07, +0.13] bet | −0.08 [−0.14, −0.02] wor | +0.03 [+0.01, +0.05] equ | −0.07 [−0.09, −0.04] wor | +0.03 [−0.01, +0.06] inc | +0.11 [+0.06, +0.16] bet | +0.21 [+0.18, +0.23] bet |
| Comparison, Devanagari forms | +0.08 [+0.03, +0.13] bet | −0.03 [−0.06, −0.01] inc | +0.07 [+0.03, +0.10] bet | −0.05 [−0.08, −0.02] inc | +0.02 [−0.02, +0.07] inc | −0.04 [−0.07, −0.02] inc | +0.02 [+0.00, +0.04] equ | +0.07 [+0.03, +0.12] bet | +0.13 [+0.10, +0.16] bet |
| Comparison, Chinese forms | +0.24 [+0.09, +0.38] bet | −0.10 [−0.21, +0.00] inc | +0.11 [−0.08, +0.30] inc | −0.23 [−0.42, −0.03] wor | +0.08 [−0.01, +0.17] inc | −0.07 [−0.14, −0.01] wor | −0.00 [−0.01, +0.00] equ | +0.26 [+0.11, +0.41] bet | +0.33 [+0.20, +0.47] bet |
| form → taps, Chinese withheld | +0.38 [+0.18, +0.57] bet | +0.04 [−0.21, +0.30] inc | +0.50 [+0.29, +0.71] bet | +0.17 [−0.05, +0.38] inc | +0.10 [−0.14, +0.34] inc | +0.04 [−0.21, +0.28] inc | +0.15 [−0.07, +0.37] inc | +0.37 [+0.13, +0.61] bet | +0.48 [+0.33, +0.64] bet |
| Chinese paired, own block | −0.16 [−0.25, −0.07] wor | +0.01 [−0.07, +0.09] inc | −0.15 [−0.21, −0.10] wor | +0.02 [−0.04, +0.08] inc | −0.24 [−0.31, −0.17] wor | +0.00 [−0.11, +0.11] inc | −0.01 [−0.12, +0.09] inc | −0.17 [−0.31, −0.03] wor | −0.19 [−0.27, −0.10] wor |
| English, end | −0.10 [−0.14, −0.07] wor | +0.02 [−0.02, +0.06] inc | −0.07 [−0.13, −0.02] wor | +0.05 [+0.01, +0.09] bet | −0.02 [−0.07, +0.03] inc | +0.03 [+0.01, +0.06] inc | −0.02 [−0.05, +0.01] inc | −0.09 [−0.13, −0.06] wor | −0.15 [−0.19, −0.10] wor |

Every contrast and measure is in the comparison artifact.

### Per seed (230–239)

| Arm | taps → withheld Chinese | English comparison |
| --- | --- | --- |
| In place | 0.37 0.52 0.72 0.22 0.00 0.32 0.62 0.45 0.00 0.58 | 0.73 0.77 0.77 0.72 0.69 0.74 0.75 0.76 0.54 0.73 |
| Move, keep weight | 0.20 0.48 0.38 0.92 0.12 0.90 0.53 0.87 0.45 0.78 | 0.74 0.75 0.72 0.66 0.67 0.67 0.57 0.61 0.62 0.55 |
| Dense in place | 0.03 0.42 0.33 0.02 0.05 0.03 0.08 0.93 0.00 0.10 | 0.53 0.50 0.54 0.51 0.50 0.53 0.52 0.57 0.50 0.53 |
| Rewire ½ | 0.13 0.70 0.02 0.38 0.00 0.53 0.48 0.85 0.32 0.65 | 0.62 0.77 0.68 0.66 0.66 0.63 0.73 0.72 0.55 0.69 |
| Rewire 2× | 0.57 0.73 0.40 0.73 0.28 0.83 0.70 0.83 0.00 0.20 | 0.78 0.79 0.76 0.76 0.76 0.77 0.77 0.76 0.73 0.75 |

The per-seed values for dense and random are in the
[fixed-coordinate results](fan-in-pilot-v1-fixed-results.md#per-seed-230239), and for
rewire in the [reset results](fan-in-pilot-v1-reset-results.md#per-seed-230239); the
reruns are identical.

### Single-coordinate probe

For each of the 16 code coordinates: an affine count map fitted on that coordinate of
collection codes only, applied unchanged to taps and forms. Means over all 16
coordinates, then over seeds; no coordinate is selected.

| | Dense | Random | Rewire | In place | Keep weight | Dense in place | ½ | 2× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Collection R² | 0.76 | 0.76 | 0.89 | 0.81 | 0.89 | 0.81 | 0.82 | 0.91 |
| Coordinates with R² ≥ 0.5 (of 16) | 13.7 | 13.3 | 15.5 | 14.3 | 15.5 | 14.9 | 14.6 | 15.9 |
| Order, collections | 0.88 | 0.89 | 0.95 | 0.91 | 0.95 | 0.91 | 0.92 | 0.96 |
| Order, taps | 0.71 | 0.74 | 0.95 | 0.81 | 0.99 | 0.83 | 0.91 | 0.98 |
| Order, English forms | 0.52 | 0.54 | 0.71 | 0.65 | 0.64 | 0.55 | 0.64 | 0.75 |
| Order, Chinese forms | 0.59 | 0.61 | 0.90 | 0.76 | 0.70 | 0.60 | 0.77 | 0.92 |
| Order, Devanagari forms | 0.60 | 0.61 | 0.77 | 0.69 | 0.73 | 0.61 | 0.70 | 0.79 |

- Count is already spread across most coordinates of collection codes in every arm
  (13–16 of 16 with R² ≥ 0.5). What differs is whether form codes carry it on the same
  coordinates: single-coordinate order on forms tracks transfer across arms.
- Dense in place raises collection R² (0.81) without raising form order (0.55–0.61):
  collections become more one-dimensional, forms do not join them.

### Readers and code

| | Dense | Random | Rewire | In place | Keep weight | Dense in place | ½ | 2× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dead units, `g` / `P` | 0 / 0 | 0 / 0 | 0.06 / 0.42 | 0.09 / 0.44 | 0 / 0 | 0.01 / 0.04 | 0.07 / 0.32 | 0.10 / 0.45 |
| Live `g` connections on the most-read dimension | — | 0.06 | 0.11 | 0.07 | 0.10 | — | 0.11 | 0.11 |
| Collection variance on the first direction | 0.88 | 0.91 | 0.98 | 0.95 | 0.96 | 0.89 | 0.94 | 0.99 |
| Code participation ratio (collections) | 1.28 | 1.21 | 1.04 | 1.10 | 1.08 | 1.25 | 1.14 | 1.01 |
| English forms' offset off the axis | 1.73 | 3.41 | 1.19 | 3.16 | 1.04 | 2.18 | 1.59 | 0.67 |
| Chinese forms' offset off the axis | 2.84 | 6.06 | 1.71 | 5.10 | 2.64 | 3.59 | 2.49 | 1.18 |
| Devanagari forms' offset off the axis | 1.00 | 2.10 | 0.76 | 2.13 | 0.80 | 1.13 | 1.11 | 0.52 |

Resets or moves per run (first layer, `g` / `P`): rewire 2,038 / 1,334; in place 2,047 /
1,316; keep weight 2,038 / 1,334 (the same draws as rewire); dense in place 32,850 /
21,161 (16 times as many connections); half 1,019 / 664; double 4,103 / 2,666. No arm
concentrates its readers: at most 11% of live `g` connections share a dimension.

- **Dead units are not needed.** Random moves that keep their weight leave no dead unit
  and give the largest taps → withheld Chinese gain. With the silenced arm of the reset
  run, this rules out unit loss as a necessary part of the effect.
- The "offset off the axis" is measured on the readers' most-read coordinate, which means
  little when readers are spread; the single-coordinate probe replaces it here.

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| In place gives most of the rewire arm's gain | **Mostly right:** equivalent on English comparison; about two thirds of rewire's gain on taps → withheld Chinese |
| Keep-weight gives little | **Wrong:** it beats random on both measures and gives the largest Chinese cross-sense gain; it falls short of rewire only on comparison |
| Dense in place gives little | **Right**, and it still pays the taught cost |
| Half gives less; double about the same, with a larger taught cost | **Right on transfer; the larger taught cost is not shown** (every taught contrast against rewire is inconclusive) |
| Single-coordinate probe higher on forms in the erasure arms | **Right** |

## Interpretation

- **The common factor is disruption of a sparse reader's connection.** Zeroing a connection
  and moving it to another coordinate both destroy what that unit computed. At one
  connection per operand, either removes the unit's whole view of that operand. Both work;
  neither needs the gradient to choose, and neither needs units to die.
- **Erasure and moving act differently.** Zeroing (in place, rewire) gives the strongest
  comparison on forms; moving with the old weight gives the strongest taps → form matching
  but weaker comparison. A moved weight is a wrong-sized, possibly wrong-signed input on a
  new coordinate, not a fresh start. Why the two favour different measures is not
  explained.
- **Dense readers do not benefit.** At the same per-connection rate, a dense unit keeps
  about two thirds of its code weights at each event, and its remaining weights can carry
  the solution. A sparse unit loses its whole input from an operand. The disruption per
  unit, not per connection, may be what matters; this is untested.
- **The effect saturates with rate.** Halving the rate loses part of the gain, doubling adds
  little on the two measures, but makes English comparison very consistent across seeds
  and places forms closest to the collection line (offsets 0.52–1.18). The reviewer's
  recurrence predicts an optimum somewhere; within ½× to 2× none is visible.
- **The code-level signature is cross-sense, not rank.** Count is readable from almost every
  coordinate of collection codes even in dense networks. The erasure arms differ in that
  form and tap codes carry count on those same coordinates. This fits the review's
  account that changing readers favour a code any reader can use, and remains a
  correlation: the fresh-reader and recovery diagnostics are still untested.
- **The cost does not depend on the mechanism.** Every arm that disrupts readers, dense or
  sparse, loses taught Chinese at its own block by 0.15–0.24.

## Computation

| | Value |
| --- | --- |
| Runs | 80 (8 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 54,441 s |
| Wall time per run, mean by arm | dense 350 s; random 343; rewire 354; in place 362; keep weight 352; dense in place 348; half 341; double 329 |
| Job wall time | 4,758 s (data 28 s, rendering 3 s, runs 4,727 s) |
| Peak RSS, largest worker | 1.18 GB |
| Analysis | 56 s wall, including the exactness check against the reset run and the single-coordinate probe |
| Selection and tuning | none; rates were fixed in the pre-registration from the fixed-coordinate run |

Per-arm wall time differences (329–362 s) reflect scheduling on a shared CPU, not the
operations: resets and moves draw one number per connection per event (45 events).

## Caveats

- Ten seeds; pilot.
- Rates are matched per connection, not per unit; the dense comparison is therefore not
  matched in disruption per unit.
- The single-coordinate probe is a linear diagnostic on final codes; it does not measure
  how fast a new reader learns.
- Immediate forgetting and recovery, and fresh readers on frozen codes, remain untested.
