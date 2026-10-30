# Fan-in pressure pilot v1, resets, rewiring and predictor capacity: results

Design, contrasts and predictions: [proposal, "Resets, rewiring and predictor capacity"](fan-in-pressure-proposal.md#resets-rewiring-and-predictor-capacity-written-before-running),
written before the run. Follows the [timing results](fan-in-pilot-v1-timing-results.md) and
an external review of them ([after-timing analysis](rigl-after-timing-analysis.md)).
Run 2026-10-30, seeds 230–239 (paired with the fixed-coordinate and timing runs),
D-varied+rehearsal. **A pilot.**

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot reset` (81 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot reset` (55 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-reset-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-reset-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-reset/` (not tracked).
- **Exactness:** the rerun arms (`dense`, random, RigL) reproduce the fixed-coordinate run
  exactly in all 30 runs.
- The runtime file records `0239188`, the commit that added the runner, and `dirty: true`,
  because another session's untracked files were in the tree.

## Summary

**Erasure matters; dead units alone do not explain RigL; the contribution of gradient
choice is not bounded.** Corrected 2026-10-30 after an external review
([after-reset analysis](rigl-after-reset-analysis.md)), which flagged two overstatements in
the first version.

- **Random rewiring with restart at zero gets most of RigL's gain.** Each connection moves
  to a random coordinate at RigL's switch rate and restarts at zero. Against random wiring
  it is better on taps → withheld Chinese (+0.38) and English comparison (+0.18). Against
  RigL it is inconclusive on both (−0.15, −0.01) and equivalent on Devanagari comparison.
  It reaches Chinese comparison 0.98, above RigL's 0.86. Its readers are **not**
  concentrated on a shared coordinate (11% of live connections on the most-read
  dimension, against RigL's 87%). Its code still becomes almost one-dimensional:
  collections put 98% of their variance on the first direction, as in RigL (99%).
- **Gradient-chosen moves that keep their weight lose much of the gain.** This arm is worse
  than RigL on taps → withheld Chinese (−0.25) and English comparison (−0.14). It keeps
  about half of RigL's Chinese gain and a quarter of its English gain over random. It has
  almost no dead units, so this arm removes erasure and dead units together.
- **Lost capacity alone does nothing.** Silencing RigL's average number of units from the
  start leaves the random arm unchanged on both measures (+0.05 inconclusive, +0.02
  equivalent).
- **The effect is not mainly in `P`.** RigL on `P` alone improves English comparison (+0.13)
  but not taps → withheld Chinese (+0.12, inconclusive). RigL on `g` alone improves both
  (+0.28, +0.13). Neither alone reaches RigL on English comparison. Each gives part of the
  effect.
- **The cost is the same as RigL's.** Every arm that erases connections in `g` loses taught
  Chinese at its own block (−0.17 to −0.31 against random), as RigL does (−0.24). RigL on
  `P` alone and the silenced arm keep it.

## Pre-declared readings

On taps → withheld Chinese forms and English comparison.

| Reading | Rule | Outcome |
| --- | --- | --- |
| Origin in `P` | `riglP` better than random on both | **No:** taps → Chinese +0.12 [−0.05, +0.29] inc; English +0.13 [+0.08, +0.19] bet |
| Origin in `g` | `riglg` better than random on both | **Yes:** +0.28 [+0.06, +0.49]; +0.13 [+0.08, +0.18] |
| Erasure matters | `riglcarry` worse than RigL on either | **Yes, on both:** −0.25 [−0.49, −0.02]; −0.14 [−0.19, −0.09] |
| Gradient choice matters | `rewire` worse than RigL on either | **Not shown:** −0.15 [−0.36, +0.06] inc; −0.01 [−0.05, +0.03] inc |
| Erasure alone suffices | `rewire` equivalent or better than RigL on both | **Not shown:** inconclusive on both, not equivalent |
| Capacity alone suffices | `dead` better than random on both | **No:** +0.05 [−0.10, +0.20] inc; +0.02 [+0.00, +0.04] equ |
| Review's hypothesis | `riglP` better than random on both and `riglcarry` worse than RigL on both | **Not supported as stated:** the erasure half holds, the `P` half does not |

Devanagari comparison, reported alongside: `rewire` is equivalent to RigL (−0.01
[−0.02, +0.01]); `riglP`, `riglcarry` and `dead` are worse than RigL; `riglg` is
inconclusive (−0.05).

## Results

Means over ten seeds. Outcome against dense: paired 95% t-interval rule.

| Measure | Dense | Random | RigL | RigL, `P` only | RigL, `g` only | RigL, keep weight | Random rewire | Silenced units |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Transfer** | | | | | | | | |
| Chinese withheld, own block | 0.28 | 0.35 inc | 0.31 inc | 0.36 inc | 0.21 inc | 0.28 inc | 0.34 inc | 0.35 inc |
| Devanagari withheld, own block | 0.20 | 0.19 inc | 0.33 bet | 0.35 bet | 0.38 bet | 0.32 inc | 0.40 bet | 0.30 bet |
| taps → Chinese form, withheld | 0.13 | 0.12 inc | **0.66 bet** | 0.24 inc | 0.40 bet | 0.40 bet | **0.51 bet** | 0.17 inc |
| taps → Devanagari form, withheld | 0.32 | 0.32 inc | **0.64 bet** | 0.37 inc | 0.43 inc | **0.61 bet** | **0.61 bet** | 0.33 inc |
| form → taps, Chinese withheld | 0.31 | 0.20 inc | 0.57 bet | 0.49 inc | 0.49 inc | 0.47 inc | 0.54 inc | 0.32 inc |
| Comparison, Chinese forms | 0.57 | 0.64 inc | 0.86 bet | 0.77 bet | 0.74 bet | 0.74 bet | **0.98 bet** | 0.70 bet |
| Comparison, Devanagari forms | 0.64 | 0.68 inc | **0.81 bet** | 0.75 bet | 0.76 bet | 0.73 bet | **0.80 bet** | 0.70 bet |
| Comparison, English forms | 0.49 | 0.56 bet | **0.75 bet** | 0.69 bet | 0.69 bet | 0.60 bet | **0.74 bet** | 0.58 bet |
| **Taught** | | | | | | | | |
| Chinese paired, own block | 0.64 | 0.62 inc | 0.38 wor | 0.59 inc | 0.31 wor | 0.45 wor | 0.45 wor | 0.63 equ |
| taps → collection | 0.64 | 0.68 inc | 0.61 inc | 0.61 inc | 0.60 inc | 0.62 inc | 0.61 inc | 0.64 inc |
| Comparison, collections | 0.97 | 0.97 equ | 0.96 equ | 0.97 equ | 0.97 equ | 0.97 equ | 0.96 equ | 0.97 equ |
| Succession pairwise, collections | 0.91 | 0.92 equ | 0.86 inc | 0.91 equ | 0.92 equ | 0.88 inc | 0.90 equ | 0.92 equ |
| **Other** | | | | | | | | |
| Chinese paired, end | 0.62 | 0.63 inc | 0.49 wor | 0.59 inc | 0.50 wor | 0.53 wor | 0.55 wor | 0.59 inc |
| English, end | 0.39 | 0.40 inc | 0.24 wor | 0.35 inc | 0.28 wor | 0.28 wor | 0.28 wor | 0.37 inc |
| Judgment revision: isolated, withheld | 0.23 | 0.30 inc | 0.36 bet | 0.36 bet | 0.33 bet | 0.34 bet | 0.41 bet | 0.34 bet |
| Judgment revision: revised, withheld | 0.59 | 0.61 inc | 0.53 wor | 0.59 inc | 0.59 inc | 0.58 inc | 0.57 inc | 0.65 inc |
| Next-judgment coherence AUROC, Chinese | 0.93 | 0.93 equ | 0.90 inc | 0.94 inc | 0.89 inc | 0.90 inc | 0.94 inc | 0.94 inc |
| Next-judgment coherence AUROC, English | 0.81 | 0.82 inc | 0.70 wor | 0.78 inc | 0.72 wor | 0.69 wor | 0.72 wor | 0.82 inc |
| Thai, 4 shots, after 200 updates | 0.19 | 0.18 equ | 0.18 equ | 0.16 inc | 0.18 equ | 0.17 inc | 0.17 inc | 0.19 equ |
| Probe order, Chinese forms | 0.57 | 0.53 inc | 0.77 bet | 0.53 inc | 0.50 inc | 0.68 inc | 0.77 bet | 0.57 inc |
| Probe order, Devanagari forms | 0.65 | 0.61 inc | 0.74 bet | 0.65 equ | 0.70 inc | 0.68 inc | 0.74 bet | 0.66 equ |
| Probe order, English forms | 0.50 | 0.51 equ | 0.68 bet | 0.60 bet | 0.65 bet | 0.57 bet | 0.68 bet | 0.57 bet |

Arm names in the artifacts: `dense`, `read-q16-untied`, `read-q16-untied-rigl`,
`-riglP`, `-riglg`, `-riglcarry`, `-rewire`, `-dead`.

### Declared contrasts

Mean paired difference [95% t-interval], outcome.

| Measure | `P` only − random | `P` only − RigL | `g` only − random | `g` only − RigL | Keep weight − random | Keep weight − RigL | Rewire − random | Rewire − RigL | Silenced − random | Silenced − RigL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| taps → Chinese form, withheld | +0.12 [−0.05, +0.29] inc | −0.42 [−0.63, −0.20] wor | +0.28 [+0.06, +0.49] bet | −0.26 [−0.59, +0.07] inc | +0.28 [+0.11, +0.46] bet | −0.25 [−0.49, −0.02] wor | +0.38 [+0.17, +0.60] bet | −0.15 [−0.36, +0.06] inc | +0.05 [−0.10, +0.20] inc | −0.49 [−0.64, −0.33] wor |
| Comparison, English forms | +0.13 [+0.08, +0.19] bet | −0.05 [−0.09, −0.02] wor | +0.13 [+0.08, +0.18] bet | −0.06 [−0.09, −0.02] wor | +0.05 [+0.00, +0.09] inc | −0.14 [−0.19, −0.09] wor | +0.18 [+0.13, +0.23] bet | −0.01 [−0.05, +0.03] inc | +0.02 [+0.00, +0.04] equ | −0.17 [−0.20, −0.14] wor |
| Comparison, Devanagari forms | +0.07 [+0.03, +0.11] bet | −0.06 [−0.08, −0.03] wor | +0.08 [+0.04, +0.11] bet | −0.05 [−0.07, −0.03] inc | +0.04 [+0.01, +0.08] inc | −0.08 [−0.10, −0.06] wor | +0.11 [+0.08, +0.15] bet | −0.01 [−0.02, +0.01] equ | +0.01 [−0.03, +0.06] inc | −0.11 [−0.13, −0.08] wor |
| Comparison, Chinese forms | +0.13 [−0.01, +0.27] inc | −0.10 [−0.33, +0.14] inc | +0.10 [−0.12, +0.32] inc | −0.12 [−0.30, +0.05] inc | +0.10 [−0.08, +0.28] inc | −0.12 [−0.38, +0.13] inc | +0.34 [+0.20, +0.47] bet | +0.11 [−0.07, +0.30] inc | +0.06 [−0.09, +0.20] inc | −0.17 [−0.34, +0.01] inc |
| taps → Devanagari form, withheld | +0.06 [−0.11, +0.22] inc | −0.27 [−0.43, −0.11] wor | +0.12 [−0.06, +0.29] inc | −0.21 [−0.40, −0.01] wor | +0.29 [+0.14, +0.44] bet | −0.03 [−0.16, +0.10] inc | +0.29 [+0.09, +0.50] bet | −0.03 [−0.17, +0.11] inc | +0.02 [−0.16, +0.20] inc | −0.31 [−0.47, −0.14] wor |
| Chinese paired, own block | −0.03 [−0.09, +0.03] inc | +0.21 [+0.08, +0.34] bet | −0.31 [−0.45, −0.17] wor | −0.07 [−0.18, +0.04] inc | −0.17 [−0.26, −0.09] wor | +0.07 [−0.06, +0.19] inc | −0.17 [−0.22, −0.13] wor | +0.07 [−0.05, +0.19] inc | +0.02 [−0.03, +0.06] inc | +0.25 [+0.12, +0.38] bet |

Every contrast and measure is in the comparison artifact.

### Per seed (230–239)

| Arm | taps → withheld Chinese | English comparison |
| --- | --- | --- |
| RigL, `P` only | 0.15 0.03 0.25 0.05 0.18 0.27 0.62 0.35 0.03 0.48 | 0.73 0.76 0.71 0.61 0.62 0.70 0.75 0.72 0.57 0.74 |
| RigL, `g` only | 0.12 0.92 0.35 0.02 0.35 0.65 0.77 0.60 0.05 0.17 | 0.67 0.67 0.71 0.75 0.67 0.73 0.71 0.73 0.58 0.66 |
| RigL, keep weight | 0.37 0.02 0.48 0.00 0.32 0.28 0.67 0.78 0.45 0.68 | 0.58 0.74 0.73 0.58 0.60 0.64 0.54 0.56 0.47 0.58 |
| Random rewire | 0.77 0.30 0.50 0.37 0.13 0.87 0.87 0.78 0.28 0.18 | 0.77 0.78 0.75 0.73 0.73 0.74 0.77 0.76 0.57 0.74 |
| Silenced units | 0.03 0.00 0.08 0.25 0.07 0.47 0.08 0.37 0.00 0.35 | 0.58 0.68 0.60 0.57 0.58 0.59 0.54 0.60 0.53 0.49 |

The per-seed values for dense, random and RigL are in the
[fixed-coordinate results](fan-in-pilot-v1-fixed-results.md#per-seed-230239); the reruns are
identical.

### Readers and code

| | Dense | Random | RigL | `P` only | `g` only | Keep weight | Rewire | Silenced |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dead units, `g` / `P` | 0 / 0 | 0 / 0 | 0.12 / 0.49 | 0 / 0.50 | 0.12 / 0 | 0.01 / 0.01 | 0.06 / 0.42 | 0.11 / 0.50 |
| Live `g` connections on the most-read dimension | — | 0.06 | 0.87 | 0.06 | 0.73 | 0.40 | 0.11 | 0.07 |
| Live `P` connections on the most-read dimension | — | 0.06 | 0.89 | 0.76 | 0.06 | 0.46 | 0.12 | 0.10 |
| Collection variance on the first direction | 0.88 | 0.91 | 0.99 | 0.97 | 0.89 | 0.94 | 0.98 | 0.94 |
| Collection variance on the reader axis | 0.14 | 0.05 | 0.48 | 0.09 | 0.48 | 0.15 | 0.08 | 0.07 |
| Code participation ratio (collections) | 1.28 | 1.21 | 1.02 | 1.07 | 1.25 | 1.14 | 1.04 | 1.14 |
| English forms' offset off the axis | 1.73 | 3.41 | 0.31 | 1.42 | 0.50 | 0.60 | 1.19 | 1.82 |
| Chinese forms' offset off the axis | 2.84 | 6.06 | 0.74 | 2.40 | 1.35 | 1.83 | 1.71 | 3.27 |
| Devanagari forms' offset off the axis | 1.00 | 2.10 | 0.19 | 1.02 | 0.44 | 0.52 | 0.76 | 1.19 |

Connections grown per run (first layer): RigL `g` 2,055, `P` 1,322; `P` only 2,300; `g`
only 2,932; keep weight `g` 3,742, `P` 3,654; rewire `g` 2,038, `P` 1,334. The rewire arm
matches RigL's switch rate, as designed. Keeping the weight almost doubles the number of
moves, because a moved connection keeps a large weight and its gradient keeps pointing
elsewhere. Share of units whose two operands read the same dimension (all units, first
layer), 0.05–0.07 at the start: at the end RigL 0.81 in `g`, 0.92 in `P`; `P` only 0.79;
`g` only 0.67; keep weight 0.25 and 0.43; rewire 0.06 in both.

- **Dead units follow erasure, not gradient choice.** Random rewiring kills 42% of `P`
  and 6% of `g`, close to RigL. Gradient moves that keep their weights kill almost none.
- **The code becomes one-dimensional without concentrated readers.** Rewire collections
  put 98% of their variance on one direction, but the readers' most-read coordinate holds
  only 8% of it: the count direction is not aligned with any one coordinate. RigL on `g`
  alone concentrates `g` (73%) and puts 48% on the axis, yet its first-direction share
  stays at 0.89, like random.

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| `riglP` carries most of the effect | **Wrong:** `g` carries more; neither alone reaches RigL |
| `riglcarry` loses most of the gain, with few dead units | **Partly right:** worse than RigL on both measures and almost no dead units, but it keeps about half the Chinese gain |
| `rewire` gives part of the gain | **Wrong in size:** it gives most of it, not distinguishable from RigL |
| `dead` gives little | **Correct** |

## Interpretation

- **Repeated erasure is the main ingredient.** Restarting connections at zero, on RigL's
  schedule and at random coordinates, reproduces most of RigL's transfer. How much gradient
  choice adds is not bounded: rewire is 0.15 below RigL on taps → withheld Chinese with an
  interval of [−0.36, +0.06], and equivalent only on Devanagari comparison.
- **Concentrated readers are not needed for transfer.** The rewire arm's readers stay
  spread across the code, unlike RigL's and the fixed arm's. The code itself becomes
  close to one-dimensional, which a spread reader can still read. Concentration onto a
  coordinate (the fixed and timing runs) and a one-dimensional code (rewire) are two
  routes to the same transfer.
- **Dead units alone are not enough.** Silencing as many units from the start does
  nothing. This does not show that units dying during learning is irrelevant, or that it
  does not interact with erasure. The keep-weight arm cannot separate erasure from dead
  units, because it removes both; it also makes about twice as many moves as RigL, so it
  does not intervene on erasure alone.
- **Low rank alone is not enough.** RigL on `P` only puts 97% of collection variance on
  one direction, close to rewire's 98%, but transfers much less across senses.
- **Both readers contribute.** Erasure in `g` alone gives the Chinese gain; erasure in
  either gives about two thirds of the English gain. The full effect needs both.
- **What erasure does is not yet shown.** A plausible account: a connection reset to zero
  must be relearned from whatever the code currently offers, so reader solutions that fit
  only some contexts are repeatedly lost, and the encoders are pushed toward a code that
  any newly grown connection can read. This is the review's hypothesis without its `P`
  part. It is an instance of forget-and-relearn (Zhou et al. 2022) and resembles
  iterated learning ([literature notes](forget-relearn-iterated-learning-notes.md)). Three
  explanations remain distinct: useful structure resists the reset, is relearned faster
  after it, or repeated relearning changes the encoders. Final transfer does not separate
  them ([reviewer's note](rigl-after-reset-analysis.md)).
- **The cost remains.** Every arm with erasure in `g` loses taught Chinese at its own block
  and English retention. Erasure trades fitting the current block for transfer.

## Computation

| | Value |
| --- | --- |
| Runs | 80 (8 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 55,642 s |
| Wall time per run, mean by arm | dense 349 s; random 354; RigL 359; `P` only 356; `g` only 357; keep weight 363; rewire 351; silenced 348 |
| Job wall time | 4,832 s (data 28 s, rendering 3 s, runs 4,801 s) |
| Peak RSS, largest worker | 1.19 GB |
| Analysis | 55 s wall, including the exactness check against the fixed-coordinate run |
| Selection and tuning | none; the rewire probabilities and silenced counts were computed from the fixed-coordinate artifacts before the run |

Random rewiring draws one number per connection per connectivity update: O(E) per update
for E reader connections (768), 45 updates per run. Keeping the weight is a copy per moved
connection. Neither is measurable against training.

## Caveats

- Ten seeds; pilot.
- The keep-weight arm removes erasure and dead units together.
- Rewiring stops at update 4,500 in every arm; the rewire arm's wiring is still changing
  then, and its rate is matched only on average, not per update.
- Silenced units are drawn at random; RigL's dead units are not random.
