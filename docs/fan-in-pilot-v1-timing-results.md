# Fan-in pressure pilot v1, timing or choice of the coordinate: results

Design, contrasts and predictions: [proposal, "Timing or choice of the coordinate"](fan-in-pressure-proposal.md#timing-or-choice-of-the-coordinate-written-before-running),
written before the run. Follows the [fixed-coordinate results](fan-in-pilot-v1-fixed-results.md).
Run 2026-10-30, seeds 230–239 (paired with the fixed-coordinate run), D-varied+rehearsal.
**A pilot.**

*Correction, 2026-10-30.* The pre-registration says the gradual schedule follows RigL's
change budget. At one connection per operand it does not: RigL drops max(1, round(αk)) = 1
connection per operand at every connectivity update, so it reconsiders every connection
every 100 updates whatever α is, and a switched connection restarts at zero. The gradual
arms move each unit once and carry its weights. This run therefore rules out that
intervention, not every explanation based on RigL's training history. Found by an external
review ([after-timing analysis](rigl-after-timing-analysis.md)); followed up in the
[resets proposal](fan-in-pressure-proposal.md#resets-rewiring-and-predictor-capacity-written-before-running).

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot timing` (60 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot timing` (44 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-timing-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-timing-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-timing/` (not tracked).
- **Exactness:** the rerun arms (`dense`, RigL, `read-q16-fixed`) reproduce the
  fixed-coordinate run exactly in all 30 runs.
- The runner was committed in `a0f0fd0`. The runtime file records `217701a`, because a
  documentation commit was made during the run, and `dirty: true`, because another
  session's untracked files were in the tree. The runner's hash is recorded.

## Summary

**Neither timing nor choice of the coordinate explains RigL.** By the pre-declared rules,
timing matters (a little), choice does not, and RigL is not explained.

- **Timing.** Concentrating gradually, on RigL's schedule, improves English comparison a
  little over concentrating from the start: 0.52 → 0.58 with the drawn coordinate, 0.53 →
  0.59 with RigL's. It also **removes the fixed arm's Chinese gains**: taps → withheld
  Chinese 0.76 → 0.33, Chinese comparison 0.99 → 0.57, Chinese probe order 0.94 → 0.52.
  Gradual concentration lands near dense on Chinese, and only part of the way to RigL on
  English.
- **Choice.** Using the coordinate RigL converged on for the same seed changes nothing.
  Every choice contrast on the three named measures is equivalent or inconclusive, with
  differences of 0.00–0.02. Collections put count on whatever coordinate the readers
  read.
- **RigL stays ahead.** The best new arm (`read-q16-gradual-rigld`) is worse than RigL on
  taps → withheld Chinese (−0.39), English comparison (−0.16) and Devanagari comparison
  (−0.11).

## Pre-declared readings

| Reading | Rule | Outcome |
| --- | --- | --- |
| Timing matters | Gradual better than start, for either coordinate, on English or Devanagari comparison | **Yes, on English:** +0.06 [+0.02, +0.10] (drawn), +0.05 [+0.02, +0.09] (RigL's) |
| Choice matters | RigL's coordinate better than the drawn one, at either timing, on English or Devanagari comparison | **No:** all equivalent or inconclusive (+0.01 to +0.02) |
| RigL is explained | `gradual-rigld` equivalent or better than RigL on taps → withheld Chinese, English and Devanagari comparison | **No:** worse on all three |

The proposal's reading for the last outcome: something else in RigL matters (its partial
concentration, its dead units, or the order in which units move).

## Results

Means over ten seeds. Outcome against dense: paired 95% t-interval rule.

| Measure | Dense | RigL | Fixed, start | Gradual | Fixed, RigL's coord. | Gradual, RigL's coord. |
| --- | --- | --- | --- | --- | --- | --- |
| **Transfer** | | | | | | |
| Chinese withheld, own block | 0.28 | 0.31 inc | 0.41 inc | 0.17 wor | 0.37 inc | 0.18 inc |
| Devanagari withheld, own block | 0.20 | 0.33 bet | 0.26 inc | 0.26 inc | 0.27 inc | 0.24 inc |
| taps → Chinese form, withheld | 0.13 | **0.66 bet** | **0.76 bet** | 0.33 inc | **0.76 bet** | 0.26 inc |
| taps → Devanagari form, withheld | 0.32 | **0.64 bet** | 0.39 inc | 0.42 inc | 0.36 inc | 0.40 inc |
| form → taps, Chinese withheld | 0.31 | **0.57 bet** | **0.59 bet** | 0.33 inc | 0.55 inc | 0.27 inc |
| Comparison, Chinese forms | 0.57 | **0.86 bet** | **0.99 bet** | 0.57 inc | **0.98 bet** | 0.57 inc |
| Comparison, Devanagari forms | 0.64 | **0.81 bet** | 0.66 inc | 0.69 bet | 0.68 inc | 0.70 bet |
| Comparison, English forms | 0.49 | **0.75 bet** | 0.52 equ | 0.58 bet | 0.53 inc | 0.59 bet |
| **Taught** | | | | | | |
| Chinese paired, own block | 0.64 | 0.38 wor | 0.56 wor | 0.46 wor | 0.57 wor | 0.50 wor |
| taps → collection | 0.64 | 0.61 inc | 0.60 inc | 0.58 wor | 0.57 inc | 0.63 inc |
| Comparison, collections | 0.97 | 0.96 equ | 0.96 equ | 0.96 equ | 0.97 equ | 0.97 equ |
| Succession pairwise, collections | 0.91 | 0.86 inc | 0.86 inc | 0.89 equ | 0.87 inc | 0.89 inc |
| **Other** | | | | | | |
| Chinese paired, end | 0.62 | 0.49 wor | 0.50 wor | 0.43 wor | 0.52 wor | 0.44 wor |
| English, end | 0.39 | 0.24 wor | 0.20 wor | 0.22 wor | 0.21 wor | 0.20 wor |
| Judgment revision: isolated, withheld | 0.23 | 0.36 bet | 0.30 inc | 0.22 inc | 0.32 inc | 0.23 inc |
| Next-judgment coherence AUROC, Chinese | 0.93 | 0.90 inc | 0.94 equ | 0.87 wor | 0.93 equ | 0.87 wor |
| Next-judgment coherence AUROC, English | 0.81 | 0.70 wor | 0.74 wor | 0.68 wor | 0.74 wor | 0.66 wor |
| Thai, 4 shots, after 200 updates | 0.19 | 0.18 equ | 0.18 inc | 0.16 inc | 0.17 inc | 0.15 inc |
| Probe order, Chinese forms | 0.57 | 0.77 bet | 0.94 bet | 0.52 inc | 0.90 bet | 0.46 inc |
| Probe order, Devanagari forms | 0.65 | 0.74 bet | 0.66 equ | 0.68 inc | 0.66 equ | 0.68 inc |
| Probe order, English forms | 0.50 | 0.68 bet | 0.51 equ | 0.56 inc | 0.51 inc | 0.56 bet |

### Declared contrasts

Mean paired difference [95% t-interval], outcome, on the named measures.

| Measure | Gradual − start (drawn) | Gradual − start (RigL's) | RigL's − drawn (start) | RigL's − drawn (gradual) | Gradual, RigL's − RigL |
| --- | --- | --- | --- | --- | --- |
| taps → Chinese form, withheld | −0.43 [−0.70, −0.16] wor | −0.50 [−0.73, −0.28] wor | +0.00 [−0.06, +0.06] inc | −0.07 [−0.34, +0.19] inc | −0.39 [−0.64, −0.15] wor |
| Comparison, English forms | +0.06 [+0.02, +0.10] bet | +0.05 [+0.02, +0.09] bet | +0.01 [−0.01, +0.03] equ | +0.01 [−0.03, +0.04] equ | −0.16 [−0.19, −0.13] wor |
| Comparison, Devanagari forms | +0.03 [−0.02, +0.09] inc | +0.02 [−0.03, +0.07] inc | +0.02 [−0.00, +0.04] equ | +0.01 [−0.05, +0.06] inc | −0.11 [−0.15, −0.07] wor |
| Comparison, Chinese forms | −0.42 [−0.58, −0.26] wor | −0.41 [−0.58, −0.24] wor | −0.01 [−0.02, +0.00] equ | +0.01 [−0.18, +0.19] inc | −0.29 [−0.48, −0.10] wor |
| Chinese paired, own block | −0.10 [−0.15, −0.05] wor | −0.07 [−0.14, −0.00] wor | +0.00 [−0.04, +0.05] inc | +0.03 [−0.02, +0.09] inc | +0.11 [−0.04, +0.27] inc |

Every contrast and measure is in the comparison artifact.

### Per seed (230–239)

| Arm | taps → withheld Chinese | English comparison |
| --- | --- | --- |
| Gradual | 0.87 0.05 0.20 0.00 0.38 0.78 0.07 0.88 0.10 0.00 | 0.47 0.65 0.62 0.60 0.56 0.62 0.60 0.59 0.53 0.56 |
| Gradual, RigL's coordinate | 0.27 0.05 0.00 0.85 0.12 0.50 0.02 0.82 0.00 0.00 | 0.58 0.65 0.70 0.55 0.54 0.59 0.60 0.61 0.50 0.56 |

The fixed-coordinate run's per-seed values for dense, RigL and the fixed arm are in
[its results](fan-in-pilot-v1-fixed-results.md#per-seed-230239); the reruns are identical.

### Readers and code

| | Dense | RigL | Fixed, start | Gradual | Fixed, RigL's | Gradual, RigL's |
| --- | --- | --- | --- | --- | --- | --- |
| Dead units, `g` / `P` | 0 / 0 | 0.12 / 0.49 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Live `g` connections on the most-read dimension | — | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 |
| Collection variance on the reader axis | 0.14 | 0.48 | 0.47 | 0.30 | 0.51 | 0.36 |
| Cosine of first direction with reader axis | 0.39 | 0.69 | 0.72 | 0.57 | 0.76 | 0.63 |
| English forms' offset off the axis | 1.73 | 0.31 | 0.71 | 0.74 | 0.72 | 0.77 |
| Chinese forms' offset off the axis | 2.84 | 0.74 | 0.66 | 2.20 | 0.64 | 2.32 |
| Devanagari forms' offset off the axis | 1.00 | 0.19 | 0.42 | 0.57 | 0.45 | 0.51 |
| Code participation ratio (collections) | 1.28 | 1.02 | 1.46 | 1.38 | 1.41 | 1.38 |

- Gradual arms end with every unit on the coordinate, like the start arms, but their
  collections put less variance on it (0.30–0.36 against 0.47–0.51). Chinese forms sit far
  off the axis in the gradual arms (2.2–2.3), as in dense.
- In seeds 231 and 239 the drawn coordinate equals RigL's, so the start arms are identical
  there, and so are the gradual arms.

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| Timing matters: gradual improves English comparison | **Correct in direction, small:** +0.05 to +0.06 |
| Choice matters less | **Correct:** choice makes no difference |
| `gradual-rigld` close to RigL, below it on English | **Wrong in size:** far below RigL on English, Devanagari and Chinese |
| The fixed arm's Chinese gains remain in all four cells | **Wrong:** they vanish in both gradual cells |

## Interpretation

- **The coordinate does not matter.** Whichever coordinate every reader reads,
  collections put count on it. RigL's converged coordinate is not special.
- **Timing trades Chinese for a little English.** Concentrating from the start gives very
  strong Chinese transfer and none for English. Concentrating gradually gives a small
  English gain and loses Chinese. Neither gives RigL's broad transfer on all three
  scripts. Why the Chinese gain depends on concentration from the first update is not
  explained; the Chinese block (block 2) begins at update 2,000, when the gradual arms
  already have 49 of 64 units on the axis.
- **What else differs in RigL** (untested):
  - **Partial concentration.** RigL's live units put 87% of their connections on the
    shared coordinate, not 100%. Some units keep reading other dimensions.
  - **Dead units.** RigL silences 12% of `g`'s units and 49% of `P`'s. The fixed and
    gradual arms keep all units, so `P` has twice RigL's live capacity.
  - **Gradient-chosen order.** RigL moves the units whose connections matter least and
    grows where the gradient is largest; the gradual arms move units in a random order.
  - **Restart from zero.** RigL's moved connections start at weight zero; the gradual
    arms carry the old weights over.
  - **Repeated rewiring.** RigL grows about 2,000 connections in `g` per run, many of them
    dropped again: units change coordinates repeatedly before settling.
- **The dead-unit point now matters.** The fixed-coordinate run showed dead units are not
  needed for the Chinese gain. They may still be needed for the English and Devanagari
  gains, which no unit-preserving arm has reproduced.

## Computation

| | Value |
| --- | --- |
| Runs | 60 (6 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 40,965 s |
| Wall time per run, mean by arm | dense 350 s; RigL 355; fixed 349; gradual 347; fixed, RigL's 355; gradual, RigL's 353 |
| Job wall time | 3,580 s (data 29 s, rendering 3 s, runs 3,548 s) |
| Peak RSS, largest worker | 1.19 GB |
| Analysis | 44 s wall, including the exactness check against the fixed-coordinate run |
| Selection and tuning | none; RigL's coordinates were read from the fixed-coordinate artifact before the run |

The gradual arms move at most 64 units per reader per run, with one copy of two weights
each; no measurable cost.

## Caveats

- Ten seeds; eight carry the choice contrast.
- One gradual schedule (RigL's change budget, front-loaded), one random order of units.
- RigL's coordinate is an oracle taken from a finished run.
