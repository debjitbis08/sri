# Fan-in pressure pilot v1, replication with a fixed coordinate: results

Design, contrasts and predictions: [proposal, "Replication with a fixed coordinate"](fan-in-pressure-proposal.md#replication-with-a-fixed-coordinate-written-before-running),
written before the run. Follows the [one-dimension RigL results](fan-in-pilot-v1-rigl1-results.md).
Run 2026-10-30, fresh seeds 230–239, D-varied+rehearsal. **A pilot.**

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot fixed` (50 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot fixed` (43 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-fixed-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-fixed-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-fixed/` (not tracked).
- The run was made from revision `567de0e` with the uncommitted runner, analysis and
  pre-registration changes in the working tree (`dirty: true` in both runtime files; module
  hashes are recorded).

## Summary

**The RigL result replicates. Fixing the coordinate in advance does not reproduce it in
general, though it does for Chinese forms. By the pre-declared rule, the learned choice
matters.**

- **Replication.** RigL against random, on fresh seeds:
  - taps → withheld Chinese forms +0.53 [+0.31, +0.76];
  - English comparison +0.19 [+0.16, +0.22];
  - better also on taps → withheld Devanagari, form → taps, Chinese and Devanagari
    comparison, Devanagari placement and all three form probes.

  The costs replicate too: Chinese paired at its block −0.24, retention of Chinese and
  English at the end −0.14 and −0.16, English coherence −0.12.
- **Fixed coordinate, every unit (`fixed`).** The effect splits by script:
  - **Chinese:** as good as or better than RigL. taps → withheld Chinese 0.76 (0.58–0.88
    in every seed), Chinese comparison 0.99 in every seed, Chinese probe order 0.94.
  - **English and Devanagari:** no gain over dense. English comparison 0.52 (RigL 0.75),
    Devanagari comparison 0.66 (RigL 0.81), taps → withheld Devanagari 0.39 (RigL 0.64).
  - **Taught costs** are similar to RigL's. Chinese paired at its block falls less (0.56
    against RigL's 0.38), English retention falls more (0.20 against 0.24).
- **Fixed coordinate, 75% of units (`fixed75`).** Mostly like the random arm. The 16 free
  units per reader appear to let the network avoid the axis. Chinese comparison still
  improves (+0.20 against dense), and retention is closer to dense.
- **Dead units are not the mechanism.** The fixed arms have no dead units; RigL has 12%
  in `g` and 49% in `P`. The fixed arm, with full capacity, transfers on Chinese as well
  as RigL does, so RigL's gain does not need lost capacity.

## Pre-declared readings

| Reading | Rule | Outcome |
| --- | --- | --- |
| Replicated | RigL − random better on taps → withheld Chinese and English comparison | **Yes** (+0.53, +0.19, both pass) |
| Concentration is enough | `fixed` or `fixed75` equivalent or better than RigL on both | **No.** `fixed`: +0.11 [−0.06, +0.27] inconclusive and −0.22 worse; `fixed75`: worse on both |
| The learned choice matters | Both fixed arms worse than RigL on either | **Yes,** on English comparison (−0.22, −0.24) |

## Results

Means over ten seeds. Outcome against dense: paired 95% t-interval rule.

| Measure | Dense | Random | RigL | Fixed | Fixed 75% |
| --- | --- | --- | --- | --- | --- |
| **Transfer** | | | | | |
| Chinese withheld, own block | 0.28 | 0.35 inc | 0.31 inc | 0.41 inc | 0.29 inc |
| Devanagari withheld, own block | 0.20 | 0.19 inc | 0.33 bet | 0.26 inc | 0.18 inc |
| taps → Chinese form, withheld | 0.13 | 0.12 inc | **0.66 bet** | **0.76 bet** | 0.31 inc |
| taps → Devanagari form, withheld | 0.32 | 0.32 inc | **0.64 bet** | 0.39 inc | 0.34 inc |
| form → taps, Chinese withheld | 0.31 | 0.20 inc | **0.57 bet** | **0.59 bet** | 0.40 inc |
| Comparison, Chinese forms | 0.57 | 0.64 inc | **0.86 bet** | **0.99 bet** | 0.77 bet |
| Comparison, Devanagari forms | 0.64 | 0.68 inc | **0.81 bet** | 0.66 inc | 0.67 inc |
| Comparison, English forms | 0.49 | 0.56 bet | **0.75 bet** | 0.52 equ | 0.51 equ |
| **Taught** | | | | | |
| Chinese paired, own block | 0.64 | 0.62 inc | 0.38 wor | 0.56 wor | 0.61 inc |
| taps → collection | 0.64 | 0.68 inc | 0.61 inc | 0.60 inc | 0.65 inc |
| Comparison, collections | 0.97 | 0.97 equ | 0.96 equ | 0.96 equ | 0.97 equ |
| Succession pairwise, collections | 0.91 | 0.92 equ | 0.86 inc | 0.86 inc | 0.91 equ |
| **Other** | | | | | |
| Chinese paired, end | 0.62 | 0.63 inc | 0.49 wor | 0.50 wor | 0.60 inc |
| English, end | 0.39 | 0.40 inc | 0.24 wor | 0.20 wor | 0.32 wor |
| Judgment revision: isolated, withheld | 0.23 | 0.30 inc | 0.36 bet | 0.30 inc | 0.23 inc |
| Judgment revision: revised, withheld | 0.59 | 0.61 inc | 0.53 wor | 0.55 inc | 0.60 inc |
| Next-judgment coherence AUROC, Chinese | 0.93 | 0.93 equ | 0.90 inc | 0.94 equ | 0.92 inc |
| Next-judgment coherence AUROC, English | 0.81 | 0.82 inc | 0.70 wor | 0.74 wor | 0.76 wor |
| Thai, 4 shots, after 200 updates | 0.19 | 0.18 equ | 0.18 equ | 0.18 inc | 0.19 equ |
| Probe order, Chinese forms | 0.57 | 0.53 inc | 0.77 bet | 0.94 bet | 0.69 bet |
| Probe order, Devanagari forms | 0.65 | 0.61 inc | 0.74 bet | 0.66 equ | 0.63 inc |
| Probe order, English forms | 0.50 | 0.51 equ | 0.68 bet | 0.51 equ | 0.50 equ |
| Probe count error, Chinese withheld | 3.39 | 3.68 | 2.58 | 1.37 | 2.64 |
| Probe count error, Devanagari withheld | 1.78 | 1.83 | 1.20 | 1.63 | 1.83 |

### Contrasts against RigL

Mean paired difference [95% t-interval], outcome.

| Measure | Fixed − RigL | Fixed 75% − RigL |
| --- | --- | --- |
| taps → Chinese form, withheld | +0.11 [−0.06, +0.27] inc | −0.34 [−0.53, −0.16] wor |
| taps → Devanagari form, withheld | −0.25 [−0.39, −0.10] wor | −0.30 [−0.45, −0.16] wor |
| form → taps, Chinese withheld | +0.03 [−0.14, +0.19] inc | −0.17 [−0.37, +0.04] inc |
| Comparison, Chinese forms | +0.13 [−0.06, +0.31] inc | −0.09 [−0.30, +0.11] inc |
| Comparison, Devanagari forms | −0.15 [−0.17, −0.12] wor | −0.14 [−0.19, −0.09] wor |
| Comparison, English forms | −0.22 [−0.24, −0.20] wor | −0.24 [−0.26, −0.22] wor |
| Chinese paired, own block | +0.18 [+0.05, +0.31] bet | +0.23 [+0.12, +0.34] bet |
| Chinese paired, end | +0.01 [−0.03, +0.05] equ | +0.11 [+0.05, +0.17] bet |
| English, end | −0.04 [−0.07, −0.01] inc | +0.08 [+0.05, +0.12] bet |

Every contrast, including fixed − fixed75 and RigL − dense, is in the comparison artifact.

### Per seed (230–239)

| Arm | taps → withheld Chinese | English comparison |
| --- | --- | --- |
| Dense | 0.00 0.00 0.10 0.00 0.05 0.13 0.00 0.87 0.00 0.13 | 0.50 0.53 0.51 0.48 0.46 0.51 0.51 0.51 0.48 0.47 |
| Random | 0.00 0.17 0.02 0.00 0.03 0.07 0.12 0.70 0.02 0.10 | 0.56 0.63 0.60 0.54 0.57 0.56 0.49 0.55 0.56 0.49 |
| RigL | 0.80 0.42 0.92 0.78 0.13 0.85 0.47 0.77 0.65 0.78 | 0.75 0.79 0.77 0.75 0.68 0.77 0.73 0.74 0.72 0.76 |
| Fixed | 0.80 0.68 0.83 0.88 0.67 0.75 0.87 0.78 0.80 0.58 | 0.56 0.54 0.52 0.54 0.46 0.53 0.56 0.53 0.48 0.51 |
| Fixed 75% | 0.58 0.00 0.40 0.75 0.00 0.23 0.00 0.87 0.08 0.20 | 0.54 0.57 0.55 0.51 0.48 0.51 0.49 0.47 0.47 0.48 |

### Readers and code

Live units only; dead units are those never active on evaluation codes.

| | Dense | Random | RigL | Fixed | Fixed 75% |
| --- | --- | --- | --- | --- | --- |
| Dead units, `g` / `P` | 0.00 / 0.00 | 0.00 / 0.00 | 0.12 / 0.49 | 0.00 / 0.00 | 0.00 / 0.00 |
| Live `g` units reading the same dimension of both operands | — | 0.05 | 0.79 | 1.00 | 0.76 |
| Live `g` code connections on the most-read dimension | — | 0.06 | 0.87 | 1.00 | 0.77 |
| Live `P` code connections on the most-read dimension | — | 0.06 | 0.89 | 1.00 | 0.77 |
| Collection variance on the first direction | 0.88 | 0.91 | 0.99 | 0.81 | 0.83 |
| Collection variance on the reader axis | 0.14 | 0.05 | 0.48 | 0.47 | 0.30 |
| Cosine of first direction with reader axis | 0.39 | 0.18 | 0.69 | 0.72 | 0.59 |
| English forms' offset off the axis | 1.73 | 3.41 | 0.31 | 0.71 | 1.81 |
| Chinese forms' offset off the axis | 2.84 | 6.06 | 0.74 | 0.66 | 1.80 |
| Devanagari forms' offset off the axis | 1.00 | 2.10 | 0.19 | 0.42 | 0.76 |
| Code participation ratio (collections) | 1.28 | 1.21 | 1.02 | 1.46 | 1.42 |

- The reader axis is the live `g` units' most-read dimension; for dense, the coordinate
  with the largest loading on the first direction. Offsets are in collection standard
  deviations along the axis.
- RigL's `g` and `P` converge on the same dimension in all ten seeds (5, 1, 15, 7, 13, 12,
  7, 2, 9, 2). The fixed coordinates were 13, 1, 6, 13, 3, 6, 10, 8, 2, 2.
- RigL's operand overlap from the earlier diagnostic, which counts dead units, is 0.81 in
  `g` and 0.92 in `P`; the live-unit measures above replace it.

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| RigL replicates in direction on both measures | **Correct;** both pass the rule |
| `fixed` loses more taught matching than RigL | **Wrong:** Chinese paired at its block falls less (0.56 against 0.38); at the end the two are equal; English retention falls slightly more |
| `fixed75` closest to RigL on transfer | **Wrong:** `fixed75` is the furthest; `fixed` is closest, and ahead on Chinese |
| Fixed arms transfer less than RigL | Partly: on English and Devanagari, yes; on Chinese, no |
| No dead units in the fixed arms | Correct (0.00) |

## Interpretation

- **The RigL effect is real on this network and curriculum.** Two independent sets of
  ten seeds give nearly the same gains and the same costs.
- **Concentration on one coordinate is enough for some forms, not all.** With every
  reader on a random coordinate chosen in advance, collections put count on it (47% of
  their variance, as under RigL), and Chinese forms follow almost perfectly. English and
  Devanagari forms sit near the axis but are not ordered along it: comparison and probe
  order stay at dense levels.
- **What RigL adds is not explained by this run.** Two candidates, both untested:
  - **Timing.** RigL starts from diverse random readers and concentrates gradually as the
    drop fraction decays. Forms taught early (English is the first block) may be placed on
    the axis while readers still see several dimensions. The fixed arm concentrates from
    the first update.
  - **The chosen coordinate.** RigL grows connections where the gradient is largest,
    likely where count already lies. A random coordinate has to be built up.

  Why Chinese in particular benefits under the fixed arm is also unexplained. It is the
  second block, and Chinese 一, 二, 三 are drawn as one, two and three strokes; neither is
  shown to matter.
- **Free reader units undo the pressure.** With 16 of 64 units free, the network largely
  behaves as with random wiring. This matches the [proposal's premise](fan-in-pressure-proposal.md#question)
  that freedom the network keeps is freedom it uses: a quarter of the units left free is
  enough to remove most of the pressure.
- **The costs are the same as before.** Squeezing the readers onto one coordinate costs
  taught matching and retention, whether or not the coordinate is learned.

## Computation

| | Value |
| --- | --- |
| Runs | 50 (5 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 33,553 s |
| Wall time per run, mean by arm | dense 347 s; random 343; RigL 360; fixed 329; fixed 75% 344 |
| Job wall time | 2,999 s (data 28 s, rendering 3 s, runs 2,968 s) |
| Peak RSS, largest worker | 1.18 GB |
| Analysis | 43 s wall, 36 s CPU (now includes the live-unit and code-structure diagnostics) |
| Selection and tuning | none; the 75% share was set from rigl1's diagnostics before the run |

The fixed arms add one coordinate draw and one row permutation per reader; no
measurable cost.

## Caveats

- One curriculum arm, ten seeds per arm.
- One fixed coordinate per seed, the same for `g` and `P`, and one share (75%) for the
  mixed arm.
- The fixed arm differs from RigL in two ways at once: the coordinate is not learned, and
  concentration is present from the start rather than reached gradually. This run cannot
  tell which matters.
