# Fan-in pressure pilot v1, learned grouping at one dimension per operand: results

Design, contrasts and predictions: [proposal, "Learned grouping at one dimension per operand"](fan-in-pressure-proposal.md#learned-grouping-at-one-dimension-per-operand-written-before-running),
written before the run. Follows the [learned-grouping results](fan-in-pilot-v1-rigl-results.md).
Run 2026-10-30, fresh seeds 220–229, D-varied+rehearsal. **A pilot: one run, not yet replicated.**

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot rigl1` (30 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot rigl1` (36 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-rigl1-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-rigl1-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-rigl1/` (not tracked).

## Summary

**The prediction was wrong. Prune and regrow at one dimension per operand produces the
largest untaught transfer of any heads-free network on the number line, at a cost to
taught matching and retention.**

- **Cross-sense matching of never-paired forms jumps:**
  - taps → withheld Chinese forms 0.05 → 0.70 (0.37–0.83 in every seed; dense 0–0.17);
  - taps → withheld Devanagari forms 0.33 → 0.70; withheld Chinese form → taps 0.29 → 0.66.
- **Untaught comparison on forms:** Chinese 0.85, Devanagari 0.81, English 0.75 (dense 0.60,
  0.62, 0.47). English comparison is 0.71–0.78 in every seed; no earlier network without
  heads exceeded 0.61, and the heads network reached 0.76.
- **The code itself is arranged by count.** The count probe, fitted on collections only,
  orders English forms at 0.71 (dense 0.48) and Devanagari at 0.77 (0.62); its count error
  on withheld Chinese forms halves (3.38 → 1.79). The collection codes' participation ratio
  falls to 1.03: count lies almost entirely on one direction.
- **Costs:** Chinese paired at its own block 0.60 → 0.39, retention of Chinese and English
  at the end of training lower (−0.11, −0.13), English coherence AUROC lower (−0.11),
  succession on collections slightly lower (−0.05). Placement of withheld Chinese forms does
  not improve.
- **Random selection at the same limit** reproduces the confirmation's modest gains
  (comparison on all three scripts better, isolated judgment better), so the jump comes
  from the learned grouping, not the limit alone.

**Mechanism.** The live reader units in each seed converge on one code dimension, chosen
differently in each seed, and mostly read that dimension of both operands. All the
pressure of `g` and `P` then falls on one coordinate, and every sense's code has to carry
count there. RigL also left many reader units dead (13–20% in `g`, 37–62% in `P`),
shrinking the readers further. The two effects are not separated here.

## Results

Means over ten seeds. Outcome against dense: paired 95% t-interval rule.

| Measure | Dense | Random | RigL |
| --- | --- | --- | --- |
| **Transfer** | | | |
| Chinese withheld, own block | 0.35 | 0.40 inc | 0.28 inc |
| Devanagari withheld, own block | 0.19 | 0.28 inc | 0.36 bet |
| taps → Chinese form, withheld | 0.05 | 0.23 inc | **0.70 bet** |
| taps → Devanagari form, withheld | 0.33 | 0.41 inc | **0.70 bet** |
| form → taps, Chinese withheld | 0.29 | 0.36 inc | **0.66 bet** |
| Comparison, Chinese forms | 0.60 | 0.78 bet | **0.85 bet** |
| Comparison, Devanagari forms | 0.62 | 0.67 bet | **0.81 bet** |
| Comparison, English forms | 0.47 | 0.53 bet | **0.75 bet** |
| **Taught** | | | |
| Chinese paired, own block | 0.60 | 0.61 inc | 0.39 wor |
| taps → collection | 0.66 | 0.60 inc | 0.60 inc |
| Comparison, collections | 0.97 | 0.97 equ | 0.96 equ |
| Succession pairwise, collections | 0.92 | 0.92 equ | 0.87 inc |
| **Other** | | | |
| Chinese paired, end | 0.61 | 0.58 inc | 0.49 wor |
| English, end | 0.39 | 0.34 inc | 0.26 wor |
| Judgment revision: isolated, withheld | 0.23 | 0.35 bet | 0.39 bet |
| Judgment revision: revised, withheld | 0.58 | 0.60 inc | 0.56 inc |
| Next-judgment coherence AUROC, Chinese | 0.92 | 0.93 inc | 0.92 inc |
| Next-judgment coherence AUROC, English | 0.79 | 0.81 inc | 0.69 wor |
| Thai, 4 shots, after 200 updates | 0.18 | 0.16 equ | 0.17 equ |
| Probe order, Chinese forms | 0.59 | 0.67 inc | 0.79 inc |
| Probe order, Devanagari forms | 0.62 | 0.61 inc | 0.77 bet |
| Probe order, English forms | 0.48 | 0.50 inc | 0.71 bet |
| Probe count error, Chinese withheld | 3.38 | 2.77 | 1.79 |
| Probe count error, Devanagari withheld | 1.79 | 2.01 | 1.27 |
| Code participation ratio (collections) | 1.26 | 1.16 | 1.03 |

### RigL against random

| Measure | RigL − random [95% interval] |
| --- | --- |
| taps → Chinese form, withheld | +0.46 [+0.29, +0.64] better |
| taps → Devanagari form, withheld | +0.29 [+0.10, +0.49] better |
| form → taps, Chinese withheld | +0.29 [+0.16, +0.42] better |
| Comparison, Devanagari forms | +0.14 [+0.10, +0.18] better |
| Comparison, English forms | +0.21 [+0.18, +0.25] better |
| Comparison, Chinese forms | +0.08 [−0.13, +0.29] inconclusive |
| Chinese withheld, own block | −0.12 [−0.27, +0.04] inconclusive |
| Devanagari withheld, own block | +0.08 [−0.02, +0.18] inconclusive |
| Chinese paired, own block | −0.22 [−0.35, −0.10] worse |
| Chinese paired, end | −0.08 [−0.14, −0.03] worse |
| English, end | −0.08 [−0.14, −0.02] worse |
| Next-judgment coherence AUROC, English | −0.12 [−0.16, −0.08] worse |
| Probe order, Devanagari / English forms | +0.16 / +0.21, better |

Per seed, taps → withheld Chinese forms:

| Arm | Seeds 220–229 |
| --- | --- |
| Dense | 0.15, 0.00, 0.00, 0.17, 0.00, 0.00, 0.07, 0.03, 0.00, 0.10 |
| Random | 0.27, 0.00, 0.30, 0.15, 0.77, 0.05, 0.05, 0.22, 0.00, 0.53 |
| RigL | 0.63, 0.37, 0.75, 0.73, 0.70, 0.55, 0.77, 0.82, 0.83, 0.83 |

### Learned wiring

- **Connections changed:** 18.6% of the final `g` connections and 93% of `P`'s were not in
  the initial mask; 2,031 and 1,382 connections grown per run.
- **Dead units and an artifact.** Units whose first-layer ReLU is never on for evaluation
  codes have zero gradient everywhere, and `topk` over equal values returns index 10. Every
  dead unit therefore ends reading dimension 10 of both operands. The analysis's
  `overlap_final` (0.72 in `g`, 0.87 in `P`) counts these units and overstates tying.
  Checked in seeds 220, 221 and 224:

| Seed | Layer | Live units | Live: tied | Live: most-read dimension (connections of 2 × live) | Dead units |
| --- | --- | --- | --- | --- | --- |
| 220 | `g` | 56 | 0.64 | 11 (88 of 112) | 8 |
| 220 | `P` | 40 | 0.93 | 11 (77 of 80) | 24 |
| 221 | `g` | 55 | 0.40 | 7 (71 of 110) | 9 |
| 221 | `P` | 24 | 1.00 | 7 (46 of 48) | 40 |
| 224 | `g` | 51 | 0.61 | 9 (80 of 102) | 13 |
| 224 | `P` | 31 | 0.84 | 9 (55 of 62) | 33 |

  `g` and `P` converge on the **same** dimension within a seed, and it differs between
  seeds. The dead units' fixed reading of dimension 10 is an artifact of tie-breaking, not
  a learned choice; it does not affect outputs, since those units contribute a constant.

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| RigL against random: no difference | **Wrong:** large gains in cross-sense matching, comparison and the probe; losses in taught matching and retention |
| Random untied recurs in direction against dense | Correct: comparison better on all three scripts, isolated judgment better |
| Switching is frequent | Correct for `P` (93% changed); moderate for `g` (19%) |

## Interpretation

- **Learned grouping matters when it can concentrate.** With one dimension per operand,
  RigL's grow rule repeatedly picks the coordinate with the largest gradient, and the
  readers collapse onto one coordinate. The shared code then carries count on that
  coordinate for every sense, and never-paired forms land on it too. At 4 dimensions per
  operand and on the encoder, the same rule produced no such concentration and no effect.
- **This fits the project's working lesson, in its strongest form so far.** Less freedom in
  how the code is read arranges the code more: one coordinate read by every reader is close
  to the least freedom possible, and the code becomes nearly one-dimensional. The cost is
  the same as before in other networks: taught matching and retention fall when the code
  is squeezed onto one line.
- **It is close to a learned number line.** Nothing specifies which coordinate, or that
  there should be one; the dimension differs by seed and is found by the grow rule. But the
  outcome resembles the removed linear `larger` head (an ordinal score along one direction),
  reached through connectivity rather than a relation-specific form. Whether this counts as
  hand-given structure is a question for the researchers.
- **Not separated:** concentration on one coordinate, and the loss of reader capacity from
  dead units (up to 62% of `P`'s units). RigL's zero initialization of grown connections
  lets units die permanently.
- **Not established:** replication on fresh seeds; behaviour in the other curriculum arms;
  whether a fixed, unlearned concentration (every reader on one chosen coordinate) gives
  the same result.

## Computation

| | Value |
| --- | --- |
| Runs | 30 (3 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 20,303 s |
| Wall time per run, mean by arm | dense 341 s; random 350; RigL 341 |
| Job wall time | 1,795 s (data 28 s, rendering 3 s, runs 1,764 s) |
| Peak RSS, largest worker | 1.18 GB |
| Analysis | 36 s wall; dead-unit check about 1 minute (not part of the analysis script) |
| Selection and tuning | none |

## Caveats

- One run of ten seeds, one curriculum arm.
- The dead-unit check covered three seeds; the analysis's overlap diagnostic includes dead
  units and should be corrected before reuse.
