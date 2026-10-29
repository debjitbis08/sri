# Fan-in pressure pilot v1, learned grouping (RigL): results

Design, contrasts and predictions: [proposal, "Learned grouping"](fan-in-pressure-proposal.md#learned-grouping-written-before-running),
written before the run. Follows the [confirmation](fan-in-pilot-v1-confirm-results.md), whose
learned selection hardly moved. Run 2026-10-30, fresh seeds 210–219, D-varied+rehearsal.

- **Commands:**
  - `NSY_WORKERS=6 NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_pilot pilot rigl` (50 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot rigl` (42 s; it reads the run's
    thread and layout settings from the runtime file).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-rigl-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-rigl-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-rigl/` (not tracked).

## Summary

**Learning which units combine gives no benefit over random choice.** Prune and regrow
changed many connections, but every learned-against-random contrast, 24 measures in each
of the two settings, is equivalent or inconclusive. None is better or worse.

- **The learner moved.** 26–31% of the final encoder connections and 20–35% of the reader
  connections were not in the initial mask. Each layer grew about five times as many
  connections as it holds, so connections were swapped repeatedly.
- **Encoder limit at φ = 1/16: the harm replicates and RigL does not reduce it.** Against
  dense, both random and RigL arms are worse on Devanagari placement, taps → withheld
  Devanagari forms, Chinese comparison, isolated judgment on withheld forms, English
  coherence and the Chinese count probe, by nearly identical amounts.
- **Readers at 4 dimensions per operand, untied: no effect of the limit and none of
  learning.** Every contrast against dense is equivalent or inconclusive, consistent with
  stage 1, where readers at 4 per operand (tied) had a weaker effect than at 1.
- **Reader operands drifted slightly toward the same dimensions** (overlap 0.99 → 1.21 of
  4 in `g`, 0.94 → 1.31 in `P`). The confirmation found that tying does not matter, and
  this drift changed no outcome.

## Results

Means over ten seeds. Outcome against dense: the paired 95% t-interval rule (**bet**ter,
**wor**se, **equ**ivalent, **inc**onclusive).

| Measure | Dense | Encoder random | Encoder RigL | Readers random | Readers RigL |
| --- | --- | --- | --- | --- | --- |
| **Transfer** | | | | | |
| Chinese withheld, own block | 0.34 | 0.32 inc | 0.26 inc | 0.36 inc | 0.35 inc |
| Devanagari withheld, own block | 0.25 | 0.15 wor | 0.16 wor | 0.29 inc | 0.30 inc |
| taps → Chinese form, withheld | 0.04 | 0.09 inc | 0.07 inc | 0.10 inc | 0.08 inc |
| taps → Devanagari form, withheld | 0.43 | 0.21 wor | 0.21 wor | 0.43 equ | 0.39 inc |
| form → taps, Chinese withheld | 0.39 | 0.24 inc | 0.29 inc | 0.26 inc | 0.28 inc |
| Comparison, Chinese forms | 0.65 | 0.53 wor | 0.54 wor | 0.67 inc | 0.67 inc |
| Comparison, Devanagari forms | 0.61 | 0.58 inc | 0.58 inc | 0.65 inc | 0.63 inc |
| Comparison, English forms | 0.49 | 0.46 inc | 0.47 equ | 0.51 equ | 0.50 equ |
| **Taught** | | | | | |
| Chinese paired, own block | 0.65 | 0.62 inc | 0.61 inc | 0.61 inc | 0.62 inc |
| taps → collection | 0.67 | 0.68 inc | 0.69 inc | 0.68 inc | 0.61 inc |
| Comparison, collections | 0.98 | 0.98 equ | 0.98 equ | 0.97 equ | 0.98 equ |
| Succession pairwise, collections | 0.91 | 0.93 equ | 0.93 inc | 0.91 equ | 0.91 equ |
| **Other** | | | | | |
| Chinese paired, end | 0.60 | 0.64 inc | 0.63 inc | 0.61 inc | 0.58 inc |
| English, end | 0.40 | 0.40 inc | 0.43 inc | 0.35 inc | 0.36 inc |
| Judgment revision: isolated, withheld | 0.31 | 0.21 wor | 0.22 wor | 0.30 inc | 0.33 inc |
| Judgment revision: revised, withheld | 0.68 | 0.68 inc | 0.67 inc | 0.63 inc | 0.65 inc |
| Next-judgment coherence AUROC, Chinese | 0.92 | 0.92 equ | 0.92 equ | 0.90 inc | 0.91 equ |
| Next-judgment coherence AUROC, English | 0.80 | 0.69 wor | 0.69 wor | 0.79 inc | 0.80 inc |
| Thai, 4 shots, after 200 updates | 0.17 | 0.20 inc | 0.19 inc | 0.17 equ | 0.16 equ |
| Probe order, Chinese forms | 0.66 | 0.50 wor | 0.54 wor | 0.64 inc | 0.68 inc |
| Probe order, Devanagari forms | 0.60 | 0.57 inc | 0.58 inc | 0.63 inc | 0.62 inc |
| Probe order, English forms | 0.50 | 0.47 inc | 0.47 inc | 0.51 equ | 0.49 equ |
| Code participation ratio (collections) | 1.30 | 1.32 | 1.41 | 1.16 | 1.18 |

### Learned against random

Mean paired difference [95% t-interval], outcome. Transfer measures and the fit check.

| Measure | Encoder: RigL − random | Readers: RigL − random |
| --- | --- | --- |
| Chinese withheld, own block | −0.06 [−0.18, +0.07] inc | −0.01 [−0.11, +0.09] inc |
| Devanagari withheld, own block | +0.01 [−0.04, +0.07] inc | +0.00 [−0.11, +0.11] inc |
| taps → Chinese form, withheld | −0.02 [−0.08, +0.03] inc | −0.02 [−0.16, +0.12] inc |
| taps → Devanagari form, withheld | +0.00 [−0.11, +0.11] inc | −0.04 [−0.19, +0.10] inc |
| form → taps, Chinese withheld | +0.05 [−0.15, +0.26] inc | +0.02 [−0.22, +0.27] inc |
| Comparison, Chinese forms | +0.00 [−0.03, +0.03] equ | +0.00 [−0.08, +0.08] inc |
| Comparison, Devanagari forms | −0.00 [−0.03, +0.03] equ | −0.02 [−0.04, −0.00] equ |
| Comparison, English forms | +0.01 [+0.00, +0.02] equ | −0.02 [−0.04, +0.00] equ |
| Chinese paired, own block | −0.01 [−0.07, +0.05] inc | +0.00 [−0.05, +0.06] inc |
| taps → collection | +0.00 [−0.05, +0.06] inc | −0.07 [−0.16, +0.02] inc |
| Comparison, collections | +0.00 [−0.00, +0.00] equ | +0.00 [−0.00, +0.01] equ |
| Succession pairwise, collections | −0.00 [−0.02, +0.02] equ | +0.01 [−0.01, +0.02] equ |

The remaining measures, and every contrast against dense, are in the comparison artifact.

### Connection changes

Means over seeds.

| Layer | Final connections not in the initial mask | Connections grown during training | Operand overlap, initial → final |
| --- | --- | --- | --- |
| image `out` (512 kept) | 27.6% | 2,519 | — |
| sequence `out` (512 kept) | 25.6% | 2,548 | — |
| image `proj` (64 kept) | 30.2% | 577 | — |
| sequence `proj` (64 kept) | 30.6% | 509 | — |
| `g` first layer (512 code connections kept) | 19.5% | 4,193 | 0.99 → 1.21 of 4 |
| `P` first layer (512 kept) | 34.7% | 4,152 | 0.94 → 1.31 of 4 |

## Predictions against outcomes

| Prediction | Outcome |
| --- | --- |
| RigL matches or beats random on taught fit | Matches (all equivalent or inconclusive) |
| RigL reduces the encoder limit's harm to transfer | **Not supported:** harm is the same with and without RigL |
| Readers: little difference | Correct |
| More than 30% of connections change | Partly: 20–35% by layer. Swaps were frequent (about five grown per kept connection), but most grown connections were dropped again before the end |

## Interpretation

- **Which units combine does not matter here; how many do, and where, does.** Across three
  experiments the fan-in limit's effect depends on the site (readers help a little at one
  dimension per operand; the encoder is harmed at 1/16) and on its strength. Random,
  straight-through and prune-and-regrow selection give the same results at the same limit.
- **Why learning the grouping may not help transfer.** RigL keeps the connections that
  matter most for the training loss: large weights and large gradients on taught
  questions. Learning the grouping from the same objective returns some of the freedom
  the limit removed, and it is spent on what is taught, not on placing what is not.
  Taught fit was already unaffected by random limits, so there was little for RigL to
  gain. This is an interpretation, not tested here; a grouping criterion aimed at
  transfer would need a different signal than the training loss.
- **The encoder harm is robust.** It replicates on fresh seeds with nearly the same
  magnitudes, and it is not a matter of unlucky random wiring.
- **Reader limits help only when strong.** At 4 per operand nothing passes the rule; the
  modest gains in the confirmation were at 1 per operand.

## Computation

| | Value |
| --- | --- |
| Runs | 50 (5 arms × 10 seeds), 6 workers × 2 threads, channels-last image encoder |
| Worker CPU, all runs | 33,825 s |
| Wall time per run, mean by arm | dense 354 s; encoder random 352; encoder RigL 350; readers random 333; readers RigL 332 |
| Job wall time | 2,993 s (data 28 s, rendering 3 s, runs 2,962 s) |
| Peak RSS, largest worker | 1.18 GB |
| Analysis | 42 s wall |
| Selection and tuning | none; RigL's interval, fraction, schedule and end point were fixed before the run |

- RigL's connectivity updates (45 per run, a dense gradient and a per-unit top-k each)
  add no measurable time.
- Compared with the confirmation's layout (11 workers × 1 thread, 966 s per run, 4,628 s
  for 50 runs), this run took 65% of the wall time for the same number of runs.

## Caveats

- One curriculum arm, ten seeds.
- One learning method for grouping (RigL with its default settings), and two settings.
- Different numerical settings from the earlier experiments (threads, layout); no arm is
  compared across experiments.
