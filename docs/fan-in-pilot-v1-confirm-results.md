# Fan-in pressure pilot v1, confirmation: results

Design, contrasts and predictions: [proposal, "Confirmation after stage 1"](fan-in-pressure-proposal.md#confirmation-after-stage-1-written-before-running),
written before the run. Follows [stage 1](fan-in-pilot-v1-results.md). Run 2026-10-29, fresh
seeds 200–209, D-varied+rehearsal. Readers only: the first layers of `g` and `P` read one
code dimension per operand per hidden unit.

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.fan_in_pilot pilot confirm` (77 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot confirm` (49 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-confirm-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-confirm-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot-confirm/` (not tracked).

## Summary

- **Replication: not confirmed on the primary measure.** Untaught comparison on Chinese
  forms rises 0.64 → 0.72 with random tied readers, but the 95% interval includes zero
  (+0.08 [−0.01, +0.18]). The stage-1 effect (+0.23) was inflated by a low dense baseline in
  seeds 100–102 (0.58 there, 0.64 here) and a high limited result.
- **A smaller, broad effect of the limit does hold.** Across the four limited arms the
  direction is consistent, and several secondary measures pass the rule:
  - isolated judgment on withheld forms improves in **every** limited arm (+0.10 to +0.12);
  - Devanagari comparison (+0.05, tied), Chinese and English comparison (+0.06, +0.05,
    untied), Chinese placement (+0.08, tied) and Devanagari placement (+0.12, untied);
  - the count probe on English forms improves (+0.07 tied, +0.09 untied): here the code
    itself is better arranged, not only the reader.

  No taught measure falls past the margin.
- **Tying does not matter.** Untied against tied random selection: every measure
  inconclusive or equivalent, none better or worse. The stage-1 gain was not a
  per-dimension-comparator artifact; it comes from the limit. This contradicts the
  prediction that untied would be weaker.
- **Learned selection is no different from random, because it barely learned.** Only
  1.4% (tied) and 4.5% (untied) of `g`'s units switched dimension during training, and
  almost none of `P`'s. The learned untied arm reads the same dimension of both operands
  in 5.5% of units, about chance (1/16). **This does not answer whether learnable grouping
  helps**; it shows that this learning method, from this starting point, hardly moved.

## Results

Means over ten seeds. Outcome against dense: the paired 95% t-interval rule (**bet**ter,
**wor**se, **equ**ivalent, **inc**onclusive).

| Measure | Dense | Random tied | Random untied | Learned tied | Learned untied |
| --- | --- | --- | --- | --- | --- |
| **Primary** | | | | | |
| Comparison, Chinese forms | 0.64 | 0.72 inc | 0.70 bet | 0.73 inc | 0.74 bet |
| **Secondary: never directly paired forms** | | | | | |
| Chinese withheld, own block | 0.29 | 0.38 bet | 0.36 inc | 0.37 inc | 0.43 inc |
| Devanagari withheld, own block | 0.16 | 0.26 inc | 0.28 bet | 0.26 bet | 0.25 inc |
| **Secondary: untaught compositions** | | | | | |
| taps → Chinese form, withheld | 0.13 | 0.09 inc | 0.10 inc | 0.12 inc | 0.13 inc |
| taps → Devanagari form, withheld | 0.35 | 0.44 inc | 0.46 inc | 0.46 inc | 0.39 inc |
| form → taps, Chinese withheld | 0.24 | 0.20 inc | 0.34 inc | 0.26 inc | 0.18 inc |
| Comparison, Devanagari forms | 0.64 | 0.69 bet | 0.69 inc | 0.70 bet | 0.67 inc |
| Comparison, English forms | 0.50 | 0.54 equ | 0.55 bet | 0.52 inc | 0.53 inc |
| **Fit check: taught** | | | | | |
| Chinese paired, own block | 0.63 | 0.60 inc | 0.64 inc | 0.59 inc | 0.64 inc |
| taps → collection | 0.65 | 0.62 inc | 0.59 inc | 0.64 inc | 0.65 inc |
| Comparison, collections | 0.98 | 0.98 equ | 0.97 equ | 0.97 equ | 0.98 equ |
| Succession pairwise, collections | 0.91 | 0.93 equ | 0.92 equ | 0.92 equ | 0.93 equ |
| **Other** | | | | | |
| Chinese paired, end | 0.60 | 0.55 inc | 0.57 inc | 0.58 inc | 0.58 inc |
| English, end | 0.38 | 0.34 inc | 0.37 inc | 0.36 inc | 0.36 inc |
| Judgment revision: isolated, withheld | 0.21 | 0.31 bet | 0.33 bet | 0.32 bet | 0.31 bet |
| Judgment revision: revised, withheld | 0.60 | 0.65 inc | 0.64 inc | 0.65 inc | 0.64 inc |
| Next-judgment coherence AUROC, Chinese | 0.92 | 0.93 equ | 0.93 equ | 0.92 inc | 0.92 equ |
| Next-judgment coherence AUROC, English | 0.81 | 0.81 inc | 0.82 inc | 0.83 inc | 0.79 inc |
| Thai, 4 shots, after 200 updates | 0.17 | 0.18 equ | 0.18 inc | 0.17 equ | 0.15 inc |
| Probe order, Chinese forms | 0.59 | 0.58 inc | 0.61 inc | 0.60 inc | 0.62 inc |
| Probe order, Devanagari forms | 0.63 | 0.64 equ | 0.64 equ | 0.65 inc | 0.61 equ |
| Probe order, English forms | 0.49 | 0.56 bet | 0.58 bet | 0.55 bet | 0.53 inc |
| Code participation ratio (collections) | 1.27 | 1.18 | 1.20 | 1.18 | 1.19 |

Chinese comparison per seed (200–209):

| Arm | Values |
| --- | --- |
| Dense | 0.43, 0.61, 0.73, 0.79, 0.52, 0.86, 0.72, 0.70, 0.43, 0.60 |
| Random tied | 0.65, 0.66, 0.84, 0.82, 0.80, 0.89, 0.84, 0.54, 0.64, 0.55 |
| Random untied | 0.52, 0.62, 0.84, 0.85, 0.62, 0.87, 0.72, 0.67, 0.67, 0.60 |
| Learned tied | 0.78, 0.77, 0.57, 0.87, 0.74, 0.62, 0.98, 0.44, 0.75, 0.78 |
| Learned untied | 0.71, 0.62, 0.93, 0.82, 0.73, 0.90, 0.86, 0.64, 0.52, 0.71 |

### Declared contrasts

Mean paired difference [95% t-interval], outcome.

| Measure | Tied − dense | Untied − dense | Untied − tied | Learned − random (tied) | Learned − random (untied) |
| --- | --- | --- | --- | --- | --- |
| Comparison, Chinese forms | +0.08 [−0.01, +0.18] inc | +0.06 [+0.00, +0.11] bet | −0.02 [−0.09, +0.04] inc | +0.01 [−0.12, +0.13] inc | +0.05 [−0.02, +0.12] inc |
| Comparison, Devanagari forms | +0.05 [+0.02, +0.09] bet | +0.05 [+0.01, +0.08] inc | −0.01 [−0.04, +0.03] equ | +0.01 [−0.02, +0.04] equ | −0.02 [−0.04, +0.01] equ |
| Comparison, English forms | +0.03 [+0.02, +0.05] equ | +0.05 [+0.01, +0.09] bet | +0.02 [−0.02, +0.05] inc | −0.02 [−0.05, +0.02] equ | −0.02 [−0.07, +0.03] inc |
| Chinese withheld, own block | +0.08 [+0.01, +0.16] bet | +0.06 [−0.06, +0.19] inc | −0.02 [−0.15, +0.11] inc | −0.00 [−0.11, +0.10] inc | +0.07 [−0.06, +0.21] inc |
| Devanagari withheld, own block | +0.10 [−0.03, +0.22] inc | +0.12 [+0.03, +0.22] bet | +0.03 [−0.09, +0.15] inc | +0.01 [−0.10, +0.11] inc | −0.03 [−0.17, +0.11] inc |
| Judgment revision: isolated, withheld | +0.10 [+0.04, +0.16] bet | +0.12 [+0.06, +0.18] bet | +0.02 [−0.07, +0.11] inc | +0.00 [−0.06, +0.06] inc | −0.02 [−0.13, +0.09] inc |
| Probe order, English forms | +0.07 [+0.02, +0.12] bet | +0.09 [+0.04, +0.13] bet | +0.02 [−0.06, +0.10] inc | −0.01 [−0.07, +0.05] inc | −0.04 [−0.11, +0.02] inc |

The full contrast table, including every secondary and fit measure, is in the comparison
artifact. Only one contrast is worse anywhere: English coherence AUROC, learned untied
against random untied (−0.03 [−0.05, −0.01]).

### Learned selection

| Arm | `g` units switched | `P` units switched | `g` units reading the same dimension of both operands |
| --- | --- | --- | --- |
| Learned tied | 1.4% | 0.0% | 100% (by construction) |
| Learned untied | 4.5% | 0.8% | 5.5% (chance about 6.3%) |

## Interpretation

- **The fan-in limit on the readers gives a modest, broad gain in untaught transfer**,
  about +0.05 to +0.12 on several measures, with no loss in taught fit. It is smaller than
  stage 1 suggested and does not pass the rule on the declared primary measure for the
  tied arm. Three seeds overstated it; ten seeds put it in the range where single measures
  are near the decision boundary.
- **The gain comes from the limit, not from tying.** Random untied selection does as well
  as tied. The limit therefore is not a hidden per-dimension comparison form; it is less
  freedom in how the reader mixes the code, consistent with the
  ["more flexible answerer arranges less"](think-pilot-v1-results.md#interpretation) pattern.
- **In part the code is better arranged too.** The count probe on English forms improves in
  three of four limited arms, and isolated judgments on withheld forms improve in all four.
  This is a change from stage 1, where the probe showed no change.
- **The learnable-grouping question is still open.** Selections started one logit unit ahead
  of the alternatives and trained at the shared learning rate (1e-3), and hardly any
  switched. Two further reasons the test is weak here:
  - the collection codes' variance lies mostly along one direction (participation ratio
    about 1.2), so with one dimension per operand many choices carry similar information
    and random choice may already be near the best;
  - the comparison is learned-from-a-random-start against that random start; a learned arm
    that cannot move is the random arm.

  A fair test needs a selection mechanism that actually explores (for example a smaller
  initial margin, a higher learning rate for the logits, or prune-and-regrow), and a
  setting where choice matters (wider inputs, or groups of more than one).

## Computation

| | Value |
| --- | --- |
| Runs | 50 (5 arms × 10 seeds), five waves on 11 workers |
| Worker CPU, all runs | 48,159 s |
| Worker CPU per run, mean by arm | dense 935 s; random tied 978; random untied 993; learned tied 992; learned untied 918 |
| Job wall time | 4,628 s (data 31 s, rendering 5 s, runs 4,592 s) |
| Peak RSS, largest worker | 1.22 GB |
| Analysis | 49 s wall |
| Selection and tuning | none |

- Learned arms add 1,024 (tied) or 2,048 (untied) selection logits per reader layer and
  compute a 16-way softmax per unit per update; their per-run CPU is within the spread of
  the other arms.

## Caveats

- One curriculum arm. Ten seeds.
- Masks and starting selections are one draw per seed and layer.
- The learned arms test one selection method from one starting point.
