# Number core network v1: pilot results

Date: 2026-10-28. Specification: [number core network v1](number-core-network-v1-spec.md)
(draft; not frozen). Pilot attempt 1. Command:
`.venv/bin/python experiments/number_core_network.py pilot` (1,000 steps, the runner
default for the pilot). Artifacts: `reports/number-core-network-v1-pilot-results.json`
(deterministic), `-runtime.json`. Checkpoints in gitignored `runs/`.

Scope: seed 100, all eight arms, 1,000 steps (a third of the spec budget), validation
pools only (MNIST validation reserve, dice and Morse validation pools). This is a
pipeline check and a first look, not headline evidence, and selects nothing.

## Pipeline checks

| Check | Outcome |
| --- | --- |
| Unit tests (Morse window bound, scramble derangement, Hessian equals the term-by-term energy for both topologies, zero gradient at equilibrium, orthonormal label codes) | pass |
| Morse overflow | none; latest end sample 105 of 128 over train and validation pools |
| Exact solve | relative residual 1e-8 to 1.2e-7 in every arm and modality (largest in the inner-only arms) |
| Numerical failures | none |
| MNIST label accuracy | 0.96–0.98 |
| Labelled dice and Morse counts (1, 3, 5) | 0.99–1.00 |

## First look

Validation accuracy, seed 100, one run per arm. Labels are the ten-way argmax over
label codes (uniform baseline 0.10).

| Arm | Dice 2, 4 label | Morse 2, 4 label | Dice 2, 4 nearest MNIST core centroid | Morse 2, 4 nearest MNIST core centroid | Dice 2, 4 *same* judge | Morse 2, 4 *same* judge | Core-inner CKA Morse–MNIST | Core-inner CKA dice–MNIST |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-M | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.34 | 0.79 |
| S-M | 0.00 | 0.00 | 0.48 | 0.43 | **1.00** | **0.99** | 0.66 | 0.89 |
| N-X | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.28 | 0.71 |
| S-X | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.66 | 0.53 | 0.79 |
| N-X-E | 0.00 | 0.47 | 0.00 | 0.48 | 0.00 | 0.46 | 0.21 | 0.72 |
| S-X-E | 0.00 | 0.00 | 0.48 | 0.18 | **1.00** | 0.60 | 0.53 | 0.90 |
| N-X-I | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.76 |
| S-X-I | 0.00 | 0.02 | 0.01 | 0.69 | 0.42 | 0.89 | 0.72 | 0.87 |

The MNIST–MNIST core-inner CKA reference is 0.81–0.88. Extrapolation (dice 6,
Morse 6–9) is near zero in every arm and readout. *Next* from Morse 0 reaches digit 1
in S-X-E and S-X-I (0.99). MNIST composition `S^k` on core-inner codes is
0.73–0.84 for k = 1, falling to 0.09–0.50 for k = 3. In the cross-relation pilot it
was 0.98–0.99.

### The label readout excludes unlabelled counts

Every arm labels unlabelled dice and Morse counts 0.00. The modal predicted label for
an unlabelled count is always a neighbouring labelled count. In S-M, dice
2 → 1, 4 → 5 and 6 → 5; Morse 0 → 1, 2 → 1, 4 → 5 and 6, 7 → 5. The readout
respects order but never outputs an unlabelled number for dice or Morse.

The cause is in the loss [derived]. Softmax cross-entropy over all ten codes,
applied to dice and Morse only at 1, 3 and 5, teaches the output path that dice and
Morse inputs are never 2, 4 or 6. That is negative evidence the spec did not intend:
"labels only at 1, 3 and 5" became "dice and Morse only take labels 1, 3 and 5".
Under this objective, the label readout cannot test transfer.

### Core codes partly align with succession

The readouts that do not pass through the label path tell a different story:

- With succession, the core-inner judge says *same* for dice 2 and 4 against digits
  2 and 4 in S-M and S-X-E (1.00), and for Morse 2 and 4 in S-M (0.99). Without
  succession these are 0.00.
- Core-inner CKA between Morse and MNIST roughly doubles with succession (0.53–0.72
  against 0.21–0.34), approaching the MNIST–MNIST reference in S-X-I.
- Nearest-centroid readout of the core is about 0.5 on dice 2 and 4 with succession,
  and 0 without.

So alignment of unlabelled interpolation counts appears in the core, in some arms,
after a third of the planned budget. Labels alone (S-X, scrambled Morse) align less
than labels plus equivalence (S-X-E) for dice. The Morse picture is mixed. One seed
cannot separate these arms.

### Capacity and budget

Composition degrades quickly with k, and losses in the succession arms were still
1.0–1.9 at step 1,000. The core inner layer has 4 units, and the whole SRI part is a
fixed linear map from the 16 front-end features at equilibrium [derived]. Both the
short budget and the narrow core may limit the successor. The pilot cannot separate
them.

## Amendments to consider before freezing

1. **Label loss without negative evidence for sparse modalities.** For example,
   pull the output toward the correct code (squared error to `q_k`) instead of
   softmax over all ten codes. That still tells dice and Morse what 1, 3 and 5 are,
   without telling them the other numbers are excluded. Alternatively, apply the
   softmax over all ten codes only for MNIST. This needs to be decided before
   freezing, since it defines the primary measure.
2. **Wider modules.** The user expects numbers need more units. A size factor (for
   example 64 → 32 → 16 against 16 → 8 → 4) is cheap with exact solves; the solve
   grows to about 270 free activities.
3. **Budget.** Run the spec's 3,000 steps, or check convergence in a second pilot.

## Cost

Wall 364 s on 8 workers (one fit each), worker CPU 2,069 s, largest worker peak RSS
0.81 GB. Training per fit: 158–357 s (no-succession arms fastest, S-X-E slowest).
8,000 optimizer steps, 7.17 M training encodings, 16,016 Cholesky factorizations.
Data preparation is included in the wall time. A 20-step smoke run (15 s wall) checked
the pipeline first. Both count as selection cost.

# Pilot 2: amended label loss and wider modules

Date: 2026-10-28. Command: `.venv/bin/python experiments/number_core_network.py pilot2`.
Artifacts: `reports/number-core-network-v1-pilot2-results.json`, `-runtime.json`.
Changes from pilot 1, both from its amendments:

1. **Label loss:** softmax over the ten codes for MNIST only. Dice and Morse use a
   positive-only cosine pull toward the correct code, `1 − cos(x_out, q_k)`, which
   never pushes away the other numbers.
2. **Width:** all eight arms at 64 → 32 → 16 (272 free activities per solve), plus
   S-M and S-X-E at the original 16 → 8 → 4 for comparison. Per-unit coupling
   strengths are unchanged (uniform 1/7 per unit, inner-only 1 per unit).

Same seed (100), 1,000 steps and validation pools. Pipeline checks pass again:
solve residuals 1e-8 to 7e-8, Morse end at most sample 105, no failures, MNIST
labels 0.95–0.98, and labelled dice and Morse 0.99–1.00.

## Results (one seed; descriptive)

| Arm | Dice 2, 4 label | Morse 2, 4 label | Dice 2, 4 / 6 nearest MNIST core centroid | Morse 2, 4 nearest centroid | Judge *same*: labelled dice / dice 2, 4 / dice 6 | Judge *same*: Morse 2, 4 | Core-inner CKA Morse / dice vs MNIST | MNIST `S^k`, k = 1 / 3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-M wide | 0.00 | 0.00 | 0.50 / 0.00 | 0.00 | 1.00 / 0.50 / 0.00 | 0.13 | 0.21 / 0.50 | — |
| N-X wide | 0.01 | 0.00 | 0.01 / 0.00 | 0.00 | 1.00 / 0.50 / 0.00 | 0.00 | 0.14 / 0.44 | — |
| N-X-E wide | 0.02 | 0.00 | 0.00 / 0.00 | 0.48 | 1.00 / 0.00 / 0.00 | 0.20 | 0.08 / 0.43 | — |
| N-X-I wide | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 | 1.00 / 0.00 / 0.00 | 0.00 | 0.18 / 0.48 | — |
| S-M narrow | 0.00 | 0.00 | 0.00 / 0.00 | 0.00 | 1.00 / 0.91 / 0.15 | 0.46 | 0.54 / 0.82 | 0.79 / 0.34 |
| S-M wide | 0.00 | 0.00 | 0.03 / 0.00 | 0.00 | 0.33 / 0.18 / 0.00 | 0.01 | 0.43 / 0.73 | 0.94 / 0.69 |
| S-X wide | 0.00 | 0.00 | 0.50 / 0.00 | 0.00 | 0.00 / 0.50 / 0.00 | 0.00 | 0.27 / 0.80 | 0.93 / 0.73 |
| S-X-E narrow | 0.00 | 0.00 | 0.00 / 0.29 | 0.00 | 1.00 / 1.00 / 0.82 | 0.77 | 0.42 / 0.87 | 0.89 / 0.53 |
| **S-X-E wide** | 0.00 | 0.00 | **0.79 / 0.79** | **0.81** | **1.00 / 1.00 / 1.00** | **0.98** | 0.55 / 0.93 | 0.93 / 0.63 |
| S-X-I wide | 0.00 | 0.00 | 0.00 / 0.03 | 0.13 | 0.67 / 0.50 / 0.89 | 0.97 | 0.42 / 0.80 | 0.96 / 0.90 |

MNIST–MNIST core-inner CKA reference: 0.82–0.87. Morse 6–9 stays near zero in every
readout. *Next* from Morse 0 reaches digit 1 in S-X-E (0.88–0.99).

## What changed and what did not

- **The label readout still never transfers** (0.00–0.02 on unlabelled counts),
  even without negative evidence. Unlabelled counts still map to the neighbouring
  labelled label: in S-X-E wide, dice 2 → 3, 4 → 5, 6 → 5. This happens although
  dice 2's core state is closest to MNIST 2's core mean (cosine 0.82, against 0.61
  for 1 and 0.66 for 3). In the output layer, the same dice-2 state is closest to
  MNIST 3 (0.81, against 0.41 for 2). So the core → output path turns an approximately
  aligned core state into the wrong label. It was fitted on dice only at 1, 3 and 5,
  so it is sensitive to exactly the dice-specific directions in which dice 2 still
  differs from MNIST 2. The orthonormal label codes have no order, so a state between
  1 and 3 reads out as whichever labelled code is nearer, never as 2 [derived from
  these measurements; one seed].
- **Labels alone do not make labelled codes coincide in the core.** In S-X wide
  (labels, succession, scrambled Morse, no equivalence loss), the core judge rejects
  every labelled dice count as the same as its MNIST counterpart (0.00), although
  both get the correct label. This is the prior-work deduction (labels constrain only
  what the output sees), observed directly.
- **With the equivalence loss and wide modules (S-X-E wide), unlabelled counts align
  in the core:** judge *same* 1.00 for dice 2, 4 and 6 and 0.98 for Morse 2 and 4;
  nearest MNIST core centroid 0.79–0.81. This is the clearest transfer so far, but
  one seed only.
- **Width helps the successor.** MNIST composition at k = 3 rises from 0.34–0.53
  (narrow) to 0.63–0.90 (wide). For alignment the width effect is mixed: S-X-E
  improves with width, S-M gets worse. One seed cannot separate width from seed
  variation.
- **Extrapolation beyond the labelled range upward (Morse 6–9) fails everywhere.**
  Dice 6 aligns only in S-X-E wide and partly S-X-I wide.

## Implications for the specification

1. **The label output cannot carry the primary measure under sparse labels.** It
   turns approximate core alignment into a neighbouring labelled class, by
   construction of an unordered code space fitted on three classes. The cross-modal
   question is answered in the core: the core judge, nearest MNIST core centroid and
   coupling measures. Propose these as primary, keep the label readout as a reported
   secondary measure, and state why.
2. **Shared labels are not a sufficient bridge in this architecture**; the
   equivalence loss is. That answers the reviewer's question in the direction the
   prior-work deduction predicted, pending seeds. It argues for making equivalence
   the main condition and label-only the comparison, rather than the reverse.
3. **Wide modules as default**, given the composition gain.

## Cost

Wall 494 s on 10 workers, worker CPU 3,836 s, largest worker peak RSS 0.81 GB.
Training per fit: 270–487 s. 10,000 optimizer steps, 9.6 M training encodings, 20,020
Cholesky factorizations. A 20-step smoke run (18 s wall) preceded it. All of this
counts as selection cost.

# Tally-mark test on the current structure

Date: 2026-10-28. Exploratory; requested by the user to see how badly the current
structure does. Command: `.venv/bin/python experiments/number_core_network.py tally`.
Artifacts: `reports/number-core-network-v1-tally-results.json`, `-runtime.json`.

**Setup.**
- New generator (`src/number_core.py`, `build_tally`): tally marks for 1–10 on
  28 × 28. Gates are four strokes struck by a diagonal; 6–9 are a gate plus loose
  strokes, 10 is two gates. Stroke length, width, spacing and slant vary, with
  per-stroke jitter, random position, brightness and noise.
- Tallies go through the image encoder, with labels only at 1, 2 and 5 (cosine
  pull, as for dice and Morse in pilot 2).
- Two variants: tally labels only, and labels plus within-tally *same*/*next*
  relations at all counts.
- Arms N-M, S-M and S-X-E at wide width; seed 100, 1,000 steps, validation pools.
- The output has codes only for 0–9, so 10 cannot be labelled correctly; its
  modal label is reported.

**Result: it does as badly as possible on the label readout.** Labelled tallies 1,
2 and 5 are 0.99–1.00. Every unlabelled tally is 0.00–0.08 in every arm and variant.
3, 4 and 6–10 are all labelled **5** (once 3 → 2). The readout behaves like three bins
ordered by amount of ink: one stroke, two strokes, more.

| Arm / tally training | Label 3, 4 | Label 6–9 | Modal label for 3, 4, 6–10 | Nearest MNIST core-inner centroid 3, 4 / 6–9 | Judge *same* 3, 4 / 6–9 | Judge *next* 3, 4 | Core-inner CKA tally–MNIST |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N-M / labels | 0.00 | 0.01 | 5 | 0.04 / 0.01 | 0.00 / 0.14 | — | 0.35 |
| N-M / relations | 0.00 | 0.00 | 5 | 0.00 / 0.00 | 0.01 / 0.00 | — | 0.25 |
| S-M / labels | 0.00 | 0.00 | 5 | 0.31 / 0.00 | 0.29 / 0.06 | 0.12 | 0.43 |
| S-M / relations | 0.00 | 0.00 | 2, 5 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 | 0.32 |
| S-X-E / labels | 0.00 | 0.02 | 5 | 0.05 / 0.03 | 0.01 / 0.07 | 0.16 | 0.46 |
| S-X-E / relations | 0.00 | 0.00 | 5 | 0.38 / 0.00 | 0.71 / 0.00 | 0.89 | 0.40 |

- Only one condition shows any core-level alignment for tallies: S-X-E with tally
  relations, for interpolation (3, 4: judge *same* 0.71, *next* 0.89). Tally marks
  are not in the equivalence set there; the alignment comes through succession,
  via the dice and Morse anchors.
- Extrapolation (6–9) is 0.00–0.14 in every readout. The gate structure is never
  recovered. Composing "a gate plus strokes" from 5 and 1/2 does not happen.
- MNIST stays at 0.97.

**Reading.** Consistent with the probe result: the core does not hold a shared
number code. A new input kind with three labels lands in its own region, its
readout learns three bins, and succession helps at most between the labelled
counts. One seed.

**Cost.** Wall 403 s on 6 workers (data preparation 20 s, including 20,000 training
and 3,000 validation tally renders), worker CPU 1,653 s, largest worker peak RSS
0.86 GB. 6,000 optimizer steps, 7.04 M training encodings, 12,012 factorizations.
A 20-step smoke run (30 s wall) preceded it.
