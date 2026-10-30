# Arbitrary pairings v1: Option A pilot (seed 299)

Debugging and sizing run for the [contract](arbitrary-pairings-v1-contract.md), Option A
(inherited schedules, rewiring and resets stopped at update 4,500), one seed (299), six arms,
six pilot configurations: pair-set size N in {18, 45, 90} × pairing-block length L in {1,000, 2,000}.
Run 2026-10-30 on the GPU (six processes) at revision `218300f`. **It yields no reading;** one seed
cannot compare arms, and the arms' values are shown only to check floor, ceiling and the controls.

- **Command:** `NAP_OPTION=A NAP_N=<N> NAP_L=<L> NAP_SEEDS=299 NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.arbitrary_pairings pilot pairs`
- **Artifacts:** `reports/arbitrary-pairings-v1-pairs-A-N{N}-L{L}-{results.json.gz,runtime.json}`.

## What the checks show

Recall on A at the end of block 4, chance 1/9 = 0.111, by arm (sparse48, dense, mixed, mixed-h125, mixed-h500, mixed-reset):

| N, L | recall on A | rescored vs shuffled labels | `P-A'` vs A (sparse48, mixed) |
|---|---|---|---|
| 18, 1,000 | 0.28 0.40 0.24 0.33 0.29 0.33 | 0.17–0.24 | 0.28 vs 0.28; 0.26 vs 0.24 |
| 45, 1,000 | 0.27 0.31 0.24 0.27 0.27 0.32 | 0.05–0.12 | 0.20 vs 0.27; 0.24 vs 0.24 |
| 90, 1,000 | 0.21 0.27 0.15 0.21 0.18 0.28 | 0.08–0.12 | 0.17 vs 0.21; 0.15 vs 0.15 |
| 18, 2,000 | 0.33 0.28 0.26 0.40 0.27 0.32 | 0.13–0.24 | 0.34 vs 0.33; 0.26 vs 0.26 |
| 45, 2,000 | 0.24 0.30 0.27 0.35 0.27 0.32 | 0.06–0.12 | 0.26 vs 0.24; 0.30 vs 0.27 |
| 90, 2,000 | 0.22 0.24 0.19 0.20 0.24 0.31 | 0.10–0.13 | 0.15 vs 0.22; 0.19 vs 0.19 |

- **No ceiling.** No arm exceeds 0.40 at any size.
- **Near the floor for large N.** At N = 90 several arms sit at 0.15–0.24, within about 0.1 of chance; at
  N = 45 every arm is at least 0.24 at L = 2,000.
- **Rescoring against shuffled labels** is at chance for N = 45 and 90 (0.05–0.13). For N = 18 it reaches
  0.17–0.24: with 18 symbols the score is noisy (sd about 0.07, since a symbol's renderings pick alike),
  so N = 18 cannot confirm the check.
- **Learning check (`P-A'` against A):** the gaps are 0.00–0.07 (0.07 at N = 45, L = 1,000 in `sparse48` and at
  N = 90, L = 2,000 in `sparse48`). With one seed and a few dozen items that is within noise, but it is a sanity
  check, not clearance: equal performance on two assignments does not rule out a shortcut common to both.
- **Rescoring against shuffled labels** is likewise a sanity check. One permutation fluctuates widely; the study
  now reports the mean and spread over 100 permutations.
- **The lookup does not show a recognition ceiling (corrected after external review).** The nearest-neighbor lookup
  measured count-label accuracy in the learned embedding, 0.22–0.57 and often above the network's recall. That shows
  another decoder recovers more from the embedding, not that recognition is the dominant limit: different symbols
  share counts and the embedding changes with pairing training. The runner now also reports recall on the training
  renderings and 1-nearest-neighbor symbol-identity accuracy, so recognition and association learning can be separated:
  high recall on training renderings with poor recall on independent ones supports a recognition bottleneck; both low
  leaves association learning or decoding open. The diagnostic pilot below reports them.
- **Curves are flat from the first point.** Recall at 250 updates is already near its end-of-block value in
  most arms, so acquisition speed is unresolved at 250-update resolution, and the runner started recording after
  training began. The runner now records update 0 of each block and adds points at 25, 50, 100, 150 and 200 updates.

## Choice, made on floor, ceiling and the controls only (not on which arm does best)

- **N = 45.** N = 18 fails the rescoring check by noise; N = 90 leaves several arms near chance; N = 45 has
  every arm above 0.24 at L = 2,000 and a rescoring score at chance.
- **L = 2,000,** matching the curriculum's block length; L = 1,000 is cheaper (about 70 against about 95 minutes
  for the study) but the pilot shows no difference in floor or ceiling, so the choice rests on matching the block length.
- **Distance threshold stays at the provisional 0.35;** the pilot did not vary it, and nothing in the controls
  points to look-alike symbols as the problem.
- One seed fixes these numbers, so they carry the pilot's noise; the study cohort is separate (seeds 320–329).

## Computation

| N, L | wall time for six arms | peak child memory |
|---|---|---|
| 18, 1,000 | 389 s | 2.5 GB |
| 45, 1,000 | 392 s | 2.5 GB |
| 90, 1,000 | 394 s | 2.5 GB |
| 18, 2,000 | 546 s | 2.5 GB |
| 45, 2,000 | 546 s | 2.5 GB |
| 90, 2,000 | 551 s | 2.5 GB |

Data preparation 28–29 s and rendering 3 s per configuration. The grid took 1.0 hour of wall time, six
configurations, 36 arm-runs. The runner writes no model checkpoints; the per-arm record is about 70 KB (compressed
results file 72 KB for six arms). N barely changes cost; block length does. At L = 2,000 the study (60 arm-seeds in ten
batches of six) is expected to take about 95 minutes.

## Diagnostic pilots (seed 299, N = 45, L = 2,000; update 0, recognition diagnostics, Option B preflight) — superseded by the fixed-font rerun below

Run after the external review, at revision `4806eba`, with `NFI_TAG=-diag`: Option A (`...-A-N45-L2000-diag-*`)
and Option B (`...-B-N45-L2000-diag-*`), six arms each, about 10 minutes each. One seed; no reading.

Recall on A at the end of block 4 (chance 0.111):

| Arm | A: independent renderings | A: training renderings | A: lookup count / identity | B: independent | B: training | B: lookup count / identity |
|---|---|---|---|---|---|---|
| sparse48 | 0.24 | 0.57 | 0.31 / 0.10 | 0.14 | 0.31 | 0.26 / 0.13 |
| dense | 0.30 | 0.63 | 0.42 / 0.28 | 0.30 | 0.63 | 0.42 / 0.28 |
| mixed | 0.27 | 0.65 | 0.30 / 0.19 | 0.27 | 0.60 | 0.31 / 0.19 |
| mixed-h125 | 0.35 | 0.77 | 0.39 / 0.29 | 0.18 | 0.55 | 0.36 / 0.28 |
| mixed-h500 | 0.27 | 0.64 | 0.48 / 0.35 | 0.26 | 0.61 | 0.41 / 0.29 |
| mixed-reset | 0.32 | 0.68 | 0.44 / 0.32 | 0.20 | 0.44 | 0.38 / 0.29 |

- **`dense` is identical in the two options** (0.30 / 0.63 in both), as it must be: it has no rewiring or reset, so
  the schedule switch changes nothing for it.
- **The associations are learned.** Recall on the exact training renderings is 0.44–0.77 against 0.14–0.35 on independent
  renderings (chance 0.111). Both are above chance, so association learning is not the failure; the gap shows the
  network separates the trained images far better than it generalizes a symbol across font, rotation and size.
  Symbol-identity accuracy by nearest neighbor is 0.10–0.35 (chance 1/45 = 0.022), so recognition across renderings
  is weak. This supports recognition across renderings as a substantial limit on the independent-rendering recall.
  It does not show recognition is the only limit.
- **Shuffled-label rescoring** is at chance in every arm (mean 0.11, sd 0.02–0.03 over 100 permutations).
- **Option A, update 0 and early window.** Recall starts near chance (0.10–0.18 at update 0) and rises through the first
  250 updates in most arms (for example `dense` 0.13 to 0.26), so the early window now resolves acquisition.
- **Option B preflight: measurable, with a floor in the rewiring arms.** `sparse48` is at 0.14 on independent
  renderings and its curve is flat at 0.14 from update 0; its training-rendering recall is 0.31. `mixed-reset` (0.20 / 0.44)
  and `mixed-h125` (0.18 / 0.55) also fall relative to Option A, while `mixed` and `mixed-h500`, which do not rewire the memory
  part's connections through the reset, stay near their Option A values. Under Option B, then, continuing RigL
  or resets while pairs are taught erodes pair memory in the arms that carry that disruption, as the contract's caveat
  expected. The contrast of `mixed` with `sparse48` could be floor-bound on the `sparse48` side on independent renderings.
- **A to B comparisons are one seed;** the size of the erosion is not established.

**Proposed adjustment before freezing (for the researchers):** make recall on the training renderings a co-primary
outcome beside independent-rendering recall. It measures memory of the arbitrary association with recognition
generalization removed, and the study's question is memory; independent-rendering recall stays as the recognition-
generalization measure. Fixed before the cohort, it adds no analysis choice after seeing results.

## Fixed-font rerun (seed 299, N = 45, L = 2,000; `NFI_TAG=-diag2`, revision `9b93805`)

A check of the symbol renderings found that one symbol drawn in different fonts differs about as much as different
symbols do: the largest cross-font distance (noise-free 28×28 images, unit norm) has median 0.62, against 0.53 to the
nearest other symbol, for 242 of 387 symbols, and 72 of 1,841 font renderings (3.9%) are closer to another symbol's
mean than to their own. Each symbol is now drawn in one font fixed by a seeded stream; rotation, size, position,
brightness and noise still vary. The symbol pool becomes 512 (covered by one or more fonts, after the 0.35 distance filter).
Both options were rerun; the earlier diagnostic numbers above used the old rendering.

Recall on A at the end of block 4 (chance 0.111), Option A, then Option B where it differs:

| Arm | A indep. | A training | lookup count / identity | Option B: indep. / training |
|---|---|---|---|---|
| sparse48 | 0.31 | 0.59 | 0.35 / 0.19 | 0.25 / 0.30 |
| dense | 0.28 | 0.62 | 0.44 / 0.30 | 0.28 / 0.62 (identical, as it must be) |
| mixed | 0.32 | 0.71 | 0.42 / 0.26 | 0.30 / 0.64 |
| mixed-h125 | 0.31 | 0.67 | 0.45 / 0.31 | 0.22 / 0.44 |
| mixed-h500 | 0.34 | 0.60 | 0.43 / 0.30 | 0.28 / 0.59 |
| mixed-reset | 0.33 | 0.74 | 0.52 / 0.41 | 0.25 / 0.43 |

- **Fixing the font did not close the gap.** Independent-rendering recall stays at 0.28–0.34 against 0.59–0.74 on the
  training renderings; identity accuracy by nearest neighbor is 0.19–0.41 (chance 0.022), a little higher than before.
  So the cross-font spread was real but not the main reason for the gap.
- **Consistent reading, not established:** each symbol has only 8 training renderings, each seen thousands of times;
  the network may be storing those bitmaps rather than the symbols. A larger rendering pool, or fresh renderings every
  update, would test this. It is a design choice for the researchers.
- **Shuffled-label rescoring** is at chance in every arm (0.11, sd 0.02–0.03).
- **Option B** remains measurable: independent-rendering recall 0.22–0.30, none at the floor; the rewiring and reset arms
  (`mixed-h125` 0.44, `mixed-reset` 0.43, `sparse48` 0.30) lose training-rendering recall relative to Option A.
- One seed; no reading.
