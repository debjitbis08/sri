# Number concept network v1: confirmation results

Date: 2026-10-28. Contract: [number concept network v1](number-concept-network-v1-contract.md)
(frozen at 344f89e). Command: `.venv/bin/python experiments/number_concept_network.py
confirmation`. Runtime hashes of the runner, module and contract match the frozen
files. `dirty: true` reflects unrelated uncommitted notes in docs/ only.

Artifacts: `reports/number-concept-network-v1-confirmation-results.json.gz`
(scientific), `-runtime.json`, `-manifest.json.gz`, `-grids/`. Checkpoints are in
gitignored `runs/` with SHA-256 values in the results file. All scores below are
held-out test sets unless marked; validation A scores agree within 0.02.

## Outcome by the frozen rules

| Claim | Rotation | MLP |
| --- | --- | --- |
| Digit gates (centroid and `[Re, Im]` probe ≥95%, every seed) | **pass** (0.972–0.974 / 0.970–0.974) | **pass** (0.973–0.975 / 0.973–0.974) |
| Primary one-anchor transfer (A mean ≥0.90, no seed <0.80, ≥0.20 over SH, AO, SW) | **fail** | **fail** |
| Composition (A, k=2 mean ≥0.90 in both modalities) | **fail** (digits 0.973, dice 0.471) | **fail** (digits 0.977, dice 0.344) |

No positive one-anchor-transfer result is established for either variant.

## Primary transfer

Macro accuracy on standard dice faces 2–6, ten-way readout (uniform guessing 0.10):

| Variant | Arm | Seeds 0–4 | Mean |
| --- | --- | --- | ---: |
| Rotation | U | 0.935, 0.000, 0.000, 0.758, 0.000 | 0.338 |
| | **A** | **0.503, 0.059, 0.015, 0.976, 0.403** | **0.391** |
| | SH | 0.256, 0.000, 0.158, 0.195, 0.200 | 0.162 |
| | AO | 0.136, 0.031, 0.102, 0.051, 0.119 | 0.088 |
| | SW | 0.001, 0.223, 0.210, 0.001, 0.200 | 0.127 |
| MLP | U | 0.004, 0.628, 0.102, 0.000, 0.580 | 0.263 |
| | **A** | **0.000, 0.876, 0.659, 0.286, 0.201** | **0.404** |
| | SH | 0.090, 0.168, 0.200, 0.200, 0.201 | 0.172 |
| | AO | 0.082, 0.085, 0.175, 0.132, 0.177 | 0.130 |
| | SW | 0.000, 0.398, 0.126, 0.040, 0.000 | 0.113 |

Paired A − control differences (mean, sample SD): rotation SH 0.229 (0.344),
AO 0.303 (0.393), SW 0.264 (0.490); MLP SH 0.233 (0.338), AO 0.274 (0.359),
SW 0.292 (0.217). The mean gaps exceed 0.20, but some individual differences are
negative: rotation seed 2 against SH and AO, rotation seeds 1–2 against SW, and
MLP seed 0 against SH and AO.
The controls cannot rescue the claim, because A itself misses both the mean and
per-seed accuracy thresholds. The seeds split into one near-complete success
(rotation seed 3, 0.976), partial runs and near-total failures. This is not a
uniform shortfall.

## Diagnosis: recurrence without reliable cross-modal alignment

The contract asks, when digit feasibility passes but A fails, whether within-dice
recurrence or cross-modal alignment failed. The diagnostics point to alignment.

1. **Dice succession is learned within the dice modality.** The final successor
   loss is 3.94 for U and A in every seed and variant. That is the
   positive-multiplicity entropy floor, `log(256/5) = 3.94`. The unanchored arm
   shows the same structure on the test set as near-pure offsets:
   rotation seed 2 predicts count + 3 for 98% of faces 2–6, and MLP seed 3
   predicts count − 2 for 99%. These are the orbit ambiguities the contract's
   prior-work section predicts: a correct internal ordering with an arbitrary
   cross-modal start. Rotation U seeds 0 and 3 land at offset 0 (0.93, 0.76)
   by that same ambiguity, not by anchoring.
2. **The anchor fits one image, not the class.** The anchor residual is 0.01–0.03
   in every anchored run. Test count-1 accuracy is 1.00 in all rotation A runs,
   but the held-out count-1 centroid's relative raw residual is 0.99–1.61
   (rotation) and 2.03–3.20 (MLP). Two MLP A runs (seeds 0 and 4) score 0.00
   on test count-1 despite fitting the anchor.
3. **Anchoring does not propagate along the chain.** Arm A's modal predictions
   for counts 1–6 are, for example, `1,2,1,4,3,0` (rotation seed 0) and
   `1,0,1,7,8,9` (rotation seed 2). Count 1 is pinned, and some counts or
   segments line up, but the chain is not carried through. Class-centroid
   recurrence residuals `||S(c_n) − c_(n+1)|| / ||c_(n+1)||` are large in both
   modalities: digits 0.57–0.65 (rotation) and 0.54–1.11 (MLP); dice 0.44–0.58
   (rotation A) and 0.64–0.81 (MLP A). The contrastive objective ranks successors
   correctly (digit k=1 composition 0.976–0.979) without making them nearly
   equal. Anchor propagation needs near-equality relative to the centroid
   margins, so an anchor at count 1 does not determine counts 2–6. The contract
   lists this as a known limitation of contrastive training.
4. **The successor operator matters, but not enough.** SW, using another seed's
   frozen S, is below A on average. The relative operator discrepancy on
   recipient digit latents is 1.66–2.77 (rotation) and 1.33–2.19 (MLP). Both
   variants pass the digit gates and digit composition, and neither aligns dice
   reliably. Rotation is not better than the MLP here: their A means differ by
   0.013, which is within the seed spread.

## Secondary diagnostics

- **Composition, arm A:** dice k=1, 2, 3 macro accuracy tracks the primary score
  per seed. Rotation seed 3 reaches 1.00 / 1.00 / 0.61, MLP seed 1
  1.00 / 1.00 / 0.99, and other seeds stay low. Digit composition k=1–3 is
  0.80–0.98.
- **Random layouts:** counts 1–6 macro accuracy is 0.03–0.23 in every arm, so the
  dice encoder does not transfer across layouts. Counts 7–9 are therefore not
  interpretable as counting (0.00–0.33, from single-class prediction patterns).
- **Magnitude probe:** `|Z|` reaches 0.81–0.88 (rotation) and 0.89–0.92 (MLP),
  below the full-latent probe, as expected if phase carries class information.
- **Clipping:** stage 1 0.8–5%; dice U ≤1%; A 4–40%, SW 6–32%, AO 7–26%,
  SH 20–57% of steps.

## Data caveat raised after freezing

After the run started, the user noted the standard dice are visibly irregular.
The contract's independent ±2 px per-pip jitter, with 6.5 px nominal spacing,
skews fives and sixes noticeably. These results hold only for that generator.
The diagnostics suggest the jitter is not the main cause of the failure: test
faces are cleanly separated and ordered within the dice modality (U offset
fractions up to 0.99, successor loss at its floor). The failure lies in the
cross-modal step. Cleaner dice could still change the margins, and a successor
contract would be needed to test that.

## Cost

| Phase | Wall s | Notes |
| --- | ---: | --- |
| Data preparation | 15 | MNIST load, all dice pools including random layouts |
| Stage 1 (10 fits, parallel) | 2,952 | 2,852–2,917 CPU s per fit incl. evaluation and probes |
| Stage 2 (50 dice fits, 11 workers) | 2,898 | ≈726 CPU s per correct/shuffled arm; AO ≈342 |
| Grids, timing, writing | 6 | |
| **Total** | **5,872** | worker CPU 59,796 s (16.6 h); largest worker peak RSS 800 MB |

Work: 89,200 optimizer steps, 63.7 M training encodings, 20.4 M successor
applications, 40 probe fits, 9,394 clipped steps. Selection cost before
confirmation: the single pilot (221 worker CPU s); no tuning. Deployment, rotation,
seed 0 A, one thread: encoding plus readout 0.24 ms (batch 1) and 6.1 ms (batch
256) median; cross-decoding 0.31 ms and 11.7 ms. Running 10–11 fits in parallel
was about 3.5× slower per step than one process alone; the pre-run 30-minute
estimate used single-process timing.
