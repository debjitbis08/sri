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
- **Learning check (`P-A'` against A):** within a few points of each other in both arms at every size
  (largest gaps 0.07 at N = 45 and N = 90), no evidence that learning depends on which counts were assigned.
- **Recall is mostly recognition-limited.** The nearest-neighbor lookup on the item code (training against evaluation
  renderings) is 0.22–0.57, often above the network's recall, so the ceiling for recall here is set by recognizing a
  symbol across font, rotation and size.
- **Curves are flat from the first point.** Recall at 250 updates is already near its end-of-block value in
  most arms, so acquisition speed is unresolved at 250-update resolution. The runner now adds points at 25, 50, 100,
  150 and 200 updates of each block.

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
