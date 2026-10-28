# Number concept network v1: pilot results

Date: 2026-10-28. Specification: [number concept network v1](number-concept-network-v1-spec.md)
(draft; not yet frozen). Pilot attempt 1 of 1; no earlier or discarded attempts.

Command: `.venv/bin/python experiments/number_concept_network.py pilot`.
Artifacts: `reports/number-concept-network-v1-pilot-results.json` (scientific),
`-runtime.json` (timing/cost), `-manifest.json.gz` (splits, dice metadata, anchors),
`-grids/` (qualitative). Checkpoints are in gitignored `runs/` with SHA-256 values
in the results file. The runtime records `dirty: true` at revision 9b4ba0b because
the runner, module and spec were uncommitted at run time. Their recorded SHA-256
values match the files committed unchanged in a719ffd.

Scope, as the specification states: seed 100 (SW donor seed 101), 500 digits per
class, 200 dice per count, 5 epochs/stage (100 stage-1 and 25 stage-2 steps),
validation pools only, random-layout results not computed. This is a pipeline and
stability check. It does not test the headline thresholds and cannot select a budget.

## Pipeline and stability checks

| Check | Outcome |
| --- | --- |
| Implementation acceptance checklist (23 items) | all pass |
| Split isolation (MNIST train/validation disjoint, 600/digit, no test data, distinct dice streams, anchor in training pool) | all pass |
| Finite values | no numerical failure in 4 stage-1 and 10 dice fits |
| Losses falling | yes in every fit (reconstruction BCE 0.69 → 0.21–0.44; successor loss 7.6–8.0 → 4.2–5.6) |
| Clip frequency (norm 5) | stage 1: 2%; dice U/A/SH/SW: 8–12%; dice AO: 52% |
| Probe convergence | full-latent probe loss 2.30 → 0.31 at step 500, still decreasing slowly (≈0.007 over the last 100 steps) |
| Dice placement | standard faces: radius means 1.78–1.82 across counts; six-face acceptance ≈ 0.57 |

No pipeline or numerical-stability amendment is indicated. AO clips on about half
its steps, since it optimizes only reconstruction plus the single-image anchor.
Clipping kept it stable, but the clip rate should be read next to AO's results.

## Diagnostic values (not headline evidence)

| Variant | Digit centroid | Digit probe `[Re, Im]` | Probe `|Z|` | Digit composition k=1 |
| --- | ---: | ---: | ---: | ---: |
| Rotation | 0.872 | 0.883 | 0.692 | 0.875 |
| MLP | 0.873 | 0.884 | 0.727 | 0.851 |

Both digit gates are below 95%, which the specification expects after 100 steps.
Digit class-centroid recurrence residuals are 0.5–1.2, far from exact recurrence.

Dice macro accuracy on counts 2–6 (ten-way readout; count-1 accuracy in parentheses):

| Variant | U | A | SH | AO | SW |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rotation | 0.209 (1.00) | 0.002 (1.00) | 0.000 (1.00) | 0.000 (1.00) | 0.005 (1.00) |
| MLP | 0.000 (0.00) | 0.000 (0.95) | 0.000 (1.00) | 0.000 (1.00) | 0.046 (1.00) |

The anchored arms score below the 10% uniform rate because their predictions
concentrate on digit 1. After 25 stage-2 steps the dice classes are barely separated.
For rotation A, cosines between dice class centroids are mostly 0.6–0.94, and
within-dice recurrence residuals are 0.84–1.27. The digit-1 centroid is the
nearest digit centroid for most dice codes. The anchor pulls the count-1 code onto
it (relative anchor residual 0.17), and the other counts follow. AO shows the same
pattern without succession. Rotation U scores 1.00 on count 1 and 0.87 on count 5
with no anchor; this is a single short run and is not interpreted.

## Cost

Wall time 57.9 s; worker CPU 221.5 s total (11 workers, one thread each). Phases:
data preparation 14.9 s (all pools, including random layouts), stage 1 22.3 s,
stage 2 15.5 s, deployment timing 4.1 s. 650 optimizer steps, 473,800 training
encodings, 8 probe fits. Largest worker peak RSS 623 MB. Checkpoints ≈1.64 MB each.

## Next

The pilot shows a working pipeline at this budget, not feasibility. The 95% digit
gates and stage-2 class separation are untested at the confirmation budget (4,220
and 940 steps). The specification fixes that budget unless a separately recorded
budget-selection study changes it. Freeze the contract (with implementation hash)
before any confirmation run.
