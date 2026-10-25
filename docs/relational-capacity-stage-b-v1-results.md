# Relational capacity Stage B v1: no witness; routing stays almost inert in every arm

**Completion:** complete, 2026-10-25.

- The recorded tests phase passed (20 tests).
- The full declared menu ran: 40 candidates, 40,000 updates.
- Every selected serialized checkpoint passed two byte-identical independent
  NumPy/Schur verifications.
- No candidate met the witness criterion. All six (configuration, rendering)
  groups are `unresolved_after_budget`.

**Decision:** do not launch a learning or transfer study with S2, P0 or D1 as
specified. The route-sensitivity screen gives the main diagnostic:

- At every one of the 200 scheduled checkpoints, routed outputs differed from
  the same parameters' fixed-route outputs by an RMS of at most 0.0079.
- Any witness needs at least 0.2546.

Every parameter setting the optimizer visited therefore behaved almost as a
linear predictor, and was excluded by the affine floor. This is an observation
about the visited trajectories, not a uniform bound, so no arm is proved
impossible. Within this budget, the bottleneck appears to be the absence of
usable input-dependent nonlinearity, not retained rank or module count.

[Contract](relational-capacity-stage-b-v1-contract.md) ·
[decisions](relational-capacity-stage-b-decisions.md) ·
[manifest](../reports/relational-capacity-stage-b-v1-manifest.json) ·
[results](../reports/relational-capacity-stage-b-v1-results.json) ·
[runtime](../reports/relational-capacity-stage-b-v1-runtime.json)

## 1. What was tested

The task, renderings and witness criterion are those of masked-capacity v1:

- 72 scenes;
- loss ≤0.10 on every scene;
- every hidden orientation correct.

All three configurations keep reciprocal energy inference, explicit
observation masks, positive local gates computed from one collective pilot,
one routed solve, and actual outer states as output.

| Configuration | Description |
| --- | --- |
| S2 | One rank-two module, α=0.1 |
| P0 | Two independent rank-one modules at α/2 each, with a fixed mean readout |
| D1(μ) | Input module (α=0.1) coupled to a designated output module through sector-aligned identity ports on all eight terminal coordinates, μ ∈ {0.001, 0.01, 0.1} |

The optimizer minimized the exact worst-scene loss using all 72 targets. This
is privileged capacity fitting, not transfer. The fitted parameters must not
initialize later studies.

## 2. Results

Selected checkpoint per group, verified after serialization:

| Group | Selected candidate | Worst loss | Mean loss | Joint orientation | Fixed-route worst loss | Route sensitivity d |
| --- | --- | --- | --- | --- | --- | --- |
| S2, identity | start 0, rate 0.01, update 1000 | 0.921330 | 0.874786 | 0/72 | 0.921335 | 1.5e-5 |
| S2, rotated | start 0, rate 0.01, update 1000 | 0.922444 | 0.878078 | 0/72 | 0.922442 | 1.3e-5 |
| P0, identity | start 1, rate 0.01, update 1000 | 0.921869 | 0.876270 | 0/72 | 0.921865 | 1.8e-4 |
| P0, rotated | start 0, rate 0.01, update 1000 | 0.917340 | 0.868031 | 0/72 | 0.917165 | 5.4e-4 |
| D1, identity | μ=0.001, start 1, rate 0.01, update 500 | 0.986319 | 0.958399 | 0/72 | 0.986318 | 4.5e-4 |
| D1, rotated | μ=0.01, start 1, rate 0.01, update 1000 | 0.986441 | 0.959111 | 0/72 | 0.986381 | 1.8e-3 |

Across all 40 candidates:

- **S2 and P0** ended with worst losses of 0.917–0.938.
- **D1** ended at 0.986–0.997, barely below the zero predictor's 1. The output
  module receives its evidence only through the ports and stayed close to
  zero.
- **μ.** Larger μ did not help; μ=0.1 gave the worst D1 results (0.994–0.997).
- **Selected checkpoints.** 36 candidates selected their final checkpoint and
  4 selected update 500. Worst-scene loss improved only slightly.
- **Rank-two arm.** S2 reached about the same level as P0: removing the
  terminal bottleneck did not by itself produce progress toward the target.

**The route-sensitivity screen excluded every checkpoint.** It applies to the
24 equal-amplitude scenes. The largest d at any checkpoint was 0.0079, against
τ = 0.2546. The parameter-specific thresholds were larger still, 0.82–0.92.

At every selected checkpoint, the fixed-route control (all routes one, i.e.
the pilot) matched the routed worst loss to within 0.004. That gap is largest
for D1 at μ=0.1, and is at most 2e-4 for S2 and P0. The gates as parameterized
change the output by a negligible amount. A larger d is necessary for a
witness, but would not be sufficient.

The disconnected-output control for D1 is analytical: with no port, the output
module's equilibrium is zero and every scene loss is exactly 1.

**Gate denominator diagnostic** (recorded as declared, not an error):

| Configuration | Minimum gate denominator |
| --- | --- |
| S2 | ≥2.9e-8 |
| P0 | down to 2.4e-12 |
| D1 | down to 2.8e-14 |

For D1, the minimum is below the gate formula's ε = 1e-12. Many output-module
gates are then dominated by ε and pushed toward fully open. Across candidates,
9,032 forward batches had some denominator below 1e-8. Weak drive in unanchored
or weakly anchored modules therefore also leaves their gates largely
uninformative. This is one plausible contributor to the inert routing. It is
not the whole explanation, since S2 shows the same inertness with healthy
denominators.

## 3. Interpretation

- **What is shown.** Under the declared optimizer and budget, none of the three
  revisions found a witness. Every visited setting was effectively linear
  with respect to the task.
- **Composition behaved as the analysis anticipated.** Passive or weakly
  coupled modules did not supply usable capacity. The designated output
  inherited only a weak signal.
- **What is not shown.**
  - That S2, P0 or D1 is impossible. The screen bounds visited settings only.
  - That routing of this form can never be sensitive. A uniform upper bound on
    d over all allowed parameters would turn the screen into an exclusion; no
    such bound has been derived.
- **What would turn this into an exclusion.**
  - An analytical bound on route-induced output change for positive gates in
    [0.325, 1] computed from pilot residual ratios.
  - A deliberately different source of input-dependent nonlinearity: for
    example, the relation-selection mechanism still missing for R/G, or gates
    that are not bounded away from zero.

  Either needs a new prospective contract. Extending μ, the step sizes or the
  number of updates here would not address the observed failure mode, and the
  contract forbids it.
- **No communication-benefit claim** follows in either direction. P0 and D1
  differ in interface and evidence placement.

## 4. Correctness and reproducibility

Order of work:

1. Contract committed before implementation: `0a61cad`. Its only amendment
   corrected the status line from draft to frozen, before any implementation
   existed.
2. Sources and tests committed: `98ecc74`.
3. Manifest committed: `8d24f78`. It records 33 frozen dependency hashes,
   including the masked-capacity v1 chain, its task audit, the decisions
   document and the bounds the contract relies on.
4. Recorded tests phase: all 8 frozen masked-capacity tests and 12 new tests
   passed.
5. Search.

**Disclosures.**

- During development, before the manifest, the unit tests were run and an
  in-memory smoke test ran three updates per configuration. Nothing was
  written to reports, and the smoke-test progress lines were filtered out
  unread.
- The frozen masked-capacity test module sets Torch's inter-op thread count at
  import. The new test module therefore tolerates an earlier setting, and the
  runner imports the frozen tests first.
- No source changed after the freeze.

**Verification.**

- Torch/NumPy disagreement at most 6.1e-14.
- Full/Schur disagreement at most 1.25e-13.
- All verification records byte-identical between their two repeats.
- No nonfinite value, constraint violation, rank failure, solver failure or
  correctness stop.
- The fixed-route affine-floor check held at every checkpoint.
- The full optimizer trajectory ran once, as declared.

## 5. Computation

Ryzen 5 3600, float64, CPU PyTorch 2.14.1, one process, one BLAS/Torch thread.

| Phase | Wall seconds | CPU seconds | Peak RSS |
| --- | --- | --- | --- |
| Manifest freeze command | see runtime JSON | | |
| Tests subprocess | 2.96 | see runtime JSON | |
| Search command | 5,629.6 | 5,628.6 | 630,480 KiB (parent, cumulative high-water) |

Candidate wall time totalled 292.6 s for S2, 551.2 s for P0 and 4,764.6 s for
D1. These components are contained in the search total and must not be added
to it. Per-candidate and per-checkpoint cumulative costs are in the runtime
JSON.

Work counts for the search, including verification forwards:

| Count | Value |
| --- | --- |
| Optimizer updates | 40,000 |
| Backward passes | 40,000 |
| Fitting-scene exposures (each independent example reused) | 2,880,000 |
| Local observation uses (P0 counts each scene twice) | 3,456,000 |
| Torch forward batches | 40,280 |
| D=80 Torch factorizations, pilot / routed | 24,168 / 1,740,096 |
| D=160 Torch factorizations, pilot / routed | 24,168 / 1,740,096 |
| QR retractions | 864,000 |
| Unit retractions | 256,000 |
| Independent NumPy Cholesky checks, D=80 / D=160 | 6,912 / 6,912 |

Only 72 independent examples were used. The NumPy checks exclude their Schur
sub-solves.

Storage is listed in the contract (§9) and the runtime JSON. FLOPs, native
autograd workspace and backward primitive-solve counts are null with reasons.

This is a privileged capacity search, not an inference benchmark or scaling
claim. The measured per-update costs (about 0.04 s for S2, 0.07 s for P0 and
0.19 s for D1) were within the contract's planning estimate.

The S1 reference (masked v1) used a mean-loss objective and different seeds.
Its best verified worst losses, 1.006 and 1.007, are context, not a matched
comparison. Its missing historical cost fields remain as disclosed in the
contract.

**Reproduce** from an isolated checkout at `8d24f78` with the pinned
environment:

```
python -m experiments.relational_capacity --phase tests
python -m experiments.relational_capacity --phase search
```

The runners refuse changed frozen inputs and phases that are already
recorded. Preserve the archived artifacts.
