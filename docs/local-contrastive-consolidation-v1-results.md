# Local contrastive consolidation v1: local learning succeeds; two-phase learning adds no advantage

Completed 2026-10-26 at revision fe06f43, run once as contracted, with no
extensions. [Contract](local-contrastive-consolidation-v1-contract.md) ·
[runner](../experiments/local_contrastive_consolidation.py) ·
[scientific results](../reports/local-contrastive-consolidation-v1-results.json) ·
[per-cycle movement, gzip](../reports/local-contrastive-consolidation-v1-movement.json.gz) ·
[runtime](../reports/local-contrastive-consolidation-v1-runtime.json)

## Outcome

Unlike [local consolidation v1](local-consolidation-v1-results.md), every learning
arm acquires the second structure, and every arm that replays both structures
reconstructs both. The main two-phase arm cuts combined whole-module RMSE from
0.366 (frozen) to 0.099. Its final error is nearly identical to the
exact-gradient reference's and within the declared match margin of ridge
regression on the same waking data. The three-module 8→4→2
architecture can learn this task with module-local update signals.

The contrast this study was designed around fails. The clamped-energy control
(the v1-style rule, with no second phase) reaches 0.092 and beats the two-phase
rule in all ten seeds, at equal update count and at equal settle count. It is
also cheaper. Both rules learn successfully; the two-phase rule adds no
demonstrated advantage here. Clamped-energy learning with learnable
inter-module translations is therefore the working baseline for follow-up
studies.

Importance inertia also fails its test. It reduces forgetting relative to no
protection, but uniform slowing with the same mean scale reduces it more, and
arm 9's movement differs from arm 8's by more than 20% in every seed, so that
comparison is unmatched.

Three conclusions are kept separate:

- **Learning:** demonstrated for this linear fixture with block-local update
  signals. Clamped energy 0.092, two-phase 0.099.
- **Inertia:** reducing plasticity protects old knowledge, but this importance
  measure is not shown to protect beyond ordinary slowing. With every
  comparison movement-unmatched, selective protection remains unresolved.
- **Hierarchy:** neither its necessity nor the sufficiency of the outer
  shortcut is established under the declared criteria.

| Criterion | Result | Pass |
| --- | --- | --- |
| P1 acquisition and retention with A+B replay | B improves 10/10; A retained 10/10; 0.099 vs frozen 0.366 | yes |
| P2 two-phase beats clamped energy | 0/10 seeds against 2a and against 2b | **no** |
| P3 gap to exact-gradient reference | ratio 0.9996 (32-step), 0.9996 (exact) | yes |
| S1 topology | arm 5/3 = 1.06, arm 5b/3 = 1.23, arm 4/3 = 1.29 | neither threshold met |
| S2 importance inertia beats B-only and uniform slowing | 0/10 seeds; all 10 movement-unmatched | **no** |
| S3 against ridge | ratio 1.011 | match |
| M1 gradient fidelity at initialization | 10/10 seeds | yes |

## Main comparison

Mean whole-module masked RMSE (32-step inference) on 12 unseen variations per
structure, over confirmatory seeds 3-12:

| Arm | After A: A | After A: B | Final A | Final B | Final combined | Exact-equilibrium combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 Frozen | | | | | 0.3662 | |
| 2a Clamped energy, C+C | 0.0958 | 0.3802 | 0.0934 | 0.0903 | **0.0919** | 0.0912 |
| 2b Clamped energy, 9C+9C | 0.0960 | 0.3815 | 0.0936 | 0.0904 | 0.0921 | 0.0914 |
| 3 Two-phase, uniform | 0.0983 | 0.3731 | 0.1004 | 0.0974 | 0.0989 | 0.0982 |
| 4 Two-phase, inner only | 0.1309 | 0.3671 | 0.1282 | 0.1270 | 0.1276 | 0.1164 |
| 5 Two-phase, no outer | 0.1046 | 0.3676 | 0.1057 | 0.1040 | 0.1048 | 0.1026 |
| 5b Two-phase, outer-only chain | 0.1235 | 0.4984 | 0.1232 | 0.1193 | 0.1213 | 0.1213 |
| 6 Exact-gradient reference | 0.0983 | 0.3732 | 0.1005 | 0.0974 | 0.0990 | 0.0982 |
| 10 Ridge (all waking traces) | | | | | 0.0979 | |

Arm 3 is worse than arm 2a in every seed, by 0.0059 to 0.0083 combined RMSE.
Arm 2b barely differs from 2a: nine times more cycles changes the final error
by 0.0002. The clamped-energy rule has converged by C cycles, so 2b's extra
settles buy nothing.

Arms 3 and 6 reach nearly identical final errors (ratio 0.9996). That
establishes a small final-performance gap between the block-local two-phase
signal and the exact gradient of the same masked loss under the same global
settling. It does not establish that their learning trajectories or later
gradients are equivalent; gradient fidelity degrades during training (see M1).
The arm 3 versus arm 2 difference is therefore not explained by locality or
finite settling. It lies between the two objectives: masked-loss gradient
versus clamped energy. This locality statement covers learning signals only;
settling still uses global solves.

No arm generalizes to the unseen structure before seeing it. After stage A,
structure-B error stays near frozen in every hierarchical arm. The outer-only
chain is worse there (0.498).

## What this says about the earlier diagnosis

The suggestion that motivated this study claimed that v1's proposals came from
the wrong objective: clamped reconstruction residuals make each module a local
autoencoder. This run does not support that as the main explanation. Once
inter-module translations are learnable at graded layers, energy minimization
at fully clamped states is the best arm in this study.

There, the translation update is essentially ΔT ∝ (s_b − T s_a) s_aᵀ: a local
regression of each neighbor layer on the other, using only activities at the
two connected layers. Such a local relationship can improve reconstruction
elsewhere through network interaction. Relative to v1, this study changed
several things together:
- learned translations at all three layers;
- coupling budget 2 instead of 0.4;
- 300 Adam cycles instead of 6 SGD proposals per phase;
- simultaneous rather than sequential module updates.

No arm here combines the clamped-energy rule with fixed identity coupling or
with v1's budget. So which of these changes rescued learning is not identified.

The two-phase rule is still a working local rule. Its fidelity at initialization
is high, and it tracks the exact gradient. It simply optimizes a different and,
here, slightly worse target.

## Exploratory observations (not criteria)

- **Effective strength.** After training, the largest outer-translation singular
  value averages 2.72 for arm 3 and 2.71 for arm 6, but 1.04 for arm 2a. The
  masked-loss arms raise effective outer coupling (κTᵀT) well above its initial
  level. The energy arm keeps it near 1 and aligns the outer translation with
  the private pattern subspace better: relative error 0.067 versus 0.168. One
  hypothesis is that minimizing masked error on 32 replayed traces uses coupling
  strength in ways that generalize slightly worse than modelling the joint
  energy. Training-set errors were not recorded, so this run cannot test that.
- **Functional alignment** (relative residual, lower is closer). The masked-loss
  arms align middle layers more closely than arm 2a (0.063 vs 0.156). Arm 2a
  aligns inner layers (0.010 vs 0.037) and outer layers (0.270 vs 0.470) more
  closely. Different arms distribute cross-module transmission across layers
  differently.
- **Abstraction metrics** are uninformative, as declared: frozen centroid
  accuracy is already 0.992, and every learned arm scores 1.0.

## Topology (S1)

Removing outer coupling costs 6% (arm 5/3 = 1.06), below the 1.25 threshold, so
this study does not establish that outer connections help. The outer-only chain
is 23% worse (1.23): not within 1.1, so the shortcut does not suffice here, and
not quite 1.25, so the declared underperformance conclusion is not reached
either. The hierarchy plus graded coupling gives the best two-phase result, but
the declared thresholds support no topology conclusion. Inner-only coupling is
clearly worst among hierarchical arms (1.29). That is consistent with the probe
and the frozen graded study, but it confounds topology, budget distribution and
parameter count, as declared.

## Forgetting and inertia (S2)

With B-only replay in stage B, structure A degrades sharply. Mean A RMSE ratios
(after B / after A) by seed:

| Arm | Mean | Range |
| --- | ---: | --- |
| 7 B-only, no protection | 2.39 | 1.89 to 3.06 |
| 8 B-only, importance inertia (λ=100) | 1.76 | 1.46 to 2.25 |
| 9 B-only, matched uniform slowing | 1.48 | 1.24 to 1.89 |

Final B RMSE is 0.0948, 0.0989 and 0.0960 respectively; arm 8's B is 1.03× arm
9's. Importance inertia protects A better than nothing in every seed, but worse
than uniform slowing in every seed. The comparison is not movement-matched.
Arms 8 and 9 have the same mean scale (0.364) at the start of stage B, yet arm
9's stage-B displacement is 0.61 to 0.85 of arm 8's and its path length 0.62 to
0.75 of arm 8's. Importance scaling slows parameters that moved in stage A but
lets the others move at full speed, so arm 8 moves more in total. All ten seeds
are flagged; the sensitivity analysis on unflagged seeds has 0/10 seeds left.
λ=100 was selected at the upper grid edge. The grid was not extended, so a
larger λ is untested.

So this run does not show that the importance measure protects useful
relationships beyond uniform slowing. Ω measures accumulated estimated learning
contribution, as stated in the contract, not usefulness across experiences.
Forgetting of the first structure without its replay is now established in this
architecture. This does not explain v1's retention result: v1 also retained and
replayed A, and its retention did not separate replay from the small amount of
learning.

## Gradient fidelity (M1)

At initialization, every arm-3 parameter group in every seed passes: minimum
cosine 0.998, maximum relative error 0.104, norm ratio 0.912 to 0.999. After
training, fidelity degrades. For arm 3 after stage B: minimum cosine 0.70,
maximum relative error 0.73, norm ratio 0.87 to 1.24. Some B-only arms reach
cosine -0.16 for one group at the end of stage B, where the exact gradient is
small. Fidelity at later checkpoints was declared reported, not a criterion.
Despite this degradation, arm 3's final error is within 0.05% of arm 6's. So
finite-settling bias produced no measurable final-performance gap, though the
trajectories may still differ. The outer-only arm stays above
cosine 0.959 throughout. No positive-definiteness check failed (0 skips), no
arm diverged, and no near-zero gradient groups occurred.

Verification passed before the pilot:
- finite-difference errors: activity 5e-9, parameter 8e-9, exact-loss adjoint
  1.3e-10;
- v1 energy equivalence;
- implicit descent in free and both nudged phases;
- hidden-target leakage, chain reversal at identity T, and private-map
  reconstruction;
- arms 3/7/8/9 share an identical stage A.

With long settling and β=1e-5, the two-phase estimate matches the exact gradient
to relative error 3e-10 (uniform) and 5e-9 (outer-only). At the contracted
β=0.01 with long settling, the deviation is 3e-4 and 5e-3, which is finite-β
bias, not a defect.

## Pilot selection

Pilot seeds 0-2, trained on even-index waking traces and scored on odd-index
traces after stage B. Selected values:
- two-phase η=0.01;
- clamped energy η=0.03;
- exact reference η=0.01;
- inertia λ=100 (grid edge).

No η was disqualified, and the pilot test sets were never scored. Full grids
are in the results JSON.

## Implementation interpretations

These choices were made in the runner where the contract left room. They are
recorded here and were not tuned.
- Ω accumulates the two-phase estimate without weight decay.
- A mask failing the PD check would contribute zero, with the average still
  divided by 3. None failed.
- The M1 pass/fail applies to arm 3. Values for the other two-phase arms are
  reported, not criteria.
- S2's B condition uses mean final B RMSE over the seeds considered.
- The pilot score is the mean squared error per coordinate.
- Translations act on the full layer, so the fidelity group `T_outer` combines
  both edges.
- A 2-cycle smoke run on seed 3 checked execution and determinism before the
  real run. Its outputs were deleted unexamined.
- The verification check originally asserted long-settle fidelity at β=0.01
  and failed on finite-β bias. It was changed to assert at β=1e-5 and report
  β=0.01 before the committed runner and the real run.
- Per-cycle movement is written to a separate deterministic file rather than
  the main results JSON. The runner writes plain JSON (13 MB). After the run it
  was compressed with `gzip -n -9` to `reports/local-contrastive-consolidation-v1-movement.json.gz`
  (5.5 MB), without changing the committed runner. The uncompressed SHA-256 is
  `7a627eab4472d33236aa1ce3970da1dc5f1289455c0a34eb28d4ff59a166f64a`, so a rerun
  can be compared byte-for-byte after decompression. A more sustainable storage
  mechanism for large artifacts is still to be chosen.

## Computation

Whole run: 252.7 s wall, 252.6 s process CPU on one thread, peak RSS 120 MB.
This includes pilot selection (60.4 s), verification (1.0 s), all evaluations
and diagnostics. Mean training-only CPU per seed and arm, excluding evaluation:

| Arm | Training CPU (s) | Activity example-steps | Free-dim² weighted estimate |
| --- | ---: | ---: | ---: |
| 2a Clamped energy | 0.49 | 460,800 | 1.49e8 |
| 2b Clamped energy, 9× | 4.35 | 4,147,200 | 1.34e9 |
| 3 Two-phase, uniform | 2.07 | 4,147,200 | 2.80e9 |
| 4 Two-phase, inner | 1.74 | 4,147,200 | 2.80e9 |
| 5 Two-phase, no outer | 1.89 | 4,147,200 | 2.80e9 |
| 5b Outer-only chain | 0.99 | 4,147,200 | 2.65e8 |
| 6 Exact reference | 1.13 | — (exact solves) | — |
| 7-9 B-only variants | 1.94-1.96 | 2,764,800 | 1.87e9 |

The best arm (2a) is also the cheapest learning arm. It needs about a quarter of
arm 3's training CPU and a ninth of its activity work. Arm 2b matches arm 3's
settle count at about twice its CPU because of per-cycle overhead, and gains
nothing. These are whole-phase mechanism timings on 42-dimensional states, not
latency benchmarks or scaling claims.

Ridge needs 3 closed-form fits per stage and has negligible cost. It sees all
three surfaces globally. Pilot selection (48 trajectories across 4 grids, all
retained) is charged separately from confirmatory training.

Storage per G-uniform learner: parameters 2,304 B, Adam moments 4,608 B plus a
step counter, replay 6,144 B and episode IDs 256 B. Inertia arms add 6,912 B.
Raw traces are retained. The exploratory probes that informed the design have
no cost receipts.

The runtime artifact's `dirty_status` lists the two result files. They are
written before the provenance step; the source tree itself was clean at
fe06f43.

## Next questions

These are not results:
- **Priority: which change rescued the simpler rule.** Do this before adding
  another learning mechanism. Use the clamped-energy rule, hold Adam, update
  count and replay fixed, and compare fixed versus learned translations and
  weaker versus stronger coupling. This tests whether learning the
  relationships between modules is the decisive ingredient.
- **The objective.** Why does energy minimization generalize better than the
  masked-prediction gradient here? Record training-set errors and effective
  coupling over training.
- **Inertia.** Use movement-matched controls, for example matching per-cycle
  update norms, and an importance measure tied to cross-experience support.
- **Harder fixtures.** Nonlinear or multistable ones, where discrimination is
  not already trivial and where a settling-based signal could matter.
