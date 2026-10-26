# Grading and feedback gain in three 8→4→2 modules

**Layer grading matters, and output-feedback gain matters independently.** At a
fixed aggregate coupling budget, uniform per-unit connections across all three
layers give the lowest final error among the six tested profiles. For every
profile/strength combination, lower output feedback gives lower final error.
Stronger inward-graded connections also let the fully reciprocal chain pass the
original thresholds within 4096 updates. Thus the previous full-reciprocity
failure was a finite-budget result, not an inability to represent the solution.

This is a 33-configuration sensitivity study with supplied frozen connections,
not learned parameter selection or proof of a generally optimal grading. No
sleep learning, STM or parameter adaptation was used.

## What changed

Retain input ↔ intermediate ↔ output with 8→4→2 continuous activities per module,
the same local maps, 28 fixtures, initial state, 0.1 step size and 4096-update cap.
Only input/intermediate outer coordinates receive cues; only output outer
coordinates are scored. The task remains supplied rank-two linear completion.

New connections can act at the outer, middle and inner layers. Fixed orthogonal
translations align the modules' different coordinate systems before applying
compatibility pressure. They operate on the full layer, not only its two latent
coordinates. This avoids forcing numerically different representations to agree
without translation. These translations are supplied, not learned; performance
with unknown or erroneous alignment remains untested.

The main six profiles share the budget `8*w_outer + 4*w_middle + 2*w_inner = 2`
on each edge. This controls total reciprocal coupling curvature despite different
layer widths. It does not match every directional gain or the spectrum of the
whole dynamics. Uniform means equal strength per unit, not equal total strength
per layer. Separate controls vary overall strength. All three output-feedback
gains (0, 0.25, 1) retain intermediate→input feedback at strength 1. Six additional
controls remove both feedback directions.

## Main comparison at equal aggregate coupling strength

Output RMSE after 4096 updates, over the 16 complete-evidence pairs:

| Relative outer:middle:inner profile | Output feedback 0 | Output feedback 0.25 | Output feedback 1 |
| --- | ---: | ---: | ---: |
| Inner only, 0:0:1 | 4.17e-6 | 1.81e-5 | 2.81e-4 |
| Inward graded, 0.05:0.2:1 | 1.21e-6 | 6.08e-6 | 1.30e-4 |
| Uniform per unit, 1:1:1 | 3.62e-7 | 2.38e-6 | 7.83e-5 |
| Outward graded, 1:0.2:0.05 | 5.50e-7 | 4.55e-6 | 1.50e-4 |
| Outer only, 1:0:0 | 8.56e-7 | 7.45e-6 | 2.21e-4 |
| Middle only, 0:1:0 | 1.30e-6 | 7.26e-6 | 1.64e-4 |

Uniform per-unit coupling reduces final error about 11.5-fold relative to inner
only when output feedback is absent. At step 1024 the corresponding errors are
0.01360 and 0.02534, so the effect is not confined to nearly converged outputs.
Both first pass the declared quality and convergence criteria at the reported
4096 horizon; our coarse readout schedule does not establish their exact crossing
times. Do not turn a final-error ratio into a latency speedup.

All main-budget configurations with feedback 0 or 0.25 pass both thresholds by
4096. At feedback 1, only uniform coupling passes the quality threshold, and
none passes the all-case convergence criterion. All these configurations have
numerically correct stationary outputs; residual finite-time error is relaxation,
not a different exact solution.

## Strength is a separate parameter

For the inward-graded profile, the fully reciprocal final RMSE changes as follows:

| Total budget relative to main comparison | Final RMSE | Quality <=1e-4 | All-case residual <=1e-5 |
| --- | ---: | --- | --- |
| 0.25× | 0.002047 | No | No |
| 1× | 0.000130 | No | No |
| 1.6×, raw weights (0.05,0.2,1) | 0.0000627 | Yes | No |
| 4× | 0.0000146 | Yes | Yes |

Higher strength helps within this tested range; it is not a monotonicity result
for arbitrary gains or step sizes. All tested constrained update operators have
spectral radius <=1 within numerical tolerance. More aggressive parameters might
require a different integration step, which we did not tune.

At 4× inward strength, removing output feedback still lowers final error to
1.12e-7, compared with 5.31e-7 for feedback 0.25 and 1.46e-5 for feedback 1.
This is the lowest observed final error in the declared grid, not an independently
validated optimum. The output still has no independent evidence in this task,
so its feedback supplies transient resistance without a new information source.

## Forward-only and missing-evidence controls

All six forward-only profiles settle but retain systematic error. Their final
RMSE ranges from 0.23958 (inner only) to 0.11995 (outer only). Grading can reduce
but does not eliminate the unsupported-input-default bias under these particular
local rules. Their dynamically selected stationary solutions have the same error.
This is not evidence that a different feedforward fusion rule could not solve
the task; the direct reference does.

Missing evidence is scored against its compatible affine output set. For uniform
coupling with no output feedback, final set-distance RMSE is 2.10e-7 with missing
context and 1.57e-7 with missing input. For inner-only/no-output-feedback these
are 3.08e-6 and 2.81e-6. An output lying in the compatible set does not express
uncertainty: this architecture still emits one chosen reconstruction.

Cue interventions agree with the aggregate result. For uniform/no-output-feedback,
changing input with context held fixed gives output-change RMSE 4.46e-7; changing
context with input fixed gives 3.88e-7 (24 pairs each). These measure correct
cue-conditioned changes, not activity movement alone.

Contradictory fixtures retain their common-code inconsistency of 0.707107 and
have no correctness target. Artifacts report each configuration's weighted
compatibility energy and distance to its selected stationary output. These energies
use different profile weights and must not be ranked across profiles as a common
accuracy measure. Stationary local incompatibility is also retained. We do not
claim that any compromise is the true answer or that the model detects conflict.

## What this means for learning

The results support exposing layer strengths and direction gains as candidate
learned parameters. They do not establish how to learn them. In particular,
minimizing energy with respect to a coupling strength can simply turn the
connection off. A future consolidation rule needs evidence of predictive
usefulness across experiences, rather than agreement or low energy alone.

The [consolidation proposal](cross-experience-consolidation-proposal.md) now
includes connection strength, feedback gain and representation translation as
distinct parameter families. Support-driven inertia can apply to these relations
as well as local reconstruction weights. Their update rules remain unimplemented.

## Verification and computation

The 33 declared configurations were all evaluated: 924 scientific episodes,
28 unique fixtures, 16 complete pairs, no new independent data from repeated
exposures. Complete trajectories are retained at all eight horizons. All 1,386
finite-difference gradient coordinates passed (maximum error 8.59e-10), as did
orthogonality/alignment, normalized budget, independent directional-gradient
assembly, topology, frozen parameters, clamping and symmetric-arm descent checks.
132 constrained stability eigendecompositions found no unstable configuration.
33 energy-Hessian eigenchecks verify the symmetric descent step-size bound.

Stationary diagnostics use left-nullspace invariants to select the reachable
solution, avoiding the previous minimum-norm ambiguity. All 924 SVD/augmented
least-squares diagnostics passed their residual and rank checks. They do not
replace measured finite-time convergence tests.

A separate artifact verification compared 672 historical trajectory rows from
three inner-only controls: output differences were exactly zero. All 7,392 new
trajectory rows have finite outputs/residuals and exact clamps; source hashes
and total update accounting match. This verification introduces no additional
candidate models or trajectories.

Main execution took 83.935 wall seconds, before runtime serialization:

- Construction: 0.0067 s; verification: 0.1446 s.
- All-candidate inference, scoring and stationary diagnostics: 47.8602 s.
- All-candidate benchmarking: 35.6486 s.
- Scientific serialization and provenance: approximately 0.274 s.

All candidate costs are included, including unsuccessful configurations. There
is no selected deployment model, separate tuning grid, or held-out confirmation.
No fitting, optimizer updates or training occurred. Manual design time is
unmeasured. The separate artifact-check cost is recorded in its own runtime file
and must be added to the main study cost when reporting the complete workflow.

There were 15,138,816 iterative updates and 7,260 iterative calls, including
warmup and measured batches, plus 92 direct calls. The main counter's SVD-based
least-squares calls are supplemented by the explicitly recorded 924 stationary
solves; they are not absent just because the imported counter stays zero.
Process CPU time exceeds wall time during the SVD-heavy phase because numerical
libraries may use multiple threads; it includes all process threads. There are
no child workers. BLAS configuration and thread environment are recorded; actual
backend thread count is unavailable because threadpoolctl is not installed.

Every candidate was benchmarked at 256, 1024 and 4096 updates using one warmup
and three batches of 16 sequential calls. Allocation/updates are included;
scoring is excluded. At 4096, median time is 12.73 ms for inner-only/no-output-
feedback and 12.71 ms for uniform/no-output-feedback. Their small timing difference
is not a speed claim: both execute the same dense matrix dimensions. Uniform
coupling gives better error at comparable measured cost. Direct cue SVD and
expansion takes 0.0249 ms median and reconstructs complete cues to numerical
precision; it remains the computational reference, not the mechanism success gate.
Full wall/CPU ranges and quality horizons are retained in the artifacts.

Executed dense cost is O(P M T S²), with P=33, M=28, T=4096, S=42;
construction is O(L S²), SVD/eigen diagnostics O(S³) each. Deployed matrix plus
activity state is 14,448 float64-array bytes regardless of grading. Candidate
dynamics/energy matrices occupy 931,392 bytes; candidate edge operators 310,464;
alignment maps 2,016; local maps 960; expanded maps 384. These named arrays plus
one activity vector total 1,245,552 bytes; the runner additionally retains baseline
operators, local residuals, reference/projection arrays, fixtures, diagnostics and
Python/report containers. Fixture numeric payload is 832 bytes. Peak single-
process RSS is 84,604 KiB including those overheads and temporary SVD workspace.
No replay, optimizer or learned statistics are retained. Deployment cost remains
construction + Q*inference. Historical timings are not mixed into these comparisons.

## Complete grid

The following table retains every declared candidate. Q denotes the first reported
horizon meeting RMSE<=1e-4; C denotes the first with every complete case's residual
<=1e-5. A dash means the criterion was not reached by the 4096 cap.

| Configuration | Final RMSE | Q | C |
| --- | ---: | ---: | ---: |
| inner_only_b0 | 4.16871e-06 | 4096 | 4096 |
| inner_only_b0.25 | 1.80622e-05 | 4096 | 4096 |
| inner_only_b1 | 0.000281146 | — | — |
| inward_b0 | 1.20869e-06 | 4096 | 4096 |
| inward_b0.25 | 6.08414e-06 | 4096 | 4096 |
| inward_b1 | 0.00012996 | — | — |
| uniform_b0 | 3.6225e-07 | 4096 | 4096 |
| uniform_b0.25 | 2.37984e-06 | 4096 | 4096 |
| uniform_b1 | 7.82565e-05 | 4096 | — |
| outward_b0 | 5.49851e-07 | 4096 | 4096 |
| outward_b0.25 | 4.55148e-06 | 4096 | 4096 |
| outward_b1 | 0.00015017 | — | — |
| outer_only_b0 | 8.55923e-07 | 4096 | 4096 |
| outer_only_b0.25 | 7.45438e-06 | 4096 | 4096 |
| outer_only_b1 | 0.000221038 | — | — |
| middle_only_b0 | 1.30457e-06 | 4096 | 4096 |
| middle_only_b0.25 | 7.26405e-06 | 4096 | 4096 |
| middle_only_b1 | 0.000164169 | — | — |
| inward_gain0.25_b0 | 3.11053e-05 | 4096 | 4096 |
| inward_gain0.25_b0.25 | 0.000171302 | — | — |
| inward_gain0.25_b1 | 0.00204695 | — | — |
| inward_gain4_b0 | 1.11878e-07 | 4096 | 4096 |
| inward_gain4_b0.25 | 5.30928e-07 | 4096 | 4096 |
| inward_gain4_b1 | 1.45888e-05 | 4096 | 4096 |
| inward_raw_b0 | 5.3553e-07 | 4096 | 4096 |
| inward_raw_b0.25 | 2.65882e-06 | 4096 | 4096 |
| inward_raw_b1 | 6.27199e-05 | 4096 | — |
| inner_only_forward | 0.239579 | — | 4096 |
| inward_forward | 0.219008 | — | 4096 |
| uniform_forward | 0.170181 | — | 4096 |
| outward_forward | 0.130112 | — | 4096 |
| outer_only_forward | 0.119951 | — | 4096 |
| middle_only_forward | 0.195878 | — | 4096 |

Artifacts: [contract](graded-three-module-v1-contract.md), [runner](../experiments/graded_three_module.py), [scientific results](../reports/graded-three-module-v1-results.json), [runtime](../reports/graded-three-module-v1-runtime.json), [verification](../reports/graded-three-module-v1-verification.json), [verification runtime](../reports/graded-three-module-v1-verification-runtime.json).
