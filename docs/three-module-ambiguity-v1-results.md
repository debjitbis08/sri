# Three 8→4→2 modules: useful input feedback, unnecessary output feedback

The three-module test reconstructs separate outer outputs from distributed
partial evidence. The strongest tested configuration is input ↔ intermediate →
output. Adding output → intermediate feedback slows inference without improving
the exact answer in this fixture. Removing intermediate → input feedback leaves
a persistent context-dependent bias.

The full reciprocal chain improves substantially over forward-only inference,
but fails the declared finite-budget quality and convergence thresholds at 4096
updates. We did not extend the cap. This is a qualified mechanism result, not
validation of full reciprocity or evidence of a computational advantage.

## Architecture and scope

Input, intermediate and output modules each contain 8 outer, 4 middle and 2 inner
continuous activities: 42 activities total. Only the input and intermediate
outer surfaces receive evidence. No output coordinate is clamped. Fixed local
linear maps reconstruct each module's own outer pattern; only aligned inner
coordinates connect adjacent modules. There is no direct input–output connection.
Weights remain byte-identical, and explicit STM is disabled.

Input supplies a/2 and intermediate context supplies b/2. The output target is a
fixed rank-two expansion of (a,b). All 16 pairs from {-1,-0.5,0.5,1}² are evaluated.
Neither cue determines the entire output. All arms use the same evidence, maps,
zero free initialization, step size and horizons. The maps and alignment are
supplied, not discovered. This is linear completion with complementary coordinates,
not competition among nonlinear semantic interpretations. It tests the requested
three-module hierarchy, but not graded inter-module coupling across all layers.

## Main comparison

RMSE includes all eight output outer coordinates, over 16 complete-evidence cases.

| Connections | Step 256 | Step 1024 | Step 4096 | Cases unconverged at 4096 |
| --- | ---: | ---: | ---: | ---: |
| Input ↔ middle ↔ output | 0.270662 | 0.068042 | 0.00028115 | 16/16 |
| Input → middle → output | 0.260206 | 0.239580 | 0.23957871 | 0/16 |
| Input → middle ↔ output | 0.270878 | 0.239581 | 0.23957871 | 0/16 |
| Input ↔ middle → output | 0.223434 | 0.025341 | 0.00000417 | 0/16 |

The declared quality threshold was RMSE <=1e-4; convergence required free-update
residual <=1e-5 for every case. Only input ↔ middle → output reaches both at a
reported horizon (4096). Full reciprocity has maximum residual 4.79e-5 at the cap;
the successful arm has maximum residual 1.11e-6. Low residual in the two biased
arms confirms again that settling does not imply correctness.

Changing each cue while holding the other fixed yields 24 paired comparisons per
cue. For input ↔ middle → output, output-change RMSE against the correct change is
5.03e-6 for the input cue and 4.58e-6 for context. Forward-only has nearly exact
input-cue changes but context-change RMSE 0.39123. Thus its problem is not failure
to move at all; its context influence is much too weak.

## What the feedback links do

With no middle → input feedback, the input module retains its unsupported second
coordinate at zero. Its outgoing compatibility pressure biases the middle's
second coordinate toward that default. At the dynamically selected equilibrium,
the middle/output second coordinate is b/7 instead of b. This persistent bias
accounts for RMSE 0.23957871. Allowing the middle to update the input's ambiguous
coordinate removes that competing default.

The output module has no independent evidence and no conflicting output prior.
At equilibrium it can reconstruct whatever code the middle supplies. Its return
connection therefore imposes no additional constraint on the correct solution,
but during settling its initially empty activities push back on the middle.
Removing this return connection improves finite-budget accuracy in this fixture.
This is a reason to test when a feedback link is useful, not a general argument
against output feedback in systems with richer priors or other evidence.

The reciprocal and input ↔ middle → output equilibria reconstruct the target to
numerical precision in an independent linear stationary analysis. Their reported
finite-time error is incomplete relaxation. The two other arms have a biased
selected equilibrium. A different forward-only rule that communicates uncertainty
instead of treating an unsupported zero as evidence might avoid this bias. Our
edge deletions do not establish that feedforward fusion generally fails.

### Stationary diagnostic clarification

The main runner initially diagnoses stationarity with minimum-norm least squares.
Some directed systems have a family of stationary states: the minimum-norm member
need not be the member reached from zero initialization. Consequently the field
`own_stationary_output_rmse` is distance to that chosen least-squares solution,
not universally distance to the trajectory's limit. In the biased complete-cue
arms it is 0.02756 despite an already tiny update residual.

A separate post-run analytic audit preserves the original artifacts and resolves
this issue using conserved left-nullspace projections. For free dynamics
`y'=-B y+r`, left-null vectors L satisfy `Lᵀr=0`, so `Lᵀy` remains at its initial
zero value. Solving the stationary equations together with these invariants gives
a unique selected stationary state for every tested case. Final-to-selected
stationary RMSE is 1.34e-12 for forward-only and 2.11e-12 when only output feedback
remains; their target bias is still 0.23958. This audit adds no trajectories,
hyperparameter choices or candidate models. It corrects interpretation of the
stationary diagnostic, not the scientific error measurements.

## Missing and contradictory evidence

Four cases omit context; four omit input. Each has a one-dimensional continuous
set of compatible output patterns. We score distance to that set, not to an
invented missing value. At 4096:

| Connections | Missing context: set-distance RMSE | Missing input: set-distance RMSE |
| --- | ---: | ---: |
| Full reciprocal | 0.00024064 | 0.00014538 |
| Forward-only | 1.34e-12 | 0.23957871 |
| Input → middle ↔ output | 2.11e-12 | 0.23957871 |
| Input ↔ middle → output | 0.00000308 | 0.00000281 |

The asymmetry exposes the unsupported-default problem: an unobserved input can
still suppress a real context cue downstream. Selecting a compatible point when
evidence is missing does not represent uncertainty explicitly; this model emits
one reconstruction. The direct reference's missing coordinate is a disclosed
minimum-norm choice, not inferred knowledge.

Four contradictory cases clamp incompatible values of b at the input and middle
surfaces. There is no common-code solution; the direct observed-code residual is
0.707107 in each case. We assign no correctness target. The exact constrained
reciprocal energy optimum is 2/13 ≈0.153846. Full reciprocity approaches that
optimum (mean energy 0.153846199) but all four cases remain above the convergence
threshold. Input ↔ middle → output reaches 0.153846154 and converges. The two
biased arms settle at energy 2/7 ≈0.285714. Clamp error is zero throughout.

These results show compromise under inconsistent observations. The evaluator
knows the inconsistency; the model has no learned conflict detector or calibrated
confidence output. Low energy is not evidence that its compromise is true.

## Computation and verification

The main run took 11.7902 wall seconds before runtime serialization. It includes
112 scientific episodes, verification, stationary diagnostics, scoring and
benchmarks. There were 2,555,780 updates, 2,804 iterative inference calls and 1,056
direct calls including diagnostics, warmups and benchmarks. Before benchmarks:
458,756 updates and 116 iterative calls. There are 28 unique fixtures, including
16 complete pairs; repeated exposures are not independent examples. No fitting,
training, validation, candidate search, tuning or parameter updates occurred.
Manual design time is unavailable.

All reciprocal activity gradients were checked (maximum error 7.76e-10), along
with independent local-versus-matrix gradient equivalence, topology, directional
derivative deletion, clamping, reciprocal descent, frozen relationships, cue
identifiability, missing-cue nullspaces and contradiction infeasibility. The
largest reciprocal Hessian eigenvalue is 4.4605, consistent with the chosen 0.1
step size's descent bound. Directed dynamics are not asserted to descend that
same energy. Checks ran inside the measured main execution.

The stationary audit adds 0.1517 wall seconds, 112 SVDs and 112 augmented
least-squares solves, with separate runtime/provenance. Main counters include
1,460 least-squares calls and 264 explicit nullspace SVDs; map-rank verification
also uses three rank decompositions and the descent check uses one symmetric
eigendecomposition. Audit time must be added when reporting the total study cost
(approximately 11.942 seconds before the two runtime serializations).

Median measured per-query time at 4096 updates:

| Method | Median | Minimum–maximum |
| --- | ---: | ---: |
| Full reciprocal | 13.035 ms | 12.768–13.469 ms |
| Forward-only | 13.025 ms | 12.762–13.989 ms |
| Input → middle ↔ output | 13.873 ms | 12.865–13.971 ms |
| Input ↔ middle → output | 12.809 ms | 12.767–12.910 ms |
| Direct joint-input SVD and expansion | 0.0210 ms | 0.0209–0.0215 ms |

Each measurement uses one warmup and five batches of 16 sequential queries.
Allocation and updates (or the direct SVD) are included; scoring is excluded.
All horizon timings and CPU measurements are retained. Small timing differences
between iterative arms are not hardware-speed claims: they execute dense matrices
of equal size. The direct solution is a correctness reference and has the better
quality–latency tradeoff here; beating it was not the mechanism criterion.

The actual implementation uses dense 42×42 matrices: O(S²) work per update,
O(TS²) per query, O(S²) matrix storage, O(S) activities; dense residual-to-Hessian
construction costs O(LS²). Stationary SVD diagnostics cost O(S³). A local sparse
implementation could cost O(K(Nh+hd)+Ed) per update but was not benchmarked.
The direct q×2 cue solve costs O(qd²+d³) and output expansion O(Nd), d=2.
Deployment cost is construction + Q×inference; there is no training/selection.

The minimal dense deployed operator plus activity state occupies 14,448 bytes.
The runner retains 85,680 bytes of model/operator arrays plus one activity vector,
including all four controls and diagnostic operators. Local maps are 960 bytes;
expanded maps are 384 bytes; fixture numeric payload is 832 bytes. No optimizer,
replay or learned statistics are stored. Python/report containers and temporary
arrays are additional; main single-process lifetime peak RSS was 41,508 KiB.
Runtime JSON records BLAS, threads, CPU limits and provenance. Historical scalar
and two-module timings are not equivalent-task comparisons.

## Conclusion and next boundary

Separate input/output with three 8→4→2 modules works as a frozen linear inference
mechanism. This task supports feedback between the evidence-bearing modules,
not the necessity of feedback from every module. A subsequent learning study
should preserve that distinction when testing experience-dependent inertia;
we have not implemented or validated sleep/dream consolidation here.

Artifacts: [prospective contract](three-module-ambiguity-v1-contract.md),
[runner](../experiments/three_module_ambiguity.py),
[science](../reports/three-module-ambiguity-v1-results.json),
[runtime](../reports/three-module-ambiguity-v1-runtime.json),
[stationary audit](../reports/three-module-ambiguity-v1-audit.json),
[audit runner](../experiments/three_module_ambiguity_audit.py),
[audit runtime](../reports/three-module-ambiguity-v1-audit-runtime.json).
