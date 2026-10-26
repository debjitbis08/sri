# Local consolidation v1: reviewer experiment

Prospective contract, 2026-10-26. Test a module-local, role-independent learning
rule in three 8→4→2 modules. Every module uses the same algorithm. Pattern IDs
exist only in the evaluator. Fixed chain inner coupling 0.2; inter-module gains
and translations are not learned in this first consolidation study. Learn each
module's W (8×4) and V (4×2), initialized independently from N(0,0.25²), seeds
0,1,2. No supplied pattern encoder, no supervised hidden code, no global task loss
or backpropagation through activities. This is block-local (module-wide), not
strictly single-synapse-local.

Local activity energy is 0.5||x-Wh||²+0.5||h-Vc||²+0.025(||h||²+||c||²).
Neighbor codes have 0.1||c_i-c_j||² compatibility. Activities perform 32 implicit
Euler steps of size 1 from zero free state, with outer evidence clamped. Solve
(I+M H_ff) once per call, then apply the affine update; M=I except the slow-inner
control uses inner mobility 0.1. No stopping early. Expose residuals and final
inner change as diagnostics; stability is not a correctness criterion.

## Data and waking/sleep protocol

For each seed independently generate random orthonormal 8×2 pattern maps per
module, kept private to the fixture/evaluator. Two structures are their two
columns. Each experience has a shared amplitude U[0.8,1.2] and independent local
Gaussian nuisance (standard deviation 0.1), projected perpendicular to those
columns. Stage A contains 16 variations of structure 0, stage B contains 16 of
structure 1. Learners receive only the 3×8 observed activities and unique episode
IDs, not the generator, class labels or amplitude. Eight even-index episodes per
stage supply proposals and eight odd-index episodes supply local replay checks.
Retain all 32 waking episodes; this is bounded explicit replay, not memory-free
learning. Waking activities/temporary state can change but long-term weights
remain byte-identical. Reset activities each episode/call; no cross-query STM.

After each waking stage, run six sleep cycles. Replay all available proposal
traces and infer activities with all observations clamped. For each module propose
ΔW=0.2*mean[(x-Wh)hᵀ]-0.0002W and
ΔV=0.2*mean[(h-Vc)cᵀ]-0.0002V, multiplied by its gate and inertia factor.
Proposal gradients use only that module's local activity and weights. The
regularizer prevents unbounded weight growth; candidate entries clip to [-2,2].
Apply proposals sequentially by module, re-inferring proposal states before each.

Stability gate g=1/(1+1000*mean||c_last-c_previous||²). Support-based arms use
inertia 1/(1+s/8), s the count of distinct supportive validation episodes for
that module. No-inertia control fixes this factor to 1. Repeated replay cannot
increase s twice for the same episode. Counts are a module-level support proxy,
not a per-synapse confidence claim. Ungated arms use g=1.

For support-based candidates, temporarily hide coordinate cycle mod 8 of that
module's outer layer in each validation trace. Its value must be excluded from
initialization and all activity updates. Other observations stay clamped. Compare
candidate and incumbent predictions of that same local coordinate. Accept only
if mean squared prediction error decreases by >1e-5, no individual validation
error increases by >0.01, and at least three improved episodes have pairwise local
observed-vector distances >0.05 (excluding the hidden coordinate). Update support
only for those distinct improved episode IDs. The gate, candidate and acceptance
rule do not see evaluator labels, final reconstruction scores or an input/output
role. Local check observations are self-supervised targets, not hidden identities.
Adaptive reuse of validation traces is training, not independent test evidence.

Eight matched arms: frozen; ungated local energy learning; stability-gated energy
learning; stability-gated energy learning with slower inner activity; predictive
support without stability gate; predictive support with stability gate; same
with inertia disabled; same using repetitions of only the first proposal/check
trace at equal replay batch sizes. This last arm has only one distinct check
identity and cannot satisfy the three-experience criterion; report this designed
rejection as a bookkeeping control, not an empirical discovery. All arms receive
the same waking stream and store the same trace pool. Counts/accepted updates
must be reported. No hyperparameter search or post-result extension.

## Reviewer evaluations

Before training and after each sleep stage, evaluate 12 unseen variations of each
structure. Compute global reconstruction by masking an entire module's outer
layer and cueing the other two, rotating the masked module over all three.
Report missing-module RMSE by structure and combined, without a designated
input/output module. All evaluations freeze weights and reset activities.

For abstraction, present complete held-out observations, record all six inner
activities, and compute within-structure squared spread, between-centroid squared
distance, and their ratio. Smaller spread alone is not success: an all-zero code
is a collapsed control. Evaluate a nearest-centroid classification probe using
separate labeled evaluator-only reference examples (8 per structure), never
feeding identities/centroids back to the learner or choosing parameters with them.
Report code norm, residuals and gate values. These metrics assess properties;
they are not a global optimization objective.

Retention: compare structure-0 reconstruction after stage A versus after stage B.
Acquisition: compare structure-1 reconstruction before versus after stage B.
Gate contribution: compare predictive-support arms with/without gate and energy
arms with/without gate. Slower-inner control tests whether apparent stability is
just inertia. Compare support/no-inertia for the intended retention/revision
tradeoff, without demanding maximal adaptability. A reviewer-positive result
requires improved unseen reconstruction for both structures relative to frozen,
noncollapsed discrimination, retention of stage-A structure within 10% RMSE, and
an additional benefit from the stability gate; report per-seed values and means,
not a significance claim from three seeds. Partial improvements do not satisfy
all these conditions.

Direct comparator: per-module ridge linear prediction from the other two outer
surfaces, trained on the same available waking traces (proposal and check pools),
ridge 0.1, no class labels. Count 3 fits per stage/seed and evaluation cost. Its
use of global observations is explicit; it is a task baseline, not the proposed
local mechanism. Test examples and labeled probes never enter any learning arm.

## Computation and verification

With K=3,N=8,h=4,d=2,S=42, B<=32, P=K(Nh+hd)=120 weights, C=36
proposals per seed/arm: local proposal O(BP), Hessian construction O(LS²), implicit
matrix solve O(S³), 32-step batched inference O(TBS²). Mask pattern is shared in
a call. Candidate checks repeat inference for incumbent and candidate; record all
rejected candidate costs. Stationarity/inference is not differentiated through.
Direct ridge fit O(BF²+F³), F=17 including intercept, workspace O(F²).
Retained model parameters O(P), traces O(BKN), support IDs O(KB), candidate copy
O(P), batched activity/workspace O(BS+S²). Include replay buffers and support ID
payload, not just weights. No optimizer momentum or covariance history retained.

Write deterministic results and separate runtime/provenance. Record construction,
all learning/validation and rejected proposals, evaluator/probe cost, direct fits,
serialization, counts, hashes/revision/dirty status, seeds, Python/NumPy/BLAS,
thread settings/limits, process CPU/wall and peak RSS. Set BLAS environment to one
thread before importing NumPy. Whole-run/phase timing is a mechanism study, not
a warmed deployment benchmark; do not make inference speed claims. No selection
or tuning; all eight arms retained. Manual design time unavailable.

Verify activity and parameter gradients by finite differences, exact clamps,
implicit reciprocal descent, frozen waking/evaluation weights, consistent
initialization across arms, no target coordinate leakage into inferred state,
no repeated-ID support inflation, and role permutation consistency. Historical
frozen studies remain unchanged and are not equal-task learning comparators.
Dreaming here is replay plus local proposal/check consolidation; recombination,
autonomous sleep scheduling and learned coupling remain unimplemented.
