# Three-module ambiguity v1: prospective contract

2026-10-26. Input ↔ intermediate ↔ output, each with continuous 8→4→2
activities. Frozen supplied linear relationships, no STM, no learning, zero free
initial activities. All 42 activities evolve except observed outer coordinates.
Read all eight output outer activities at 0,1,4,16,64,256,1024,4096 steps.

Each module has local residuals x-D_m h and h-H_m c. Let R have rows
(1,0),(0,1),(1,1)/sqrt(2),(1,-1)/sqrt(2), divided by sqrt(2).
Let U stack I4 and the normalized 4×4 Hadamard matrix, divided by sqrt(2).
H_input=R, H_middle=R with rows 0 and 1 exchanged, H_output=R.
D_input=D_middle=U; D_output is U with reversed rows and alternating row signs.
Local energy is half the sum of squared residuals. Inner codes share coordinates;
neighbor compatibility adds half ||c_i-c_j||² on each edge. No direct input–output
edge, no outer equality coupling. Inner-only inter-module coupling is an explicit
scope limitation; this does not test graded all-layer coupling or learned maps.
The two-layer reconstruction has rank two; the hierarchy is linear and supplied.

For each of 16 pairs (a,b) in {-1,-0.5,0.5,1}², clamp input x[0]=a/2
and middle x[0]=b/2. Output has no cue. Its target is D_output H_output (a,b).
Neither cue alone determines the full output. Evaluate changes of each cue while
holding the other fixed. This is an exhaustive fixture, not learned generalization.

Four arms retain both forward edges: reciprocal, forward-only, remove middle→input,
and remove output→middle. Each directed edge uses the same compatibility residual
but applies its derivative only to the receiving module. Reciprocal updates are
simultaneous gradient descent; directed updates are not asserted to minimize the
reciprocal energy. Rate 0.1, exactly 4096 steps, no cap extension or tuning.
Compare equal counts and measured latency at each horizon. Residual tolerance 1e-5;
quality threshold 1e-4 RMSE. Record first qualifying declared horizon, never stop
readout pending convergence. Frozen oracle uses least squares on the declared
linear dynamics to diagnose final bias separately from finite-time error.

Additional fixtures: four missing-context cases (a from the same four values,
no middle clamp); four missing-input cases (b likewise); four contradictory cases
(a,b) in {-1,1}² with input x[0]=a/2, input x[1]=b/2,
and middle x[0]=-b/2. Thus the last group has no common latent solution.
For missing evidence report distance to the compatible continuous affine set of
outputs, its dimension, and the selected inner code. Set membership is not a claim
of certainty or recovery of a hidden missing value. Contradictory cases have no
correct target: report common-code evidence inconsistency, reciprocal energy,
clamp residual, and distance from the exact reciprocal constrained energy optimum.
Never score them against a manufactured correct answer.

The direct joint-input reference solves the 2-column observed-code equations by
SVD least squares and expands the output using the same maps. Report rank and
residual, with minimum-norm selection explicitly disclosed for missing cues.
For inconsistent cues its least-squares compromise is diagnostic, not ground truth.
The constrained reciprocal optimum additionally uses SVD least squares of the
free-state Hessian; neither oracle output enters dynamics. No labels or desired
inner activities are supplied during inference.

Checks: finite-difference reciprocal gradients; local map shapes/rank; clamping;
frozen maps; reciprocal energy descent; zero direct input–output derivative;
correct directional derivative deletion; exact consistent direct reconstruction;
each cue changes output; missing-cue nullspace; contradiction infeasibility;
free-state recurrence versus matrix update. Record stationary residuals and error
to each arm's own stationary solution. No training, candidates, validation or tuning.

For K modules, outer N, middle h, inner d: local map construction/storage
O(K(Nh+hd)), local sparse update O(K(Nh+hd)+Ed). This prototype explicitly
assembles a dense S×S dynamics matrix, S=K(N+h+d): construction O(S²) workspace
plus local products, update O(S²), T steps O(TS²), activity O(S). Do not attribute
sparse cost to the executed dense implementation. With L residual rows, dense
Hessian construction O(LS²), SVD stationary diagnostics O(S³), workspace O(S²).
Direct cue least squares O(qd²+d³) and output expansion O(Nd); retained output
map O(Nd). State and all stored maps/matrices are float64 and counted in runtime.
No optimizer, replay or learned statistics. Report fixture/report arrays separately
from model state, peak single-process RSS, versions, BLAS/thread configuration,
limits, hashes, revision/dirty status and counters including verification and
benchmarks. Warm up once, then measure five batches of 16 sequential calls at each
nonzero horizon and direct reference; report median/range wall and CPU per query.
Allocation, SVD for direct, and updates included; scoring excluded. Separate
scientific JSON from runtime JSON. Count all diagnostic SVDs and construction,
no discarded candidates. Manual design cost is unavailable. Deployment cost is
construction + Q*inference; no learning or selection cost. Historical timings
are different-task measurements and will not be used as equivalent comparisons.
