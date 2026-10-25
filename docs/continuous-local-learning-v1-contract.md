# Continuous local learning v1: prospective contract

Status: frozen before execution, 2026-10-25. A new graded candidate on the known
four-bit development fixture. No claim of general learning, equivalence to the
earlier discrete circuit, or success on the frozen 72-scene task. Preserve all
previous sources/artifacts. Follow [computation reporting](computation-reporting.md).

## 1. Candidate and locality

Four cyclic outer activities x_i in [0,1] feed four shared edge channels z_i in
[0,1]^4. The pooled edge state feeds eight decoder activities h in [0,1]^8 and
four real output activities y. Define sigmoid sigma and append a bias coordinate
where shown:

```
phi_i = [x_i, x_(i+1), 1]
f_i = sigma(W phi_i)          W shape 4x3
v = [mean_i z_i, 1]
g = sigma(A v)                A shape 8x5
r = B [h,1]                  B shape 4x9
E = 0.5 sum_i ||z_i-f_i||^2 + 0.5 ||h-g||^2 + 0.5 ||y-r||^2
    + 0.0001/2 (||W||^2+||A||^2+||B||^2).
```

There are 88 learned float64 parameters (704 bytes). Mutable activity is
4+16+8+4=32 float64 values (256 bytes). E is a local-factor energy: each edge
prediction compares neighboring outer values with its own edge channels, and
decoder factors compare adjacent levels. Activity derivatives include reciprocal
effects through the same weights. Parameter derivatives use local residuals and
their factor inputs; no unrolled inference/backpropagation is used by this arm.

Pooling and four-location weight sharing are supplied. Implementing their common
parameter requires combining four local contributions; this is not a claim of
independent biological synapses learning without communication. Labels define
the four classes from the earlier study. The four y coordinates encode a class
by argmax, not a binary inner mode. This is a new continuous representation and
does not preserve the exact quotient identity automatically.

## 2. Simultaneous state and parameter dynamics

At every stream step, compute all gradients from the old state and parameters:

```
state_new = project(state - a*0.1*grad_state E)
theta_new = clip(theta - a*0.005*grad_theta E, -3,3).
```

Project x,z,h to [0,1]; impose observed x and supplied y target exactly. Unclamped
y is unconstrained. The same multiplier a starts at 1 each step and is halved
up to 12 times until joint energy does not increase beyond 1e-12. No new accepted
step is chosen by task accuracy. If none is accepted, stop as a numerical
correctness failure, not a scientific negative. Count every energy evaluation
and rejected trial. This shared line search is an explicit global coordination
mechanism: the update derivatives are local-factor derivatives, but the numerical
integrator is not wholly decentralized.

For fixed evidence, continuous projected gradient flow of the joint energy has
a descent property. Discrete line search checks descent on each unchanged-evidence
step. Evidence changes may raise energy; do not conflate that with instability.
Energy decrease does not prove correct grouping, global minimization, avoidance
of collapse, or persistence of earlier knowledge.

During the learning stream, all four x values and a one-hot class label y are
clamped, while z,h and weights update simultaneously on every step. States carry
across presentations; there is no free/nudged or settle-then-learn cycle. Labels
arrive as external evidence. This is supervised continuous adaptation, not learning
from its own inferred answer. Diagnostic copies may freeze parameters for valid
evaluation without changing the main stream's update rule.

## 3. Direct predictive comparator

The same functions have a direct prediction: z=f(x), h=g(mean z), y=r(h).
With fixed weights, full x observation, and free y, it has zero prediction
residuals and is the unique zero-residual activity solution. Use this direct
prediction for classification, never the training-clamped y. It is also the
mandatory simpler inference comparator.

The training comparator is projected online gradient descent on
0.5||prediction(x)-target||^2 + weight_decay, differentiating through the direct
functions, with parameter step 0.005 and the same clipping/backtracking rule.
It has the same initialization, stream, number of parameter-update attempts,
and labels, but no latent settling steps. Its objective/gradients differ from
joint energy adaptation, so equal update counts do not imply equal compute.
Charge both full costs and do not infer an optimal optimizer from this comparison.

## 4. Fixed stream, seeds, and checkpoints

Root seed 2026102511, replicas 0,1,2. NumPy SeedSequence([root,replica,0]) creates
independent normal initial W and A entries with standard deviation 0.5, B with
standard deviation 0.2, clipped to [-3,3]. Paired arms use identical arrays.
Initial z,h are 0.5. x,y are set by the first external observation. A separate
SeedSequence([root,replica,1]) controls presentation permutations.

Block A contains 200 passes over the same eight labeled patterns used in the
local-circuit study (0000,1111;0001,1110;0011,1100;0101,1010). Each pass uses
one seeded permutation. Each presentation lasts 40 joint/parameter steps:
1,600 presentations and 64,000 steps per arm/replica.

Block B then presents two previously withheld rotations, 0010 (one different)
and 0110 (adjacent pairs), alternating for 100 presentations, 40 steps each.
This introduces new patterns and shifts the class frequency, not new class
semantics. It is a retention/interference probe, not evidence of new relational
discovery. The same learning rule stays active in both blocks.

At initialization, after A and after B record direct classification, squared
prediction errors, class counts, parameter norms, and prediction spread on all
16 known patterns. The eight originally withheld patterns are an evaluation
set after A; after B, two have entered learning and cannot be called held out.
Report original training, A-held-out, B-exposed, and still-unseen sets separately.
The fixture is already known development material, not fresh confirmation data.

No early stop on accuracy, parameter search, additional seeds, or extension after
failure. Six trajectories (2 arms x 3 replicas), 408,000 update attempts total.
There are 10,200 presentations and only 10 distinct labeled patterns ever
presented; exposures and repeated integration steps are separate counts.

## 5. Completion while parameters are fixed or adapting

After A and B, create independent diagnostic copies of each model. For each of
the eight original labeled patterns and each of the six pairs of observed
coordinates, expose those two bit values and clamp y to the true class one-hot.
This is a fixed 48-query battery (possible repeated evidence contexts are reported
as repeated queries, not independent examples). Hidden values are not supplied.
The class is an explicit conditioning signal, not a model prediction or hidden
target leak. This is stricter class conditioning than the old soft inner prior.

Initialize hidden x at stored default s=0011 and observed x at evidence. Initialize
z,h from the direct functions at that x. Add

```
E_default = 0.05/2 ||x-s||^2
E_binary = 0.05 sum_i x_i^2 (1-x_i)^2.
```

The default is a preference, not extra evidence. For 300 steps, run either frozen
weights or the same simultaneous weight dynamics. Both retain exact observed
values and the supplied class. Every query starts from a fresh copy; plastic
probe outcomes never initialize a later query or the main learning stream.

Report thresholded class validity, membership in the nearest-default discrete
completion set, hidden continuous error to that set, energy monotonicity,
parameter drift, and projected activity residual. The task oracle enumerates the
16 binary patterns satisfying class and observed bits; it is NOT a global
minimizer of the new nonconvex continuous energy. Scores compare behavior, not
claim exact energy equivalence. All 48 queries have feasible semantic completions
by construction. Inferred h/z values must not be scored as observed target answers.

Define activity-settled as max absolute projected unit-step gradient residual
<=1e-5 at the end, excluding clamped coordinates. Report all nonsettled cases
as truncated, even if thresholded outputs happen to be right. For plastic probes,
also report gradient residual of parameters and their norm drift; activity
settling alone does not imply a stationary coupled state. Compare frozen/plastic
accuracy and drift, and compare pre/post-block-B grouping for interference.

48*2 checkpoints*2 plasticity settings*6 models*300=345,600 probe steps, all
charged. No probe score changes the learned model or hyperparameters.

## 6. Correctness checks and inference benchmark

Before trajectories, finite-difference-check every state/parameter derivative of
the joint energy including completion terms at an interior deterministic point;
central step 1e-6, max absolute disagreement <=2e-7. Check every direct predictor
parameter derivative similarly. Check a projected joint step lowers fixed-evidence
energy and preserves all clamps. No development fit or score-based pilot.

At post-A replica 0 of each arm, compare direct prediction with frozen-weight
activity settling on all 16 fully observed patterns, free y. Initialize z,h=0.5,
y=0. Run up to 500 steps, stopping when projected activity residual <=1e-5;
record truncated cases and maximum direct-output disagreement. One warmup batch
and three measured batches per method/arm. Report batch median and min/max,
amortized per-query time, iteration counts and convergence; include allocations,
line search and gradients. The settled benchmark also computes one direct
reference per query for its disagreement check; this overhead is included and
disclosed. No inference speed claim if error tolerance is unmet.

## 7. Computation and storage

For n outer units, k edge channels, m decoder units, c classes, each dense local
energy/gradient evaluation costs O(nk+km+mc) time and O(nk+km+mc) workspace.
Parameters take O(k+km+mc), activity O(n+nk+m+c), and no sample replay or optimizer
state is used. With J backtracking evaluations and T integration steps, work is
O(T*(J+1)*(nk+km+mc)); count J rather than hiding it in "one update". The direct
predictor has O(nk+km+mc) cost without settling. Completion is nonconvex and no
uniform bound on required settling steps is claimed.

Float64 arrays: parameters 704 bytes, activities 256 bytes. Gradient arrays and
candidate copies are transient allocations of comparable size; no autograd tape,
momentum, or covariance. Report retained trained-array bytes, query copies,
training/probe update counts, backtracking, gradients, objective/forward calls,
and process peak RSS separately. The driver retains checkpoint arrays and logs
for research; this is not part of the model's inference memory. Raw samples are
externally provided by the fixture generator, not retained as replay by the model.

Set BLAS/OMP thread environment to one before NumPy import. One CPU process, no
workers or accelerator. Record NumPy version/backend, interpreter/platform/CPU,
source/contract hashes, revision/dirty status, seed policy, command, named phase
wall/process-CPU seconds, counts and RSS units. Deterministic science and timing
receipts are separate. Treat unavailable allocator-level workspace as null with
a reason. Include rejected line-search proposals and both arms in total costs;
all six declared trajectories are reported, with no best-seed selection.

The runner is `experiments/continuous_local_learning.py`, launched using
`.venv/bin/python -m experiments.continuous_local_learning`. First execution
freezes source/contract hashes in its receipt. Do not overwrite existing reports.
Failure after that point must be disclosed before a corrected rerun. Manual
development cost is unmeasured. Whole-run timing excludes imports/provenance and
runtime-receipt serialization; named phases include numerical checks, learning,
all shadow diagnostics, benchmarks, and scientific serialization.
