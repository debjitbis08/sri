# Stored-pattern hierarchy v1

Prospective contract, 2026-10-26. Mechanism test informed by the negative joint
learning studies and the outer-anytime result; historical artifacts are unchanged.
No hyperparameter search or outcome-based extension. This is a constructive,
realizable synthetic domain, not a general-learning or discovered-mode claim.

## Architecture and algebra

Two distinct modules each have layers of widths 8,4,2,1. Only the input module's
outer eight values are clamped. Only the output module's outer eight values are
task outputs. The two innermost scalar activities couple with strength 4.
There is no supplied inner target, class label, learned external decoder, or
output clamp during evaluation. Read continuous outputs at steps
0,1,4,16,64,256,1024,2048, irrespective of residual.

For children a,b and parent c, m=(a+b)/2 and d=(a-b)/2. The local energy is
E_local = ((c-m)^2 + (d-f(c))^2)/2,
where f(c)=w0+w1*c+w2*c^2 is a stored preferred detail. The inward map is exact
pair averaging; the outward preferred pair is (c+f(c),c-f(c)). This is nonlinear
interpretation from stored relationships, not recovery of arbitrary lost detail.
Pair averaging removes the local antisymmetric Haar mode while retaining the
constant mode, recursively. This is a hierarchical mode decomposition, not an
ideal global Fourier low-pass filter. Grouping is supplied and fixed.

Total energy sums the 14 local factors, adds 4/2*(core_in-core_out)^2 and
0.005/2*(core_in^2+core_out^2). The weak core prior selects zero in the absence
of evidence; details at zero need not be zero. Thus no evidence yields a stored
pattern. A clamped all-zero input is evidence and is NOT equated to absent input.
Activity is unbounded float64, not binary or clipped. Dynamics are simultaneous
reciprocal gradient updates, not a completed inward pass followed by an outward
pass. Initial state is each module's preferred reconstruction from core zero.
Use state rate 0.1, backtrack by halves up to 20 times for non-increasing energy.

## Data and fitting

Generate paired input/output signals from the same scalar s by recursive expansion
using module-specific quadratic detail functions. At level l=0,1,2 (outer to
inner), parent index j, n=4,2,1 respectively, generator coefficients are
w0=0.08*(j+1)/n+0.03*module,
w1=0.12*(-1)^(j+l+module), w2=0.35*(-1)^(j+module).
These coefficients construct the task; they are not copied into fitted models.
Train on nine equally spaced s in [-0.8,0.8]. Test on the eight interval midpoints.
Fit each local conditional detail using least squares on observed parent averages
and child differences, separately per module and factor. This uses paired
training signals but no hidden labels. Full training examples are discarded at
inference. Quadratic and affine models each require 14 fits, batch size 9.
There are 9 independent training pairs and 8 test pairs; module views do not double
that count. Training is offline for this first capacity test.

Predeclared arms: quadratic frozen; quadratic with simultaneous local parameter
updates (rate 0.0001); affine frozen; zero-detail frozen; quadratic frozen with
core coupling removed. Plastic gradients are local energy gradients, not task
loss gradients; include drift and do not imply they discover a useful code.
All arms run all eight held-out queries for 2048 updates. Plastic query copies
start at the same pretrained parameters (not a continual stream). Include no-
evidence equilibrium and finite-difference checks. Include direct reconstruction:
mean of input outer values, then output preferred expansion, for each fitted
model. This comparator bypasses settling and the weak core prior, so compare
quality as well as cost. It demonstrates that this task itself requires no
iterative network. Zero-initialized output-only MSE and per-horizon RMSE give
baselines; no binary threshold or inner classification counts as output success.

## Checks and cost

Check energy gradients centrally (step 1e-6, tolerance 1e-7), averaging identities,
fitted training residuals, no-evidence gradient, clamp preservation, finite state,
and descent on every step. Scientific failures do not abort. Record all outputs,
energy, projected/free gradient norm, parameter drift and horizon error. A residual
above 1e-5 marks an unsettled readout; it never gates action.

For outer width N=2^L and p<=3 polynomial coefficients, construction and local
update cost O(Np), activity and parameter storage O(N) and O(Np). Fitting M
examples per factor with dense least squares costs O(N*(M*p^2+p^3)); extracting
averages costs O(MN). A T-step query with J backtracking trials costs O(TJNp).
Direct reconstruction costs O(Np). No grouping search, tuning, optimizer moments,
replay or eigendecomposition. This synthetic family and fixed grouping determine
what is learnable; asymptotics do not establish accuracy at other widths.

Charge feature construction, 28 fits, checks, all five arms, comparator and reporting.
Record whole and phase wall/CPU times and peak process RSS, source/contract hashes,
revision/dirty status, command, environment/BLAS/threads, counts and array bytes.
Time direct and frozen quadratic 64/2048-step inference on eight queries with one
warmup and three measured batches; report median/min/max, including allocation
and gradient/backtracking but excluding scoring. Account for those extra runs.
Timing JSON is separate from deterministic scientific JSON. No speed claim from
this small case. Manual development time and Python allocator peak are unavailable.
Run once: .venv/bin/python -m experiments.standing_pattern_hierarchy.
