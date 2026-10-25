# Single-module masked capacity v1: task audit and bounded witness search

**Status:** prospective, 2026-10-25. Commit this contract before implementation
and finite execution. Freeze source/test hashes and a manifest before tests,
task audits or capacity fits. All abstraction-preflight-v1 results, including
the sign shortcut and affine obstruction, have been seen. This study uses a
new task and seeds. It is not confirmation of an abstraction or learning claim.

Follows [preflight v1](single-module-abstraction-preflight-v1-results.md) and
the [computation protocol](computation-reporting.md). Preserve prior sources,
contracts, scientific results and runtimes. No execution of the separate
plasticity/retention diagnostics is included here.

## 1. Questions and invariant architecture

1. Does a revised holdout exclude exact lookup under the module's sign
   symmetry and under supplied common rotation/scale transformations?
2. Can one allowed parameter setting complete all scenes through the masked
   module with matched gates, without a decoder, output gain or scene-specific
   parameters?

Retain the v1 four rings (32,24,16,8), four sectors, phase zero, 328 parameters,
Stiefel radial maps, unit terminal vectors, energy coefficients, positive gates
and two-pass inference. Use the explicit observation mask in both passes:
only outer sectors 0 and 1 have anchor 0.1, and the other two have no data
anchor. Lateral coupling remains 0.001. Gate formula and epsilon remain v1.
Missingness is observation metadata, independent of familiarity. No learned
mask, extra trainable connections, nonlinear decoder, bias or output rescaling.

The solver is float64 dense Cholesky, with a direct Schur reference. The
existing uniqueness proof applies. This is an interface revision already
analyzed in preflight v1, not a replacement of the module architecture.

## 2. Revised finite task

Let r in {1,2,3}, q in {1/3,1,3}, t in {0,...,7}, and

```
theta_j = 2*pi*(t+j*r)/8
a = sqrt(2/(1+q*q)); b=q*a
amplitudes = (a,b,a,b)
s_j = amplitude_j * (cos(theta_j), sin(theta_j), 0,0,0,0,0,0).
```

There are 72 scenes, ordered lexicographically by (r,q_index,t), with q_index
0,1,2. Reveal parts 0 and 1 and predict parts 2 and 3. Both visible and hidden
signals have squared norm 2, so hidden squared error divided by 2 remains the
loss and zero output scores 1. Relative angle determines r and norms determine
q; the rule oracle then determines the missing parts. Labels and clean targets
are privileged diagnostic information, not a proposed learner interface.

Hold out exactly `q_index == r-1`, for all eight t. Thus training has 48 scenes
and holdout 24. Each relation and each amplitude ratio occurs in training, but
three relation-ratio combinations do not. Entire sign, common-rotation and
common-positive-scale classes are withheld: these transformations preserve
both the relative angle and the amplitude ratio. This is a deliberate change
from holding out orientations in the old task. Report all factor marginals.
The ratio spacing is chosen analytically before execution: even a template
with the correct angular relation incurs at least 0.20 normalized loss when
only a common scale is allowed to bridge adjacent ratios. This avoids the
0.10 threshold coincidence that the closer grid {1/2,1,2} would have produced.

Use two fixed renderings: world 0 is identity; world 1 independently rotates
each part's eight coordinates using a Gaussian orthogonal matrix. These are
two capacity fixtures, not independent statistical evidence for generalization.
There is no texture/noise and no finite-sample confidence interval.

## 3. Task audits and controls

Fit and score the following before the capacity search. All numerical ties
within absolute 1e-12 use lexicographic scene order, then positive before
negative sign. Orientation ties use the smallest orientation index. Affine
fits use float64 SVD, cutoff 1e-12 relative to the largest singular value,
minimum-norm weights and an intercept.

- Zero and training-mean controls.
- First-part and two-part affine predictors fitted to the 48 training scenes.
- Ordinary lookup and signed lookup using training scenes only.
- A privileged common-rotation/scale lookup: decode into the known signal
  planes; for each training input fit one complex scalar c to its two visible
  complex coordinates by least squares, and apply that same scalar to both
  stored hidden coordinates. Choose by visible discrepancy only. This allows
  arbitrary common rotation and nonnegative scale. It gets renderer knowledge;
  it is a shortcut audit, not a fair learned comparator.
- A privileged factored-template reference: normalize visible parts separately,
  match a training angular template up to common rotation, rotate its normalized
  hidden directions, and restore hidden amplitudes from observed parts 0 and 1
  respectively. This supplies the true amplitude correspondence and renderer.
  It is expected to solve the task; its success demonstrates that the task is
  compatible with a simple factored relational representation, not that every
  template explanation has been excluded.
- The modular-rule oracle, evaluator-only.
- A two-part affine fit to all 72 scenes, explicitly privileged, plus a fit to
  the q=1 subset of 24 scenes to check the old exact floor (8−2√2)/21.

For common transformation lookup, the optimal c is
`sum(conj(z_train)*z_query)/sum(abs(z_train)^2)` over two visible parts.
All targets are excluded from template selection. For the factored reference,
compute c on normalized visible parts and use its phase only; a zero magnitude
uses phase 1 and records the degeneracy. Normalize stored hidden parts before
applying phase and the query's respective visible amplitudes.

Record per-scene losses, nearest-orientation predictions, per-rule/per-ratio
means, worst loss and joint correctness. Do not select a different task or
split after any audit outcome. The narrow split gate requires: identifiable
oracle targets; no exact signed/common-transform training match for any held-out
input (minimum squared discrepancy >1e-10); and neither signed nor common-
transform lookup meeting loss <=0.10 and joint orientation correctness on
every held-out scene. Failure limits the task's interpretation; the independent
capacity search still runs on the fixed 72 scenes to answer representability.
Success excludes only the named shortcuts. It does not exclude the explicitly
supplied factorization or establish abstraction by itself.

## 4. Bounded capacity search

Root `2026102503`. Renderer world 1 uses NumPy
`SeedSequence([root,1,0])`. For each world w in {0,1} and start s in {0,1},
initialize with `SeedSequence([root,w,1,s])`: Gaussian radial blocks in
ring-major then sector-major order, v1 positive-diagonal QR, followed by four
Gaussian unit terminal vectors. Identity and rotated fixtures have independent
initializations. Starts are identical between step-size candidates.

Each rendering receives two starts times Adam step sizes {0.01,0.05}, in
start-major, ascending-step-size order. Each candidate has at most 1,000
updates. Full batch includes all 72 clean input/target scenes, including the
nominal holdout. This is intentional privileged capacity fitting; its scores
are not learned transfer. No retained fitted parameters may initialize a
future learning experiment.

Use CPU PyTorch from the pinned repository environment, float64, one process,
one intra-op and inter-op thread, deterministic algorithms. Minimize mean
hidden loss through both masked solves and through the gate computation. This
full derivative is for the capacity optimizer only; it is not the frozen-route
learning rule of plasticity v1. Validate it independently by finite differences,
including off-manifold terminal vectors with Q^T Q curvature.

Project gradients to the Stiefel/unit-sphere tangents; clip their combined
Euclidean norm to at most 10. Adam uses betas (0.9,0.999), epsilon 1e-8 and no
weight decay or schedule. After each update apply v1 positive-diagonal QR and
unit normalization. Retain Adam moments in ambient coordinates without reset
or transport, explicitly treating this as a bounded heuristic constrained
optimizer. No line search, extra seeds, new step sizes or budget extension.

Evaluate checkpoints {0,100,250,500,1000}. Track all metrics and choose the
checkpoint with smallest worst-scene loss within a candidate; ties within
1e-12 prefer the earlier checkpoint. A witness requires one parameter setting
whose loss is <=0.10 on every one of the 72 scenes and whose two missing
orientations are correct on every scene, including distinct relations. A low
mean alone cannot pass.

At the first checkpoint apparently meeting the witness criterion, serialize
parameters to 12 significant digits and verify by independent NumPy/Schur
inference. If verified, stop that rendering's search and mark later candidates
unexecuted under this stopping rule; continue the other rendering. If none
passes, finish all four candidates. Retain and independently verify the best
checkpoint of every executed candidate. A verification mismatch is a correctness
failure, not permission to choose another checkpoint. Saved witness parameters
must satisfy constraints and the target after serialization.

Classify each rendering as `constructive_witness` or `unresolved_after_budget`.
The existing affine proof still excludes fixed-route solutions: the q=1 subset
alone has optimal average error 0.246265, incompatible with <=0.10 on every
scene. Failure of nonlinear optimization is not a representational impossibility
proof. A witness on one rendering supports only that fixture's feasibility.

## 5. Numerical gates, provenance and reproducibility

Before the main audit/search require tests for masking, source-equivalent
energy/Hessian expansion, full/Schur agreement, full gradients versus finite
differences, tangent projection, optimizer retraction, no hidden-input dependence,
the revised split and template selection without targets. Keep frozen dependencies
hash-identical. Check rank min |diag R|>1e-10, Stiefel/unit constraints within
1e-9, solve relative residual <=1e-10, Schur/state agreement within 1e-9, and
finite inputs, gradients and losses. No diagonal repair or reseeding.

Run two byte-identical task audits. Run the bounded optimization once; disclose
that the full optimizer trajectory is not rerun. Reload each selected saved
parameter setting and independently score it twice, requiring byte-identical
verification metrics. This checks deployment of serialized parameters, not
bitwise reproducibility of a second training execution.

Commit contract first, then source/tests/manifest before any execution. Record
all hashes, commands, dirty paths, test outcomes, every candidate and skipped
candidate, and all timing. Any correctness failure stops the study incomplete.
An implementation-only correction needs a disclosure, refreeze and rerun; no
silent replacement of failed work. Scientific JSON contains no wall times.

## 6. Computation and storage

N=72,d=32,M=80,J=L=4,P=328. Construction/rendering costs O(N J q_dim^2),
q_dim=8. Affine fitting costs O(n p min(n,p)+n p d), p<=17. Lookup audits cost
O(N n d), with n=48, doubled for signed lookup. Orbit fits have O(1) complex
work per candidate after O(N d q_dim) decoding. All privileged and ordinary
reference fits, audits and repeats count as study work, not free baselines.

One matched forward constructs O(LJ M^2) block curvature, one O(M^3) pilot
factor, N O(M^2) pilot solves, O(N LJ M^2) routed Hessian assembly and N
O(M^3) second factors/solves. Dense automatic differentiation has an upper
bound O(N M^3+N LJ M^2) in backward work, and retains O(N M^2+LJ M^2+P)
state. Record actual forward factors/solves and backward calls; do not label
library backward internals as measured adjoint solves without instrumentation.
QR adds O(sum_l J s_l s_(l+1)^2) per update, Adam O(P).

Maximum search is 2 renderings*2 starts*2 step sizes*1,000=8,000 updates,
576,000 scene exposures, plus 40 scheduled checkpoint forwards. These imply
at most 8,040 matched forward batches and 586,920 forward Cholesky factorizations
before independent verification. No tuning beyond those eight candidates.
Count early stops, parameter-state copies, retractions, gradients, validations
and all non-winning candidates. Report quality versus cumulative updates,
scene exposures and measured search cost, clearly labelled privileged fitting.

Parameters occupy 2,624 bytes. Adam retains two float64 moment arrays (5,248
bytes) plus step counters with actual dtype/bytes recorded. Inputs and hidden
targets occupy N*(32+16)*8=27,648 bytes per rendering; these are retained
capacity-training examples. A dense curvature cache is 819,200 bytes; each
N*80*80 tensor is 3,686,400 bytes. Report actual retained tensors, estimated
simultaneously live tensor bytes where feasible, and process peak RSS with
units/scope. Native autograd/solver workspace and FLOPs can be null with reasons.
Harness reports/checkpoints are separate from deployable learner parameters.

Runtime JSON records pinned versions, CPU/BLAS/thread configuration, hashes,
revision, command, dirty paths, wall/process CPU by phase and candidate, peak
RSS, selection and rejected-candidate costs, all counts and failures. Nested
totals are not added to their components. This is a one-size capacity search;
no inference-speed or scaling claim and no separate timing sweep are required.

## 7. Deliverables and decision

New source/runner/tests and `single-module-masked-capacity-v1` manifest,
scientific results, runtime, and human-readable results in docs/. Save full
selected parameter arrays and their independent verification. Preserve the
previous preflight's artifacts and its negative split finding.

Report observation semantics, narrow shortcut exclusion and masked routed
capacity separately. If both renderings have a witness and the narrow split
gate passes, the next step can be a separately frozen learning/transfer study
with fresh worlds, nuisance and strong comparators. A privileged factored
reference remains a disclosed alternative explanation. If capacity is unresolved,
report that limit and the completed search budget rather than launching an
unbounded experimental program or replacing the architecture silently.
