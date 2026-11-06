# Mathematical analysis of the relearning objective

2026-11-04. Derived analysis and small numerical checks, not a contract or a
trained remedy. Requested after the stage-A screen. No checkpoint was loaded
and no new continuation was run. The numerical checks validate equations, not
their efficacy in the full model.

## Conclusion and concrete recommendation

Do not launch a grid of partial candidate gradients or EMA teachers yet.
The strongest analytically defensible next candidate is **the existing
distance-based prediction objective with a local constraint on representation
movement**. Keep candidate learning, but remove the component of the proposed
update that changes collection spread or increases mismatch on the existing
paired collection/taps context examples. This is constrained optimization,
not a claim that another pointwise distance solves interference.

The immediate inexpensive check is a one-step feasibility audit on the captured
states: compute the correction below for actual Adam proposals, measure the
retained prediction descent and actual geometry changes, and reject the proposal
if essentially no useful update survives. Only then is one continuation warranted.
This does not authorize a run. It does not require replay, sleep, count labels
in memory, or training the reverse comparator.

This recommendation changes the optimization rule as well as specifying a
constrained objective. It cannot honestly be called a scalar loss replacement.

## 1. Exact gradients of the current loss [derived]

For prediction p, candidates c_j, true index y and fixed positive beta:

    d_j = ||p-c_j||²
    pi_j = softmax(-beta d)_j
    L = beta d_y + log sum_j exp(-beta d_j)

The gradients treating output vectors as independent variables are:

    dL/dc_j = 2 beta (1[j=y] - pi_j) (c_j-p)
    dL/dp   = 2 beta (sum_j pi_j c_j - c_y).

Thus a mistaken prediction attracts its target and repels other candidates.
These statements do not yet specify the change in network outputs: that depends
on shared Jacobians and the optimizer.

For jointly scaling p and every c_j by alpha, at alpha=1:

    dL/d log(alpha) = 2 beta (d_y - sum_j pi_j d_j).

Contraction helps when the true distance exceeds the probability-weighted
distance, not on every row. Correct, confident rows often have the opposite
sign. Multiplying L by any positive constant preserves this directional sign.
The weight-0.1 failure therefore does not rule out all magnitude effects, but
it is unsurprising that it retains the contraction incentive.

The logit expansion is:

    -beta ||p-c_j||² = 2 beta p·c_j - beta ||c_j||² - beta ||p||².

The last term cancels in softmax. Candidate norms are score biases. Replacing
the score with an unnormalized dot product removes that bias but changes the
classification geometry and creates a norm-growth route; it is not an
equivalent repair.

## 2. Why the obvious alternatives lack the needed guarantee [derived]

**Individual cosine normalization.** For positive scalar codes c_k=k v, all
normalized codes equal v/||v||. A valid magnitude-coded number line becomes
indistinguishable. The actual codes need not be exactly of this form, but this
counterexample rules out blindly enforcing per-item norm invariance while
claiming to preserve ordinal/compositional information. A learned code system
could reorganize from scratch; that is a different empirical claim.

**Candidate-spread normalization.** Joint scaling of p and C leaves
||p-c_j||² / spread(C) unchanged, ignoring epsilon. But the actual predictor is
a biased ReLU MLP (GenericSequenceNet.P). In general P(alpha x) != alpha P(x).
Thus score invariance does not establish network-update invariance. With
L(C,p,s(C)), the candidate gradient additionally contains

    (partial L / partial s) grad_C s.

This permits changes in spread to influence the objective. Expansion in stage A
is measured, but it is not proof that changing the denominator alone increased
confidence. With fixed numerator and temperature, a larger denominator softens
logits. Also, setting s0=mean(s) does not make s0/s average to one, or preserve
the original first-batch logits exactly.

**Partial candidate gradient.** For positive alpha, replacing candidate
gradient G by alpha G retains its direction. It trades its size against other
paths and Adam state; it does not remove the contraction component. There is
no analytic reason to expect one universal intermediate coefficient to preserve
geometry and composition simultaneously.

**EMA teacher.** This still has zero direct gradient into detached candidates.
Targets move because teacher parameters average student parameters. It is not
a partial candidate gradient. Lagged targets can also have a different latent
coordinate system from the current predictor. It is a separate model of target
dynamics, not the next point on an alpha curve.

## 3. Why output-level fixes are not enough [derived]

Let theta be shared parameters, C(theta) the candidate codes, and J_C their
Jacobian. With candidate loss gradient G_C, plain gradient descent gives the
candidate-path contribution

    delta C = -eta J_C J_C^T G_C + O(eta²).

Projecting G_C perpendicular to C in code space does not generally make
C^T delta C zero: J_C J_C^T need not preserve that perpendicularity. Adam adds
coordinate-dependent preconditioning and momentum. For the same reason,
projecting a raw gradient and then running Adam does not guarantee an allowed
actual step. Constrain or check the proposed parameter displacement itself.

Nor can a sequence loss alone guarantee preservation of an arbitrary separate
reader g. Reader parameters can change the reverse score while leaving the
sequence objective and code geometry unchanged. The present goal can therefore
be a targeted protection of the identified mechanism, not a theorem of complete
retention or learned symmetry.

## 4. A specified local constraint [derived proposal, untested in the core]

Use only currently available training context, with its existing paired
collection/taps experiences. No evaluation examples, new reverse-direction
labels, or retained memory anchors are required. For their current codes c_i,t_i:

    S(theta) = mean_i ||c_i - mean(c)||²
    E(theta) = mean_i ||c_i-t_i||².

At the pre-update state, require first-order preservation of log S and no
increase of E. Nonzero S is required; declare a numerical floor and stop the
analysis if the context is degenerate. These aggregates are not complete
geometry: they allow some distortions, cancellations and reader changes.

Let delta0 be Adam's proposed displacement from all existing losses. Compute

    a = grad_theta log S
    b = grad_theta E.

Choose the smallest Euclidean correction:

    minimize_delta  1/2 ||delta-delta0||²
    subject to     a·delta = 0, b·delta <= 0.

For a nonzero a:

    delta1 = delta0 - a (a·delta0)/(a·a)
    n = b - a (a·b)/(a·a).

If b·delta1 <= 0, use delta1. Otherwise, for nonzero n:

    delta = delta1 - n (b·delta1)/(n·n).

If n vanishes exactly, the second condition is redundant after the first.
Numerically near-dependent constraints need an explicit tolerance or a rank-
revealing two-row solve. This is the nearest feasible proposal under the stated
metric; it is not necessarily the best task-descent direction. Check
grad(L_sequence)·delta, acquisition gradients, and retained update norm.

Only two parameter-space directions are constrained when they are independent.
Both candidate and predictor learning otherwise remain possible. In contrast,
candidate stop-gradient removes every direct candidate-side learning direction.
Few constrained directions do not guarantee little loss of useful descent.

Parameters unused by S and E have zero entries in a,b and are not corrected
directly. In particular this does not enforce the comparator's symmetry.
All participating trainable encoders/projections must be included, not just
the final image matrix. Do not silently substitute code-space projection.

**Finite steps and state.** These conditions hold only to first order. A
finite-step implementation must reevaluate S,E on the same context and report
deviations. A trust region/backtracking policy and tolerance must be declared
before a training test. Zero tolerance can force a zero step because second-
order mismatch can be positive even for a valid tangent direction. Local
tolerances can accumulate over time; held-out geometry and memory-present
performance remain necessary. Specify how Adam's moments are maintained when
its proposed step is corrected. Initially keep its standard moment update and
correct only the parameter displacement, exposing the size of that correction.

This is a diagnostic candidate for the identified failure, not a guaranteed
remedy or a claim that all code scale changes are undesirable forever.

## 5. What to do next

1. On captured early relearning batches, audit actual proposals using the two
   constraints. Reuse data and seeds. Check rank, finite-step feasibility and
   whether useful sequence/acquisition descent survives. No training grid.
2. If feasible, specify one constrained-update continuation, compared with the
   unchanged and candidate-stop endpoints. Keep the original prediction loss,
   losses for relations and original data. Do not train reverse matching.
3. Evaluate transient and deadline behavior of both interpretation and
   composition. If geometry is protected but reverse scores still fail, the
   existing evidence already points to a separate reader-calibration issue;
   do not call that a failure to prevent contraction.

An alternative penalty on S,E would require weights and, if centered on the
current pre-step values, has zero gradient at the first step. It does not provide
the immediate step constraint above. This is why the proposal uses constrained
optimization rather than another weighted scalar penalty.

## Prior work and evidence

Search date 2026-11-04. Queries: "site.proceedings.mlr.press orthogonal gradient
descent continual learning Farajtabar 2020" and "site.arxiv.org 1704.06369
NormFace normalized features scale".

- **[abstract; analogy]** [NormFace](https://arxiv.org/abs/1704.06369) studies
  normalized face features and separate scale. It does not establish that
  normalizing our magnitude-bearing codes preserves their information.
- **[read: abstract and section 3 excerpt; analogy]**
  [Orthogonal Gradient Descent](https://proceedings.mlr.press/v108/farajtabar20a.html)
  projects updates to preserve earlier outputs locally. The proposal here uses
  two current-context geometry statistics and corrects an Adam displacement.
  It does not implement OGD's retained historical output-gradient subspace,
  and inherits no general retention guarantee from that paper.
- **[derived]** All equations and counterexamples above are local calculations
  for the stated objective. No novelty claim is made. The projection is an
  elementary equality/half-space constrained least-squares problem.

## Numerical verification and computation

`experiments/relearning_loss_analysis.py` checks the candidate gradient and
joint-scale derivative in float64: maximum errors 1.11e-16 and 3.55e-15.
It demonstrates loss of all nine distinctions under normalization of a positive
scalar number line, and nonhomogeneity of a biased ReLU predictor.

A two-modality linear toy with shared image parameters checks the constrained
step. The harmful proposal's log-spread and mismatch derivatives are -1.386
and +0.314. After correction they are zero to rounding, with nonzero retained
update norm 1.821. Finite-step geometry deviations scale quadratically with
step size, illustrating why first-order protection is not exact protection.

Scientific JSON: `reports/relearning-loss-analysis-v1.json`.
Separate runtime: `reports/relearning-loss-analysis-v1-runtime.json`.
Whole-check computation excluding interpreter/import startup: about 0.004 s
wall/CPU; process peak RSS about 228 MiB including PyTorch. CPU, one thread,
float64, one deterministic seed, zero fits and zero checkpoint reads. These
are arithmetic checks, not deployment timings or new scientific accuracy results.

For K candidates in d dimensions, the loss and output gradients cost O(Kd)
time and storage. Computing statistics on n paired contexts costs O(nd), plus
their network forward/backward costs. For P participating parameters and m=2
constraints, the correction costs O(mP + m³) time and O(mP) gradient storage,
in addition to Adam. It needs two additional scalar backward passes and may
need extra forwards for finite-step checks. The teacher alternative requires
an additional parameter copy and forward path; none is used here. Any actual
feasibility audit must separately report these costs, all candidates and step
trials, model revision, source hashes and training/selection state.
