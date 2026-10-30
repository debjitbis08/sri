# Mathematical conditions for transfer and retention

**Scope update:** the memory proposal now explicitly decays dense reader
parameters (revision `217701a`). The results below retain their stated scope;
FIFO results are no longer about the current proposal. See
[decaying-reader theory](decaying-readers-theory.md) for selective parameter
decay, handoff timescales and the shared-layer rescaling loophole.

2026-10-30. A mathematical follow-up to [the empirical review](fan-in-memory-analysis.md).
These are conditional results, counterexamples and proposed diagnostics, not a
theory fitted to the observed accuracies or a new experiment contract.

The useful distinction is between three claims: forcing readers through a scalar;
identifying that scalar across input kinds; preserving the parameters that map
old inputs to it. None implies the next without further assumptions.

## 1. A scalar bottleneck alone does not identify transferable relations

**Proposition (non-identifiability).** On a finite training set, if scalar encoders
and relation readers have unrestricted interpolation capacity, restricting every
reader to one shared scalar per input does not determine its outputs on untaught
input-pair/relation combinations.

**Proof.** Assign a distinct real number to each input identity. Every taught
pair/relation becomes a distinct point in the reader's input space. A function
can interpolate all its prescribed outputs and independently assign any value
at an untaught point. For example, put a small disjoint bump around each finite
point. The sequence predictor can likewise interpolate its finite observed
histories. Cross-entropy targets can be approached arbitrarily closely by making
the prescribed logits large. One real coordinate suffices for these distinct
identities. QED.

This is a counterexample to a bottleneck-only guarantee, not a claim that the
particular 64-unit readers realize every such lookup. Width, smoothness, noise,
weight constraints and optimization can rule out some interpolants. It explains
why reader sparsity is an inductive bias whose effect must be measured.

**What to use:** seek an identifiable relational structure and a quantitative
smoothness/margin condition, rather than using low rank as a certificate.

## 2. A grounded relation graph gives an explicit transfer guarantee

### Assumptions and model

Let V contain N items (initially idealized count identities, not raw images).
Associate scalar q_i with each. A taught edge e=(i,j) specifies

    q_j - q_i ≈ d_e.

Same-count edges have d_e=0; a unit succession edge has d_e=1. Let B have rows
b_e^T=e_j^T-e_i^T, W=diag(w_e)>0, and C select grounded reference items with
positive anchor weights Λ. Consider

    E(q) = 1/2 ||W^(1/2)(Bq-d)||²
         + 1/2 ||Λ^(1/2)(Cq-a)||²,
    A = B^T W B + C^T Λ C.

This is weighted least squares on a grounded graph. It is **not** the present
generic neural loss: unit differences, scalar equality, and anchor values impose
stronger semantics than unconstrained `g` and `P`. It is a tractable reference
model for what successful relational alignment would mean. Numerical identities
of withheld forms are evaluator information, never extra permitted training labels.

### Proposition 2a: exact identifiability

A is positive definite iff every connected component has an anchored node.
If there exists q* with Bq*=d and Cq*=a, it is then the unique minimizer.

**Proof.** For any vector u,

    u^T A u = Σ_e w_e(u_j-u_i)² + Σ_anchor λ_i u_i².

This vanishes precisely when u is constant on each component and zero on anchored
components. An unanchored component admits an arbitrary constant shift. Otherwise
the nullspace is trivial. Subtracting the consistent q* gives
E(q)=1/2(q-q*)^T A(q-q*), proving uniqueness. QED.

**Consequence:** unpaired forms can acquire an exact quantity through succession
paths and a few grounded links. Direct pairing of every count is not necessary
under these assumptions. If only order inequalities replace unit differences,
exact cardinal spacing is generally not identified; this theorem no longer applies.

### Proposition 2b: a sharp error certificate

For any query vector v and E(q)≤ε,

    |v^T(q-q*)| ≤ sqrt(2ε R_v),   R_v = v^T A^(-1) v.             (1)

**Proof.** Apply Cauchy–Schwarz to A^(-1/2)v and A^(1/2)(q-q*).
Equality is attained by q-q* proportional to A^(-1)v. QED.

For v=e_i-e_j, R_v is the effective resistance between two nodes when anchor
edges are interpreted as conductances to a grounded node. It measures how weakly
the available constraints determine their relative position. For v=e_i it
measures absolute uncertainty relative to ground. A coarser global bound is
||q-q*||≤sqrt(2ε/λ_min(A)).

To certify a positive quantity difference Δ, it suffices that
sqrt(2ε R_v)<Δ. Low training residual alone is inadequate if R_v is large.
These bounds apply to represented nodes; unseen renderings require an additional
encoder generalization/noise bound.

### What happens when a script's bridges fade?

Let a script contain n nodes, keep its internal succession constraints, and
give its grounding edges total weight β. Suppose it has no direct anchor.
Use the unit vector u equal to 1/sqrt(n) on that script and zero elsewhere:

    λ_min(A) ≤ u^T A u = β/n.                                  (2)

All internal edges cancel; only grounding bridges contribute. As β tends to
zero, the absolute alignment certificate degenerates. At β=0 the script can
translate arbitrarily while retaining its internal order exactly.

**What to use:** track cross-script calibration separately from within-script
order, and preserve a few grounding bridges. A fading store can leave a perfectly
ordered but incorrectly positioned script. Equation (2) concerns the constraints
still enforced, not a claim that stored weights instantly forget at eviction.

## 3. Functional bypass, not edge count, determines transfer robustness

Suppose the actual score decomposes on the relevant evaluation domain as

    g(a,b,r)=G(q_a,q_b,r)+h(a,b,r),  |h|≤δ,

where G is L-Lipschitz in its two scalars under the L1 norm. Let ideal scalar
codes produce a correct-vs-incorrect candidate score gap at least γ, and let
each actual scalar differ from its ideal by at most η.

**Proposition.** The actual gap is positive whenever

    4 L η + 2 δ < γ.                                           (3)

**Proof.** Each candidate score moves by at most 2Lη+δ. The difference between
two scores moves by at most twice that amount. QED.

Equation (1) supplies η≤max_i sqrt(2ε(A^(-1))_ii) in the graph model; observation
noise can be added to η. Margin γ and the Lipschitz domain must actually hold;
the result does not assume a globally monotonic generic reader.

For a fixed75 network, δ can be large even with few residual units: arbitrary
outgoing weights can amplify those units. Conversely a small nonzero residual
path may be harmless. No bound on δ follows from the percentage of masked edges.

**What to use:** ablate nonmodal units and measure score perturbations relative
to margins. This directly tests whether the apparent scalar reader actually
controls decisions. Penalizing or limiting functional bypass is a more direct
intervention than picking another arbitrary concentration percentage.

## 4. A scale-invariant alternative to raw gradient growth

For a local candidate weight increment w_j, suppose a quadratic approximation is

    L(w_j)=L(0)+g_j w_j+1/2 h_j w_j²,  h_j>0.

**Proposition.** The best unrestricted one-coordinate update is w_j*=-g_j/h_j;
its predicted loss decrease is

    D_j = g_j²/(2h_j).                                         (4)

For fixed-feature squared regression, g_j=-x_j^T r and h_j=||x_j||²,
so (4) is exact for that coordinate with the other coefficients fixed.
Under x_j→s_j x_j, g_j→s_j g_j and h_j→s_j²h_j. Therefore D_j is invariant,
whereas RigL's raw |g_j| is not. The result follows by completing the square.

In a fixed ReLU activation region with a squared output loss, a Gauss–Newton
diagonal uses squared output Jacobians. General cross-entropy networks require
a local curvature approximation. Dividing |g_j| by input RMS is a cheaper proxy,
not generally the exact curvature correction. Fixed numerical damping also
breaks exact invariance unless transformed consistently.

**What to use:** compare raw selection with normalized selection before claiming
that its chosen coordinate is semantically privileged. At k=1 the current code
reconsiders every edge, so growth-score bias is particularly consequential.
This is a local optimization result, not a proof that normalized growth transfers
better or that all layer rescalings preserve the full sequence-prediction loss.

## 5. Which memories preserve alignment most efficiently?

Start from positive definite A. Adding one available constraint b with weight w
changes it to A'=A+wbb^T. Sherman–Morrison gives

    A'^(-1)=A^(-1) - w A^(-1)bb^T A^(-1)/(1+w b^T A^(-1)b).

**Proposition.** The query-resistance reduction is exactly

    R_v - R'_v = w(v^T A^(-1)b)²/(1+w b^T A^(-1)b) ≥ 0.         (5)

Summing over query vectors v gives a score for choosing a constraint under a
memory budget. For all coordinate queries, the numerator is w||A^(-1)b||²,
which is the one-step A-optimal-design reduction in trace(A^(-1)).

**Interpretation:** retaining an old bridge can matter more than retaining a
recent redundant within-script observation. The score measures contribution to
identifiability, whereas age alone does not. This is a standard optimal-design
criterion applied to the reference model, not a novel replay algorithm.

The criterion needs a current model of constraints; it is not directly a score
for raw images. Evaluate only permitted stored edges. In particular, do not add
withheld label edges because an evaluator knows their true counts. If A is
singular, first connect ungrounded components; adding a ridge is a prior, not
evidence of grounding.

### Statistical interpretation and a limit on the forgetting hypothesis

Suppose independent observed edge/anchor noises have variance σ² divided by
their weights. The unbiased weighted least-squares estimate has covariance
σ²A^(-1). Consequently, adding correct independent observations cannot increase
any query's variance, by (5). Removing observations cannot improve it in this
model. Finite residual certification in (1) and this stochastic statement are
different claims: changing observations also changes the fitted residual.

Replaying the same noisy sample is not a new independent observation. Replay
can restore optimization pressure on a constraint; it must not be credited with
the statistical information of another independent measurement.

For a simple example of when forgetting *can* help, suppose n current independent
observations have mean μ and variance σ², while m old independent observations
have mean μ+b and the same variance. The mean of all n+m has target MSE

    σ²/(n+m) + (m b/(n+m))².

Using only current observations has MSE σ²/n. Dropping the old data improves MSE
iff

    b² > σ²(n+m)/(nm).                                        (6)

Thus a benefit from fading needs sufficient bias/drift, model mismatch,
optimization interference, or another mechanism beyond "less memory". For b=0,
retaining valid observations is better. This toy threshold is not an estimate
of the network's actual interference.

## 6. Eviction does not imply neural forgetting

Take a fixed quadratic context loss

    L_c(θ)=1/2(θ-θ_c)^T H(θ-θ_c),  H positive semidefinite.

Under continuous gradient flow, starting at θ_0,

    θ(t)=θ_c + exp(-Ht)(θ_0-θ_c).                              (7)

**Proof.** Diagonalize H, solve each scalar ODE, and rotate back. In an eigenmode
with eigenvalue λ and old-new displacement Δ,

    θ_i(t)-θ_i(0)=(1-exp(-λt)) Δ_i.

Three cases follow exactly:

- λ=0: no change, even if the memory is no longer replayed.
- Δ_i=0: the old value is compatible with current training and is preserved.
- λ>0 and Δ_i≠0: drift toward the context optimum; faster for larger λ.

For a linear old prediction a^Tθ, the exact drift is
a^T(I-exp(-Ht))(θ_c-θ_0). It depends on parameter overlap and disagreement,
not simply the rank of the output code. Weight decay adds a force toward zero
and changes the nullspace conclusion. Adam and changing nonlinear features do
not obey this fixed-H flow exactly.

For smooth nonlinear loss and a small ordinary gradient step, the related local
condition is ΔL_old≈-η grad(L_old)^T grad(L_new); at an old optimum the linear
term vanishes and second-order effects matter. This is why a low old gradient
alone is not a retention guarantee.

**What to use:** distinguish passive storage eviction, continued context training
and explicit weight decay. Measure old-output Jacobians against current update
directions, or cheaply measure per-step old-output drift. The memory proposal's
delay tests active interference, not a necessary consequence of fading.

### A consolidation deadline under an explicitly solvable model

For one correctly specified scalar mode with curvature λ and exposure weight
p(t), gradient flow gives

    e(T)=e(0) exp(-ηλ ∫_0^T p(t)dt).

Reaching |e(T)|≤ε before eviction requires

    ηλ ∫_0^T p(t)dt ≥ log(|e(0)|/ε).                          (8)

For the full-store age distribution in the earlier review, four draws per update
and one fresh insertion give four expected lifetime replays per inserted episode,
independent of half-life. With fixed curvature and no interference, changing the
half-life redistributes a fixed dose; it does not create additional consolidation.
Equation (8) is exact for deterministic exposure-weighted flow. Substituting an
expected stochastic replay count describes a mean-exposure approximation, not
the expected error of a random SGD trajectory. Time-varying curvature, conflicts,
multiple episodes sharing a mode and finite-step effects can make timing matter.

## 7. Arbitrary bindings have an irreducible memory requirement

Suppose nine distinguishable glyphs are assigned bijectively to nine counts by
an arbitrary permutation. A system that must recover every assignment exactly
needs at least 9! distinguishable retained states: at least ceil(log2(9!))=19
bits in a fixed-length binary representation (entropy log2(9!)≈18.47 bits for
a uniform permutation). This ignores image recognition and therefore is only
a lower bound.

**Proof.** If two different assignments produce the same retained state, the
system must give the same answers for both and be wrong on at least one. Pigeonhole
principle. A number concept cannot remove this dictionary information.
Weights, sufficient statistics and replay all count as retained state. Infinite
precision reals evade finite-bit bounds unless precision is accounted for.

If a full ordered chain of those glyphs and its orientation are retained, that
chain itself carries assignment information; it is not free recovery from
structure. Conversely, once the binding is known, the relational rule can be
shared across scripts. The useful compression is sharing the rule while keeping
the necessary bindings, not erasing all script-specific information.

## Concrete uses, ordered by how directly the theory supports them

1. **Measure functional reliance:** compare nonmodal ablation effects with score
   margins (3), rather than mask percentages.
2. **Separate scale from signal in growth:** inspect (4), using a local curvature
   estimate or an explicitly labelled RMS proxy. Run a controlled comparison
   only after specifying cost and permitted tuning.
3. **Keep a small set of grounding constraints:** use (5) as a reference selector,
   or the cheaper baseline of preserving at least one bridge per connected
   script plus its internal relations. One bridge suffices only for the exact
   increment model; noisy data benefit from redundancy.
4. **Make the delay diagnostic:** report mapping drift separately from relation
   skill; compare it with the overlap/disagreement prediction in (7).
5. **Require a reason for fading to help:** test bias/interference, rather than
   treating eviction itself as a compression objective. Bound (6) gives a simple
   example of the competing variance and bias terms.

Nothing here predicts the numerical .75 versus .52 English comparison result.
The theory identifies missing quantities—constraint strength, reader sensitivity,
margin, coordinate scale and parameter interference—that can explain or refute
candidate mechanisms with targeted measurements.

## Prior work, evidence and scope

All propositions above are **[derived]** under their displayed assumptions.
These are applications of established mathematics, not novelty claims.
Sources searched/opened 2026-10-30:

- Query: "Zhu Ghahramani Lafferty 2003 semi supervised learning Gaussian fields
  harmonic functions pdf". [Zhu et al. 2003](https://mlg.eng.cam.ac.uk/pub/pdf/ZhuGhaLaf03a.pdf)
  **[read: abstract and introduction]** places graph learning in Gaussian-field,
  electrical-network and spectral terms. Our nonzero edge increments and
  quantity anchors are a specified reference model, not their classification
  experiment or a characterization of the neural pilot.
- Query: "optimal experimental design rank one inverse Sherman Morrison A optimal
  design graph effective resistance". [Lu and Pong](https://arxiv.org/abs/1009.1909)
  **[abstract]** studies established optimal-design criteria. Equation (5) is
  derived here directly; the paper does not establish this replay intervention.
- Query: "Lopez Paz Ranzato Gradient Episodic Memory 2017 paper".
  [GEM](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf)
  **[abstract]** reports a continual-learning method to reduce forgetting while
  allowing transfer. The local gradient-interference connection is contextual;
  (7) is a separate exact quadratic derivation, not a GEM performance claim.

Cheaper established references are weighted least squares, graph grounding,
one-coordinate quadratic minimization, and uniform/coverage-preserving replay.
No cited source implies recency eviction produces abstractions in this network.

## Numerical verification and computation contract

`experiments/fan_in_memory_theory.py` verifies the identities on a declared
27-node fixture (collections, taps and forms, nine counts each), with unit
succession edges, all collection–tap matches, seven collection–form matches
(counts 4 and 6 withheld), and one anchor. It varies only the form-bridge weight
over {1, .1, .01, 0}; these are illustrative calculations, not tuned experiments.

Checks: exact grounding; loss of identifiability at zero bridge weight; the
Rayleigh bound (2); equality construction for (1); exact rank-one updates (5)
for two allowed candidate edges; quadratic growth and scale invariance (4);
the bias threshold (6); nullspace preservation and ODE identity (7); and the
deadline identity (8). All use float64, fixed explicit arrays, no sampled data,
no neural fits and no hyperparameter selection. Illustrative least-squares
solves and candidate scores are counted in the receipt.

For N nodes and E edges, dense construction here costs O(EN²), storage O(EN+N²);
full symmetric eigendecomposition and dense solves/inversion cost O(N³) each,
O(N²) workspace. Scoring J candidate edges with an existing inverse and one query
costs O(JN²), or O(JN) for sparse incidence edges with direct column access;
this implementation uses the dense form. A rank-one inverse update costs O(N²).
The local coordinate example costs O(Bd) time/storage. Full neural curvature,
candidate selection and retention costs are not benchmarked or claimed cheap.

Deterministic calculations are saved to `reports/fan-in-memory-theory.json`;
whole-process wall/CPU/RSS, library/backend environment and provenance are saved
separately to `reports/fan-in-memory-theory-runtime.json`. No training or old
artifacts are changed. Numerical identities check implementation, while the
proofs establish the mathematical claims under their assumptions.
