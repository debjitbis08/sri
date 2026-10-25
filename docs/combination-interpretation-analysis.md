# Combination, defaults, and interpretation between layers

Status: analytical preflight, 2026-10-25. No operator family selected for SRI;
no capacity fitting, learning, or transfer experiment. The finite check below
illustrates proofs, not success on the frozen 72-scene task. Existing artifacts
remain unchanged.

## 1. The problem we are solving

The proposed structure is two outer units participating with one inner unit:

```
a1 = C_outer(u1, s1)
a2 = C_outer(u2, s2)
h  = C_inner(M(a1, a2), si).
```

Here C combines incoming activity with stored preference, M aggregates outer
contributions, and the symbols are unknown operations, not assumed arithmetic.
They may have different types at different layers. The neutral incoming message
is written 0, with C(0,s)=s. It is not necessarily the numerical zero of a unit's
state space. An observed zero is evidence, distinguished by an observation mask.

An inner state may be identical for distinct outer patterns. The outward
operation interprets that state using outer context and stored preferences;
it need not reconstruct the original pattern. No exact residual memory is
assumed. Persistent structure is fixed during inference; transient activity is
not silently stored as learned knowledge after a query.

This follows the separation of admissibility, realization, and learning in
[interaction semantics](interaction-semantics.md). The
[coarse-interior analysis](coarse-interior-analysis.md) already proves that a
summary can omit detail without erasing it at the surface. The new question is
how to select an outer interpretation when that detail is missing. Fewer inner
response modes do not establish semantic abstraction, consistent with the
[foundations](foundations.md).

The [Stage B negative](relational-capacity-stage-b-v1-results.md) motivates
requiring an explicit nonlinear response example. It does not select the new
operators or prove that the previous operators are uniformly incapable.

## 2. Define a compatibility relation before an inverse

Let X be the outer configuration space and H the inner state space. A candidate
specifies R_theta(x,h), or a graded compatibility cost K_theta(x,h). For a
deterministic inward descriptor q:X->H, the compatible outer interpretations are

```
F_h = {x : q(x)=h}.
```

F_h is the fiber of h. Decomposition means selecting within this fiber using
evidence and stored preference. One candidate semantics is

```
D(h,e;theta) = argmin_{x in F_h and consistent with e} P_theta(x).
```

This is a set until uniqueness or an explicit tie rule is established. An empty
set means incompatible evidence, not permission to overwrite observations.
Joint inference may instead revise h. Soft evidence requires a declared cost
and strength; this note uses hard evidence for the finite illustration.

**No-inverse proposition.** If q(x)=q(y) with x!=y, no function D(h) can recover
both x and y. Proof: D(q(x)) is a single value. Context or retained detail may
distinguish the cases, but that is additional information. A right inverse
q(D(h))=h can select one representative; it is not an inverse on all of X.

**Preference/symmetry proposition.** Suppose a transformation g preserves h,
the evidence, and stored preferences, but fixes no member of F_h. A deterministic
equivariant selector cannot choose a member: invariance of its inputs would
require D=gD. A unique choice must use asymmetry in the preference/context,
break equivariance explicitly, or relax uniqueness. Thus defaults carry actual
information and cannot be treated as a free neutral convention.

## 3. What the neutral law does and does not determine

C(0,s)=s leaves infinitely many possibilities: translation, a group action,
nonlinear relaxation toward a preference, and other operations. It does not
imply associativity, commutativity, invertibility, saturation, or convergence.
Require those only when the intended interaction needs them.

A useful candidate family is an action A of a message algebra on stored states:
A(0,s)=s. For example, a transformation message can act on a stored reference.
This is a candidate, not an assertion that all states form a group. If messages
compose, specify their order; order independence is an additional restriction.

Do not mix a raw inner descriptor with a deviation message. If h=q(x) is an
absolute descriptor and si is a stored reference, a message encoder B must obey

```
C_inner(B(q(x),si),si)=q(x).
```

Otherwise the stored preference can be counted twice. The act of expressing a
descriptor relative to a default is not itself evidence that the default biases
inference; a preference or response law must supply that behavior separately.

## 4. Three candidate families and their obligations

| Family | Construction | What it supplies | Missing obligation |
| --- | --- | --- | --- |
| Linear summary with defaults | Quadratic compatibility and fixed reference patterns | Exact solves, compression, stable unique solution when Hessian is positive definite | With a fixed mask, the response is affine; defaults do not remove the existing affine obstruction |
| Relational fibers with preference | Inner state labels a set of compatible outer patterns; evidence and preference select a member | Explicit contextual interpretation and possible nonlinear response | How relations/preferences are learned; scalable realization; frequency structure |
| Smooth state-dependent compatibility | A nonquadratic joint energy over outer and inner activity | Continuous nonlinear interactions and a common descent objective | A concrete energy with the desired relations, convergence conditions, and acceptable conditioning |

For the first family, E(z;e)=z^T H z/2-z^T(Be+b)+constant with fixed SPD H
has z*=H^-1(Be+b). A stored default changes b, not affinity. This is an
exclusion of that family for escaping an affine task floor, not an exclusion of
all smoothing or all defaults. Nonlinear output decoding would change the model.

For the third, gradient flow decreases energy wherever differentiable:
dE/dt=-||grad E||^2. This rules out a nonconstant periodic orbit, but by itself
does not prove boundedness, convergence to one point, or a correct interpretation.
Strong convexity and suitable smoothness give stronger guarantees; nonconvex
compatibility needs explicit local/global distinctions. A coordinate change
that makes a linear process look nonlinear is not enough: assess responses in
the declared observation/output coordinates.

## 5. A complete two-outer, one-inner interpretation example

This is a deliberately supplied binary relation, not a discovered SRI algebra.
It extends the parity illustration already present in interaction semantics by
giving defaults, evidence, selection, and a settling counterexample.

Let x1,x2,h be bits and q(x1,x2)=x1 XOR x2. The fibers are

```
F_0 = {00,11}       (same)
F_1 = {01,10}       (different).
```

Four outer configurations become two inner possibilities. Simultaneously
flipping both outer bits leaves the inner descriptor unchanged. Thus relative
structure is retained and absolute orientation is not. This is a quotient by
a specified symmetry, not conventional spatial low-pass smoothing.

A candidate neutral combination is C(u,s)=u XOR s, with u a change message.
It satisfies C(0,s)=s. Use stored outer preference (s1,s2)=(0,1) and inner
preference si=1. The inward change message is
B(q,si)=q XOR si, giving C(B(q,si),si)=q. The raw aggregator q and the message
encoder are distinct operations; no claim of a universal untyped operator follows.

For interpretation, use one common energy with hard outer observations:

```
E(x1,x2,h) = 4 [h != (x1 XOR x2)]
           + 2 [h != 1]
           + [x1 != 0] + [x2 != 1].
```

Brackets denote indicator costs. The weights are constructed analytically:
4>2 means compatibility beats any change in inner preference at fixed x;
2>1 means the preferred relation beats a single missing unit's outer default.
No coefficients are tuned on examples. The finite verifier enumerates all
eight states and all nine outer evidence patterns in {missing,0,1}^2.

| Observed x1 | Observed x2 | Selected (x1,x2,h) | Energy |
| --- | --- | --- | --- |
| missing | missing | (0,1,1) | 0 |
| 0 | missing | (0,1,1) | 0 |
| 1 | missing | (1,0,1) | 2 |
| missing | 0 | (1,0,1) | 2 |
| missing | 1 | (0,1,1) | 0 |
| 0 | 0 | (0,0,0) | 3 |
| 0 | 1 | (0,1,1) | 0 |
| 1 | 0 | (1,0,1) | 2 |
| 1 | 1 | (1,1,0) | 3 |

All optima are unique. With one observation the same inner state h=1 supports
different outer completions. With complete equal observations, evidence is
preserved and the inner interpretation changes to h=0. With no evidence,
the stored whole-state preference is recovered. Complete-data h is XOR, which
is not affine over real-valued input coordinates: h(00)=0, h(10)=h(01)=1
would force an affine h(11)=2, but the actual value is 0. Single-missing-bit
completion alone is 1-x and is affine; do not claim otherwise.

This demonstrates existence of a nonlinear relation descriptor with contextual
decomposition and defaults. It supplies the relation by hand, uses a ternary
interaction, and does not solve or approach the 72-scene task by itself.

## 6. Stable settling can still select the wrong interpretation

On a finite state space, sequentially accepting only strict energy decreases
terminates, since no state can be revisited. Retain the current value on ties.
This excludes cycles but only guarantees a local minimum under the allowed moves.

For the example, observe x1=1 and initialize the other units at their defaults:
(1,1,1), E=5. Updating h first decreases energy to (1,1,0), E=3.
Neither changing h alone (E=5) nor x2 alone (E=8) improves it. This is a strict
local minimum. The globally preferred compatible interpretation is (1,0,1),
E=2, requiring a joint change of x2 and h. Updating x2 first instead reaches
that optimum immediately. Update order matters even though both schedules stop.

Exhaustive joint minimization is the reference realization for this tiny example;
it is not a scalable local neural algorithm. A later candidate must either prove
its allowed updates reach the desired interpretation or explicitly report traps,
initialization dependence, and cost of escaping them. Reciprocal compatibility,
invertibility, and preference alone establish none of these properties.

## 7. Modes and higher dimensions: two notions must not be conflated

For fixed linear summaries q_l, decreasing retained rank describes fewer linear
modes. Exact hierarchy consistency is q_{l+1}=r_l composed with q_l; the induced
equivalence classes become coarser inward. For relational summaries the same
factorization defines a hierarchy of fibers, without requiring linear modes.
A relation descriptor t survives exactly iff it is constant on each fiber of q,
equivalently t=f composed with q for some f. This is a precise preservation test.

Conventional frequency requires a geometry and an operator such as a Laplacian.
The XOR illustration does not supply them: it distinguishes 00 from 01, while
identifying 00 with 11. Ordinary two-site averaging instead identifies 01 and
10 but distinguishes 00 and 11. Neither notion of simpler representation should
be silently substituted for the other. One option is a retained low-frequency
subspace after a learned relational encoding; both the encoding and its cost
would then be part of the model, not a free decoder.

For differentiable q, local detail directions satisfy J_q(x) delta=0. These
directions preserve the inner state to first order only; finite displacements
need not do so. A state-dependent kernel is a local interpretation-dependent
mode notion, not a globally ordered Fourier basis.

The finite candidate extends to d independent binary feature coordinates:
h_j=x1_j XOR x2_j. Each fiber has 2^d outer interpretations. It reduces 2d
outer bits to d inner bits, with context/defaults resolving missing detail.
Coordinate-independent energy factorizes and can be solved exactly in O(d).
Coupling feature coordinates generally destroys this factorization; naive
enumeration becomes O(d*2^(3d)). No learned geometry or cross-feature relation
capacity follows from the independent extension.

## 8. Declared finite verification and computation scope

The one-off checker is `experiments/combination_interpretation_analysis.py`.
Before its first execution this section declares the whole verification menu:
all 9 evidence patterns, all 8 states per pattern, the two specified update
orders above, all 4 neutral-combination cases and all 4 message round trips.
No randomness, optimization, parameter sweep, eigensolver, or numerical tolerance
is used. Integer arithmetic suffices. The direct equivalent is exhaustive joint
selection; the two local schedules are diagnostic counterexamples, not alternatives
selected by success rate. Tests are the exhaustive assertions inside the checker.

For q-valued units, explicit two-outer/one-inner enumeration evaluates q^3 states
per evidence pattern; the complete evidence menu has (q+1)^2 patterns. With O(1)
compatibility evaluation this costs O((q+1)^2 q^3) time. Streaming selection
needs O(1) candidate workspace excluding output; this small checker materializes
O(q^3) states and O((q+1)^2) results. A general compatibility table could require
O(q^3) persistent storage; the supplied XOR predicate has constant description.

There is no training, learned parameter state, optimizer, covariance, replay,
or retained raw dataset. Persistent hand-supplied preference is three bits;
three integer coefficients (4,2,1) specify costs, with the last shared by the
outer units. These logical counts are not Python object byte counts. Runtime
records the Python object footprint of this configuration, its JSON bytes, and
process peak RSS separately. No claim of compact machine implementation follows.

Deterministic results and nondeterministic runtime receipts are separate files.
The receipt records hashes, revision/dirty status, command, interpreter/platform,
CPU, process/thread scope, elapsed wall/CPU time, counts and RSS units. Whole-run
measurement covers checking and result serialization after imports, not interpreter
startup or the final runtime receipt write. It is not an inference benchmark.
No BLAS, accelerator, or worker pool is used. Selection work includes every
candidate, including rejected/infeasible states. Manual mathematical development
and source editing time are not measured; no end-to-end research-cost claim is made.

## 9. Decision after this preflight

Do not extend Stage B or choose a gate formula yet. The next mathematical target
is an operator family that jointly provides (a) a declared mode geometry,
(b) relational fibers and defaults, and (c) a tractable realization procedure
with characterized failure cases. The example meets (b), provides an exact tiny
reference for (c), and deliberately leaves (a) open. It also disproves the idea
that strict local descent alone suffices for the intended interpretation.

Before a new learning study, require a constructive relational example in its
actual output coordinates, a direct simpler comparator, a scaling derivation,
and a prospective computation-reporting contract. Learning the relation algebra
and preference from evidence is a separate obligation from supplying them.
