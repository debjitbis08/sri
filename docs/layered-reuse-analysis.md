# Reconnecting learned locality to the circular layered module

Status: analytical architecture study, 2026-10-24. A concrete candidate and
constructed witnesses, not an experiment contract, trained model, or new
confirmation result. Frozen historical artifacts remain unchanged.

Connects the [original disk](layered-module-v1-contract.md),
[layered diagnosis](layered-module-v2-results.md),
[selective-filter analysis](layered-module-design-analysis.md), and
[representation reuse results](representation-reuse-v1-results.md).

## Scope fixed before computation

Retain the four distinct rings of 32,24,16,8 scalar units and a scalar centre.
Analyze learned rectangular maps between adjacent rings, reciprocal quadratic
inference, a circular outer input/output interface, and different persistent
learning rates by depth. Derive rank limits, equilibrium/direct equivalence,
the effect of the centre, coordinate transport under drift, and the bias/noise
tradeoff of slow updates. Verify on explicit matrices only; no sampled worlds,
training, parameter search, or inspection of new confirmation outcomes.

Numerical witnesses use four eight-site exterior sectors, shrinking through
six, four and two coordinates per sector; local DCT bases and fixed coordinate
inclusions; k=4 signal directions and a separate texture direction per sector.
Check exterior tie strengths lambda={1,0.1}, radial strengths all one,
terminal nuisance penalty one, and drift angles {0,30,90} degrees. These are
constructed illustrations, not selected operating parameters. Compare the
original centre spring with an explicitly changed scalar-summary coupling.
Learning-rate checks use eta={0.5,0.1,0.01}, unit observation noise and drift
v=0.001, with 10,000 deterministic moment recurrences per rate.

## Computation contract

With total state dimension M=sum_l n_l+1=81, dense Hessian construction and
factorization/checking cost O(M³) time and O(M²) storage. Solving for the whole
boundary transfer needs O(M²d) after factorization, storing O(Md). Explicit
maps occupy sum_l n_l n_(l+1) in a dense reference, or only their declared
sector supports in a sparse implementation. A gradient application costs
O(M+sum_l nnz(D_l)+sum_l nnz(B_l)) using residual factors; t relaxation steps
cost t times that, with convergence tolerance and conditioning reported.
The witness uses dense direct solves, not a speed benchmark.

An independent coordinate-wise energy expansion checks the assembled Hessian.
EMA moment checks cost O(T) time and O(1) live state per rate. A separate
runtime JSON records measured phase wall/CPU time, RSS scope, versions,
threads, source hashes, command, deterministic work counters, and repeat
verification. No fitted models, candidate selection or tuning; those costs
are zero here, not inferred for a future learner. Unmeasured allocation and
FLOP quantities are null with reasons.

## Architecture after combining the two lines

The candidate is an **81-unit reciprocal module**, not a projector substituted
for the module. It keeps a 32-unit outer ring, distinct inner rings of 24,16,8
units, and one scalar centre. Immutable input h is an external drive at the
outer ring. The interpreted/reconstructed output is that ring's settled
state x_0, in the same 32 coordinates. This is a single-module reconstruction
interface; separate global Input/Output modules and satellite/core coupling
from the larger research program have not been implemented by this analysis.

[Architecture drawing](../reports/layered-reuse-architecture.svg).

```mermaid
flowchart LR
    H["External evidence h"] --> O["Outer ring: 32 units\noutput x₀"]
    O <-->|"learned D₀"| A["Ring: 24 units"]
    A <-->|"learned D₁"| B["Ring: 16 units"]
    B <-->|"learned D₂"| C["Inner ring: 8 units"]
    C <-->|"summary coupling"| S["Scalar centre"]
```

Ring geometry and individual units remain explicit. Each rectangular
D_l maps inner coordinates into the adjacent outer ring. Its transpose
appears in the reverse influence through differentiation of the same energy;
there are not independently trained inward and outward wires. Learned local
grouping restricts map supports to circular neighborhoods. In the witness,
four sectors have sizes 8→6→4→2. Sector identities align maps across unequal
widths; selecting an exterior phase does not mean applying the same integer
cyclic shift to rings of different sizes.

| Retained from the original module | Explicit change or remaining hypothesis |
| --- | --- |
| Circular exterior, distinct shrinking rings and scalar units | No claim that ring geometry alone creates selectivity |
| Reciprocal joint settling | Local reconstruction residuals replace ordinary positive pairwise radial springs |
| Exterior evidence and observable response | Input h remains immutable; output x_0 is a soft response, not clamped data |
| Locality of interactions | Block-supported maps allow signed weights and induced within-block couplings; not the original nearest-neighbor graph |
| Lateral interactions | PSD penalties must spare learned signal directions; generic smoothing is not imposed |
| Scalar centre | A summary constraint replaces the old all-units-to-one agreement spring in the recommended witness |
| Persistent learned structure | Faster outer/slower inner updates are a hypothesis to test, not a demonstrated property |

This is an explicit architectural revision. The old positive-spring disk is
not silently claimed to have acquired these properties. The 81 units are
transient activity state. Maps, support assignments, lateral penalties and
any retained sufficient statistics are separate persistent learned state.

## 1. Reciprocal energy, stability and preserved signal

Use x_l in R^(n_l), l=0,...,3, and centre c. A tractable starting family is

```
E(x,c;h) = lambda/2 ||x_0-h||²
         + sum_(l=0)^2 alpha_l/2 ||x_l-D_l x_(l+1)||²
         + sum_(l=0)^3 beta_l/2 ||B_l x_l||²
         + gamma/2 (c-aᵀx_3)².
```

Require lambda,alpha_l,gamma>0 and each D_l full column rank. The B_l terms
are optional PSD residual penalties, not arbitrary signed energies. For the
witness, alpha_l=gamma=1, B_0=B_1=B_2=0, and B_3 is the projector onto the
four nonsignal coordinates of the eight-dimensional inner ring. The four
signal coordinates have zero penalty. a is the unit-normalized all-ones vector.

For a homogeneous zero-energy state, the boundary anchor implies x_0=0;
full column rank successively implies x_1=x_2=x_3=0; the summary constraint
then implies c=0. Hence the Hessian H is positive definite. The equilibrium
is unique. With fixed learned structure and positive diagonal mobility M,
`dx/dt = -M grad(E)` converges because MH is similar to the positive-definite
matrix M^(1/2) H M^(1/2). This proves fixed-structure inference stability,
not convergence of learning or multistability, memory, or abstraction.

For an interior ring, adjacent residuals contribute

```
alpha_(l-1) D_(l-1)ᵀ(D_(l-1)x_l-x_(l-1))
  + alpha_l(x_l-D_l x_(l+1))
```

to its gradient. This explicitly couples it to both neighbors.

If signal coordinates satisfy U_l=D_l U_(l+1) and B_l U_l=0, then for a
pure input h=U_0 z, setting x_l=U_l z and c=aᵀU_3 z makes every residual
zero. By uniqueness this is the equilibrium: **no signal attenuation is
forced by depth**, despite the shrinking rings. This is a conditional
existence result; learning the compatible U_l and maps remains necessary.

### The centre cannot be a mandatory one-dimensional bottleneck

The original spring has form gamma/2 ||x_3-a c||² (with a normalized centre
coordinate). Eliminating c adds gamma/2 x_3ᵀ(I-aaᵀ)x_3. Only one inner
direction lies in its nullspace; it penalizes the rest. A k=4 signal space
cannot all pass through that scalar unchanged.

The proposed summary constraint `(c-aᵀx_3)²` is a deliberate reversal of
what is required to agree. Eliminating c adds zero energy to x_3. The centre
is reciprocally coupled during transients, but has **no effect on the ring
equilibrium in this witness**. It must not be credited with mediation or
extra computational power. Additional useful centre interactions need their
own specification; this scalar centre is not the multi-module shared core.

## 2. Shrinking widths: capacity, not automatic abstraction

For a linear encoding T_l from a k-dimensional signal family U into ring l,
lossless recovery requires rank(T_l U)=k, hence n_l>=k. For a rank-r
projector and unit-eigenvalue signal covariance of rank k, normalized signal
loss is at least max(0,1-r/k). A rank-3 bottleneck loses at least 1/4 of a
rank-4 isotropic signal; the numerical witness attains it.

The widths 32→24→16→8 have enough dimension for k=4. That does not ensure
their *local supports* or learned maps contain the useful signal. To preserve
two rank-4 source/target subspaces with nonzero equal drift angles, a common
linear code needs dimension at least dim(span(U_source,U_target))=8. Eight
inner units have enough nominal capacity in this example, but a terminal
penalty preserving only four modes does not allocate that capacity. At zero
drift the required union dimension is four.

Different layers cannot be called progressively more abstract just because
they have fewer units or update more slowly. The construction preserves a
specified subspace and filters its complement. It supplies no semantics for
abstraction and no learned method for deciding which distinctions to keep.

## 3. Outer output is not the interior projector

Partition H into outer and interior variables. Exact elimination gives

```
H_eff = H_oo - H_oi H_ii^(-1) H_io
x_0* = lambda H_eff^(-1) h.
```

This direct map is the mandatory non-iterative control. An equilibrium-only
test cannot establish that reciprocal settling computes a function unavailable
to a direct linear encoder. Learning, changing interactions, partial work or
local execution could distinguish them, but require explicit tests and costs.

With orthonormal nested D_l and unit radial strengths, a mode rejected at
the j-th serial penalty has exterior gain

```
g_j = lambda*j/(1+lambda*j),  j=1,2,3,4.
```

The four retained signal modes have gain one. In the witness, multiplicities
for j=1,2,3,4 are 8,8,8,4. Thus a softer exterior tie suppresses nuisance
without suppressing the exact learned signal nullspace, but conditioning and
sensitivity to errors in that nullspace become important. Modes rejected
only deep inside are attenuated less at the exterior.

Crucially, this exterior map G is not an orthogonal projector. Its correct
signal error and reconstruction risk are

```
L(G) = tr((G-I) S (G-I)ᵀ)/k
E(G) = [tr((G-I) S (G-I)ᵀ) + tr(G N Gᵀ)]/k.
```

The projector-only formula `1-tr(GS)/k` from the reuse experiments cannot be
substituted here. Evaluation uses S=UUᵀ and N=4VVᵀ+0.25I, with V chosen
in the first rejected subspace.

| Constructed configuration | Exterior signal error | Exterior E | Hessian condition number |
| --- | ---: | ---: | ---: |
| Summary centre, lambda=1 | <1e-28 | 2.0385 | 40.5 |
| Summary centre, lambda=0.1 | <1e-28 | 0.3481 | 192.7 |
| Original centre spring, lambda=1 | 0.0356 | 1.9825 | 23.6 |
| Original centre spring, lambda=0.1 | 0.4817 | 0.6012 | 52.6 |

For comparison, the oracle signal projector has E=0.25 and raw input E=6;
zero output has E=1. Preserving signal exactly is therefore not enough to
make the outer response useful: the lambda=1 summary case loses to zero
output on reconstruction. These are analytical witnesses, not fitted quality
comparisons or a selected lambda for the next experiment.

One can decode only the four retained inner coordinates back to the boundary;
in this orthonormal witness that yields exactly UUᵀh. But that is a different
readout, not x_0. It must be an explicit control or architectural change,
not an unreported switch to the successful projector result.

## 4. A precise fast-exterior / stable-interior mechanism exists

Let R be an orthogonal, sector-local transformation of exterior observations.
With B_0=0, replace only the first map by D_0'=R D_0, and drive with h'=Rh.
Leave all deeper maps and penalties unchanged. The energy is invariant under

```
x_0' = R x_0,  x_1'=x_1,  x_2'=x_2,  x_3'=x_3,  c'=c.
```

Consequently the new equilibrium has **exactly the same internal coordinates**
for corresponding transformed inputs. This gives a concrete role to an
adaptive exterior interface: it can absorb a changed local realization while
the deeper representation remains compatible. The witnesses verify the full
transfer identity at 0°,30°,90° to error below 2e-15.

This is an existence theorem for a known transformation, not an algorithm
that can infer it. Paired isotropic coefficients identify UUᵀ, not U itself:
UO for orthogonal O gives the same observed covariance; even sector-local sign
flips are invisible. Thus matching deeper *semantic coordinates* across contexts
needs a declared alignment convention, identifiable cross-context evidence,
or an objective invariant to that ambiguity. An oracle rotation cannot enter
a learner. Changing supports, anisotropic changes that do not obey the
orthogonal transport, and deeper structural drift need separate analysis.

It also does not preserve old input decoding automatically. Once a rank-4
readout follows U_theta, its source signal loss is sin²(theta): 0,0.25,1 at
0°,30°,90°. A slow interior behind a rewritten exterior map is not a guarantee
of source recovery. Keeping both requires enough represented directions and
a learning/use rule, or separately retained mappings with a counted selection
mechanism. The raw exterior response may retain attenuated residuals, but that
does not establish useful memory either.

## 5. Three different time scales must not be conflated

1. **Inference relaxation:** how fast current states approach equilibrium.
2. **Persistent plasticity:** how fast D_l, B_l or support assignments change.
3. **Retention:** how well old responses remain recoverable after learning.

The stable-inference proof permits any positive layer mobility. Making inner
state relaxation slow does not change equilibrium or confer memory. In the
lambda=0.1 summary witness, mobilities 1,0.5,0.1,0.01 across rings (0.01 at
the centre) reduce the slowest relaxation rate from 0.0200 to 0.000468:
approximately 43 times slower for the worst mode. These are continuous-time
spectral rates, not measured wall-time speed ratios.

For persistent learning, a proposed hierarchy is eta_0>eta_1>eta_2, with
support changes and the deepest retained-mode penalty slower still. A larger
learning rate alone does not ensure a larger actual update: gradient scales,
parameter normalization and constraints matter. Hold structure fixed during
each inference solve. If it changes concurrently,
`dE/dt = -grad_x(E)ᵀ M grad_x(E) + partial_Theta(E) dot(Theta)`, and the last
term need not be negative. Fixed-structure convergence is insufficient.

A scalar estimation model illustrates the statistical tradeoff without
pretending to prove a layer-learning theorem. Let
`a_t=(1-eta)a_(t-1)+eta y_t`, where observation noise has variance sigma².
For a constant underlying mean,
`Var(a)=eta*sigma²/(2-eta)`. For a linearly drifting mean mu_t=v*t, steady
tracking bias is `-(1-eta)v/eta`. Slower updates reduce noise but increase
lag. At sigma²=1,v=0.001 the deterministic moment checks give:

| eta | Noise variance | Drift lag magnitude |
| --- | ---: | ---: |
| 0.5 | 0.333333 | 0.001 |
| 0.1 | 0.052632 | 0.009 |
| 0.01 | 0.005025 | 0.099 |

These rates are examples, not optimized or proposed as experiment settings.
Slow learning is justified only if the quantity learned inward actually varies
more slowly. Nesting does not establish that assumption. Uniform and reversed
rate orderings, and a non-layered matched fast/slow model, remain essential
controls for any future learning claim.

## What has been combined, and what remains to design

Learned locality supplies a candidate **support pattern and signal subspace**
for the radial maps and compatible lateral penalties. The circular module
supplies **distinct state spaces and reciprocal realization**. The reuse study
supplies evidence that stable grouping can outlive fitted weights under its
generator. The transport calculation supplies a precise sufficient condition
under which changing an outer map leaves deeper responses intact.

The numerical maps and terminal signal penalty here are constructed with
known signal directions. A learned version could initialize compatible local
bases from paired-data estimates and complete unused coordinates by a declared
orthogonal convention. Extending v2's one-feature-per-block estimator to
multiple ring coordinates is a new design choice; it has not been validated.
Neither repeated use of the same subspace at depth nor orthogonal completion
constitutes a learned hierarchy of relevance.

Before freezing an experiment, the remaining decisions are concrete:

- How to learn local maps and align their coordinates using only permitted
  observations, with normalized gradients and full-rank constraints.
- How to update the retained-mode penalties without oracle signal knowledge.
- Whether the actual output is the settled outer state or a separately
  declared reconstructed component; these have different risks.
- Whether the centre is only a summary or has an additional justified role.
- What persistent state and represented capacity allow old and new contexts
  to coexist, and how it is accessed without a supplied context oracle.

The present result supports feasibility of a **learned, locally coupled,
shrinking-ring module with an adaptive exterior and more stable interior**.
It does not establish a trained learning rule, an advantage of depth or
reciprocity, source retention, or emergent abstraction. The next experiment
should address those remaining learning decisions within this explicit module,
rather than replacing it with a single projection model again.

## Verification receipt

Run `.venv/bin/python -m experiments.layered_reuse_analysis`.
The runner checks independent energy/Hessian assembly, positive definiteness,
Schur/direct equivalence, closed-form mode gains, preserved signal, a separate
projector readout control, full-state transport, source loss, union rank,
latent sign ambiguity, and stationary/tracking EMA moments. Both complete
passes must produce byte-identical scientific JSON. There are no iterative
inference runs, fitted models, sampled examples, tuning trials or selected
candidates. Direct gradient-flow convergence is proved and its spectral rates
checked; an iterative solver performance claim is not made.

[Scientific results](../reports/layered-reuse-analysis.json),
[runtime receipt](../reports/layered-reuse-analysis-runtime.json), and
[architecture drawing](../reports/layered-reuse-architecture.svg) are new
artifacts. Whole-process RSS includes imports; isolated temporary allocation
and FLOP measurements are unavailable. All final verification repetitions are
counted separately from the initial development execution. The computation
receipt is not mixed into deterministic scientific output.
