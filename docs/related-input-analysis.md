# Response to new but related inputs in the coarse hierarchy

**Status:** analytical study, 2026-10-24. No sampled data, training, or changes
to an ongoing experiment. Builds on
[coarse-interior-analysis.md](coarse-interior-analysis.md).

## 1. What can be established

For the current linear 32 -> 8 -> 4 hierarchy, the response to **every**
new input can be calculated exactly. The useful distinction is between:

- invariance: different detailed inputs produce the same coarse state;
- adaptation: a real change in broad structure changes that state;
- transfer: an earlier coarse estimate improves the response to a new input.

The first two are properties of the present architecture. The third requires
an explicit source of retained context. The architecture has fixed averaging
maps, no learned weights, and no persistent memory mechanism. Thus these
calculations establish structural response and conditional reuse, not learned
semantic generalization.

## 2. An exact definition of relatedness

Recall the orthonormal restriction matrices `A` (32 -> 8), `B` (8 -> 4),
and `C=BA` (32 -> 4). Let `h'=h+delta` be a new input. Decompose its change
into three orthogonal parts:

```
delta_fine = (I-A^T A) delta
delta_mid  = (A^T A-C^T C) delta
delta_core = C^T C delta.
```

These occupy 24, 4, and 4 independent directions respectively. The squared
norm of the full change is the sum of their squared norms.

Under the declared representation, inputs are exactly related at the deep
level when `Ch'=Ch`. They may differ arbitrarily in the 28 remaining
directions. They are approximately related when `||C(h'-h)||` is small.
Relatedness at the middle level is the stronger condition `Ah'=Ah`.

This definition is supplied by the architecture, not discovered from data.
It captures shared local averages. It does not imply that translated objects,
semantic categories, or every intuitively similar pair receive similar
representations. Fixed block boundaries can separate related-looking inputs,
and unrelated-looking inputs can share the same block averages.

## 3. Exact response with no retained context

With no coarse prior, the settled state is

```
x=h,       z=Ah,       y=Ch.
```

For a new input, therefore,

```
Delta x = delta
Delta z = A delta
Delta y = C delta.
```

No approximation or fitting is involved. The operator norm of each
restriction is one, so changes cannot become larger in normalized internal
coordinates: `||Delta y|| <= ||Delta z|| <= ||delta||`.

For unit-length changes in each orthogonal component:

| Type of change | Surface change norm | Middle change norm | Core change norm |
| --- | --- | --- | --- |
| Fine: preserve every four-site average | 1 | 0 | 0 |
| Intermediate: change four-site averages but preserve eight-site averages | 1 | 1 | 0 |
| Broad: change the four deep coordinates | 1 | 1 | 1 |

Example: increasing one surface value and decreasing its neighbour by the
same amount within a four-site block leaves every internal state unchanged.
Increasing all four values of one block while decreasing the next block
changes the middle state but leaves the deep state unchanged. Increasing
all eight positions of one coarse block changes all three levels.

Thus novel detail does not have to be represented again at every depth.
The core retains the shared broad structure, while the surface expresses
the new detail. However, in this context-free case the output surface is
still the input: representation alone is not inference or correction.

## 4. Response when a coarse context is available

Use the energy already defined:

```
E = lambda/2 ||x-h||² + kappa/2 ||Ax-z||²
  + kappa/2 ||Bz-y||² + eta/2 ||y-p||².
```

Here `p` is a supplied coarse prior, prediction, or stored summary. Let

```
q = kappa eta/(kappa+2 eta)
g = lambda/(lambda+q)
P = C^T C,       Q=I-P.
```

Then

```
x* = Qh + C^T [g Ch + (1-g)p].
```

With the same context before and after the new input,

```
Delta x = Q delta + g P delta.
Delta y = [kappa/(kappa+2 eta)] g C delta.
Delta z = (I-B^T B) A delta + 1/2 B^T (C Delta x + Delta y).
```

For `lambda=kappa=eta=1`, these become:

| Type of unit-length change | Surface | Middle | Core |
| --- | --- | --- | --- |
| Fine | 1 | 0 | 0 |
| Intermediate | 1 | 1 | 0 |
| Broad | 0.75 | 0.50 | 0.25 |

The response preserves new fine detail while negotiating broad change with
the retained context. A small core response can mean successful resistance
to nuisance, or excessive resistance to a genuine change. It is not itself
a success metric. If the context also changes, its contribution must be
included; the table assumes it remains fixed.

## 5. When reusing an old summary actually improves accuracy

Suppose the true old coarse structure is `c0` and the new one is
`c1=c0+d`. Assume current observed coarse coordinates are `c1+xi`, while
the retained prior is `p=c0+epsilon`. Let independent, zero-mean errors
have covariance `vI` and `wI`, with coarse dimension `m=4`.

The coarse surface response is

```
c_hat = g(c1+xi) + (1-g)(c0+epsilon).
```

Its error relative to the new truth is

```
c_hat-c1 = g xi + (1-g)epsilon - (1-g)d.
```

Consequently the exact mean squared coarse-coordinate error is

```
E||c_hat-c1||² = g² m v + (1-g)² (m w + ||d||²).
```

Using only the current observation gives error `m v`. Context is beneficial
precisely when, for `0<=g<1`,

```
m w + ||d||² < [(1+g)/(1-g)] m v.
```

This is an analytical boundary between useful reuse and bias from obsolete
structure. New detail can be arbitrarily different without appearing in
this condition: it is transmitted through the surface residual rather than
forced to match the old context.

Example: let old and new coarse noise each have variance 0.5 per coordinate,
independently, and use the existing `g=0.75` response. These values correspond
to separate spherical nuisance draws at `alpha=1` before orthonormal
restriction. Then:

| Squared true coarse change `||d||²` | Error using current input only | Error reusing old summary |
| --- | --- | --- |
| 0 | 2.00 | **1.25** |
| 4 | 2.00 | **1.50** |
| 12 | 2.00 | 2.00 |
| 16 | 2.00 | 2.25 |

For unchanged broad structure, reuse reduces coarse error by 37.5%, despite
the prior being just as noisy as the new observation. Beyond squared broad
change 12, the same rule hurts. These are errors in the four normalized
coarse coordinates, not whole-surface errors or classification accuracies.

The independent-error assumption matters. Correlated old/new errors add
`2g(1-g) trace(Cov(xi,epsilon))`. If the prior is simply the same noisy
current estimate again, reuse adds no evidence and gives no improvement.
There is no claim here that the system has learned when to trust a prior.
Implementing retention and choosing context strength are separate tasks.

## 6. A limited, explicit mismatch signal

Minimizing over all states gives

```
E_min(h,p) = lambda q/[2(lambda+q)] ||Ch-p||².
```

At unit parameters this is `||Ch-p||²/8`. It reports disagreement between
the observed broad structure and the supplied context, regardless of fine
detail. It could be an observable for deciding whether the old context
fits, but it is not a learned novelty detector or an error against truth.

Under the independent noise model above, its expectation is the coefficient
times `||d||²+m(v+w)`. Noise alone raises it. With no context (`eta=0`), the
minimum energy is zero for every input: the present summary-only module has
no intrinsic familiar-versus-new distinction.

## 7. Related inputs and settling from the previous state

For a fixed context and fixed quadratic Hessian `H`, suppose the module
starts at the old equilibrium and the input changes abruptly. Let `G` map
surface input to full equilibrium state. The new transient is exactly

```
u(t)-u_new* = -exp(-Ht) G delta.
```

For a fine-only change, the interior stays unchanged at every time, not
just after settling. The new surface detail approaches its target at rate
`lambda`. For an intermediate-only change, the middle layer responds but
the core stays unchanged at every time. Broad changes activate the coupled
surface/middle/core dynamics. The linear system separates these components.

Starting from a related state can reduce the time to a fixed absolute
error tolerance because the initial discrepancy is smaller. It does not
change the eigenvalues or provide a learned acceleration. For a fixed
relative tolerance within the same component, changing the amplitude alone
does not change the settling time. With a fixed current input, the unique
equilibrium remains independent of history unless retained context is
explicitly provided.

## 8. Verification and the experimental implication

Run

```
.venv/bin/python -m experiments.related_input_analysis
```

The script writes [related-input-analysis.json](../reports/related-input-analysis.json).
It solves old and new full spatial equilibria independently, checks their
differences against the transfer formula, checks energy against the minimized
expression, and evaluates the matrix-exponential transient through an
eigendecomposition. Core invariance for fine/intermediate changes and middle
invariance for fine changes hold within `1e-12` at every checked time. All
results are deterministic, with no sampled training/test inputs.

A targeted follow-up should distinguish three questions rather than combine
them into one accuracy score:

1. Do novel fine details preserve the same broad internal state?
2. Does genuine broad change update the appropriate internal state?
3. Does an explicitly retained noisy coarse summary improve response to a
   new related input, and degrade it when the relationship no longer holds?

The present calculations answer the first two exactly and establish a
conditional prediction for the third. They do not establish that fixed
block averages are the right notion of relatedness for real tasks. Learning
that notion would be the next substantive generalization problem.
