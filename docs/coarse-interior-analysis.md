# A sparse interior for broad structure, with detail retained at the surface

**Status:** analytical construction and comparison, 2026-10-24. No sampled
experiment. Existing frozen experiments and their outputs are unchanged.
This revises the design objective in light of the intended shared input/output
surface: deep units should retain broad structure, not every surface distinction.

## 1. The distinction that changes the design

A small interior can serve as either:

1. a complete reconstruction of the surface; or
2. a summary of some properties of the surface, leaving other properties
   represented closer to the surface.

These have different mathematical consequences. Let `U` have orthonormal
columns representing broad patterns and `z` be a small internal state.

**Reconstruction coupling** penalizes `||x-Uz||²`. It treats detail that
the interior cannot represent as an error. With a soft input tie, eliminating
`z` gives

```
x* = UU^T h + lambda/(lambda+kappa) (I-UU^T)h.
```

This keeps broad patterns and suppresses detail at the surface. It was the
logic behind several earlier filtering constructions.

**Summary coupling** penalizes `||U^T x-z||²`. It compares only the broad
projection with the interior. With no additional internal constraint,

```
x*=h,        z*=U^T h.
```

The interior is a compact summary while the surface keeps all detail. Thus
the requirement to keep broad strokes inside does not entail destroying
fine distinctions outside. It also does not, by itself, correct the outside:
an unconstrained summary tracks its input rather than providing new evidence.

## 2. A concrete 32 -> 8 -> 4 hierarchy

Let `x` have 32 outer units, `z` eight intermediate units, and `y` four deep
units. Build restriction matrices:

- `A` averages each contiguous block of four surface positions, with
  normalized coefficients `1/2`, so `AA^T=I_8`.
- `B` combines adjacent pairs of intermediate positions, with coefficients
  `1/sqrt(2)`, so `BB^T=I_4`.
- `C=BA` represents four contiguous blocks of eight outer positions, with
  coefficients `1/sqrt(8)`; `CC^T=I_4`.

The normalized coordinates simplify energy and covariance calculations.
Physical block averages are `z/2` and `y/sqrt(8)`. These scalings must be
included when comparing per-unit variance, noise floors, or physical dynamics.

Use reciprocal energy

```
E = lambda/2 ||x-h||²
  + kappa/2 ||Ax-z||²
  + kappa/2 ||Bz-y||²
  + eta/2 ||y-p||².
```

Every outer unit is still both input and output. The optional `p` is an
explicit coarse context, prediction, or command. It is not silently assumed
to be correct. With `eta=0`, there is no such contextual constraint.
For `lambda,kappa>0` and `eta>=0`, the driven Hessian is positive definite:
zero homogeneous energy forces `x=0`, then `z=0`, then `y=0`.

These squared aggregate constraints induce some same-layer interactions
when expanded as a pairwise energy. They are not merely independent positive
springs from individual surface units to an average unit. Such ordinary
springs would additionally penalize within-group differences and return to
the reconstruction problem. A physical implementation must support the
aggregate constraint or its equivalent mixed-sign interactions.

## 3. Exact broad-stroke representation

Without contextual forcing (`eta=0`), the unique equilibrium is

```
x*=h,        z*=Ah,        y*=Ch.
```

There is an exact orthogonal decomposition of surface information:

```
h = (I-A^T A)h + (A^T A-C^T C)h + C^T C h.
```

| Component | Dimension | Meaning |
| --- | --- | --- |
| `(I-A^T A)h` | 24 | Differences within four-site blocks |
| `(A^T A-C^T C)h` | 4 | Differences between adjacent four-site blocks |
| `C^T C h` | 4 | Broad eight-site block structure |

The deeper state does not carry all 32 independent directions. It carries
four broad coordinates. Intermediate and outer states preserve the
remaining variations. This is information separation by scale, not proof
of semantic abstraction.

For the old test waves, the exact power retained in the reconstructed
summary is:

| Signal | Intermediate summary | Deep summary | Actual surface, eta=0 |
| --- | --- | --- | --- |
| Mode 1, either phase coordinate | 95.27% | 81.32% | 100% |
| Mode 8, either phase coordinate | 0% | 0% | 100% |

Mode 8 cancels within each four-site block. Its absence deep inside is
therefore a **desired architectural result** under this objective, provided
the surface can still express it. Requiring a deep decoder to recover it,
as in the earlier profile, would test a different objective.

The unretained part of mode 1 remains in the shallower residuals, so these
percentages are not total information loss from the module. Four piecewise
constant block values approximate a broad wave rather than reproducing it
exactly. Block averaging is not an ideal anti-aliasing Fourier filter:
other fine patterns with nonzero block averages can influence the core.
Moving block boundaries changes the representation. Smooth or learned
restrictions are possible alternatives, not established by this example.

## 4. Noise and the meaning of a quieter interior

For input nuisance covariance `v I_32`, where `v=alpha²/2`, physical
intermediate averages have variance `v/4` and deep averages have variance
`v/8`. At `alpha=1`, these are 0.125 and 0.0625, versus 0.5 at each surface
unit. A signal constant within an averaging block retains its amplitude.

This calculation requires the covariance, not independence of the
spherical nuisance coordinates. It gives a precise sense in which broad
structure is less contaminated by local variation.

In the normalized coordinates `z,y`, each noise coordinate instead has
variance `v`. The reconstructed deep field has total noise power only
`4/32` of the full field because it has four independent directions.
Do not claim that an arbitrary coordinate rescaling itself creates better
signal-to-noise ratio. Averaging only improves estimation of structure that
is genuinely shared across the averaged positions; coherent broad noise
survives.

## 5. The interior can guide the surface without erasing its detail

Set `P=C^T C`, the coarse projector, and `Q=I-P`. Eliminate `z`, then `y`.
Two equal coupling penalties in series produce effective stiffness

```
q = kappa eta/(kappa+2 eta).
E_surface = lambda/2 ||x-h||² + q/2 ||Cx-p||².
```

The surface solution is

```
x* = Qh + C^T [lambda/(lambda+q) Ch + q/(lambda+q) p].
```

This is an explicit shared-surface computation. External data and internal
context negotiate the broad components, while

```
Qx*=Qh
```

holds exactly. The correction has rank at most four. A change in `p`
coordinates a whole group of surface units rather than specifying their
individual detail.

For `lambda=kappa=eta=1`, `q=1/3`: coarse surface coordinates are 75% supplied
coarse data and 25% contextual values. Fine residual coordinates are 100%
supplied data. The preserved "detail" here means the kernel of the declared
block-averaging map, not every high Fourier frequency. Piecewise constant
coarse corrections can themselves introduce sharp block boundaries.

If `p` agrees with the clean coarse structure, it can correct coarse error.
If it is wrong, it introduces a coordinated bias. If `p` is just recomputed
from the same noisy `Ch`, there is no additional evidence and no denoising
benefit. Learning or another module would have to supply useful constraints;
this construction does not assume that problem solved.

## 6. Compare layered, compact, and surface-only realizations

The following three systems produce **exactly the same equilibrium surface
map for every h and p**, not merely equal performance on selected examples:

1. **Layered:** 32 surface + 8 intermediate + 4 deep variables, with the
   energy above.
2. **Compact:** 32 surface + 4 deep variables, using
   `lambda/2||x-h||² + kappa/4||Cx-y||² + eta/2||y-p||²`.
   The effective surface-to-core spring is `kappa/2`, the series value.
   Its deep equilibrium also equals the hierarchy's deep equilibrium.
3. **Surface-only:** 32 variables, using the eliminated energy
   `lambda/2||x-h||² + q/2||Cx-p||²`.

For `lambda=kappa=eta=1`:

| Realization | State variables | Scalar pair interactions | Slowest state decay time |
| --- | --- | --- | --- |
| Layered 32 -> 8 -> 4 | 44 | 92 | 2.618 |
| Compact 32 -> 4 | 36 | 144 | 1.000 |
| Surface-only | 32 | 112 | 1.000 |

Pair interactions count nonzero off-diagonal entries in the symmetric
Hessian once per pair, excluding external forcing. They include the
same-layer interactions implied by the aggregate constraints. These are
counts for explicit pairwise realizations, not general computation or
parameter-storage lower bounds. The matrices have shared coefficients;
a program can apply the surface-only operator through block sums without
storing 112 independent weights.

The hierarchy's surface interaction neighborhoods cover four sites rather
than eight. Its extra states allow more local, fewer explicit pairwise
connections in this example, but make relaxation slower under the stated
unit-mobility dynamics. Eliminating the intermediate layer makes surface
interactions broader. Exact transient behaviour is not preserved by static
elimination.

Thus **a sparse hierarchy has a demonstrable representational role and a
possible wiring benefit, but no equilibrium accuracy advantage in this
linear comparison**. The compact model is sufficient to retain and mediate
the four coarse coordinates. The intermediate layer is justified only if
its intermediate summaries or implementation constraints are useful.

## 7. Limits implied by sparsity

**Correction capacity.** In this aggregate model, surface correction lies
in a four-dimensional subspace. More generally, a linear interior coupled
through `m` aggregate coordinates cannot independently prescribe corrections
in more than `m` surface directions through those coordinates. Separate
surface processing or additional shallow states can handle others.

**Broad correction is not general denoising.** If all 28 detail directions
must remain unchanged, their white nuisance remains too. Even a perfect
coarse correction can remove at most `4/32=12.5%` of total isotropic nuisance
power. With the example's 75/25 data/context blend and a perfectly accurate
context, 94.53% of the total nuisance remains. That is compatible with a
useful coarse state but would fail the earlier aggressive whole-surface
denoising objective. The earlier projection design removed most noise by
also suppressing detail; it solved a different problem.

**Missing detail cannot be inferred from a coarse summary alone.** If a
surface unit is unobserved and its fine residual is unconstrained, broad
averages do not generally determine it. Completion needs local detail
priors or other observations. The old known-subspace completion theorem
must not be transferred to this broader detail-preserving design.

**Finite dimension.** A few scalar deep units cannot linearly retain every
independent detail. That is the intended bottleneck. For example, preserving
DC and both phases of the first Fourier mode requires at least three
independent coordinates; a scalar centre cannot carry all three. The present
four-block summary is one specific coarse representation, not the unique
or universally optimal choice.

## 8. Checks and a better experimental question

Run

```
.venv/bin/python -m experiments.coarse_interior_analysis
```

The script writes [coarse-interior-analysis.json](../reports/coarse-interior-analysis.json).
It independently assembles all three Hessians and solves for the full
response to every surface/context basis vector. Surface maps agree with
the closed form within `4e-16`; compact and layered deep states agree;
all driven Hessians are positive definite. It also verifies the orthogonal
24+4+4 decomposition, preserved detail under context, mode-1/mode-8 retention,
pair counts, and dynamical eigenvalues. No sampled data are used.

A subsequent experiment should test the architecture's intended separation:

- change within-block detail while holding broad averages fixed: deep state
  should stay fixed and the surface should retain those changes;
- change broad structure: deep state should track it, with the analytically
  predicted noise level;
- provide an explicit coarse context: surface broad components should move
  by the predicted amount while fine residuals stay unchanged;
- compare layered and compact realizations on settling and wiring, while
  expecting identical equilibrium outputs;
- include inaccurate context and missing-detail cases to expose bias and
  underdetermination.

This supports a revised design principle: **keep broad summaries deeper,
keep unresolved detail nearer the shared surface, and compare only the
properties that each level is intended to represent**. It does not require
forcing all distinctions through the smallest interior state.
