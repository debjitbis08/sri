# Shared input/output surface: analytical reinterpretation of the layered module

**Status:** analytical study and experiment proposal, 2026-10-24. No
confirmation run. This work does not change or inspect the scored outputs
of the concurrently running layered-selectivity-v1 experiment.

## 1. The architectural correction

The outer ring is both the input and output surface. External influence
changes its state; the interior responds; that response changes the same
surface. Depth is internal structure supporting the surface's computation,
not a route to a separate output at the centre.

The current energy already allows this feedback. However, the earlier
experiments decode the innermost ring. They test inward transmission, not
the intended shared-surface function. Their results remain valid for their
declared question but do not establish useful outward response.

Three distinctions are essential:

- The externally supplied value `h` is not the resulting surface state `x`.
  A soft tie allows them to differ.
- A surface unit has no permanent input/output identity. Which units are
  externally influenced can change between interactions.
- Clamping every unit fixes every surface value. In that protocol an output
  could be the reaction force, but cannot be a changed surface value. State
  response and reaction force must not be conflated.

This analysis uses surface state as the output. Partial observations are a
test of the shared surface, not a requirement that it have permanent input
and output partitions. All-site soft stimulation remains another valid use.

## 2. Eliminate the interior to see exactly what it contributes

Write the boundary state as `x` and all interior states as `z`. For a
quadratic module, partition its *undriven* stiffness matrix as

```
H = [ Hbb  Hbz ]
    [ Hzb  Hzz ].
```

For fixed `x`, the settled interior is

```
z*(x) = -Hzz^{-1} Hzb x.
```

Its effective surface energy is

```
E_surface(x) = 1/2 x^T S x
S = Hbb - Hbz Hzz^{-1} Hzb.
```

`S` is the Schur complement: the matrix of constraints the interior imposes
on the surface. It is symmetric and positive semidefinite for the designs
here. It generally couples distant surface positions even though the full
network uses local connections. This is an explicit way for the interior
to organize relationships across the shared surface.

With an observation mask `M` (ones at influenced units, zeros elsewhere) and
soft tie strength `lambda`, the surface settles to

```
(S + lambda M) x* = lambda M h.
```

Only `Mh` enters this equation: unknown target values are never supplied to
the module. They are not softly tied to zero. A zero value at an observed
position and a position with no observation are different boundary conditions.

For all-site influence, `M=I`, the response is
`x*=(S+lambda I)^{-1}lambda h`. This is already a simultaneous input/output
surface. We do not have to introduce separate output neurons.

## 3. What the current disk does at its surface

For a Fourier mode with interior lateral stiffness `p_k`, compute its
terminal load and move outward:

```
a_L = p_k + b_k
a_l = p_k + a_{l+1}/(1+a_{l+1}),  l=L-1,...,2
s_k = a_2/(1+a_2).
```

The current scalar centre gives `b_0=0` and `b_k=1` for every non-DC mode.
Its four target coordinates have `p_k=0`, hence `s_k=1/L`. With a unit tie,
the target surface gain is `L/(L+1)`, or 0.8 at depth 4.

There is a general surface-filtering limit here: with unit first radial
springs and no exterior lateral penalty, `0 <= S <= I`. To see the upper
bound, hold `x` fixed and choose every interior state zero. That costs only
`||x||²/2`; allowing the interior to minimize cannot cost more. Consequently
all-site unit-tie gains lie between 1/2 and 1. Increasing interior stiffness
cannot make arbitrary surface modes vanish through this interface.

For the selective depth-4 disk, the surface nuisance-to-signal ratio,
relative to the input ratio, is about **0.478**. This is useful filtering,
but much weaker than the innermost layer's filtering. An interior statistic
must not be presented as a surface result.

More importantly, the centre assigns positive surface energy to every
target pattern. It is a single DC-compatible termination, not a neutral
reservoir for the four target coordinates. With only part of the surface
observed, this biases completion toward smaller amplitudes. The calculations
below quantify that bias. This is a finding about this particular centre
connection, not an objection to having an interior or a central region.

## 4. A sufficient condition for useful surface completion

Let `F` have four orthonormal columns: cosine and sine for modes 1 and 8.
A valid surface signal is `s=Fa`, with arbitrary real coefficients `a`.
Require the effective surface energy to satisfy

```
S >= 0,     null(S) = range(F).
```

Thus every valid signal is a zero-energy pattern, and every incompatible
direction has positive energy. With exact observations `Mh=Ms`, `x=s`
has zero total energy, including the external tie. It is the **unique**
minimum precisely when

```
rank(F_observed) = 4.
```

Proof: a zero-energy difference must lie both in `null(S)` and in
`null(M)`. Such a difference is `F delta_a`, with
`F_observed delta_a=0`. It vanishes exactly when the observed basis has
full column rank. Equivalently, `S+lambda M` is positive definite under
that condition, for any positive `lambda`.

This gives exact noiseless pattern completion at the shared surface,
without a classifier, a privileged output end, or fitting a decoder.
Changing the observation mask changes the computational question while
leaving the same module in place.

Observing enough units by count is not sufficient. On even-indexed sites,
`sin(8 theta)` is identically zero; on odd-indexed sites, `cos(8 theta)`
is zero. Both 16-site masks have rank three and hide a target coordinate.
No method can uniquely recover that coordinate for unrestricted coefficients
from those observations alone. Arbitrarily returning zero is not successful
completion. This is a useful mandatory ambiguity control.

## 5. A minimal realization and its limitations

Keep the earlier selective lateral penalty

```
K = tau [(C-e_1 I)(C-e_8 I)]^T [(C-e_1 I)(C-e_8 I)]
tau = 1/min_positive [(e_k-e_1)(e_k-e_8)]².
```

Keep 32 units per ring, unit radial springs, no exterior lateral penalty,
and a masked unit soft tie at the surface. For this new analysis only,
leave the last ring **unterminated**: omit its springs to the scalar centre.
The energy is

```
E = 1/2 ||M(x-h)||²
  + 1/2 sum_{l=2}^L ||u_l-u_{l-1}||²
  + 1/2 sum_{l=2}^L u_l^T K u_l,    x=u_1.
```

Its undriven nullspace consists exactly of the four target patterns,
identical on every ring. Hence its effective `S` has exactly the required
nullspace. The masked external tie makes the full system strictly stable
when the mask is identifiable. The scalar centre can be removed from the
state rather than retained as an isolated, uncertifiable zero mode.

If a central region is essential, an alternative is four latent centre
coordinates `c`, with terminal energy `||u_L-Fc||²/2`. Eliminating them
penalizes only the complement of the target subspace, preserving the same
completion theorem. That alternative introduces explicit mode-specific,
nonlocal centre connections; it is not the same local architecture and is
not the numerical candidate evaluated here.

The unforced, unterminated module has four flat directions. It is not
strictly stable or uniquely determined with no observations, or with an
unidentifiable mask. This is the cost of allowing all target amplitudes
without bias. A small additional anchor would restore uniqueness but also
introduce shrinkage; it must be declared as a different design. A pure
linear field along these flat directions would make the energy unbounded,
so it cannot replace the masked soft tie without another anchor.

These flat directions are not demonstrated robust memory. Under a valid
fixed mask the equilibrium is unique and independent of the starting state.

## 6. Exact noisy-completion predictions

Use four rings. Observe 16 consecutive surface sites, with the other 16
left free. All 32 rotations of this mask are identifiable. The eigenvalues
of `F_observed^T F_observed` are approximately
`0.41031, 0.49117, 0.50883, 0.58969`: no target direction is nearly hidden.
The complement is another such mask, so input/output roles can be reversed.

For calculations, give the coefficient ensemble second moment
`E[aa^T]=4I`, so mean signal power is 1/2 per site. One realization is
uniform coefficients on the radius-four sphere in four dimensions.
Nuisance is independently uniform on the radius-`4 alpha` sphere in 32
dimensions, as before, but only its observed coordinates enter the drive.

Let `T=(S+M)^{-1}M`, `O` denote observed sites, and `U` free sites. Then
the expected free-site mean squared error, including independent readout
noise of standard deviation `sigma`, is exactly

```
MSE_U = 4/|U| ||(TF-F)_U||_F²
      + alpha²/(2|U|) ||T_{U,O}||_F²
      + sigma².
```

No Gaussian approximation or Monte Carlo is needed: this follows from
linearity, independence, and the exact covariances. Normalize by signal
power 1/2 to obtain NMSE. At `alpha=1`, `sigma=0.1`:

| Architecture, depth 4 | Clean bias MSE | Nuisance MSE | Total free-site NMSE |
| --- | --- | --- | --- |
| Current selective disk, scalar centre | 0.18573 | 0.02007 | 0.43160 |
| Selective interior, unterminated last ring | ~0 | 0.12956 | **0.27912** |
| Ordinary smoothing, unterminated | 0.59958 | 0.05767 | 1.33451 |
| Wrong bands 1 and 4, unterminated | 0.26832 | 0.14341 | 0.84346 |
| No lateral penalty, unterminated | Nonunique | Nonunique | Not evaluable |
| Direct four-coordinate least-squares reconstruction | 0 | 0.12919 | 0.27839 |

Returning zero at the free sites would give NMSE 1 before readout noise,
or 1.02 with the same readout noise. The candidate is therefore expected
to improve substantially on that baseline. Its predictions at `alpha=0.5`
and zero nuisance are NMSE 0.08478 and 0.02 respectively, with `sigma=0.1`.
Without readout noise, clean completion is exact.

The direct least-squares baseline knows exactly the same four-dimensional
subspace and uses only observed values. The layered design nearly matches
it; it does not outperform it. Its benefit here is realizing completion
through local reciprocal interactions, not statistical superiority.

The current disk trades lower variance for substantial bias. At other
noise levels that tradeoff could help; the table is not a universal
ranking. The unpenalized target nullspace is chosen to establish exact
clean consistency, not claimed to minimize risk at every noise level.

## 7. Deterministic verification

Run

```
.venv/bin/python -m experiments.shared_surface_analysis
```

The standalone script writes
[shared-surface-analysis.json](../reports/shared-surface-analysis.json).
It assembles the full spatial matrices independently of the scalar load
recurrence; verifies their Schur complements; compares masked full-state
solutions with reduced boundary solutions; checks symmetry and the surface
stiffness bound; and verifies exact clean completion for all 32 rotated
contiguous half-masks. Maximum clean transfer error is below `9e-13`.

The even/odd masks are explicitly reported as nonunique for the candidate.
They are not silently inverted, regularized, or counted as zero-accuracy
examples. Ordinary smoothing can have a unique but wrong completion;
uniqueness alone is not evidence of recovering the missing information.

These are deterministic operator calculations. No new training/test drives
were sampled. The running experiment's source, frozen contract, manifest,
and results were not modified, and its outcomes were not used to select
this candidate.

## 8. Proposed next experiment: shared-surface completion

This is a proposal, not a frozen contract. Keep the current experiment
running and report it under its existing claims. A subsequent contract
should test the following separately:

1. **State consistency:** settle exact partial patterns and test the actual
   free surface values, not a learned classifier. Check all four basis
   responses and combinations. Require clean error below the numerical
   tolerance established by independent spatial and boundary solves.
2. **Role reversibility:** repeat with masks starting at sites 0, 8, 16,
   and 24, each observing 16 consecutive sites. Every unit is observed in
   some trials and read as output in others. Use the same module weights.
3. **Noisy completion:** use fresh continuous mixtures of both modes,
   rather than just the previous eight phase-class patterns. At each
   `alpha in {0,0.5,1}`, report free-site NMSE with fixed readout noise
   `sigma=0.1`, clean bias, and observed-site deviations separately.
4. **Ambiguity:** include even-only and odd-only masks. Verify the rank
   failure and exhibit two different valid patterns with identical cues.
   Report the unresolved direction, not a successful forced guess.
5. **Controls:** current scalar-centre disk, no lateral penalty, ordinary
   smoothing, wrong bands, zero completion, and direct four-coordinate
   least squares. The no-lateral case is underdetermined here, even though
   it nearly passed the previous inward-decoding profile.

A reasonable primary engineering target is exact clean consistency and
mean NMSE at most 0.35 at `alpha=1` across all four identifiable masks. The
analytic expectation is 0.27912, leaving room for finite-sample variation.
The eventual contract should specify sample size and uncertainty estimation
before generating new data; for example, 200 independent patterns per noise
level, paired across arms and masks. Reusing a pattern across masks creates
dependence, so uncertainty must be computed by whole-pattern groups, not
by treating all output units or masks as independent samples.

No classifier training is needed. The supplied subspace is an architectural
prior, disclosed as such. Acceptance would demonstrate shared-surface
completion of a known pattern family, not learning or general intelligence.

## 9. What this says about the architecture itself

For static quadratic behaviour, a boundary-only network with energy
`x^T S x/2` is exactly equivalent to the entire layered interior. A static
completion experiment cannot by itself establish that depth is necessary.
The direct boundary model and least-squares reconstruction should remain
explicit reference implementations.

The internal architecture can still matter for local wiring, parameter
sharing, physical realization, learning rules, transient responses, and
interactions with other modules. In particular, eliminating dynamic interior
states produces time-dependent boundary effects; a single static matrix
does not reproduce all transients. Those are separate questions from the
equilibrium completion result.

Reciprocity also constrains the interface. With fixed mask and an added
boundary force `f`, the linear susceptibility is
`dx*/df=(S+M)^{-1}`, a symmetric matrix. The observed-value transfer
`(S+M)^{-1}M` need not be symmetric. Thus the architecture supports reciprocal
constraint satisfaction, not an arbitrary directed input/output mapping
merely by relabeling neurons. Nonlinear or learned energies would require
their own analysis.

The immediate correction in research direction is therefore to study
**what the interior makes the shared surface do**. Pattern completion is
a concrete first demonstration, with a proof of when it can work, an exact
noise prediction, and an explicit condition under which it cannot work.
