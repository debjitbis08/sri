# Analytical design after layered module v2

**Status:** design analysis, 2026-10-24. Includes explicitly exploratory
finite-sample pilots. This is not a frozen contract or a confirmatory run.
The proposed experiment is in
[layered-selectivity-v1-contract.md](layered-selectivity-v1-contract.md).

## 1. Conclusions

There is a feasible, stable, finite-range quadratic design for both declared
signal families. Learning and a double well are not required for this
specified task. Replace ordinary lateral smoothing in the interior by a
positive-semidefinite penalty whose nullspace contains modes 1 and 8.
Keep 32 units per ring, reciprocal radial springs, the original centre, and
the original exterior tie strength. Remove exterior lateral smoothing so
that the exterior remains the least-filtered layer.

At depth 4 the proposed design has, analytically:

- equal signal gains of 0.8, 0.6, 0.4, 0.2 on successive rings;
- nuisance suppression ratio 0.01658 (target: at most 0.25);
- population Gaussian-reference accuracy 0.99891 for both families at
  `alpha=1`, absolute readout noise `sigma=0.1`;
- a rigorous lower bound of 0.99034 on population nearest-mean accuracy for
  the actual spherical nuisance plus Gaussian readout noise;
- positive-definite energy, a unique equilibrium, and no gain above one.

An important simpler control also works: **remove all lateral smoothing**.
It nearly always meets the old profile in exploratory pilots. Thus passing
T1–T3 alone does not demonstrate selective preservation: broadband
attenuation can pass when enough signal amplitude remains. The proposed
experiment keeps T1–T3 and adds a separate, explicit selectivity measurement.

These findings concern a designed filter for known signal bands. They do
not establish abstraction, learned relevance, temporal memory, or useful
inter-module communication.

## 2. Feasibility conditions before choosing a mechanism

Use orthonormal Fourier coordinates on a 32-unit ring. A phase class has
signal norm 4; the four means are a square in its two-dimensional mode
plane. Let `g_k(l)` be a real, phase-independent mode gain at layer `l`.
The nuisance covariance is exactly `(alpha²/2) I`, although its distribution
is spherical, not Gaussian. Therefore

```
P(l) = alpha² / 64 * sum_{k=0}^{31} g_k(l)²
S(l) = g_n(l)² / 2 + P(l)
P(L)/P(1) = sum_k g_k(L)² / sum_k g_k(1)².
```

The sums use the full complex Fourier index set, counting both real
coordinates of every non-DC, non-Nyquist mode. These expressions are
population quantities, without readout noise. They are not fitted to v2
test outcomes.

For signal gain `a`, absolute noise `sigma`, and known class means, the
Gaussian reference is

```
A_G(a) = Phi(4a / sqrt(alpha² a² + 2 sigma²))².
```

There is no requirement to preserve unit signal amplitude to pass T3.
For example, at `alpha=1`, `sigma=0.1`, `a=0.2` gives `A_G=0.99891`.
A uniform interior contraction by 0.2 relative to an identity exterior
would already reduce nuisance variance by 96% while retaining this accuracy.
That is why a stronger mechanism claim needs a signal-normalized measure.

Conversely, a linear map cannot selectively remove in-band nuisance while
leaving signal in the same direction unchanged. On that direction it
multiplies both by the same gain. It can reject other directions. Preserving
the four target coordinates at gain one requires total squared gain at
least four; relative to an identity exterior the best nuisance ratio is
therefore `4/32=1/8`. This is a bound for that gain requirement, not a bound
on ratios when signal amplitude is also reduced.

## 3. A local penalty with the correct nullspace

Let `C` be the ordinary ring Laplacian:

```
(Cu)_j = 2u_j - u_{j-1} - u_{j+1}
e_k = 2(1 - cos(2 pi k/32))
B = (C - e_1 I)(C - e_8 I)
q_k = [(e_k - e_1)(e_k - e_8)]².
```

`B` uses only positions `j-2,...,j+2`. The energy `||Bu||²/2` has Hessian
`B^T B`, which couples positions at circular distance at most four. This
is finite-range, with mixed-sign effective couplings; it is **not** a sum
of positive nearest-neighbour difference springs. Its stability follows
from the squared residual, not from each effective pairwise coefficient
being attractive.

Exactly the four coordinates for modes 1 and 8 have `q_k=0`; DC is penalized.
Choose the coefficient from the spectrum, without sample fitting:

```
tau = 1 / min_{k not in {1,8,24,31}} q_k
    = 169.2824418989257.
```

Every unwanted mode then has interior stiffness at least one. The smallest
nonzero `q_k` is 0.00590728719 (DC). The largest lateral stiffness is
10626.90254. This is a real numerical cost: the design is stable but stiff.
No claim is made that this coefficient is optimal or inexpensive to realize.

## 4. Recommended disk energy and exact solution

There are `L` rings of 32 units and one scalar centre `c`. Use

```
E = 1/2 ||u_1 - h||²
  + 1/2 sum_{l=2}^L ||u_l - u_{l-1}||²
  + tau/2 sum_{l=2}^L ||B u_l||²
  + 1/2 sum_j (u_{L,j} - c)².
```

There is no exterior lateral penalty and no scalar local potential.
All radial and centre spring strengths and the exterior tie strength are
one. This is reciprocal gradient dynamics: differentiating the energy
couples both sides of every radial edge. It is not a feed-forward update.

The homogeneous quadratic form is positive definite. Zero energy forces
`u_1=0`, then equality along every radial connection forces every ring to
zero, and the centre term forces `c=0`. Adding the nonnegative lateral
penalty preserves positive definiteness. Thus every drive has one certified
minimum in exact arithmetic, and `du/dt=-grad E` converges to it.

Eliminating the centre adds `I - 11^T/32` to the final ring: a stiffness of
one for every non-DC mode and zero for DC. For each mode solve the symmetric
tridiagonal system

```
H_k g_k = (1,0,...,0)^T
diagonal: 2, 2+p_k, ..., 2+p_k, 1+p_k+b_k
off-diagonals: -1
p_k = tau q_k; b_0=0; b_k=1 otherwise.
```

For `L=2` the diagonal is simply `(2, 1+p_k+b_k)`. In a target mode `p_k=0`,
so the exact solution is

```
g_1(l) = g_8(l) = (L-l+1)/(L+1).
```

The centre still attenuates these modes, but only algebraically with depth,
not the exponential frequency-dependent attenuation from ordinary lateral
smoothing. This matters: removing the centre is **not necessary** to pass
the profile at these depths.

All gains are nonnegative, at most one, and nonincreasing inward. This
follows by working backward from the terminal equation and using
`g_{l-1}=(2+p)g_l-g_{l+1}`. Parseval's identity then implies T1, even for
finite-sample variances computed from the same drives on every layer.

Analytic predictions at `alpha=1`, `sigma=0.1`:

| Depth | Interior signal gain | P(L)/P(1) | Gaussian reference, both families |
| --- | --- | --- | --- |
| 2 | 1/3 | 0.06202 | 0.99977 |
| 4 | 1/5 | 0.01658 | 0.99891 |
| 6 | 1/7 | 0.00804 | 0.99553 |

The centre-free variant is also feasible: omit the centre and use exterior
tie strength four. Target gains then equal one at every depth; at depth 4
the suppression ratio is 0.18100. With tie strength one that variant has
ratio 0.35649 and fails T2 because the exterior has already suppressed much
of the nuisance. The earlier suggestion that changing both tie and centre
was necessary applies only to seeking depth-independent unit gain, not to
meeting the original profile. The disk above changes fewer structural rules.

## 5. Accuracy for the actual nuisance distribution

The Gaussian reference is useful, but it is not the only support for T3.
For a uniformly distributed vector on the radius-`4 alpha` sphere in 32
dimensions, every unit-direction projection is sub-Gaussian with variance
proxy `alpha²/2`. One direct proof compares its even moments with Gaussian
moments: their ratio is
`32^m / [32(32+2)...(32+2m-2)] <= 1`; odd moments vanish. Comparing the
moment-generating series gives the stated bound.

In a signal plane with gain `a`, adding independent Gaussian readout noise
gives proxy `v = alpha² a²/2 + sigma²`. Each of the two adjacent class
boundaries is distance `2 sqrt(2) a` from the true mean. If both adjacent
comparisons are correct, the opposite-class comparison is also correct.
A union bound gives

```
Pr(error of population NM) <= 2 exp(-4a²/v).
```

At depth 4, `a=0.2`, `alpha=1`, `sigma=0.1`, the lower bound on accuracy is
`1 - 2 exp(-16/3) = 0.9903441`. This establishes a margin without replacing
the true nuisance by Gaussian noise. It does **not** bound the fitted
nearest-mean readout with 25 training examples per class. That separate
finite-sample question is examined by the exploratory pilot and reserved
confirmation run.

## 6. The simpler control and what counts as selectivity

Set the interior lateral penalty to zero, keeping all other disk parameters.
Every non-DC mode then has gain `(L-l+1)/(L+1)`; DC has gain one. Thus

```
P(L)/P(1) = [1 + 31/(L+1)²] / [1 + 31 L²/(L+1)²].
```

At `L=4` this is 0.10749 and the signal gain remains 0.2. This control can
pass the original profile without favoring either target band. It even
worsens nuisance relative to signal because DC survives unchanged.

Define, separately from T1–T3,

```
J_n = [P(L)/P(1)] / [g_n(L)/g_n(1)]².
```

This is the change in nuisance-to-pure-signal power ratio from exterior to
interior. Uniform scaling gives one. At depth 4:

| Design | Original suppression ratio | J_1 = J_8 |
| --- | --- | --- |
| No lateral penalty | 0.10749 | 1.71977 |
| Proposed selective penalty | 0.01658 | 0.26528 |

The proposed design improves the nuisance-to-signal ratio by about 3.77
times across layers. This improvement comes from rejecting out-of-band
nuisance, not separating signal from nuisance in the same band. An explicit
criterion `J_n <= 0.5` is proposed for a new selectivity claim. It is not
retroactively attached to v1 or v2.

## 7. Numerical checks and exploratory pilots

Reproduce with:

```
.venv/bin/python -m experiments.layered_module_design_analysis
```

The script writes [design-analysis.json](../reports/layered-module-design-analysis.json).
It builds the spatial Hessian independently of the mode equations, including
the scalar centre, and checks its inverse against the Fourier transfer.
Maximum error is below `1e-10`; it also checks the centre-free spectrum,
monotone gains, and the no-amplification property of the centre-free variant.

For finite-training behaviour, exploratory pilots use 200 independent
replicates, each with the v1 sample sizes (25 training and 25 test drives per
class per family), true spherical nuisance, `alpha=1`, and fixed `sigma=0.1`.
Nearest mean operates on all 32 noisy interior coordinates, without a
mode-aware projection. Seeds are `[20261029,0,rep]` for nuisance and
`[20261029,1,rep,family]` for readout noise. Draws are paired across designs.
Direct equilibrium maps are used; no optimizer certification claim is made
for these pilots. They are exploratory observations, not analytic guarantees.

| Depth-4 design | Replicates meeting original profile in both families | Mean accuracy F-low / F-high |
| --- | --- | --- |
| Proposed selective disk | 200/200 | 0.9990 / 0.99865 |
| No lateral penalty | 199/200 | 0.9887 / 0.98485 |
| Ordinary interior smoothing | 0/200 | 0.97915 / 0.26845 |

For the selective disk, minimum observed accuracies were 0.98 / 0.99, and
suppression ratios ranged from 0.01486 to 0.01829. Since the target gain
ratio is exactly 1/4, these also put every pilot `J_n` below 0.293.
The same selective design passed the original profile in all 200 pilots
at depths 2 and 6. These are correlated comparisons across depths, not 600
independent confirmations. Additional centre-free and wrong-band controls
are recorded in the JSON.

Candidate selection was analytical, followed by exploratory comparisons;
the final choice of the simpler disk and the additional claim criterion
were informed by this analysis and its pilots. Fresh confirmation seeds
are therefore essential. No v1/v2 test arrays or proposed confirmation
arrays were used, and no v1/v2 result files were changed.

## 8. Costs and limits

The design requires a hand-specified relevant subspace, mixed-sign lateral
interactions out to four sites, 32 units per ring, and stiff coefficients.
It is a constructive existence result under those allowances. It is not
a proof that the original positive-spring architecture can work unchanged.

At depth 4 the centre-free Hessian has condition number about 61,389; the
full disk Hessian extrema are reported separately in the JSON. Stability
does not imply fast explicit-gradient settling. The confirmation should use
an SPD direct solve as its exact reference, retain optimizer certification,
and report conditioning and iterations. It should not claim biological
plausibility or efficient physical dynamics from these calculations.

The nullspace depends on the declared modes and discretization. Success
does not imply tolerance to arbitrary frequency shifts or parameter error.
The wrong-band control is intended to make that dependence visible. A later
study can derive tolerance bounds or learn the penalty; neither is needed
to establish this first constructive result.
