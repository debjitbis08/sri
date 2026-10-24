# Full shared-surface response: exact denoising, coupling, and settling analysis

**Status:** deterministic analytical study, 2026-10-24. No new sampled
experiment, no changes to the running experiment. This extends
[shared-surface-analysis.md](shared-surface-analysis.md) to the case where
every outer unit receives an external value and every outer unit is also
read as output.

## 1. The full surface remains both input and output

Let `h=s+z` be the supplied surface pattern: clean signal `s` plus nuisance
`z`. Let `x` be the actual outer state. Every unit receives the soft influence
`lambda/2 (x_j-h_j)²`; no unit is permanently assigned an output-only role.
The result is the settled `x`, not the interior state or a classifier label.

Eliminating the linear interior gives its effective surface energy
`x^T S x/2`. Hence

```
E_surface = 1/2 x^T S x + lambda/2 ||x-h||²
x* = (S+lambda I)^{-1} lambda h.
```

For `lambda>0` and `S>=0`, this is a unique stable surface equilibrium.
It allows all units to receive and express influence simultaneously.
Hard-clamping all states is the limit `lambda -> infinity`, where `x*=h`:
the state output then becomes a copy, although reaction forces can still
carry information about the interior.

## 2. Each surface pattern has an exact gain

The ring construction is translation invariant, so `S` is diagonal in
Fourier coordinates. If its eigenvalue for coordinate `k` is `s_k`, then

```
x*_k = g_k h_k,       g_k = lambda/(lambda+s_k).
```

There are no hidden fitted quantities in this equation.

- `s_k=0`: the pattern passes unchanged, for every positive `lambda`.
- `s_k>0`: the interior suppresses that pattern.
- Gains lie in `[0,1]`, so the mechanism never amplifies a mode.
- Decreasing `lambda` increases the influence of the internal constraints.

For the selective four-ring module with an unterminated last ring,
`s_k=0` exactly for the four real target coordinates (modes 1 and 8), while
the remaining eigenvalues lie between `8/13=0.6153846` and one. The minimum
is the DC coordinate. This spectrum is derived from the load recurrence in
the preceding study, not fitted to noisy samples.

Consequently this module leaves every clean combination of the two target
waves unchanged and attenuates every other pattern. As `lambda -> 0+`, its
surface response approaches the orthogonal projection onto the four target
coordinates. At `lambda=0` the target amplitudes are undetermined; the
limit is not a unique unforced equilibrium.

## 3. Exact error for the actual nuisance distribution

Let `F` contain the four orthonormal target vectors and write `s=Fa`.
Use an ensemble with `E[aa^T]=4I`, giving clean signal power `1/2` per unit.
The earlier single-family phase ensembles also give the same formulas below
when the two target modes have equal gain. Nuisance is independent and
uniform on the radius-`4 alpha` sphere in 32 dimensions, with exact covariance
`alpha² I/2`. Add independent readout noise with variance `sigma²` per unit
after settling.

For `T=(S+lambda I)^{-1}lambda`,

```
MSE = 4/32 ||(T-I)F||_F² + alpha²/64 ||T||_F² + sigma².
```

The first term is signal distortion; the second is residual nuisance; the
third is the fixed measurement floor. This identity needs only second
moments, so it is exact for the spherical nuisance, not a Gaussian reference.
There is no classifier training or class-mean estimation.

In the unterminated selective design, the distortion term is zero. Define
the retained nuisance fraction

```
R(lambda) = 1/32 sum_k g_k²
          = 1/8 + 1/32 sum_{k outside target} [lambda/(lambda+s_k)]².

NMSE = MSE/(1/2) = alpha² R(lambda) + 2 sigma².
```

Each non-target summand increases with `lambda`; thus weaker coupling
monotonically improves this particular equilibrium denoising objective.
The input nuisance and in-band signal are treated identically, so the
four target coordinates retain all their nuisance. That sets the floor
`R -> 4/32=1/8` for linear maps that leave the whole target subspace unchanged.

This is not a universal lower bound for all estimators. Allowing signal
shrinkage can lower average risk under a specified amplitude prior. For the
coefficient covariance above and `alpha=1`, the best linear mean-square
estimator projects onto the target subspace and multiplies it by `8/9`.
Its NMSE is 0.13111 with `sigma=0.1`, compared with 0.145 for exact unbiased
projection. It gives up exact clean-pattern preservation. Report that
tradeoff explicitly rather than claiming the proposed module is optimal.

## 4. Numerical predictions and a coefficient chosen analytically

For the unterminated selective module, `L=4`, `alpha=1`, `sigma=0.1`:

| External tie lambda | Signal gain | Nuisance fraction R | Surface NMSE | Surface step t95 |
| --- | --- | --- | --- | --- |
| 10 | 1 | 0.85173 | 0.87173 | 1.21 |
| 3 | 1 | 0.62433 | 0.64433 | 7.50 |
| 1 | 1 | 0.35084 | 0.37084 | 17.86 |
| 0.3 | 1 | 0.17428 | 0.19428 | 46.73 |
| **0.1** | **1** | **0.13277** | **0.15277** | **126.77** |
| 0.03 | 1 | 0.12580 | 0.14580 | 406.41 |
| Projection limit, lambda -> 0+ | 1 | 0.125 | 0.145 | Diverges |

`t95` is defined in section 6; it is in model-time units, not seconds or
optimizer iterations. Copying the noisy input gives NMSE 1 before readout
noise or 1.02 with the same readout noise. At `lambda=0.1`, the predicted
NMSE at `alpha=0.5` is 0.05319; with no nuisance it is 0.02, entirely due
to readout noise. The underlying clean surface state is unchanged.

There is no need to choose `lambda` by noisy trial-and-error. Let
`s_min=8/13`. A sufficient bound is

```
R <= [4 + 28 (lambda/(lambda+s_min))²]/32.
```

For a desired nuisance fraction at most 0.15, require

```
epsilon = sqrt((32*0.15-4)/28)
lambda <= epsilon*s_min/(1-epsilon) = 0.1251779.
```

The round value **lambda=0.1** therefore satisfies a worst-mode analytical
bound before any sampled experiment. It is a design choice balancing a
noise target and settling time, not a claim of a universally optimal value.
Smaller coupling yields progressively less additional noise improvement
as it approaches the in-band floor.

## 5. Why weakening the tie is not enough in the current disk

The original scalar-centre termination gives target stiffness `s_k=1/4`
at depth 4. Its signal gain is therefore

```
g_target = lambda/(lambda+1/4).
```

At `lambda=1` the target gain is 0.8, and at `lambda=0.1` it is only 2/7.
Weakening the external tie suppresses the desired pattern too.

| Original scalar-centre disk | Signal gain | Surface NMSE, alpha=1, sigma=0.1 |
| --- | --- | --- |
| lambda=1 | 0.8 | 0.36584 |
| lambda=0.3 | 0.54545 | 0.31308 |
| lambda=0.1 | 0.28571 | 0.54818 |

The original disk can trade bias against noise, but cannot preserve the
target subspace exactly while rejecting other directions through arbitrarily
weak coupling. The terminal condition and lateral operator must be analyzed
together. None of these proposed changes should be made inside the already
frozen experiment.

## 6. Settling speed is part of the analytical result

Assume unit-rate continuous gradient dynamics in the full physical state:

```
du/dt = -H_lambda u + lambda P_surface h.
```

Here `H_lambda` includes the external tie. For a step input and a zero start,

```
u(t) = (I-exp(-H_lambda t)) u*.
```

The slowest state decay time is `1/min_eigenvalue(H_lambda)`. This is not
necessarily the same as the time for a particular observable to settle.

In a target mode of the unterminated module, the ring chain has stiffness

```
H_target = tridiag(-1, [1+lambda, 2, ..., 2, 1], -1).
```

For a unit step, diagonalize `H_target=V diag(d_i) V^T`. The surface error
relative to its final unit response is exactly

```
1-x_1(t) = sum_i w_i exp(-d_i t)
w_i = lambda V_{1i}²/d_i,     w_i>=0,     sum_i w_i=1.
```

The table's `t95` is the time this residual first reaches 0.05. It measures
the target-pattern surface response, not all interior states or a noisy
transient. As `lambda -> 0+`, the smallest eigenvalue is approximately
`lambda/L`, so the slow time constant grows like `L/lambda`. Small forcing
still selects the correct equilibrium amplitude, but needs time to do so.

This tradeoff assumes fixed internal spring strengths and unit mobility.
Multiplying the entire energy by a constant preserves the equilibrium and
speeds continuous gradient dynamics by that constant, at the cost of larger
forces/rates. The times are not an architecture-independent speed limit.
The high-order lateral penalty also remains stiff, which limits naive
explicit time-stepping. A direct linear solve is not evidence of fast
physical settling.

## 7. Two limits that the ideal equilibrium model leaves out

**Imperfect pattern preservation.** If a desired mode has a small residual
surface stiffness `delta>0`, its gain is `lambda/(lambda+delta)`. To retain
at least 95% of its amplitude requires `lambda>=19 delta`. At `lambda=0.1`
this means `delta<=0.005263`. Excessively weak coupling makes a design more
sensitive to errors that shift its zero eigenvalues. This bound assumes
the perturbation preserves the eigenvectors; general mode-mixing errors
require an operator perturbation analysis.

**Noise during internal dynamics.** The present predictions include static
input nuisance and post-settling readout noise, not continually injected
internal noise. Under the additional model

```
du = -H_lambda(u-u*)dt + sqrt(2D)dW,
```

with independent equal-strength noise on every state coordinate, stationary
covariance is `D H_lambda^{-1}`. Its surface block is
`D(S+lambda I)^{-1}`. The four target coordinates alone contribute surface
variance `D/(8 lambda)` per unit, or NMSE `D/(4 lambda)`. Thus weakening the
tie indefinitely is not beneficial in that extended model. This is a
conditional analytical result, not a claim that such internal noise was
present in the previous experiments.

## 8. Verification and a focused experiment proposal

Reproduce all table entries with

```
.venv/bin/python -m experiments.full_surface_analysis
```

The script writes [full-surface-analysis.json](../reports/full-surface-analysis.json).
It compares three independently evaluated transfers: the full spatial
solve, the Schur-complement surface solve, and the Fourier gains. It checks
agreement within `1e-9`, zero clean bias for the unterminated candidate,
and the positive exponential weights for the settling calculation. It
uses no random draws or outputs from the running experiment.

A subsequent full-surface experiment can use a single analytically chosen
candidate: the unterminated selective module at `L=4`, `lambda=0.1`.
Measure the actual output surface, with all units softly influenced, using
continuous mixtures of both target modes and the declared nuisance model.

The primary predictions are unit clean gain and surface NMSE 0.15277 at
`alpha=1`, `sigma=0.1`. Include a copy baseline, the current scalar-centre
disk, direct orthogonal projection, and prior-dependent linear mean-square
estimation as distinct controls. A lambda=1 comparison isolates coupling
strength. Report actual transient response separately if physical settling
is part of the claim. Sample sizes, thresholds and seeds should be fixed
in a new contract before sampling.

An off-subspace clean input is also necessary as a diagnostic: the module
will suppress it even if an external evaluator considers it meaningful.
That exposes the supplied relevance assumption. A low error on the allowed
patterns demonstrates consistency with that assumption, not discovery of
what should count as relevant.

The central analytical conclusion is that a fully observed shared surface
can correct its inputs, provided the interior leaves desired patterns
unpenalized and the external coupling allows movement. The resulting
equilibrium accuracy, ambiguity-free response for lambda>0, settling time,
and limitations can all be calculated before running an experiment.
