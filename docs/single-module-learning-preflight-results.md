# Learning preflight — locality helps aligned learning, but the linear loop is equivalent to its FF control

**Completion:** complete, 2026-10-24. This is a development preflight, not a
neural benchmark. Only exact linear checks and finite-pair spectral estimates
were run. No neural suite, PyTorch installation, nonlinear energy, or local
plasticity rule was implemented.

[Specification](single-module-learning-preflight.md) ·
[manifest](../reports/single-module-learning-preflight-manifest.json) ·
[results](../reports/single-module-learning-preflight-results.json) ·
[runtime](../reports/single-module-learning-preflight-runtime.json).

## 1. Main findings

1. **The current learned linear hierarchy is not a distinct learning
   architecture under the matched objective and parameterization.** Its
   complete equilibrium map equals an explicit feed-forward map. Exact
   gradients of the same differentiable objective therefore agree too.
2. **The locality prior supplies a real finite-sample benefit when correct.**
   With nuisance variance 0.25, the local spectral learner first reaches
   mean subspace loss <=0.10 at 16 pairs, versus 64 for the unrestricted
   spectral learner on the declared budget grid.
3. **The same prior creates large irreducible errors when wrong.** A
   four-site shift gives mean population loss 0.3568; a dense rotation gives
   0.5330. The unrestricted population loss is zero in all three regimes.
4. **The hierarchy and local compact model have the same allowed coarse
   subspaces.** The observed benefit belongs to locality, not to hierarchy,
   a shared surface, or reciprocal settling.

These results support a narrow quick-learning claim for a correctly supplied
locality prior. They do not demonstrate an improvement over established
neural architectures or resilience to unknown changes in locality.

## 2. Exact equivalence and its implications

Each normalized row of the compact restriction `C` occupies one eight-site
group. Splitting it into two four-site segments, normalizing each segment,
and passing the two lengths to `B` produces `C=BA` exactly. The hierarchy
and compact local representation each have 28 continuous degrees of freedom,
versus 112 for an unrestricted four-dimensional subspace of R^32.

The zero-segment case is handled explicitly: its corresponding `B` coefficient
is zero, and its normalized `A` row can be any fixed unit vector. There is
no representational advantage hidden in that degeneracy.

For arbitrary learned `A,B`, the full spatial solve and explicit surface,
middle, and core formulas agree. If both use the same parameterization and
loss, equality as functions implies equality of exact parameter gradients.
Different optimizers, parameterizations, approximate solves or stopped
gradients can change learning, but those are additional algorithmic choices.
They cannot be attributed to the existence of an equilibrium loop alone.

Merely retaining the first view's core does not automatically defeat this
equivalence. With fixed linear weights, `p=C h1` and the second view's core
has the form `a C h2+(1-a)C h1`, for a scalar determined by coupling strengths.
A feed-forward comparator given both views can implement exactly that map.
A proposed local learning rule must specify an actual update difference,
and the comparison must give the baseline the same information.

## 3. What was measured

The run used 64 independent task worlds, each with eight independent training
streams. Unknown templates are smooth within eight-site blocks. Aligned,
shifted and dense versions rotate both the signal and nuisance, preserving
the unrestricted statistical problem while changing its compatibility with
fixed groups. Two views share latent coefficients but have independent
Gaussian nuisance with variance 0.25 or 1 per coordinate.

Pair budgets are nested prefixes `{4,8,16,32,64,128,256}`. The two spectral
estimators maximize the same symmetric cross-covariance objective, globally
or one direction per local group. Fixed averaging is an unfitted control.
The population local optimum is computed separately from ground truth and
is never supplied to an estimator.

Loss is

```
L = 1 - ||V^T U||_F²/4.
```

It is the fraction of clean signal power lost on independent new latent
combinations. Zero is perfect recovery; one loses all four signal directions.
Uncertainty averages the eight streams within a world first, then uses the
64 world means. Intervals below are mean +/- 1.96 standard errors, an
approximation rather than a guaranteed finite-sample confidence bound.

## 4. Learning curves when locality is aligned

Mean subspace loss:

| Pairs | Global, noise 0.25 | Local, noise 0.25 | Global, noise 1 | Local, noise 1 |
| --- | --- | --- | --- | --- |
| 4 | 0.5816 | 0.3056 | 0.7478 | 0.6166 |
| 8 | 0.4433 | 0.1664 | 0.6878 | 0.5057 |
| 16 | 0.2803 | **0.0751** | 0.6085 | 0.3606 |
| 32 | 0.1446 | 0.0352 | 0.5013 | 0.2152 |
| 64 | **0.0706** | 0.0175 | 0.3475 | 0.1104 |
| 128 | 0.0348 | 0.0087 | 0.1969 | **0.0551** |
| 256 | 0.0172 | 0.0044 | 0.1041 | 0.0279 |

For noise 0.25 at 16 pairs, the local mean interval is [0.0726,0.0776],
versus [0.2749,0.2857] globally. At noise 1 the local mean reaches the
0.10 threshold at 128 pairs; the global mean remains 0.1041 at 256 pairs,
with interval [0.1027,0.1054]. Do not extrapolate beyond the declared grid.

Fixed averaging has mean loss 0.6546 at every budget. Learned local spectral
directions improve substantially because the unknown broad templates are
not restricted to constant block means.

The 16-versus-64 crossing is a fourfold difference on this particular grid,
not a universal fourfold learning-speed theorem. All methods have the same
paired observations. The local estimator additionally has a correctly
specified support prior.

The noiseless case makes that information advantage especially clear:
one nondegenerate observation identifies each of four local one-dimensional
patterns within its known block. Without those supports, at least four
independent spanning observations are needed to identify an arbitrary
four-dimensional signal subspace. Neither observation requires a neural loop.

## 5. The cost of misaligned locality

| Regime | Population local loss floor | Approximate interval over worlds | Global population floor |
| --- | --- | --- | --- |
| Aligned | 0 | 0 analytically | 0 |
| Shifted four sites | **0.3568** | [0.3416,0.3720] | 0 |
| Dense rotation | **0.5330** | [0.5247,0.5414] | 0 |

These are representational errors, not optimization failures or finite-data
noise. A shifted support can put two independent signal components inside
one fixed block, while the local learner is allowed only one direction there.

Finite-pair results at noise variance 0.25:

| Regime | Global loss, 64 pairs | Local loss, 64 pairs | Global loss, 256 pairs | Local loss, 256 pairs |
| --- | --- | --- | --- | --- |
| Aligned | 0.0706 | 0.0175 | 0.0172 | 0.0044 |
| Shifted | 0.0706 | 0.3948 | 0.0172 | 0.3690 |
| Dense | 0.0706 | 0.5822 | 0.0172 | 0.5484 |

Global losses are equal by joint rotational invariance, verified numerically.
The aligned/shifted difference therefore cannot be explained by changing the
global signal-to-noise ratio. At 256 pairs, shifted local loss is already
close to its population floor; gathering much more data will not solve it.

The dense regime is intentionally a severe nonlocal stress test. It does
not establish failure on every real-world broad pattern. The four-site
shift is the more direct warning about fixed group boundaries.

## 6. Analytical finite-sample reference and new-input error

For true covariance `S=UU^T` and `T=S+nu I`, Gaussian fourth moments give

```
E||M_N-S||_F² = [(tr T)²+tr(T²)+(tr S)²+tr(S²)]/(2N)
              = (20+132nu+528nu²)/N.
```

The aligned local estimator only uses four diagonal covariance blocks. Their
summed expected error is `(8+36nu+144nu²)/N`. At noise 0.25 the coefficients
are 86 globally and 26 locally; at noise 1 they are 680 and 188. This is
one explicit source of the variance reduction. Subspace error also depends
on the eigenvalue gap, so it is not numerically equal to covariance error.

Across every reported budget/noise setting, empirical covariance-error means
are between 0.9812 and 1.0212 times their exact predictions. These are
diagnostics, not tuned pass/fail thresholds. Aligned local population gaps
are one. Shifted local gaps can be small (minimum 0.00279 among the declared
worlds), adding estimation difficulty on top of the approximation floor.

For a fitted rank-four orthogonal projector and an independent new input,
the exact relative reconstruction error is `L+nu_test`. Results record this
at the training nuisance level and at four times that level. No held-out
sample was needed to compute this expectation. In this isotropic linear
model, the increased test noise adds the same term to all rank-four
projectors: there is no extra shift-resilience advantage beyond learning
the subspace better. This limitation should not be hidden by a favourable
plot of only one architecture.

## 7. Checks and provenance

| Check | Result |
| --- | --- |
| Compact/hierarchical factorization, including zero segments | max error 0 |
| Full spatial vs explicit complete-state map | 1.33e-15 |
| Finite-difference gradients of same parameterized loss | 8.33e-12 |
| Noiseless local recovery from one pair | loss 0 |
| Noiseless global recovery from four spanning pairs | loss 0 |
| Future unused rows have no effect on a fitted prefix | exact |
| Joint-rotation invariance of global loss | max difference 3.22e-15 |
| Any local fitted loss below its population optimum | no violations |
| Dedicated unit tests | 6 passed |
| Regeneration in a fresh temporary directory | manifest/results byte-identical |

The specification and source hashes were recorded before finite-sample
training streams were generated. Runtime is separate (about 24 seconds for
the initial run, one BLAS thread). The data are development data and will
not be reused for neural confirmation. No original selectivity artifacts
or dependencies were changed.

Reproduce with:

```
.venv/bin/python -m unittest tests.test_learning_preflight -v
.venv/bin/python -m experiments.learning_preflight
```

## 7a. Supplementary checks: asymptotics, pairing, resilience, identifiability

A second, independently written development script covers questions the
main preflight does not. It uses its own generator (random within-block
profiles rather than DCT templates, unequal signal variances `(8,6,4,2)`,
200 independent tasks per cell, budgets `{4,...,128}`), so its numbers are
not interchangeable with the tables above. It uses a remap-invariant
population metric: the fraction of broad variance linearly decodable from
the four-dimensional code, `R2`. It is zero for a collapsed code.

**First-order sample-complexity ratio.** For Gaussian `c` and isotropic
Gaussian detail of variance `s`, the symmetrized cross-covariance estimator
has

```
E||sin Theta||_F² ≈ (1/n) sum_i k (l_i+s) s / (2 l_i²),
```

with `k=28` orthogonal directions globally and `k=7` per aligned block.
To first order the locality prior is therefore worth exactly a factor
`28/7=4` in pairs. Monte Carlo agrees within about 5% at `n>=64` for
`s<=1`; at small `n` or `s=4` the error saturates and the formula is
not informative. Pairs needed to reach 90% of the dense ceiling, aligned
random profiles: 11.6 vs <=4 (`s=1`) and 96 vs 26 (`s=4`), global vs local.
This is consistent with the 16-versus-64 crossing above and explains it.

**Pairing matters only with structured nuisance.** With isotropic detail,
PCA of pooled views (ignoring pair identity) is within 0.03 `R2` of the
cross-covariance estimator at every budget. When training nuisance has
four extra directions of variance 12 orthogonal to the signal, pooled PCA
collapses to `R2≈0.06` at 128 pairs, while cross-covariance still learns
(global 0.74, local 0.83, ceiling 0.84). **A benchmark using isotropic
detail does not test invariance learning from pairs**; the neural
benchmark must include structured, view-independent nuisance.

**New detail and changed nuisance.** At 32 pairs, `s=1`, aligned tasks:
adding detail variance `4s` only orthogonal to the signal lowers `R2` from
0.835 to 0.817 for the local learner and from 0.817 to 0.758 globally.
Locality limits leakage into new detail. Fixed averaging is exactly
invariant to within-block detail, but only useful when templates are
constant. Under four-times isotropic test noise, every estimator falls by
the same population term, matching section 6.

**Held-out combinations are uninformative for linear models.** A linear
code treats every coefficient combination identically. That test can only
separate nonlinear models.

**Persistent distractor.** Adding a two-dimensional component shared by
both views lowers the population cross-covariance solution's `R2` from 0.84
to 0.71 (variance 4) and 0.61 (variance 10). When distractor and signal
have matched statistics, swapping them leaves the joint view distribution
unchanged. No pair-only method can tell them apart; this is an
identifiability limit, not an estimator weakness.

**Benchmark feasibility.** VICReg's per-branch centred covariance has rank
at most `n-1`, so with 4 pairs it cannot whiten a four-dimensional code.
Held-out top-1 retrieval among 64 candidates, whitened with training views
only, peaks near 0.31 at `s=1` even for the oracle-quality estimators. It
is a coarse, low-ceiling metric. A linear probe with counted labels, or
subspace/`R2` diagnostics where defined, discriminates better.

Equivalence checks in this script agree with section 7: hierarchy/compact
factorization error 2e-16, equilibrium vs feed-forward map 4e-12 (random
non-orthonormal `A,B`, random `lambda,kappa`), gradient through the
equilibrium vs chain rule 4e-8 (finite differences). With context
`eta=1`, the map departs from feed-forward (core moves 2.9 from `Ch`).

Reproduce (about 5 minutes, one BLAS thread):

```
.venv/bin/python -m experiments.learning_preflight_supplement
```

Output: [supplement results](../reports/single-module-learning-preflight-supplement.json).
Development data only; not reused for confirmation.

## 8. Candidate decision before neural implementation

**Do not proceed as written with the unchanged linear equilibrium candidate
as a purportedly distinct neural architecture.** The matched FF control
ties it as a function, including exact gradients under the same loss and
parameterization. No full neural suite is justified merely to rediscover
that identity.

**Keep the compact local spectral learner as the mandatory sample-efficiency
reference.** Its aligned learning advantage is substantial and explainable.
Any neural candidate should at least reproduce that benefit under the same
prior, rather than claim success only against fixed averaging or a poorly
optimized dense network.

**For the next candidate-design study, prefer the retained-context/local-rule
route over adding nonlinearity solely to break equivalence.** That route
addresses the stated goal of reusing broad structure across related
observations. This is a research-priority choice, not evidence that such a
rule will outperform its baselines. Specify the plasticity update,
anti-collapse constraint, state retention, and differentiation conventions
before implementing it; give the controls the same history and supervision.
Retaining a linear first-view context alone is still FF-reproducible, as
section 2 shows.

The candidate must also confront the locality failure: use an explicit
adaptive/overlapping grouping mechanism, or scope its claim to aligned
local structure and keep the shifted cases as failures. The present data
do not choose or validate an adaptive mechanism. Do not silently replace
the difficult cases with aligned ones.

Thus the concrete decisions are: retain the local spectral/compact reference;
reject an unchanged linear loop as the distinct candidate; prioritize a
specified retained-context learning mechanism next; and defer neural
implementation until that mechanism has an analytically identifiable
difference. No learned superiority claim is made by this preflight.
