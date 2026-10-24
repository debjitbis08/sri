# Single-module learning benchmark v1 — locality-prior coarse module vs established paired-view learners

**Status:** prospective, 2026-10-24. Development selection runs first on
declared development worlds. The manifest then records hashes of this
contract, the implementation, the selected configurations and the
environment. Only after that are confirmation worlds generated.

**Disclosed exploration before freezing.** Two smoke runs used non-declared
seed `1` (worlds 900–903 and 999), which are never reused. They led to three
changes before this contract was frozen. (1) Training is vectorized across
ensemble members because the per-model loop was too slow. (2) Ridge-CCA
references were added: under view-specific structured nuisance, the
cross-covariance spectral reference was weaker than learned VICReg, and the
analytical reference must be the strong one. (3) Isotropic nuisance was
lowered from 0.5 to 0.25 so the probe ceiling is not compressed near 0.67.
No confirmation world was generated.

Follows the [proposal](single-module-learning-benchmark-proposal.md) and
the [preflight](single-module-learning-preflight-results.md).

## 1. What this tests, and what it cannot test

The preflight proved that the settled linear coarse module (32→8→4,
summary coupling, no context) equals a feed-forward local encoder. Under
the same objective and parameterization its gradients are identical too.
Proposal §4.1 (architecture-controlled: reciprocal settling vs matched
feed-forward) is therefore **decided analytically as an exact tie** and is
not re-run. Nothing here tests reciprocal dynamics, retained context, or a
local plasticity rule.

This experiment is proposal §4.2, the **method comparison**. Can the learned
coarse module, whose distinctive content is its fixed-locality hierarchy,
learn broad structure from fewer pairs than established paired-view
learners, and keep it under prespecified shifts? And what does its
locality cost when the structure is misaligned? A win is attributable to
the locality prior, not to circularity or depth (preflight §2).

## 2. Task

Dimensions `D=32`, `M=4`, four eight-site blocks. Per world:

- **Broad templates `U`:** in block `k`, a random combination of DCT-II
  modes 0–2, normalized (smooth, unknown shape and sign).
- **Texture nuisance `V`:** in block `k`, a random combination of DCT-II
  modes 3–7, normalized. It is exactly orthogonal to `U`.
- **Rotations:** `aligned` (identity), `shifted` (circular shift by four
  sites), `dense` (random orthogonal from QR with positive `diag(R)`).
  Signal and nuisance rotate together, so the unrestricted statistical
  problem is identical across regimes.

A view is `h = R(U c + 2 V g + 0.5 e)`, with `g ~ N(0,I_4)` and
`e ~ N(0,I_32)` drawn independently per view. That is texture variance 4
and isotropic variance 0.25. The two views of a pair share only `c`.
Training `c ~ N(0,I_4)` conditioned on `c0 c1 > 0`, implemented by setting
`c1 = |c1| sign(c0)`.

Texture variance exceeds signal variance in every direction. Pooled PCA
would select nuisance. Learning the broad structure requires pair identity
(preflight §7a).

Pair budgets are nested prefixes `n ∈ {4,8,16,32,64,128,256}` of one
256-pair stream per (world, regime). There are no augmentations, epochs
count nothing, and every method receives exactly the same pairs.

## 3. Methods

| Arm | Role | Parameters |
| --- | --- | --- |
| `coarse_local_vicreg` | **Candidate**: learned coarse module, A (8 × 4-site), B (4 × 2) | 40 |
| `dense_linear_vicreg` | Same objective without locality (isolates the prior) | 128 |
| `mlp_vicreg` | Established paired-view learner, MLP 32→w→w→4, w ∈ {16,64} | 868 / 6532 |
| `conv_vicreg` | Established spatial bias: circular conv (k=5) + residual block, w ∈ {4,8} channels | 708 / 1732 |
| `dae_paired` | Paired-view denoising AE: encode one view, reconstruct the other, 32→w→4→w→32 | 1220 / 4772 |
| `local_cca`, `global_cca` | Analytical references: ridge CCA, per block or global | — |
| `local_spectral`, `global_spectral` | Cross-covariance eigenvectors (preflight estimators) | — |
| `fixed_averaging` | Disclosed fixed prior; no fit | — |

VICReg is applied directly to the 4-dim code with weights (25, 25, 1), std
floor 1 and no expander head. The DAE is labelled the paired-view adaptation
and never sees clean targets. CCA ridge is `0.1 ×` mean variance, fixed by
declaration rather than tuned. Code dimension is 4 for every method.

Exact parameter matching to the candidate's 40 parameters is not attempted
for nonlinear networks. The capacity-controlled pair is candidate vs
`dense_linear_vicreg`, and nonlinear arms get a two-point width range.

**Training.** Full-batch AdamW for 1500 steps, float32, one model per
(world, configuration), with the vectorized optimizer written out explicitly.

**Equal development resources.** Every learned arm gets exactly six
configurations:

- linear arms: lr ∈ {3e-3, 1e-2, 3e-2} × weight decay ∈ {0, 1e-2};
- nonlinear arms: lr ∈ {1e-3, 3e-3, 1e-2} × width, weight decay 1e-4.

## 4. Metric

**Linear-probe R².** Each trained encoder is frozen. A ridge linear probe
(ridge `1e-3·n_probe`, standardized features, bias) is fitted on 64 fresh
labelled examples of `(view, c)` from the in-distribution condition. These
labels are counted supervision and identical across methods. Then

```
R2 = 1 - sum ||c_hat - c||² / sum ||c||²
```

is computed on 1024 fresh test views per condition. A collapsed code
cannot score above about 0. R² is invariant to invertible remapping of
the code.

**Test conditions.** All are generated after the probe fit and never used
in training or selection:

| Condition | Change |
| --- | --- |
| `in_distribution` | as training |
| `heldout_combination` | `c0 c1 < 0`: coefficient-sign combinations never seen in training |
| `changed_nuisance` | texture variance ×4 (16) |
| `new_detail` | extra variance 2 in all directions orthogonal to both `U` and `V` |

## 5. Development selection

Development uses worlds 0–7 (seed `20261201`) in all three regimes. For each
arm and budget, select the configuration with the highest in-distribution
probe R², averaged over the development worlds and **pooled across
regimes**. The regime is not known to a practitioner. Selection uses only
development worlds. The selected table is written into the manifest before
confirmation.

## 6. Confirmation and statistics

Confirmation uses worlds 1000–1031, 32 independent worlds with the same
seed root, so there is no overlap with development. One training stream
and one initialization per (world, regime, arm, budget). The world is the
unit of replication.

- **Low-data score (per world, regime, arm):** mean in-distribution R²
  over the seven budgets. These are equally spaced in `log2 n`, so this is
  the normalized area under the curve.
- **Resilience score:** mean R² over the three shifted conditions at
  `n=32`. Each condition is also reported separately, with its drop from
  `in_distribution`.

**Primary hypotheses** (aligned regime only, candidate minus comparator, for
the four learned comparators: dense linear, MLP, conv and DAE):

- **H1 (low-data):** paired difference in low-data score.
- **H2 (resilience):** paired difference in resilience score.

For each family of four comparisons, use two-sided paired t tests over the
32 worlds with Holm correction at 0.05. The candidate is declared **better**
than a comparator when the Holm-adjusted p < 0.05 **and** the mean
difference ≥ 0.02. It is **worse** under the mirror condition. Otherwise
the result is **no practical difference**. A claim of "faster learning than
established architectures" needs **better** against all three established
neural arms: MLP, conv and DAE.

**Secondary results** (reported, with no claim from them):

- the same comparisons in the shifted and dense regimes, which measure the
  cost of locality;
- candidate vs `local_cca` and `global_cca`;
- full learning curves, with 95% intervals over worlds;
- the first budget where mean in-distribution R² ≥ 0.6, or "not reached by
  256";
- convergence: the fraction of members with non-finite loss or relative
  change > 5% over the last 10% of steps.

**Predictions recorded before confirmation**, from the preflight and the
disclosed smoke runs:

- the candidate is close to `local_cca` on aligned tasks and beats dense
  learners at small n;
- on shifted and dense tasks it falls below `global_cca` and dense learners
  at large n;
- held-out combinations do not separate linear methods.

## 7. Gates before confirmation

- The unit tests pass: equivalence of the coarse arm to its explicit
  `B A h` map, the orthogonality of `U` and `V`, regime rotations
  orthogonal, nested prefixes, held-out sign conditioning, probe R² near 0
  for a constant code and near the ceiling for the oracle code, and
  determinism of the vectorized trainer.
- Every development run produces finite losses. Any non-finite member is
  reported and scored as R² = 0.

## 8. Reproduction

```
.venv/bin/python -m unittest tests.test_learning_benchmark -v
.venv/bin/python -m experiments.learning_benchmark --phase all
```

Outputs:

- `reports/single-module-learning-benchmark-v1-development.json`
- `reports/single-module-learning-benchmark-v1-manifest.json`
- `reports/single-module-learning-benchmark-v1-results.json`
- `reports/single-module-learning-benchmark-v1-runtime.json`

Jobs run in parallel processes with one thread each. Results do not depend
on the number of workers.
