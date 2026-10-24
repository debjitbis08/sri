# Layered selectivity v1 — the band-selective disk meets T1–T4 in 20 of 20 primary replicates; broadband attenuation passes the old profile but never T4

**Completion:** `complete`. Every correctness gate passed, all 320
(configuration, replicate) units are eligible (all 128,000 canonical
settlings certified), and regeneration is byte-identical.

**Headline:**

- **Primary outcome: `success`.** The selective arm at `L=4`, `alpha=1`
  meets T1–T4 in both families in **20 of 20** replicates (18 required).
  The two-sided Clopper–Pearson 95% interval for the replicate success
  proportion is [0.832, 1]. No replicate fails any criterion.
- **Selectivity is measured, not only suppression.** Primary `J_n` lies in
  0.246–0.287 (F-low) and 0.253–0.279 (F-high), around the population value
  0.2653. Interior nuisance power relative to signal power falls about 3.7
  times from the exterior inward.
- **Broadband attenuation passes the old profile without selectivity.** At
  `L=4` the no-lateral control meets T1–T3 in 20 of 20 replicates at both
  `alpha` and fails T4 in every one (`J_n` 1.52–1.84). Passing T1–T3 alone
  therefore does not demonstrate selective preservation, as the design
  analysis argued.
- **The relevant band must be known.** The wrong-band control behaves like
  the selective arm for F-low, including T4, and decodes F-high at chance
  (means 0.24–0.26, lowest 0.18) at both depths.
- **Ordinary smoothing** fails T4 in every replicate and family. At `L=4` it
  also loses F-high (mean accuracy 0.28–0.29).
- Every selective configuration (`L` in {2, 4}, `alpha` in {0.5, 1})
  succeeds in 20 of 20. These are secondary comparisons.

This supports the existence of a stable, finite-range, task-informed filter
that preserves both declared bands while rejecting irrelevant variation. It
does not show that relevance can be discovered without supervision, that the
construction tolerates other bands or coefficient errors, or anything about
learning, abstraction, temporal retention or coupled modules. No Stage 2
configuration is selected.

[Contract](layered-selectivity-v1-contract.md) (sha256 `9abd7ed8…`) ·
[manifest](../reports/layered-selectivity-v1-manifest.json) (sha256 `f4ca391c…`) ·
[results](../reports/layered-selectivity-v1-results.json) ·
[runtime](../reports/layered-selectivity-v1-runtime.json) ·
[design analysis](layered-module-design-analysis.md).

**Provenance.**

- The contract was revised after review against the implementation, before
  any confirmation draw. The revisions are listed at its top. Thresholds,
  factors, seeds, the primary configuration and the success rule were not
  changed.
- Before freezing, the pipeline was smoke-tested on one replicate per
  configuration with the non-declared seed `1`. No confirmation seed
  (`[20261030, 0, r]`, `[20261030, 1, r, l, family]`) was used before the
  freeze.
- The contract, manifest, code and tests were committed (`b309d41`) before
  the run. The run was made once, on 8 worker processes with one BLAS thread
  each, in 1 h 52 min.
- A second run in a separate directory produced byte-identical manifest and
  results files.

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Contract, code and prior-artifact hashes; pinned environment (numpy 2.5.4, scipy 1.18.1, one BLAS thread) | yes |
| Hessian exactly symmetric and positive definite, every arm and depth | yes |
| Lateral coupling distance (band arms 4, smoothing 1, none 0); radial and centre structure | yes |
| Analytic gradient / Hessian against central differences, scaled discrepancy | 2.3e−9 / 5.3e−10 (limit 1e−5) |
| Term-by-term energy against the assembled quadratic form | 4.4e−16 relative |
| Spatial direct-solve transfer against tridiagonal Fourier gains; centre = ring-`L` mean | 3.5e−12 |
| Gains in [0, 1], nonincreasing inward | yes |
| Pure-signal responses equal declared gains | 3.3e−11 (limit 1e−9) |
| v1 `nearest_mean` against an independent implementation, including a tie | exact |
| Every canonical state within 1e−7 of the Cholesky direct solve | max 2.6e−11 |
| All canonical settlings certified | 128,000 / 128,000 |
| Byte-identical regeneration (manifest, results) | yes |
| Tests | 17 new, all passing |

## Results

### Primary configuration (selective, `L=4`, `alpha=1`)

| Family | Strict accuracy, mean (min) | `P(L)/P(1)` range | `J_n` range | Failures |
| --- | --- | --- | --- | --- |
| F-low | 0.9995 (0.99) | 0.0154–0.0179 | 0.246–0.287 | none |
| F-high | 1.000 (1.00) | 0.0158–0.0174 | 0.253–0.279 | none |

Population predictions (contract section 5): suppression 0.01658, `J_n`
0.2653, Gaussian-reference accuracy 0.9989, spherical known-mean lower bound
0.9903. The fitted-NM minimum, 0.99 (one error in 100 test drives),
is just below that known-mean bound; the bound does not cover fitted means. Pure-signal
gains are exactly 0.8, 0.6, 0.4, 0.2 on successive rings (`Vsig` 0.32,
0.18, 0.08, 0.02).

Relative to the input, ring powers in replicate 0, F-high, are:

| Ring | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| Nuisance `P(l)` / input | 0.307 | 0.047 | 0.020 | 0.0051 |
| Signal `Vsig(l)` / input | 0.64 | 0.36 | 0.16 | 0.04 |

The exterior itself already passes signal about twice as well as nuisance,
because the tie couples it to an interior that rejects out-of-band modes.
`J_n` is measured from the exterior, so this filtering is not counted in it.

### Every configuration

Counts are out of 20 replicates. "Profile" is T1–T3 in both families.
Accuracy is the mean strict NM accuracy at layer `L`.

| Configuration | Profile | Success | Failures by criterion | Accuracy F-low / F-high | `J_n` range F-low; F-high |
| --- | --- | --- | --- | --- | --- |
| selective `L=2` `alpha=0.5` | 20 | 20 | — | 1.000 / 1.000 | 0.225–0.267; 0.234–0.259 |
| selective `L=2` `alpha=1` | 20 | 20 | — | 0.9995 / 0.9995 | 0.225–0.267; 0.234–0.259 |
| selective `L=4` `alpha=0.5` | 20 | 20 | — | 1.000 / 1.000 | 0.246–0.287; 0.253–0.279 |
| **selective `L=4` `alpha=1`** | **20** | **20** | — | 0.9995 / 1.000 | 0.246–0.287; 0.253–0.279 |
| no lateral `L=2` `alpha=0.5` | 0 | 0 | T2 20+20, T4 20+20 | 1.000 / 1.000 | 1.16–1.23; 1.15–1.24 |
| no lateral `L=2` `alpha=1` | 0 | 0 | T2 20+20, T4 20+20 | 0.9995 / 0.998 | 1.16–1.23; 1.15–1.24 |
| no lateral `L=4` `alpha=0.5` | 20 | 0 | T4 20+20 | 1.000 / 0.9995 | 1.58–1.82; 1.52–1.84 |
| no lateral `L=4` `alpha=1` | 20 | 0 | T4 20+20 | 0.982 / 0.9905 | 1.58–1.82; 1.52–1.84 |
| ordinary smoothing `L=2` `alpha=0.5` | 20 | 0 | T4 20+20 | 1.000 / 0.997 | 0.68–0.78; 2.50–3.04 |
| ordinary smoothing `L=2` `alpha=1` | 16 | 0 | F-high T3 4; T4 20+20 | 0.9995 / 0.9455 | 0.68–0.78; 2.50–3.04 |
| ordinary smoothing `L=4` `alpha=0.5` | 0 | 0 | F-high T3 20; T4 20+20 | 1.000 / 0.292 | 1.43–1.91; 212–318 |
| ordinary smoothing `L=4` `alpha=1` | 0 | 0 | F-low T3 1, F-high T3 20; T4 20+20 | 0.960 / 0.282 | 1.43–1.91; 212–318 |
| wrong bands `L=2` `alpha=0.5` | 0 | 0 | F-high T3 20, T4 20 | 1.000 / 0.253 | 0.233–0.264; 1.3e7–1.5e7 |
| wrong bands `L=2` `alpha=1` | 0 | 0 | F-high T3 20, T4 20 | 0.9995 / 0.256 | 0.233–0.264; 1.3e7–1.5e7 |
| wrong bands `L=4` `alpha=0.5` | 0 | 0 | F-high T3 20, T4 20 | 1.000 / 0.2455 | 0.245–0.278; unavailable |
| wrong bands `L=4` `alpha=1` | 0 | 0 | F-high T3 20, T4 20 | 0.9995 / 0.242 | 0.245–0.278; unavailable |

"T4 20+20" means 20 failures in each family. T1 held in every replicate of
every configuration. For wrong bands at `L=4`, F-high `J_8` is unavailable
in all 20 replicates because `Vsig(L)` is below the `1e-12` floor
(population gain `1.4e-13`); per the contract this counts as T4 not met.

### Against the population predictions

Every predicted pass/fail pattern in contract section 5 held, with two
finite-sample departures, reported as measured:

- **Ordinary smoothing, `L=2`, `alpha=1`, F-high:** mean accuracy 0.9455
  (min 0.83) against a Gaussian reference of 0.9955. T3 failed in 4 of 20
  replicates. A likely contributor, not tested here, is that the fitted
  32-dimensional class means carry training nuisance that the known-mean
  reference does not, which matters more when the target mode is attenuated
  more than the low modes (gain ratio 0.25 for mode 8 versus 0.49 for mode 1).
- **Ordinary smoothing, `L=4`, `alpha=1`, F-low:** T3 failed in 1 of 20
  replicates (min accuracy 0.88; reference 0.9982).

The no-lateral control at `L=4`, `alpha=1` likewise averages below its
reference (0.982 / 0.9905 against 0.9989) but always passes T3. The selective
arm stays at or above 0.99 in every replicate.

### Matched controls (selective minus control, layer `L`)

Mean differences over the 20 paired replicates; the selective arm's accuracy
was never below a control's in any replicate.

| Control | `L` | `alpha` | Accuracy F-low / F-high | `J_n` F-low / F-high |
| --- | --- | --- | --- | --- |
| no lateral | 2 | 0.5 | 0.000 / 0.000 | −0.95 / −0.95 |
| no lateral | 2 | 1 | 0.000 / 0.0015 | −0.95 / −0.95 |
| no lateral | 4 | 0.5 | 0.000 / 0.0005 | −1.45 / −1.41 |
| no lateral | 4 | 1 | 0.0175 / 0.0095 | −1.45 / −1.41 |
| ordinary smoothing | 2 | 0.5 | 0.000 / 0.003 | −0.49 / −2.53 |
| ordinary smoothing | 2 | 1 | 0.000 / 0.054 | −0.49 / −2.53 |
| ordinary smoothing | 4 | 0.5 | 0.000 / 0.708 | −1.44 / −264 |
| ordinary smoothing | 4 | 1 | 0.0395 / 0.718 | −1.44 / −264 |
| wrong bands | 2 | 0.5 | 0.000 / 0.747 | +0.001 / −1.4e7 |
| wrong bands | 2 | 1 | 0.000 / 0.7435 | +0.001 / −1.4e7 |
| wrong bands | 4 | 0.5 | 0.000 / 0.7545 | +0.006 / unavailable |
| wrong bands | 4 | 1 | 0.000 / 0.758 | +0.006 / unavailable |

On F-low the wrong-band arm matches the selective arm to within 0.006 in
`J_n`: both keep mode 1 in their nullspace. The difference between them is
confined to the band each one was told to keep.

### Settling

All settlings certified. The stiff arms cost iterations, not reliability:

| Arm | L-BFGS-B iterations, mean (max), `L=2` / `L=4` | Newton steps, max | Worst final gradient |
| --- | --- | --- | --- |
| selective | 488–503 (651) / 980–988 (1,223) | 4 | 1.0e−12 |
| no lateral | 15 (18) / 21 (25) | 1 | 1.0e−12 |
| ordinary smoothing | 29 (35) / 46–47 (57) | 1 | 3.6e−15 |
| wrong bands | 2,210–2,279 (3,069) / 4,944–5,033 (6,585) | 20 | 5.8e−11 |

In the wrong-band arm the Newton polish uses all 20 steps because rounding
at stiffness `3.6e5` keeps the gradient above the `1e-12` polish target. It
stays far below the `1e-8` certification threshold. By iteration counts, the wrong-band
arm took most of the compute.

## Interpretation (per contract section 7)

All four criteria pass for the primary configuration, so the declared
reading applies: a stable finite-range quadratic disk with a task-informed
lateral penalty preserves both declared signal bands while rejecting
irrelevant variation, under fixed absolute readout noise and an unweighted
nearest-mean readout on all 32 coordinates.

The controls separate three things that the original profile conflated:

1. **Attenuation alone** (no lateral penalty) passes T1–T3 at `L=4` but
   worsens nuisance relative to signal (`J_n` ≈ 1.7).
2. **Generic smoothing** (ordinary Laplacian) also worsens it and, at depth,
   erases the high band.
3. **Selectivity for the wrong band** helps only the band it keeps.

Limits, as declared: the relevant subspace is hand-specified; the penalty
has mixed-sign couplings out to four sites and is stiff (largest stiffness
about `1.1e4`, condition number about `2.8e4` at `L=4`); success does not
imply tolerance to frequency shifts or coefficient error; and nothing here
concerns learning, temporal retention or inter-module coupling. The next
analytical questions named in the contract are coefficient tolerance,
less-stiff implementations, and learning the relevant subspace. Moving to
coupled modules is not automatic.

This experiment stops here. No depth, noise, band or coefficient will be
extended after these results.
