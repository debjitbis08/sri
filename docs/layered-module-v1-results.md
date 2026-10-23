# Layered module v1 — no configuration meets the profile: linear interiors calm down but lose fine distinctions; double-well interiors amplify incidental variation

**Completion:** `complete`. Every correctness gate passed, all 80
configurations are eligible (all 32,000 canonical settlings certified), and
regeneration is byte-identical.

**Headline:**

- **No configuration meets the target profile.** The declared candidate depth
  for Stage 2 is therefore `None`.
- **Linear (harmonic) arm.** Interiors are calmer, and low-frequency
  distinctions survive at moderate depth. High-frequency distinctions do not
  survive in any configuration. Ten linear configurations fail **only** on
  F-high kept distinctions (T3). This is reservation R1 in action.
- **Double-well arm.** It does the opposite of what was hoped. Interior
  incidental variance is 0.52–9.2 times the exterior's, and T2 fails in every
  double-well configuration. Responses are multistable (up to 6 distinct
  minima per probe drive), strongly non-additive (superposition defect
  0.29–2.0), and keep F-high distinctions no better than the linear arm. Here
  nonlinearity amplifies nuisance instead of protecting distinctions, so R2 is
  not answered positively by this potential.
- **Two causes of lost F-high distinctions, from diagnostics.** These are not
  primary results:
  - F-high separation collapses on rings of 21 or fewer units, in both arms.
    Rings of 16 or fewer cannot represent the mode-8 pattern at all.
  - On larger rings (24–27 units) the information is still present: the
    Mahalanobis separation is 4.7–6.2 at `σ₀ = 0.03`. The declared
    nearest-mean classifier cannot use it, because low-frequency nuisance
    dominates an unweighted distance.

[Contract](layered-module-v1-contract.md) (sha256 `7cb93213…`) ·
[manifest](../reports/layered-module-v1-manifest.json) ·
[results](../reports/layered-module-v1-results.json) ·
[runtime](../reports/layered-module-v1-runtime.json).

**Provenance.**

- The contract, manifest, code and tests were committed in `2d48ac5` before
  any scored computation.
- The first scored run crashed while writing its report: a numpy `int64` was
  not JSON-serializable. It produced no output file, and no scored values were
  inspected.
- The only change since is in the report-rounding helper, which now converts
  numpy scalars (`experiments/layered_module.py`, uncommitted). No measure,
  threshold or setting changed. The run was then repeated.

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Contract hash and pinned environment (numpy 2.5.4, scipy 1.18.1, one BLAS thread) | yes |
| Graph construction: tabulated ring sizes, connected, centre present, no duplicate edges, Laplacian rows sum to 0 | yes |
| Analytic gradient and Hessian against finite differences | max error 1.7e−7 |
| Linear arm: every canonical state equals the direct sparse solve | max deviation 3.8e−15 |
| Linear superposition (absolute) | yes |
| Attenuation sanity (`μ=0`, `γ=1`, `L=4`): interior RMS for `cos θ` vs `cos 8θ` | 0.169 vs 0.0155 |
| All canonical settlings certified | 32,000 / 32,000 |
| Byte-identical regeneration (manifest, results) | yes |
| Tests | 16 new; full suite 287 passing |

## Results

### Which targets fail

Counts are failing configurations per family. Each arm has 32 linear or 48
double-well configurations.

| Arm | Family | T1 sensitivity | T2 suppression | T3 distinctions |
| --- | --- | --- | --- | --- |
| Linear | F-low | 0 | 4 | 18 |
| Linear | F-high | 0 | 4 | **32 (all)** |
| Double well | F-low | 18 | **48 (all)** | 24 |
| Double well | F-high | 44 | **48 (all)** | **48 (all)** |

### Linear arm

- **Suppression.** The ratio `P(L)/P(1)` falls with depth: 0.37 at `L=2`, then
  0.25, 0.21 and 0.19 for `μ=0`. With `μ=1` it falls to 0.14 at `L=2` and 0
  (rounded) at `L=6`. T2 fails only for `μ=0` at `L=2`.
- **F-low.** Strict interior accuracy at `σ₀ = 0.1` is at least 0.97 for every
  `μ=0` configuration with `L ≤ 3`, and 1.00 for `μ=0`, `γ=1`, `L=4`. It
  degrades at `L = 6` and with `μ = 1` beyond `L = 2`.
- **F-high.** Interior accuracy is 0.22–0.47 (chance 0.25) at `σ₀ = 0.1`, and
  at most 0.63 at `σ₀ = 0.03`.
- **Configurations failing only on F-high T3:**
  - `μ=1` at `L=2`, both `γ`, both `α`;
  - `μ=0` at `L=3`, both `γ`, both `α`;
  - `μ=0`, `γ=1`, `L=4`, both `α`.
- **Multistability and superposition.** Always 1 minimum; superposition defect
  0, as expected.

### Double-well arm

- **Suppression.** Interior incidental variance exceeds the exterior's in most
  configurations, with ratios from 0.52 up to 9.2. Example: `β=1`, `γ=1`,
  `L=4`, F-low. Total variance `S` by layer is 0.66, 0.65, 0.57, 0.42 and
  nuisance variance `P` is 0.16, 0.22, 0.27, 0.31, so nuisance *grows* inward
  as units commit to one well or the other.
- **Kept distinctions.** F-low interior accuracy falls with depth, to 0.35 at
  `L=6`. F-high is near chance beyond the exterior. Even the exterior layer
  decodes F-high at only 0.46–0.77, against 1.00 in the linear arm.
- **Multistability.** 1–6 distinct minima per probe; responses depend on the
  start.

### Kept-distinction diagnostic (Mahalanobis, `σ₀ = 0.03`, `γ=1`, `α=1`)

Values are listed by layer, exterior first.

| Arm, `L` | F-high | F-low |
| --- | --- | --- |
| linear `μ=0`, `L=4` (rings 32/24/16/8) | 9.2, 4.7, 0.9, 0.7 | 9.1, 8.8, 7.8, 6.8 |
| linear `μ=0`, `L=6` (rings 32/27/21/16/11/5) | 8.9, 6.2, 1.5, 0.9, 0.6, 0.1 | 9.4, 9.1, 8.1, 7.4, 6.8, 4.1 |
| double well `β=1`, `L=4` | 8.4, 4.0, 0.9, 0.7 | 7.9, 6.6, 5.0, 4.3 |

F-high separation survives on 24- and 27-unit rings and collapses on rings of
21 or fewer units. A cos 8θ pattern needs more than 16 samples per ring to be
represented without aliasing, so rings of 16 or fewer cannot carry it at all.
The 21-unit ring can represent it but has already attenuated it. The
shrinking-ring disk geometry therefore caps fine distinctions at a fixed
depth, on top of harmonic attenuation.
This is the depth confound declared in contract section 2.1, now with a
concrete mechanism.

## Readings

1. **R1 is real in this family.** The harmonic interior is calmer, but fine
   distinctions do not reach it. Two mechanisms act together: ring sizes too
   small to represent the pattern, and attenuation relative to coarse nuisance.
2. **The double well is not the hoped-for remedy.** Local bistability turns
   small interior differences into ±1 commitments, but these follow nuisance
   as much as signal. The result is noisier, start-dependent interiors.
3. **Classifier choice matters on larger inner rings.** The information
   survives there but is not linearly separable by unweighted distance. This
   is reported as a diagnostic. The primary criterion is unchanged.

## What this does not establish

- Anything about modules with constant ring sizes, field drives, other
  nonlinearities, temporal retention, holes, or coupled modules. None was
  tested.
- That nonlinearity cannot help. Only one potential family, applied uniformly
  to every unit, was tried.
- Anything about abstraction, learning or the core.

## Accounting

| Item | Value |
| --- | --- |
| Canonical settlings / multistability settlings / superposition settlings | 32,000 / 10,240 / 1,920 |
| Grid and gates wall time | ≈218 s (one BLAS thread) |
| Largest module | 113 units (`L = 6`) |

## Next obligations (not run)

1. The stopping rule applies. Do not retune thresholds, classifiers, ring sizes
   or potentials inside v1.
2. A follow-up contract could separate the two F-high mechanisms. One option
   is constant ring sizes, to remove aliasing. Another is a declared whitened or
   learned readout, to separate information present from information usable by
   distance. Field drives (contract section 10) remain a required arm.
3. A nonlinear arm that calms nuisance would need a different mechanism from
   independent double wells. Candidates include saturating rather than
   bistable units, or interior-only nonlinearity. Any such arm needs a new
   contract.
