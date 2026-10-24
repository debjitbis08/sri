# Layered module v2 — separating geometric loss, attenuation and readout limits in one module, with soft-tie and field drives

**Status:** prospective; frozen for implementation on 2026-10-24. Freeze this
contract's hash in the manifest before any scored computation.

**Pre-implementation specifications (2026-10-24, before any scored
computation).** The draft left these details open. They are fixed here, and
no threshold, factor or readout changed.

1. **Monte Carlo pairing.** In the population reference (section 5.3), draw
   `i` is paired with class `i mod 4`, so each class gets 250,000 draws. Each
   draw is assigned to the nearest whitened class mean.
2. **Whitening.** `Σ_ℓ` is handled through the singular value decomposition
   `G_ℓ = U S Vᵀ`. Because `n_ℓ ≤ 32`, `U` is square, and
   `Σ_ℓ⁻¹ = U diag(1/((α²/2) s² + σ²)) Uᵀ` exactly. This avoids an
   ill-conditioned Cholesky factorization when mode gains span many orders of
   magnitude. The class means are `±a, ±b`, so the Monte Carlo works in at
   most 2 dimensions, even if rounding adds tiny further components.
3. **Oracle-covariance LDA** is `null` if any class has no certified training
   drive.
4. **Spectra** use the certified drives of both halves, without readout
   noise, like `S` and `P`.
5. **Reading G.** For the linear arm, if only the LDA difference reaches the
   margin, the outcome is "not shown"; both values are still reported.
6. **Reading R.** The projection and covariance-estimation sub-readings are
   "not evaluable" when their own readout is `null`. The NM sub-reading
   still stands.
7. **Prediction check.** It uses the closed form at absolute `σ₀ = 0.1`, for
   both `α`, at layer `L`. A difference counts as larger than sampling
   variation when `|LDA − A_ref| > 0.1`.
8. **Estimation-gap table.** At layer `L`, for absolute `σ₀ = 0.1` and
   relative `σ_rel = 0.1`. Values for every layer are in the per-configuration
   records.
9. **Field versus tie.** Strict accuracies at layer `L` and absolute
   `σ₀ = 0.1`. The LDA difference is reported only when both LDA readouts are
   available.
10. **Gate inputs.** The derivative and LDA-implementation gates draw from
    `default_rng(20261028, …)`. The per-mode gain gate settles pure
    `cos(nθ)` and `sin(nθ)` drives. None of these settlings enters any
    measure.
11. **v1's Mahalanobis diagnostic** is not carried into v2's outputs, because
    it counts readout noise twice (section 5.4). The v1 replication gate
    compares the quantities listed in section 11 only.

## 0. Where this sits

[Layered module v1](layered-module-v1-results.md) found no configuration that
meets the declared profile. In the linear arm the declared nearest-mean
readout does not recover F-high distinctions at the interior in any
configuration. Three candidate explanations remain unseparated:

- **Geometric loss.** Rings shrink inward, so the inner rings cannot represent
  mode 8 faithfully.
- **Attenuation.** The mode-8 response shrinks with depth, and the readout
  noise `σ₀` is fixed.
- **Readout limits.** The unweighted nearest-mean readout may not use
  information that is present.

In v1 these vary together, because smaller rings are also deeper.

v2 is a new contract. It does not retune v1, and v1's stopping rule still
applies to v1. v2 changes three things:

1. It adds a **constant-width geometry** (every ring has 32 units), alongside
   v1's shrinking geometry.
2. It declares three readouts: the v1 nearest-mean readout, a covariance-aware
   readout (LDA), and a diagnostic projection onto the signal's mode. All are
   scored on held-out data. For the linear arm, an oracle-covariance LDA
   control isolates the cost of estimating the covariance.
3. It adds the required **field-drive arm** (v1 contract, section 10).

It also adds **relative readout noise**, which scales with each layer's
signal size. For the linear arm it adds an exact **population reference**
computed from the module's linear map.

v2 is a diagnostic study. **It selects no depth or configuration for
Stage 2.** Its readings inform the Stage 2 contract.

## 1. Question

In one layered disk module with continuous states, settled independently per
drive:

1. **Geometric loss.** Does the shrinking geometry lose class information
   that the constant-width geometry keeps, at matched depth and layer, once
   attenuation is factored out by using relative readout noise?
2. **Attenuation.** With constant width, is attenuation relative to a fixed
   readout noise enough to explain the loss of F-high distinctions at the
   interior?
3. **Readout limits.** Does a covariance-aware readout recover held-out
   distinctions that the nearest-mean readout misses?
4. **Drive form.** Do these answers change when the soft exterior tie is
   replaced by an exterior field?

**What this does not measure.** As in v1:

- Nothing here tests memory, retention or persistence over time.
- Layers are not levels of abstraction.
- Nothing here concerns inter-module coupling, the core, or learning.

## 2. Module

### 2.1 Geometries

Units, layers, unit order, centre and radial rule are as in v1 (section 2.1).
Only the ring sizes differ.

- **Shrinking (v1).** `n_ℓ = max(4, ⌊32·(L−ℓ+1)/L + ½⌋)`:

  | `L` | Ring sizes |
  | --- | --- |
  | 2 | 32, 16 |
  | 3 | 32, 21, 11 |
  | 4 | 32, 24, 16, 8 |
  | 6 | 32, 27, 21, 16, 11, 5 |

- **Constant.** `n_ℓ = 32` for every `ℓ`. Under v1's radial rule, unit `j` of
  ring `ℓ` connects to unit `j` of ring `ℓ+1` only. The centre unit is joined
  to every unit of ring `L`. The largest module (`L = 6`) has 193 units.

**What still varies with depth.** With constant width, increasing `L` still
adds units, lengthens the radial path to the centre and (with `γ = 2`) raises
the maximum lateral stiffness. Ring width at a given layer no longer changes.
Comparing geometries at matched `(L, ℓ)` isolates ring width. That
comparison is the purpose of this factor.

### 2.2 Energy and drives

```
E(u) = ½ Σ_lateral κ_ℓ (u_i − u_j)²  +  ½ Σ_radial κ_r (u_i − u_j)²  +  Σ_i V(u_i)  +  D(u; h)
```

`κ_ℓ = γ^{ℓ−1}`, `κ_r = 1` and the potentials are as in v1. The drive term
`D` has two forms:

- **Tie (v1).** `D = ½ λ Σ_{i ∈ layer 1} (u_i − h_i)²`, with `λ = 1`.
- **Field.** `D = −Σ_{i ∈ layer 1} h_i u_i`.

Up to a constant, the tie equals a field `λh` plus an exterior-only anchor
`½ λ Σ_{layer 1} u_i²`. For the linear arm the two forms differ only in
gain and in that diagonal term. For the double well the field acts as a tilt,
with no restoring term besides the wells. That is where the drive form is
expected to matter.

**Field arm potentials.** Linear `μ = 1` and double well
`β ∈ {0.5, 1, 2}`. Linear `μ = 0` is excluded: the Hessian is then the graph
Laplacian, which is singular, and a field with a nonzero DC component makes
`E` unbounded below.

The gradient and Hessian are analytic, as in v1, with the drive term changed.

### 2.3 Settling

v1's conventions (section 2.3) apply unchanged: L-BFGS-B options, the Newton
polish, certification (`‖∇E‖∞ < 1e−8` and smallest Hessian eigenvalue
`> 1e−8`), the multistability diagnostic (seeds `[20261026, L, s]`) and
rounding to 10 significant digits. Dense Hessians reach at most 193 units.

## 3. Perturbation ensembles

These are identical to v1 section 3: signal families F-low (`n = 1`) and
F-high (`n = 8`), 4 classes; the 32-vector Fourier nuisance basis; draws from
`default_rng(20261024)` with shape `(2, 4, 50, 32)`; rescaling to RMS `α/√2`;
draws 0–24 for training and 25–49 for testing. Each configuration therefore
sees the same drives in both geometries and both drive forms (paired).

**Pure-signal drives.** For each family, the four drives `h^{sig}_c` with no
nuisance are also settled from the canonical start. They define the relative
noise scale only (section 4).

## 4. Readout noise

Each layer-`ℓ` state receives additive noise `σ·ε` before decoding.
`ε ~ default_rng([20261025, L, ℓ, family])`, with shape `(4, 50, n_ℓ)`, as
in v1. The same `ε` is used across geometries, drives, potentials, `γ`, `α`
and noise levels (where shapes match).

- **Absolute (v1).** `σ = σ₀ ∈ {0.03, 0.1, 0.3}`.
- **Relative.** `σ = σ_rel · R_ℓ`, with `σ_rel ∈ {0.03, 0.1, 0.3}`.
  - `R_ℓ` is the RMS over units of layer `ℓ` and over the four classes of the
    certified pure-signal states, restricted to layer `ℓ`.
  - If any pure-signal drive is unsettled, or `R_ℓ < 1e−12`, relative-noise
    results for that layer are `null`.

Relative noise asks whether class information is present regardless of
overall size. Absolute noise asks whether it is present above a fixed floor.

## 5. Measures (per configuration, family, layer and noise setting)

### 5.1 Response measures

- `S(ℓ)`, `P(ℓ)` and the suppression ratio `P(L)/P(1)`: as in v1 section 4,
  without readout noise.
- **Spectral decomposition.** Project each layer-`ℓ` state onto the
  orthonormal real Fourier basis of `R^{n_ℓ}`. This basis has a DC vector;
  `cos(kφ)` and `sin(kφ)` for `k = 1..⌈n_ℓ/2⌉−1`; and, for even `n_ℓ`, a
  Nyquist vector. Here `φ_j = 2πj/n_ℓ`. For each mode `k`, report:
  - `P_k(ℓ)`: within-class variance of its coefficients (cos and sin summed),
    averaged equally over classes;
  - `B_k(ℓ)`: between-class variance of the class-mean coefficients.

  This shows which frequency bands carry the calming and which carry the
  distinctions.

### 5.2 Readouts

All readouts are fitted on noisy, certified training states only and scored
on noisy test states.

1. **Nearest mean (NM, v1).** Euclidean nearest class mean. Ties go to the
   lowest class index.
2. **LDA (covariance-aware).**
   - Fit class means `m̂_c` and the pooled within-class covariance `Σ̂`, with
     denominator `N_cert,train − 4`.
   - Score `δ_c(x) = m̂_cᵀ Σ̂⁻¹ x − ½ m̂_cᵀ Σ̂⁻¹ m̂_c`, with equal priors,
     solved through a Cholesky factor of `Σ̂`. Assign the largest score; ties
     go to the lowest index.
   - LDA is `null` if `N_cert,train − 4 < n_ℓ`, if any class has no certified
     training drive, or if the Cholesky factorization fails.
   - No shrinkage or regularization is applied. Readout noise keeps `Σ̂` full
     rank in expectation.
3. **Mode projection (diagnostic).**
   - Features are the two coefficients of the layer state on the orthonormal
     `cos(nφ)` and `sin(nφ)` vectors of ring `ℓ`, where `n` is the family's
     signal mode. Classify by NM in these 2 dimensions.
   - It is defined only when `n_ℓ > 2n`, so that mode `n` is faithfully
     representable. Otherwise it is `null`. For F-high this excludes
     shrinking rings of 16 or fewer units.
   - This readout uses knowledge of the signal mode. It never enters the
     profile or the readings in section 9, except as stated there.
4. **Oracle-covariance LDA (diagnostic, linear arm only).** The same scores
   as LDA, with the estimated class means `m̂_c` but the population covariance
   `Σ_ℓ` of section 5.3 in place of `Σ̂`. It isolates the cost of estimating
   the covariance, because the means are estimated exactly as for LDA. It uses
   knowledge of the nuisance distribution, but no test data. It never enters
   the profile.

**Accuracy.** Strict accuracy is correct test assignments out of 100;
unsettled test drives, and every test drive when the readout is `null`, count
as incorrect. Secondary accuracy is out of the certified test drives only.

A strict accuracy of zero from a `null` readout is used **only for profile
accounting** (section 10). For the diagnostic readings in section 9, a `null`
readout is unavailable, not zero (section 9, validity conditions).

### 5.3 Population reference (linear arm only)

For a linear configuration, the layer-`ℓ` state is `U_ℓ = G_ℓ h`. `G_ℓ` is
the layer-`ℓ` block of `H⁻¹ P₁` (field) or `λ H⁻¹ P₁` (tie), computed by a
direct solve.

- **Exact covariance.** The nuisance draws are uniform on a sphere of radius
  `4α` in `R^{32}`, so their covariance is exactly `(α²/2) I`. The noisy
  layer state therefore has covariance
  `Σ_ℓ = (α²/2) G_ℓ G_ℓᵀ + σ² I` and class means `G_ℓ h^{sig}_c`.
- **Reference accuracy `A_ref`.** The Bayes accuracy for four equal-prior
  Gaussian classes with these means and this shared covariance.
  - In whitened coordinates the four means lie in a 2-dimensional subspace.
    Compute the accuracy by Monte Carlo in that subspace: `10⁶` standard
    normal draws from `default_rng(20261027)`, with the same draws used for
    every evaluation (common random numbers). Report the number of
    misclassified draws and the two-sided Clopper–Pearson interval for the
    error rate (section 11 gives its level). This interval stays valid when
    no draw is misclassified, unlike the empirical binomial standard error.
  - `A_ref` is a **linear-Gaussian reference**, not a strict ceiling: the true
    nuisance is not Gaussian, and `A_ref` assumes known parameters.
- **Population separation.** Report the minimum pairwise Mahalanobis distance
  between the class means under `Σ_ℓ`.

### 5.4 Empirical separation

Report the minimum pairwise Mahalanobis distance between `m̂_c` under `Σ̂`,
using LDA's own fit on noisy training states.

**Note on v1.** v1's Mahalanobis diagnostic added `σ₀² I` to a covariance
estimated from states that already carried readout noise, so the noise was
counted twice. Its separations are therefore mild underestimates. No v1
conclusion depends on this. v2 does not add the extra term.

### 5.5 Interference and multistability

As in v1: the superposition defect (same 8 pairs) and distinct minima per
probe.

## 6. Unsettled samples and eligibility

As in v1 section 5. A configuration is eligible only if all 200 canonical
settlings in both families are certified. Pure-signal drives do not affect
eligibility. Unsettled pure-signal drives make relative-noise results `null`.

## 7. Grid

Factors:

- geometry ∈ {shrinking, constant}
- drive and potential:
  - tie: linear `μ ∈ {0, 1}`, double well `β ∈ {0.5, 1, 2}`;
  - field: linear `μ = 1`, double well `β ∈ {0.5, 1, 2}`
- `L ∈ {2, 3, 4, 6}`
- `γ ∈ {1, 2}`
- `α ∈ {0.5, 1}`

That gives `2 × 9 × 4 × 2 × 2 = 288` configurations. Each runs both
families with 200 drives each (115,200 canonical settlings), plus 8
pure-signal settlings per configuration, plus the diagnostics. The 80
shrinking-geometry, tie-drive configurations are v1's grid and serve as a
replication check (section 11).

Every configuration is reported. There is no best-configuration selection and
no per-configuration tuning.

## 8. Analytic expectations (disclosed before computation)

**Constant width, linear arm.** Rings of equal size are joined `j ↔ j`, and
the centre couples only to the DC mode. The linear map is therefore diagonal
in the angular Fourier basis.

- **Per-mode gain.** Mode `n` (cos and sin alike) reaches layer `ℓ` with gain
  `g_n(ℓ)`. The gains solve an `L × L` tridiagonal system with:
  - diagonal `γ^{ℓ−1} e_n + μ + 2` for `ℓ ≥ 2`, and `e_n + μ + 1` for
    layer 1, plus `λ` on layer 1 for the tie. Ring `L` counts the centre as
    its inner neighbour; the centre is at 0 for `n ≥ 1`;
  - off-diagonal `−1`;
  - right-hand side `λ` (tie) or `1` (field) on layer 1;
  - `e_n = 2(1 − cos(2πn/32))`.
- **Reference accuracy in closed form.** In this geometry the classes form a
  square of radius `4 g_n(ℓ)` in the mode-`n` plane. The noise there is
  isotropic, with per-axis variance `g_n(ℓ)² α²/2 + σ²`. So

  ```
  A_ref = (1 − Q(4 g / (√2 · √(g² α²/2 + σ²))))²,   g = g_n(ℓ),  Q = standard normal tail.
  ```

  Mode 8 attenuates by about `2 − √3 ≈ 0.27` per layer (`μ = 0`, `γ = 1`). Mode 1
  attenuates much less, about 0.82 per layer at `μ = 0`, `γ = 1`, far from
  both boundaries.

**Predicted `A_ref` at layer `L`, absolute `σ₀ = 0.1`, `α = 1`, constant
width** (F-low / F-high; computed from the formula above, with no drives):

| Drive, potential, `γ` | `L=2` | `L=3` | `L=4` | `L=6` |
| --- | --- | --- | --- | --- |
| tie, `μ=0`, `γ=1` | 1.00 / 0.91 | 1.00 / 0.48 | 1.00 / 0.31 | 0.98 / 0.25 |
| tie, `μ=0`, `γ=2` | 1.00 / 0.77 | 1.00 / 0.30 | 0.99 / 0.25 | 0.54 / 0.25 |
| tie, `μ=1`, `γ=1` | 0.99 / 0.76 | 0.79 / 0.36 | 0.47 / 0.27 | 0.28 / 0.25 |
| tie, `μ=1`, `γ=2` | 0.99 / 0.63 | 0.77 / 0.28 | 0.43 / 0.25 | 0.26 / 0.25 |
| field, `μ=1`, `γ=1` | 1.00 / 0.84 | 0.93 / 0.39 | 0.60 / 0.28 | 0.30 / 0.25 |
| field, `μ=1`, `γ=2` | 1.00 / 0.71 | 0.92 / 0.29 | 0.55 / 0.25 | 0.27 / 0.25 |

At `α = 0.5` the values change by at most 0.02, because in-band nuisance
reaching the interior is small compared with `σ₀`. The run recomputes these
values through the general Monte Carlo procedure (section 5.3), and a gate
checks the two against each other.

**Predictions:**

- **E1. Constant width does not rescue F-high under absolute noise.** Even an
  optimal readout should fail T3 (`≥ 0.9`) for F-high in every
  constant-width linear configuration except tie, `μ = 0`, `γ = 1`, `L = 2`
  (0.91, a borderline case). If the empirical LDA results agree, attenuation
  relative to a fixed readout floor is enough to explain the F-high failure,
  without any geometric loss.
- **E2. Relative noise makes the constant-width reference independent of
  depth.** Signal, nuisance and readout noise all scale with `g`, so the
  argument of `Q` becomes `4/√(α² + σ_rel²)`. At `σ_rel = 0.1`, `A_ref ≈ 1.00`
  at every layer, in both families, for `α ∈ {0.5, 1}`. Any drop in the
  shrinking geometry under relative noise is then attributable to geometry.
- **E3. NM falls below LDA in deep layers, even under relative noise.** NM's
  estimated class means carry training-set nuisance in every mode. Deep in
  the module, low-mode nuisance (gain `g_1`) dominates the mode-`n` signal
  (gain `g_8 ≪ g_1` for F-high). This error does not shrink when readout
  noise is scaled. LDA and mode projection are expected to stay close to
  `A_ref`, and NM to degrade with depth for F-high.
- **E4. Field and tie differ little in the linear arm.** The table shows
  small, quantitative differences only.
- **E5. Double well.** No prediction. Under the field drive, units are tilted
  toward the sign of the local drive with no exterior anchor. This could make
  the units' commitments follow the drive more faithfully than under the tie,
  or still let them follow nuisance.

## 9. Declared readings

These rules fix in advance which statements the results can support. Each
threshold is a convention. Full values are reported, so a reader can apply
other thresholds. "Layer" means any layer `ℓ ≥ 2`; matched comparisons hold
`L, ℓ, γ, α`, drive and potential fixed. Accuracies are strict.

**Validity conditions.** A reading is evaluated at a given configuration (or
matched pair of configurations), family, layer and noise setting only if all
of the following hold. Otherwise the reading is **not evaluable** there.

- Every participating configuration is eligible (section 6). For eligible
  configurations, strict and secondary accuracy coincide.
- Every readout that the rule compares is non-`null`.
- Every `A_ref` value the rule uses is available (linear arm only).
- For relative noise, `R_ℓ` is non-`null` in every participating
  configuration.

Each reading table lists the not-evaluable cases with their reasons and
counts. A not-evaluable case never supports or counts against a claim.

- **G — geometric loss.** Under relative noise `σ_rel = 0.1`, for a linear
  configuration and family: geometric loss is **shown at layer ℓ** if both
  `A_ref(constant) − A_ref(shrinking) ≥ 0.10` and
  `LDA(constant) − LDA(shrinking) ≥ 0.10`.
  - If only the first holds, the report says the loss is **predicted by the
    linear map but not confirmed by held-out decoding**.
  - For the double well, only the empirical LDA comparison is available. It
    is reported under the label "consistent with geometric loss", never
    "shown".
- **A — attenuation sufficiency.** For a constant-width linear configuration
  and family at layer `L`: attenuation relative to fixed readout noise is
  **sufficient to explain** a T3 failure if all four hold:
  - `A_ref` at absolute `σ₀ = 0.1` is `< 0.9`;
  - `A_ref` at `σ_rel = 0.1` is `≥ 0.9`;
  - LDA at absolute `σ₀ = 0.1` is `< 0.9`;
  - LDA at `σ_rel = 0.1` is `≥ 0.9`.
- **R — readout limitation.** At layer `L`, at absolute `σ₀ = 0.1` or
  relative `σ_rel = 0.1`: the NM readout is **limiting** if
  `LDA − NM ≥ 0.10`.
  - If `projection − LDA ≥ 0.10`, the report says only that the
    mode-projection readout performs better on held-out data than LDA. That
    is a **possible estimation or model limitation of LDA**, not an
    identified cause. Projection also discards irrelevant features and
    estimates class means in 2 dimensions instead of `n_ℓ`. In the
    double-well arm, a shared Gaussian covariance may not describe the
    states at all.
  - **Covariance-estimation control (linear arm only).** If
    `oracle-covariance LDA − LDA ≥ 0.10`, the report says that estimating the
    covariance from 100 training drives limits LDA there. This is the only
    rule that attributes a gap to covariance estimation. No such attribution
    is made for the double-well arm.
- **Estimation gap.** `A_ref − LDA` is reported for every linear
  configuration. No claim rests on it.
- **Prediction check.** For each constant-width linear configuration, the
  report lists the empirical LDA accuracy next to the `A_ref` value in
  section 8. With 100 test drives, differences up to about 0.1 are within
  sampling variation. Larger differences are reported as unexplained, not
  corrected.

## 10. Outputs

- For every configuration: all measures in section 5, for every layer, family
  and noise setting, plus certification counts.
- **Reading tables** G, A and R, by family, arm, geometry and drive.
- **Profile counts.** The v1 profile (T1, T2, T3 at absolute `σ₀ = 0.1`)
  evaluated separately with NM and with LDA as the T3 readout, by arm,
  geometry and drive. Meeting the profile with any readout is reported, but
  **selects nothing for Stage 2.**
- **Spectral tables.** `P_k(ℓ)` and `B_k(ℓ)` for `k ∈ {0, 1, 8}` and in
  total, for each linear configuration at `α = 1`. Full spectra are stored in
  the results file.
- **Field versus tie.** Matched differences in LDA and NM accuracy at layer
  `L`, by potential and geometry.
- Multistability counts and superposition defects for every configuration.

## 11. Correctness gates

- **v1 replication.** For the 80 shrinking-geometry, tie-drive
  configurations, these match v1's reported results exactly after rounding:
  certification counts, `S`, `P`, suppression ratios, NM strict and secondary
  accuracies, distinct minima and superposition defects.
- **Constant-width graph.** Every ring has 32 units, radial edges are exactly
  `j ↔ j` between consecutive rings, the centre is joined to all of ring `L`,
  the graph is connected and has no duplicate edges.
- **Gradient and Hessian.** Analytic forms for both drive forms agree with
  central finite differences (step `1e−6`, tolerance `1e−5`) at seeded
  random states.
- **Linear direct solve.** For both drive forms, every canonical linear state
  equals the direct sparse solve to within `1e−7` (∞-norm).
- **Per-mode gains.** For every constant-width linear configuration and
  `n ∈ {1, 8}`, the settled responses to pure `cos(nθ)` and `sin(nθ)` equal
  `g_n(ℓ)` times the same pattern on ring `ℓ`, to within `1e−10` (∞-norm).
- **Reference procedure.** For every constant-width linear configuration,
  layer, family and noise setting, the closed-form error rate `1 − A_ref`
  from section 8 lies inside the two-sided Clopper–Pearson interval, at
  level `1 − 10⁻⁶`, for the Monte Carlo error count out of `10⁶` draws.
  - With zero misclassified draws, the interval is about `[0, 1.4·10⁻⁵]`,
    so a correct implementation passes near perfect accuracy.
  - The level allows for the roughly 2,000 checks in this gate.
- **LDA implementation.** On seeded synthetic Gaussian data, the Cholesky
  scores match an explicit-inverse implementation to within `1e−9`. Accuracy
  is unchanged under a seeded invertible linear transform of the features.
- **Attenuation sanity** (v1 gate), and superposition in every linear
  configuration (v1 gate, both drive forms).
- **No leakage.** Test drives never enter means, covariances, `R_ℓ` or any
  other fitted quantity. `R_ℓ` uses pure-signal drives only.
- **Determinism and environment.** As in v1: declared seeds recorded in the
  manifest, byte-identical regeneration in a temporary directory, pinned
  numpy and scipy, one BLAS thread, runtime metadata in a separate file.

## 12. Deferred (not in this run)

- **Band-limited nuisance**, with nuisance kept out of the signal's band.
  This tests whether a fixed module can calm nuisance that it can tell apart
  from the signal.
- **Learned or adapted per-mode gains** ("Stage 1.5"). This would supply what
  counts as relevant, which a fixed module cannot supply.
- **Other nonlinearities:** divisive normalization, saturating
  (non-bistable) potentials, interior-only nonlinearity, phase (XY) units.
  Each needs its own contract.
- **Temporal retention** and **topological modes**, as in v1.

## 13. Stopping rule and scope

Run the declared grid once. Do not add potentials, geometries, readouts,
noise levels or thresholds after seeing results. v2 characterizes response
properties of a single disk module under the two drive forms, settled
independently per drive, and separates three explanations for v1's interior
failure. It selects no Stage 2 configuration. It does not test inter-module
coupling, the core, learning, temporal persistence or abstraction.
