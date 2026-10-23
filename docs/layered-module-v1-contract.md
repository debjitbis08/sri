# Layered module v1 — exterior sensitivity, interior nuisance suppression and kept distinctions in one module

**Status:** prospective; frozen for implementation on 2026-10-24. Freeze this
contract's hash in the manifest before any scored computation.

**Pre-implementation change (2026-10-24, before any scored computation).** The
draft ran the Newton polish only when `‖∇E‖∞ ≥ 1e−8`. Certification at
`1e−8` bounds the state error only by `1e−8/λ_min`, which can exceed the
linear gates' `1e−7` and `1e−8` tolerances. The polish now always runs, aiming
at `‖∇E‖∞ < 1e−12` (section 2.3). Near a minimum, energy differences fall
below floating-point resolution, so a step is also accepted when it does not
raise `E` beyond `1e−12·max(1,|E|)` and strictly lowers `‖∇E‖∞`.
Certification thresholds are unchanged.

**Scope decisions at review (2026-10-24):**

- Topological modes (holes, tori) are left out for now. Only the disk shape is
  used.
- Perturbations start simple: a soft exterior tie to the drive. Field drives
  are a required later arm (section 10), not part of this run.

**Second review (2026-10-24).** Six points were tightened:

1. The former "persistence" ratio is renamed **nuisance suppression**. It is
   not temporal persistence.
2. Nuisance generation is now exact.
3. Rules for unsettled samples are defined.
4. Numerical conventions are frozen.
5. Evaluation details are explicit.
6. The confounds that come with changing depth are declared.

## 0. Where this sits

This is **Stage 1** of a new line of work. It follows the
[v3 results](shared-core-abstraction-v3-results.md) and the design discussion
of 2026-10-24.

The earlier substrate has five scalar units and no layers. It cannot show
modules that have a detailed exterior, a calmer interior, or a core that
builds up and helps many satellites. The planned sequence is:

1. **Stage 1 (this contract).** One module, no learning. Characterize how a layered
   module responds to exterior perturbations at each depth.
2. **Stage 2.** Global Input and Output modules (which have layers), two
   satellites, and a core.
   - Satellites do not couple to each other directly. All satellite-to-satellite
     influence goes through the core.
   - Every satellite couples directly to Input and Output (decision (a); this
     keeps the current 5-unit model as a special case).
   - Satellite–core coupling exists at every layer and is graded: stronger
     inward, weaker outward.
3. **Stage 3.** Many satellites and a learning stream, measuring cross-satellite
   transfer over time.

Direct satellite–satellite links are the more general alternative. They are
deferred, not rejected. Stage 1 involves a single module, so the mediator
decision does not affect it.

## 1. Question

Can one layered module with continuous states respond to a wide variety of
exterior perturbations at its exterior, while its interior is
(i) less affected by incidental variation and (ii) still retains task-relevant
distinctions?

How do linear versus nonlinear local dynamics, depth, and an inward
lateral-coupling gradient change this?

The **harmonic extension** on a disk motivates the question. For boundary data
`f(θ)=Σ a_n e^{inθ}`, the interior is `u(r,θ)=Σ a_n r^{|n|} e^{inθ}`. High
frequencies fade inward. This contract takes up the two reservations recorded for
that analogy:

- **R1.** Attenuation might erase distinctions that are essential for
  interpretation.
- **R2.** Harmonic extension is linear. Linear modes may not be expressive
  enough.

The harmonic extension is the minimizer of a Dirichlet energy with the
boundary held fixed. The linear arm below is that analogy, made discrete. The
nonlinear arm adds local multi-well terms to the same energy.

**What this does not measure.**

- Every drive is settled independently from the same start, so nothing here
  tests memory, retention, or persistence over time. "Interior suppression"
  means only that independently settled interior states vary less with
  incidental input variation.
- Layers are **not** levels of abstraction. Stage 1 measures response
  properties by depth only. It makes no claim about abstraction or
  intelligence.

## 2. Module

### 2.1 Graph (disk)

A module is a graph of continuous units `u_i ∈ R`, arranged in layers
`ℓ = 1..L`. Layer 1 is the exterior and layer `L` is the innermost.

- **Layers.** Layer `ℓ` is a ring `C_{n_ℓ}`, with units `j = 0..n_ℓ−1` at
  angles `2πj/n_ℓ`. Edges within a layer are *lateral*.
- **Ring sizes.** `n_1 = 32` and `n_ℓ = max(4, ⌊n_1·(L−ℓ+1)/L + ½⌋)`, rounding
  half up. This gives:

  | `L` | Ring sizes |
  | --- | --- |
  | 2 | 32, 16 |
  | 3 | 32, 21, 11 |
  | 4 | 32, 24, 16, 8 |
  | 6 | 32, 27, 21, 16, 11, 5 |

- **Centre.** One centre unit is joined to every unit of ring `L`. It belongs to
  no layer and is reported separately.
- **Radial edges.** Each unit `j` of ring `ℓ` connects to the unit `k` of ring
  `ℓ+1` that minimizes the circular distance between `j/n_ℓ` and
  `k/n_{ℓ+1}`. Distances are compared in exact rational arithmetic. Ties go to
  the lowest `k`. Any unit of ring `ℓ+1` still unconnected is then joined to
  its nearest unit of ring `ℓ`, by the same rule. Duplicate edges are merged.
- **Unit order.** Ring 1 units in index order, then ring 2, and so on, with the
  centre unit last.

**Depth changes several properties together.** Increasing `L` adds units,
changes ring sizes at a given layer index, lengthens the radial path from
exterior to centre, and (with `γ = 2`) raises the maximum lateral stiffness to
`γ^{L−1}`. Conclusions describe **this declared architecture family**. They do
not attribute effects to depth alone.

### 2.2 Energy

```
E(u) = ½ Σ_lateral κ_ℓ (u_i − u_j)²  +  ½ Σ_radial κ_r (u_i − u_j)²
     + Σ_i V(u_i)  +  ½ λ Σ_{i ∈ layer 1} (u_i − h_i)²
```

- `κ_ℓ = γ^{ℓ−1}` (with `κ = 1`) is the lateral coupling in layer `ℓ`.
- `κ_r = 1` for every radial edge, including the centre edges.
- `λ = 1`. The exterior is softly tied to the drive `h`, so exterior
  sensitivity is a measured quantity, not a clamp identity.
- Local potentials, applied to every unit including the centre:
  - **Linear arm.** `V(u) = ½ μ u²` with `μ ∈ {0, 1}`. With `μ = 0`, the
    interior is the discrete harmonic extension of the exterior state.
  - **Nonlinear arm.** `V(u) = ¼ β (u² − 1)²` with `β ∈ {0.5, 1, 2}`. This is a
    double well with stable values near ±1.

The gradient and Hessian are analytic: graph Laplacian terms, plus `V'` and
`V''` on the diagonal, plus `λ` on layer-1 units.

### 2.3 Settling (numerical conventions)

- **Optimizer.** `scipy.optimize.minimize(method='L-BFGS-B')`, with no bounds
  and the analytic gradient. Options: `maxiter=20000`, `maxcor=20`, `ftol=0`,
  `gtol=1e−11`, `maxls=50`. Start from `u = 0` (the canonical start).
- **Polish.** Then take up to 20 Newton steps with the exact (dense) Hessian,
  stopping once `‖∇E‖∞ < 1e−12`, when the Hessian is singular, or when no step
  is accepted. Each step uses backtracking (factor ½, at most 30 halvings). A
  step is accepted if it lowers `E`, or if it raises `E` by at most
  `1e−12·max(1,|E|)` while strictly lowering `‖∇E‖∞`.
- **Certification.** A state is *certified* if `‖∇E‖∞ < 1e−8` and the smallest
  eigenvalue of the dense Hessian is `> 1e−8` (`numpy.linalg.eigvalsh`; at
  most 113 units). Otherwise it is *unsettled*. A saddle counts as unsettled.
- **Multistability diagnostic.**
  - Probe drives: draw 0 of each class in each family at the configuration's
    `α` (8 drives).
  - For each probe, settle from 16 random starts `u₀ ~ N(0, 1)` per unit, using
    generator `default_rng([20261026, L, s])`, with `s = 0..15` the start
    index.
  - Merge certified states greedily, in start order, when their RMS distance
    over all units is `< 1e−6`.
  - Report distinct minima per probe and unsettled starts. The diagnostic
    never affects primary measures.
- **Reporting.** Floats in scientific reports are rounded to 10 significant
  digits. Byte-identical regeneration is claimed for the same machine and
  pinned environment only.

## 3. Perturbation ensembles

Drives act on layer 1 (32 units, `θ_j = 2πj/32`).

- **Signal families.** There are 4 classes `c = 0..3`, with
  `h^{sig}_c(θ_j) = cos(n θ_j + πc/2)`.
  - **F-low:** `n = 1`.
  - **F-high:** `n = 8`.

  `RMS(h^{sig}) = 1/√2` in both families.
- **Nuisance basis.** An orthonormal real Fourier basis of `R^32`, with all 32
  vectors:
  - DC: `1/√32`;
  - `cos(kθ)·√(2/32)` and `sin(kθ)·√(2/32)` for `k = 1..15`;
  - Nyquist: `cos(16θ)/√32`.
- **Nuisance draws.**
  - Coefficients are i.i.d. `N(0,1)`, from generator `default_rng(20261024)`.
  - They are drawn once as an array of shape `(family 2, class 4, draw 50,
    basis 32)`, in C order, families ordered (F-low, F-high).
  - Each draw `z` is the basis combination of its coefficients, rescaled
    exactly to RMS `α/√2`: `z ← z · (α/√2) / RMS(z)`.
  - The same underlying draws serve every configuration and both `α` values.
    This gives a paired design.
- **Samples.** `h = h^{sig}_c + z`. Draws 0–24 form the training half and
  draws 25–49 the test half: 100 training and 100 test drives per family.

## 4. Measures (per configuration, family and layer)

Let `U_ℓ(h)` be the certified settled state restricted to layer `ℓ`. The centre
unit is reported separately with the same measures and enters no target
predicate.

1. **Exterior sensitivity** `S(ℓ)`: the population variance (`ddof=0`) of each
   unit across all certified drives in the family (both halves), averaged over
   the units in `ℓ`.
2. **Nuisance variance** `P(ℓ)`: the population variance of each unit within
   each class, over that class's certified drives, averaged over units and then
   equally over classes. The **interior suppression ratio** is `P(L)/P(1)`.
3. **Kept distinctions** `D(ℓ)`:
   - **Readout noise.** For each `σ₀ ∈ {0.03, 0.1, 0.3}`, add `σ₀·ε` to each
     unit, where `ε` comes from `default_rng([20261025, L, ℓ, family])` with
     shape `(class 4, draw 50, n_ℓ)`. The same `ε` is used across potentials,
     `γ`, `α` and `σ₀` (paired).
   - **Classifier.** Class means come from noisy, certified training states.
     Test states are assigned to the nearest mean in Euclidean distance, with
     ties going to the lowest class index.
   - **Primary (strict) accuracy.** Correct test assignments out of 100;
     unsettled test drives count as incorrect.
   - **Secondary accuracy.** The same, out of the certified test drives only.
     Both are reported.
4. **Mahalanobis diagnostic.** Using the **training half only**: class means,
   plus the pooled within-class covariance (denominator `N_train − 4`) plus
   `σ₀² I`. Report the minimum pairwise Mahalanobis distance between the class
   means.
5. **Superposition defect** (interference diagnostic).
   - Pairs `i = 0..7`: `h₁ =` F-low class `i mod 4`, draw `25+i`; `h₂ =`
     F-high class `(i+1) mod 4`, draw `25+i`; both at the configuration's `α`.
   - Defect: `‖U_L(h₁+h₂) − U_L(h₁) − U_L(h₂)‖₂ / ‖U_L(h₁+h₂)‖₂`.
   - If the denominator is `< 1e−12`, or any of the three states is unsettled,
     the pair's defect is `null`. It is excluded from the mean, and the
     `null` count is reported.

## 5. Unsettled samples and eligibility

- Unsettled drives are never dropped from reports. Their counts are given per
  configuration, family and half.
- `S`, `P`, the Mahalanobis diagnostic and secondary accuracy use certified
  drives only, with their counts reported.
- Classifier training uses certified training drives only. A class with no
  certified training drive has no mean, and its test drives count as
  incorrect.
- A configuration is **eligible** for the target profile only if all 200
  canonical settlings in **both** families are certified. Ineligible
  configurations are reported with the label `unsettled` and cannot meet the
  profile.

## 6. Grid

Factors:

- depth `L ∈ {2, 3, 4, 6}`
- potential: {linear `μ=0`, linear `μ=1`, nonlinear `β=0.5`, `β=1`, `β=2`}
- `γ ∈ {1, 2}`
- `α ∈ {0.5, 1}`

Each of these 80 configurations is run on both families (F-low, F-high), with
200 drives each: 32,000 canonical settlings, plus diagnostics. `σ₀` affects
only decoding.

Every configuration is reported. There is no best-configuration selection and
no per-configuration tuning. The parameter values are declared conventions,
not fitted quantities.

## 7. Declared target profile and expectations

An eligible configuration shows the **target profile** at `σ₀ = 0.1` if,
**in each of F-low and F-high separately**:

- **(T1) Exterior sensitivity:** `S(1) ≥ S(ℓ)` for every `ℓ`.
- **(T2) Interior suppression:** `P(L)/P(1) ≤ 0.25`.
- **(T3) Distinctions kept:** strict `D(L) ≥ 0.9`.

The thresholds are conventions. Full curves are reported for every `σ₀`, layer
and family, so a reader can apply other thresholds.

**Analytic expectations, disclosed before computation:**

- **Linear arm.** The settled state is a linear map of `h`. Class-mean
  separation and nuisance shrink by similar factors in each mode, so the main
  thing that changes with depth is signal size relative to the fixed `σ₀`.
  F-high is expected to lose accuracy with depth faster than F-low (R1 in
  action), and `μ = 1` faster than `μ = 0`. T3 is expected to fail for F-high
  at larger `L` and `σ₀`.
- **Nonlinear arm.** Unknown. A double well could turn a small interior
  difference into a full ±1 difference, keeping distinctions against `σ₀` (an
  answer to R1/R2). It could equally erase them by snapping different classes
  to the same well, or make responses depend on the start.
- **Gradient `γ = 2`.** Expected to suppress lateral (high-frequency) variation
  inward, reducing nuisance but also F-high signal.

## 8. Outputs

- **Results by depth.** For each potential and `γ`, the `L` values that meet
  the target profile.
- **Candidate depth for Stage 2.** By this declared rule: nonlinear `β = 1`,
  `γ = 1`, `α = 1`, the smallest `L` meeting the profile; otherwise none. The
  result is stated as a property of the declared architecture family (section
  2.1), not of depth alone.
- **Linear versus nonlinear.** At matched `L`, `γ` and `α`, report the strict
  `D(L)` differences for F-high, with the full `σ₀` curves.
- **Multistability counts and superposition defects** for every configuration.
- **If no configuration meets the profile,** report which of T1, T2 or T3
  fails, by arm and family, plus eligibility failures. A negative result is
  valid.

## 9. Correctness gates

- **Graph construction.** Matches the declared rule on hand-built small cases
  and on the ring sizes tabulated in section 2.1. The graph is connected, the
  centre is present, and there are no duplicate edges.
- **Gradient and Hessian.** The analytic forms agree with central finite
  differences (step `1e−6`, tolerance `1e−5`) at seeded random states.
- **Linear arm.** Every canonical settled state equals the direct sparse solve
  of `H u = λ P₁ h` to within `1e−7` (∞-norm). `H` is the constant Hessian and
  `P₁` the layer-1 selector.
- **Attenuation sanity case** (`μ = 0`, `γ = 1`, `L = 4`, no nuisance): the
  layer-`L` RMS of the response to `cos(8θ)` is below that to `cos(θ)`.
- **Superposition.** For every linear configuration, the absolute defect
  `‖U_L(h₁+h₂) − U_L(h₁) − U_L(h₂)‖₂` is `≤ 1e−8·max(1, ‖U_L(h₁)‖₂ + ‖U_L(h₂)‖₂)`.
- **Certification.** Every reported state carries its certification status.
- **No leakage.** Test-half drives never enter class means, covariances, or any
  other fitted quantity.
- **Determinism.** Seeds and generators are as declared and recorded in the
  manifest. Regeneration in a temporary directory is byte-identical. Runtime
  metadata is kept in a separate file.
- **Environment.** Pinned numpy/scipy versions from `requirements.txt`,
  recorded in the manifest. BLAS runs on one thread, for speed on these small
  matrices and for reproducible floating-point results.

## 10. Deferred arms (required later, not in this run)

- **Field drives.** Replace the soft tie with an exterior field, `−Σ h_i u_i`.
  In the linear arm this needs `μ > 0` or another anchor, because with no
  restoring term the energy is unbounded below. It must be tested before
  Stage 1 findings are treated as robust to the form of the perturbation.
- **Temporal retention.** Sequential drives without resetting to the canonical
  start, to test whether interior states persist across time.
- **Topological modes.** Holes, tori, and edge-valued states (the Hodge
  1-Laplacian) are left out for now.

## 11. Stopping rule and scope

Run the declared grid once. Do not add potentials, depths or thresholds after
seeing results. Stage 1 establishes response properties of a single disk
module under soft exterior perturbations, settled independently per drive. It
does not test inter-module coupling, the core, learning, temporal persistence,
abstraction, dreaming or memory. Stage 2 needs its own contract and may use
Stage 1's declared depth rule.
