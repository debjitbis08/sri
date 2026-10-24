# Layered module v2 — attenuation against a fixed readout floor explains the interior F-high failure; shrinking rings add geometric loss at 16 units or fewer; nearest-mean is limiting once attenuation is factored out

**Completion:** `complete`. Every correctness gate passed, all 288
configurations are eligible (all 115,200 canonical settlings certified), v1
replicates exactly, and regeneration is byte-identical.

**Headline:**

- **No configuration meets the profile with either readout** (0 of 288 with
  nearest mean, 0 of 288 with LDA). v2 selects nothing for Stage 2, as
  declared.
- **Attenuation (reading A).** In the constant-width linear arm, nearly every
  interior T3 failure meets the declared sufficiency rule, 67 cases in all:
  the reference and LDA both fail under absolute `σ₀ = 0.1` and both pass
  under relative noise. Rings of constant width keep the class information.
  Its size relative to the fixed readout floor is what is lost. The
  analytic predictions (contract section 8) held: the F-high reference at
  layer `L` is at most 0.48 for every `L ≥ 3`.
- **Geometric loss (reading G).** For F-high, under relative noise, the
  shrinking geometry loses class information that constant width keeps. This
  is **shown** on every shrinking ring of 11 or fewer units, on 20 of 36
  cases with 16-unit rings, and on none with 21 or more. It never appears for
  F-low (largest reference difference 0.001).
- **Readout limits (reading R).** Under relative noise, nearest mean is
  **limiting** for F-high in 37 of 48 constant-width and 27 of 48 shrinking
  linear cases. LDA and mode projection sit at the reference. With constant
  width, `μ = 0`, `γ = 1` and `α = 1`, nearest mean falls to 0.24–0.33 by
  layer 3, while LDA and projection stay at 1.00 at every layer (prediction
  E3). Under absolute noise nearest mean is rarely limiting, because every
  readout falls with the reference.
- **Covariance estimation** limits LDA in only 3 linear cases. Projection
  outperforms LDA in 75 rows; per the contract, that is a possible estimation
  or model limitation, not an identified cause.
- **Field drive.** In the linear arm it changes little (prediction E4). In
  the double-well arm it removes the tie's amplification of F-high nuisance
  (suppression ratio about 1.0 instead of about 2), but it does not suppress
  nuisance: T2 fails in all 96 double-well field configurations.

[Contract](layered-module-v2-contract.md) (sha256 `eb497614…`) ·
[manifest](../reports/layered-module-v2-manifest.json) ·
[results](../reports/layered-module-v2-results.json) ·
[runtime](../reports/layered-module-v2-runtime.json).

**Provenance.**

- The contract, manifest, code and tests were committed before any scored
  computation.
- A first scored run, and a regeneration run started with it, were stopped
  by request a few minutes in. Neither wrote an output file, and no scored
  values were inspected.
- Nothing changed before the run was repeated. A regeneration run in a
  separate directory ran alongside it, and its manifest and results files
  are byte-identical.

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Contract hash, pinned environment (numpy 2.5.4, scipy 1.18.1, one BLAS thread), v1 results hash | yes |
| Graph construction, both geometries (constant width: radial edges exactly `j ↔ j`) | yes |
| Analytic gradient and Hessian against finite differences, both drives | max error 5.2e−7 |
| Linear arm: every canonical state equals the direct sparse solve, both drives | max deviation 1.8e−14 |
| Per-mode gains: pure `cos`/`sin` responses equal `g_n(ℓ)` × pattern | max deviation 1.0e−13 |
| Reference procedure: closed form inside the Clopper–Pearson interval (level `1 − 10⁻⁶`) | 2,160 / 2,160 |
| LDA: Cholesky against explicit inverse; invariance under a linear transform | 3.6e−15; yes |
| Attenuation sanity and linear superposition (v1 gates) | yes |
| v1 replication (80 configurations; certification, `S`, `P`, ratios, NM accuracy, minima, superposition) | 80 / 80 exact |
| All canonical settlings certified | 115,200 / 115,200 |
| Byte-identical regeneration (manifest, results) | yes |
| Tests | 19 new; full suite 306 passing |

## Results

### Profile

No configuration meets the profile with either readout. Switching the T3
readout from nearest mean to LDA changes only F-low counts, by at most 9.
F-high T3 fails everywhere with both. Full counts, including T1, are in
`summary.profile`.

| Arm / geometry / drive | Configs | F-high T3 fails (NM / LDA) | F-low T3 fails (NM / LDA) | T2 fails (F-low, F-high) |
| --- | --- | --- | --- | --- |
| linear / shrinking / tie | 32 | 32 / 32 | 18 / 18 | 4, 4 |
| linear / constant / tie | 32 | 32 / 32 | 15 / 14 | 4, 4 |
| linear / shrinking / field | 16 | 16 / 16 | 8 / 9 | 0, 0 |
| linear / constant / field | 16 | 16 / 16 | 9 / 10 | 0, 0 |
| double well / shrinking / tie | 48 | 48 / 48 | 24 / 21 | 48, 48 |
| double well / constant / tie | 48 | 48 / 48 | 42 / 33 | 48, 48 |
| double well / shrinking / field | 48 | 48 / 48 | 19 / 16 | 48, 48 |
| double well / constant / field | 48 | 48 / 48 | 30 / 25 | 48, 48 |

### Attenuation: prediction check and reading A

At layer `L`, absolute `σ₀ = 0.1`, the empirical LDA accuracy lies within 0.1
of the closed-form reference in 77 of 96 constant-width linear rows.

- **All 19 larger differences have LDA below the reference**, by 0.11–0.21.
  LDA never exceeds the reference by more than 0.04.
- In these 19 rows, the mode-projection readout is within 0.07 of the
  reference. Across all constant-width linear rows and layers it falls short
  of the reference by 0.005 on average (0.043 for LDA, 0.034 for nearest
  mean). This locates the shortfall in the full-dimensional readouts. Per the
  contract, these differences are reported as unexplained.
- Under relative noise `σ_rel = 0.1`, the reference is 1.000 at every layer
  of every constant-width linear configuration (prediction E2). LDA,
  projection and oracle LDA match it exactly.

**Reading A**: 67 sufficient, 29 not shown, 0 not evaluable.

- 26 of the 29 "not shown" rows are F-low cases where the reference passes
  T3, so there is no failure to explain.
- The other 3 are F-high:
  - tie, `μ=0`, `γ=1`, `L=2`, at both `α`: the reference passes (0.914,
    0.935) but LDA fails (0.85, 0.86). This is the borderline case named in
    E1.
  - tie, `μ=0`, `γ=2`, `L=6`, `α=1`: relative-noise LDA reaches 0.89.

### Geometry: reading G

Under relative noise `σ_rel = 0.1`, F-high outcomes by the size of the
shrinking ring at that layer (linear arm, all drives, `γ`, `α`):

| Shrinking ring size | 5 | 8 | 11 | 16 | 21 | 24 | 27 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shown | 12 | 12 | 24 | 20 | 0 | 0 | 0 |
| not shown | 0 | 0 | 0 | 16 | 24 | 12 | 12 |

Example: tie, `μ=0`, `γ=1`, `α=1`. Reference / LDA on the shrinking ring;
both are 1.00 on the matched constant-width ring:

| `L`, layer (ring size) | Reference | LDA |
| --- | --- | --- |
| 2, 2 (16) | 0.785 | 0.73 |
| 4, 3 (16) | 0.781 | 0.78 |
| 6, 4 (16) | 0.964 | 0.91 |
| 3, 3 (11) | 0.567 | 0.56 |
| 4, 4 (8) | 0.326 | 0.33 |
| 6, 6 (5) | 0.309 | 0.32 |

On 16-unit rings the outcome depends on the configuration, which fits
mode 8 being the Nyquist mode there: the cosine component survives and the
sine component does not. Wider rings at the same layer can still feed it
through radial coupling. "Predicted but not confirmed" never occurs: in
every linear row where the reference difference reaches 0.10, the held-out
LDA difference does too.

For the double-well arm, reading G is **not evaluable** in 504 of 528 rows.
In constant width, the pure-signal drives that set the relative-noise scale
settled on saddles in 172 of 192 configuration–family pairs. These drives
are exactly symmetric, so units where the drive vanishes stay at the
unstable point `u = 0`. (Canonical drives carry nuisance, which breaks the
symmetry; all of them certified.) The 24 evaluable double-well rows (F-low,
`L ≤ 3`) are all "not indicated".

### Readout: reading R

Rows at layer `L`, linear arm (limiting = `LDA − NM ≥ 0.10`):

| Geometry | Noise | F-high NM limiting | F-low NM limiting |
| --- | --- | --- | --- |
| constant | absolute 0.1 | 2 / 48 | 1 / 48 |
| constant | relative 0.1 | **37 / 48** | 4 / 48 |
| shrinking | absolute 0.1 | 1 / 48 | 2 / 48 |
| shrinking | relative 0.1 | **27 / 48** | 8 / 48 |

Depth profile, constant width, tie, `μ=0`, `γ=1`, `α=1`, F-high, relative
noise. Values are NM / LDA / projection by layer, exterior first:

| `L` | Layers |
| --- | --- |
| 4 | 1.00/1.00/1.00, 0.61/1.00/1.00, 0.31/1.00/1.00, 0.24/1.00/1.00 |
| 6 | 1.00/1.00/1.00, 0.60/1.00/1.00, 0.33/1.00/1.00, 0.30/1.00/1.00, 0.24/1.00/1.00, 0.25/1.00/1.00 |

- **Covariance-estimation control:** limiting in 3 of 384 evaluable linear
  rows, all at tie, `μ=0`, `γ=2`, `L=6` (oracle − LDA = 0.11–0.12).
- **Projection outperforms LDA** in 75 rows: 22 linear constant-width and
  53 double-well constant-width. No cause is identified. On shrinking
  geometry the projection is not evaluable for F-high (rings ≤ 16 or
  undefined at `L`).
- **Double-well arm:** nearest mean is limiting in 16–31 of 96 rows per
  family and geometry under absolute noise.

### Spectra (where the calm sits)

Constant width, tie, `μ=0`, `γ=1`, `L=4`, `α=1`, F-high. `P_k` is
within-class variance and `B_k` between-class variance for mode `k`:

| Layer | `P_0` | `P_1` | `P_8` | `B_8` | total `P` |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.497 | 0.541 | 0.068 | 1.11 | 2.77 |
| 2 | 0.497 | 0.279 | 0.0049 | 0.080 | 1.18 |
| 3 | 0.497 | 0.116 | 0.00035 | 0.0057 | 0.73 |
| 4 | 0.497 | 0.028 | 0.000022 | 0.00035 | 0.55 |

- With `μ = 0`, the DC component of nuisance is not attenuated at all. At
  layer 4 it is 91% of the within-class variance. This sets the floor of the
  suppression ratio near 0.2.
- The F-high signal (`B_8`) shrinks by about 0.07 per layer, exactly with
  its in-band nuisance (`P_8`). Its ratio to in-band nuisance stays constant
  (about 16), as the linear analysis says it must.
- With `μ = 1`, every mode is attenuated, including DC. The suppression ratio
  falls to 0.001 at `L = 4`, and F-low distinctions fail too.

### Field versus tie

Matched differences (field − tie), strict accuracy at layer `L`, absolute
`σ₀ = 0.1`, mean over configurations:

| Arm | Geometry | F-high NM / LDA | F-low NM / LDA |
| --- | --- | --- | --- |
| linear `μ=1` | constant | +0.02 / +0.02 | +0.09 / +0.09 |
| linear `μ=1` | shrinking | +0.02 / +0.01 | +0.06 / +0.08 |
| double well | constant | +0.01 / −0.02 | +0.16 / +0.10 |
| double well | shrinking | −0.01 / −0.01 | +0.04 / +0.08 |

Double-well suppression ratio `P(L)/P(1)`, median over configurations:

| Geometry / drive | F-low | F-high | Max distinct minima |
| --- | --- | --- | --- |
| constant / tie | 2.46 | 2.09 | 4 |
| constant / field | 1.36 | 1.01 | 7 |
| shrinking / tie | 1.86 | 1.75 | 6 |
| shrinking / field | 0.69 | 0.98 | 10 |

Without the exterior anchor, the field drive stops the interior from
amplifying nuisance relative to the exterior. It is more multistable,
though, and it never reaches T2.

## Readings

1. **Attenuation is sufficient for the constant-width interior failure.**
   Under a fixed readout floor, the constant-width linear module fails T3
   for F-high from `L = 3` with any readout tested, and the linear-Gaussian
   reference fails too. When noise scales with the signal, every
   covariance-aware readout recovers the classes at every layer. The
   information is present but small. This answers v1's first candidate
   explanation for this geometry.
2. **The shrinking geometry adds a separate geometric loss for mode 8 on
   rings of 16 units or fewer.** This is shown by both the population
   reference and held-out LDA, with attenuation factored out. On rings of 21
   or more units no such loss appears. v1's F-high collapse on small inner
   rings therefore combined attenuation and geometric loss. v1's
   Mahalanobis clue for the 24- and 27-unit rings is now confirmed by
   held-out decoding, under relative noise.
3. **The declared nearest-mean readout is limiting once attenuation is
   factored out.** Under relative noise it falls to chance for deep F-high
   layers, while LDA and projection stay perfect. The spectra show deep
   layers dominated by low-mode (especially DC) nuisance, which is
   consistent with E3's mechanism: estimated class means carry
   out-of-band nuisance. The control that isolates covariance estimation
   rarely fires, so covariance estimation is not the main limit on LDA.
4. **The field drive does not rescue the double well.** It removes the tie's
   nuisance amplification but adds multistability and still never
   suppresses nuisance.

## What this does not establish

- The reference is a linear-Gaussian reference, not a strict ceiling.
  Across all linear rows, layers and noise levels, held-out readouts exceed
  it by at most 0.13. That is within sampling variation for 100 test drives.
- Only broadband nuisance was tested. Whether a fixed module can calm
  nuisance that differs from the signal in band, amplitude or time is
  untested.
- The double-well geometry comparison is almost entirely not evaluable,
  because of the saddle settlings described above.
- Anything about learning, temporal retention, coupled modules, the core or
  abstraction.

## Accounting

| Item | Value |
| --- | --- |
| Configurations / canonical settlings | 288 / 115,200 |
| Pure-signal, multistability and superposition settlings | 2,304 / 36,864 / 6,912 |
| Reference evaluations (Monte Carlo, `10⁶` draws each) | 4,320 |
| Grid and gates wall time | ≈1,795 s (one BLAS thread) |
| Largest module | 193 units (constant width, `L = 6`) |

## Next obligations (not run)

1. The stopping rule applies. Do not retune thresholds, readouts, noise
   levels or geometries inside v2.
2. **For Stage 2 design:** whatever reads a satellite's interior (the core's
   coupling) has to cope with signal that shrinks with depth and with
   low-mode nuisance that does not. A gain-normalized or covariance-aware
   coupling is indicated; an unweighted comparison is not. Rings should not
   shrink below the width the finest declared distinction needs.
3. **Deferred arms** (contract section 12) remain open: band-limited
   nuisance, learned or adapted per-mode gains, and other nonlinearities.
4. A future contract that uses a pure-signal reference with a nonlinear
   potential should declare how symmetric drives avoid saddles, for example
   a declared symmetry-breaking start, or a scale taken from canonical
   (nuisance-carrying) drives.
