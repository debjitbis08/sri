# Locality adaptation v1 — phase selection recovers shifted structure but fails aligned noninferiority; overall `fail`

**Completion:** `complete`. All correctness gates passed (22 tests), the
repeated development run was byte-identical, and all 64 confirmation worlds
were scored with no solver, rank or nonfinite failure.

**Headline:**

- **Primary outcome: `fail`.** H1 and H3 pass by a wide margin. H2 and H4
  fail. All four were required.

| Claim | Regime / score | Mean improvement | One-sided 98.75% lower bound | Threshold | Outcome |
| --- | --- | --- | --- | --- | --- |
| H1 recover shifted signal | half-shifted / L | 0.311 | 0.291 | > 0.02 | pass |
| H2 preserve aligned signal | aligned / L | −0.049 | −0.058 | > −0.02 | **fail** |
| H3 improve shifted usefulness | half-shifted / E | 0.321 | 0.298 | > 0.02 | pass |
| H4 preserve aligned usefulness | aligned / E | −0.060 | −0.071 | > −0.02 | **fail** |

  Improvement is `fixed_cca_all − adaptive_cca` in the mean of
  n ∈ {16,…,256}, computed per world (t₆₃ quantile 2.296).
- **The aligned failure is mostly the cost of withholding data.** It is not
  wrong selection. Of the 0.049 aligned signal-loss deficit, 0.033 is the cost
  of fitting on 3n/4 pairs (`fixed_cca_fit − fixed_cca_all`). Phase
  selection on top of that adds 0.016. For NMSE the split is 0.041 and 0.019.
  The selection term goes to zero by n=128. At n=16 (4 validation pairs) it
  is 0.065.
- **Phase selection removes the representational floor where the bank
  contains it.** Every world's half-shifted and unknown-shift best-phase floor
  is 0. Phase 0's floor is 0.356 and 0.249 respectively. At n=256 the selected
  phase is population-optimal in 64/64 worlds for both regimes. Because
  circular shifts permute the phase bank, adaptive CCA's scores are identical
  across aligned, half-shifted and unknown-shift regimes.
- **Within this generator, adaptive local CCA beats global CCA at these
  budgets in every non-dense regime** (descriptive). Its signal loss is
  0.122 lower [0.113, 0.131] and its NMSE 0.160 lower [0.148, 0.171]. In the
  dense regime global CCA is far better (adaptive is worse by 0.369 L and
  0.417 E). As predicted, the best-phase floor there is 0.491.
- **Computation: adaptation costs ≈8× fixed fitting and buys no deployment
  advantage over global CCA in this implementation.** At d=32 the medians
  are 6.2 ms for adaptive fit+selection, 0.74 ms for fixed CCA and 0.41 ms
  for global CCA. NumPy single-input inference is about 14 µs local versus
  2.8 µs global. The asymptotic O(d) versus O(dk) advantage is swamped by
  gather/allocation overhead at these sizes.

This establishes that a validated choice among eight partition phases
recovers translated broad structure under this generator, after paying for
validation examples and all candidate fits. It does **not** establish
adaptation without an aligned cost: under the declared equal-pair rule,
adaptive CCA is measurably worse than fixed locality when the fixed partition
is already right. It establishes nothing about dense mixing, real data,
reciprocal dynamics, retained context or subspace drift.

[Contract](locality-adaptation-v1-contract.md) (sha256 `1231719cd110…`) ·
[manifest](../reports/locality-adaptation-v1-manifest.json) (sha256 `24ad9aa45564…`) ·
[results](../reports/locality-adaptation-v1-results.json) ·
[development](../reports/locality-adaptation-v1-development.json) ·
[runtime](../reports/locality-adaptation-v1-runtime.json)

## Procedure and disclosures

- The contract was reviewed and edited for clarity only before
  implementation (commit `61fe6ca`). See its "Pre-implementation review"
  note. No threshold, arm, budget, regime, seed or decision rule changed.
- Implementation, development report and manifest were committed in
  `7e42b6d` before any confirmation world was generated. The confirmation
  runner refuses to start if any frozen source differs from its manifest hash
  or is uncommitted.
- Development worlds were used only for correctness. They showed the aligned
  deficit (development mean −0.018 L, −0.027 E). Nothing was tuned in
  response, as the contract requires.
- Environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1 (OpenBLAS, one
  thread), AMD Ryzen 5 3600, float64, one process. The `.venv` was recreated
  with `uv` for this run. Its Python build date differs from the one recorded
  in earlier receipts. Torch was not installed, so the earlier
  `tests.test_learning_benchmark` suite was not rerun. This experiment does
  not import it.
- Runtime records mark `worktree_dirty: true` for the confirmation and timing
  phases. The only uncommitted files were the result/runtime JSON being
  written. Frozen sources were verified clean by hash and `git status`.

## Primary scores (mean over n = 16–256, 64 worlds, 95% t intervals)

| Arm | aligned L | aligned E | half-shifted L | half-shifted E | dense L | dense E |
| --- | --- | --- | --- | --- | --- | --- |
| `adaptive_cca` | 0.170 [0.159, 0.181] | 0.443 [0.430, 0.456] | 0.170 | 0.443 | 0.661 | 1.019 |
| `fixed_cca_all` | 0.121 [0.116, 0.126] | 0.383 [0.377, 0.389] | 0.481 | 0.764 | 0.668 | 1.018 |
| `fixed_cca_fit` | 0.154 | 0.424 | 0.504 | 0.799 | 0.683 | 1.047 |
| `global_cca_all` | 0.292 | 0.603 | 0.292 | 0.603 | 0.292 | 0.603 |
| `global_cca_fit` | 0.332 | 0.671 | 0.332 | 0.671 | 0.332 | 0.671 |
| `adaptive_spectral` | 0.208 | 0.973 | 0.208 | 0.973 | 0.660 | 1.570 |
| `fixed_spectral_all` | 0.133 | 0.803 | 0.493 | 1.343 | 0.657 | 1.527 |
| `global_spectral_all` | 0.304 | 1.429 | 0.304 | 1.429 | 0.304 | 1.429 |

Unknown-shift (secondary): fixed CCA L 0.374, E 0.654. Adaptive matches its
aligned values. Descriptive controls, in distribution: zero output NMSE 1,
raw input 6, oracle signal projector 0.25. Global arms are identical across
regimes because every regime is an orthogonal transform of the same world.

The spectral arms keep signal nearly as well as CCA but take in much more
nuisance (projected noise 0.67–1.13 versus 0.26–0.34). This is the case
section 3 warned about: a good validation score is not the same as useful
reconstruction. `adaptive_spectral` does not rescue or alter the primary result.

## Failure diagnosis (adaptive CCA, mean signal-loss terms)

| n | aligned fitting | aligned selection | dense floor | dense fitting | dense selection | aligned selected optimal |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.524 | 0.103 | 0.491 | 0.272 | 0.043 | 16/64 |
| 16 | 0.362 | 0.065 | 0.491 | 0.234 | 0.031 | 35/64 |
| 32 | 0.209 | 0.022 | 0.491 | 0.189 | 0.032 | 53/64 |
| 64 | 0.108 | 0.002 | 0.491 | 0.136 | 0.019 | 62/64 |
| 256 | 0.029 | 0.000 | 0.491 | 0.078 | 0.015 | 64/64 |

Classification per section 9:

- **Representation:** zero floor for aligned and translated worlds. The
  dense floor (0.491) is a representational limit that phase choice cannot
  remove.
- **Fitting:** this is the dominant term at every budget. Part of it is
  structural, because adaptive fits use only 3n/4 pairs.
- **Selection:** regret is material only at n ≤ 32, where at most 8
  validation pairs exist.
- **Usefulness:** shifted gains in L carry over to E (H3). Under texture
  variance 16 the NMSE increases are 0.069 adaptive, 0.036 fixed (aligned)
  and 0.182 global. Under the extra-noise stress they are 0.329, 0.236 and
  0.554. These are descriptive only.
- **Computation:** see below. Global CCA is cheaper to fit at d=32 and
  faster to deploy at every measured d in this implementation.

## Computation

Confirmation (64 worlds × 4 regimes × 7 budgets × 8 arms) took 22.8 s wall
and 22.7 s CPU in one process, with a parent peak RSS of 190 MiB (cumulative,
not per arm). Fit-stage time summed over the run was 9.6 s for adaptive CCA
(57,344 Cholesky and 57,344 eigh of 8×8; 14,336 candidate scores), 1.2 s for
fixed CCA all and 0.5 s for global CCA all. Validation scoring took 0.5 s.
Exposures: adaptive CCA used 780,288 fit-row exposures plus 260,096
validation-row exposures, from 256 distinct pairs per world and regime.

Timing panel (stage 2, n=256, k=d/8, aligned; dense within ~10%). Medians
are over 7 fresh fits after 1 discarded warm-up; IQRs were within 1% of the
median.

| d | arm | fit+select | 1-input inference | 256-batch inference | persistent bytes | peak NumPy fit bytes | total at Q=10⁴ (estimate) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | adaptive | 6.2 ms | 14.4 µs | 49 µs | 256 + 24 | 137 kB | 0.150 s |
| 32 | fixed | 0.74 ms | 14.0 µs | 50 µs | 256 + 24 | 41 kB | 0.141 s |
| 32 | global | 0.42 ms | 2.8 µs | 11 µs | 1,024 | 89 kB | 0.028 s |
| 128 | adaptive | 21.6 ms | 14.8 µs | 229 µs | 1,024 + 24 | 438 kB | 0.170 s |
| 128 | global | 2.7 ms | 3.3 µs | 61 µs | 16,384 | 1.3 MB | 0.036 s |
| 512 | adaptive | 99.7 ms | 22.9 µs | 2.18 ms | 4,096 + 24 | 1.6 MB | 0.328 s |
| 512 | fixed | 12.7 ms | 22.7 µs | 2.17 ms | 4,096 + 24 | 49 kB | 0.240 s |
| 512 | global | 82.2 ms | 10.1 µs | 0.78 ms | 262,144 | 21 MB | 0.183 s |

Fit cost grows roughly linearly in d for local arms and superlinearly for
global. Global fitting matches fixed at d=128, is 6.5× slower at d=512, and
nearly reaches adaptive there. Local codes need 64× fewer persistent bytes
at d=512. The local inference path (index gather plus scatter into a new
output) is nevertheless slower than a dense BLAS matmul at every measured
size, so the amortized totals favour global CCA throughout. Those totals are
arithmetic estimates, not end-to-end runs. Spawned-worker peak RSS was
98–135 MiB against a 95 MiB idle-worker baseline. LAPACK workspace is not
measured, and the manifest gives an analytical estimate. FLOPs were not
measured. This panel is descriptive and is not a speed hypothesis.

## Interpretation and scope

- A bounded, validated phase bank is a working correction for **translated**
  locality. It turns the fixed code's 0.31 shifted signal-loss penalty into
  the same loss it achieves when aligned.
- That aligned loss is worse than the fixed code with full data, because 1/4
  of the pairs are withheld for validation and small-budget selection is
  noisy. The contract forbids refitting after selection, so this cost is
  declared and visible rather than hidden. Whether refitting on all n
  pairs, or other selection schemes, removes it is a new question needing a
  new contract. It is not a rescue of this one.
- Global CCA is the escape route for dense mixing. In this implementation it
  is also the cheaper deployment, despite larger state.
- Untested: subspace drift and source-to-target learning, overlapping or
  adaptive memberships, real data, reciprocal dynamics, retained context.
  The dense floor motivates the overlapping/fallback direction. The small
  selection regret does not justify further selection engineering inside
  this study.

## Reproduction

```
.venv/bin/python -m unittest tests.test_locality_adaptation -v
.venv/bin/python -m experiments.locality_adaptation --phase dev
.venv/bin/python -m experiments.locality_adaptation --phase freeze   # then commit
.venv/bin/python -m experiments.locality_adaptation --phase confirm
.venv/bin/python -m experiments.locality_adaptation --phase timing
```
