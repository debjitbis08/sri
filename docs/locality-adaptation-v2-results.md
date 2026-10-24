# Locality adaptation v2 — cross-validated selection with refit passes H1–H3 and misses H4 by 0.002; overall `fail`

**Completion:** `complete`. All correctness gates passed (34 tests, including
v1's unchanged suite), v1's frozen sources matched the v1 manifest, the
repeated development run was byte-identical, and all 64 confirmation worlds
were scored with no solver, rank or nonfinite failure.

**Headline:**

- **Primary outcome: `fail`.** H1–H3 pass. H4 (aligned reconstruction
  noninferiority) misses its threshold by 0.0018. All four were required.

| Claim | Regime / score | Mean improvement | One-sided 98.75% lower bound | Threshold | Outcome |
| --- | --- | --- | --- | --- | --- |
| H1 recover shifted signal | half-shifted / L | 0.350 | 0.330 | > 0.02 | pass |
| H2 preserve aligned signal | aligned / L | −0.011 | −0.017 | > −0.02 | pass |
| H3 improve shifted usefulness | half-shifted / E | 0.368 | 0.346 | > 0.02 | pass |
| H4 preserve aligned usefulness | aligned / E | −0.014 | **−0.022** | > −0.02 | **fail** |

  Improvement is `fixed_cca_all − cv_refit_cca` in the mean over
  n ∈ {16,…,256}, computed per world (t₆₃ quantile 2.296, SE of the H4
  difference 0.0032).
- **What the failure means.** As the contract states, this four-fold
  CV-and-refit procedure did not establish aligned reconstruction
  noninferiority at the −0.02 margin. Non-significance is inconclusive. It
  does not show that the procedure is inferior by more than 0.02, and it
  sets no lower bound on the cost of phase adaptation in general.
- **Where the procedure loses.** The aligned deficit is almost entirely at
  n=16. There the candidate selects a population-optimal phase in 42/64
  worlds, and E is 0.650 versus 0.578 for fixed. From n=32 on, selection is
  optimal in 63–64/64 worlds and the curves agree to within 0.001. A
  0.072 deficit at one of five budgets contributes 0.0144 to the 0.0145 mean.
- **Refit and cross-validation both help, and refit helps more** (descriptive,
  aligned, 95% t intervals):
  - refitting removes 0.027 L and 0.032 E (`holdout_refit − holdout`);
  - cross-validation removes a further 0.0105 L [0.005, 0.016] and 0.012 E
    [0.005, 0.019] (`cv_refit − holdout_refit`);
  - the candidate stays 0.011 L above the finite-sample ceiling (minimum L
    over the eight all-n fits); holdout-refit stays 0.022 above it.
- **Shifted recovery improved over v1.** Signal loss went from 0.170 to
  0.132, because the deployed fit now uses every pair. `holdout_cca`
  replicates v1's `adaptive_cca` on fresh seeds (L 0.1695 versus 0.170,
  unpaired). In the dense regime global CCA remains far better (candidate
  worse by 0.349 L), and the best-phase floor is 0.495.
- **Computation:** cross-validated selection costs 33 local fits. At d=32 the
  fit medians are 25.6 ms, versus 6.9 ms holdout-refit, 0.73 ms fixed and
  0.41 ms global. Deployed state and inference are identical to v1.

The study supports this much: under this generator, validated phase
selection followed by refit recovers translated structure. Its aligned
signal-loss cost is noninferior at a 0.02 margin. Its aligned reconstruction
cost could not be shown to be within that margin; the gap is concentrated at
the smallest primary budget. It establishes nothing about dense mixing,
real data, reciprocal dynamics, retained context or subspace drift.

[Contract](locality-adaptation-v2-contract.md) (sha256 `00099acca6bd…`) ·
[manifest](../reports/locality-adaptation-v2-manifest.json) (sha256 `68d7a44c7448…`) ·
[results](../reports/locality-adaptation-v2-results.json) ·
[development](../reports/locality-adaptation-v2-development.json) ·
[runtime](../reports/locality-adaptation-v2-runtime.json) ·
[v1 results](locality-adaptation-v1-results.md)

## Procedure and disclosures

- Contract committed in `803f4bf` after one review round and before any
  implementation. Implementation, development report and manifest were
  committed in `2744282` before any confirmation world was generated. The
  runner refuses to confirm or time unless v2 and v1 frozen sources match
  their hashes and are committed.
- v2 calls v1's generator, estimators and metrics unchanged. Its root seed
  (`2026102408`) is installed into v1's module-level seed before any draw. A
  gate test checks that the drawn stream equals a direct
  `SeedSequence([2026102408, stage, w, part])` draw and differs from v1's.
- Development worlds showed the same aligned pattern (development means
  −0.019 L, −0.021 E). Nothing was tuned in response.
- Environment as in v1: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1 (OpenBLAS,
  one thread), AMD Ryzen 5 3600, float64, one process. Runtime records now
  list the exact dirty paths. During confirmation the only one was the
  results file being written; during timing it was the results and runtime
  JSON.

## Primary scores (mean over n = 16–256, 64 worlds)

| Arm | aligned L | aligned E | half-shifted L | half-shifted E | dense L | dense E |
| --- | --- | --- | --- | --- | --- | --- |
| `cv_refit_cca` | 0.132 [0.124, 0.140] | 0.396 [0.387, 0.406] | 0.132 | 0.396 | 0.645 | 0.979 |
| `fixed_cca_all` | 0.121 [0.114, 0.127] | 0.382 [0.375, 0.389] | 0.482 | 0.764 | 0.675 | 1.011 |
| `holdout_refit_cca` | 0.142 | 0.408 | 0.142 | 0.408 | 0.650 | 0.987 |
| `holdout_cca` | 0.170 | 0.440 | 0.170 | 0.440 | 0.661 | 1.012 |
| `oracle_structural_phase_cca_all` | 0.121 | 0.382 | 0.121 | 0.382 | 0.648 | 0.983 |
| `global_cca_all` | 0.297 | 0.605 | 0.297 | 0.605 | 0.297 | 0.605 |

Unknown shift (secondary): fixed L 0.398, E 0.678. Adaptive arms match their
aligned values because circular shifts permute the phase bank. Controls in
distribution: zero output 1, raw input 6, oracle signal projector 0.25.

The structural-phase reference is not a ceiling. In the dense regime the
candidate is slightly better than it (−0.0025 L, interval spanning 0),
because another phase can fit better on a given sample.

## Signal-loss diagnosis (L only)

Mean terms of `L(selected) = floor + fitting + selection`. Fitting is
identical for both refit arms, since they share the eight all-n fits.

| n | aligned fitting | cv_refit selection | holdout_refit selection | cv_refit optimal phase | dense floor | dense cv_refit selection |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.508 | 0.121 | 0.130 | 15/64 | 0.495 | 0.038 |
| 8 | 0.427 | 0.084 | 0.110 | 31/64 | 0.495 | 0.038 |
| 16 | 0.294 | 0.056 | 0.095 | 42/64 | 0.495 | 0.036 |
| 32 | 0.160 | 0.001 | 0.014 | 63/64 | 0.495 | 0.032 |
| 64 | 0.086 | 0.000 | 0.000 | 63/64 | 0.495 | 0.017 |
| 128 | 0.043 | 0.000 | 0.001 | 64/64 | 0.495 | 0.009 |
| 256 | 0.020 | 0.000 | 0.000 | 64/64 | 0.495 | 0.006 |

Reconstruction is diagnosed through L and projected noise, not by
substituting E. On aligned worlds the candidate's projected noise exceeds
fixed's by only 0.003 (0.264 versus 0.261). Most of its E deficit is signal
loss from wrong-phase selections at n=16.

Classification per section 9:

- **Representation:** zero floor for aligned and translated worlds; 0.495
  in the dense regime.
- **Full-data fitting:** this is the dominant term at every budget, and it
  is shared with the fixed code when phase 0 is chosen.
- **Selection:** material only at n ≤ 16. Cross-validation reduced it
  relative to holdout at every budget where holdout's was nonzero.
- **Usefulness:** shifted gains carry over to E. Under texture variance 16
  the aligned NMSE increases are 0.043 candidate, 0.033 fixed and 0.175
  global. Under extra noise they are 0.257, 0.236 and 0.564. These are
  descriptive only.

## Computation

Confirmation took 77.9 s wall and 77.9 s CPU in one process, with a parent
peak RSS of 188 MiB (cumulative, not per arm). Candidate fit-stage time
summed to 38.3 s for `cv_refit_cca`. That covered 59,136 local fits plus
1,792 refits, 236,544 Cholesky and 236,544 eigh of 8×8, 57,344 candidate
scores, 3.25 M fit-row exposures and 1.04 M validation-row exposures, from
256 distinct pairs per world and regime. For comparison: holdout-refit
10.4 s, `holdout_cca` 9.3 s, fixed 1.1 s, global 0.5 s. The evaluator's
diagnostic all-n fits took 9.4 s and are counted separately, not as
candidate cost.

Timing panel (stage 2, n=256, k=d/8, aligned; dense within ~7%). Medians
are over 7 fresh fits after 1 warm-up.

| d | arm | fit+select | 1-input inference | persistent bytes | peak NumPy fit bytes | total at Q=10⁴ (estimate) |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | cv_refit | 25.6 ms | 14.0 µs | 256 + 24 | 137 kB | 0.166 s |
| 32 | holdout_refit | 6.9 ms | 14.1 µs | 256 + 24 | 135 kB | 0.148 s |
| 32 | fixed | 0.73 ms | 14.0 µs | 256 + 24 | 41 kB | 0.141 s |
| 32 | global | 0.41 ms | 2.7 µs | 1,024 | 89 kB | 0.028 s |
| 128 | cv_refit | 87.8 ms | 14.5 µs | 1,024 + 24 | 438 kB | 0.233 s |
| 128 | global | 2.5 ms | 3.1 µs | 16,384 | 1.3 MB | 0.034 s |
| 512 | cv_refit | 412 ms | 20.8 µs | 4,096 + 24 | 1.6 MB | 0.620 s |
| 512 | holdout_refit | 112 ms | 22.6 µs | 4,096 + 24 | 1.6 MB | 0.339 s |
| 512 | fixed | 11.7 ms | 21.0 µs | 4,096 + 24 | 49 kB | 0.221 s |
| 512 | global | 80.0 ms | 9.1 µs | 262,144 | 21 MB | 0.171 s |

Cross-validated fitting costs about 3.7× holdout-refit and 35× fixed. Its
peak NumPy bytes equal holdout-refit's, because fold candidates are discarded
after scoring. Global CCA is cheapest in amortized total at every measured d
in this NumPy implementation. Local codes keep a 64× persistent-byte
advantage at d=512. Totals are estimates. LAPACK workspace and FLOPs are not
measured. This panel is descriptive, not a speed hypothesis.

## Interpretation and scope

- Refit was the main repair. v1's aligned signal-loss deficit of 0.049
  becomes 0.011 here (on different seeds, so unpaired). CV selection adds a
  smaller, measurable improvement.
- The remaining cost is selection error at n=16, where four folds of four
  pairs each must distinguish phases whose fitted quality differs little.
  Within this contract that does not license more folds, a different margin
  or budget-specific selection.
- For dense mixing, and for total cost in this implementation, global CCA
  remains the better choice. Local phase adaptation is the better
  quality-per-pair choice for translated structure at these budgets. Its
  advantages are 64× smaller deployed state and, in principle, O(d)
  inference; neither turned into a speed advantage here.
- The contract tied the next step (source-to-target reuse and subspace drift)
  to a pass, and this run did not pass. Whether to proceed anyway with
  `cv_refit_cca` as a disclosed local reference, or to address small-budget
  selection first in a new version, is a decision for a new contract. This
  result does not settle it.

## Reproduction

```
.venv/bin/python -m unittest tests.test_locality_adaptation_v2 tests.test_locality_adaptation -v
.venv/bin/python -m experiments.locality_adaptation_v2 --phase dev
.venv/bin/python -m experiments.locality_adaptation_v2 --phase freeze   # then commit
.venv/bin/python -m experiments.locality_adaptation_v2 --phase confirm
.venv/bin/python -m experiments.locality_adaptation_v2 --phase timing
```
