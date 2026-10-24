# Representation reuse v1 — all four primary claims pass; overall `pass`

**Completion:** `complete`. All correctness gates passed: 56 tests, including
the unchanged v1 and v2 suites. v1 and v2 frozen sources matched their
manifests. The repeated development run was byte-identical. All 64
confirmation worlds were scored with no solver, rank, projector or nonfinite
failure.

**Headline:**

- **Primary outcome: `pass`.** All four bounds clear their margins (s=256,
  window m ∈ {16, 32, 64}, 64 worlds, t₆₃ quantile 2.296).

| Claim | Angle | Difference (positive favours reuse) | Mean | SE | One-sided 98.75% lower bound | Margin | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H1 stable reuse preserves signal | 0° | L(fresh_local_cv) − L(frozen_local) | 0.180 | 0.006 | 0.166 | > 0.02 | pass |
| H2 stable reuse reconstructs better | 0° | E(fresh_local_cv) − E(frozen_local) | 0.203 | 0.008 | 0.186 | > 0.02 | pass |
| H3 updated weights beat frozen | 30° | E(frozen_local) − E(phase_reuse_target) | 0.066 | 0.006 | 0.052 | > 0.02 | pass |
| H4 reused grouping noninferior | 30° | E(fresh_local_cv) − E(phase_reuse_target) | 0.019 | 0.005 | 0.008 | > −0.02 | pass |

- **As predicted before implementation, H1, H2 and H4 passed by wide
  margins.** They confirm what 256 extra paid source pairs and a correct
  source phase are worth. They are not a surprising transfer result. H1/H2
  compare a learner with 256+m pairs of history against one with m pairs.
  H4 passes, and its point estimate even favours reuse, because the source
  phase spares the target the n ≤ 16 selection error that v2 had already
  located.
- **H3 was the informative claim, and it passes.** At 30° drift, refitting
  target weights at the reused phase beats the frozen source code by
  0.066 E. That comes from 0.515 frozen versus 0.449 for phase reuse over
  the window. The win depends on the window: at m=16, phase reuse (0.591) is
  still worse than frozen (0.515). It wins from m=32 on (0.420, then 0.335).
- **History-matched controls do not explain the drift result. They lose to
  it.** At 30° the pooled local fit (source+target sums, s+m pairs) has
  E 0.466. That is better than frozen (0.515) but worse than target-only
  phase reuse (0.449; contrast −0.017 [−0.028, −0.006]). Pooling keeps
  reconstructing the old direction. At 60° and 90°, pooling is 0.49–0.80 E
  worse than phase reuse. At 0° pooling is the best arm (0.269 versus 0.271
  frozen). With a stable subspace, retaining covariance explains the whole
  advantage, exactly as the contract anticipated.
- **The tradeoff is source forgetting.** Phase reuse releases the source
  basis, and its post-update source E rises by 0.18 (0°), 0.38 (30°),
  0.78 (60°) and 0.97 (90°) over the released source model. Frozen arms
  forget nothing. Pooled arms forget at most 0.006.
- **Global controls are worse throughout.** This generator has a stable
  local partition, so that is expected. Frozen global trails frozen local
  by 0.065 E at 0°. Pooled global trails pooled local by 0.045 E at 30°.
  Fresh global trails fresh local CV by about 0.29 E at every angle.

The study supports this much, under this generator only: when the partition
stays fixed and signal directions rotate inside it, keeping the source-selected
phase and refitting weights on target pairs beats freezing at 30°. It is
noninferior to fresh target phase selection, and it does better than pooling
source covariance once drift is material. The cost is losing source
reconstruction, which grows with the drift angle. It establishes nothing about
changing boundaries, dense mixing, correlated coefficients, retained
activation, drift detection or reciprocal dynamics.

[Contract](representation-reuse-v1-contract.md) ·
[manifest](../reports/representation-reuse-v1-manifest.json) ·
[results](../reports/representation-reuse-v1-results.json) ·
[development](../reports/representation-reuse-v1-development.json) ·
[runtime](../reports/representation-reuse-v1-runtime.json) ·
[v2 results](locality-adaptation-v2-results.md)

## Procedure and disclosures

- The draft contract was committed in `c180506`. Review clarifications were
  committed in `6e6adac` before implementation (contract section 10). None
  changed a hypothesis, margin, budget or arm. That section also records the
  expectation stated before implementation that H1, H2 and H4 would pass
  widely.
- Implementation, tests, development report and manifest were committed in
  `677d1f7` before any confirmation world was generated. The runner refuses
  confirmation and timing unless its own frozen sources and the v1/v2
  dependencies match their hashes and are committed.
- Estimators come from frozen v1/v2 code. Source selection calls v2
  `cv_scores` and v1 `select`. Refits and pooled fits solve the exposed sums
  with v1 `cca_top`. Gates check bitwise equality with v2 `fit_cv_refit` and
  v1 `fit_global`. Pooled and phase-reuse projectors match direct
  concatenated fits within 1e-9. Templates reuse v1's part-0 recipe, and a
  gate checks them against v1 `make_world` under a scoped root override.
- The development report includes descriptive primary differences, as v2's
  did. They were 0.166, 0.186, 0.068 and 0.019 on 8 development worlds.
  Nothing was tuned or changed after seeing them.
- Environment: Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1 (OpenBLAS, one
  thread), AMD Ryzen 5 3600, float64, one process. During confirmation the
  only dirty path was the results file being written.

## Window scores at s=256 (mean over m = 16, 32, 64; in-distribution)

E for target, with source E in brackets. Source arms see 256+m pairs; fresh
arms see m.

| Arm | 0° | 15° | 30° | 60° | 90° |
| --- | --- | --- | --- | --- | --- |
| `frozen_local` | 0.271 (0.271) | 0.336 | 0.515 | 1.004 | 1.249 (0.271) |
| `phase_reuse_target` | 0.454 (0.454) | 0.451 | 0.449 | 0.447 | 0.449 (1.240) |
| `pooled_local` | 0.269 (0.269) | 0.320 | 0.466 | 0.939 | 1.248 (0.271) |
| `frozen_global` | 0.336 | 0.396 | 0.562 | 1.016 | 1.244 |
| `pooled_global` | 0.327 | 0.374 | 0.511 | 0.952 | 1.238 |
| `fresh_local_cv` | 0.474 (0.474) | 0.473 | 0.467 | 0.458 | 0.461 (1.242) |
| `fresh_global` | 0.765 | 0.764 | 0.759 | 0.753 | 0.754 |

Target signal loss L at 30°: frozen local 0.264, phase reuse 0.178, pooled
local 0.214, fresh local CV 0.192. Controls (in distribution): zero output 1,
raw input 6, oracle target projector 0.25. The frozen oracle source projector
has L = sin²θ and E = sin²θ + 0.25, which is 0.50 at 30°. The structural floor
at the true phase is zero at every angle to 1e-15. Source phase selection was
correct in 64/64 worlds at s=256, 62/64 at s=64 and 52/64 at s=16.

At 30°, E learning curves (s=256):

| m | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `frozen_local` | 0.515 | 0.515 | 0.515 | 0.515 | 0.515 | 0.515 | 0.515 |
| `phase_reuse_target` | 0.988 | 0.792 | 0.591 | 0.420 | 0.335 | 0.296 | 0.273 |
| `pooled_local` | 0.508 | 0.503 | 0.492 | 0.471 | 0.434 | 0.382 | 0.326 |
| `fresh_local_cv` | 1.153 | 0.912 | 0.643 | 0.424 | 0.336 | 0.296 | 0.273 |
| `fresh_global` | 1.705 | 1.280 | 0.961 | 0.760 | 0.556 | 0.421 | 0.337 |

Phase reuse and fresh CV converge once fresh CV selects the true phase. Its
true-phase rate at 30° is 23/64 at m=4, 46/64 at m=16 and 62–64/64 from m=32.
For m ≤ 8, the frozen and pooled codes are better than any target-only fit.

**Smaller source budgets (descriptive).** At s=16, frozen local at 0° has
E 0.624 and is worse than fresh local CV (0.474). Sixteen source pairs do
not buy a better representation than 16–64 target pairs. Pooled local is
still the best arm at 0° for s=16 (0.413), because it uses s+m pairs. At
s=64, frozen local at 0° has E 0.340. Phase reuse is unaffected by s except
through p̂: its 30° E is 0.494 at s=16 (12/64 wrong phases), 0.450 at s=64
and 0.449 at s=256.

**Evaluation stresses (unchanged projectors, descriptive).** Under low noise
(isotropic 0.01) at 30°, phase reuse has E 0.209 and frozen local 0.275.
Under texture variance 16, they have 0.510 and 0.519. Arms that keep more
signal also project more texture. That shrinks the 30° phase-reuse gain
under texture-16 to 0.009. At 0°, frozen local is better under both stresses
(0.031 and 0.276, against 0.214 and 0.511).

## Computation

Confirmation took 95.1 s wall in one process, with a parent peak RSS of
251 MiB (cumulative, not per arm). Executed work, counted once per actual
execution:

- **Source acquisition:** 192 local CV acquisitions (64 worlds × 3 budgets)
  in 8.4 s. That is 6,336 local fits, of which 192 are refits, plus 6,144
  candidate scores and 21,504 independent source pairs. The 192 global
  acquisitions took 0.07 s.
- **Fresh local CV:** 2,240 CV runs (one per world, angle and checkpoint),
  51 s. That is 73,920 local fits and 4.06 M fit-row exposures from 81,920
  target-pair arrivals.
- **Updating source arms:** 245,760 single-pair updates and 6,720 checkpoint
  solves each. Phase reuse took 8.1 s, pooled local 8.0 s and pooled global
  6.5 s.

Every source arm is charged the full standalone acquisition once per branch,
even where execution was shared. The closed-form per-branch charges are in
the runtime JSON (`charged_counts_per_branch`), and gate tests check them
against measured counters. Frozen arms do no target work.

Timing panel: stage 2, s=256, θ=30° (0° is within noise of it), k=d/8. Each
figure is the median of 7 fresh executions after 1 warm-up. The endpoint is
an isolated run from the source checkpoint: copy the state, apply m updates,
solve once. Totals are arithmetic estimates using the m=32 endpoint.

| d | arm | acquisition | endpoint m=32 | endpoint m=256 | 1-input inference | deployed bytes | state after 256 updates | total Q=10⁴ | 100 episodes × Q=100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | frozen_local | 25.6 ms | — | — | 14.0 µs | 280 | 264 | 0.166 s | 0.166 s |
| 32 | phase_reuse_target | 26.2 ms | 1.3 ms | 6.1 ms | 14.2 µs | 280 | 4,368 | 0.170 s | 0.299 s |
| 32 | pooled_local | 25.3 ms | 1.2 ms | 5.6 ms | 13.1 µs | 280 | 4,376 | 0.158 s | 0.277 s |
| 32 | pooled_global | 0.43 ms | 1.0 ms | 5.7 ms | 2.8 µs | 1,024 | 17,424 | 0.029 s | 0.130 s |
| 32 | fresh_local_cv | — | 22.6 ms | 24.0 ms | 13.1 µs | 280 | 131,344 | 0.153 s | 2.39 s |
| 32 | fresh_global | — | 1.0 ms | 5.8 ms | 2.8 µs | 1,024 | 17,416 | 0.029 s | 0.130 s |
| 512 | frozen_local | 416 ms | — | — | 22.6 µs | 4,120 | 4,104 | 0.641 s | 0.642 s |
| 512 | phase_reuse_target | 416 ms | 10.0 ms | 21.2 ms | 21.0 µs | 4,120 | 69,648 | 0.636 s | 1.62 s |
| 512 | pooled_local | 385 ms | 9.9 ms | 21.3 ms | 20.9 µs | 4,120 | 69,656 | 0.604 s | 1.58 s |
| 512 | frozen_global | 80.8 ms | — | — | 9.6 µs | 262,144 | 262,144 | 0.177 s | 0.181 s |
| 512 | pooled_global | 80.9 ms | 144 ms | 660 ms | 9.1 µs | 262,144 | 4,456,464 | 0.317 s | 14.6 s |
| 512 | fresh_local_cv | — | 329 ms | 380 ms | 20.9 µs | 4,120 | 2,101,264 | 0.538 s | 33.1 s |
| 512 | fresh_global | — | 415 ms | 652 ms | 8.7 µs | 262,144 | 4,456,456 | 0.502 s | 41.5 s |

- **Episodes:** in the repeated-episode estimates the source is acquired once
  and reused. Fresh policies pay their full fit every episode. Reuse arms see
  s+Rm pairs and fresh arms Rm; both counts are in the runtime JSON.
- **Local target updates are cheap. Local acquisition is not.** At d=512,
  refitting local weights at the reused phase costs 10 ms for 32 pairs, about
  33× less than a fresh CV run. The one-off CV acquisition (416 ms) dominates
  single-episode totals. With 100 episodes the source is acquired once, and
  phase reuse (1.6 s) is about 20× cheaper than fresh local CV (33 s).
- **Global costs grow faster.** At d=512, per-pair d² updates plus d³ solves
  make pooled and fresh global adaptation 15–65× slower than local. Their
  deployed basis is 64× larger, but inference is about 2× faster in this
  NumPy implementation.
- **Retained state differs by policy.** Phase reuse and pooled local hold
  block sums of O(db), 68 KiB at d=512. Fresh local CV holds its raw target
  pairs (2 MiB at d=512). Pooled global holds d×d sums (4.3 MiB).
- **An unexplained timing:** at d=512, fresh global's m=32 endpoint (415 ms)
  is slower than pooled global's (144 ms), although both do the same updates
  and one solve. A plausible but unverified cause is a solver speed effect on
  the poorly conditioned 32-pair statistics. It is reported as measured.
- **Not measured:** LAPACK workspace (estimates are in the runtime JSON) and
  FLOPs. The full panel took 264 s across 42 spawned configurations, all
  counted as study cost. This panel is descriptive, not a speed hypothesis.

Historical v1/v2 studies did not report per-episode adaptation cost, so no
cost comparison with them is made here.

## Interpretation and scope

- In this generator, reusing a learned grouping is worth more than reusing
  learned weights once directions drift by 30° or more. The grouping costs
  nothing to keep (one phase index) and still holds. The weights go stale at
  the rate sin²θ. Pooling source covariance carries the stale weights
  forward. It helps only at small drift and small m.
- In this window, frozen local stops beating fresh target learning between
  15° and 30°. Pooled local is level with fresh at 30° (0.466 versus 0.467). It moves with m, because target-only
  fits keep improving and frozen codes do not. No angle-dependent policy was
  declared, so none is claimed. Choosing between frozen and refit would need
  a drift detector, which is out of scope.
- Phase reuse discards source reconstruction. To keep both, a learner would
  need to retain the source basis as well (another d weights). This study
  measured that tradeoff but tested no policy that keeps both.
- H1/H2 measure acquisition savings for paid history. They do not show
  equal-total-example superiority: at s=16, frozen reuse loses to target-only
  learning.

## Reproduction

```
.venv/bin/python -m unittest tests.test_representation_reuse tests.test_locality_adaptation_v2 tests.test_locality_adaptation -v
.venv/bin/python -m experiments.representation_reuse --phase dev
.venv/bin/python -m experiments.representation_reuse --phase freeze   # then commit
.venv/bin/python -m experiments.representation_reuse --phase confirm
.venv/bin/python -m experiments.representation_reuse --phase timing
```
