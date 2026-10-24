# Single-module plasticity v1: H4 passes, H1–H3 fail; overall `fail`

**Completion:** `complete`. All correctness gates passed: 91 tests, including
the unchanged v1, v2 and reuse suites. Frozen v1/v2/reuse sources matched
their manifests. The repeated development run was byte-identical. All 64
confirmation worlds were scored with no factorization, rank, constraint,
residual, Schur or nonfinite failure.

**Headline:**

- **Primary outcome: `fail`.** Regional-drift branch, window m ∈ {16, 32, 64,
  128, 256} B pairs, 64 worlds, t₆₃ quantile 2.296. C = `graded_matched`.

| Claim | Difference (positive favours C) | Mean | SE | One-sided 98.75% lower bound | Margin | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| H1 matching beats mean routing on target | E_B(graded_mean) − E_B(C) | −0.029 | 0.0005 | −0.030 | > 0.01 | fail |
| H2 matching beats shifted routing on target | E_B(graded_shifted) − E_B(C) | −0.024 | 0.0006 | −0.025 | > 0.01 | fail |
| H3 slower inner learning preserves source | E_A(uniform_matched) − E_A(C) | 0.0012 | 0.0003 | 0.0006 | > 0.01 | fail |
| H4 target adaptation noninferior to uniform | E_B(uniform_matched) − E_B(C) | 0.007 | 0.0004 | 0.0065 | > −0.02 | pass |

- **H1 and H2 do not merely fall short; they reverse.** Mean and shifted
  routing both beat matched routing by more than twice the margin. The
  reversal is already there at the source checkpoint, before any target
  learning. With identical parameters, E_A is 0.496 matched, 0.468 mean and
  0.473 shifted; B-set E at m=0 is 0.615, 0.586 and 0.591. Through B the
  matched-versus-mean gap barely moves (0.029 at m=0, 0.028 at m=256). The
  failure is mainly an **inference** effect of per-sector matched coupling.
  It is not caused by the plasticity reallocation away from the drifted
  sector that the pre-implementation note worried about.
- **The gate does detect the drifted sector, but using it hurts.** At m=0 the
  drifted sector's g is 0.04–0.07 below the other sectors in every ring (all
  64 worlds acquired the true phase). Per (example, sector), g₀ correlates
  −0.14 with output error. Lower familiarity marks higher error, as intended.
  A likely mechanism, which this study did not test: on single observations,
  noise-dominated sectors also look unfamiliar. Matched routing lowers their
  coupling, which weakens shrinkage exactly where noise must be suppressed.
  Averaging (mean) or misassigning (shifted) the gates removes that
  per-observation correlation.
- **H3 fails as predicted; the depth profile has almost no effect on source
  risk.** The difference is 0.001. No adapting arm forgets at all: E_A
  *improves* during B learning (C: −0.005 over the window) because the module
  is still undertrained on the source distribution (see below). The contract's
  retention question therefore never became active. The predicted
  "graded forgets faster" outcome did not occur either.
- **H4 passes as predicted.** Through its 32/15 outer rate, C adapts the
  drifted sector faster than uniform. Drifted-sector B error at m=256 is
  0.787 for C, 0.845 uniform and 0.969 frozen. Window E_B is 0.591 C,
  0.599 uniform and 0.606 reversed, which matches the rate ordering.
- **The layered module is not practically useful in this configuration.**
  Window E_B on regional drift is 0.59 for C, against 0.377 for the
  phase-target CCA reference and 0.401 for pooled local CCA. E_A is 0.49,
  against 0.27 for pooled CCA and 0.25 for the oracle projector. Under global
  drift, layered E_B (0.88–0.97) is close to the zero-output control (1.0).
  The fixed rate eta=0.1 with four updates per batch moves parameters
  slowly. Source warm-up improved E_A only from 0.504 to 0.496, and the
  warm-up loss fell from 0.409 to 0.401. Development is correctness-only, so
  no rate was tuned.

The study supports this much, for this one circular module and stream:
matched familiarity routing, as specified, lowers reconstruction quality
compared with spatially averaged or misplaced routing, and the effect comes
mostly from inference rather than plasticity. A decreasing depth-rate profile
adapts the drifted sector faster without measurable source cost, because no
arm reached the regime where forgetting occurs. It establishes nothing about
the gate formula being calibrated familiarity, about layering versus flat
models, or about multiple modules.

[Contract](single-module-plasticity-v1-contract.md) ·
[design](single-module-plasticity-design.md) ·
[manifest](../reports/single-module-plasticity-v1-manifest.json) ·
[results](../reports/single-module-plasticity-v1-results.json) ·
[development](../reports/single-module-plasticity-v1-development.json) ·
[runtime](../reports/single-module-plasticity-v1-runtime.json)

## Procedure and disclosures

- The draft contract was committed in `b9dcc59`. Review clarifications
  (contract section 13) were committed in `927e050` before implementation.
  They changed no hypothesis, margin, budget, arm, rate or gate. That section
  also records the expectations stated before implementation: H1/H2 had
  undetermined sign and might well fail, H3 was expected to fail, and H4 was
  expected to pass.
- Implementation, tests, development report and manifest were committed in
  `3556b24` before any confirmation world was generated. Confirmation and
  timing refuse to run unless their own frozen sources and the v1/v2/reuse
  dependencies match their hashes and are committed. Confirmation ran at
  `3556b24` with only `reports/single-module-plasticity-v1-results.json`
  dirty (its own output).
- Development quality was seen before freezing (8 worlds: H1 −0.028, H2
  −0.024, H3 0.001, H4 0.008). As the contract requires, nothing was changed
  in response.
- Source acquisition calls reuse v1 `acquire_local` (v2 CV + refit,
  33 local fits). A gate checks it bitwise against v2 `fit_cv_refit`. The
  generator reproduces the reuse recipe under root 2026102410; a gate checks
  this bitwise against `src/representation_reuse.py` under a scoped root
  override. Direct references are the frozen reuse `PhaseReuseTarget` and
  `PooledLocal`, never reset at the return.
- Inference is a float64 Cholesky solve on all 80 coordinates for both
  passes. Across confirmation, the maximum relative solve residual was
  6.9e-15. The maximum pilot condition number was 146, and the largest
  second-pass Cholesky-diagonal lower bound was 18.7. Schur-complement LU
  references matched both passes on 8 inputs per evaluator set at every
  layered checkpoint, within 1e-9. Analytic fixed-route gradients matched
  central differences, including off-manifold terminal vectors with QᵀQ
  curvature. Every update kept the Stiefel and unit constraints within 1e-9.
- Evaluator controls on the shared draws: zero output 1.00, raw input 6.00,
  oracle projection 0.250 (A) and 0.247–0.249 (B).

## Descriptive results

Window means over m ∈ {16,…,256} B pairs, with 95% t intervals over 64
worlds. Branches share draws and are not independent replicates.

| Arm | Regional E_B | Regional E_A | Stable E_B | Global E_B | Global E_A |
| --- | --- | --- | --- | --- | --- |
| graded_matched (C) | 0.591 [0.582, 0.601] | 0.490 | 0.486 | 0.918 | 0.500 |
| uniform_matched | 0.599 | 0.492 | 0.488 | 0.942 | 0.496 |
| reversed_matched | 0.606 | 0.494 | 0.491 | 0.965 | 0.496 |
| graded_mean | 0.563 | 0.464 | 0.459 | 0.884 | 0.472 |
| graded_shifted | 0.567 | 0.469 | 0.463 | 0.886 | 0.477 |
| frozen_layered | 0.615 | 0.496 | 0.494 | 0.991 | 0.496 |
| phase_target_cca | 0.377 | 0.537 | 0.375 | 0.374 | 1.028 |
| pooled_local_cca | 0.401 | 0.284 | 0.266 | 0.820 | 0.334 |

- **Return to A (regional, final m=256 of return).** E_A is 0.461 for C,
  0.473 uniform, 0.485 reversed, 0.438 mean and 0.496 frozen. The layered arms
  keep improving on A, because they are still learning the source
  distribution. Phase-target CCA recovers to 0.324, and pooled stays at 0.275.
- **Ring parameter change by m=256 B (regional), outer→inner.** Distance from
  source: C (0.31, 0.11, 0.04, 0.01), uniform (0.15, 0.10, 0.08, 0.04),
  reversed (0.04, 0.06, 0.08, 0.08). Cumulative applied update norms are
  C (1.03, 0.37, 0.13, 0.03) and uniform (0.48, 0.35, 0.25, 0.13). The
  mean-one rate profiles therefore do not give equal update norms (contract
  section 4). C moves the outer ring most, which is consistent with its faster
  drifted-sector adaptation.
- **Clean-input distortion** (regional window): 0.116 for C, 0.121 uniform,
  0.133 frozen. It is a separate diagnostic and is not summed with noise.

## Computation

Measured on an AMD Ryzen 5 3600 with one process, one BLAS thread, float64,
NumPy 2.5.4 and SciPy 1.18.1. Full data, raw repeats and environment are in
the runtime JSON.

| Phase | Wall (s) | Process CPU (s) | Peak RSS (KiB, cumulative parent) |
| --- | --- | --- | --- |
| Development (two full passes, 8 worlds) | 660 | 660 | 162,544 |
| Confirmation (64 worlds) | 2,696 | 2,695 | 235,540 |
| Timing panel (16 spawned workers) | 37 | — | idle worker 98,072 |

- **Confirmation split.** About 1,885 s went to evaluator inference, mostly
  per-input second-pass factorizations and solves (370 s and 359 s inside
  it), and 82 s to Schur references. Each adapting layered arm used about
  238 s for learning over 64 worlds × 3 branches × 256 updates (49,152
  updates; 786,432 pilot, second and adjoint solves each). Source acquisition
  took 3 s and warm-up 16 s, shared per world. Direct references took 3 s
  each.
- **Per-call timings** (stage-2 world 0, medians of 7 repeats). One routed
  update over 16 directions takes 3.5–4.6 ms; the largest components are
  dense Hessian assembly (~0.5 ms), second factor and solve, and adjoint.
  Two-pass inference takes about 1.1 ms for one input and 4–5.5 ms for 32,
  including fresh pilot-factor construction per call. A direct reference
  needs 0.76 ms per batch statistic update plus solve and 0.012 ms per
  projector inference. Acquisition (CV + refit) takes about 25 ms, and
  layered warm-up about 140 ms.
- **Arithmetic cost estimates** through the end of return (512 pairs):
  acquisition + warm-up + 256 updates + Q × single inference is about 1.1–1.3 s
  at Q=1 and 11–13 s at Q=10,000 for layered arms. The figures for direct
  references are 0.07 s and 0.20 s. These are estimates from isolated
  timings, not measured episodes. The direct references are both cheaper and
  more accurate here.
- **Persistent learner state.** Layered: 328 float64 values (2,624 bytes),
  phase metadata and no optimizer state. Direct references: block sums plus
  count and weights (sizes in the runtime JSON). The dense 80×80 solver does
  not exploit the parameters' sparsity, so no efficiency claim follows.

## Classification

| Question | Outcome |
| --- | --- |
| Compatibility recognition | Gates are lower at the drifted sector and correlate with error; not calibrated |
| Matched routing (H1/H2) | Reversed: matched is worse than mean or shifted, already at the source checkpoint |
| Target adaptation | C adapts fastest among layered arms (H4 pass); all far behind direct CCA |
| Source retention (H3) | Not tested in practice: no layered arm forgot; profile effect 0.001 |
| Return recovery | Layered arms improve slowly on A; direct references recover faster |
| Ring-rate ordering | Graded > uniform > reversed for target adaptation, matching outer rate |
| Computation | Layered is about 10–100× slower than direct references for worse quality |
| Absolute usefulness | No: E_B 0.59 against 0.38 (phase-target CCA) and 0.25 (oracle) |

A negative result narrows this design and nothing more. In particular, it
does not justify moving to multiple modules or falling back silently to a
single projector. Two of its findings are concrete enough to direct a
follow-up contract: per-observation matched coupling is harmful at this
signal-to-noise ratio, and the fixed learning rate leaves the module
undertrained, so the retention question remains open.
