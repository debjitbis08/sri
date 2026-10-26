# Inner-channel diagnostic v1: results

Results for the frozen [contract](inner-channel-diagnostic-v1-contract.md),
run 2026-10-26 at revision 4fc5a45. Follows
[correlated nuisance v1](correlated-nuisance-v1-results.md).

[Runner](../experiments/inner_channel_diagnostic.py) ·
results (`reports/inner-channel-diagnostic-v1-results.json.gz`, gzip -n -9,
10.9 MB uncompressed, SHA-256 e44f537b…8ff9) ·
[runtime](../reports/inner-channel-diagnostic-v1-runtime.json). The runner
also writes a movement file with per-cycle parameter movement and the
per-update settle checks (50.5 MB uncompressed, SHA-256 66d01fdb…49be,
recorded in the results). It is not kept in the repository; rerun to
regenerate it.
Reproduce with `.venv/bin/python -m experiments.inner_channel_diagnostic`
(about 24 minutes). A concurrent repeat run was byte-identical in every
output except the movement file name recorded in the results.

## Summary

- **Rank restriction is not the barrier.** Uncentered rank-2 regression
  filters (γ 0.05) and has decorrelated error 0.00037, 4.8-9.7× lower than
  uniform clamped-energy coupling. A 2-dimensional channel can keep the
  filtering and the signal at once (D5).
- **Removing clips and weight decay helps the capacity fit 3.6-3.8×** (D2).
  The unconstrained fit S′ filters (γ 0.13-0.17), and at b = 10 its
  decorrelated error (0.00097) is 3.8× lower than uniform coupling's. It
  gets there with large weights (|W| up to 26), outside the learner's clip
  bounds, so there is no feasible witness. Post hoc, the constrained fit
  never touched its clips. That makes weight decay a candidate cause, not
  an isolated one.
- **Within the learner's constraints, no improvement was found.** The
  constrained capacity fit S did not beat the clamped-energy learner E
  (D1a), and did not beat uniform coupling (D1b). Masked completion did not
  beat E either (D3).
- **The shrinkage hypothesis is not supported as a learner-specific cause**
  (D6). E's hidden codes do shrink at b = 2 (ρ 0.85), but so do the capacity
  fit's (0.88).
- **Training settles are not exact** (D7). The gradient discrepancy is
  largest during stage A, reaching 0.065. The final-parameter check that
  motivated "not expected to matter" missed this. In stage B it is about
  1e-5, still above the declared tolerance of 1e-6. Its effect on the
  results is not measured.

## Results

X is exact-equilibrium avoidable MSE per coordinate, mean over seeds 33-42.
"Signal" is the signal-error part of X_dec. ρ is hidden-code shrinkage.

**Weak regime, b = 2 / b = 10:**

| Arm | η | X_in | X_dec | γ | Signal | ρ | Learns (N1) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E clamped energy, inner | 0.1 / 0.1 | 0.0133 / 0.0098 | 0.0064 / 0.0029 | 0.042 / 0.043 | 0.0062 / 0.0026 | 0.85 / 0.94 | yes / no (A 2/10) |
| U clamped energy, uniform | 0.1 / 0.03 | 0.0025 / 0.0005 | 0.0018 / 0.0037 | 0.62 / 1.05 | 0.0004 / 0.0000 | 0.95 / 0.99 | no (A 5/10) / yes |
| M masked completion, inner | 0.01 / 0.01 | 0.0130 / 0.0107 | 0.0061 / 0.0038 | 0.042 / 0.042 | 0.0058 / 0.0036 | 0.88 / 0.93 | yes / no (A 5/10) |
| S capacity fit | 0.003 / 0.003 | 0.0128 / 0.0105 | 0.0057 / 0.0035 | 0.011 / 0.006 | 0.0057 / 0.0035 | 0.88 / 0.93 | fit |
| S′ fit without clips or decay | 0.1 / 0.1 | 0.0086 / 0.0081 | **0.0015 / 0.0010** | 0.17 / 0.13 | 0.0012 / 0.0008 | 0.90 / 0.95 | fit |

Comparators (weak regime, all waking traces):

| Comparator | X_in | X_dec | γ |
| --- | ---: | ---: | ---: |
| uncentered rank-2, no intercept | 0.0073 | **0.00037** | 0.050 |
| centered rank-2, recentred | 0.0046 | 0.0025 | 0.67 |
| ridge, no intercept | 0.0002 | 0.0030 | 0.94 |
| ridge, with intercept | 0.0002 | 0.0031 | 0.94 |

Signal oracle X_in and full oracle X_dec on these seeds are close to
correlated nuisance v1's (0.0064 and 0.0036 there). The uncentered
comparator's X_in, 0.0073, is about the signal oracle's: it passes on
almost no nuisance and loses little signal.

**Independent regime** (X_in = X_dec), b = 2 / 10: E 0.0065 / 0.0030;
U 0.0010 / 0.0015; M 0.0062 / 0.0039; S 0.0057 / 0.0035; S′ 0.0034 / 0.0020.
Comparators: uncentered rank-2 0.00046, centered rank-2 recentred 0.0011,
ridge 0.0023-0.0025.

S's chosen start was split between the shared initialization and E's final
parameters; S′ chose E's final parameters in 19 of 20 weak-regime fits. No
fit was disqualified.

## Criteria

**N1** (reported, not a gate). Failures: weak E 10 (A retained 2/10), weak
M 10 (5/10) and weak U 2 (5/10). All learn the second structure and are
within 0.08× of their frozen error; they fail retention.

**D0 control.** E filters and U transmits at both budgets, as in
correlated nuisance v1. E against U on X_dec: clearly worse at b = 2 (3.6×,
10/10 seeds); at b = 10 E is lower in 9/10 seeds (0.79×), unresolved.

**D1 capacity fits.**

- D1a: **no improvement found** at either budget. S and E are comparable
  (1.12× and 0.83×; at b = 10 E is better in 10/10 seeds).
- D1b: S filters, but **no such solution found**. At b = 2 S is clearly
  worse than U (3.2×); at b = 10 they are comparable (1.06×).

**D2 removing clips and decay.** **Helps the fit** at both budgets: S has
3.8× (b = 2) and 3.6× (b = 10) S′'s decorrelated error, worse in 10/10
seeds. **No feasible witness:** S′'s final parameters lie outside the clip
bounds in every seed (|W| up to 26, |V| and |T| up to about 5). By the
contract this does not show that capacity was restricted.

**D3 changing the objective.** **No difference established.** At b = 2 M
and E are comparable (1.06×). At b = 10 M is worse in 10/10 seeds (1.32×),
unresolved. M also filters (γ 0.042).

**D4 summary.** Constructive improvements found at both budgets: D2 and D5.

**D5 uncentered rank restriction.** **Achieves both** at both budgets: it
filters (γ 0.050) and U is clearly worse on X_dec (4.8× at b = 2, 9.7× at
b = 10, 10/10 seeds). The centered comparator transmits (γ 0.67) with X_dec
0.0025.

**D6 shrinkage.** **Not consistent** with the hypothesis. At b = 2, E's ρ is
0.85 and S's 0.88, both below 0.9 in every seed. At b = 10 both are about
0.93. Shrinkage is present but not specific to the learner.

**D7 training settles.** **Not exact for practical purposes.** Across all
confirmatory E and U runs, the largest 32-step settle distance is 0.18 and
the largest gradient discrepancy r is 0.065. The discrepancy is largest
during stage A:

- r exceeds 1e-6 in about 280-300 of each run's 600 updates (from the
  movement file, not kept; per-run maxima and where they occur are in the
  results JSON), and falls
  below 1e-6 within a few updates of stage B starting in most runs (in some
  runs much earlier, around update 20-35);
- the largest stage-B value is 4.3e-5 (independent E 10), and about 1e-5
  in the weak regime. Both exceed the 1e-6 tolerance.

What this did to the trajectories is not measured. E ending up comparable
to S does not show the settling errors are harmless, since both could fall
short. Only a matched comparison of 32-step and exact training would
measure the effect.

## Post-hoc readings

Outside the contract; they change no criterion.

- **Weight decay is a candidate cause of the constrained fit's shortfall,
  not an isolated one.** S had zero clip activations in every start and
  seed, with |W| and |V| at most 1.44, so clipping did not alter those
  trajectories. That does not show clipping would stay irrelevant once
  decay is removed: S′ ends well outside the bounds. S's pilot objective
  was flat across the learning-rate grid (b = 2: 0.0229-0.0236), but that
  does not rule out optimization effects. S′, without decay, improved
  steadily with η up to the grid's top edge (b = 10: 0.0099 at 0.001 to
  0.0017 at 0.1); it may not have converged, and its advantage may be
  larger. Separating decay-only and clipping-only interventions is needed.
- **The learner does hit its clips.** E had about 2,600 clip activations per
  regime and budget across seeds, mostly on W. So for E, both clipping and
  decay may bind.
- **Why large weights help is not established.** One hypothesis: the fixed
  activity priors, 0.025(‖h‖² + ‖c‖²), act like ridge penalties on the
  codes. Their effect relative to the reconstruction terms falls as W and V
  grow, so large weights escape the shrinkage the priors impose. ρ rises
  only slightly for S′ (0.90 / 0.95), so ρ may not capture this; it is
  untested.
- **The unconstrained fit beats surface coupling under decorrelation.** At
  b = 10, S′ (0.00097) is 3.8× below U (0.0037) while filtering. D1b is
  declared on S, so this is descriptive. It is constructive evidence that
  parameters of this energy architecture, outside the clip bounds and found
  by an evaluator-assisted fit, can keep the filtering and recover most of
  the signal.

## Reading

The question was whether inner coupling can keep its filtering and recover
the signal.

- **For a rank-2 channel in general, yes** (D5): uncentered rank-2
  regression does both, and does best of everything here on the
  decorrelated test.
- **For this energy architecture, yes, with an evaluator-assisted fit**
  (D2, post hoc). The successful solutions found were outside the learner's
  bounds and used large weights. The unsuccessful constrained fits do not
  show that comparable solutions within the bounds are impossible.
- **Within the constraints, nothing found did better than the learner.**
  Changing the objective to masked completion did not help. The learner,
  masked completion and the constrained fit stayed close despite different
  targets and protocols, while removing decay and clips changed the outcome
  substantially. That makes regularization and parameter scaling a strong
  next target, without ruling out the learning rule.

The next question is therefore whether the clamped-energy learner reaches
S′-like solutions when its regularization and constraints change. Before
running a sweep, an analytical scaling check should establish how W, V and
the translations jointly change the effective activity penalties and
coupling: larger weights need not pose the same inference problem. A small
follow-up would then:

- compare 32-step and exact training with everything else fixed (D7);
- separate removing decay, removing clips, and removing both;
- keep signal error and nuisance gain separate, since better reconstruction
  is useful only if filtering survives.

## Limitations

- 8 of 20 selected learning rates are at a grid edge, including every S′
  condition (at 0.1, where its pilot was still improving) and four E or U
  conditions.
- S and S′ are evaluator-assisted fits from two starts of 3,000 cycles.
  "No improvement found" is not a capacity limit.
- D2 mixes clipping, decay and optimization. Post-hoc evidence makes decay
  a candidate, but the contract did not separate them.
- One fixture, two regimes, two budgets, one chain topology.

## Computation

Measured on one CPU thread (float64):

| Phase | Wall (s) |
| --- | ---: |
| Verification | 5.4 |
| Selection pilot (180 learner trajectories, 120 fits of 3,000 cycles) | 620.8 |
| Confirmatory runs (10 seeds) | 799.8 |
| Total before runtime serialization | 1,427.8 |

- Wall time per seed, summed over both regimes and budgets, including
  evaluations: S 35.1 s, S′ 35.0 s, M 3.8 s, U 3.4 s, E 2.6 s.
- Training CPU per run (one regime and budget), excluding evaluation:
  S and S′ 8.5 s each (both starts together), M 0.84 s, U 0.50 s,
  E 0.36 s. These exclude the settle check: the training timers stop
  before it. Whole-phase wall times include it.
- The capacity fits and their pilot are about 90% of the run. The pilot is
  43% of wall time.
- Counts: 1,020,602 Adam steps; 2,826,325 exact solves; 2,700,015 adjoint
  solves; 213,270 settles; 48,000 settle-check exact solves; 249 comparator
  fits.
- Peak RSS 443 MB, mostly in-memory reporting structures.
- My pre-run estimate of 35-40 minutes was high; the run took 24.

Runtime `dirty_status` lists only the uncompressed results and movement
files, which are written before the provenance step.
