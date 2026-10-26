# Correlated nuisance v1: results

Results for the frozen [contract](correlated-nuisance-v1-contract.md), run
2026-10-26 at revision a727c1f. Fixture 2 of the
[inner-coupling research direction](inner-coupling-hypothesis.md).

[Runner](../experiments/correlated_nuisance.py) ·
results (`reports/correlated-nuisance-v1-results.json.gz`, gzip -n -9, 12.6 MB
uncompressed, SHA-256 a3f68a65…af24; the contract names the uncompressed
file) ·
[runtime](../reports/correlated-nuisance-v1-runtime.json) ·
movement (`reports/correlated-nuisance-v1-movement.json.gz`, gzip -n -9,
uncompressed SHA-256 fa5ee60d…8b1a, recorded in the results).
Reproduce with `.venv/bin/python -m experiments.correlated_nuisance` (about
4.5 minutes). A concurrent repeat run was byte-identical in every output
except the movement file name recorded in the results.

Post-hoc analysis, outside the contract:
[script](../experiments/correlated_nuisance_posthoc.py) ·
[results](../reports/correlated-nuisance-v1-posthoc-results.json) ·
[runtime](../reports/correlated-nuisance-v1-posthoc-runtime.json); reproduce
with `.venv/bin/python -m experiments.correlated_nuisance_posthoc` (under 1 s).

## Summary

- **Inner coupling filters correlated nuisance; every surface-coupled family
  transmits it.** In the weak regime, inner coupling's nuisance gain γ is
  about 0.05 in every seed. Uniform, outer and outer-matched coupling have
  γ between 0.58 and 1.13.
- **Filtering does not pay off overall here.** The 2-dimensional inner
  channel reconstructs the signal much worse than surface coupling. At b = 2,
  inner coupling is clearly worse than uniform and outer coupling even on
  the decorrelated test (3.3× and 2.7×), so N4 **counts against the
  mechanism on this fixture**. At b = 10 inner coupling is better on the
  decorrelated test in 10/10 seeds (1.35× against uniform, 1.58× against
  outer), but it fails the retention part of the learning criterion, so no
  conclusion is drawn. Its in-distribution error is 18-21× worse.
- **N5, regime dependence, is descriptive only.** In the strong regime
  inner coupling's γ rises to about 0.8, the direction the variance-ranking
  hypothesis expects. But the strong-regime inner conditions fail N1, so
  under the contract this supports no conclusion.
- **Adding deeper coupling to surface coupling changes nothing measurable.**
  At fixed surface strength, uniform and outer-only coupling have the same γ
  (ratio 1.02) and comparable errors. In the independent regime, outer-only
  coupling at the same budget matches uniform coupling. In this linear
  architecture, the surface channel decides what is shared.
- **The inner channel does not behave like reduced-rank regression.** Rank-2
  regression transmits in the weak regime (γ 0.68) where inner coupling
  filters. A derived post-hoc explanation accounts for the comparator:
  regression ranks centered variance, under which one nuisance direction
  outranks the signal's amplitude direction. It does not show how the
  energy learner selects directions.

## Fixture references

Means over seeds 23-32, exact Bayes references on each seed's test sets:

| Regime | Signal oracle, X_in | Full oracle, X_dec |
| --- | ---: | ---: |
| weak | 0.0064 | 0.0036 |
| strong | 0.243 | 0.131 |

The contract's derived values were 0.0066 / 0.0034 and 0.250 / 0.125; the
Monte Carlo check in verification reproduced them to within 1%. Differences
here are finite-sample variation on 24 test examples.

## Results

X is exact-equilibrium avoidable MSE per coordinate, mean over seeds 23-32.
"Signal" is the signal-error part of X_dec (the error inside the hidden
module's pattern span). "Learns" is N1.

**Weak regime** (shared nuisance variance 0.027 per direction):

| Condition | η | X_in | X_dec | γ | τ | Signal | Learns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| uniform 2 | 0.1 | 0.00236 | 0.00195 | 0.62 | 0.49 | 0.00044 | no (A retained 8/10) |
| outer 2 | 0.03 | 0.00160 | 0.00237 | 0.75 | 0.58 | 0.00024 | yes |
| outer-matched 2 | 0.1 | 0.00279 | 0.00197 | 0.58 | 0.45 | 0.00062 | yes |
| inner 2 | 0.1 | 0.01301 | 0.00647 | 0.048 | −0.01 | 0.00619 | yes |
| uniform 10 | 0.03 | 0.00051 | 0.00406 | 1.04 | 0.82 | 0.00004 | yes |
| outer 10 | 0.03 | 0.00045 | 0.00474 | 1.13 | 0.87 | 0.00004 | yes |
| outer-matched 10 | 0.03 | 0.00059 | 0.00394 | 1.02 | 0.79 | 0.00006 | yes |
| inner 10 | 0.1 | 0.00953 | 0.00299 | 0.049 | −0.01 | 0.00265 | no (A retained 2/10) |
| ridge | | 0.00025 | 0.00332 | 0.95 | | | |
| reduced-rank 2, frozen | | 0.04483 | 0.04285 | 0.68 | | | |
| reduced-rank 2, recentred (post hoc) | | 0.00484 | 0.00286 | 0.68 | | | |

**Strong regime** (shared nuisance variance 1.0 per direction):

| Condition | X_in | X_dec | γ | Signal | Learns |
| --- | ---: | ---: | ---: | ---: | --- |
| uniform 2 / 10 | 0.00854 / 0.00116 | 0.193 / 0.209 | 1.19 / 1.26 | 0.0058 / 0.0006 | yes / yes |
| outer 2 / 10 | 0.00493 / 0.00075 | 0.199 / 0.219 | 1.22 / 1.28 | 0.0030 / 0.0004 | yes / yes |
| outer-matched 2 / 10 | 0.00916 / 0.00119 | 0.193 / 0.214 | 1.19 / 1.27 | 0.0062 / 0.0006 | yes / yes |
| inner 2 / 10 | 0.162 / 0.155 | 0.184 / 0.190 | 0.77 / 0.80 | 0.107 / 0.106 | no / no (vs frozen 0.44 / 0.42) |
| ridge | 0.00023 | 0.134 | 1.01 | | |
| reduced-rank 2, recentred (post hoc) | 0.0733 | 0.176 | 0.96 | | |

**Independent regime** (v1's fixture; X_in = X_dec):

| Condition | b = 2 | b = 10 |
| --- | ---: | ---: |
| uniform | 0.00101 | 0.00144 |
| outer | 0.00102 | 0.00210 |
| outer-matched | 0.00119 | 0.00147 |
| inner | 0.00643 | 0.00300 |
| ridge | 0.00255 | |
| reduced-rank 2, recentred (post hoc) | 0.00134 | |

Uniform / 2 is close to the ablation's uniform L/2 (exact 0.00099 there, on
seeds 13-22). Seed 3's exact reproduction of v1 arm 2a is in verification.

## Criteria

**N1 learning.** 20 of 24 conditions learn. The failures:

- weak inner 10 and weak uniform 2 fail A retention (after-B/after-A ratio
  ≤ 1.5 in 2/10 and 8/10 seeds). Both reach low final error; stage B
  raises structure A's error by up to 1.9× while lowering B's.
- strong inner 2 and 10 fail the frozen-ratio test (0.44 and 0.42).

The runner's N4, N5 and N6 verdicts do not apply the contract's rule that
comparisons involving a non-learning condition support no conclusion. The
readings below apply it; the runner's raw values are in the results JSON.

**N2 control (independent regime).**

- Inner coupling is clearly worse than uniform at both budgets: 6.3× at b = 2
  and 2.1× at b = 10.
- At b = 2, uniform, outer and outer-matched are comparable (outer/uniform
  0.998). Outer-only coupling is enough. With the layers isolated, the
  ablation's uniform advantage over inner coupling is a surface-coupling
  advantage.
- At b = 10, uniform and outer-matched are comparable (0.98); full-budget
  outer is worse (1.46×, 10/10 seeds, unresolved under the 1.5 threshold).

**N3 transmission.** Every weak and strong condition is in an affirmative
category; none is unresolved.

- Filters: weak inner 2 and 10 (γ 0.02-0.07 in every seed).
- Transmits: all others, including strong inner (γ 0.57-0.87).

τ agrees in direction with γ everywhere (inner −0.01, others 0.45-0.99).

**N4 robustness (weak regime).**

- **b = 2: counts against the mechanism on this fixture.** Inner against
  outer, both learning: outer is better on X_dec (inner 2.7× worse, 10/10
  seeds), so the pair is dominated. Inner against uniform is also dominated
  (3.3×) but uniform 2 does not learn, so that pair is set aside. One
  dominated pair and no advantage gives "counts against".
- **b = 10: no conclusion.** The raw verdicts are an advantage against outer
  (1.58×, 10/10 seeds) and unresolved against uniform (1.35×, 10/10 seeds),
  with inner filtering. Inner 10 does not learn, so neither pair supports a
  conclusion.
- In-distribution cost: inner is 5.5-21× worse on X_in in every pair.

**N5 regime dependence: no conclusion.** Strong-regime inner coupling fails
N1 at both budgets (it removes only about 58% of its frozen error), so the
same gate as N4 and N6 applies. Descriptively, inner γ is 0.05 in the weak
regime (filters, learning at b = 2) and about 0.8 in the strong regime
(transmits, not learning). The direction matches the variance-ranking
hypothesis, but this shows what a failing inner channel carries, not what a
trained one would. The runner's raw verdict, "filtering depends on regime",
does not apply the gate.

**N6 adding deeper coupling (weak regime).**

- **b = 10: no effect on transmission.** γ ratio uniform/outer-matched is
  1.02 (uniform higher in 10/10 seeds but well inside 1.25). X_dec and X_in
  are comparable.
- **b = 2: no conclusion**, because uniform 2 does not learn. The raw γ ratio
  is 1.07.
- Fixed-budget uniform against outer: γ ratio 0.93 (b = 10) and 0.83
  (b = 2), within "no effect"; X_dec comparable.

**R1 references.** In the weak regime, every surface-coupled condition sits
between the two oracles on X_in but exceeds the full oracle on X_dec at
b = 10 (0.0039-0.0047 against 0.0036). See the gain split below.

**R2 reduced-rank analogy.**

- Strong regime: inner and reduced-rank regression both transmit.
- Weak regime: inner filters (γ 0.05) and reduced-rank regression transmits
  (γ 0.68). The analogy fails where it mattered.

## Post-hoc analyses

These are outside the contract and change no criterion.

**Reduced-rank comparator offset.** The frozen comparator keeps the ridge
intercept after projecting the slopes. Every prediction is then shifted by
x̄ B (I − V₂V₂ᵀ), which inflates its error about 10×. Recentring the intercept
so the training-mean prediction is preserved gives the "recentred" rows
above. Slopes, and therefore γ and R2, are unchanged. The frozen X values
for this comparator should not be used. With recentring, rank-2 regression
on the decorrelated weak test (0.0029) is close to inner 10 (0.0030).

**Why reduced-rank regression transmits in the weak regime** [derived,
post hoc]. Regression ranks centered fitted variance. Training data mix two
structures. The centered signal has a label-contrast direction with large
variance, and an amplitude direction whose variance is Var(a)/2 ≈ 0.0067.
That is below the shared nuisance's 0.027 per direction, so rank 2 keeps the
contrast and one nuisance direction. Keeping one of two nuisance directions
predicts γ ≈ 1/√2 ≈ 0.71, close to the observed 0.68 and its symmetric gain
of 0.68. This explains the comparator's behaviour directly. For the energy
learner it suggests only a reason the two differ: the energy has no bias
terms, and the signal's uncentered second moment (0.51 per pattern
coordinate) outranks the nuisance. That is the centered-versus-uncentered
distinction the contract review raised. It does not establish that the
energy objective selects directions by second-moment ranking; inner
filtering in the weak regime is consistent with that, but was not derived
for this objective.

**Over-transmission by surface coupling.** In distribution, both observed
modules carry the same shared latent. The sum of their shared-direction
projections carries 2s z, while their difference carries only independent
noise, c(U_o₁ᵀe_o₁ − U_o₂ᵀe_o₂), with variance 2 × 0.01c² = 0.002 per direction
in the weak regime. So the sum of the two blocks of G_h is well identified
by training data, and their difference (antisymmetric part) only weakly.
Decorrelation raises the difference's variance to 2s² + 0.002 ≈ 0.056, about
28×. An antisymmetric part that costs little in distribution then costs much
more. Relative to the oracle's symmetric gain, at exact equilibrium:

| Weak regime | Symmetric | Antisymmetric |
| --- | ---: | ---: |
| uniform 2 / 10 | 0.50 / 0.83 | 0.36 / 0.64 |
| outer 2 / 10 | 0.60 / 0.88 | 0.44 / 0.71 |
| outer-matched 2 / 10 | 0.47 / 0.80 | 0.33 / 0.63 |
| inner 2 / 10 | 0.05 / 0.05 | 0.02 / 0.01 |
| ridge | 0.93 | 0.15 |
| reduced-rank 2 | 0.68 | 0.07 |

The surface-coupled learners carry a large antisymmetric part that ridge
does not, which is why their X_dec exceeds the full oracle's. A likely
reason, not tested: in the 0 ↔ 1 ↔ 2 chain, an end module is predicted
mainly through its single neighbour, so the learned map weights the two
observed modules unequally, while the oracle weights them equally.

## Reading against the hypothesis

The mechanism the hypothesis relies on exists in this architecture: a
module coupled only at its inner layer passes on almost none of a surface
correlation that does not persist (γ ≈ 0.05), while surface coupling passes
on substantial nuisance (γ 0.58-0.75 at b = 2, 1.02-1.13 at b = 10). But here the narrow channel also loses signal. Inner coupling's
signal error alone (0.0027-0.0062) is comparable to the nuisance error
surface coupling incurs on the decorrelated test (about 0.0015-0.0047). Inner coupling was better on the decorrelated test only
at b = 10, by 1.35-1.58×, in a condition that fails retention, and it paid
18-21× in-distribution.

So this fixture gives the hypothesis a mechanism but no net win. Two
qualifications:

- **Linear modules.** The inner layer here is a 2-dimensional projection, so
  "deeper" structure cannot exist. The comparison tests a bottleneck, not
  abstraction.
- **Adding inner coupling to surface coupling did not gate transmission**
  (N6). Whatever surface coupling exists decides what is shared. A system
  wanting inner-level sharing would have to weaken or learn to gate surface
  coupling, not just add inner coupling.

## Limitations

- 10 of 24 selected learning rates are at a grid edge, mostly the top edge
  (0.1), with inner conditions at 0.1 throughout. The pilot curves are flat
  between 0.03 and 0.1 (for example weak inner 10: 0.01103 against 0.01100),
  so the edge is unlikely to matter much, but it was not extended.
- 32-step settling has not converged for inner coupling (distance to exact
  about 0.1 at b = 10 weak). The primary endpoint is exact, so criteria are
  unaffected; training settles still use 32 steps.
- The reduced-rank comparator as frozen has the intercept defect above.
- Several conclusions rest on retention failures of a few conditions; the
  stage protocol (A then A+B) interacts with these comparisons.
- One fixture, one signal-to-nuisance ratio per regime, one chain topology.

## Next questions

- Learn or gate per-layer coupling strength, so a module can weaken its
  surface channel when surface correlations do not persist (N6 shows adding
  inner coupling is not enough).
- Nonlinear modules (fixture 1), the real test of the hypothesis.
- All-to-all instead of chain coupling, to test the asymmetric-weighting
  explanation of over-transmission.
- A wider inner channel to trade filtering against signal capacity.

## Computation

Measured on one CPU thread (float64):

| Phase | Wall (s) |
| --- | ---: |
| Verification (including the ablation's checks and two 600-cycle reproductions) | 4.5 |
| Selection pilot (360 trajectories, exact scoring) | 140.6 |
| Confirmatory training and evaluation (240 runs) | 115.9 |
| Frozen baselines, comparators, oracles | 3.7 |
| Total | 265.5 |

- Training CPU, excluding evaluation: 95.9 s for 240 runs. Per run: uniform
  0.50 s, outer and outer-matched 0.37 s, inner 0.36 s. The pilot is 53% of
  wall time.
- Peak RSS 441 MB, mostly in-memory reporting structures (the ablation's was
  125 MB).
- Counts: 360,600 Adam steps; 401,820 settles and implicit factorizations;
  50,064 exact solves; 6,062,400 replay-trace exposures; 372 comparator fits.
  These include the pilot and verification. Phase and per-run timings, hashes
  and versions are in the runtime JSON.
- Post-hoc analysis: 0.6 s.

Wall-clock times come from runs on a shared machine with the concurrent
repeat and are not latency benchmarks.

Runtime `dirty_status` lists only the results and uncompressed movement
files, which are written before the provenance step.
