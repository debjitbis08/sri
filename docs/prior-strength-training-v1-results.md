# Prior-strength training v1: results

Results for the frozen [contract](prior-strength-training-v1-contract.md),
run 2026-10-26 at revision 4921353. Follows the
[energy scaling analysis](energy-scaling-analysis.md).

[Runner](../experiments/prior_strength_training.py) · results
(`reports/prior-strength-training-v1-results.json.gz`, gzip -n -9, 31.7 MB
uncompressed, SHA-256 f3f40ebb…d8cb) ·
[runtime](../reports/prior-strength-training-v1-runtime.json). No movement
file. Reproduce with `.venv/bin/python -m experiments.prior_strength_training`
(about 16 minutes). A concurrent repeat run produced a byte-identical
results file.

## Summary

- **Training with weaker activity priors keeps the repair and the
  filtering.** At priors 0.005 and 0.001, the clamped-energy inner learner's
  decorrelated error is 4.2-9.4× lower than at 0.05 (10/10 seeds at every
  setting). It still filters (nuisance gain about 0.07), and its signal gain
  is restored to 0.965-0.987 (P1, P4).
- **Training preserves the inference-only repair** (P2). Trained at a weak
  prior, the learner matches the 0.05-trained learner evaluated at that
  prior (ratios 0.96-1.00).
- **The inner learner now beats surface coupling under decorrelation** (P3).
  At matched weak priors, uniform coupling's decorrelated error is 1.9-5.3×
  higher (10/10 seeds). At prior 0.001 the inner learner (0.00072) matches
  uncentered rank-2 regression (0.00072), the best reference so far. This is
  the first learning run in this line in which a locally trained inner
  channel beats surface coupling on the decorrelated test.
- **Clipping and exact settling made no detectable difference; removing
  decay cost filtering** (P5, P6). The no-decay arms filter less (nuisance
  gain 0.21-0.26, unresolved category) and their error differences are
  unresolved. Exact training matched 32-step training everywhere, although
  32-step training-settle errors grow large at weak priors.
- **Most weak-prior learners fail the retention part of the learning
  criterion**, because stage A's error becomes very low and stage B raises
  it by more than the 1.5× ratio allowed. Their final errors are still
  lower than the 0.05-prior learner's.

## Results

X is exact-equilibrium avoidable MSE per coordinate, mean over seeds 43-52.
"Gain" is signal gain on the decorrelated set.

**Weak regime**, arm E (inner, clamped energy, decay and clips, 32-step
settles), b = 2 / b = 10:

| Prior | η | X_in | X_dec | γ | Gain | Smallest Q eigenvalue | Top W / V / T |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| 0.05 | 0.3 / 0.1 | 0.0133 / 0.0098 | 0.0069 / 0.0033 | 0.057 / 0.059 | 0.778 / 0.853 | 0.0094 | 2.48 / 1.41 / 1.00 |
| 0.005 | 0.03 / 0.1 | 0.0073 / 0.0072 | **0.00082 / 0.00078** | 0.070 / 0.066 | 0.965 / 0.972 | 0.0021-0.0023 | 2.08 / 1.03 / 1.00-1.04 |
| 0.001 | 0.1 / 0.1 | 0.0072 / 0.0072 | **0.00072 / 0.00072** | 0.067 / 0.066 | 0.986 / 0.987 | 0.00067 | 2.08 / 0.83 / 1.00 |

References at the same seeds (weak regime):

| Reference | X_dec |
| --- | ---: |
| uniform coupling U, prior 0.05, b = 2 / 10 | 0.0019 / 0.0039 |
| uniform coupling U, prior 0.005, b = 2 / 10 | 0.0016 / 0.0039 |
| uniform coupling U, prior 0.001, b = 2 / 10 | 0.0016 / 0.0039 |
| uncentered rank-2 regression | 0.00072 |

Uniform coupling's error barely moves with the prior: it is dominated by
transmitted nuisance (γ 0.61-1.04, "transmits" at every prior).

**Other arms, weak regime**, X_dec and γ:

| Arm | Prior 0.05, b = 2 / 10 | Prior 0.005, b = 2 / 10 | Prior 0.001, b = 2 / 10 |
| --- | --- | --- | --- |
| E_exact | 0.0069 / 0.0033; γ 0.06 | 0.00083 / 0.00078; γ 0.07 | 0.00073 / 0.00072; γ 0.07 |
| E_noclip | 0.0069 / 0.0033; γ 0.06 | 0.00082 / 0.00078; γ 0.07 | 0.00072 / 0.00072; γ 0.07 |
| E_nodecay | 0.0082 / 0.0035; γ 0.15 / 0.24 | 0.0011 / 0.0010; γ 0.26 / 0.21 | 0.00087 / 0.00092; γ 0.21 |
| E_free | 0.0082 / 0.0035; γ 0.15 / 0.25 | 0.0011 / 0.0010; γ 0.26 / 0.21 | 0.00087 / 0.00092; γ 0.21 |

The no-decay arms selected smaller learning rates (0.003-0.03), and in most
settings never reached their clips, which is why E_nodecay and E_free
coincide.

**Independent regime** (X_in = X_dec): E falls from 0.0066 / 0.0031 at
prior 0.05 to 0.00053 at 0.001 (both budgets). Uniform coupling is
0.0011 / 0.0016 at 0.05 and 0.0007 / 0.0016 at 0.001. Uncentered rank-2
regression is 0.00052. On the original v1 data, then, the inner learner
trained at prior 0.001 beats uniform coupling at either prior. The ablation's
finding that uniform coupling beats inner coupling about 4× reverses once
the priors are weak. This is descriptive; no criterion covers it.

## Criteria

**N1** (reported, not a gate). At prior 0.05, b = 2, E learns. Every E,
E_exact and E_noclip condition at priors 0.005 and 0.001 fails retention
(A retained in 1/10 seeds), while B's error falls in 10/10 seeds and final
error is 0.043-0.048 of frozen. The cause is the ratio form of the
criterion, not a larger final error. On seed 43, for example, at prior 0.005
and b = 2, structure A's error is 0.0020 after stage A and 0.0051 after
stage B. The 0.05-prior learner's is 0.0082 and 0.0110. Uniform coupling at
b = 10 learns at every prior.

**P0 control.** E at prior 0.05 filters (γ 0.057-0.059), with gain
0.778 / 0.853 and X_dec 0.0069 / 0.0033. That matches the inner-channel
diagnostic on its seeds (gain 0.781 / 0.856, X_dec 0.0064 / 0.0029).

**P1 weaker-prior training.** **Improves decorrelated completion** at both
weaker priors and both budgets, with filtering. E(0.05) is 8.3× (0.005,
b = 2), 4.2× (0.005, b = 10), 9.4× (0.001, b = 2) and 4.5× (0.001, b = 10)
worse, 10/10 seeds each.

**P2 training against inference-only repair.** **Training preserves the
repair** in all four settings: the 0.05-trained learner evaluated at the
weaker prior and the learner trained at it are comparable (ratios
0.959-1.000).

**P3 against surface coupling.** **The inner learner beats surface
coupling under decorrelation** at both weaker priors and both budgets:
U(λ) is 1.9× (0.005, b = 2), 4.9× (0.005, b = 10), 2.1× (0.001, b = 2) and
5.3× (0.001, b = 10) worse, 10/10 seeds, with E filtering. At prior 0.05
there is no such result: U is clearly better at b = 2 (3.5×) and they are
comparable at b = 10.

**P4 signal gain.** **Restored** at both weaker priors and budgets (mean
0.965-0.987, ≥ 0.9 in 10/10 seeds). Not restored at 0.05 (0.778 / 0.853).

**P5 decay and clipping.** **No difference established** for any arm at any
prior.

- Removing clips changed nothing measurable (ratios 0.999-1.000), although
  E hits its clips 135-5,700 times per setting.
- Removing decay, alone or with clips, raised X_dec 1.07-1.36×, which is
  unresolved. The no-decay arms also lost the "filters" category at most
  settings (γ 0.21-0.26), keeping it only at prior 0.05, b = 2 (γ 0.15).

**P6 exact training.** **No effect detected** in all six settings (ratios
0.98-1.00).

- At prior 0.05, the learner's 32-step training settles reach gradient
  discrepancy 0.049 in stage A and about 1e-5 in stage B, as in the
  diagnostic.
- At weaker priors the discrepancy is much larger: up to 0.38 (0.005) and
  0.49 (0.001), with stage-B maxima of 0.008-0.015 and about 0.10, above the
  1e-6 tolerance in nearly every update.

Even so, exact training gave the same final errors.

**P7 conditioning.** No final model is flagged. The smallest Q eigenvalue
is at least 0.00057 and the largest masked condition number at most 1,563.
Weak priors lower the eigenvalue floor as expected (0.0094 at 0.05,
about 0.0006-0.0007 at 0.001). Weights did not grow: at weak priors, the
learner's top V singular value falls from 1.41 to 0.83 and W's from 2.48 to
2.08.

## Cross-evaluation (descriptive)

Final E models evaluated at exact equilibrium under the other priors,
weak regime, X_dec, b = 2 / 10:

| Trained at | Evaluated at 0.05 | at 0.005 | at 0.001 |
| --- | --- | --- | --- |
| 0.05 | 0.0069 / 0.0033 | 0.00081 / 0.00075 | 0.00072 / 0.00072 |
| 0.005 | 0.0085 / 0.0056 | 0.00082 / 0.00078 | 0.00071 / 0.00072 |
| 0.001 | 0.0135 / 0.0106 | 0.00093 / 0.00087 | 0.00072 / 0.00072 |

- Weakening the priors at inference repairs every model.
- Restoring strong priors hurts models trained at weak priors more. Their
  smaller V leaves less margin against the stronger shrinkage.
- So training adapted to the weak priors by shrinking weights slightly, not
  by over-compensating.

## Reading

The question was whether training with weaker activity priors keeps the
repaired completion and the filtering. Here it does:

- the repair survives local learning unchanged;
- filtering holds;
- signal gain is restored;
- the inner learner beats surface coupling on the decorrelated test and
  matches the best non-local reference.

That turns the inner channel's filtering, first seen in correlated nuisance
v1, from an interesting mechanism into a measured advantage on this
fixture.

What this does and does not show:

- **The earlier inner-coupling deficit was mostly the activity priors.** On
  v1's own data the ranking reverses at weak priors. Comparisons between
  topologies in earlier studies (local contrastive v1, the clamped-energy
  ablation, correlated nuisance v1) were all made at prior 0.05. Their
  conclusions hold for that prior and should not be read as properties of
  the topologies alone.
- **Linear modules and one fixture.** As before, this tests a low-rank
  channel's filtering, not abstraction.
- **Decay matters for filtering, clips do not** under this protocol, at
  these budgets.
- **Training-settle errors are large at weak priors but did not change
  results here.** That is specific to this rule, protocol and fixture; the
  32-step settle's slower convergence at weak priors could matter
  elsewhere.
- **The retention criterion is a poor measure at low error.** Its ratio
  form fails runs whose absolute error after stage B is lower than a passing
  run's. A future contract should declare retention on absolute avoidable
  error or with a floor.

## Limitations

- Four selected learning rates are at a grid edge (E and E_exact at prior
  0.05, b = 2, and U at 0.05, b = 2, at 0.3; independent E_noclip 0.005,
  b = 10).
- Both layers share one prior value. Separate layer priors and learned
  priors are untested.
- Prior 0.001 is the smallest tested. Smaller priors approach singular
  systems.
- One fixture, two regimes, two budgets, one chain topology, linear modules.

## Next questions

- Update the research-direction note and the prior-work survey: the
  topology comparisons need re-reading at weak priors.
- A study that carries the weak-prior setting into the correlated-nuisance
  topology comparison (inner, outer, outer-matched, uniform) and the strong
  regime.
- Retention criteria on absolute error.
- Then the planned move to nonlinear modules and non-quadratic energies,
  where prior strength becomes a design choice to declare up front.

## Computation

Measured on one CPU thread (float64):

| Phase | Wall (s) |
| --- | ---: |
| Verification (including the diagnostic's checks and one reproduction) | 6.0 |
| Selection pilot (1,296 trajectories) | 502.7 |
| Confirmatory runs (720) | 430.5 |
| Total before runtime serialization | 940.7 |

- Training CPU per run, excluding evaluation and the settle check: E 0.38 s,
  E_exact 0.32 s, E_noclip, E_nodecay and E_free 0.38 s, U 0.52 s. Exact
  training was cheaper per update than 32 implicit steps.
- Counts: 1,210,204 Adam steps; 1,116,945 settles; 765,432 exact solves;
  360,000 settle-check exact solves; 19,713,728 replay-trace exposures.
  These include the pilot and verification.
- Peak RSS 443 MB. The pilot is 53% of wall time.
- My pre-run estimate of 25-35 minutes was high; the run took 16.

The runtime `dirty_status` lists only the uncompressed results file, which
is written before the provenance step.
