# Weak-prior topology v1: results

Results for the frozen [contract](weak-prior-topology-v1-contract.md), run
2026-10-26 at revision 8fe7fec. Checkpoint study for the continuous linear
line; see the [linear-line checkpoint](linear-line-checkpoint.md).

[Runner](../experiments/weak_prior_topology.py) · results
(`reports/weak-prior-topology-v1-results.json.gz`, gzip -n -9, 27.1 MB
uncompressed, SHA-256 80655b4c…c08066) ·
[runtime](../reports/weak-prior-topology-v1-runtime.json). No movement file.
Reproduce with `.venv/bin/python -m experiments.weak_prior_topology` (about
11 minutes). A concurrent repeat run produced a byte-identical results file.

## Summary

Each correlated-nuisance conclusion, re-tested at the original prior (0.05)
and a weak prior (0.001) on fresh seeds 53-62:

| Conclusion | Prior 0.05 | Prior 0.001 | Property of |
| --- | --- | --- | --- |
| Surface-coupled families transmit correlated nuisance (weak and strong regimes) | yes | yes | topology |
| Inner coupling filters it (weak regime) | yes | yes | topology |
| Inner coupling transmits when nuisance dominates (strong regime) | descriptive only: inner does not learn | descriptive only: inner does not learn | not established |
| Adding deeper coupling at fixed surface strength changes transmission | no effect | no effect | topology |
| Inner coupling is more robust to the correlation breaking (main test) | counts against | **supports the mechanism** | prior |
| On v1's data, uniform coupling beats inner coupling | yes | **reversed** at b = 10; unresolved at b = 2 | prior |

- **Every transmission and filtering category is the same at both priors.**
  What each topology passes on is a property of the topology.
- **Every performance ranking that involved inner coupling changed with the
  prior.** At prior 0.001 the inner family beats uniform and outer coupling
  on the decorrelated test by 2.4-7.1×, in every seed. It matches
  uncentered rank-2 regression (0.00057 against 0.00057).

## Results

X is exact-equilibrium avoidable MSE per coordinate, mean over seeds 53-62;
"gain" is signal gain on the decorrelated set.

**Weak regime**, X_dec / γ / gain, b = 2 and b = 10:

| Family | Prior 0.05, b = 2 | Prior 0.05, b = 10 | Prior 0.001, b = 2 | Prior 0.001, b = 10 |
| --- | --- | --- | --- | --- |
| inner | 0.0067 / 0.05 / 0.78 | 0.0031 / 0.05 / 0.86 | **0.00057 / 0.06 / 0.99** | **0.00057 / 0.06 / 0.99** |
| uniform | 0.0018 / 0.62 / 0.94 | 0.0035 / 1.05 / 0.99 | 0.0014 / 0.63 / 0.99 | 0.0035 / 1.06 / 1.00 |
| outer | 0.0021 / 0.76 / 0.96 | 0.0041 / 1.14 / 0.99 | 0.0019 / 0.76 / 0.99 | 0.0042 / 1.14 / 1.00 |
| outer-matched | 0.0018 / 0.58 / 0.93 | 0.0035 / 1.04 / 0.98 | 0.0014 / 0.59 / 0.99 | 0.0034 / 1.04 / 1.00 |

Comparators (weak): uncentered rank-2 regression 0.00057 (γ 0.06); ridge
with intercept 0.0028 (γ 0.95).

In-distribution cost: inner's X_in at prior 0.001 is 0.0064, against
0.0004-0.0021 for the surface families. Inner coupling passes on almost no
predictable nuisance, so it gives up the in-distribution gain from
predicting it.

**Independent regime** (v1's data; X_in = X_dec):

| Family | Prior 0.05, b = 2 / 10 | Prior 0.001, b = 2 / 10 |
| --- | --- | --- |
| inner | 0.0066 / 0.0031 | **0.00052 / 0.00052** |
| uniform | 0.0010 / 0.0015 | 0.0007 / 0.0015 |
| outer | 0.0011 / 0.0023 | 0.0009 / 0.0022 |
| outer-matched | 0.0012 / 0.0016 | 0.0008 / 0.0015 |

Uncentered rank-2 regression: 0.00051.

**Strong regime:** every family transmits at both priors (γ 0.78-1.29).
Inner coupling does not learn at either prior: it removes only 58-67% of its
frozen error, and its signal gain is 0.11-0.15. The surface families' X_dec
is 0.157-0.184, above the full oracle's (about 0.13 on earlier seeds), and
inner's is 0.179-0.206.

## Criteria

**N1 learning** (reported, not a gate), with retention now on absolute
error:

- At prior 0.001, weak inner learns at both budgets (A retained 9/10 seeds).
  Under the old ratio form it would have failed (3/10).
- At prior 0.05, weak inner fails absolute retention (0-1/10). Its A error
  after stage B stays above 5% of frozen; the ratio form passes it at b = 2.
- Strong inner fails at both priors.
- Every surface-family condition learns.

**T1 control (independent regime).**

| Comparison | Prior 0.05, b = 2 | b = 10 | Prior 0.001, b = 2 | b = 10 |
| --- | --- | --- | --- | --- |
| inner vs uniform | inner clearly worse (6.3×) | inner clearly worse (2.0×) | unresolved (uniform 1.34× higher) | **uniform clearly worse (2.8×)** |
| inner vs outer | inner clearly worse (5.9×) | unresolved (1.36×) | **outer clearly worse (1.7×)** | **outer clearly worse (4.2×)** |
| inner vs outer-matched | inner clearly worse (5.5×) | inner clearly worse (1.9×) | unresolved (1.4×) | **outer-matched clearly worse (2.9×)** |
| uniform vs outer-matched | comparable | comparable | comparable | comparable |

**T2 transmission.** Identical categories at both priors and budgets:

- weak inner filters (γ 0.05-0.06);
- weak uniform, outer and outer-matched transmit (γ 0.58-1.14);
- every strong-regime condition transmits.

**T3 robustness (weak regime).**

- **Prior 0.05: counts against the mechanism** at both budgets, replicating
  correlated nuisance v1 on fresh seeds. Inner is dominated by uniform and
  outer at b = 2 (3.1-3.7×), and by uniform at b = 10. Inner does not learn
  here.
- **Prior 0.001: supports the mechanism** at both budgets, with inner
  filtering and learning:
  - at b = 2, uniform is 2.4× and outer 3.3× worse than inner on X_dec;
  - at b = 10, 6.1× and 7.1×;
  - 10/10 seeds each.

  The in-distribution cost is inner 3.6-14.6× worse on X_in.

**T4 regime dependence.** **Descriptive only** at both priors, because
strong-regime inner does not learn. Inner filters in the weak regime and
transmits (γ 0.78-0.92) in the strong one, the direction the
variance-ranking hypothesis predicts. A trained inner channel in the strong
regime has not been obtained in this line.

**T5 deeper coupling at fixed surface strength.** **No effect** at both
priors and budgets (γ ratio 1.02-1.07). X_dec and X_in are comparable.
Adding middle and inner coupling to the same surface coupling does not
change what is passed on.

**T6 prior dependence.**

- **Unchanged between priors:** T2 (all categories), T4 (descriptive at
  both), T5 (no effect at both), and uniform against outer-matched in T1.
- **Changed:** T3 (counts against → supports) and every T1 comparison
  involving inner coupling.

## Reading

This is the checkpoint the line needed:

- **What a topology transmits is a property of the topology**, at both
  priors tested:
  - surface coupling passes on non-persistent correlations;
  - a narrow inner channel filters them when they are weaker than the
    signal;
  - adding inner coupling to surface coupling does not gate the surface.
- **How well the inner channel performs depended on the activity prior.**
  At prior 0.05 its completion is shrunk, and it loses to surface coupling
  even on v1's data. At prior 0.001 it recovers its signal and is the best
  learned family, robust to the correlation breaking and matching the best
  non-local reference. Earlier rankings of inner against surface coupling
  belong to the prior 0.05 setting.
- **Two things remain open:**
  - a learning inner channel in the strong regime;
  - the strong regime itself, where every family transmits and exceeds
    the full oracle's decorrelated error.

## Limitations

- Six selected learning rates are at a grid edge, all at b = 2: weak inner
  at 0.05 and the outer-matched and uniform conditions at the top edge 0.3.
- Two priors only; both layers share one value.
- Linear modules, one fixture, one chain topology.
- The runtime JSON's `dirty_status` lists
  `docs/relational-abstraction-learning-proposal.md` as modified during the
  run. That edit was outside this study, and the runner and contract hashes
  are recorded. It also lists the uncompressed results file, written before
  the provenance step.

## Computation

Measured on one CPU thread (float64):

| Phase | Wall (s) |
| --- | ---: |
| Verification (including prior-strength training's checks and one reproduction) | 6.7 |
| Selection pilot (864 trajectories) | 348.4 |
| Confirmatory runs (480) | 303.2 |
| Total before runtime serialization | 669.4 |

- Training CPU per run, excluding evaluation and the settle check: inner
  0.38 s, outer and outer-matched 0.39 s, uniform 0.51 s.
- Counts: 807,604 Adam steps; 895,295 settles; 433,194 exact solves;
  288,000 settle-check exact solves; 13,161,728 replay-trace exposures.
- Peak RSS 443 MB.
