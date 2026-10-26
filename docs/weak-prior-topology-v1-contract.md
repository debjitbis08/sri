# Weak-prior topology v1: do the correlated-nuisance topology conclusions survive weak activity priors?

Prospective contract, 2026-10-26. Frozen before execution. Checkpoint study closing the
continuous linear line before the move to
[relational abstraction learning](relational-abstraction-learning-proposal.md).

[Prior-strength training v1](prior-strength-training-v1-results.md) found
that the inner learner's earlier deficit was mostly the fixed activity
priors. Trained at prior 0.001, the inner learner kept its filtering,
recovered its signal, and beat uniform coupling under decorrelation. On
v1's own data the uniform-over-inner ranking reversed.

Every topology comparison in
[correlated nuisance v1](correlated-nuisance-v1-results.md) was made at
prior 0.05. This study repeats that comparison at a weak prior, beside the
original prior on the same fresh seeds. It re-checks which conclusions
belong to the topologies and which belonged to the prior:

- whether surface coupling transmits non-persistent correlations;
- whether inner coupling filters them;
- whether filtering depends on the noise regime;
- whether adding deeper coupling to surface coupling changes transmission.

It adds no new mechanism.

## Design provenance and contamination

Informed by results on seeds 23-52. Seeds 0-2 are the pilot. Confirmatory
claims use new seeds 53-62. Seed 43 is used once, only for the reproduction
check.

## Prior work

Reviewed under [literature review for contracts](literature-review.md); no
new searches. The design repeats correlated nuisance v1's, whose prior-work
section and the shared [survey](prior-work.md) cover the methods: early
versus late fusion, rank constraints and spurious correlation, alignment
versus cross-prediction under correlated nuisance.

What earlier results already predict:

- **Transmission is about topology, not priors** [project result]. In
  prior-strength training, uniform coupling transmitted at every prior
  (γ 0.61-1.04), and lowering the priors barely changed its decorrelated
  error.
- **The inner family should filter in the weak regime and beat the surface
  families on the decorrelated test** [project result, at b = 2 and 10
  against uniform only].
- **In the strong regime the inner channel transmits** [hypothesis].
  Correlated nuisance v1's strong-regime inner conditions did not learn
  (frozen ratio about 0.42), so this was never established. The
  variance-ranking hypothesis still predicts it: shared nuisance (1.0 per
  direction) outranks the signal's second moment (0.51).

What this adds: the topology conclusions restated at the prior that the
last study showed matters. "Nothing new beyond that" is an acceptable
answer.

## Fixture, architecture and protocol

Unchanged from [correlated nuisance v1](correlated-nuisance-v1-contract.md):

- the fixture in its **independent, weak and strong** regimes;
- three 8 → 4 → 2 linear modules in a 0 ↔ 1 ↔ 2 chain, with free
  translations clipped to ±3;
- W and V clipped to ±2, with weight decay 1e-3;
- the clamped-energy rule, 32-step training settles, Adam, stage A then
  A+B replay, 300 cycles per stage.

The activity prior λ is applied as in
[prior-strength training](prior-strength-training-v1-contract.md), to both
layers, in training and evaluation.

## Factors

| Factor | Levels |
| --- | --- |
| Regime | independent, weak, strong |
| Topology family | uniform (κ = b/14 per layer), inner (κ_inner = b/2), outer (κ_outer = b/8), outer-matched (κ_outer = b/14) |
| Budget level | b ∈ {2, 10} |
| Activity prior λ | 0.05 (the original), 0.001 |

That gives 48 conditions. No decay, clip or exact-training arms: in
prior-strength training, clipping and exact training made no detectable
difference, and decay is kept.

**Comparators:** uncentered rank-2 regression and ridge with intercept,
fitted on all waking traces.

## Pilot selection (seeds 0-2 only)

For each condition, select η from {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}
exactly as in prior-strength training:

- exact-equilibrium observed masked MSE under the condition's own λ, scored
  on odd-index waking traces after training on even-index traces;
- ties go to the smaller η;
- non-finite values disqualify an η, and a condition with no eligible η is
  recorded as failed.

Grid-edge selections are reported, with no extension.

## Evaluations

As in prior-strength training, exact-equilibrium inference primary:

- X_in, X_dec, nuisance gain γ and its symmetric and antisymmetric parts,
  alignment τ, and signal error and signal gain;
- conditioning (smallest Q eigenvalue, largest masked condition number) and
  top singular values;
- the training-settle check as per-run summaries;
- no per-cycle movement file.

## Criteria (confirmatory seeds 53-62, declared before execution)

Decision rules are correlated nuisance v1's:

- ratios (X₁ + δ) / (X₂ + δ), δ = 1e-5;
- **comparable** ≤ 1.25;
- **clearly better / worse:** ≥ 1.5 with the ordering in ≥ 9/10 seeds;
- otherwise **unresolved**;
- **filters:** γ ≤ 0.2 in ≥ 9/10 seeds; **transmits:** γ ≥ 0.5 in ≥ 9/10
  seeds; otherwise unresolved.

Each criterion is evaluated at each budget and at each λ. The λ = 0.001
outcome is the new result; the λ = 0.05 outcome is the same-seed
replication of correlated nuisance v1.

**Learning (N1)** is reported for every condition. It is not a gate, and its
retention part is redefined on absolute error, as prior-strength training
recommended. A condition **learns** if:

- B's avoidable error falls during stage B in ≥ 9/10 seeds;
- A's avoidable error after stage B is ≤ 0.05 × its frozen value in
  ≥ 9/10 seeds;
- final X_in is ≤ 0.1 × its frozen X_in.

The correlated-nuisance ratio form (A after B ≤ 1.5 × A after A) is also
reported, for comparison only.

- **T1 control (independent regime).** All six pairwise family comparisons
  on X (X_in = X_dec). Descriptive, at both priors.
- **T2 transmission (weak and strong).** The N3 category of every family.
- **T3 robustness (weak regime).** Inner against uniform and inner against
  outer on X_dec, with correlated nuisance v1's pairwise verdicts:
  - **advantage:** the other family is clearly worse;
  - **dominated:** inner is clearly worse, or comparable on X_dec while
    clearly worse on X_in;
  - **neutral:** comparable on X_dec and not dominated;
  - otherwise **unresolved**.

  The aggregation applies in order:
  1. **mixed** if one pair is an advantage and the other dominated;
  2. **counts against** if any pair is dominated and none is an advantage;
  3. **supports the mechanism** if both pairs are advantages and inner
     filters;
  4. **partial support** if one pair is an advantage, the other neutral,
     and inner filters;
  5. **advantage not explained by filtering** if 3 or 4 holds except that
     inner does not filter;
  6. otherwise **unresolved**.

  X_in for the same pairs is reported as the in-distribution cost.
- **T4 regime dependence (inner family, weak against strong).**
  - **Filtering depends on the regime** if inner filters in the weak regime
    and transmits in the strong one, and both inner conditions learn (N1).
  - **Filtering persists** if inner filters in both, both learning.
  - Otherwise the categories are reported descriptively.
- **T5 deeper coupling at fixed surface strength (weak regime).** Uniform
  against outer-matched on γ, with correlated nuisance v1's γ-ratio rule:
  - ratio (γ₁ + 0.01) / (γ₂ + 0.01);
  - **reduces** if ≤ 1/1.5 and lower in ≥ 9/10 seeds;
  - **increases** if ≥ 1.5 and higher in ≥ 9/10 seeds;
  - **no effect** if within [1/1.25, 1.25];
  - otherwise unresolved.

  X_dec, X_in and the signal-error split are reported beside it.
- **T6 prior dependence of each conclusion.** For each of T1-T5, report
  whether the λ = 0.001 outcome differs from the λ = 0.05 outcome on the
  same seeds. A difference means that conclusion belonged to the prior. No
  difference means it is a property of the topology at both priors tested.
- **R references.** For every condition, (X_in, X_dec) and γ beside the two
  comparators.

These are screening rules on ten seeds, not significance tests. Partial
passes are reported as such. There are no post-result extensions.

## Computation

Per-run costs as in prior-strength training (about 0.4-0.5 s of training CPU
per run, plus evaluation).

- Confirmatory runs: 48 conditions × 10 seeds × 600 cycles.
- Pilot: 48 × 6 × 3 = 864 trajectories.

The expected total is about 15 minutes. Storage per learner as in
prior-strength training (P = 128 inner, 248 outer and outer-matched,
288 uniform).

Outputs, following [computation reporting](computation-reporting.md):

- a deterministic scientific JSON, compressed with `gzip -n -9` before its
  first commit, with its uncompressed SHA-256 recorded;
- a runtime JSON;
- no movement file.

## Verification

- **Reproduction:** on seed 43, weak / inner / b = 10 at λ = 0.001 and its
  prior-strength η (0.1) reproduces prior-strength training's arm E final
  X_in and X_dec to 1e-12.
- At λ = 0.05 the prior-modified Q equals `lcc.build` exactly.
- Q is positive definite at initialization for every family, budget and λ.
- The absolute-retention criterion and the ratio form are both computed
  from the same recorded evaluations.
- Every applicable check carried over from prior-strength training.

Out of scope: decay, clip and exact-training arms, other priors, separate
layer priors, nonlinear modules and non-quadratic energies.
