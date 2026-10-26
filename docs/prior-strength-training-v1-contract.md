# Prior-strength training v1: does training with weaker activity priors keep the repaired completion and the filtering?

Prospective contract, 2026-10-26. Frozen before execution. Follows the
[inner-channel diagnostic](inner-channel-diagnostic-v1-results.md) and the
[energy scaling analysis](energy-scaling-analysis.md).

The scaling analysis found that inner coupling's lost signal is a scalar gain
loss: completions reproduce the signal at 0.78-0.86 of its size, and the
fixed activity priors cause it. At the clamped-energy learner's own fixed
parameters, lowering both priors from 0.05 to 0.005, without retraining,
raised its signal gain to 0.97-0.98 and cut its decorrelated error 7-14×,
with filtering intact (nuisance gain about 0.05).

That established the inference mechanism. This study asks the learning
question: when the learner is **trained** with weaker priors, do the learned
parameters keep that improvement and the filtering? Training adapts W, V and
the translations to the new priors. It could undo the gain, as the
unconstrained capacity fit's weights over-compensated, or lose the filtering.

It also separates, as the diagnostic's review asked:

- weight decay from clipping, as separate arms;
- 32-step from exact-equilibrium training, with everything else fixed.

As before, this is a search for constructive results. A change that does not
help is reported as "no improvement found", and its cause stays unresolved.

## Design provenance and contamination

This design was informed by results on seeds 23-42 (correlated nuisance v1
and the inner-channel diagnostic) and by the post-hoc scaling and
prior-intervention analyses of seeds 33-42. Seeds 0-2 are the pilot again.
Confirmatory claims use new seeds 43-52. Seed 33 is used once, only for the
reproduction check in Verification.

## Prior work

Reviewed under [literature review for contracts](literature-review.md).
Shared survey: [prior work](prior-work.md). No new searches were run for
this draft: the components are established, or are project results. Tags
as defined in the survey.

- **Related methods.**
  - The activity priors are Gaussian priors on latent activities, the same
    role as the unit prior on latents in factor analysis and probabilistic
    PCA [familiar]. Their effect on completion is posterior shrinkage, as in
    ridge regression [familiar].
  - In those models, latent scale is not identified apart from the prior:
    weights and latents can trade scale, and the prior fixes it [familiar].
    The scaling analysis's identity is this project's version: scaling W and
    V by (s, t) is equivalent to reweighting every deeper energy term
    [derived].
  - Exact-equilibrium training is the clamped-energy rule with its settle
    replaced by an exact solve: the same gradient, evaluated at the settled
    state the rule assumes.
- **What theory and earlier results already predict.**
  - *At fixed parameters [project result].* Lowering both priors to 0.005
    gives the learner gain 0.97-0.98 and decorrelated error 0.00040-0.00046,
    with nuisance gain about 0.05.
  - *Scale drift [derived from the identity, a hypothesis for training].* The
    priors and weight decay are what fix the scale shared between weights
    and activities. Weaker priors leave decay as the main scale anchor. With
    neither decay nor clips, weights and codes may drift in scale, and Q may
    approach singularity. This study measures it.
  - *Unit translation gain [hypothesis].* The learner trains with every outer
    layer clamped, so completion-time shrinkage never appears in its training
    states, and its translations stayed at gain 1.00. If that persists,
    weaker priors reduce the shrinkage it fails to compensate, and the gain
    should approach the fixed-parameter values.
- **Known failure modes.**
  - Weak priors weaken the positive definiteness margin: the smallest
    eigenvalue of Q scales with the priors. Conditioning is therefore
    measured, and the smallest prior is 0.001, never 0.
  - Removing decay and clips can let weights grow without bound. A run that
    diverges is disqualified, not rescued.
- **Baselines.** Uniform coupling trained at the same prior strength;
  uncentered rank-2 regression (the best decorrelated reference so far,
  0.00037); ridge with intercept.
- **What this adds.** Whether the inference repair survives learning, for
  this local rule, plus the separated effects of decay, clipping and exact
  settling. "Nothing new beyond that" is an acceptable answer.

## Fixture and architecture

[Correlated nuisance v1](correlated-nuisance-v1-contract.md)'s fixture,
unchanged, in the **independent** and **weak** regimes. The architecture
and energy are unchanged, except for the activity priors. Three 8 → 4 → 2
linear modules sit in a 0 ↔ 1 ↔ 2 chain, with free translations.

**Activity priors.** The energy's prior term becomes
(λ/2)(‖h‖² + ‖c‖²) for every module, so λ is the Hessian value (0.05 in all
earlier studies). Both layers use the same λ. Training and evaluation use
the same λ, except in the cross-evaluation below.

**Budgets** b ∈ {2, 10}. **Families:** inner (κ_inner = b/2) for every
learner arm, and uniform (κ = b/14 per layer) for the reference arm.

## Factors and arms

**Prior strength** λ ∈ {0.05, 0.005, 0.001}.

**Learner arms**, all with the clamped-energy rule, Adam (β₁ = 0.9,
β₂ = 0.999, ε = 1e-8), stage A then A+B replay and 300 cycles per stage:

| Arm | Family | Settle used for the gradient | Weight decay on W, V | Clips (W, V ±2; T ±3) |
| --- | --- | --- | --- | --- |
| **E** | inner | 32-step | 1e-3 | yes |
| **E_exact** | inner | exact clamped equilibrium | 1e-3 | yes |
| **E_noclip** | inner | 32-step | 1e-3 | no |
| **E_nodecay** | inner | 32-step | 0 | yes |
| **E_free** | inner | 32-step | 0 | no |
| **U** | uniform | 32-step | 1e-3 | yes |

That gives 6 arms × 3 priors × 2 budgets × 2 regimes = 72 conditions. E at
λ = 0.05 is the inner-channel diagnostic's arm E.

All arms start from the shared W, V initialization (SeedSequence
[seed, 1001]) with T = I. Each run has its own fresh Adam state.

**Comparators**, fitted on all waking traces at each stage: uncentered
rank-2 regression and ridge with intercept, as in the diagnostic. They do
not depend on λ.

## Pilot selection (seeds 0-2 only)

For each of the 72 conditions, select η from
{0.001, 0.003, 0.01, 0.03, 0.1, 0.3}:

- train on even-index waking traces;
- score exact-equilibrium observed masked MSE, under the condition's own λ,
  on odd-index traces after stage B;
- ties go to the smaller η;
- a non-finite parameter, score or energy disqualifies that η;
- if every η is disqualified, the condition is recorded as failed and
  counts as "no improvement found" in every comparison.

The grid adds 0.3 above the diagnostic's grid, because its unconstrained
fit's pilot was still improving at 0.1. Grid-edge selections are reported,
with no extension. The pilot never scores test or decorrelated sets. All
pilot runs are charged to a named selection phase.

## Evaluations

At initialization, after each stage and every 50 cycles; exact-equilibrium
inference is primary:

- X_in, X_dec, nuisance gain γ and its symmetric and antisymmetric parts,
  alignment τ, and the signal-error part of each;
- **signal gain** on the decorrelated set: the regression coefficient of
  the completion's pattern-span component on μ_sig, pooled over hidden
  modules (as in the scaling analysis);
- observed RMSE, training-set error, generalization gap and N1 learning;
- top singular values of each module's W and V and of each translation;
- **conditioning:** the largest condition number of the three masked free
  blocks, and the smallest eigenvalue of Q;
- clip activations by parameter group;
- **training-settle check** for the 32-step arms, at every update: the
  gradient discrepancy r = ‖g − g*‖ / max(‖g*‖, 1e-8), as in the diagnostic.
  Only summaries are stored: the per-run maximum and the update where it
  occurs, the stage-A and stage-B maxima, and the number of updates with
  r > 1e-6.

**Cross-evaluation (descriptive).** Each final E and U model is also
evaluated, at exact equilibrium, under the other two prior strengths. The
learned parameters are kept, and only Q's prior terms change, as in the
prior-intervention analysis. This shows whether training at weak priors
adapted the parameters to them.

## Criteria (confirmatory seeds 43-52, declared before execution)

All criteria use the weak regime and final exact-equilibrium values, and
are evaluated at each budget separately. Decision rules are the
diagnostic's:

- ratios of avoidable errors use (X₁ + δ) / (X₂ + δ), δ = 1e-5;
- **comparable:** larger ratio ≤ 1.25;
- **clearly better / worse:** ratio ≥ 1.5 with the ordering in ≥ 9/10
  seeds;
- otherwise **unresolved**;
- **filters:** γ ≤ 0.2 in ≥ 9/10 seeds; **transmits:** γ ≥ 0.5 in ≥ 9/10
  seeds.

**N1 is reported, not a gate**, as in the diagnostic. It gates only claims
that a learning protocol succeeds. A failed condition from the pilot counts
as "no improvement found".

- **P0 control.** E at λ = 0.05: report its filtering category, signal gain
  and X_dec beside the diagnostic's values. Descriptive.
- **P1 weaker-prior training (main).** For each λ ∈ {0.005, 0.001}, compare
  E(λ) with E(0.05) on X_dec.
  - **Training with weaker priors improves decorrelated completion** if
    E(0.05) is clearly worse than E(λ) and E(λ) filters.
  - **Improves but loses filtering** if E(0.05) is clearly worse and E(λ)
    does not filter; γ is reported.
  - Otherwise **no improvement found**.
- **P2 training against inference-only repair.** For each λ ∈ {0.005,
  0.001}, compare E trained at λ with E trained at 0.05 and evaluated at λ
  (cross-evaluation) on X_dec.
  - **Training improves on the inference-only repair** if the
    cross-evaluated model is clearly worse.
  - **Training erodes the repair** if E trained at λ is clearly worse.
  - **Training preserves it** if they are comparable.
  - Otherwise unresolved.
- **P3 against surface coupling.** For each λ, with E(λ) filtering:
  - **The inner learner beats surface coupling under decorrelation** if
    U(λ) is clearly worse than E(λ) on X_dec;
  - otherwise **no such result**, with the ordering reported.

  Also reported against U(0.05) and against uncentered rank-2 regression.
- **P4 signal gain restored.** For each λ, the gain is **restored** if E(λ)'s
  mean signal gain is ≥ 0.95 and it is ≥ 0.9 in ≥ 9/10 seeds. Otherwise
  report the gain.
- **P5 decay and clipping.** At each λ, compare each of E_noclip, E_nodecay
  and E_free with E on X_dec.
  - **Removing clips / decay / both helps under this protocol** if E is
    clearly worse. The arm's filtering category is reported beside it.
  - **Does worse** if the arm is clearly worse.
  - Otherwise **no difference established**.

  An improvement that loses filtering is reported as such, not as a pass.
- **P6 exact training.** At each λ, compare E_exact with E on X_dec.
  - **Exact training helps** if E is clearly worse.
  - **Exact training does worse** if E_exact is clearly worse.
  - **No effect detected** if they are comparable.
  - Otherwise unresolved.

  The comparison holds everything else fixed, so a detected difference is
  attributed to the settle used in training under this protocol. E's
  settle-check summaries are reported beside it.
- **P7 conditioning and scale.** Report, for every arm and λ, the
  conditioning measures and top singular values. Flag any final model with
  smallest Q eigenvalue ≤ 1e-6 or largest masked condition number ≥ 1e8.
  Comparisons involving a flagged model are reported but support no
  conclusion. Descriptive otherwise.

These are screening rules on ten seeds, not significance tests. Partial
passes are reported as such. There are no post-result extensions of arms,
priors, cycles, grids, budgets or seeds.

**Reading.**

- A pass on P1 with filtering, and P2 "preserves" or "improves", would show
  that the repair survives local learning. With P3, it would show that the
  inner learner can beat surface coupling under decorrelation on this
  fixture.
- "Erodes" on P2 would mean training partly compensates the weaker priors
  away. The cross-evaluation and singular values would show how.
- P5 and P6 locate whether decay, clipping or settling matter once the
  priors are weaker.

## Computation

Per learner cycle, as in the diagnostic:

- 32-step arms: one 18-dimensional clamped factorization, 32 batched steps
  and one energy derivative, O(18³ + B·32·18² + BP).
- E_exact: one exact clamped solve instead of the 32 steps.
- The settle check adds one exact clamped solve and one energy derivative
  per update for the 32-step arms.
- Evaluations add masked exact solves, completion maps and a 42 × 42
  eigendecomposition.
- Cross-evaluation adds two exact evaluations per final E and U model.

Executed totals:

- confirmatory runs: 72 conditions × 10 seeds × 600 cycles;
- pilot: 72 × 6 η × 3 seeds = 1,296 trajectories of 600 cycles.

From the diagnostic's measured costs (about 1.2 ms per cycle with the settle
check), the pilot should take about 15-20 minutes and the confirmatory runs
about 10-15 minutes. Report executed counts, CPU per arm, and quality
against CPU.

Storage per learner, float64 (P = 128 inner, 288 uniform):

| Item | Bytes |
| --- | ---: |
| Parameters | 8P |
| Adam moments and step counter | 16P + 8 |
| Replay traces, 32 × 24 | 6,144 |
| Dense Q, 42 × 42 | 14,112 |
| Clamped settle workspace, 18 × 18 plus 32 × 18 states | 7,200 |
| Settle-check accumulators | 48 |

Outputs, following [computation reporting](computation-reporting.md):

- a deterministic scientific JSON, compressed with `gzip -n -9` before its
  first commit, with its uncompressed SHA-256 recorded;
- **no per-cycle movement file;** per-run summaries go in the results;
- a runtime JSON with hashes, revision, dirty status, versions, thread
  settings, per-phase wall and CPU (selection charged to a named phase),
  peak RSS, and counts of settles, exact solves and clip activations.

## Verification

- **Reproduction:** on seed 33, weak / E / b = 10 at λ = 0.05 and η = 0.1
  reproduces the inner-channel diagnostic's final X_in and X_dec to 1e-12.
- The prior-modified Q equals `lcc.build` exactly at λ = 0.05, and matches
  central finite differences of the explicit energy at λ = 0.005 and 0.001.
- The exact-training gradient equals the energy derivative at the exact
  clamped equilibrium, and the 32-step gradient converges to it as the step
  count grows.
- E_free and E_noclip apply no clips; E_free and E_nodecay apply no decay
  (checked on constructed parameters, as in the diagnostic).
- Q is positive definite at initialization for every λ, arm and budget.
- Cross-evaluation at the training λ reproduces the condition's own final
  values exactly.
- Every applicable check carried over from the diagnostic.

Out of scope: masked-completion and capacity-fit arms, separate middle and
inner prior strengths, learned priors, the strong regime, outer families,
nonlinear modules and non-quadratic energies.
