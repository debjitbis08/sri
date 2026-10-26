# Inner-channel diagnostic v1: can inner coupling keep its filtering and recover the signal?

Prospective contract, 2026-10-26. Frozen before execution. Follows
[correlated nuisance v1](correlated-nuisance-v1-results.md).

In that study, inner-only coupling filtered weak correlated nuisance almost
completely (nuisance gain γ ≈ 0.05) while every surface-coupled family
transmitted it. But the narrow inner channel lost so much signal that it was
worse than surface coupling even on the decorrelated test at b = 2, and
better only at b = 10 in a condition that failed retention. The question
here is whether the filtering can be kept while the missing signal is
recovered.

This is a search for constructive improvements. It can show that these
parameters can do better than the learner, or that a given change helps. It
cannot reliably assign a failure to architecture, objective or
optimization: a fit that does not improve leaves the cause unresolved, and
claims that no procedure can succeed would need an analytical bound or
certified optimization. The candidate causes it probes:

- **Training settles.** Not expected to matter, but not ruled out. On
  correlated nuisance v1's final parameters (seeds 23-25, inner, uniform and
  outer conditions), 32-step training settles were within about 2e-14 of
  exact equilibrium, and the learning gradient differed from the exact one
  by under 1e-12. Training clamps all three outer layers, so the free system
  is small and settles fast. That check was unrecorded and used final
  parameters only. This study measures the gradient discrepancy at every
  update.
- **Capacity.** These parameters, under these constraints, may not be able
  to represent filtered and accurate completion. Only a successful fit is
  evidence here: it shows the capacity exists.
- **Objective.** The clamped-energy objective may not ask for accurate
  completion, even if the architecture could deliver it.
- **Rank restriction itself.** A rank-2 channel may be unable to filter and
  keep the signal at once. The centered rank-2 comparator in correlated
  nuisance v1 could not, but it centered the data and the energy does not.

## Design provenance and contamination

This design was informed by correlated nuisance v1's results on seeds 23-32
and by the settle check above. Seeds 0-2 are the pilot again. Confirmatory
claims use new seeds 33-42. Seed 23 is used once, only for the reproduction
check in Verification.

## Prior work

Reviewed under [literature review for contracts](literature-review.md).
Shared survey: [prior work](prior-work.md). No new searches were run for
this draft: the components are established methods already covered there,
or project results. Tags as defined in the survey.

- **Related methods.**
  - The masked-completion arm M uses the adjoint gradient of the
    exact-equilibrium masked loss. That is local contrastive v1's
    exact-gradient reference (arm 6). It is a masked-prediction objective
    (familiar from denoising and masked autoencoding) with a non-local
    gradient [familiar].
  - The supervised arms S and S′ are evaluator-assisted capacity probes:
    fitting the architecture directly to the target the learner should
    produce. They are diagnostics, not learners.
  - The uncentered comparator is reduced-rank regression without an
    intercept (Izenman 1975) [familiar].
- **What theory or earlier results already predict.**
  - *Uncentered rank restriction [derived].* Without an intercept, rank-2
    regression keeps the directions of largest uncentered fitted second
    moment. Signal has about 0.51 per pattern coordinate; weak shared
    nuisance has 0.027 per direction. The expectation is that it keeps the
    pattern span, filters (γ near 0) and reconstructs signal about as well
    as ridge. If so, rank restriction is not the barrier, and the question
    moves to the energy architecture and its learning.
  - *Objective [project result].* In local contrastive v1, under uniform
    coupling, the exact masked-gradient arm had 2.15× the avoidable error of
    clamped energy ([floor analysis](local-contrastive-consolidation-v1-floor-analysis.md)).
    The generative energy objective used the data better there. That is
    context for inner coupling, not a prediction; the generative-versus-
    discriminative small-sample result (Ng & Jordan 2002) [familiar] is
    context too.
  - *Shrinkage [hypothesis].* In Gaussian latent models, posterior codes
    shrink toward zero as priors compete with evidence [familiar]. One
    candidate cause of inner coupling's signal loss is that a hidden
    module's inner code, set only by coupling, is shrunk or distorted
    relative to the code the module would infer from its own data. This is
    measured, not assumed.
- **Known failure modes.** Masked objectives can overfit 32 training traces
  per stage; a capacity fit can stop at a local optimum or depend on its
  start, so it uses two starts. Evaluator-assisted fits can exploit
  information a learner never has; they bound what the parameters can
  represent, not what any learner can find.
- **What this adds.** An internal diagnostic that searches for
  constructive ways past a measured limitation: parameters that do better,
  or a change that helps. It can leave the cause unresolved. "Nothing new
  beyond that" is an acceptable answer.

## Fixture and architecture

[Correlated nuisance v1](correlated-nuisance-v1-contract.md)'s fixture,
unchanged, in the **independent** and **weak** regimes. The strong regime is
not run. Architecture and energy are unchanged: three 8 → 4 → 2 linear
modules in a 0 ↔ 1 ↔ 2 chain, free translations clipped to ±3, W and V clipped
to ±2, weight decay 1e-3 on W and V, Adam (β₁ = 0.9, β₂ = 0.999, ε = 1e-8).

Budgets b ∈ {2, 10}. Families: **inner** (κ_inner = b/2) for every arm, and
**uniform** (κ = b/14 per layer) for the reference arm only.

## Arms

All arms start from the same W, V initialization per seed (SeedSequence
[seed, 1001]) and T = I, except the second start of S and S′.

| Arm | Family | Objective | Data and protocol | Deployable |
| --- | --- | --- | --- | --- |
| **E** clamped energy | inner | as correlated nuisance v1 | stage A then A+B replay, 300 cycles per stage | yes, local |
| **U** clamped energy, reference | uniform | as correlated nuisance v1 | as E | yes, local |
| **M** masked completion | inner | exact-equilibrium whole-module masked squared error on the observed hidden module, hidden module rotated over all three; adjoint gradient | as E | yes, non-local |
| **S** capacity fit | inner | the same masked loss with the hidden module's target replaced by μ_sig | all 32 waking traces together, 3,000 cycles; clips and decay as E | no, evaluator-assisted |
| **S′** capacity fit without clips or decay | inner | as S | as S, without clips or weight decay | no, evaluator-assisted |

Arms S and S′ each run from two starts: the shared initialization, and arm
E's final parameters for the same seed, regime and budget. Each start
begins with fresh Adam moments and step counter. The start with the lower
final training objective is reported as the fit; both are recorded.
Selection uses only the S objective on training traces. A start whose
parameters or objective become non-finite is disqualified. If both starts
are disqualified, the fit is recorded as failed, which counts as "no
improvement found".

**Masked loss.** M and S minimize

    L = (1/3) Σ_h (1/n) Σ_i ½ ‖ŷ_h,i − target_h,i‖²

over the three hidden modules h and n training traces, with ŷ the
exact-equilibrium completion of h's outer layer. The squared error is
summed over h's 8 outer coordinates, matching local contrastive v1's
`exact_grad` normalization. The target is the observed outer activity for M
and μ_sig for S and S′. Weight decay 1e-3 on W and V is added to the
gradient for M and S, as for E; Adam's ε is 1e-8 throughout. The loss scale
differs from the clamped energy's, which interacts with ε and decay; that is
accepted and reported, not tuned.

**Comparators**, fitted on all waking traces at each stage:

- ridge without intercept (ridge 0.1);
- **uncentered rank-2 regression:** that fit, with slopes replaced by
  B V₂V₂ᵀ, where V₂ holds the top two right singular vectors of the
  uncentered fitted values;
- for reference, the recentred centered rank-2 regression from correlated
  nuisance v1's post-hoc analysis, and ridge with intercept.

## Pilot selection (seeds 0-2 only)

- **E, U, M:** for each regime and budget, select η ∈ {0.001, 0.003, 0.01,
  0.03, 0.1} as in correlated nuisance v1: train on even-index waking traces
  and score exact-equilibrium observed masked MSE on odd-index traces after
  stage B. Ties go to the smaller η; non-finite values disqualify.
- **S, S′:** select η from the same grid by the mean final S objective on
  pilot training traces, from the shared initialization only.

Grid-edge selections are reported, with no extension. The pilot never
scores test or decorrelated sets. All pilot runs are charged to a named
selection phase.

## Evaluations

As in correlated nuisance v1, with exact-equilibrium inference primary:

- X_in, X_dec, nuisance gain γ and its symmetric and antisymmetric parts,
  alignment τ, and the signal-error part of each;
- observed RMSE, training-set error and generalization gap;
- N1 learning, for E, U and M only.

**Schedule.** E, U and M are evaluated at initialization, after each stage
and every 50 cycles. S and S′ are evaluated at initialization, every 250
cycles and at the end, for each start.

New measures:

- **Training-settle convergence (E and U), at every update:** the maximum
  over replayed traces of the distance between the 32-step clamped settle
  and the exact clamped equilibrium, and the gradient discrepancy
  r = ‖g − g*‖ / max(‖g*‖, 1e-8), where g uses the 32-step settle and g* the
  exact equilibrium, both before weight decay. The floor in the denominator
  is the near-zero policy: when the exact gradient is below 1e-8, r measures
  absolute discrepancy scaled by 1e8. Per-run maxima and the update where
  each occurs are recorded.
- **Hidden-code shrinkage ρ:** for each hidden module h and test example,
  compare the inner code c_h settled with h masked against c_h settled with
  every module observed. ρ is the regression coefficient
  Σ⟨c_masked, c_full⟩ / Σ‖c_full‖², pooled over modules and examples; the
  residual fraction ‖c_masked − ρ c_full‖² / ‖c_full‖² is reported beside
  it. ρ < 1 means the masked code shrinks.
- **Objective values:** each arm's own training objective, and every arm's
  value on the S objective, so arms can be compared on one scale.
- Clip activations, and whether S′'s final parameters lie inside S's clip
  bounds (W and V within ±2, T within ±3).

## Criteria (confirmatory seeds 33-42, declared before execution)

All criteria use the weak regime and final exact-equilibrium values, and
are evaluated at each budget separately. Decision rules are correlated
nuisance v1's:

- ratios of avoidable errors use (X₁ + δ) / (X₂ + δ), δ = 1e-5;
- **comparable:** larger ratio ≤ 1.25;
- **clearly better / worse:** ratio ≥ 1.5 with the ordering in ≥ 9/10
  seeds;
- otherwise **unresolved**;
- **filters:** γ ≤ 0.2 in ≥ 9/10 seeds; **transmits:** γ ≥ 0.5 in ≥ 9/10
  seeds.

**N1 is reported, not used as a gate here.** Learning (N1) for E, U and M is
reported beside every diagnostic outcome. It gates only claims that a
learning protocol succeeds. A diagnostic comparison with a failing arm
still counts: for example, S clearly better than a failing E shows that
improvement is achievable.

Every positive outcome below is constructive: it exhibits parameters or a
change that does better. Every other outcome is "no improvement found" or
"no difference established", and leaves its cause unresolved.

- **D0 control.** Report whether E filters and whether U transmits, as in
  correlated nuisance v1, and E against U on X_dec. Descriptive.
- **D1 capacity fits (S).**
  - **D1a, S against E on X_dec.** **These parameters can do better than the
    learner** if E is clearly worse than S. Otherwise **no improvement
    found**.
  - **D1b, S against U on X_dec, with S filtering.** **The inner
    architecture can beat surface coupling under decorrelation** if U is
    clearly worse than S. Otherwise **no such solution found**, with the
    ordering reported. If S does not filter, D1b reaches no conclusion and
    S's γ is reported.

  Neither outcome is a capacity limit: two finite-duration starts can stall
  or miss better solutions.
- **D2 removing clips and decay (S′ against S on X_dec).**
  - **Removing clips and decay helps the fit** if S is clearly worse than
    S′. This mixes a feasibility constraint (clips) with a training
    preference (decay) and an optimization change, so it does not by itself
    show that representational capacity was restricted.
  - **Feasible witness:** if, in addition, S′'s final parameters lie inside
    S's clip bounds in ≥ 9/10 seeds, the bounds did not exclude that
    solution; S's shortfall then comes from decay or optimization, not from
    clipping.
  - Otherwise **no improvement found**. A comparable result does not show
    that clips and decay leave capacity unrestricted.
- **D3 changing the objective (M against E on X_dec).**
  - **Masked completion helps under this protocol** if E is clearly worse
    than M. This does not show that E failed to reach its own objective's
    optimum.
  - **Masked completion does worse under this protocol** if M is clearly
    worse.
  - Otherwise **no difference established**. Both objectives could prefer
    nuisance transmission or otherwise differ from the signal-only target,
    so this rules the objective neither in nor out.

  M's N3 category is reported: the masked objective rewards transmitting
  predictable nuisance, so M may filter less.
- **D4 summary.** List the constructive improvements found (D1a, D1b, D2,
  D3, D5). If none is found at a budget, the cause of the signal loss at
  that budget is unresolved.
- **D5 uncentered rank restriction.** The uncentered rank-2 comparator
  **achieves both** if it filters and U is clearly worse than it on X_dec.
  Otherwise report its γ and X_dec. Compare with the centered comparator.
- **D6 shrinkage.** Report ρ and the residual fraction for E, M, S and S′.
  Descriptive. The shrinkage hypothesis is **consistent** with the data if
  E's ρ is below 0.9 in ≥ 9/10 seeds and S's is not.
- **D7 training settles.** For E and U, over **every update** of every
  confirmatory run: settles are **exact for practical purposes** if the
  maximum settle distance is ≤ 1e-8 and r ≤ 1e-6 throughout. Otherwise
  report the largest values, the updates where they occur, and their
  frequency. The conclusion covers the measured updates of these runs only.

These are screening rules on ten seeds, not significance tests. Partial
passes are reported as such. There are no post-result extensions of arms,
cycles, grids, budgets or seeds.

**Reading.**

- A positive D1b shows that inner-architecture parameters exist that filter
  and beat surface coupling under decorrelation. That alone says nothing
  about the learner.
- Whether the learner falls short of such a solution is D1a: E clearly worse
  than S. If both hold, the learner misses an available solution, and D3
  says whether changing the objective helps.
- If no constructive improvement is found, the cause stays unresolved. A
  negative claim about the architecture would need an analytical bound,
  pursued separately where possible.

Nonlinear modules and non-quadratic energies are out of scope and planned
separately.

## Computation

Notation as in correlated nuisance v1 (S = 42 activities; 18 free
dimensions for a clamped settle and 26 for a whole-module masked solve).

- E, U: as before. The settle check adds, per update, one exact clamped
  solve of the 18-dimensional free system, O(18³) plus O(B·18²), and one
  extra energy derivative, O(BP).
- M: per cycle, three masked exact solves and three adjoint solves of the
  26-dimensional free system, O(26³) each, plus two energy derivatives per
  mask. This is the non-local cost the local rule avoids.
- S, S′: the same per-cycle cost as M, for 3,000 cycles on 32 traces, from
  two starts.
- Comparators: one ridge solve and one 8-column SVD per fit.

Executed totals per confirmatory seed: E, U, M at 600 cycles each, and
S, S′ at 2 × 3,000 cycles each, for each of 2 regimes × 2 budgets. Pilot:
(E + U + M) × 4 settings × 5 η × 3 seeds = 180 trajectories of 600 cycles,
plus (S + S′) × 4 × 5 × 3 = 120 fits of 3,000 cycles. Report executed
counts, CPU per arm, and quality against CPU, following
[computation reporting](computation-reporting.md).

Storage per learner, float64 throughout (P = 128 for inner, 288 for
uniform):

| Item | Bytes |
| --- | ---: |
| Parameters | 8P (1,024 inner; 2,304 uniform) |
| Adam moments and step counter | 16P + 8 |
| Replay or training traces, 32 × 24 | 6,144 |
| Dense Q, 42 × 42 | 14,112 |
| Masked free block, 26 × 26 (M, S, S′) | 5,408 |
| Equilibrium and adjoint states, 2 × 32 × 26 (M, S, S′) | 13,312 |
| Clamped settle workspace, 18 × 18 plus 32 × 18 states (E, U) | 7,200 |
| Second start's parameters and moments (S, S′) | 24P + 8 |

Process peak RSS is reported separately and will be dominated by reporting
structures.

Outputs:

- a deterministic scientific JSON, compressed with `gzip -n -9` before its
  first commit, with its uncompressed SHA-256 recorded;
- the same for any per-cycle movement file;
- a separate runtime JSON with hashes, revision, dirty status, versions,
  thread settings, per-phase wall and CPU (selection charged to a named
  phase), peak RSS, and counts of settles, exact and adjoint solves, and
  clip activations.

## Verification

- **Reproduction:** on seed 23, weak / inner / b = 10 under arm E at
  η = 0.1 reproduces correlated nuisance v1's final X_in and X_dec to 1e-12.
- The adjoint gradient of the masked loss, with observed and with μ_sig
  targets, matches central finite differences of the exact loss to 1e-6 on
  random parameters.
- With μ_sig targets replaced by observed targets, the S gradient equals
  the M gradient exactly.
- S′ applies no clip and no decay; S applies the same clips and decay as E.
- The uncentered comparator's slope has rank ≤ 2, and when the targets are
  an exact rank-2 linear function of the inputs it equals no-intercept ridge
  to 1e-10.
- Shrinkage: ρ = 1 and residual 0 when the masked and full codes are equal
  (checked on a constructed case).
- Every check carried over from correlated nuisance v1 that applies.

Out of scope: the strong regime, outer and outer-matched families, learned
or gated coupling, a free-running phase, designated module roles, other
energies and nonlinear modules.
