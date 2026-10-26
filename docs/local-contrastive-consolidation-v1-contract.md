# Local contrastive consolidation v1: two-phase learning in graded three-module chains

Prospective contract, 2026-10-26. Frozen before execution. Follows the negative
[local consolidation v1](local-consolidation-v1-results.md) result and its audit.
That study's proposals came from fully clamped reconstruction residuals, so each
module learned a local autoencoder. The predictive-support check could veto a
proposal but never steer it toward cross-module prediction; inter-module coupling
was fixed identity between independently random codes; and in a convex quadratic
energy the stability gate stayed near 1. This study replaces the proposal signal
with a two-phase local rule whose update direction is masked cross-module
prediction. It also learns inter-module translations at graded layers.

## Design provenance and contamination

Two exploratory probes informed this design:
[`local_contrastive_probe.py`](../experiments/local_contrastive_probe.py) and
[`local_contrastive_probe_graded.py`](../experiments/local_contrastive_probe_graded.py).
They write no artifacts and have no cost receipts. They used the v1 fixture, seeds 0-2, the v1 initial weights, exact
equilibrium solves, Adam (lr 0.02, 300 steps per stage) and scored on the v1
held-out test sets. Mean whole-module masked RMSE, stage A then A+B replay:

| Probe configuration | Combined | A | B |
| --- | ---: | ---: | ---: |
| Frozen (v1 coupling) | 0.376 | 0.360 | 0.391 |
| Two-phase, inner-only identity coupling 0.2, fixed | 0.134 | 0.128 | 0.140 |
| Two-phase, inner-only 0.2, learned translation | 0.124 | 0.120 | 0.128 |
| Two-phase, inner-only, budget 2, learned translation | 0.103 | 0.102 | 0.104 |
| Two-phase, uniform per-unit, budget 2, learned translations | 0.090 | 0.089 | 0.090 |
| Two-phase, inward graded, budget 2, learned translations | 0.091 | 0.089 | 0.092 |
| Same inner-only 0.2 rule, stage B replays B only | 0.444 | 0.615 | 0.117 |
| Direct ridge comparator | 0.0955 | — | — |

At initialization, the two-phase estimate matched the exact gradient of the masked
loss with cosine >= 0.9999 for every parameter group. These probe numbers are
design evidence, not results. Seeds 0-2 and their test sets were therefore seen
during design. They serve only as the pilot below, and confirmatory claims use
fresh seeds 3-12. The probe settings (exact solves, step counts, learning rate) are not
the confirmatory settings, so its numbers are not targets this run must
reproduce.

## Architecture

Three modules, each 8 outer → 4 middle → 2 inner continuous activities (S=42),
in a chain 0 ↔ 1 ↔ 2 with no designated input or output during learning. This
keeps the role-independent v1 chain rather than the graded three-module study's
input/intermediate/output roles, because masking rotates over all three modules.
Initialize W (3×8×4) and V (3×4×2) from v1's `initialize(seed)`, using
SeedSequence [seed,1001].

Energy per example:

    E = Σ_m [0.5||x_m - W_m h_m||² + 0.5||h_m - V_m c_m||² + 0.025(||h_m||² + ||c_m||²)]
      + Σ_edges (a,a+1) Σ_layers ℓ  0.5 κ_ℓ ||s_{a+1,ℓ} - T_{a,ℓ} s_{a,ℓ}||²

where s_{m,ℓ} is module m's outer (8), middle (4) or inner (2) layer. The learned
translations T_{a,ℓ} are 8×8, 4×4 and 2×2, initialized to identity. Identity
carries no alignment knowledge here because each module's private maps and
encoders are random. Couplings are symmetric: every direction has feedback gain
1, so all dynamics descend this one energy. Q stays positive definite for any
W, V, T, because coupling terms are PSD and the 0.05 diagonal covers middle and
inner units.

The graded study found lower error with weak or absent output feedback. Its
directed gains, however, give non-gradient dynamics, which breaks the two-phase
rule's correspondence with a loss gradient. Every module is also sometimes the
hidden one, so no feedback direction is privileged. Directed gains and learned
strengths are out of scope.

Coupling families. Per-unit strengths are fixed, with per-edge budget
8κ_outer + 4κ_middle + 2κ_inner = 2, as in the graded study's main comparison:

- **G-uniform** (main): κ = (1/7, 1/7, 1/7).
- **G-inner**: κ = (0, 0, 1). Bridges to v1's topology at equal budget.
- **G-no-outer**: κ = (0, 1/3, 1/3). Removes outer connections. This also
  redistributes strength to the remaining layers and removes 128 translation
  parameters, so it is not a pure shortcut test.
- **Outer-only, no hierarchy**: each module is only its 8 outer units, with no
  W, V, middle or inner units. Adjacent surfaces couple through learned 8×8
  translations at κ_outer = 1/4 (budget 2). A small 0.025||x||² term on outer
  units keeps a hidden end module positive definite when T is rank-deficient;
  this term is used in this arm only. It is the shortcut predictor: a learned
  outer translation chain with the hierarchy removed. The term adds 0.05I to the
  activity Hessian and, with β=0.01, keeps a positive curvature margin for the
  negative nudge. It is not only numerical conditioning: it changes
  predictions. For a hidden receiving end module with one neighbor and fixed T,
  equilibrium is y = 0.25/(0.25+0.05) Tx = (5/6) Tx. Learning T can compensate,
  but arm 5b differs from arm 3 in regularization and optimization as well as
  topology.

Inward-graded coupling is dropped; it matched uniform in the probe.

Fixed κ does not fix effective strength. For an edge term κ||y − Tx||²/2 the
curvature is κI on the target side and κTᵀT on the source side, so learning T
also learns effective coupling strength. This study therefore does not isolate
alignment from strength learning. Translation singular values and effective
coupling curvature are recorded (see Evaluations).

Inference: identical to v1, 32 implicit Euler steps of size 1 from zero free state
with observed outer coordinates clamped, solving (I + Q_ff) once per call. No early
stopping. There is no slow-inner variant and no stability gate.

## Data and protocol

Use v1's `fixture(seed)` unchanged: per-seed random orthonormal 8×2 maps per
module, amplitude U[0.8,1.2], perpendicular nuisance with standard deviation 0.1,
16 waking variations per structure, 12 test variations per structure and 8
evaluator-only reference variations per structure. Stage A presents structure 0,
stage B presents structure 1. Learners receive only observed 3×8 activities and
episode IDs. Waking leaves long-term parameters byte-identical. There is no
accept/reject step, so v1's proposal/check split is unnecessary: all available
traces are used for learning. Explicit replay is bounded at 32 traces.

Each sleep phase has C=300 cycles. Each cycle averages one full-batch update over
the three whole-module masks m ∈ {0,1,2}. All parameters update simultaneously
(v1 updated modules sequentially). The optimizer is Adam with learning rate η,
β₁=0.9, β₂=0.999 and ε=1e-8, applied per parameter. Its first and second
moments and step counter are learned history: they persist across stages, are
never reset, and are counted as persistent state. Weight decay 1e-3 is added to
the W and V gradients before Adam; T is not decayed. Clip W
and V entries to [-2,2] and T entries to [-3,3]. Report clip activations.
Wherever this contract writes Δθ, it means the actual parameter change after
any inertia scaling and clipping.

Replay: in stage A, the 16 A traces. In stage B, all 32 traces (A+B), except the
B-only arms, which replay only the 16 B traces. Assert that B-only arms never
touch A traces in stage B.

## Two-phase local rule

For mask m, the free phase s⁰ settles with module m's outer surface hidden and
the other two clamped. Two nudged phases start from s⁰ and run 32 more steps with
energy E ± (β/2)||x_m - x̂_m||². Here x̂_m is the replayed trace's actually
observed value; it enters only the nudged phases. The update is

    g = [∂E/∂θ(s^{+β}) - ∂E/∂θ(s^{-β})] / (2β),   β = 0.01,

and g, averaged over the three masks, is the gradient passed to Adam.

The symmetric difference cancels the unnudged continuation shared by the two
nudged phases. It does not remove finite-settling bias: when s⁰ is not an exact
equilibrium, g can still differ from the equilibrium gradient in direction and
magnitude. Fidelity is therefore measured, not assumed (see Evaluations and M1).

Locality boundary. ∂E/∂W_m and ∂E/∂V_m read only module m's activities, and
∂E/∂T_{a,ℓ} reads only the two connected layers. The parameter-update signals
are therefore block-local, and rotating masks avoids permanent input/output
roles. The target is a self-supervised observed value arriving after the free
prediction; no label, global loss differentiation or input/output role enters
the update. The implementation, however, still settles activities with global
matrix solves and coordinates the phases centrally. This study tests local
learning signals, not fully distributed computation.

Before each −β phase, check that λ_min(I + Q_ff − βP_m) > 0 and λ_min(Q_ff − βP_m) > 0,
where P_m projects onto the hidden outer units. If either check fails, skip that
mask's contribution for the cycle and count the skip. Never alter β.

## Arms

Each arm runs on confirmatory seeds 3-12:

1. **Frozen**, G-uniform parameters at initialization.
2. **Clamped energy, G-uniform**: v1-style positive-phase rule. Adam receives
   g = ∂E/∂θ at fully clamped settled states, applied to W, V and T. Two separate
   trajectories from the same initialization:
   - **2a**, C cycles per stage (C + C): matched parameter-update count.
   - **2b**, 9C cycles per stage (9C + 9C): matched settle count (9 settles per
     cycle in the two-phase arms, 1 here).

   Neither matches computation. A clamped settle has 18 free activities (middle
   and inner), a two-phase settle has 26 (plus the hidden outer surface), and
   solver costs differ. Measured wall/CPU time and operation counts are reported
   separately from the matched-count comparisons.
3. **Two-phase, G-uniform** (main).
4. **Two-phase, G-inner**.
5. **Two-phase, G-no-outer**.
5b. **Two-phase, outer-only, no hierarchy**: shortcut predictor.
6. **Exact-gradient reference, G-uniform**: exact equilibrium solve plus adjoint
   gradient of the exact masked loss 0.5||x_m* - x̂_m||², same Adam settings and
   cycles. This is non-local (it differentiates the exact global solution) and
   is a reference, not a candidate. It is not a guaranteed ceiling: it optimizes
   exact-equilibrium predictions, while the primary evaluation scores 32-step
   predictions, and Adam on a nonconvex parameter problem need not reach the
   best attainable solution.
7. **Two-phase, G-uniform, B-only replay in stage B**: forgetting control.
8. **Same plus importance inertia**: per-parameter path integral during learning,
   ω_i += max(0, -g_i Δθ_i), where g is the two-phase estimate. At each stage end,
   Ω_i += ω_i / ((θ_i^end - θ_i^start)² + 1e-3), then reset ω. Later Adam updates
   for parameter i are scaled by 1/(1 + λΩ_i) after Adam's normalization;
   scaling the gradient would be normalized away. Δθ_i in ω is the actual change
   after scaling and clipping. All quantities are per-parameter local. The scale
   reduces plasticity; it is not a quadratic anchor penalty. Ω measures
   accumulated estimated learning contribution. It is not independent evidence
   that a relationship stays useful across diverse experiences.
9. **Same B-only replay with matched uniform slowing**: one scalar factor on
   the post-normalization Adam update,
   equal to the mean over parameters of arm 8's 1/(1 + λΩ_i), computed per seed
   at the start of stage B. It runs after arm 8 by declared deterministic
   dependence. It is only an approximate control: matching the mean scale does
   not match actual movement when the per-parameter scales correlate with Adam's
   update sizes. For arms 7-9, record applied update norms per cycle and
   cumulative parameter displacement per parameter group.
10. **Direct ridge comparator**: v1's `direct_predictor`, ridge 0.1, on all waking
    traces available at each stage.

## Pilot selection (seeds 0-2 only)

The pilot selects η separately for each rule family: two-phase (shared by arms
3-5b and 7-9), clamped energy (shared by 2a and 2b, selected at C cycles) and
exact reference (arm 6). The grid is
η ∈ {0.003, 0.01, 0.03, 0.1}, run on the G-uniform family. The probe's 0.02,
used with exact solves, lies inside it. Pilot runs train only on
even-index waking traces of each stage. The score is mean whole-module masked
loss on the odd-index waking traces after stage B, averaged over seeds 0-2. Pick
the minimum; ties go to the smaller η. If two-phase arms diverge (non-finite
values, or failed PD checks in more than 1% of masks), η is disqualified.

With η fixed, select λ ∈ {1, 10, 100} for arm 8 under the B-only protocol, using
the same odd-trace score over both structures. Pilot test sets are never scored
during selection. If a selected value lies at a grid edge, report it as a
limitation and do not extend the grid. All pilot runs, including rejected
values, are charged as selection cost. Freeze all values before any
confirmatory seed runs.

## Evaluations

At initialization, after stage A and after stage B, plus every 50 cycles for
learning curves (evaluation only, never used for selection):

- Whole-module masked RMSE on held-out test variations, rotating the hidden module
  over all three modules, by structure and combined. Use 32-step inference as
  the primary metric and an exact-equilibrium solve as secondary, as in the v1
  audit.
- Single-coordinate completion RMSE (secondary; v1's local check criterion).
- Translation diagnostics, evaluator only. The private generator does not
  identify a unique full-dimensional translation, so score only the supported
  pattern subspace:
  - Outer layer: relative error ||T M_a − M_{a+1}||_F / ||M_{a+1}||_F and the
    principal angles between span(T M_a) and span(M_{a+1}), where M_a holds
    module a's two private pattern columns.
  - Middle and inner layers have no private reference, because the encoders are
    learned. Score functional alignment instead: the relative residual of
    T s_{a,ℓ} against s_{a+1,ℓ} on held-out fully observed inferred states.
  - For every T: singular values, plus eigenvalues of the effective coupling
    curvature κTᵀT (source side; target side is κI).
- v1's abstraction metrics: spread ratio, centroid accuracy on reference
  examples, code norm, zero-code control. Random codes already separate the two
  structures, so these are descriptive only and never success criteria.
- Free-phase convergence: final-step update norm, and distance to exact
  equilibrium at the end of each stage.
- Gradient fidelity at initialization and at each stage end, per parameter group
  (W, V, each T layer), comparing the finite-settle symmetric estimate g with
  the exact equilibrium gradient g* (arm 6's adjoint) on the same batch:
  - cosine;
  - relative error ||g − g*|| / ||g*||;
  - norm ratio ||g|| / ||g*||.

  If ||g*|| < 1e-10 for a group, mark it near-zero. Report both absolute norms
  and exclude that group from the cosine and ratio criteria, but count it.
- Parameter movement for every learning arm, per parameter group and for the
  whole parameter vector: applied update norm ||Δθ|| per cycle, cumulative
  displacement ||θ − θ_init||, and cumulative path length Σ||Δθ||. Similar
  displacement can hide very different amounts of movement. For stage B, also
  report displacement ||θ − θ_B,start|| and path length measured from the start
  of stage B.

## Criteria (confirmatory seeds 3-12, declared before execution)

Primary:

- **P1 acquisition and retention with replay.** Arm 3 lowers B RMSE from before
  to after stage B in >= 9/10 seeds, keeps A's after-B/after-A RMSE ratio <= 1.1
  in >= 9/10 seeds, and has mean final combined RMSE <= 0.5× frozen.
- **P2 the two-phase signal, not update count.** Arm 3 beats arm 2a (matched
  updates) and arm 2b (matched settles) in final combined RMSE in >= 9/10 seeds
  each. Report measured computation alongside; P2 makes no equal-compute claim.
- **P3 gap to the exact-gradient reference.** Arm 3 mean final combined RMSE
  <= 1.15× arm 6, on both 32-step and exact-equilibrium evaluation. This
  measures what the local learning signal loses against an exact gradient with
  the same global settling. It is not the cost of distributed computation.

Secondary:

- **S1 topology.** Report arms 3, 4, 5 and 5b. If arm 5 >= 1.25× arm 3, conclude
  only that outer connections help this configuration. Removing them also
  redistributes strength and removes parameters. If arm 5b <= 1.1× arm 3, the
  hierarchy is not shown to add anything beyond an outer translation chain: the
  shortcut suffices here. If arm 5b >= 1.25× arm 3, conclude only that this
  particular regularized shortcut underperforms. Regularization and
  optimization differ too, so a weak 5b does not establish that the hierarchy
  explains arm 3's advantage.
- **S2 forgetting and inertia.** Report arm 7's A ratio. Importance inertia
  succeeds if arm 8 has a lower A ratio than both arm 7 and arm 9 in >= 8/10
  seeds, while its final B RMSE is <= 1.1× arm 9's. The all-ten-seed result
  is primary.

  Movement warning flag, per seed, on the whole parameter vector during
  stage B only. Let D8, D9 be stage-B displacement from θ_B,start, and L8, L9
  stage-B path length, for arms 8 and 9. Flag the seed movement-unmatched if
  |D9 − D8| / D8 > 0.2 or |L9 − L8| / L8 > 0.2; arm 8 is the denominator. If
  D8 < 1e-6 or L8 < 1e-6, use the absolute difference instead and flag when it
  exceeds 1e-6. As a sensitivity analysis only, also report S2 on unflagged
  seeds, stating how many remain (k/10). The 20% threshold is a judgment call
  for a warning, not a matching guarantee.
- **S3 comparator.** Report arm 3 against ridge. Call it a match if mean final
  combined RMSE <= 1.1× ridge. This is not a success criterion; ridge sees
  global observations directly.
- **M1 mechanism fidelity.** At initialization, in every confirmatory seed and
  for every non-near-zero parameter group: cosine >= 0.95, relative error <= 0.3,
  and norm ratio in [0.8, 1.25]. A failure is reported as finite-settling bias
  and qualifies P1-P3. Stage-end values are reported, not criteria. These
  thresholds are judgment-based screening criteria, not theoretical guarantees.
  Report the continuous cosine, relative error and norm ratio for every group
  and checkpoint; passing M1 does not mean g equals the exact gradient.

Ten seeds support per-seed counts and means, not significance claims. Partial
passes are reported as such. There are no post-result extensions of cycles,
grids or seeds.

## Computation

Notation: K=3, S=42, outer/middle/inner widths 8/4/2, B <= 32 traces.
Parameters: P = 120 (W, V) + translations; G-uniform 2(64+16+4) = 168, so P=288;
G-inner P=128; G-no-outer P=160; outer-only P=128 with S=24.

- Q construction: O(S²) per parameter change, once per cycle.
- Two-phase cycle: 3 masks × 3 settles (free, +β, −β). Each settle needs its own
  implicit factor, O(S³), so 9 factorizations, plus 9 × 32 batched updates at
  O(B S²) each. Energy derivatives cost O(B P) per settle, and PD checks are 6
  eigenvalue computations at O(S³) each.
- Clamped energy cycle: 1 factorization of an 18-dimensional free system,
  32 updates, O(B P) derivative. Two-phase settles use 26 free dimensions.
- Exact reference cycle: 3 × (one O(S³) solve plus one adjoint solve), O(B P).
- Fidelity diagnostics: one exact gradient per checkpoint and seed, charged to a
  named diagnostic phase.
- Adam: O(P) per cycle in every learning arm.
- Inertia: O(P) per cycle.
- Ridge: O(B F² + F³), F=17, 3 fits per stage.
- Totals per seed and arm: C=300 cycles per stage, 2 stages; arm 2a runs 300
  and arm 2b runs 2700 per stage. Report executed counts, not these estimates.

Storage per learner (float64): parameters 8P bytes (2,304 for G-uniform); replay
32×24×8 = 6,144; episode IDs 256. Adam moments add 2×8P bytes (4,608 for
G-uniform) plus a step counter. Inertia arms add ω, Ω and stage-start
parameters, 3×8P = 6,912. Workspace: three
activity batches up to 3×32×42×8 = 32,256; dense Q 14,112; up to 9 implicit
matrices, reused, peak 2 × 14,112. Report process peak RSS separately.

Follow [computation reporting](computation-reporting.md):

- Deterministic scientific JSON (`reports/local-contrastive-consolidation-v1-results.json`)
  and separate runtime JSON, with hashes, revision, dirty status, versions,
  BLAS/threads set to 1 before importing NumPy, per-phase wall/CPU time, peak RSS
  and counts: settles, factorizations, PD skips, clip activations, cycles,
  replay exposures.
- Charge pilot selection (all grid values, both selections) to a named
  selection phase, separate from confirmatory training.
- Report quality versus cumulative activity-update work, and versus measured
  CPU time, for arms 2a, 2b, 3 and 6.

v1 and the frozen graded studies remain unchanged and are not equal-task learning
comparators. The exploratory probes have no cost data; state that explicitly in
the results.

## Verification

- Finite differences for activity and parameter energy gradients, including T.
- The adjoint reference gradient against finite differences of the exact masked loss.
- Symmetric two-phase estimate against the exact gradient on a small fixed batch.
- Exact clamps, and implicit descent in free and nudged phases.
- Hidden-target leakage: perturbing x̂_m by 1000 leaves the free phase unchanged.
- Waking and evaluation parameters stay byte-identical.
- B-only arms never access A traces in stage B.
- Identical initialization across arms per seed.
- PD-check skip accounting.
- Chain-reversal consistency for fixed-parameter inference when T is the
  identity. Learned T breaks exact reversal symmetry under this parameterization;
  disclose rather than test it.

Out of scope: learned coupling strengths and directed feedback gains, nonlinear
or multistable dynamics (where a settling-stability signal could carry
information), trace recombination and dreams, autonomous sleep scheduling,
contradictory-evidence revision, and harder fixtures where discrimination is
not already trivial.
