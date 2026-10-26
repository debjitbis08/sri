# Clamped-energy ablation v1: translations, alignment and coupling strength

Prospective contract, 2026-10-26. Frozen before execution. Follows
[local contrastive consolidation v1](local-contrastive-consolidation-v1-results.md).
In that study, the clamped-energy rule with learnable inter-module translations
learned the task (combined RMSE 0.092 against 0.366 frozen) and beat the
two-phase rule. [Local consolidation v1](local-consolidation-v1-results.md)
used a related positive-phase signal and barely learned. Several things changed
together between those studies:
- learned translations at all three layers instead of fixed identity coupling;
- coupling budget 2 instead of 0.4;
- 300 Adam cycles per stage instead of 6 SGD proposals per module;
- simultaneous rather than sequential module updates, with no proposal check.

This study holds the rule, optimizer, update count and replay fixed and varies
only the coupling: whether translations are fixed, learned as rotations or
learned freely, how strong the coupling is, and which layers it connects. It
asks whether learning the relationships between modules is the decisive
ingredient, and whether isometric alignment suffices or non-isometric
translations help.

## Design provenance and contamination

No new probes informed this design. Everything it knows comes from the two
studies above. Seeds 0-2 were seen during design of the earlier study and serve
again as the pilot. Seeds 3-12 are no longer fresh: the uniform / learned /
budget-2 condition below is local contrastive v1's arm 2a, whose results on
those seeds are published. Confirmatory claims therefore use new seeds 13-22.
Seed 3 is used once more, only as the reproduction check in Verification.

## Prior work

Reviewed under [literature review for contracts](literature-review.md); the
shared survey and sources are in [prior work](prior-work.md).

Evidence levels: **[full]** read in full, **[abstract]** abstract, proceedings
page or summary only, **[derived]** derived here for this model, **[analogy]** a
structural resemblance, not an established result about this model.

- **Related methods.**
  - The clamped-energy update has the same local form as the predictive coding
    learning rule (error × activity at the settled state) [abstract]. That is a
    similarity of update equations. It is not an equivalence of statistical
    estimators.
  - The positive-definite quadratic energy defines a Gaussian over activities,
    and exact masked completion is that Gaussian's conditional mean [derived].
    Clamped-energy learning is not maximum-likelihood factor analysis or EM.
    The normalized joint negative log-likelihood is E − ½ log det Q, up to
    constants, and Q depends on the parameters; marginal-likelihood learning
    adds further latent-variable terms [derived]. Probabilistic CCA and factor
    analysis are relevant comparators, not equivalents of this training
    procedure. The fixture is not jointly Gaussian either: it has discrete
    structure identities and uniform amplitudes.
  - Fixed versus learned translations resembles coupled versus semi-coupled
    dictionary learning, which compares identical codes with a learned linear
    map between codes [abstract; analogy]. Rotation versus free translation
    resembles Procrustes versus affine alignment in model stitching [abstract;
    analogy]. Those methods use non-local optimization and two domains; this
    study asks the questions for local learning in a role-free chain coupled at
    several levels.
- **What theory already predicts.**
  - *Gauge redundancy, unconstrained [derived].* With unconstrained learnable W
    and V and isotropic priors, an orthogonal translation at the middle or inner
    layer is a reparameterization. Rotate a module's h or c and absorb the
    rotation into W and V, and the energy is unchanged; along a chain this holds
    for every edge. Only the outer layer, where x is observed, makes an
    orthogonal translation non-redundant. Non-orthogonal T is not redundant.
  - *Clipping breaks exact equivalence [derived].* Entrywise clipping of W and V
    to ±2 is not rotation-invariant: a row (2, 2) rotates to (2√2, 0), outside the
    box. Under the implemented constraints, F and O are therefore not
    automatically identical feasible model classes. Their unconstrained energies
    are equivalent, and observed differences may come from optimization, from
    clipping, or both. Clip activations are reported per group.
  - *Noise floor [derived].* Nuisance is independent across modules and
    orthogonal to the pattern columns, so the noiseless value μ of a hidden
    module is its projection onto its own pattern span, and an oracle knowing
    the maps can recover it exactly. Local contrastive v1's arms sit close to the
    floor (RMSE about 0.086), and raw RMSE ratios understate differences in
    avoidable error ([post-hoc floor analysis](local-contrastive-consolidation-v1-floor-analysis.md)).
    Criteria are therefore declared on avoidable error ‖ŷ − μ‖².
- **Context, not prediction.**
  - Semi-coupled dictionary learning was introduced because forcing identical
    codes across domains was limiting [abstract]. GLOM implementations report
    difficulty forming agreement between columns [abstract]. Neither directly
    predicts the F/O/L ranking here. The project's hypothesis, to be tested, is
    that F is weakest where codes are private, at the outer layer.
- **Alternatives not used.** A factor-analysis or probabilistic-CCA fit is a
  standard non-local estimator for multi-view linear data. It is not run here:
  the oracle mean bounds achievable error, and ridge remains the non-local
  comparator. Bidirectional predictive coding bears on the objective question,
  not this one, and is out of scope.
- **What this adds.** Mostly an internal control: which change rescued
  learning in this project's architecture. These searches found no precedent for
  the local, multi-level, role-free form of the translation question, which is
  not a novelty claim.

## Architecture and energy

Unchanged from local contrastive v1. Three modules, each 8 outer → 4 middle → 2
inner continuous activities (S=42), in a role-independent chain 0 ↔ 1 ↔ 2.
Initialize W and V from v1's `initialize(seed)` (SeedSequence [seed,1001]).

    E = Σ_m [0.5||x_m - W_m h_m||² + 0.5||h_m - V_m c_m||² + 0.025(||h_m||² + ||c_m||²)]
      + Σ_edges (a,a+1) Σ_layers ℓ  0.5 κ_ℓ ||s_{a+1,ℓ} - T_{a,ℓ} s_{a,ℓ}||²

Inference: 32 implicit Euler steps of size 1 from zero free state, observed outer
coordinates clamped, as before. There is no outer-only arm and no two-phase rule.

## Factors

A full factorial over three factors gives 18 learning conditions.

**Topology family** (per-unit strengths in the ratio shown, scaled to the
budget):

- **uniform**: κ_outer = κ_middle = κ_inner = b/14.
- **inner**: κ_inner = b/2, no outer or middle coupling. At b = 0.4 this is
  v1's coupling (κ_inner = 0.2), and with fixed translations the energy equals
  v1's energy.

**Per-edge budget** b = 8κ_outer + 4κ_middle + 2κ_inner ∈ {0.4, 2, 10}: v1's
budget, local contrastive v1's budget, and a stronger one five times larger.

**Translation mode:**

- **F, fixed identity.** T = I throughout and is not a parameter. Only W and V
  learn.
- **O, learned rotation.** T is initialized to I, takes the ordinary Adam step,
  and is then projected to the nearest orthogonal matrix (polar factor U Vᵀ
  from the SVD of T). Its singular values stay 1, so effective coupling
  curvature stays κI on both sides of every edge. O can learn alignment but not
  any non-isometric scaling. The projection reads only T, so it is local to the edge. Orthogonal
  includes reflections; the polar factor keeps whichever determinant sign the
  stepped matrix has.
- **L, learned freely.** As in local contrastive v1: T initialized to I, clip
  entries to [-3,3], no weight decay. Effective source-side curvature κTᵀT is
  learned too, so L can learn alignment, overall strength and anisotropic
  scaling together.

A fixed identity is not just uninformed at the outer layer. Each module's outer
surface comes from its own random private map, so identity coupling there pulls
a hidden surface toward its neighbor's unrelated coordinates. Middle and inner
identity coupling can be made correct by learning W and V; outer identity
coupling cannot. In the uniform family, F therefore combines "no alignment
learning" with "a harmful fixed outer prior". The inner family has no outer
coupling and isolates the first effect. Read F results by family.

Because of the unconstrained gauge redundancy (see Prior work), F and O would
differ in what they can represent only at the outer layer if W and V were
unclipped. In the inner family, F versus O therefore mainly tests optimization
and clipping. O versus L tests the benefit of non-isometric translations. That
includes overall strength but also anisotropic scaling, for example suppressing
nuisance directions, so it does not isolate scalar strength learning.

The condition uniform / L / 2 is local contrastive v1's arm 2a. The condition
inner / F / 0.4 has v1's coupling and energy under this study's rule,
optimizer, update count and simultaneous updates.

## Data, protocol and learning rule

Data: v1's `fixture(seed)` unchanged. Stage A presents structure 0, stage B
structure 1. Waking leaves long-term parameters byte-identical. Replay: the 16 A
traces in stage A, all 32 traces in stage B. There are no B-only arms.

Rule: clamped energy, as in local contrastive v1 arm 2a. Each cycle settles
every replayed trace with all three outer surfaces clamped and passes
g = ∂E/∂θ at the settled state, averaged over traces, to Adam. Every block of g
reads only its own module's layers or the two layers its translation connects.

Optimizer: Adam (β₁=0.9, β₂=0.999, ε=1e-8), moments and step counter
persistent across stages. Weight decay 1e-3 on W and V, clips ±2 on W and V.
C = 300 cycles per stage, all learnable parameters updated simultaneously.
Every condition has the same update count and the same settle count. Δθ means
the actual parameter change after projection and clipping.

Frozen baselines: one per distinct coupling with T = I (6 couplings: two
families × three budgets). Frozen performance depends on coupling, so each
condition is also compared with its own frozen baseline.

Ridge comparator: v1's `direct_predictor`, ridge 0.1, on all waking traces at
each stage, as before.

## Pilot selection (seeds 0-2 only)

Select η ∈ {0.003, 0.01, 0.03, 0.1} separately for each of the 18 conditions,
on seeds 0-2. Train on even-index waking traces of each stage, score mean
whole-module masked MSE per coordinate on odd-index waking traces after stage B,
averaged over seeds. Pick the minimum, with ties going to the smaller η. A
condition's η is disqualified if any value is non-finite. Pilot test sets are
never scored. If the selected η lies at a grid edge, report it as a limitation
and do not extend the grid. The full grid for every condition is reported, so
sensitivity to η is visible. All pilot runs are charged to a named selection
phase. Freeze all values before any confirmatory run.

Selecting η per condition reduces the learning-rate confound. It does not
remove it: the grid is coarse, selection uses three seeds and a pilot score, and
η interacts with Adam's other settings. Differences between conditions can still
partly reflect learning-rate effects.

## Evaluations

At initialization, after stage A, after stage B and every 50 cycles (evaluation
only):

- Whole-module masked RMSE on held-out test variations, hidden module rotated
  over all three, by structure and combined. 32-step inference is primary,
  exact-equilibrium solve secondary.
- **Avoidable error.** Evaluator-only. For each hidden module, μ is the
  projection of its observation onto its own private pattern span. That is the
  noiseless conditional mean, since nuisance is orthogonal to that span.
  Avoidable MSE is mean ‖ŷ − μ‖² per coordinate, by structure and combined,
  for 32-step and exact inference. Also reported:
  - the sample floor, the mean nuisance ‖y − μ‖²;
  - noise-subtracted MSE (observed minus floor), which can be near zero or
    negative on a finite sample;
  - the cross term −2(ŷ − μ)ᵀ(y − μ).
- **Training-set error**: the same masked RMSE on the replayed traces, and the
  generalization gap (test minus training). Local contrastive v1 did not record
  this.
- Single-coordinate completion RMSE (secondary).
- Translation diagnostics as in local contrastive v1: outer-layer relative error
  and principal angles against the private pattern subspace; functional
  alignment residuals at middle and inner layers; singular values of every T and
  eigenvalues of κTᵀT. For F conditions these are computed on T = I.
- Parameter movement per group: per-cycle update norm, displacement from
  initialization and path length.
- Free-phase convergence: final-step update norm and distance to exact
  equilibrium at stage ends.
- v1's abstraction metrics, descriptive only.

## Criteria (confirmatory seeds 13-22, declared before execution)

Each criterion is evaluated separately in each topology family. **X** means a
condition's mean final combined **avoidable MSE** with 32-step inference.
Observed RMSE and noise-subtracted MSE are reported alongside but are not
criteria. "Seed count" means the number of seeds in which the stated ordering
holds for per-seed avoidable MSE.

Every ratio of avoidable errors is computed as (X₁ + δ) / (X₂ + δ), with
δ = 1e-5. This is the near-zero policy: when both errors are far below δ,
the ratio tends to 1 and the conditions count as comparable instead of
producing an undefined or unstable ratio. For reference, local contrastive
v1's arm 2a had avoidable MSE 0.00101 to 0.00122, about 100δ.

- **Comparable:** the larger ratio is <= 1.25.
- **Clearly worse:** the ratio is >= 1.5 and the condition is worse in
  >= 9/10 seeds.
- Otherwise the comparison is unresolved.

- **A1 learning in every condition.** For each condition, report:
  - whether B's avoidable error falls during stage B in >= 9/10 seeds;
  - whether A's after-B/after-A avoidable-error ratio is <= 1.5 in >= 9/10
    seeds;
  - the ratio of final X to the condition's own frozen X.

  A condition **learns** if all three hold with the last ratio <= 0.1.
- **A2 learned translations (b = 2).** Compare F with L.
  - **Decisive** if F is clearly worse than L.
  - **Not decisive** if they are comparable.
  - Otherwise unresolved.

  In the uniform family, a decisive result does not separate alignment learning
  from removing the harmful outer identity prior; the inner family does.
- **A3 alignment versus strength (b = 2).** Compare O with L.
  - **Isometric translations suffice** if O and L both learn (A1) and are
    comparable. Comparable but failing performance supports no conclusion.
  - **Non-isometric translations help** if O is clearly worse than L. That may
    reflect overall strength, anisotropic scaling or nuisance suppression; A3
    does not separate them. Singular values of learned T are reported as
    descriptive evidence.
  - Otherwise unresolved. Also report O against F. In the inner family that
    comparison reflects optimization and clipping, not representational
    capacity in the unconstrained model.
- **A4 coupling strength.** For each family and translation mode, compare
  b = 0.4 and b = 10 with b = 2. **Weak coupling limits learning** if b = 0.4
  is clearly worse than b = 2; the same for strong coupling with b = 10. Otherwise report the ratios without a
  conclusion. Effective strength in L conditions is learned, so A4 concerns the
  declared budget, not the final effective coupling.
- **A5 bridge to v1.** inner / F / 0.4 has v1's coupling.
  - If it learns (A1), v1's coupling **can** support learning under this
    protocol: this rule, Adam and this update budget. That does not show the
    coupling played no role in v1's failure. Interactions between coupling and
    v1's other features (proposal signal, SGD, small budget, sequential updates,
    proposal check) remain possible and are not separated here.
  - If it does not learn but inner / L / 0.4 or inner / F / 2 does, then under
    this protocol v1's coupling is limiting.
- **R1 comparator.** Report every condition's X against ridge's; comparable
  means a match. Descriptive only. Ridge is not a ceiling. The oracle mean
  gives the bound, X = 0.

The generalization gap, translation diagnostics and effective coupling are
reported for every condition. They are explanatory, not criteria. Many
comparisons are made on ten seeds; the thresholds and seed counts are screening
rules, not significance tests. Partial passes are reported as such. There are no
post-result extensions of cycles, grids, budgets or seeds.

## Computation

Notation: S=42, free dimension 18 per clamped settle, B <= 32 traces, P
parameters. W and V give 120. Translations add, per family and mode:

| Family | F | O and L |
| --- | ---: | ---: |
| uniform | 0 (P=120) | 168 (P=288) |
| inner | 0 (P=120) | 8 (P=128) |

- Q construction: O(S²) per cycle.
- Clamped settle: one factorization of the 18-dimensional free system, O(18³),
  plus 32 batched updates at O(B·18²) each, per cycle.
- Energy derivative: O(B P) per cycle. Adam: O(P).
- Orthogonal projection (O only): one SVD per translation per cycle, O(d³) for
  d ∈ {8, 4, 2}; per cycle 2(512 + 64 + 8) for uniform, 2·8 for inner.
- Evaluation: whole-module masked inference, three masks, on test and training
  sets every 50 cycles; exact solves at stage ends.
- Ridge: O(B F² + F³), F=17, 3 fits per stage.
- Avoidable error: one 8×8 projection per module and example, O(n·64), at each
  evaluation.
- Executed totals: 18 conditions × 10 seeds × 600 cycles, plus 6 frozen
  baselines and ridge per seed, plus a pilot of 18 × 4 × 3 = 216 trajectories.
  Report executed counts, not these estimates.

Storage per learner (float64): parameters 8P bytes (960 for F, 2,304 for
uniform O and L); Adam moments 2×8P plus a step counter; replay 32×24×8 =
6,144; episode IDs 256. Workspace: activity batch 32×42×8 = 10,752, dense Q
14,112, one implicit matrix. Report process peak RSS separately.

Follow [computation reporting](computation-reporting.md):

- Deterministic scientific JSON
  (`reports/clamped-energy-ablation-v1-results.json`) and a separate runtime
  JSON with hashes, revision, dirty status, versions, BLAS/threads set to 1
  before importing NumPy, per-phase wall/CPU time, peak RSS, and counts of
  settles, factorizations, SVD projections, clip activations, cycles and
  replay exposures.
- Charge pilot selection to a named selection phase.
- Report training CPU per condition, and quality against measured CPU.
- Per-cycle movement goes to a separate deterministic file, compressed after
  the run with `gzip -n -9` if it is large, with its uncompressed SHA-256
  recorded in the results.

The earlier studies remain unchanged. Comparisons with their numbers name the
seeds involved; v1's proposal-based learner is not rerun here.

## Verification

- **Reproduction:** on seed 3, uniform / L / 2 at η = 0.03 reproduces local
  contrastive v1 arm 2a's final per-structure RMSE from that study's results
  JSON to within 1e-12.
- inner / F / 0.4 builds the same Q as v1's `build` for random W and V.
- F conditions leave T byte-identical through training.
- O conditions: ||TᵀT − I|| < 1e-10 after every projection.
- Finite differences for activity and parameter energy gradients, including T.
- Exact clamps and implicit descent in settles.
- Waking and evaluation leave parameters byte-identical.
- Identical W, V initialization across all conditions per seed.
- Avoidable error: μ equals the noiseless generator value (amplitude × pattern
  column) recomputed from the fixture's private draws, to 1e-12, on one seed;
  observed − floor − avoidable equals the cross term to 1e-12.

Out of scope: the two-phase rule and other objectives, sequential updates and
v1's proposal check, SGD or other optimizers, update-count variation, B-only
replay and inertia, learned or directed coupling strengths, and harder
fixtures.
