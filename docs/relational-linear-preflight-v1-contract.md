# Relational linear preflight v1: can the weak-prior SRI module represent shared action-conditioned linear dynamics?

Prospective contract, 2026-10-26. Frozen before execution. First deliverable of the
[relational abstraction proposal](relational-abstraction-learning-proposal.md):
the generator, identifiability and capacity preflight for its linear fixture.
Revised the same day after an external review and the decision to keep all
three modules.

It asks four questions, in order, and no learning question:

1. Does the generator have the properties the later study needs
   (observability, boundedness, identifiable target binding, detectable
   scrambles)?
2. Can the existing three-module chain, with banks of per-action inner
   translations, represent long-horizon action-conditioned prediction on
   this fixture?
3. With the banks frozen, does refitting a new domain's bindings leave the
   effective dynamics unchanged up to one shared basis change?
4. At which calibration budgets and horizons do the direct linear references
   predict a measurable transfer gap?

The clamped-energy learning rule is not run. Capacity searches here use
exact gradients through the exact solve, some with privileged targets. They
are not evidence about local learning.

## Design provenance and contamination

New fixture; no earlier result used these worlds. Draft checks on one
hand-built world informed the [prior-work](#prior-work) predictions below.
They are reproduced by `.venv/bin/python -m experiments.relational_linear_preflight_checks`
(world seed 1, rotation 0.6, reflection axis 0.4), with outputs in
`reports/relational-linear-preflight-v1-draft-checks.json` (SHA-256
3cfdce37…93bb71d). Preflight worlds are seeds 0-9. Seeds 1000 and above are
reserved for the later learning/transfer contract and are not generated here.

## Prior work

Reviewed under [literature review for contracts](literature-review.md); see
the [linear-dynamics entry](prior-work.md#linear-action-conditioned-dynamics-and-transfer)
of the shared survey, searched 2026-10-26. Component names:

- per-action inner translations: a switched (action-indexed) linear latent
  model, as in switched-linear identification and Embed to Control
  [abstract];
- domain bindings with frozen shared dynamics: transfer of a latent dynamics
  model across observation spaces (Sun et al. 2022) [abstract];
- decode-and-re-encode rollout: iterated window prediction, the
  equation-error setting whose compounding errors DaD addresses [abstract];
- direct references: per-action least squares, reduced-rank regression,
  and a shared-encoder latent model with latent consistency [familiar];
- scramble and drift tests: joint similarity of matrix tuples, which is
  linear in the basis change [familiar].

What follows without running the module [derived; numerically checked on the
draft world by the script above]:

- **The effective operators exist exactly.** With module 0's outer clamped
  and modules 1 and 2 free, module 0's coupled core activity is a fixed
  linear function of its isolated encoding. Both window outputs therefore
  factor through the encoder (residuals about 1e-16).
- **Per-step gain loss compounds.**
  - With T = F_a at both couplings, prior 0.001 and κ = 1, the one-step gain
    is 0.985 and the two-step window gain 0.983. After 64 windows
    (128 steps) about 0.34 is left.
  - Scalar rescaling of the two banks restores 0.99997 per window (0.998
    after 128 steps), with every |T| ≤ 0.84.
  - The [energy scaling analysis](energy-scaling-analysis.md) found that the
    clamped-energy learner keeps unit translation gain. The later learning
    study should therefore expect geometric decay unless something
    compensates.
- **One-step optimality conflicts with long-horizon accuracy**, even without
  the module's priors.
  - Write Kx = E[z | x]. The population-optimal one-step map is R F_a K.
    Composing it inserts KR between transitions, while the oracle applies K
    only once.
  - On the draft world the singular values of KR are 0.998 and 0.997, and the
    composed map keeps 0.73 of the oracle's gain at L = 128.
  - Capacity is therefore searched and accepted against the oracle at every
    horizon. One-step fitting is kept as a diagnostic of objective mismatch.
- **Nuisance rejection pulls against the weight clip.** The constructed
  encoder W = [R, βB] has relative leakage ‖(Φ₀R)⁻¹Φ₀B‖ of 0.34, 0.14, 0.07
  and 0.03 for β = 1, 2, 3 and 5. Only β ≤ 2 fits the ±2 clip. At β = 2 the
  leakage alone gives avoidable error of very roughly 0.14² × 0.25 ≈ 5e-3 at
  every horizon (a rough estimate).
- **No action back-reaction for this witness.** Module 1's prediction does
  not depend on the second action (relative change about 1e-16), because the
  translations are orthogonal and the effective prior is isotropic. This does
  not hold in general, so it is measured.
- **Target binding is identifiable up to scale** with a rotation and a
  reflection fixed (commutant dimension 1), and up to rotation and scale
  with a rotation alone (dimension 2).
- **Scramble detectability, certified lower bounds** (method in P0):
  - family A swap: 0.17, or 0.67 with the probe;
  - family B swap: 0 without the probe, which the bisector reflection
    attains exactly, and 0.78 with the probe;
  - one operator replaced by a random matrix of the same eigenvalues: 0.009,
    which is weak;
  - broken rotation (Δ = 0.3): 0.022.

Known failure modes accepted here: the priors make every equilibrium a
shrinkage estimator, and the clips bound the attainable gains and rejection.
Both are measured, not removed.

What this adds: a capacity witness, or the record of searches that found
none, for the architecture the proposal makes primary; and budgets and
horizons for the learning contract. "Nothing new beyond that" is an
acceptable answer.

## Generator

Each world draws, independently:

- **Family A** (primary): a rotation F_R by angle θ ~ U[π/8, 3π/8] and a
  reflection F_Q about an axis at angle φ ~ U[0, π).
- **Family B** (symmetry audit and capacity check): two reflections about
  axes φ₁ ~ U[0, π) and φ₂ = φ₁ + δ, δ ~ U[π/8, 3π/8].
- **Broken-structure target** (audit and P2 diagnostic only): family A
  with θ′ = θ + Δ, Δ ~ ±U[π/12, π/6], same reflection.
- **Three domains** (sources s1 and s2, target t), each with a rendering
  R_i = 2G_i/√8 and nuisance loading B_i = G′_i/√8, G entries N(0, 1),
  redrawn until every singular value of [R_i, B_i] lies in [0.3, 3].
- **Probe:** a unit vector u, giving the scalar outcome y = uᵀz.

Observations follow x = R_i z + B_i n + ε, with z_next = F_a z,
n ~ N(0, 0.25 I₂) redrawn at every time step, and ε ~ N(0, 0.05² I₈).
Initial states are z₀ ~ N(0, I₂). All operators are orthogonal, so ‖z‖ is
constant along every trajectory and no normalization is applied.

The unit of evidence is a **window**: (x_t, a_t, x_{t+1}, a_{t+1}, x_{t+2})
with y at all three times. A window contains two transitions, and budgets
are counted in transitions. Actions are uniform and independent. Per world
and domain:

- **Source** (s1, s2): 256 fitting windows (512 transitions) and 128
  selection windows. Selection windows are source data and are used only
  where stated.
- **Target** (t): 128 calibration windows. Budget n uses the first n/2.
  There is no target selection set.
- **Test:** 256 initial states. Each is paired with 8 random mixed-action
  sequences per horizon L ∈ {1, 2, 4, 8, 16, 32, 64, 128}, plus the two
  pure sequences.
- **Privileged capacity sequences** (P1 only): 256 fitting and 128
  selection initial states from domain s1. Each has 8 mixed sequences per
  horizon L ∈ {1, 2, 8, 32, 128}, with oracle targets.

Learners, binding fits and direct references never see z, F_a, R_i, B_i or
u. The privileged capacity search in P1 and the evaluator use them.

**Oracle.** x*_L = R_i F_{a_L} ··· F_{a_1} E[z₀ | x₀], where the Gaussian
conditional mean is computed exactly. Primary error measure:
X_L = Σ‖x̂_L − x*_L‖² / Σ‖x*_L‖² over the test set, which is the avoidable
error. Probe error P_L is defined the same way on y.

## Architecture

The weak-prior inner learner of
[prior-strength training](prior-strength-training-v1-contract.md), as a
two-step temporal window:

- **Three 8 → 4 → 2 modules** in the 0 ↔ 1 ↔ 2 chain, coupled only at the
  inner layer. Module 0 receives x_t, module 1 predicts x_{t+1}, and
  module 2 predicts x_{t+2}.
- **Activity prior** λ = 0.001 on both hidden layers. Weight decay is not
  applied in capacity or binding fits; the fitted weights' decay penalty is
  reported.
- **Action banks:** at each coupling, one 2 × 2 translation per action,
  T⁰¹_a and T¹²_a, with coupling terms κ/2·‖c₁ − T⁰¹_{a_t} c₀‖² and
  κ/2·‖c₂ − T¹²_{a_{t+1}} c₁‖². κ ∈ {1, 5}, matching the inner family at
  b = 2 and b = 10. The banks are an extension, and action identity is
  supplied.
- **Domain bindings:** each domain has its own (W_m, V_m) for m = 0, 1, 2.
  The banks are shared across domains.
- **Clips** as in the learner: W, V within ±2 and T within ±3. A
  relaxed-clip search reruns P1 and P2 with W, V and T within ±6.

Operational definitions:

- **Encoder** φ(x): the exact equilibrium of module 0 alone with its outer
  layer clamped to x. It does not depend on the actions. The coupled
  encoding inside the window is reported as a diagnostic, by its relative
  discrepancy from φ.
- **Window predictions** M¹_{a,b} and M²_{a,b} (each 8 × 8): the outer
  activities of modules 1 and 2 at the exact equilibrium, with module 0's
  outer clamped and actions (a, b) selected. Each is extracted by solving
  for the 8 unit vectors.
- **Effective operators** G¹_{a,b} and G²_{a,b} (each 2 × 2): the
  least-squares solutions of Φ₀M = GΦ₀. A relative residual above 1e-8
  fails the factorization check.
- **Rollout** for even L: x̂ ← M²_{a,b} x̂, repeated L/2 times over
  consecutive action pairs, starting from x̂ = x₀. Each prediction is
  re-encoded through module 0, and no intermediate observation is clamped.
  For L = 1, x̂₁ = M¹_{a,a}, where the second action is a declared filler
  equal to the first. Its back-reaction is reported as the relative change
  in M¹ when the second action is switched.
- **Latent form:** φ(x̂_L) = G² ··· G² φ(x₀). Both forms are computed, and
  their agreement is a verification check.
- **Composition consistency:** ‖G²_{a,b} − G¹_{b,·} G¹_{a,b}‖ / ‖G²_{a,b}‖,
  where G¹_{b,·} is module 1's operator for b with filler b. It is
  reported, not required.
- **Probe readout:** a linear map from φ(x) to y, fitted by least squares on
  the evidence of the fit in question. It is a declared adapter and is
  counted in the parameters.

## Stages

**P0 generator audit** (evaluator, no fitting), per world:

- **Observability:** the rank of [u, F_aᵀu, …] for each action, and for
  mixed sequences up to length 4.
- **Rendering conditioning:** the condition number of [R_i, B_i].
- **Precision:** ‖z_L‖ is constant by construction; check that float64
  keeps it to 1e-12 over 128 steps.
- **Commutant:** the dimension of each family's commutant, as the null space
  of the stacked commutation operator (SVD, relative tolerance 1e-10).
- **Scramble mismatch, by exact linear algebra:**
  - **unconstrained:** J = min over S with ‖S‖_F = 1 of
    Σ_a ‖S F′_a − F_a S‖² / Σ_a ‖F_a‖². This is the smallest squared
    singular value of the stacked Sylvester operator.
  - **with the probe:** J_u minimizes the same residual subject to uᵀS = uᵀ,
    solved exactly as an equality-constrained least-squares problem.
  - **What the minimum certifies:** it is taken over all S, singular
    included, so it is a certified lower bound for invertible alignments.
  - Report the minimizer's condition number.
  - **Classification:**
    - **detectable** if the bound is ≥ the stated threshold;
    - **equivalent** if J ≤ 1e-12 with a minimizer of condition number
      ≤ 1e6;
    - otherwise **unresolved**.
- **Scrambles:** the swap of the two operators, and 10 draws replacing each
  operator by P F_a P⁻¹. Each P has N(0, 1) entries and is redrawn until its
  condition number is ≤ 10.
- **Broken target:** J and J_u against family A.

**P1 capacity witness**, per world, family, κ and clip setting:

- **Constructive witness** (privileged):
  - W_m = [R_{s1}, βB_{s1}] for all three modules, with β the largest value
    inside the clip;
  - V_m = [I; 0];
  - T⁰¹_a = F_a/g₁ and T¹²_a = F_a/g₂, where g₁ is the measured one-step
    gain and g₂ the remaining two-step gain, as in the draft-check script.

  It is evaluated directly against the oracle on the test set. A passing
  constructive witness counts.
- **Multi-horizon search** (privileged): start from the constructive
  witness.
  - Objective: Σ_{L ∈ {1, 2, 8, 32, 128}} X_L on the privileged fitting
    sequences.
  - Optimizer: Adam, 2,000 full-batch steps, projecting onto the clips
    after each step.
  - Gradients: automatic differentiation in float64 (PyTorch) through the
    exact solves.
  - Learning rate: η ∈ {0.001, 0.003, 0.01}, selected by the same objective
    on the privileged selection sequences, with ties to the smaller η.

  This search, like the witness, is never used to initialize a learner.
- **One-step diagnostic** (observed data only): from the same start, minimize
  the window prediction loss (squared error of modules 1 and 2 against
  x_{t+1} and x_{t+2}) on s1's fitting windows. η is selected on s1's
  selection windows. Its X_L beside the multi-horizon search measures
  objective mismatch. It is not a candidate witness.
- **Reported for each of the three:**
  - X_L at every horizon, and P_L;
  - the eigenvalue moduli of every G, and their L-th powers;
  - factorization residuals, composition consistency, back-reaction and the
    coupled-encoder discrepancy;
  - the singular values, numerical rank (relative tolerance 1e-8) and
    condition number of Φ₀R;
  - the relative leakage ρ = ‖(Φ₀R)⁻¹Φ₀B‖₂, which is undefined when the rank
    is below 2;
  - the active clips;
  - for each action pair, ‖M¹ − M¹_LS‖ / ‖M¹_LS‖, where M¹_LS is the
    least-squares 8 × 8 map on the same windows.

**P2 binding invariance**, per world, family, κ and clip setting where C1
found a witness:

- **Frozen banks:** use the multi-horizon search's banks if it passed C1,
  otherwise the constructive witness's.
- **Fit new bindings** (W_m, V_m and the probe readout) for s2 on its 256
  fitting windows, and for t on all 128 calibration windows. The banks are
  never updated.
- **Objective:** the observed window prediction loss plus the squared probe
  error.
- **Optimizer:** Adam, 2,000 full-batch steps, with η fixed at P1's
  one-step-diagnostic selection, so no target selection data is used.
- **Initialization and restarts:** W entries N(0, 0.1²) and V_m = [I; 0]/2.
  Run 3 restarts and keep the lowest final training objective.
- **Drift:** J_G, the unconstrained mismatch of P0 applied jointly to all
  eight operators {G¹, G²} over the four action pairs. Also report its
  minimizer's condition number.
- **Also reported:** one-step and rollout errors, and eigenvalue moduli
  before and after.
- **Broken-target diagnostic:** repeat the fit against the broken-structure
  target and report its best training objective, X_L and J_G.

The binding fits use observed data only, so their rollouts carry P1's
objective mismatch. J_G separates drift in the dynamics from that.

**P3 inference.** On each P1 multi-horizon result, compare 32-step
implicit-Euler settling (step 1, zero start) with the exact solve. Report:

- relative activity error;
- X_2 and X_32 from settled rollouts;
- the number of steps needed to reach relative error 1e-3, capped at 1,024.

**P4 direct-reference design audit** (no SRI). The shared-encoder latent
model has encoder E (2 × 8), operators F̂_a (2 × 2), decoder D (8 × 2) and
probe readout û (2). Its objective, over every transition in the evidence,
is

    Σ ‖x_{t+1} − D F̂_{a_t} E x_t‖²   (prediction)
    + Σ ‖x_t − D E x_t‖²             (reconstruction)
    + Σ ‖E x_{t+1} − F̂_{a_t} E x_t‖² (latent consistency)
    + Σ (y_t − ûᵀ E x_t)²            (probe, at every observed time)
    + 10⁻⁶ (‖E‖² + ‖D‖² + Σ ‖F̂_a‖² + ‖û‖²).

The consistency term ties F̂ to encoded successors. That removes the
one-step scale gauge (D/c, cF̂) that would otherwise leave rollouts
undetermined. The remaining invertible gauge (E → SE, F̂ → SF̂S⁻¹,
D → DS⁻¹, û → S⁻ᵀû) leaves rollouts unchanged.

Fitting is by alternating least squares over the blocks E, {F̂_a}, D and û,
each solved exactly given the others. It runs 200 sweeps from 5
initializations (one rank-2 reduced-rank start, four random), keeping the
lowest objective on the fitting evidence itself. No held-out target data is
used. Rollouts are latent: x̂_L = D F̂_{a_L} ··· F̂_{a_1} E x₀.

The arms:

- **Target-only:** the latent model fitted on the target budget.
- **Frozen-dynamics transfer:** the same model and objective. F̂_a and û
  are taken from a fit on s1's 512 transitions and held fixed, and E and D
  are fitted on the target budget. Both arms use the target's y.
- **Per-action references:** per-action full least squares and per-action
  rank-2 reduced-rank regression, rolled out in observation space. They
  use x only.

Budgets are n ∈ {4, 8, 16, 32, 64, 128, 256} transitions, at every horizon.
Report X_L for each arm and the paired per-world ratio of target-only to
frozen-dynamics X_L.

## Criteria (worlds 0-9, declared before execution)

These are screening rules on ten worlds, not significance tests.

- **A1 generator.** The fixture passes if in all 10 worlds:
  - observability rank is 2;
  - the condition number of [R_i, B_i] is ≤ 10;
  - family A's commutant dimension is 1;
  - family A's swap is detectable with bound ≥ 0.05, both J and J_u;
  - the broken target is detectable with bound ≥ 0.005.

  Family B's swap is expected to be equivalent without the probe and
  detectable with it; that is reported, not required. Same-eigenvalue
  replacements are reported as distributions. The later contract may use
  only draws classified as detectable. If A1 fails, the generator is
  redesigned before any other criterion is read.
- **C1 representability** (per family, κ and clip setting). **Witness found**
  if, in ≥ 9/10 worlds, the constructive witness or the multi-horizon search
  has:
  - X_L ≤ 1e-2 for every L ≤ 128;
  - factorization residuals ≤ 1e-8.

  Which of the two qualified is reported. **Witness found only with relaxed
  clips** if C1 fails at ±2 but passes at ±6. **No witness found by these
  searches** if both fail. That is a record of search failure, not an
  impossibility claim, unless an analytical bound rules the task out.
- **C2 binding invariance** (where C1 found a witness):
  - **Invariant** if, for both s2 and t in ≥ 9/10 worlds, J_G ≤ 1e-3 with a
    minimizer of condition number ≤ 10.
  - **Drifts** if J_G ≥ 1e-2, a certified lower bound, in ≥ 2/10 worlds.
  - Otherwise **unresolved**.

  Rollout errors and the broken target are reported descriptively.
- **C3 inference.** Exact and settled inference **agree** if the settled
  relative activity error is ≤ 1e-3 in ≥ 9/10 worlds. Otherwise the steps
  needed are reported. This is not a gate: the later contract uses exact
  inference as primary in either case.
- **C4 design audit.** For each (n, L), the direct references show a
  **qualifying gap** if the target-only to frozen-dynamics ratio is ≥ 1.5,
  with that ordering in ≥ 9/10 worlds. The learning contract chooses its
  budgets and horizons from the qualifying set. If the set is empty, report
  **no qualifying gap found with these references**.

**Outcome for the proposal** (a project decision, not an inference from
these searches):

- **Proceed** to the learning/transfer contract if A1 passes, C1 finds a
  witness at ±2 clips for family A at some κ, C2 is invariant there, and C4
  has a qualifying gap.
- **Witness only with relaxed clips:** the learning contract must declare
  its clips as a design choice, with this result as context.
- **Drifts:** the action interface or binding scope is redesigned first.
- **No qualifying gap:** the fixture or budgets are revised before a
  learning contract.

## Computation

Sizes:

- d = 34 free activities for the window solve (module 0's 6 plus 14 each
  for modules 1 and 2);
- 6 for the encoder.

Theoretical costs:

- **Exact solves:** each costs O(d³) to factorize and O(d²) per right-hand
  side, with O(d²) workspace. M¹ and M² come from one factorization and 8
  right-hand sides per action pair. G costs a 2 × 8 least-squares fit.
- **Rollouts:** an L-step rollout costs O(L) 2 × 2 products in latent form,
  and O(4L) operations in 8 dimensions per state in observation form.
- **Gradient searches:**
  - each multi-horizon Adam step costs 4 window factorizations (one per
    action pair) plus a backward pass of the same order, and latent rollouts
    over 256 states × 8 sequences × 5 horizons;
  - P1 runs 10 worlds × 2 families × 2 κ × 2 clips × 3 η × 2,000 steps for
    the multi-horizon search, and the same again for the one-step
    diagnostic;
  - P2 adds 3 restarts × 2,000 steps for each of s2, t and the broken
    target.
- **Exact mismatch computations:** a 4-column SVD or a 6 × 6 KKT solve each,
  for 2 + 20 + 1 scrambles per world and one per binding fit.
- **P4:** 5 initializations × 200 sweeps of least-squares blocks with at
  most 16 unknowns per solve, × 7 budgets × 2 latent arms, plus one source
  fit per world.

The expected total is tens of minutes on one core, dominated by the P1
searches. This is an estimate; the runtime JSON measures it.

Storage, float64:

- **Parameters:** bindings are 120 per domain (3 × (W 32 + V 8)), banks 16
  (2 couplings × 2 actions × 4) and the probe readout 2. That is 378 values
  for three domains, about 3 KB. Adam state is twice that.
- **Evidence:** a window is 29 values (3 × 8 observations, 2 actions, 3
  probes).
  - Per world and family: 2 × 384 source windows, 128 target windows and
    the broken target's 128, about 240 KB.
  - Test initial states: 256 × 8 values.
  - Privileged capacity sets: 384 initial states and their oracle targets at
    5 horizons, about 2 MB including the latent oracle trajectories at those
    horizons.

  The raw samples are retained.
- **Rollout workspace:** only checkpoint horizons are kept, about 160 KB
  per evaluation. The autograd graph for the multi-horizon objective is
  about 256 × 8 × 128 latent steps, a few MB.

Expected peak RSS is below 1 GB (the PyTorch import dominates). The runtime
JSON measures it.

Outputs, following [computation reporting](computation-reporting.md):

- a deterministic scientific JSON, compressed with `gzip -n -9` if it is
  over 1 MB, with its uncompressed SHA-256 recorded;
- a runtime JSON with wall and CPU time per stage (P0-P4), every search,
  restart and rejected η counted, solve and Adam-step counts, peak RSS,
  PyTorch and NumPy versions, thread settings, and environment;
- no movement file.

## Verification

- **Energy:** the prior-modified Q for one module equals `lcc.build`'s
  module block at λ = 0.05, and matches a finite-difference Hessian of the
  explicit energy at λ = 0.001.
- **Gradients:** autograd gradients match central finite differences to
  relative 1e-6 on a random parameter set.
- **Rollout:** the latent and observation-space rollouts agree to 1e-10.
- **Oracle:** the conditional mean matches a Monte-Carlo estimate within its
  standard error on one world.
- **Mismatch computation on known cases:**
  - a joint basis change of family A gives J ≤ 1e-12, with a minimizer of
    condition number ≤ 1e6;
  - two commuting rotations have commutant dimension 2;
  - family B's swap gives J ≤ 1e-12, attained at condition number 1.
- **P4 gauge:** a gauge-transformed fit leaves rollouts unchanged to 1e-10.
- **Draft reproduction:** the runner's three-module construction reproduces
  the draft-check JSON's window gains, residual magnitudes, leakage and
  scramble bounds to 1e-10.

## Declared choices

- **Thresholds.** The draft predicts leakage-dominated avoidable error near
  5e-3 for the constructive witness at ±2 clips, against C1's 1e-2. The 1e-3
  drift threshold and the A1 bounds (0.05, 0.005) are set from the draft
  world's values (0.17 and 0.022) with margin. None is derived from a
  downstream requirement.
- **Filler action at L = 1.** Declared as the first action. It is
  immaterial for this witness, where there is no back-reaction, but not in
  general.
- **Nuisance scale.** Variance 0.25 per direction with these loadings gives
  nuisance about 1/16 of the expected signal power in observation space: a
  weak regime. A strong regime is left to the learning contract.
- **Hebbian or fast-weight binding arms** are deliberately excluded.

Out of scope: the clamped-energy rule and any learning claim; correlated
nuisance across observations; action renaming or learned action
correspondence; the finite quotient fixture; consolidation and sleep.
