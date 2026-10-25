# Relational capacity: composition, task and harness handoffs

**Status:** handoff audit, 2026-10-25, for the
[relational capacity and composition proposal](relational-capacity-and-composition-study-proposal.md).
These are not a frozen contract or experimental results. No model was fitted
and no arm was scored. Frozen sources, contracts and reports are unchanged.
Stage A, architecture/interface selection and contract integration remain with
the analytical lead. Everything below is a finding to check or a proposed
interface that lead can accept or reject. It is not a selection.

Small read-only checks used the frozen task generator and random constrained
parameters with no fitting, in scratch scripts outside the repository
(§5). They verify algebraic and structural claims. They are not capacity or
transfer evidence.

## 1. Composition audit

### 1.1 Existing multi-module designs and their interfaces

The repository has **no continuous multi-module implementation**. Every study
with several modules is discrete or uses ternary integer energies. The current
circular module is a single module whose four sectors are coupled only by fixed
lateral ring springs.

| Design | State and communication | Output interface | Lesson for this study |
| --- | --- | --- | --- |
| Core mediation v1, phase B (`src/core_mediation.py:251`) | Mode labels; integer compatibility tables φ_AK, φ_BK, ψ_ABK, φ_AB | Satellite pair (a,b) | "Remove the core" was defined as deleting u_K and C while keeping U. That gives a precise, matched no-communication meaning. |
| Core mediation v1, phase C (`src/core_structured.py:45`) | Binary-unit Hamming-prototype modules; interaction **only through readouts**, with a penalty for the ⊥ readout | Satellite readouts | Robust codes made readout coupling inert. A hard-descriptor channel can carry no usable signal. |
| Graded core v1/basins (`src/core_graded.py:48`) | Multilinear coupling on unit fractions (popcount) of A, B, K | Majority readouts | Graded unit-level coupling restored participation, at the cost of tie-driven spurious minima. |
| Graded cellular v1 (`src/graded_cellular.py:42`) | Three cells plus context b; coupling `gamma/2 (mean - b)^2` | Mean descriptor | Coupling through a declared summary descriptor. |
| Reciprocal participation, shared input, mechanism/local learning, core abstraction v1–v3 (`src/reciprocal_participation.py:81`, `src/shared_input_mediation.py:41`) | Ternary z=(i1,i2,k,a,b); bilinear `-w x y` couplings; input anchored by `2(i-e)^2`; family C adds `-w_12 i1 i2` | Complete reachable-terminal **set** projected to (a,b) | Feedback through the input needed one input coordinate shared by both satellites. A shared observed coordinate is itself a communication channel. |
| Role-binding example (`src/mediation.py`) | Core is a relation over (satellite, role, site) tuples; output module constrained to agree | Separate output module | A generic matcher reproduces the responses, so the core was not shown to be essential. |
| Shared-surface analyses ([shared surface](shared-surface-analysis.md), [full surface](full-surface-analysis.md)) | One module; interior eliminated into the surface Schur complement S | Outer ring is both input and output | Zero-energy surface patterns are exactly null(S). |
| Circular module (`src/single_module_plasticity.py:123,186,284`; masked: `src/masked_capacity.py:183`) | Within a sector: reciprocal radial maps. **Between sectors: only the fixed `0.001` cycle Laplacian on every ring** | Settled outer state at unobserved coordinates | Every visible→hidden influence currently passes through λ=0.001 edges; this is the force the obstruction bounds. |

Organizational constraints from the record:

- [Research program §2](research-program.md): core and satellites, with
  stronger permitted inter-module coupling toward the interior. Input and
  output may be modules in their own right.
- [Plasticity design](single-module-plasticity-design.md): the outer ring is
  the input/output surface, with no separate global I/O modules. It forbids
  "moving directly to multiple modules" without motivation; the obstruction now
  supplies that motivation.
- [Interaction semantics](interaction-semantics.md): conjoining constraints
  onto an unchanged relation cannot enlarge what is admissible. Finding C1
  below is the continuous counterpart of that result.

### 1.2 Findings

**C1. Energy composition intersects free outputs; readout composition spans
them.** Suppose modules share an output state y through energy terms: a common
outer ring, soft ties `β/2||x_0^(k) - y||^2`, or anchors on y. Eliminating
each module's interior leaves `y^T (Σ_k S_k) y / 2` with every S_k PSD. Hence
`null(Σ_k S_k) = ∩_k null(S_k)`. Additional modules can only stiffen an
existing output subspace; they cannot enlarge the set of freely expressed
output patterns. Numerically, two rank-one modules sharing one outer ring have
**0** free outer directions per sector, against 1 for each module alone.

A fixed readout combination of separate states, such as their mean, instead
has hidden outputs within the bounded perpendicular slack C of
`span{w^(1)_j, w^(2)_j}`. That span is 2-dimensional in every sector for
generic parameters. The enlargement belongs to the readout, not to
communication, which is exactly the confound the proposal anticipates.

**C2. Sector-aligned communication opens no new visible→hidden path.** If
the coupling joins sector j of one module only to sector j of the other, at
any ring, hidden sectors still receive all input-dependent force through the
λ-weighted lateral edges. `E(x) ≤ E(0)` still bounds those edges' total force
by `sqrt(8 α λ)` under a shared total anchor budget α. Coupling changes the
response operator, not the force: longer and softer chains enlarge the chain
inverse up to `sqrt(Σ k^2)/min(a_min, γ)`. For example, two coupled length-four
chains give `sqrt(204)/0.325`, which makes the old bound inconclusive at about
1.24. Any M1 gain over M0 would therefore come from amplifying a weak λ-scale
force, an ill-conditioning route the harness must expose. Correspondences that
cross sectors (parts) create strong visible→hidden paths. They are the
relational arm R, not M1.

**C3. Interior correspondences are gauge up to the weak lateral term.**
Re-gauge one module's ring-3 sector coordinates by sector-local orthogonal O:
`D_2 → D_2 O`, `v → O^T v`, `T → T O`. Without the inner-ring Laplacian, the
outer equilibrium is unchanged; the measured difference was 1.8e-15. With it,
the difference was 8.4e-3. For Haar-invariant initialisation, "aligned"
(T=I) and "misaligned" (fixed T in the sector-local orthogonal group) interior
couplings are therefore the same *trained* arm, apart from O(λ) effects.
Misalignment remains meaningful as a frozen-parameter intervention, and for
cross-sector permutations that a sector-local gauge cannot absorb.
Outer-surface correspondences are anchored to observation coordinates and are
not gauge.

**C4. Flag for Stage A item 1 (rank-two, S2).** With the terminal penalty
removed, each sector has a 2-dimensional free outer plane, and the masked
Hessian stays positive definite (condition about 2.1e3 for one random draw).
Stationarity along the free directions N gives `N^T L x = 0`, so λ cancels
from the leading-order determination of the free components. For one random
draw, hidden output changed 13% when λ went from 1e-3 to 1e-5, and by at most
7.7% across random routes in [0.325, 1]. If this is general, S2's hidden
output is close to a fixed, route-insensitive harmonic extension through ring
adjacency, i.e. close to a fixed linear map. The existing affine floor
(0.246 on the q=1 subset) might then nearly bind. This is a hypothesis to
derive, not a result.

### 1.3 Proposed interfaces (for the analytical lead to accept or reject)

| Arm | State / P | Proposed energy | Output | Free outer dims per hidden sector | Notes |
| --- | --- | --- | --- | --- | --- |
| S2 | 80 / 320 (v removed) | v1 energy without terminal blocks | Outer hidden coordinates | 2 | See C4. |
| M0 | 2×80 / 656 | Two independent masked modules, each anchored at **α/2** to the same observations (total budget α) | **Mean** of the two outer states at hidden coordinates (fixed, parameter-free, declared readout) | 2 (span; readout-supplied) | Equivalent to M1 at γ=0. |
| M1 | 160 / 656 (+0 if T fixed) | M0 + `γ/2 Σ_j ||x_3,j^(1) - x_3,j^(2)||^2` (ring-3, sector-aligned, T=I as gauge convention) | Same mean | 2 | Interior placement follows the research program. Fix γ analytically or on a declared, fully charged grid. |
| M1-alt | as M1 | Outer-ring sector-aligned coupling | Same mean | 2 → intersection as γ→∞ | Not recommended: in the limit it becomes C1's shared surface. |
| Shared-output control | 32+2×48 | Two inner stacks on one outer ring | Outer ring | **0** | Optional named control showing that energy composition stiffens. |
| R (candidate) | ≈80 + family | `ρ/2 Σ_r π_r(x) Σ_{j∈{0,1}} ||x_0,j+2 - T_r x_0,j||^2`; the Hessian has blocks `[T^T T, -T^T; -T, I]`, so visible sectors feel the transpose reaction | Outer hidden coordinates | Hidden sectors anchored through ρ | Exact for the dev task: `z_{j+2} = e^{2ir} z_j` (§2.1). A fixed T expresses one relation; π_r(x) is an input-dependent selection from the pilot. In world 1, `T_r = M_{j+2} R(2r) M_j^T` depends on the renderer; supplying it is privileged. |
| G | R with T restricted | T in the commutant of the declared C8 rotation representation, i.e. similitudes `s R(φ)` on the signal plane | As R | As R | The representation is public only in the identity world. Elsewhere it needs the renderer: either restrict G to declared-representation worlds or charge an alignment learner. |

Before seeing the analytical preflight, this audit recommended the mean
readout as the primary interface. The preflight, written concurrently, chose
a designated output module coupled through terminal ports as primary, with
summed outputs as a labelled control. **That decision supersedes this
recommendation.** The interface decision belongs to the analytical lead, and
the designated output follows the repository's evidence/output separation.

### 1.4 Reconciliation with the concurrent analytical preflight

[Analysis](relational-capacity-and-composition-analysis.md) ·
[results](relational-capacity-and-composition-analysis-results.md) ·
[weak-coupling bound](relational-composition-weak-coupling-bound.md)

- **Agreement.** C1 matches its "passive helpers cannot enlarge the
  zero-penalty subspace" (§4) and its point that summed outputs enlarge the
  span (§1.2). C2 is the sector-aligned counterpart of its weak-coupling bound
  and of its fixed-route rank ≤ rank(B) result.
- **C3 still applies to its terminal ports.** For `mu ||P x_in - T P x_out||^2`
  on ring-3 coordinates, any sector-local orthogonal T is gauge-equivalent to
  T=I up to the λ=0.001 inner-ring lateral term. The port correspondence's
  "alignment" therefore needs a declared gauge convention. It is meaningful as
  a frozen intervention but not as a trained arm difference.
- **R candidate and its §6 closure/amplitude issue.** Adjacent-part rotations
  need amplitude factors (q, 1/q, q), and a closed four-cycle is inconsistent
  for r=1,3. The j→j+2 correspondence `z_{j+2} = e^{2ir} z_j` avoids both. It
  is a pure rotation, amplitude-preserving because parts j and j+2 share an
  amplitude, and uses two open edges (0→2, 1→3) with no cycle closure.
  Selecting r is still the unresolved relation-dependent inference that §6
  identifies.
- C4 refines its §2 (unique equilibrium, capacity unproved): even with
  uniqueness, S2's free components may be nearly route-insensitive.

### 1.5 Response to the analytical results

**R1. The weak-coupling bound excludes only very weak coupling.** With
alpha=0.1, lambda=0.001 and a_min=0.325, its threshold is
`mu < 0.5[a_min^2 (1-sqrt(0.2))^2/(60 alpha) - 4 lambda] = 6.90e-4`.
It therefore rules out only coefficients comparable to the lateral λ. Any
port coefficient at or above about 7e-4 is already inconclusive under this
bound, so it constrains little of a plausible grid. The bound is still
correct and useful; it just does not narrow the search much.

**R2. A route-sensitivity necessary condition applies to every arm,
including "ready" S2.** Let f be any arm's routed predictor. Let g be the same
parameters with routes held at any fixed value. g is then linear in the masked
observations, so on the 24 q=1 scenes `mean ||g-t||^2/2 >= F = (8-2sqrt2)/21`.
If f meets the target, `||f-t||^2/2 <= 0.10` on every scene. Minkowski's
inequality on RMS over those 24 scenes gives

```
RMS(f - g) >= sqrt(2F) - sqrt(0.2) = 0.25459,
```

or 18.0% of the hidden target norm sqrt(2). Routes must move the hidden
output at least this far from **every** fixed-route version of the same
parameters. This is a prospective screen that needs no fitting. The route-sensitivity
numbers in C4 (one random draw) are not comparable to it, because they are
relative to that draw's own small output norm. A Stage B contract can apply
the screen as a solver-checked diagnostic, and analytically where the route
range permits a bound. It is necessary, not sufficient.

**R3. "Analytic extractor" comparators.** The results call supplying the
relation or its analytic extractor a privileged reference. For this task, the
cubic rational map in §2.1 is that extractor. Its renderer-free polynomial
form is a learnable, chargeable comparator, not only a privileged reference,
so the next contract should include it. With odd relations (§2.2) the fixed
quadratic floor becomes 1.0, which makes the "cannot pass the affine bound"
finding uniform rather than specific to the q=1 subset.

**R4. A shared transient relation state.** The results suggest this as a
candidate for selecting relations. Under odd relations it would have to detect
a relation that is invisible to every pairwise ensemble moment (§2.3). A
per-scene compatibility gate like the existing ratio gate can still do this,
because it is evaluated per scene rather than through ensemble statistics.
The discrimination comparators must then include the quadratic detector to
show that this is more than covariance modelling.

## 2. Task audit

### 2.1 Facts about the development fixture (verified)

- Hidden parts are a fixed equivariant rational function of the visible parts:
  `z2 = z1^2 conj(z0)/|z1|^2` and `z3 = z1 (z1 conj z0)^2/(|z0||z1|)^2`
  (max error ≤8.5e-16). Equivalently, `z_{j+2} = e^{2ir} z_j` (5.0e-16).
  This map stores nothing specific to any relation and solves every
  relation–ratio group, including the held-out ones.
- Sign is rotation by t+4, so signed lookup adds nothing beyond rotation
  lookup in this generator.
- Reflection alone maps r→8−r, outside {1,2,3}: no held-out scene matches a
  training scene. **Reversal of part order composed with reflection maps
  (r,q)→(r,1/q).** Under full observation, 16 of 24 held-out scenes exactly
  match training scenes up to common rotation: (1,1/3)↔(1,3) and (3,3)↔(3,1/3).
  This does not leak into completion, where visible and hidden roles differ. It
  does leak into discrimination scored on all four parts, and into any arm
  whose correspondences are symmetric under reversal. Reciprocal T terms are
  symmetric by construction.
- Under common rotation, each relation fixes every pairwise Hermitian moment
  `E[z_j conj z_k]`, with phases (0,−r,−2r,−3r); pseudo-covariances vanish.

### 2.2 Selective compositional transfer

Risks:

1. **Formula shortcut.** A learner whose function class contains the cubic
   rational map above transfers to every held-out combination without
   compositional binding. Add a required comparator in two forms, both charged:
   a latent version (privileged, using renderer planes) and a renderer-free
   version (polynomial or normalized-monomial features of the visible
   observation coordinates, fitted on training scenes only).
2. **Orbit closure.** Withhold held-out combinations closed under every
   transformation the *scoring* is invariant to, not only the input
   nuisances. For full-observation tasks that includes reversal∘reflection.
   Audit exact matches by canonicalising each scene over the declared group in
   latent coordinates (privileged audit) and asserting none remain.

Proposed fresh-generator choice, an option for the contract: use **odd
relations {1,3,5,7}**, or all of Z8 with degenerate r=0,4 flagged. Every
pairwise moment is then exactly zero (§2.3). Consequences:

- The best affine predictor of hidden from visible parts is zero, so affine
  loss is 1 and any fixed-route linear module is uninformative. The "need for
  nonlinearity" becomes clean rather than a 0.246 floor.
- Reflection pairs relations (1↔7, 3↔5). Hold out reflection-paired
  combinations jointly, or declare reflection relation-changing and include a
  reflection-lookup control.
- Identifiability: the visible increment determines r, with a minimum
  separation of π/2 between the odd relations. Norm ratios determine q.
  Report margins under the declared noise.

Add continuous common rotation and positive scale as nuisances, with
independent recurrence of each factor value. Report all factor marginals.

### 2.3 Relationship discrimination

At fixed routes, every reciprocal quadratic energy scores a fully observed
scene by `x^T S x / 2`: a second-order statistic. An LP over distributions of
phase increments shows:

| Relation set | Max pairwise moment | Minimum valid mass of any distribution matching all six pairwise moments |
| --- | --- | --- |
| {1,2,3} (dev task) | 0.805 | **1.0** |
| {1,3,5,7} | 0 | 0 |
| Z8 | 0 | 0 |

For the development generator, every violation set differs from the valid
set in some pairwise second moment. A fixed quadratic detector can in
principle detect any violation, so success there is not evidence beyond
covariance modelling. Matching only adjacent-pair moments is feasible. With
odd or full relations, violation sets can match every second moment, so
fixed quadratic scores have equal ensemble means and detection needs
higher-order (routed or nonlinear) computation.

Proposed elements:

- Comparators: a train-only Gaussian/quadratic likelihood-ratio detector
  (charged) for every discrimination claim.
- Score: a conditional completion residual
  `||f(x_vis) - x_target||^2 / ||x_target||^2` under a declared mask. It uses
  each arm's own completion output, so it is comparable across architectures
  without calibrating raw energies. An energy score is allowed only with a
  separately charged train-only calibration probe.
- Violation types, labelled separately:
  - V1: wrong hidden increment, with per-part amplitude and orientation
    marginals preserved exactly.
  - V2: reflected hidden parts (relation-changing).
  - V3: amplitude-correspondence swap (a detail violation, not a relation).

### 2.4 Contextual disambiguation

- The observation model supports per-coordinate masks
  (`src/abstraction_preflight.py:15,181`), so ambiguity can be declared
  without a new likelihood.
- **Ambiguity is rendering-dependent.** Observing only coordinate 0 of part 1
  (with part 0 full) leaves 12/72 targets non-identifiable in the identity
  world and 0/72 in world 1. A generic functional breaks the cos-only symmetry
  on the 8-point grid. Define ambiguity in latent terms. With continuous
  orientation, one observed functional is two-fold ambiguous (θ vs 2φ_u−θ) in
  every rendering, and amplitude is also unresolved. Enumerate or derive
  identifiability per rendering, and generate masks generator-side (masks are
  observation metadata).
- On the dev grid there are 255 pairs with the same part-1 evidence, different
  part-0 context and different targets: material for matched context pairs.
- **Irrelevant-context probe within four parts.** z3 is invariant to |z0|,
  whereas z2 scales with it. Rescaling part 0's amplitude should therefore
  change the part-2 prediction proportionally and leave part 3 unchanged.
  Score each hidden part separately. This probe shifts q off-grid; label it a
  distribution shift.

### 2.5 Geometric consistency

- On observations, common rotation acts as `M_j R(φ) M_j^T` per part. Only
  the identity world exposes it without renderer knowledge.
- Measure:
  - equivariance error `||f(g·x) - g·f(x)||` for continuous φ, reported
    separately for the C8 subgroup;
  - invariance of declared relation channels;
  - sensitivity to relation-changing reflections.
- For C8, equivariant linear maps on the signal plane are exactly similitudes.
  Equivariance success with a supplied representation is "supplied
  symmetry", not discovery.

## 3. Harness plan (implementation not started)

**Ownership.** Use new files only: proposed `src/relational_harness.py`,
`experiments/relational_capacity.py` and `tests/test_relational_harness.py`.
Import frozen helpers (`experiments/locality_adaptation.py:60,74,78,82,235`
for canonical encoding, hashing, git and environment;
`experiments/masked_capacity.py:52,68,98` for the freeze/refusal pattern);
never edit them.

**Arm interface.** `ArmSpec` declares K, m, D, P, E, d, the anchor budget,
coupling coefficients, the output interface, the symmetry representation and
privileged flags. An `Arm` exposes:

- `curvature_blocks()`, `hessian(routes, mask)`, `rhs(observed, mask)`;
- `routes(pilot)`, `output(state)`;
- an independent `energy(state, observed, mask, routes)` expansion;
- `act(g, state)`.

Interventions return new Arms: `without_coupling()` (γ=0, anchors unchanged),
`fixed_routes(value)` and `misaligned(T→T P)`. Each re-runs the solver checks.

**Solver checks, run on every solve:**

- symmetry;
- Cholesky success;
- relative residual ≤1e-10;
- Cholesky condition bound, plus an exact condition number at D≤160;
- Schur elimination onto output coordinates agreeing with the full solve
  within 1e-9;
- `E(x*) ≤ E(0)` and ≤ `E(x*+δ)` for random δ;
- a nullspace-dimension report. A singular arm must declare an identifiable
  output and show output invariance along the computed null directions.

**Metrics:**

- per-scene loss, decoded orientation (grid codebook) or continuous angle
  error, and joint correctness;
- means and worst case per factor group;
- the §2 evaluators;
- a comparator registry: zero, mean, affine, orbit lookups (including
  reversal∘reflection), rotation/scale lookup, factored template (privileged),
  formula comparators (privileged and renderer-free) and the quadratic
  detector. Every comparator is charged.

**Cost accounting:**

- One meter with a nested phase tree storing inclusive and exclusive
  wall/CPU, so nested totals are never added to their components.
- Counts of pilot/second/adjoint factorizations by matrix size, solves,
  exposures (one scene through the whole composed system), independent
  examples, optimizer steps and retractions.
- A candidate ledger that includes rejected and unexecuted candidates.
- Bytes by category: parameters, optimizer state, retained examples, caches
  and dtype.
- Peak RSS from `RUSAGE_SELF` and `RUSAGE_CHILDREN` (Linux KiB), labelled as
  process scope.
- Null plus a reason for anything unmeasured.

**Complexity and planning budget** (per-forward Cholesky ≈ D³/3 flops):

| Arm | D | P | Cholesky flops (one factor) | Note |
| --- | --- | --- | --- | --- |
| S1/S2 | 80 | 328/320 | 1.7e5 | Historical S1: no repeat search |
| M0 | 160 | 656 | 1.4e6 dense, or 2×1.7e5 block | Report both. Use the dense direct method for parity with M1. |
| M1 | 160 | 656 | 1.4e6 | — |
| R/G | ≈80 + family | TBD | ≈1.7e5 per route set | Depends on Stage A |

The proposal's ceiling of 40,000 updates, 2.88M exposures and about 2.92M
second-pass factorizations holds before checkpoints. Masked v1 measured 311 s
for 8,000 updates at D=80. Assuming cost is cubic-dominated, M0/M1 would take
at most about 8× that per arm, so the whole Stage B search would be under
about 2 h on one thread. This is an **extrapolated planning estimate, not a
measurement or speed claim**.

**Provenance:**

- Freeze contract → sources/tests/manifest, recording the hashes of all
  frozen dependencies, then run tests, then byte-identical double task
  audits, then the search, then verification.
- Runners refuse changed inputs and recorded phases.
- Scientific JSON is deterministic; runtime JSON is separate.

**Missing historical cost data to disclose.** The S1 reference (masked v1)
lacks native workspace, FLOPs, backward primitive-solve counts and
per-scene inference latency; all are null with reasons in its runtime. The
discrete multi-module studies are not cost comparators.

## 4. Unresolved decisions for the contract

1. The preflight has chosen the designated-output interface. Still open: the
   port gauge convention (C3), the coupling coefficient, and the summed-output
   control's matched anchor budget.
2. Decide whether C2 and C4 are derived into bounds before M1/S2 receive any
   search budget. Adopt the R2 route-sensitivity screen for every routed arm.
3. Choose the R selection mechanism π_r: a finite route family or a learned
   T with a gate. Decide whether R/G run only in declared-representation
   worlds.
4. Choose the fresh relation set (odd/Z8 vs {1,2,3}) and the orbit group
   used for holdout closure.
5. Decide whether the formula comparator and the quadratic detector are gate
   comparators or reported references.
6. Choose the discrimination score: the completion residual (recommended) or
   calibrated energy.

## 5. Scratch verification and its cost

The scripts live in the session scratchpad, not the repository. They used the
pinned `.venv`, float64 and one process, read frozen generators/parameter
samplers, and wrote nothing to the repository. The task identities, orbit
links, ambiguity enumeration and moments took 0.035 s. The structural ranks,
conditioning, λ/route sensitivity and gauge test took 0.19 s wall (2.18 s
CPU, multithreaded BLAS). The gauge confirmation took 0.009 s and the moment
LPs 0.045 s. Peak RSS was 258,616 KiB in the first, import-dominated process.

The structural checks use one or two random draws (seed
`[20261025, 77/78]`), so they illustrate generic behaviour without proving
it. The prospective task audit should re-implement them as deterministic,
versioned checks before any fit.
