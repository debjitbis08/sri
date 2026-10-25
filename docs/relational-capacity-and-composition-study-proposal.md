# Relational capacity and composition: study proposal

Status: design proposal, 2026-10-25. Not a frozen execution contract. No new
experiments have run. Resolve the analytical and interface decisions below,
then freeze an executable contract before implementation and numerical work.
Follow [computation reporting](computation-reporting.md); preserve historical
sources and artifacts.

Analytical follow-up: [derivations and decision table](relational-capacity-and-composition-analysis.md),
[weak-coupling obstruction](relational-composition-weak-coupling-bound.md), and
[checked results](relational-capacity-and-composition-analysis-results.md).
These refine the proposed interfaces and exclude fixed-route capacity searches;
the comparison set below is the original proposal, not a frozen final factorial.

## Question and scope

Can reusable predictive relationships be expressed and learned through extra
capacity inside a circular reciprocal module, through interaction between
modules, or through structured relational connections? Does geometric structure
help, and which part of any success was supplied rather than learned?

The existing [obstruction](single-module-masked-capacity-obstruction.md) applies
to the declared isolated module, not every composition of modules. More retained
dimensions or more modules do not by themselves establish relational capacity.
We need selective transfer across details, sensitivity to relation violations,
and evidence that relevant context changes inference.

Keep explicit observation masks, reciprocal energy-based inference, and declared
outer-state outputs throughout. Any change to constraints, connectivity, output
composition, or learning rule is a named intervention. No hidden target input,
scene-specific fitted parameters, silent decoder, or post hoc gain.

## Stage A: analytical preflight

Analyze the following before searching for parameters:

1. **Retained dimension.** Replace the rank-one terminal passthrough with a
   rank-two passthrough and derive the resulting nullspaces and masked Hessian.
   In the current two-coordinate terminal sector this removes the terminal
   penalty entirely; it is not merely adding a second vector to the old model.
   Do not reuse the old uniqueness proof without checking its assumptions.
2. **Composition.** Specify exactly which module states communicate, their
   coordinate correspondence, the coupling energy and its coefficient, and
   which state is the prediction. Count shared observation anchors once under
   a common total data-fidelity budget. Derive the block Hessian and its Schur
   complement at the output. Determine whether extra modules enlarge the
   accessible output subspace or only alter stiffness in an existing subspace.
3. **Relational connections.** Compare an equality constraint with
   `0.5 * ||x_j - T_ij x_i||^2`, including the transpose contribution required
   for reciprocal inference. A fixed correspondence expresses one relation;
   identify explicitly how multiple relations could be represented or selected.
4. **Geometry.** Specify the action of common rotation on every state and
   connection. Distinguish scalar invariant channels from vector equivariant
   channels. Sector-specific renderings require coordinate alignment; their
   true signal planes must not be silently supplied to a learned arm.
5. **Solvability and expressivity.** For each retained design prove uniqueness
   or declare a principled identifiable output despite state nonuniqueness.
   Analyze fixed routes separately from input-dependent routes. A composition
   with fixed quadratic coefficients and fixed masks remains a linear input
   map; adding depth/modules alone cannot escape the existing affine task floor.

Produce a table of allowed variations, excluded violations, free dimensions,
identifiability, and known capacity bounds. Prefer constructive solutions or
impossibility bounds to optimization. A bound becoming inconclusive is not a
positive capacity result. Bound coupling strengths analytically where possible;
otherwise declare their finite search grid and charge its full cost.

## Stage B: shared capacity screen

Use the frozen 72-scene task and two renderings from masked-capacity v1 as
development fixtures. Their outcomes are already known; they are not fresh
confirmation data. Retain its per-scene error <=0.10 and joint orientation
criterion, independent serialization checks, and strong shortcut controls.

Use the following small comparison set, subject to Stage A feasibility:

| Arm | Purpose |
| --- | --- |
| S1: existing isolated rank-one module | Historical obstruction/reference; no repeat failed search |
| S2: isolated rank-two retained space | Test removal of the established bottleneck |
| M0: two rank-one modules, no communication | Control for multiplicity and parallel capacity |
| M1: same two modules with reciprocal communication | Test whether communication adds usable capacity |
| R: simplest feasible scaffold with relational correspondence constraints | Test relation-preserving constraints versus equality/smoothing |
| G: the same R scaffold with declared geometric restrictions | Test the contribution of a symmetry prior |

Freeze the scaffold choice for R/G from analytical feasibility, before capacity
scores. Do not choose it because it won a search. R and G must have the same
observation/output semantics. Their restriction changes effective parameter
count; report this rather than claiming exact parameter matching.

M0/M1 must use the same fixed output interface. A sum of outputs can itself
create extra capacity even without communication; it cannot be credited to
interaction. An output in one designated module tests a different hypothesis.
Stage A must choose and justify one interface prospectively, and derive its
effective rank. Do not give M1 a readout unavailable to M0.

Proposed ceiling: five new arms, two worlds, two initializations, two declared
optimizer settings, 1,000 full-batch updates: at most 40,000 optimizer steps
and 2,880,000 scene exposures, plus checkpoints and verification. Analytically
excluded arms get no search. Resolve optimizer objectives, minimax versus mean
loss, selection rules, coefficient grids, and early stopping before freezing;
the earlier mean-loss search worsened worst-case loss. These choices cannot be
changed after seeing arm outcomes. No automatic extension after failure.

All-scene fitting is privileged capacity search, not learned transfer. Separate
full-gradient witness optimization from the eventual local/frozen-route learning
rule. Failure under a budget remains unresolved unless proved otherwise.

## Stage C: learning and abstraction

Only capacity-feasible designs advance. Freeze a fresh learning contract with
training/validation/test worlds and seed counts justified by precision or power;
do not use capacity-fitted parameters as initialization. Select checkpoints and
hyperparameters using training/validation only. Never retune on final test.

One shared generator supplies four complementary evaluations:

- **Selective compositional transfer:** hold out relation/detail combinations,
  with entire common-transformation orbits kept within one split. Each factor
  occurs in training; held-out combinations do not. Include independent nuisance
  variation and recurrence. Fix the generator and audits before scoring models.
- **Relationship discrimination:** match appearance/norm statistics while
  violating the relation. Report discrimination and reconstruction separately.
  Use a declared conditional score or a separately costed train-only probe;
  do not assume raw energies are calibrated across architectures or masks.
- **Contextual disambiguation:** identical ambiguous local evidence paired with
  different contexts requiring different interpretations. Score whether context
  changes the prediction correctly, while irrelevant context changes do not.
  Ambiguity must be representable by the declared observation model, such as
  partial coordinate masks; an arbitrary two-mode likelihood is an extension.
- **Geometric consistency:** measure common-rotation equivariance of outputs,
  invariance of declared relation states, and sensitivity to relation-changing
  transformations. Reflections are not automatically nuisances: they can reverse
  signed relations. Supplied symmetry success is not learned symmetry discovery.

Use the same instances across arms. Retain affine, signed/orbit lookup and the
privileged factored-template reference. Clearly label renderer access and true
amplitude correspondence in privileged controls. Include a learned factored
comparator if making a claim beyond those controls; charge its fitting cost.

Do not require unspecified internal states to become invariant: pose information
may be necessary for reconstruction. Probe declared relation channels and assess
collapse through both variation in outputs and sensitivity to valid violations.

## Mechanism tests and attribution

On the same trained models, freeze parameters and evaluate removal of
communication, fixed versus adaptive routes, and aligned versus deliberately
misaligned correspondence. Check numerical solvability for each intervention.
These interventions measure deployed dependence, not what could be relearned.
The separately trained M0/M1 comparison addresses learning without communication.

Credit between-module relational computation only if communication improves
selective transfer beyond M0 and its effect follows relevant contextual changes.
Redundancy, ensemble averaging, extra observation weight, and a richer output
interface are competing explanations to measure explicitly.

The separate plasticity questions remain a distinct follow-up: sector error
versus plasticity allocation, actual constrained parameter updates, and source
forgetting under outer/inner interventions. They can reuse logging infrastructure
but should not be folded into this capacity/abstraction factorial. Successful
abstraction and beneficial retention are separate claims.

## Computation and reporting

Let K be module count, m states/module, D=Km, B batch size, P total trainable
parameters, E coupling blocks, and d their width. Dense block construction costs
up to O(D^2 + E d^2) per scene once local terms are available; a direct solve
costs O(D^3) time and O(D^2) workspace. A two-pass batch with a shared pilot
costs O(D^3 + B D^2 + B D^3), plus route construction. Full differentiation
through dense solves has O(B D^3) time and O(B D^2 + P) stored tensor order;
declare actual implementation retention and native workspace separately.
Coupling construction/application adds O(E d^2) for dense blocks or the declared
sparse alternative. Retractions, optimizer work, probes, task audits, parameter
selection and all failed candidates are additional named costs.

If block elimination or iterative settling is used, derive that algorithm's
cost separately and compare with a direct equivalent. Report condition numbers,
tolerances, iterations and failures; do not infer speed from parameter count.

Record wall/CPU time by phase and candidate, peak RSS, solver/factorization
counts, exposures, independent examples, persistent state, optimizer/replay
bytes, and temporary workspace. Report quality versus total selection/training
cost, inference cost and retained bytes. Use both equal-example and bounded
compute comparisons; matched architecture controls do not imply matched compute.
Do not add nested timing components to their enclosing total. Scientific outputs
are deterministic and separate from runtime JSON. Disclose missing historical
computation fields rather than rewriting old artifacts.

## Parallel work and deliverables

The analytical lead owns Stage A, architecture/interface decisions and final
contract integration. Independent handoffs can proceed without numerical fits:

1. **Task audit:** design fresh compositional/ambiguity/violation fixtures;
   analyze identifiability, symmetry-orbit leakage and privileged shortcuts.
2. **Composition audit:** read the repository's existing multi-module designs;
   document actual communication and output interfaces, candidate block energies,
   and whether proposed composition preserves the project's organization.
3. **Harness plan:** specify reusable solver checks, per-scene metrics, cost
   instrumentation and provenance. Plan implementation against explicit interfaces;
   do not modify frozen sources or begin runs before the contract is integrated.

Each handoff returns assumptions, derivations or design artifacts, unresolved
decisions and exact file references. No independent architecture selection,
hyperparameter search, or overlapping source ownership. The next deliverable is
an analytical decision table and an executable Stage B contract, not a claim that
all proposed arms are already well defined or feasible.
