# Comparative modeling program

**Status:** first bounded iteration complete; continuing evaluation protocol with no preferred substrate or scored
winner. This replaces the assumption that the next step is to develop the
role-binding or structural-rewriting candidate further. Both remain candidates
or controls. OP-02 through OP-03B retain their original scope and certificates.

**First completed iteration:** [v1 contract and candidate cards](comparison-v1-contract.md),
[results and grades](comparison-v1-results.md), and a frozen 50-case certificate
now cover all four initial families. The protocol below remains the standard for
future comparisons; no architecture has been selected by this iteration.

**Combined follow-up:** [graded cellular v1](graded-cellular-v1-results.md) tests
the cellular/nonlinear combination against matched controls. It finds recoverable
post-input modes in a narrow proxy and explicitly withholds credit for omitted
layers, graded layer connections, core structure, and learning.

## Separate the unresolved choices

| Axis | Choices to investigate, not established equivalences |
| --- | --- |
| Mode | Spectral eigenmode of a declared operator; element/state of a finite algebra; structured configuration; invariant behavior; another precisely defined object. |
| Nesting | Geometric containment; topological inclusion; algebraic filtration/substructure; classes ordered by response sensitivity. |
| Collective interaction | Compatible coexistence, graded assessment, joint transformation, overlap agreement, or a specified combination of these aspects. |
| Core substrate | Cell/simplicial complex, graded relational structure, finite algebra, operator-bearing structure, or another finite representation. |
| Realization | A declared construction, dynamics, selection relation, or distribution; admissibility alone does not choose a response. |
| Persistence and approximation | Properties to derive or refute from the preceding choices and their assumptions, not capabilities assigned by the substrate's name. |

These axes are not necessarily independent. A spectral mode requires an operator
and a space on which it acts; a cell complex alone supplies neither response nor
persistence. An algebraic filtration need not imply graded sensitivity. Record
such dependencies explicitly rather than test every Cartesian combination.
Layers remain response-mode families, not processing stages or abstraction ranks.

## Initial comparison set

Prepare one small, fully specified candidate card for each entry below. These
are coverage requirements for the comparison, not four incompatible architectures.

| Candidate family | Definition required before its first experiment |
| --- | --- |
| Spectral/operator-based | Configuration space, operator, mode identity including degeneracy, collective coupling, response selection, arithmetic precision. Do not assume linear superposition suffices or disqualifies it in advance. |
| Finite algebraic | Elements and operations, interpretation of modes and nesting, joint interaction, observation and continuation semantics. No unconstrained truth-table expansion. |
| Relational/transformational | Structured configurations, compatibility and/or transformation descriptions, higher-arity scope, grading, and an actual response constructor. Binding and rewriting are examples in this family, not defaults for SRI. |
| Complex-based | Cells, incidence/boundary data, configurations carried by the complex, mode and nesting definitions, and a response law that actually uses this structure. Geometry or a hole count alone is not a model. |

Every card must include input/output interfaces, immutable evidence versus
internal interpretation, core/satellite roles, an observation-equivalence rule,
disturbance and edit metrics, bounds, controls, and a resource account. It must
separate collective admissibility, realized behavior, and persistent learning.
Mark missing components as unspecified rather than fill them silently with an
external solver or an arbitrary readout. Such a solver may be used, but its
contribution, assumptions, and cost must be included.

## Shared questions and controlled tests

1. **Collective distinction:** can two declared perturbation families yield
   distinguishable responses, and can another satellite's participation change
   an existing response under unchanged external evidence? Compare coupled and
   disabled-interaction controls with the same module inventory.
2. **Selective mediation:** what contribution comes from core structure, and what
   comes from the execution engine? Test enabled combinations and retained
   incompatibilities. Distinguish changed admissibility from changed reachability.
   Keep evidence, response meaning, and the generic engine fixed where possible.
3. **Persistence and recovery:** test driven response, behavior after evidence
   withdrawal, and recovery after a declared internal disturbance separately.
   Report surviving distinctions, recurrent behavior, schedule assumptions, and
   whether a result is bounded-time or eventual. Static admissibility earns no
   persistence credit without a continuation semantics.
4. **Grading and nesting:** test sensitivity to coupling changes and compare a
   matched ungraded or flattened variant when one is well-defined. Declare what
   strength orders. A layer label alone earns no explanatory credit.
5. **Topology:** when a candidate claims a topological contribution, vary relevant
   incidence while matching resources and local semantics as far as possible.
   Record any unavoidable confound. No topology claim is required merely to be
   a comparison candidate; absence is visible in its profile.
6. **Approximation:** compare bounded work or omitted interactions with a declared
   exact reference. Seek an observable-error bound under explicit assumptions,
   or retain a counterexample. Test weak but decisive interactions; low average
   error does not establish a worst-case guarantee.
7. **Construction cost:** measure response construction, matching/search,
   rejected candidates, memory, precision, and stopping checks, including external
   machinery. Distinguish compact representation from efficient realization.
8. **Structural extensibility:** state what a learning update could modify and
   what preservation obligation would result. Do not award learned reuse until
   a separate acquisition and held-out test demonstrates it.

Use common evidence and response tasks only where equivalent encodings and
decoders can be declared before results. Adapters must not perform the inference
being credited to the model. Compare encoding information and stored structure,
not just equal node counts or alphabet sizes. For native disturbances, publish
their meanings and magnitudes; a bit flip and a unit-norm perturbation are not
automatically equivalent. Where fair matching is unavailable, report separate
scoped results instead of a misleading ranking.

Within a family, change one modeling axis at a time where possible. Between
families, attribute results to the entire specified candidate unless a controlled
comparison isolates the contribution. A failed instance does not refute its
family, and a successful instance does not validate every choice it contains.

## Grading without invented certainty

For each question, publish the following separate fields:

| Field | Allowed content |
| --- | --- |
| Outcome | Pass, fail, mixed across declared cases, unresolved, or not claimed/not applicable, with justification. |
| Evidence | Scoped proof, exhaustive bounded check, measured sample, or untested. |
| Measurement | Distinction count, preservation/error measure, work and storage, or another predeclared quantity with units. |
| Assumptions | Configuration bounds, schedules, initial conditions, precision, disturbances, and completion status. |
| Explanation | Mechanism, ablation, counterexample, or unresolved attribution. |

Do not collapse these into one weighted score initially. In particular, unknown
is not failure, a theorem is not automatically broader than a measurement, and
not claiming topology does not constitute passing a topology test. Compare
trade-offs only over shared tasks and assumptions. Select a shortlist for further
work rather than declare a universal winner.

The existing role-binding example currently has scoped construction and
selectivity evidence, plus a same-evidence context witness. Persistence, recovery,
graded nesting, topology benefits, and learned mediation remain untested or
unspecified. Its runtime and success must not stand in for those missing grades.
No other candidate receives a score before it is specified and evaluated.

## Execution order and stopping rules

1. Write the four candidate cards, shared task definitions, native tests, and
   resource limits. Evaluate whether each is sufficiently specified to run.
2. Implement the smallest auditable prototypes and exact controls appropriate
   to each. Preserve theoretical impossibilities and failed cases, not only
   successful response tables. No single candidate gets an unlimited rescue search.
3. Populate the evidence matrix from proofs and reproducible reports. Keep
   `complete`, `truncated`, and `uncertified` outcomes distinct.
4. Refine choices whose failure mechanism suggests a specific falsifiable change.
   Promote a candidate only for named requirements it actually satisfies; retain
   alternatives with different trade-offs.

The immediate deliverable is the candidate cards and shared tests, followed by
bounded comparative implementations. A replacement OP-04 architecture should
emerge from this comparison rather than precede it. STM, dreaming, and abstraction
remain subsequent obligations, not bonuses assigned to a promising substrate.
