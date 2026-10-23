# Learned reciprocal participation v1 — agent implementation contract

**Status:** prospective, not implemented. Implement the bounded study below.
Preserve earlier code, contracts and reports. A negative feasibility or learning
result completes the study; do not broaden the model to manufacture a pass.

**Implemented (2026-10-23).** Exact dynamics, oracle and phase A ran as specified;
the gate failed (condition 3 for T1–T3), so phase B was not run. See
[results](learned-reciprocal-participation-v1-results.md). No amendment to the
rules below.

## 1. Intent

The user's hypothesis is that appropriate satellite participation emerges through
learned interference between an input satellite and the other satellites. Input
interpretation is revisable, and participating satellites feed back into it.
There is no separate router, expert ranking, top-k selection, or module-dropping
decision. All modules remain in the system throughout inference. An inactive
module is a state of that system, not an omitted constraint.

Read `interaction-semantics.md`, `core-graded-v1-results.md`, and
`selective-satellite-mediation-v1-results.md` first. The previous selective study
has an additional reduction: under a fixed hypothesis its output equals the
intersection of independently reachable A and B mode sets. Its gates depend on
external context, not the other satellite's evolving configuration. Do not carry
that separability into the claimed reciprocal mechanism.

This experiment asks two separate questions:

1. Can a declared coupled finite dynamics exhibit input–satellite feedback and
   core-dependent participation while retaining local constraints?
2. Can observations teach the interaction parameters well enough to reproduce
   held-out participation and response sets?

The specified signed energy is one finite candidate for interference. It is not
adoption of numerical energy minimization as the foundation of SRI. First prove
or refute feasibility in this candidate, then evaluate learning if justified.

## 2. Modules, evidence, and state

There are four modules: input I, core K, and specialized satellites A and B.

- I has two ternary interpretation coordinates `(i1,i2)`, each in `{-1,0,1}`.
- K has transient state `k in {-1,0,1}`.
- A and B each have one ternary mode `a,b`, in immutable local domains.
- Each satellite domain is one of `D0={-1,0,1}`, `D1={0,1}`, `D2={-1,0}`,
  in that order. Zero denotes inactive but available; nonzero denotes active.

The state is `z=(i1,i2,k,a,b)`, at most 243 configurations. The modules have
separate roles: a and b need not equal. Domains represent immutable internal
exclusions, including mutually exclusive signed modes and disallowed signs.
These are hand-declared constraints, not learned exclusions.

Evidence is `x=(e1,e2,D_A,D_B)`, with `e1,e2 in {-1,0,1}`. It never changes during
inference. Evidence biases I through the energy below; it is not an instruction
selecting A or B. The canonical initial state is `(e1,e2,0,0,0)`.

There are 81 queries: nine evidence pairs times nine domain pairs. Sort by
`(e1,e2,domain_A_index,domain_B_index)`. Satellite structures and domains are
identical across experimental controls. Domain constraints remain enforced when
a coupling is removed.

## 3. Learned structure and interaction law

Persistent parameters are six pairwise coupling coefficients:

`w=(w_1A,w_2A,w_1B,w_2B,w_KA,w_KB)`, each in `{-3,0,3}`.

Exactly 729 hypotheses exist, indexed lexicographically in that coefficient
order with `-3 < 0 < 3`. Use exact integer arithmetic:

```text
E_w(z;x) = 2*((i1-e1)^2 + (i2-e2)^2)
           + 2*(a^2+b^2) + k^2
           - w_1A*i1*a - w_2A*i2*a
           - w_1B*i1*b - w_2B*i2*b
           - w_KA*k*a - w_KB*k*b
```

Every coupling is reciprocal: the same term influences both endpoints' moves.
For example A's field is `w_1A*i1+w_2A*i2+w_KA*k`. Signed contributions can
reinforce or cancel. I's interpretation updates depend on satellite states.
The core changes its own transient state in response to A and B.

The activity penalties are fixed, not fitted. A field summation is an explicit
numerical interaction law; do not call it nonadditive merely because it is named
interference. Any nonseparability claim must concern the resulting dynamics or
response relation and have a counterexample to the stated independent control.

## 4. Exact dynamics and observations

An atomic update changes exactly one of the five coordinates to any other
locally allowed value, provided total energy **strictly decreases**. Even I's
two coordinates update separately. Ties do not move. Examine coordinates in
`i1,i2,k,a,b` order and proposed values in `-1,0,1` order for reproducible traversal;
do not use this order to discard other decreasing transitions.

Enumerate the entire reachable transition graph from the canonical start. A
terminal state has no permitted decreasing move. Report all reachable terminals
and their projections:

- `Y_w(x) = {(a,b) at reachable terminal states}`;
- full terminal configurations, including input interpretation and core state;
- participation signatures `({A if a!=0, B if b!=0})`;
- possible and necessary active satellites over the terminal set;
- shortest witness paths, terminal multiplicity, path lengths, and work counts.

Different terminal outcomes are schedule sensitivity, not permission to select
the convenient one. The primary prediction is the complete set Y, never a
minimum-energy terminal selected after the fact. Inactivity is not automatically
success, and a terminal is not a proof of task correctness.

Strict descent on a finite state space proves termination for sequences of
actual updates. Do not infer convergence for schedulers that indefinitely refuse
to execute an available move. Do not claim recovery or schedule-independent
output merely from termination.

Use a candidate-edge inspection cap for optional diagnostics. Truncation returns
unknown completeness and no certified exact response set. Main runs are uncapped.
Separate discoveries from certified results. Evidence and local constraints
must be preserved in every intermediate state, not only at termination.

## 5. Phase A: mechanism feasibility before learning

Use these four fixed teachers; no teacher search or parameter tuning:

| Teacher | w_1A | w_2A | w_1B | w_2B | w_KA | w_KB |
| --- | --- | --- | --- | --- | --- | --- |
| T0: independent control | 3 | 0 | 0 | -3 | 0 | 0 |
| T1: complementary core | 3 | 0 | 0 | -3 | 3 | -3 |
| T2: matching core signs | 3 | 0 | 0 | -3 | 3 | 3 |
| T3: crossed input associations | 0 | 3 | -3 | 0 | -3 | 3 |

The labels describe coefficients, not expected outcomes. Determine what they
actually do. For all 81 queries and all four teachers, require these diagnostics:

1. **State-conditioned influence:** give reachable-state witnesses where changing
   a coupled neighbor, while holding evidence and other coordinates fixed,
   changes whether an A/B move decreases energy. Also give the reverse influence
   on an input-coordinate move. State whether both witness states are reachable
   from the same query; only such pairs count for the mechanism gate.
2. **Realized input revision:** some reachable terminal changes I from its initial
   interpretation, without changing evidence. Report both possible revision and
   revision at every terminal separately.
3. **Feedback necessity:** Y differs from the input-frozen control on at least one
   query. Count changes to participation separately from changes only to sign.
4. **Core contribution:** Y and participation differ on at least one query when
   core couplings are removed. Compare K fixed at each of -1,0,1 as well; determine
   whether one fixed k reproduces all 81 response sets. A query-wise choice of k
   is an oracle, not a fixed-core control.
5. **Selective participation:** across queries, obtain at least two distinct
   nonempty participation signatures. Report whether any queries determine a
   unique signature across all terminals, and whether a mixed-sign `(a,b)` with
   both active occurs. No equality requirement may be added.
6. **Local exclusion preservation:** zero disallowed signs or domain mutations.
   Separately identify cross-satellite response pairs excluded by the teacher;
   these are dynamical exclusions, not a blanket global incompatibility theorem.
7. **Cross-satellite influence through K:** for each query, additionally clamp B
   at each value in its domain, initialize it to that value, and compute A's
   terminal mode set with I and K free. Find a pair of B clamps giving different
   A sets. Repeat in the other direction. Hold evidence and domains fixed within
   each comparison. Re-run these interventions with both core couplings removed;
   the witnessed difference must disappear to count as core-mediated influence.
   Record these as interventions, not canonical-start trajectories. A clamp
   imposes a local state, never a forbidden value or a changed domain.

The phase-A gate requires a single teacher among T1–T3 to satisfy 1–7, with both
possible and necessary outcomes reported (only possible revision is required by
item 2). Item 4 additionally requires that no one fixed k reproduces all queries.
For item 5, at least two queries must each have a unique nonempty signature, and
those signatures must differ. The other requirements can use different queries.
Publish all teachers' results, including failed conditions.

If no teacher qualifies, STOP after the validated phase-A report and exact failure
certificates. Do not run a learning campaign on a mechanism that fails this gate.
The completion status is `mechanism_refuted_in_declared_family`, not agent failure.
Do not remove the core, alter penalties, add teachers, or relax the gate to pass.

## 6. Controls and separability checks

Implement these controls with the same evidence and local domains:

- **Uncoupled:** all six coefficients zero.
- **Core disconnected:** set only w_KA and w_KB to zero; K may still settle.
- **Input frozen:** fix I to `(e1,e2)`, retain other couplings.
- **Core fixed:** fix k separately to -1,0,1, initialize it to that value, and
  update the remaining coordinates. State this initialization intervention.
- **Input feedback removed:** allow I to update using only its evidence penalty;
  other coordinates retain their original local energy differences. From the
  canonical start I stays at evidence, so equality with input-frozen is expected;
  report that equivalence rather than treating these as independent controls.
- **Independent satellite reachability:** fix I at evidence, remove K couplings,
  evolve A and B separately, and take the Cartesian product of local terminal
  mode sets. Compare with the full Y. A mismatch refutes only this scoped control.

Additionally prove that A's local move rule depends on k and I when relevant
coefficients are nonzero. Merely executing A and B in one graph is insufficient.
For the fixed-input independent control, prove factorization analytically.

No claim that an arbitrary core-free algorithm cannot simulate the system is
allowed. A general transition table can reproduce finite dynamics; report its
storage and semantics if discussed. The target here is reciprocal mechanism,
not universal computational uniqueness of a named module.

## 7. Phase B: learn the interactions

Run only if phase A passes. Run all four teachers, retaining T0 as a negative
mechanism control; never discard a difficult teacher from learning results.

Feedback for a training query is its complete terminal output set `Y_T(x)`.
The learner receives neither teacher coefficients nor trajectories, terminal I/K
states, nor module relevance labels. This is strong outcome supervision within a
realizable hypothesis class. Teacher imitation is not yet an externally grounded
semantic task or autonomous discovery of what a satellite should know.

### Exact identification reference

Maintain the 729-bit version space V, initially all hypotheses. Update by
`V <- {w in V : Y_w(x) == feedback}`. Persist V, not training examples. An empty V
is `inconsistent_feedback`; do not reset or silently pick an approximate model.

Primary predictions are agreement or uncertainty over V:

- certain outputs = intersection of Y_w over V;
- possible outputs = union of Y_w over V;
- certified response set only if all survivors agree;
- otherwise `uncertain`, including when the certain set is empty.

There is no smallest-ID executable-hypothesis selection. This removes the prior
that masked exclusion learning in the previous study. Evaluate exact predictions
for every survivor as a distribution, but do not cherry-pick one for the headline.
Report possible/necessary participation over both hypothesis and terminal-state
uncertainty, and distinguish those two sources.

Required ablation: positive-only filtering `feedback subset Y_w(x)`. Compare
coverage, possible-output false positives, version-space size and uncertainty,
not a conservatively selected single hypothesis.

### Local structural-update candidate

This is a separate, more restrictive learner, not a substitute for the reference.
Persist one six-coefficient vector, initially all zero. Define the discrepancy on
the current feedback item as `loss=|Y_w(x) symmetric_difference feedback|`.

Allow edits within one of three blocks: `(w_1A,w_2A)`, `(w_1B,w_2B)`, or
`(w_KA,w_KB)`. A candidate changes one block to one of its other eight assignments:
24 neighbors, plus unchanged. Evaluate all neighbors using the same exact dynamics.
Choose the candidate minimizing `(loss, number_of_changed_coefficients,
lexicographic_coefficient_tuple)`, but commit only if its loss is strictly lower
than current loss. Make at most one block update per example, with one pass over
the stream. No replay, retained examples, hidden teacher access, or parameter
search beyond those neighbors.

This is local in parameter-edit support, **not local in credit information**:
it uses global output error and exact inference. Say so explicitly. Its failures
to learn coupled changes or retain earlier responses are important results.
Do not describe it as a biological or established SRI plasticity law.

## 8. Frozen train/test protocol

Training contexts are axis evidence pairs `(-1,0),(0,-1),(0,0),(0,1),(1,0)`,
in that order. Mixed evidence pairs `(-1,-1),(-1,1),(1,-1),(1,1)` are held out.

For split seeds 0 through 4, shuffle the nine domain-pair queries of each training
context with `random.Random(1000*split_seed + context_index)`, taking the first
three. Thus there are 15 training queries, 30 within-context held-out queries,
and 36 mixed-evidence held-out queries. Selection must use no labels.

For order seeds 0 through 4, sort training query IDs then shuffle with
`random.Random(10000*split_seed + order_seed)`. Checkpoints are 0,5,10,15.
Freeze a manifest containing every ID before evaluating learning. Orders are
repeated streams over the same split, not independent datasets.

Phase A inspects the finite mechanism space, including mixed queries; therefore
describe phase-B holdout as **withheld from the learner**, not as data unseen by
the researchers. No protocol may change after phase-A or phase-B metrics are seen.

Compare full-feedback reference, positive-only reference, local-update learner,
exact-query memory and zero-coupling baseline. Memory returns stored sets for seen
queries and `unknown` for unseen queries, not an invented negative response.
The teacher is a separately labeled upper reference. Core/input ablations from
phase A remain available for the learned local model and each surviving hypothesis.

## 9. Metrics, success criteria, and interpretation

For each teacher, partition, split, order and checkpoint report:

- certified response-set coverage and accuracy conditional on certification;
- possible-output precision and certain-output recall, with counts and zero
  denominators represented as null; uncertainty is not an incorrect empty output;
- local learner exact-set accuracy, output TP/FP/FN, and participation-signature
  accuracy separately from signed-mode accuracy;
- rates of no activity, single-satellite and joint activity;
- terminal multiplicity and input revision;
- observed-response retention after every update (evaluator histories only);
- gains from negative information versus the positive-only reference;
- survivor count, coefficients changed, evaluations, inspected transitions,
  persistent logical storage, actual memory method, and total wall time including
  preprocessing. Separate inference, learning, and evaluator caches.

Correctness gates: no invalid state/evidence changes; exact reference retains the
teacher and all previous feedback; certain subset teacher subset possible; no
cycles in full strict-descent dynamics; deterministic regeneration; oracle tests.

Prospective identification target: for each phase-A-qualifying teacher, at final
checkpoint certify at least 80% of each held-out partition, pooled over splits
(orders have equal final V and must not be counted as new datasets), with no
incorrect certified set. Report worst split. Accuracy given certification follows
from realizability and is not by itself a learning achievement; coverage is key.

Prospective local-learning target: on each qualifying teacher/partition, at least
80% exact-set accuracy, strict improvement over zero coupling, and at least 95%
exact retention of previously observed items after updates. Report all conditions
separately. No retrospective threshold changes. Failure of local learning does
not invalidate exact identification, nor vice versa.

For negative-information benefit, report whether full feedback strictly increases
certified coverage or strictly reduces false possible outputs. No difference is
a valid negative result. Constructed local-domain safety receives no credit as
learned exclusion. Learned dynamical exclusions must be demonstrated separately.

## 10. Implementation, proofs, and tests

Use standard-library finite reference code and unittest. Add:

- `src/reciprocal_participation.py`;
- `experiments/reciprocal_participation.py`;
- `tests/test_reciprocal_participation.py`;
- `docs/learned-reciprocal-participation-v1-results.md`;
- versioned phase-A, split, phase-B (if run), and runtime JSON reports under
  `reports/learned-reciprocal-participation-v1-*`.

Implementation order: exact energy/state semantics; independent oracle and
proofs; frozen manifests; phase A and gate; then phase B if allowed. Do not run
the previous study again or overwrite historical artifacts.

Prove finite strict-descent termination, reachable-terminal completeness,
constraint preservation, reciprocal local-field dependence, reference teacher
retention/order independence, and conditional certification soundness. Derive
the fixed-input disconnected factorization explicitly.

The independent oracle should enumerate allowed states and edges directly from
the written energy formula, without production energy/successor helpers. Compare
all 729 hypotheses on all 81 canonical queries (59,049 inference cases), using
exact dynamic programming on descending energy or independent graph traversal.
Include controlled tests for ties, truncation, contradictory feedback, simultaneous
opposing field contributions, input revision, core clamps, and all control
invariants. If a requested positive witness does not exist in the declared space,
retain its exhaustive absence as a finding rather than fabricate a fixture.

Test version-space learning on the declared streams and the block learner's
one-update/no-replay behavior. Never test only selected terminal representatives.
Run the full existing test suite after new tests pass.

Keep volatile timestamps/timings apart from deterministic scientific reports.
Record source hashes, interpreter, git revision and dirty status. Start timing
before preprocessing. Provide reproduction commands and byte-identical
regeneration verification using temporary output paths. Update README and the
open-problems register only with the scoped findings actually established.

## 11. Scope and stopping rule

This is a mode-level interaction-and-learning study with small signed numerical
couplings. It does not establish learned module contents, unit-level recovery,
topological benefits, STM, dreaming, CPU superiority, or semantic intelligence.
Even successful local updates use global supervised error. Defer those claims.

Do not introduce a router, choose experts externally, drop conflicting modules,
or conceal unresolved behavior through favorable terminal selection. Do not
increase the grid after a failure. The deliverable is evidence about this precise
candidate, including a clean failure if it cannot realize the intended mechanism.
