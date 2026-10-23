# Selective satellite mediation v1 — implementation handoff

**Status:** prospective specification; no results claimed. Implement this bounded
study, run its declared grid, and report negative results without enlarging the
search. This is a finite mechanism probe, not selection of an SRI architecture.

**Implemented (2026-10-23).** The declared grid ran once as specified; see
[results](selective-satellite-mediation-v1-results.md). No amendment to the rules
below. A reporting-only addition after the first run lists the matched controls in
the target table; no learner, split, threshold or generator changed.

## 1. Question and relationship to earlier work

Can learned, context-dependent transformations resolve conflicts between
satellites while preserving their internal constraints and leaving genuinely
unresolvable conflicts unresolved? Can the learned rules transfer to new
combinations of familiar states and contexts?

Read `docs/interaction-semantics.md`, `docs/mediation-example.md`, and
`docs/learned-compatibility-v1-results.md` before implementation. Existing studies
and frozen reports must remain unchanged.

The earlier compatibility learner completes connected components and loses
meaningful exclusions on overlapping relations. This experiment instead learns
which transformations are permitted. It measures actual changes in satellite
states. It does **not** claim to expand the set of structurally admissible states:
mediation changes reachability within fixed constraints.

The equality target below deliberately supplies a minimal agreement task. Success
does not distinguish general compatibility mediation from constrained agreement.
It establishes a prerequisite: selective learned repair under immutable local
constraints. Richer relations and graded dynamics are subsequent studies.

## 2. Fixed mathematical model

Two satellites A and B each have four modes, integers 0 through 3, interpreted as
two-bit vectors. A satellite occupies exactly one mode at a time. Its immutable
allowed domain D_i is one of these seven sets, in this order:

`{0}, {1}, {2}, {3}, {0,1}, {0,2}, {0,1,2,3}`.

A query is `x = (D_A, D_B, a0, b0, c)`, where `a0 in D_A`, `b0 in D_B`, and
context `c=(c0,c1)` is one of `00,01,10,11`, in that order. Context coordinates
are named explicitly; do not infer their meaning from integer endianness.

Immutable evidence comprises the domains, starting modes, and context. The
starting modes remain recorded in the query while transient modes `(a,b)` may
change. The external context is supplied, not learned or inferred.

Local validity is `a in D_A and b in D_B`. A successful joint response additionally
requires `a == b`. Unequal states are valid transient **conflicts**, not successful
outputs. No inference or learning operation may modify a domain, context, or
the equality requirement.

Four primitive transformations, in this canonical order, are available:

1. `A_low`: replace a with `a XOR 1`.
2. `A_high`: replace a with `a XOR 2`.
3. `B_low`: replace b with `b XOR 1`.
4. `B_high`: replace b with `b XOR 2`.

A transformation is executable only if its gate is true in context c AND the
destination remains locally valid. Each gate belongs to the fixed vocabulary
`never, c0, c1, always`, in that order. A hypothesis h assigns one gate to each
primitive: exactly `4**4 = 256` hypotheses, lexicographically indexed.

Persistent core structure is a learned hypothesis/version space over these
gates. A transient core certificate records the selected hypothesis ID and the
primitive used at each step. This is a rule controller and certificate, not a
demonstrated autonomous recurrent core. State that limitation in the results.

## 3. Exact realization and response semantics

For a query and hypothesis, construct the reachable graph from `(a0,b0)` using
the executable transformations. There are at most 16 states. Use breadth-first
search and the canonical primitive order. Expand successful states too: the
observable is all reachable successful states, not only first-hit terminals.

Define `R_h(x) = {t : (t,t) is reachable}`. Compute this set completely. When a
single representative is requested, choose its smallest t and return the first
BFS shortest path to it. The path must list real satellite states and gates;
relabeling an unchanged state does not count as repair.

Return separately:

- execution status: `complete` or `truncated`;
- feasibility: true/false for complete results, null for truncated results;
- exact response set for complete results; partial discoveries in a separately
  named field when truncated;
- representative and path, only committed after complete inference;
- visited-state count, candidate-edge checks, and maximum frontier size.

For optional caps, one unit of work is examining one of the four primitives at a
dequeued state, including disabled/invalid primitives. Check exhaustion before
the next examination. A zero cap is truncated, including an initially successful
query, because the complete response set is not certified. If all work finishes
exactly at the cap, status is complete. Never call a capped search infeasible.
The main experiment is uncapped.

No fairness, fixed-point, disturbance-recovery, or local-convergence claim follows
from this centralized BFS. Cycles are handled by visited-state tracking.

## 4. Ground truth and incompatibility classes

Use four teacher hypotheses, written in primitive order:

| ID | A_low | A_high | B_low | B_high |
| --- | --- | --- | --- | --- |
| T0 | c0 | never | c1 | never |
| T1 | c0 | c1 | c1 | c0 |
| T2 | always | never | never | c0 |
| T3 | never | never | never | never |

The evaluator, never the learner, receives the teacher. Ground-truth responses
are exact `R_teacher(x)`. Classify queries using separate fields:

- already compatible: `a0 == b0`;
- resolvable initial conflict: `a0 != b0` and teacher response set nonempty;
- unresolved initial conflict: `a0 != b0` and teacher response set empty.

For unresolved conflicts, distinguish a disjoint-domain obstruction, an
obstruction even with all gates enabled, and a teacher-permission obstruction
that the all-enabled control can resolve. These are exclusive classes, tested
in that order. Preserve domain-invalid attempted transitions as separate
diagnostics; they must never become committed states.

“Unresolvable” always means relative to this fixed generator set and teacher
permissions. It is not a universal statement about other architectures.

## 5. Learning and supervision

Use exact finite version-space learning as the primary reference learner.
Initially `V = all 256 hypotheses`. Each training feedback item supplies a query
and its **complete valid response set**, including the empty set when appropriate:

`V_next = {h in V : R_h(x) == feedback}`.

This is strong supervised feedback: it provides both valid and invalid outputs.
Do not describe it as learning from positive pairs alone, sparse rewards, or
autonomous experience. The hypothesis language and generators are hand-designed.

After each update choose the lexicographically smallest h in V as the executable
model. If V becomes empty, return `inconsistent_feedback`, commit no replacement
model, and fail the clean-teacher experiment. Do not silently reset V.

Persist V as a 256-bit membership structure (plus fixed schema/version metadata).
The learner need not retain past query/label records. Report actual Python memory
separately from this logical bit count. Teacher tables, evaluation caches and
archived histories are instrumentation and cannot be accessible to learning.

For evaluation also calculate:

- `certain(x) = intersection of R_h(x) over h in V`;
- `possible(x) = union of R_h(x) over h in V`;
- whether every survivor predicts the same set.

Keep these separate from the selected model's executed response. If certain is
empty but possible is nonempty, the learner is uncertain, not certain of failure.
All-survivor emptiness is a certified infeasibility prediction only conditional
on the teacher lying in the declared hypothesis class and clean feedback.

Required ablation: positive-only updates retain h when `feedback subset R_h(x)`;
empty feedback makes no change. Use the same examples, selection rule and budget.
This isolates information supplied by exclusions. Its preservation claim is only
for observed positive outputs, not for complete earlier response sets.

## 6. Frozen split and sequential protocol

Enumerate domains in the order above and modes in ascending order, then form
queries sorted by `(context_index, domain_A_index, domain_B_index, a0, b0)`.
There are 12 domain/start choices per satellite, 144 queries per context, and
576 queries per teacher.

For each split seed in `0..4`:

1. For each context `00,01,10`, shuffle its 144 query indices with
   `random.Random(1000*seed + context_index)` and take the first 16.
2. Their union is the 48-query training set, chosen without labels.
3. The other 384 queries in those three contexts form within-context transfer.
4. All 144 queries in context `11` form unseen-context-combination transfer.
5. Sort the training set canonically, then shuffle a copy with
   `random.Random(10000*seed + order_seed)` for order seeds `0..4`.

Run each teacher, split and order: 100 runs per learner. Evaluate checkpoints
`0,12,24,48`. At each checkpoint evaluate both held-out partitions and all seen
training queries. Do not train on held-out diagnostics. Record a hash of the
split manifest and freeze that manifest before computing learning metrics.

The teacher remains fixed during sequential exposure. This is preservation under
new evidence, not nonstationary learning. Version-space elimination should retain
all earlier complete feedback exactly; changing the selected hypothesis can
still change earlier **unobserved** predictions. Measure both separately.

No sample rejection, seed selection, additional generator, or gate-vocabulary
change is allowed after results are seen. Empty metric strata are reported as
null with denominator zero, never as perfect scores.

## 7. Required matched controls

All controls receive identical evidence, domains, generators, response semantics,
and training stream except where the declared ablation explicitly differs.

1. **No mediation:** all gates never; only the starting state is reachable.
2. **Permissive repair:** all gates always; still enforce local domains. This
   tests loss of teacher-specific incompatibilities, not just domain safety.
3. **Exact-query memory:** store feedback keyed by the entire query; on an unseen
   query use no-mediation response. It may have nonzero transfer performance on
   already compatible or impossible queries; do not assert zero recall by design.
4. **Core-free rule learner:** use the identical V and gate selection, but execute
   gates directly without a distinguished core/certificate object. Its response
   sets must equal the primary learner's. This is an equivalence control, not a
   weak baseline expected to lose.
5. **Positive-only learner:** as specified above.
6. **Teacher oracle:** compute achievable responses; clearly mark as an upper
   reference with access to hidden rules, not a learner.

The core-free equivalence is expected and should be proved. This study cannot
establish a distinct-core advantage. A future architecture claim needs a further
resource, locality, dynamics, or transfer distinction that this control lacks.

## 8. Metrics and preregistered interpretation

Report per teacher, split, order, checkpoint, held-out partition and conflict
class; aggregate only with explicit denominators:

- Exact response-set accuracy.
- Micro output precision and recall, using summed TP/FP/FN over mode outputs;
  undefined zero denominators are null.
- False-feasible rate on teacher-empty queries; missed-feasible rate on
  teacher-nonempty queries.
- Exact repairs of resolvable initial conflicts; include shortest path length,
  and whether A, B, or both changed in the selected witness.
- Domain violations, context/evidence mutations, and gate-invalid path steps.
- Exact retention on observed training items; changes and correctness changes
  in checkpoint-0 held-out predictions, without using them as feedback.
- Size of V, survivor agreement fraction, certain/possible set sizes, and
  uncertainty frequency.
- Logical persistent storage, actual memory measurement method, hypothesis
  evaluations, edge checks, and runtime. Separate evaluation/oracle/cache cost
  from learner update/inference work.

**Correctness gate:** zero hard-constraint/path violations; exact observed-item
retention for the full-feedback learner; teacher never eliminated; core-free
equivalence; all exactness tests pass. A failure invalidates performance claims.

**Transfer target:** at checkpoint 48, for each of T0–T2 and each held-out
partition, pooled over all 25 split/order runs: at least 90% exact-set accuracy,
zero false-feasible predictions, and strictly more exactly repaired resolvable
queries than exact-query memory. Report each condition separately. T3 is the
negative control: require zero false-feasible predictions, without assigning a
repair score to its empty repair stratum. Missing strata do not pass a target.

These thresholds are prospective targets, not expected findings. Failure is a
valid study result; do not tune to pass. Report distributions and worst runs so
pooled values do not conceal order/split failures. Safety by domain checks is a
constructed invariant, not learned selectivity. Selectivity concerns additional
teacher-permission exclusions that the permissive control can violate.

## 9. Proof obligations and meaningful tests

Write short scoped proofs, backed by independent tests:

1. BFS soundness, termination and completeness for finite reachable response sets.
2. Every committed path preserves local constraints and immutable evidence.
3. Clean realizable feedback retains the teacher in V.
4. Full-feedback elimination preserves all earlier observed response sets for
   every survivor; update intersection is order-independent for the same dataset.
5. `certain subset R_teacher subset possible` under those assumptions.
6. The core-free rule learner is observationally equivalent.

Test BFS against an independently implemented adjacency/transitive-closure oracle
on all 256 hypotheses and 576 queries (147,456 small cases). The oracle must not
call production successor or BFS helpers. Cache only pure inference computations;
do not obscure cache misses in work accounting.

Retain explicit fixtures: already compatible; repair requires A; repair requires
B; repair requires both; disjoint-domain impossibility; permission-only
impossibility; shared domain but unreachable intersection; several valid outputs;
different teachers indistinguishable on a training subset but different on a
held-out query; contradictory feedback; zero and exact-boundary budgets.

Fixtures may be derived analytically or by bounded exhaustive enumeration inside
the frozen space. Their existence is not evidence that all learned models attain
them. Also test input validation, deterministic serialization, split disjointness,
absence of context 11 from training, no label access in sampling, and identical
final V across orders of the same dataset.

## 10. Deliverables and execution order

Implement with the standard library and repository unittest conventions:

- `src/selective_mediation.py`: immutable query/schema, hypotheses, exact inference,
  learner, controls and metrics. Keep teacher/evaluator access out of learner APIs.
- `experiments/selective_mediation.py`: deterministic manifest, runner and reports.
- `tests/test_selective_mediation.py`: independent oracle and tests above.
- `reports/selective-satellite-mediation-v1-splits.json`: frozen split manifest.
- `reports/selective-satellite-mediation-v1.json`: configurations, complete rows,
  aggregates, witnesses/counterexamples, status and provenance.
- `reports/selective-satellite-mediation-v1-runtime.json`: volatile timings and
  machine/runtime details, separate from deterministic scientific output.
- `docs/selective-satellite-mediation-v1-results.md`: results, proofs, failures,
  scope discounts and next obligations.

First implement the semantics and oracle tests, then freeze the manifest, then
implement learning and controls, then run the frozen grid. Do not edit historical
reports. Record Python version, command, git revision, dirty-worktree status, and
hashes of the relevant source/contract files. Scientific output must regenerate
identically with the same code/interpreter; keep volatile provenance separate.

Reproduction commands:

```bash
.venv/bin/python -m unittest tests.test_selective_mediation -v
.venv/bin/python -m experiments.selective_mediation
.venv/bin/python -m unittest discover -s tests -v
```

If the repository's tests directory does not support dotted module loading, use
`unittest discover -s tests -p 'test_selective_mediation.py' -v` instead.

After results are validated, add a concise README link and evidence entry to
`docs/open-problems.md`. Mark this contract implemented without rewriting the
prospective rules. Corrections require a dated amendment retaining original text
and explaining whether any run preceded the amendment.

Stop after this grid. No dreaming, online topology changes, learned domains,
multilayer/graded realization, additional satellites, noisy-feedback campaign,
benchmark claims, or implementation of a general intelligence system is in scope.
If successful, the next experiment should test non-equality compatibility or
realize the learned transformations inside graded satellite dynamics, retaining
these exclusions and transfer tests.
