# Local update diagnosis v1 — why the block learner fails under IB supervision

**Status:** prospective; frozen for implementation on 2026-10-23. Preserve every
earlier contract, code file and report unchanged, including all
local-learning-supervision v1 and learned-input-mediation v1 artifacts and the
frozen v1 split manifest. Freeze this contract's hash in the new manifest before
any computation.

**Pre-implementation review (2026-10-23).** This revision corrects the replay
preservation claim, makes loss-barrier diagnostics exact within their stated
scope, includes repetition's stopping-check cost, and distinguishes experience
outcomes from variant decision-level classifications. The variants, teachers,
supervision and target thresholds are unchanged. No diagnosis, new simulation
or rule-variant run was performed during review.

## 1. Question

> Under IB supervision, which updates of the block-local learner fail, and why:
> because no permitted edit improves the current example, because improving
> edits break earlier behaviour, or because progress needs moves that are
> neutral or worse on the current example?

In [local learning supervision v1](local-learning-supervision-v1-results.md),
the exact learner identified every teacher from IB's 15 experiences. The block
learner did not: it recovered S1 in 5/25 runs and C2 in 7/25. The stream
contains enough information, so the failure lies in how the rule uses it. This
study first diagnoses the existing IB trajectories (phase A). It then runs a
small set of rule variants, each changing one property, declared here before
any diagnosis (phase B). Phase A supplies hypotheses for interpreting those
outcomes; it does not
select which variant to run. These diagnoses establish properties of the tested
states, objectives and neighborhoods, not a unique cause of a whole-stream failure.

Three failure types are separated:

| Type | Meaning |
| --- | --- |
| **Restricted edits** | No single-block edit lowers the current example's loss, but a larger edit would |
| **Preservation failure** | An improving edit exists, and the committed one breaks previously correct items |
| **Current-example loss barrier** | Under a frozen current example and single-block transitions, the teacher class is unreachable by strict descent, or even by non-increasing moves |

The study also measures **under-updating**: whether more updates on the current
example would have helped.

Replay appears only as a diagnostic comparator. The target architecture keeps
retention within learned structure, without an external experience store. A
replay result shows what access to earlier feedback would buy; it is not a
proposed mechanism.

## 2. Knowledge at declaration (disclosure)

- All local-learning-supervision v1 outcomes are known to the researchers:
  every IB stream's checkpoint vectors, per-item own-map and B1 timelines, final
  vectors, and the timeline categories (S1: 117 never acquired, 119 forgotten;
  C2: 101 never acquired, 52 forgotten, of 375 each).
- No per-update diagnosis has been computed. No candidate neighbourhood was
  inspected beyond what the v1 learner evaluated in passing, and no rule variant
  has been run.
- "Cross-block coordination is needed" is a hypothesis. The restricted-edit and
  short-sighted diagnostics test it directly; it is not assumed.

## 3. Fixed setting

Everything is as local-learning-supervision v1 condition IB: the battery,
2,187-vector space, teachers S1, C2, S2 and T1, the frozen splits and orders,
15 axis training queries per stream, IB feedback (B1, B6x, B7x), the projected
simulation service, partitions `within` and `mixed24`, and the evaluator. Import
them; do not modify them. S1 and C2 are primary. S2 and T1 are reported and not
scored.

Notation for one stream: `w_{t-1}` is the learner's state before experience t,
`(x_t, f_t)` the experience, and `ℓ_t(w)` the v1 loss of w on that experience
(sum over keys of the symmetric difference, after projection). **Previously
correct items** at step t are the earlier training items that `w_{t-1}`
reproduces on the own map. The **teacher class** is the teacher and its
core-sign mirror. `d(w)` is the number of the four coefficient blocks in which w
differs from the nearer member of the teacher class (0–4).

Neighbourhoods:

- `N1(w)`: the v1 neighbourhood, 26 single-block edits.
- `N2(w)`: all edits changing exactly two blocks, 240 of them (3 pairs of
  two-coefficient blocks × 64, and 3 pairs with `w_12` × 16).
- The full space: all 2,187 vectors, for exhaustive bounds only.

Use the four unchanged blocks `(w_1A,w_2A)`, `(w_1B,w_2B)`, `(w_KA,w_KB)` and
`(w_12)`. Let `N≤2 = N1 ∪ N2`. A single-block local minimum has no strictly
lower-loss neighbor in N1; an up-to-two-block local minimum has none in N≤2.
Ties are allowed at a local minimum. An up-to-two-block local minimum is therefore
always a single-block local minimum, not conversely. Neither implies a global
minimum or a strict minimum.

The teacher class and d are evaluator-only and must be verified against the
frozen complete-information IB class. d is an exact shortest **unconstrained**
single-block edit distance to that class, but it is not a task loss, a measure
of semantic quality, or a necessary monotone progress measure for learning.
Use the unchanged local-supervision coverage manifest to obtain `mixed24`; do
not resample coverage or retrain on any mixed-evidence query in this study.

## 4. Phase A — diagnosis of the existing IB trajectories

**Reproduction first.** Rerun the v1 IB block learner. Every per-step state, the
checkpoint vectors and the per-item timelines must equal the frozen v1 learning
report wherever those values were archived. Checkpoint and final vectors, row
values, work counters and timelines are archived; if an intermediate coefficient
vector was not archived, do not claim a direct frozen comparison for it. Verify
such intermediate states by lockstep execution of the unchanged v1 learner and
the diagnosis harness on the same feedback. Then freeze the reproduced per-step
trace and diagnose it. Never fill missing states by inference from correctness
timelines, which need not uniquely identify a coefficient vector.

**Evaluator boundary.** Every diagnostic is computed from the battery by
evaluator code, after the learner's update and outside it. Previously correct
items, teacher identity and `d(w)` are evaluator knowledge. Nothing computed in
phase A feeds into any learner, in phase A or phase B. Diagnostic loss
evaluations are counted separately from learner work.

### 4.1 Per-experience classification

For each experience t, with `ℓ0 = ℓ_t(w_{t-1})`, assign exactly one top-level
class:

- **E0, already exact on this example:** `ℓ0 = 0`. No update lowers this example's
  loss, although the model may still be wrong elsewhere. Report separately
  whether `w_{t-1}` is in the teacher class.
- **E1, stuck:** `ℓ0 > 0` and no edit in `N1` has lower loss. Subclasses:
  - **E1a:** some edit in `N2` has lower loss (a restricted-edit case);
  - **E1b:** none in `N2` does, and some vector in the full space has lower loss
    (an improving endpoint differs in at least three blocks). Report the minimum
    number of blocks differing from the current state among all lower-loss
    vectors. This is a direct-edit bound, not proof that every sequence of
    single-block edits fails. The clean teacher has zero current loss, so for
    ℓ0 > 0 some full-space improvement must exist; absence is a correctness error.
- **E2, committed:** `ℓ0 > 0` and the learner commits an improving edit.
  Subclasses by preservation, on the own map:
  - **E2a:** the committed edit breaks no previously correct item;
  - **E2b:** it breaks at least one, but some other improving edit in `N1`
    breaks none (selection by current loss ignored preservation);
  - **E2c:** every improving edit in `N1` breaks at least one previously correct
    item (a preservation conflict within single-block edits).

  For E2, also report whether the committed edit makes the current example
  exact, and how many items it breaks. For E2b, report the best current loss
  among preserving improving edits versus the selected edit's current loss.
  Equality indicates a tie-selection opportunity; a higher preserving loss
  indicates a current-example/preservation trade-off. Distinguish these rather
  than attributing both to the lexicographic tie-break.

An experience is **failing** if, after the update, the current example is not
exact or a previously correct item is broken. Report every class and subclass
as counts and as fractions of all experiences and of failing experiences.
For the latter fraction use `count(class AND failing)/count(failing)`; a class
such as E2a contains both failing and non-failing experiences. Empty denominators
are null. Also report E0 outside the teacher class separately: it is not a failed
update under this definition even if the final mechanism will be wrong.

### 4.2 Progress toward the teacher class

For each experience with `w_{t-1}` outside the teacher class:

- the change in `d` caused by the actual update (−, 0, +);
- whether any `N1` edit reduces `d`, and for those, the change in current loss
  of each: improving, neutral or worse. Report the best such edit's category and
  the counts;
- the same for `N2` edits that reduce `d` by two.

"Best" here means lowest current loss, with the existing changed-coefficient and
lexicographic tie-break. These are proximity diagnostics only. A valid descending
path might temporarily increase d; unfavorable teacher-ward edits do not prove
that neutral or worsening moves are necessary.

To test necessity exactly, fix the current query/feedback and construct two
graphs over all 2,187 vectors with N1 adjacency:

- **G↓:** edge `w→u` iff `ℓ_t(u) < ℓ_t(w)`;
- **G≤:** edge `w→u` iff `ℓ_t(u) ≤ ℓ_t(w)`.

For each evaluated state outside the teacher class, report one of:

1. Teacher class reachable in G↓: some strict-descent route exists. If the
   deterministic repetition probe does not reach it, the greedy selection loses
   a possible route under this fixed example.
2. Unreachable in G↓ but reachable in G≤: at least one neutral move is necessary
   along a non-increasing route; no worsening move is required for existence.
3. Unreachable in G≤: every N1 path to the class includes a loss-increasing move.

The unrestricted N1 graph is connected, so a path always exists if loss
restrictions are removed. Compute reachability exactly, including equal-loss
cycles in G≤; memoize by teacher/query rather than rebuilding for each stream.
Archive a shortest witness path for positive reachability and reverse-reachable
sets or their reconstructible hashes for negative certificates. No rollout cap
or selected-path failure may substitute for non-reachability.

These are bounds for **one example held fixed**, not for learning across the
stream. The example need not identify the teacher; its zero-loss set can contain
other mechanisms. Later experiences change the objective and can permit a move
blocked here. Do not interpret the graph categories as a proof that a sequential
learner must accept worsening moves, or that all zero-loss alternatives are bad.

### 4.3 Repetition probe (under-updating)

From `w_{t-1}`, apply the unchanged v1 rule to `(x_t, f_t)` repeatedly until no
edit strictly improves. This terminates because the loss is a non-negative
integer. Report the number of extra commits, whether the current example
becomes exact, how many previously correct items are broken at the end, and the
change in `d`. This is a counterfactual probe; the actual trajectory is not
altered.

Record total commits c, extra commits beyond B0's first decision
`c - 1[B0 commits]`, every intermediate state, and whether the terminal state
reaches the teacher class. The probe makes c+1 update-decision calls, including
the final no-improvement check, each evaluating 27 candidate losses. Report
previously correct items lost at any intermediate commit as well as those wrong
at the end; repetition may lose and restore an item within one experience.

### 4.4 Stream-end state

For each stream's final state `w_15`, report:

- `d(w_15)` and whether it is in the teacher class;
- the cumulative own-map loss over all 15 training maps;
- whether any N1 edit lowers that cumulative loss, and whether any N≤2 edit
  does; report single-block and up-to-two-block local-minimum status separately;
- for each item never acquired or forgotten in v1, the class (4.1) of the
  experience at which it was presented and of the experience at which it was
  lost.

For an item never acquired, loss-time attribution is null. For an acquired item,
list every correct→incorrect transition and the later experience whose update
caused it; distinguish the last loss leading to final forgotten status. An item's
presentation-time class is not necessarily the cause of its later loss. Keep
breaks at presentation separate as in the preceding study. The cumulative loss
has a realizable zero-loss teacher, so a nonzero local minimum is not global.

### 4.5 Readings (declared before computation)

No threshold makes one cause "the" cause; report the distribution. The declared
readings are:

| Pattern among failing experiences | Reading |
| --- | --- |
| E1a common | Single-block edits are too restricted; a two-block edit would improve the current example |
| E1b common | Even two-block edits are too restricted at those states |
| E2b common | Improving, preserving edits exist but the selection criterion ignores preservation |
| E2c common | Within single-block edits, improving the current example and preserving earlier items conflict |
| Repetition often reaches exactness without new breaks | The rule under-updates; one commit per experience is too few |
| Teacher-ward edits mostly neutral or worse | Direct proximity improvement is disfavored; consult the exact graphs before claiming a necessary barrier |
| G↓ cannot reach the class but G≤ can | This frozen-example route requires a plateau move |
| G≤ cannot reach the class | This frozen-example route requires an uphill move |
| A positive-loss end state is a single-block but not an up-to-two-block local minimum | A two-block edit can improve cumulative loss where a single-block edit cannot |
| A positive-loss end state is an up-to-two-block local minimum | A broader edit or an indirect path is required to lower cumulative loss; the teacher still supplies a zero-loss global solution |

Patterns can co-occur. Report them jointly, and do not name a cause that the
counts do not show.

## 5. Phase B — single-property rule variants (declared now)

All variants run on the same IB streams, from seven zeros, with one pass and the
same loss, tie-break `(loss, changed coefficients, lexicographic vector)` and
strict-improvement commit rule unless the variant changes exactly that one
property. Each uses its own simulation service and counters.

| Variant | One property changed | Candidates evaluated per experience |
| --- | --- | --- |
| **B0** | none (the v1 IB learner, reproduced) | 27 |
| **SCOPE** | Neighbourhood `N1 ∪ N2` (266 edits plus unchanged) | 267 |
| **REPEAT** | Repeat the v1 update on the current experience until no strict improvement | 27 × (c + 1), where c is total commits; the extra decision is the final non-commit check |
| **REPLAY** (diagnostic comparator) | Selection with history access: commit exactly when B0 would at the same state (some `N1` edit strictly improves the current example), but choose among those improving edits by `(summed loss over all earlier feedback maps, current loss, changed coefficients, lexicographic)` | 27, each scored on the current map and all stored earlier maps |

REPLAY stores every earlier feedback map. It is an external experience store
and is reported only as a comparator: it shows how much an improving edit chosen
with access to earlier feedback would preserve. It is never described as a
candidate SRI mechanism. No variant combines two properties, and none is added,
removed or retuned after phase A.

For REPLAY, "when B0 would" always means at the **same current parameter state
and example**, not at the same step of B0's diverged trajectory. Its history
contains all previous supplied IB maps, including those not currently correct;
it never includes held-out labels, teacher coefficients or phase-A diagnoses.
At the first experience the past-loss term is zero, so selection equals B0.
Score all 27 candidates on the current map and on each earlier stored map as
declared; only current-improving candidates are eligible for commitment. Append
the current map to history after the decision. Keep history storage and history
simulation costs explicit. REPEAT retains only its current experience during
that experience, not a historical replay store.

Minimizing summed past loss is **not** the same as protecting every previously
correct item. REPLAY can sacrifice a correct item to improve several incorrect
ones, even when a preserving current-improving candidate exists. For example,
two eligible edits can have earlier-item loss vectors `(0,4)` and `(1,0)`, where
the first earlier item was correct before the update. The preserving edit has
larger summed past loss, so REPLAY chooses the damaging one. Therefore no E2b
elimination guarantee is claimed. Report actual preservation rather than
substituting the past-loss objective for the preservation metric.

There is no variant that accepts a neutral or worsening **committed** move.
SCOPE can jump to a better two-block endpoint even when intermediate one-block
states are not improving; it does not traverse those intermediate states. It is
not a general treatment of longer plateaus or uphill paths. Its comparison
changes search scope and computation together, so any benefit must be reported
with cost and is not a matched-compute superiority claim. Stochastic or annealed
rules are out of scope. Phase A's exact path analysis is evaluation only.

## 6. Metrics

For every variant, teacher, split, order and checkpoint (0, 5, 10, 15):

- all local-learning-supervision v1 local metrics: exact-set accuracy on
  `within` and `mixed24` against the zero baseline, behavioural recovery,
  convergence to the teacher or mirror, false B8 atoms, and forward-witness
  rates for S2 and T1;
- per-item own-map and B1 timelines, categories, latency, loss and reacquisition
  events, `broken_at_presentation`, and minimum seen-item accuracy, as in v1
  §6.2;
- **cost:** candidate vectors evaluated, (candidate, key) predictions,
  simulation requests (total and unique), commits, coefficients changed, and,
  for REPLAY, stored maps and the predictions spent on them. Report recovery
  against candidate predictions. No variant's gain is reported without its cost.
- the diagnosis records described below, separating the variant's opportunity
  set from the baseline opportunity set at the same state.

### Classification across different update rules

E0–E2 in §4.1 describe B0's single decision per experience. For every phase-B
experience, first record that **counterfactual B0 classification at the variant's
starting state**, including the edit B0 would choose; do not apply it to learning.
Also record the actual experience outcome: current loss before/after, commits,
prior items broken at the end or at any intermediate commit, and d before/after.
The experience-level failing flag keeps §4.1's endpoint definition and is counted
once per original experience.

For actual decision-level E0–E2 labels use N1 for B0/REPLAY/REPEAT and N≤2 for
SCOPE. E1 means no improvement in that decision's allowed neighborhood; E2b/c
test preserving alternatives in that same neighborhood. E1a (N2 can improve)
can occur for an N1 decision, but cannot occur for a SCOPE-stuck decision.
E1b reports the minimum block distance to a lower-loss endpoint as before.

For REPEAT archive every decision, including its final non-commit decision.
At each subdecision, previously correct items are earlier experiences correct
immediately before that subdecision; the current experience is not part of that
set. Separately evaluate the whole experience against the prior-correct set at
its start. This distinguishes intermediate damage from endpoint preservation.
Do not pool REPEAT's multiple decision labels as if they were extra training
experiences. Keep decision-count and experience-count denominators separate.

Recompute the proximity, exact frozen-example path, cumulative-loss endpoint,
and per-item loss-attribution diagnostics for variant states using evaluator-only
code. Shared pure caches are allowed for diagnostics, never as a source of learner
labels or rule selection. The per-item timelines used for L3 remain sampled after
each original experience; intermediate REPEAT losses are additional diagnostics.

## 7. Correctness gates

- The local-learning-supervision v1 battery, IB ceiling and coverage hashes
  equal their frozen values.
- **B0 reproduction:** every per-step state, checkpoint row, timeline and work
  counter of the v1 IB block learner is reproduced exactly.
- The `N1` and `N2` enumerations have 26 and 240 distinct members, all differing
  from w in exactly one or exactly two blocks; tested on vectors with `w_12` at
  each value.
- Diagnostic losses computed from the battery equal the learner-facing
  simulation service's losses, for every candidate any variant evaluated.
  Count diagnostic comparisons separately; they must not change learner service
  counters, cache state or outputs. Shared-input and IB projection oracle checks
  remain the underlying semantics reference.
- E0–E2 classes are exhaustive and mutually exclusive, and the subclasses
  partition their class for the specified decision neighborhood. The frozen
  graph reachability categories also partition every non-teacher state examined.
- The teacher and its mirror have zero loss on every current and cumulative
  training objective. N1 connectivity, exact graph certificates and reproduced
  cumulative losses are checked independently of classification labels.
- Evaluator isolation: phase-A quantities and teacher identity never reach a
  learner; REPLAY's store contains only the feedback maps it was given.
- REPEAT terminates in every experience, with a strict loss decrease at each
  commit.
- REPLAY eligibility matches B0 at the same state; its tie-break matches the
  declared summed-past-loss objective. No claim that it eliminates E2b is used
  as a correctness gate.
- **Completion** is `complete` only if every gate and the test suite pass.
- Regeneration into a temporary directory with a different worker count is
  byte-identical for the manifest and the diagnosis and variant reports.

## 8. Prospective targets (phase B; S1 and C2; 25 streams per variant)

The v1 thresholds, unchanged, at checkpoint 15:

- **L1:** behavioural recovery in at least 23 of 25 runs, at least 4 per split.
- **L2:** at least 80% exact-set accuracy on `within` and `mixed24`, and above
  zero coupling on each.
- **L3:** own-map final retention at least 95% and forgetting rate at most 5%.

A variant **suffices** when it meets L1–L3 for both teachers. Report the cost of
every variant beside its result. A REPLAY pass is reported as "access to
earlier feedback suffices for this selection rule", not as a solution to
retention without an experience store.

Use the previous study's raw-count pooling and denominator rules: L2 pools the
25 final models on each partition; L3 pools own-map item counts across the 375
presented item instances per teacher/variant. Zero denominators are null and
do not pass; do not round before threshold comparisons. Report every split and
worst split. The 25 order/split streams are not independent datasets. L3 is
end-of-stream preservation, not absence of temporary forgetting; retain the
timelines and REPEAT subdecision diagnostics even when L3 passes. Additional
compute is not additional experience, and candidate counts must not be called
compute-matched without a separate matched budget.

Phase A has no pass/fail target; it is descriptive (§4.5).

## 9. Proofs and tests

- **Proofs:**
  - `N1 ⊂ N1 ∪ N2`, and both variants commit only on strict improvement, so
    SCOPE never stops where B0 could still improve. This does not make SCOPE's
    trajectory better than B0's: they diverge after the first differing commit.
  - REPEAT terminates: each commit strictly lowers a non-negative integer loss.
    The total commits on one fixed example are at most its initial integer loss;
    the final non-commit decision still incurs evaluation cost.
  - REPLAY commits exactly when B0 would at the same state, because both commit
    iff some `N1` edit strictly improves the current example. It changes only
    which eligible edit is chosen, using additional historical feedback. At a
    shared E2c state it cannot avoid a break with an eligible edit, but its
    diverged trajectory might avoid visiting such states. No E2b guarantee
    follows from minimizing summed past loss; retain the counterexample in §5.
  - A teacher path in G↓ is also in G≤. If G≤ has no teacher path, connectedness
    of the unrestricted N1 graph implies every teacher path has an uphill edge.
    State the fixed-example scope explicitly. A shortest path in edit distance
    need not be a descending path in loss.
  - Every N≤2 local minimum is an N1 local minimum. Validate positive-loss local
    minima against the teacher's zero-loss global solution.
- **Tests:**
  - `N1`/`N2` counts, block-difference counts and absence of duplicates;
  - E0–E2 classification on hand-built loss tables, including ties, a stuck
    state with a two-block improvement, and E2b versus E2c;
  - progress classification (`d` change; improving, neutral, worse);
  - strict and non-increasing reachability on hand-built graphs: a strict route
    initially increasing d, a plateau bridge, and a case requiring an uphill edge;
  - a greedy repetition path missing another existing strictly descending route;
  - local-minimum nesting and the impossible two-block-only local-minimum label;
  - repetition probe on a hand-built case needing two commits;
  - B0 reproduction of one v1 IB stream;
  - REPLAY commits exactly when B0 would, on a shared state;
  - a preserving eligible edit losing to a damaging edit with lower summed past
    loss, demonstrating that REPLAY need not eliminate E2b;
  - REPEAT's stopping-check cost for zero, one and several commits, and a
    within-experience loss followed by recovery;
  - variant-specific E0–E2 classification, SCOPE's absent E1a case, and separate
    decision/experience denominators;
  - evaluator boundary: learner APIs receive no phase-A quantity;
  - cost counters for each variant;
  - completion status requires passing tests; deterministic regeneration;
    unchanged prior artifact hashes.

## 10. Deliverables and stopping rule

- Code: `src/update_diagnosis.py`, `experiments/update_diagnosis.py`,
  `tests/test_update_diagnosis.py`. Import earlier code; add nothing to it.
- Reports, prefix `local-update-diagnosis-v1`: `-manifest.json`,
  `-diagnosis.json` (phase A), `-variants.json` (phase B, with per-variant
  diagnosis), `-runtime.json`.
- Results: `docs/local-update-diagnosis-v1-results.md`, reporting phase A before
  phase B. README and open-problems entries after validation.

The manifest must record the contract/source hashes, teacher vectors, prior
battery/IB ceiling/learning/coverage/split hashes, query and order IDs, neighborhood
and objective definitions, all tie-breaks, stopping rules, cost units and targets.
Store reproduced baseline traces, per-decision variant traces and reconstructible
loss-graph certificates in the scientific reports or explicitly linked compressed
artifacts. Do not require opaque summary counts to stand in for those witnesses.

Order: freeze the manifest; verify hashes; reproduce B0; run and write phase A;
then run every phase B variant in full. Never continue through a failed gate.
Stop after this grid. Do not add variants, combine properties, change
thresholds or add teachers. A complete run with negative results is a completed
study.

Use standard-library reference code and unittest. Run targeted tests, then the
full existing suite; completion requires both tests and correctness gates to
pass. Support `--out-dir` and `--workers`, report source/interpreter and prior
artifact hashes, and separate volatile git/timing metadata from deterministic
scientific output. Start timing before preprocessing. Keep battery construction,
diagnostic graph work, learner work, evaluation and tests separate. Record main
and worker memory with the measurement method; neither a single worker nor the
main-process peak alone is total concurrent memory.

Memoize compact per-example loss vectors and graph reachability; avoid retaining
all full dynamics graphs. Resource exhaustion must report incomplete diagnostics,
not a proven absence of paths. Do not prune the graph, reduce the space, alter
variants or tune a rule after diagnosis to finish a favorable run.

## 11. Scope

The study diagnoses one rule under one supervision protocol, on four
hand-chosen teachers with noise-free, complete-set feedback and a finite
battery. Diagnostics use the teacher class and information about earlier-item
correctness that no learner receives. REPLAY alone receives stored earlier
feedback, never diagnostic labels or teacher identity. The diagnostics describe
local opportunities and barriers in this setting; they are not themselves a
learning procedure. A variant that succeeds shows that changing
that one property suffices in this grid, not that it is necessary or that it
matches SRI's intended mechanism. REPLAY in particular is outside the target
architecture.
