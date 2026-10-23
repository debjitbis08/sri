# Local update diagnosis v1 — why the block learner fails under IB supervision

**Status:** prospective, not implemented. Draft of 2026-10-23. Preserve every
earlier contract, code file and report unchanged, including all
local-learning-supervision v1 and learned-input-mediation v1 artifacts and the
frozen v1 split manifest. Freeze this contract's hash in the new manifest before
any computation.

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
any diagnosis (phase B). Phase A explains phase B's outcomes; it does not
select which variant to run.

Three failure types are separated:

| Type | Meaning |
| --- | --- |
| **Restricted edits** | No single-block edit lowers the current example's loss, but a larger edit would |
| **Preservation failure** | An improving edit exists, and the committed one breaks previously correct items |
| **Short-sighted improvement** | Progress toward the teacher class needs moves that are neutral or worse on the current example |

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

## 4. Phase A — diagnosis of the existing IB trajectories

**Reproduction first.** Rerun the v1 IB block learner. Every per-step state, the
checkpoint vectors and the per-item timelines must equal the frozen v1 learning
report. Then diagnose those exact trajectories.

**Evaluator boundary.** Every diagnostic is computed from the battery by
evaluator code, after the learner's update and outside it. Previously correct
items, teacher identity and `d(w)` are evaluator knowledge. Nothing computed in
phase A feeds into any learner, in phase A or phase B. Diagnostic loss
evaluations are counted separately from learner work.

### 4.1 Per-experience classification

For each experience t, with `ℓ0 = ℓ_t(w_{t-1})`, assign exactly one top-level
class:

- **E0, already exact:** `ℓ0 = 0`. No update is needed. Report separately
  whether `w_{t-1}` is in the teacher class.
- **E1, stuck:** `ℓ0 > 0` and no edit in `N1` has lower loss. Subclasses:
  - **E1a:** some edit in `N2` has lower loss (a restricted-edit case);
  - **E1b:** none in `N2` does, and some vector in the full space has lower loss
    (needs three or more blocks). Report the minimum number of blocks needed.
- **E2, committed:** `ℓ0 > 0` and the learner commits an improving edit.
  Subclasses by preservation, on the own map:
  - **E2a:** the committed edit breaks no previously correct item;
  - **E2b:** it breaks at least one, but some other improving edit in `N1`
    breaks none (selection by current loss ignored preservation);
  - **E2c:** every improving edit in `N1` breaks at least one previously correct
    item (a preservation conflict within single-block edits).

  For E2, also report whether the committed edit makes the current example
  exact, and how many items it breaks.

An experience is **failing** if, after the update, the current example is not
exact or a previously correct item is broken. Report every class and subclass
as counts and as fractions of all experiences and of failing experiences.

### 4.2 Progress toward the teacher class

For each experience with `w_{t-1}` outside the teacher class:

- the change in `d` caused by the actual update (−, 0, +);
- whether any `N1` edit reduces `d`, and for those, the change in current loss
  of each: improving, neutral or worse. Report the best such edit's category and
  the counts;
- the same for `N2` edits that reduce `d` by two.

If, at most steps, the edits that move toward the teacher class are neutral or
worse on the current example, the greedy criterion is blocking the route. That
is the short-sighted-improvement signature.

### 4.3 Repetition probe (under-updating)

From `w_{t-1}`, apply the unchanged v1 rule to `(x_t, f_t)` repeatedly until no
edit strictly improves. This terminates because the loss is a non-negative
integer. Report the number of extra commits, whether the current example
becomes exact, how many previously correct items are broken at the end, and the
change in `d`. This is a counterfactual probe; the actual trajectory is not
altered.

### 4.4 Stream-end state

For each stream's final state `w_15`, report:

- `d(w_15)` and whether it is in the teacher class;
- the cumulative own-map loss over all 15 training maps;
- whether any `N1` edit lowers that cumulative loss, and whether any `N2` edit
  does (whether the end state is a single-block or two-block local minimum of
  the cumulative loss);
- for each item never acquired or forgotten in v1, the class (4.1) of the
  experience at which it was presented and of the experience at which it was
  lost.

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
| Teacher-ward edits mostly neutral or worse (4.2) | Greedy improvement on the current example blocks progress |
| End states are two-block but not single-block local minima | Restricted edits trap the end state |

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
| **REPEAT** | Repeat the v1 update on the current experience until no strict improvement | 27 × (1 + extra commits) |
| **REPLAY** (diagnostic comparator) | Selection only: commit exactly when B0 would (some `N1` edit strictly improves the current example), but choose among those improving edits by `(summed loss over all earlier feedback maps, current loss, changed coefficients, lexicographic)` | 27, each scored on the current map and all stored earlier maps |

REPLAY stores every earlier feedback map. It is an external experience store
and is reported only as a comparator: it shows how much an improving edit chosen
with access to earlier feedback would preserve. It is never described as a
candidate SRI mechanism. No variant combines two properties, and none is added,
removed or retuned after phase A.

There is no variant that accepts worsening moves. SCOPE covers paths through one
neutral or worse single-block move, because a two-block edit is such a path
taken in one step. A stochastic or annealed rule would add a second property
(randomness) and is out of scope.

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
- phase A's per-experience classes, recomputed on each variant's trajectory
  (using that variant's own update), so a variant's success can be traced to the
  failure types it removes.

## 7. Correctness gates

- The local-learning-supervision v1 battery, IB ceiling and coverage hashes
  equal their frozen values.
- **B0 reproduction:** every per-step state, checkpoint row, timeline and work
  counter of the v1 IB block learner is reproduced exactly.
- The `N1` and `N2` enumerations have 26 and 240 distinct members, all differing
  from w in exactly one or exactly two blocks; tested on vectors with `w_12` at
  each value.
- Diagnostic losses computed from the battery equal the learner-facing
  simulation service's losses, for every candidate the B0 learner evaluated.
- E0–E2 classes are exhaustive and mutually exclusive, and the subclasses
  partition their class.
- Evaluator isolation: phase-A quantities and teacher identity never reach a
  learner; REPLAY's store contains only the feedback maps it was given.
- REPEAT terminates in every experience, with a strict loss decrease at each
  commit.
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

Phase A has no pass/fail target; it is descriptive (§4.5).

## 9. Proofs and tests

- **Proofs:**
  - `N1 ⊂ N1 ∪ N2`, and both variants commit only on strict improvement, so
    SCOPE never stops where B0 could still improve. This does not make SCOPE's
    trajectory better than B0's: they diverge after the first differing commit.
  - REPEAT terminates: each commit strictly lowers a non-negative integer loss.
  - REPLAY commits exactly when B0 would at the same state, because both commit
    iff some `N1` edit strictly improves the current example. It changes only
    which improving edit is chosen, so it isolates selection. In particular, it
    removes E2b at every state where E2b occurs, and cannot remove E2c.
- **Tests:**
  - `N1`/`N2` counts, block-difference counts and absence of duplicates;
  - E0–E2 classification on hand-built loss tables, including ties, a stuck
    state with a two-block improvement, and E2b versus E2c;
  - progress classification (`d` change; improving, neutral, worse);
  - repetition probe on a hand-built case needing two commits;
  - B0 reproduction of one v1 IB stream;
  - REPLAY commits exactly when B0 would, on a shared state;
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

Order: freeze the manifest; verify hashes; reproduce B0; run and write phase A;
then run every phase B variant in full. Never continue through a failed gate.
Stop after this grid. Do not add variants, combine properties, change
thresholds or add teachers. A complete run with negative results is a completed
study.

## 11. Scope

The study diagnoses one rule under one supervision protocol, on four
hand-chosen teachers with noise-free, complete-set feedback and a finite
battery. Its diagnostics use evaluator knowledge, including the teacher class
and earlier items, that no learner has. They describe why this rule fails here;
they are not a learning procedure. A variant that succeeds shows that changing
that one property suffices in this grid, not that it is necessary or that it
matches SRI's intended mechanism. REPLAY in particular is outside the target
architecture.
