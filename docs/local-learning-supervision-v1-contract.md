# Local learning supervision v1 — what makes local mechanism learning succeed

**Status:** prospective; frozen for implementation on 2026-10-23. Preserve every
earlier contract, code file and report unchanged. That includes the
learned-input-mediation v1 contract, manifest, battery summary, ceilings and
learning report, and the frozen v1 split manifest. Freeze this contract's hash
in the new manifest before any computation.

**Pre-implementation review (2026-10-23).** This revision clarifies the remaining
supervision confounds, per-item acquisition/forgetting, end-of-stream versus
continuous preservation, IB's observation boundary, and v1 reproduction. No
teacher, condition, split rule, learner rule or numerical threshold was changed.
No new dynamics or learning were run during this review.

## 1. Question

> With the local update rule held fixed, which supervision makes the learned
> mechanism recoverable reliably, and does it preserve earlier behaviour?

In [learned input mediation v1](learned-input-mediation-v1-results.md) the block
learner recovered C2's behaviour in 9/25 runs from canonical outputs (arm O) and
in 25/25 runs from satellite clamps with internal input readouts (arm I). Arm I
differs from arm O in three ways at once: it intervenes, it reads internal input
coordinates, and it supplies more supervised terms per experience. This study
partially separates those factors and measures their costs; it is not a fully
factorial or supervision-budget-matched comparison. It does not change the learner.

Two new conditions, O+ and IB, are compared with the O and I references from v1:

| Condition | Training feedback per experience | Experiences | Role |
| --- | --- | --- | --- |
| **O** | B1 | 15 axis | Reference; reproduces v1 arm O |
| **O+** | B1 | 15 axis, then 12 mixed-evidence | 1. More observational coverage, no intervention |
| **IB** | B1, plus B6/B7 projected to the other satellite's mode only | 15 axis | 2. Intervention without internal readout |
| **I** | B1, B6, B7 | 15 axis | 3. Intervention with internal readout; reproduces v1 arm I |

O+ changes both evidence coverage and the number of update opportunities (27
versus 15). There is no extra-axis-only or repeated-axis control, so a benefit
cannot be attributed specifically to mixed evidence rather than additional
experiences. IB and I have the same experience and typed-entry counts but
different readout detail and tuple-level loss contributions. O versus IB also
changes both intervention access and supervision volume. Retain these limits
when interpreting a pass or failure; do not add controls after seeing results.

For each condition the study reports supervision cost, final mechanism recovery,
and acquisition and forgetting as separate quantities.

## 2. Knowledge at declaration (disclosure)

- The v1 run is complete and all its outcomes are known to the researchers. They
  include every per-run final vector for arms O and I, the identification
  ceilings, and the S2 survivor sets. "Held out" means withheld from the learner,
  not unseen by the researchers.
- v1 found that only 2 of S2's 7 arm-O survivors (splits 0–3) match S2 on the
  36 mixed-evidence queries. So it is known that full mixed coverage separates
  them. It is **not** known whether the 12-query coverage subset declared below
  does, or what the local learner does with it. No O+ or IB computation of any
  kind has been run.
- v1 computed C2 arm-I retention as 77.2% pooled, with 19 break events among 598
  failed checks. The per-item decomposition required here was not computed.
  Those 19 are correct-to-incorrect item/update comparisons, not necessarily 19
  distinct forgotten items. All C2 arm-I final models were exact, so a new
  end-of-stream retention criterion is already known to be easier for that
  reference than v1's throughout-training criterion.
- The hypothesis that the arm-O failure needs coordinated cross-block edits is
  not tested here. This study keeps the rule fixed so that a later rule change
  can be compared against a known supervision baseline.

## 3. Model, hypothesis space, teachers, battery

Everything is exactly as in the v1 contract §§3–5: the `E_C` dynamics and
interventions of `src/shared_input_mediation.py`, the 2,187-vector hypothesis
space, teachers S1, C2, S2 and T1, the B1–B8 battery with its frozen
fixed-witness atom schema, the four chain templates, mechanism equivalence, and
the core-sign symmetry. Import them from `src/mechanism_learning.py`; do not
modify them.

Rebuild the battery or load it through the v1 reconstruction interface. Either
way, its `battery_sha256` and `atom_schema_sha256` must equal those in the v1
battery summary before anything else runs.

The v1 published summary contains hashes rather than the full candidate battery;
use its deterministic reconstruction interface when no validated full cache is
available. Hash agreement is a provenance/regression check, not an independent
oracle proof. Reuse the previous oracle-verified dynamics and add the projection
checks below. Teacher labels, B8 witnesses, ceilings and held-out outcomes remain
evaluator-only; they must not enter updates or tie-breaking. S2 remains negative
for forward attribution, not for its known reverse witnesses.

**New observation keys (training only).** IB uses two projected keys:

- `B6x(v)`: for each `v` in D_A, the set of B's modes `b` in the terminal set
  under `do(A=v)`. This is B6 projected from `(i1, i2, b)` to `b`.
- `B7x(v)`: for each `v` in D_B, the set of A's modes `a` in the terminal set
  under `do(B=v)`. This is B7 projected from `(i1, i2, a)` to `a`.

These are functions of the existing B6/B7 atoms. They add no battery items and
no evaluation atoms. They are external readouts: the clamped satellite's
partner mode, with no input coordinate.

## 4. Splits and coverage queries

Use the frozen v1 split file unchanged: per split, 15 training (axis) queries,
30 within-context queries and 36 mixed-evidence queries, with 5 order seeds.

**Coverage subset.** For each split, select 12 of its 36 mixed-evidence queries,
3 from each of the four `(e1, e2)` evidence combinations. Within a combination,
take the 3 queries with the smallest
`sha256("local-learning-supervision-v1:coverage:{split_seed}:{e1},{e2},{da},{db}")`,
with the hex digest compared as a string. The rule uses no labels. Record the
selected IDs in the manifest before computation.

Hash the literal interpolated string as UTF-8, using ordinary decimal integers
without padding or added whitespace. `da` and `db` are the frozen domain indices;
query IDs are the indices in the frozen v1 query list. Break any digest tie by
ascending query ID. Check the frozen query list against the implementation before
selecting anything. Use the same selected queries for every teacher and condition
within a split; only their feedback labels differ.

**Held-out partitions for all conditions:**

- `within`: the 30 within-context queries (as v1);
- `mixed24`: the 24 mixed-evidence queries not in the coverage subset;
- `mixed36`: all 36, reported for O, IB and I only, for continuity with v1.

Comparisons between conditions use `within` and `mixed24`. O+ is never scored on
its coverage queries as held-out.

**O+ stream.** The 15 axis queries in the frozen order for that order seed, then
the 12 coverage queries in increasing order of
`sha256("local-learning-supervision-v1:order:{split_seed}:{order_seed}:{query_id}")`.
Coverage comes after axis training so that checkpoint 15 of O+ is identical to
O, and so that any loss of axis behaviour during coverage is measured directly.
Use the same UTF-8/integer conventions and query-ID tie-break for order hashes.

**Checkpoints:** 0, 5, 10, 15 for every condition; O+ adds 18, 21, 24, 27.

Evaluate mixed24 for O at checkpoint 15 and O+ at both 15 and 27 so that the
before/after coverage comparison uses identical query IDs. Differences between
the two checkpoints reflect the entire extra-training phase, not an isolated
causal effect of evidence diversity. `seen` contains all queries presented so far;
split O+'s seen metrics into axis and coverage groups as well as the combined set.

## 5. Learners and update rules

The learners and rules are exactly those of v1 §6, applied to each condition's
feedback map. Each experience is one map, processed as one update.

- **Block-local learner (primary).** Start at seven zeros; 26 block neighbours
  plus the unchanged vector; `loss = sum over keys of |predicted ⊖ feedback|`
  with unit weight per tuple; minimise `(loss, changed coefficients,
  lexicographic vector)`; commit only on strict improvement; one update per
  experience, one pass, no replay. No change to neighbourhood, loss weights,
  tie-breaking or the commit rule.
- **Full-feedback and positive-only version spaces (secondary).** As v1, with
  equality or containment on every supplied key. They answer the coverage
  question for S2: which feedback removes false-chain survivors.
- **Exact-query memory** and the **zero-coupling baseline**, as v1.

The simulation service answers `(candidate, current training query, supplied key)`
for the keys of the current condition only. In IB it must answer `B6x`/`B7x`
as deduplicated sets of partner modes; the learner must never receive the
unprojected B6/B7 sets, terminal states, or any `i1`, `i2` value. Private simulation
necessarily uses internal states; projection happens before any result crosses
the service's learner-facing boundary. Oracle/evaluator code may inspect full
states. Test the returned schema and access boundary, not an impossible claim
that internal coordinates are never computed. Memory answers only keys it was
given. Project first and then compute the symmetric-difference loss: duplicated
triples that map to one mode must not increase IB's loss or positive-tuple count.

Maintain separate simulation services/caches for O, I, IB and O+. For O and I,
preserve v1's within-condition cache reuse across teachers, splits and orders;
additional evaluator/timeline calls use separate services and counters. Count
all 27 local candidate losses per experience even when a result is cached, as a
new `candidate_losses` counter. Keep v1's `evaluations` counter unchanged in
meaning: one per (candidate, supplied key) prediction, so it is 27 per experience
only when the map has one key. O+
starts a fresh learner and has its own counters; equality with O is equality of
learned state at a matching prefix, not sharing mutable objects or prior feedback.
Full feedback is equality per supplied key and positive-only is containment per
key in every condition. Empty version spaces are errors, never certificates.
Preserve v1 observation-key order for O/I. For IB use B1, then B6x in ascending
allowed A values, then B7x in ascending allowed B values. This makes short-circuit
filtering work counters reproducible. Do not alter the block enumeration or
lexicographic tie-break in order to improve a result.

## 6. Metrics

All v1 §7 metrics are reported for every condition, teacher, split, order and
checkpoint, on `seen`, `within`, `mixed24` and (where defined) `mixed36`. The
following are added.

### 6.1 Supervision cost

Per stream and checkpoint, cumulative:

- experiences;
- typed feedback entries;
- **supervised slots**: the sum over entries of the size of that key's output
  space (B1: 9 `(a,b)` pairs; B6/B7: 27 `(i1,i2,·)` triples; B6x/B7x: 3 modes);
- **positive tuples**: the sum of feedback-set sizes;
- physical clamp runs: one per `B6`, `B7`, `B6x` or `B7x` entry;
- learner work: candidate evaluations, simulation requests (unique and total),
  commits, coefficients changed.

No single scalar is declared as "the" cost. Report recovery against each of
supervised slots, clamp runs and experiences.

Supervised slots use the fixed nominal alphabets above, even when a local domain
rules out some elements. They are encoding-space counts, not independent bits of
information or measured teaching effort. Deduplicate every response set; retain
distinct observation keys even if their results happen to coincide. A clamp run
is an intervention supplied with complete-set feedback, not one physical sample
that could by itself reveal every possible terminal outcome. Count canonical
response requests separately from clamp requests. Do not infer equal information
from equal slot counts or interpolate recovery to unobserved cost checkpoints.

### 6.2 Acquisition and forgetting (per item)

For each stream, each training item `s` presented at step `t_s`, and each step
`t ≥ t_s` after that step's update, record whether the current model reproduces
`s`. Do this twice:

- **own map**: every key the condition supplied for `s`;
- **B1 only**: the canonical output, which is comparable across conditions.

These timelines and the L3 metrics concern the block-local learner. Version-space
retention remains the exact/positive invariant from v1. Evaluate timelines with
held-out evaluator code; no old item or correctness label is replayed to learning.
Also record the item's correctness immediately **before** its own update, so
"correct after presentation" is not mistaken for newly learned at that step.

From that timeline report, per item and pooled:

- **acquired at presentation**: correct immediately after its own update;
- **ever acquired**, and acquisition latency in steps;
- **loss events**: correct at `t−1` and wrong at `t`; **reacquisitions**;
- **final category** at the last checkpoint: *never acquired*; *held* (acquired
  and never lost); *recovered* (lost, then correct at the end); *forgotten*
  (acquired, wrong at the end).

Use the post-presentation timeline for these four mutually exclusive categories.
First acquisition time is the first correct post-update step t ≥ t_s; latency
is t−t_s, so immediate acquisition has latency zero. Never-acquired latency is
null/right-censored, not zero; report its count separately from latency summaries.
Loss events compare consecutive post-update entries, starting at t_s+1.
Reacquisition is a wrong→correct transition after an earlier correct entry;
the first acquisition is not a reacquisition. If the item was correct before
its presentation but wrong immediately afterwards, record a separate
`broken_at_presentation` event. This is not silently counted as "never learned".

Derived rates, each with its denominator:

- acquisition rate = ever acquired / presented;
- forgetting rate = forgotten / ever acquired;
- final retention = correct at the last checkpoint / presented.

Also report ever-lost / ever-acquired, loss/reacquisition event counts and the
minimum prefix seen-item accuracy per stream. These expose temporary disruption
that final retention and final forgotten status can hide. Latency in steps is
not normalized for the greater supervision per step in I or IB.

For O+, also report these for the 15 axis items over the coverage phase
(steps 16–27) alone: how much axis behaviour coverage training breaks.

Use checkpoint 15 as this phase's baseline. Report the number of axis items
correct at 15; of those, how many are ever lost during 16–27 and how many remain
wrong at 27. Separately count initially incorrect axis items acquired during
coverage. Give an axis correctness table for (correct/wrong at 15) versus
(correct/wrong at 27), separately for own-map and B1 views. Do not infer damage
to axis knowledge from combined retention over axis plus new coverage items.

Also report the v1 pooled retention metric, labelled as such, so v1 numbers can
be compared. It is not a forgetting rate.

### 6.3 Mechanism recovery

As v1: behavioural recovery (every positive teacher fixed-witness tuple
reproduced, and the teacher-specific full gate passed), battery accuracy per
item type, false B8 atoms, and convergence to the teacher versus its core-sign
mirror. For S2 and T1, the false-forward-attribution rate of local models and of
version-space survivors, with SF, CF and their union separate.

Reproduce v1's teacher-specific gate/template definitions and Boolean witness
semantics. A surviving wrong-mechanism candidate represents unresolved mechanism
uncertainty, not a false certificate from an all-survivor-agreement learner.
Distinguish the contract's candidate-level attribution rate from certified
attribution. Define convergence here as reaching the teacher or core-sign mirror
at a checkpoint; it is not a proof of convergence of the update rule in general.

## 7. Identification ceilings (before learning)

O and O+ share the v1 arm-O ceiling (B1 on all 81 queries), and I shares the v1
arm-I ceiling; reproduce both. Compute the IB ceiling: the partition of the
2,187 hypotheses by B1, B6x and B7x on all 81 queries. For each teacher report
the class, its size, whether it is mechanism-homogeneous, and any false-forward
member, as v1 §5. Write the ceiling report before any learning.

These are complete-information ceilings on all 81 queries, not guarantees about
the finite 15- or 27-query streams. Preserve known sign-equivalent models and
report mechanism classes rather than demanding a unique coefficient vector.

## 8. Correctness gates

All v1 §8 gates, plus:

- battery and atom-schema hashes equal v1's;
- **reproduction**: conditions O and I reproduce every v1 per-run final vector,
  work counter and checkpoint row value on the partitions v1 defined (`seen`,
  `within`, `mixed36`);
- **prefix identity**: for every O+ stream, the learner states at checkpoints
  0–15 equal those of the O stream with the same split and order;
- `B6x`/`B7x` sets equal the projection of the independent oracle's B6/B7
  terminal sets, for the four teachers and the 50 sample indices of v1;
- for every prefix (after every step 0–15) and filtering rule,
  `V_I ⊆ V_IB ⊆ V_O`, and `V_O+` at step 15 equals `V_O`;
- after O+'s added observations, its V can only shrink from its step-15 V;
  this monotonicity applies to version spaces, not local-model accuracy;
- IB learner isolation: no internal coordinate or unprojected B6/B7 set crosses
  the learner-facing boundary of the simulation/feedback service;
- coverage subset and O+ order match the manifest, and no coverage query is in
  any O+ held-out partition;
- **completion** is `complete` only if every gate and the test suite pass
  (`stopped_failed_correctness_gate` or `stopped_failed_tests` otherwise);
- regeneration into a temporary directory with a different worker count is
  byte-identical for manifest, ceilings and learning report.

For reproduction, compare v1's existing row fields and logical learner counters
exactly, mapping new `mixed36` to v1 `mixed`. New timeline/cost fields have no v1
counterpart. Keep evaluator, battery and timing work separate: those naturally
change with the added conditions and metrics and must not be required to match
v1. For unique learner-simulation counts, use the isolated cache scope specified
in §5. Fail on unexplained differences rather than silently accepting approximate
reproduction. Through step 15, O/O+ prefix identity compares version membership,
failure state, local coefficients and memory entries; it does not compare cache
objects or elapsed times.

## 9. Prospective targets (no retrospective changes)

For S1 and C2, block-local learner, at each condition's final checkpoint,
over its 25 streams, reported per split:

Score each teacher/condition separately. L2 pools exact/n query counts over the
25 final models on each common partition. L3 pools per-item counts over those
streams (375 presented item instances for O/IB/I; 675 for O+). Report own-map and
B1-only values, but only own-map values determine L3. Repeated orders share a
split and are not 25 independent datasets. Report worst-split results alongside
pooled values. Compare thresholds using unrounded counts; a zero denominator
is null and does not pass a target.

- **L1 recovery:** behavioural recovery in at least 23 of 25 runs, with at least
  4 runs in every split.
- **L2 outputs:** at least 80% exact-set accuracy on `within` and on `mixed24`,
  and a strict improvement over zero coupling on each.
- **L3 preservation (own map):** final retention at least 95%, and forgetting
  rate at most 5%.

L1's threshold is stricter than v1's 13/25, because the question here is
reliability. v1 arm I would pass it (25/25). Reaching 13/25 is reported as well.

A condition **suffices for final recovery and end-of-stream preservation** for
a teacher when it meets L1–L3 for that teacher. The
study's primary answer is the set of conditions that suffice for both S1 and C2,
with their supervision cost.

L3 intentionally does not require continuous preservation: an item lost and
later recovered is correct at the end. A pass can coexist with substantial
temporary forgetting. Always accompany L3 with the timeline diagnostics in §6.2;
do not describe a pass as uninterrupted retention. These thresholds are goals
for this fixed grid, not estimates of reliability on arbitrary future streams.

For the version spaces (secondary), at the final checkpoint:

- **V1 (S2 coverage):** zero false-forward survivors for S2 in the full-feedback
  reference, reported separately for O+ and IB, in every split. If the IB
  ceiling contains a false-forward member, mark V1 for IB unattainable by
  identifiability rather than failed.
- **V2 (no regression):** all final survivors of the full-feedback reference
  are mechanism-equivalent to S1 or C2, respectively, in all 5 splits in O+ and
  IB. This is a consistency check implied by the previously identified O class
  and the subset relations, once reproduction is established. It is not an
  independent discovery of improved identifiability.

For S2 and T1 local learners, report false-attribution rates; no threshold.

### Reading the results (declared before computation)

| Outcome | Reading |
| --- | --- |
| IB suffices | The declared clamp/external-readout protocol meets the final targets; internal readouts are unnecessary for this grid and rule |
| IB fails, I suffices | The richer I protocol succeeds where this IB protocol fails; readout detail and tuple-level supervision volume remain confounded |
| O+ suffices | The 27-query observational protocol meets the targets; clamps are unnecessary for this grid and rule, but extra updates and mixed coverage are confounded |
| O+ recovers but fails L3 | Final mechanism recovery and prior-item accuracy differ; use the axis/coverage timelines to determine whether the cause is non-acquisition, forgetting or both |
| No condition except I suffices | Only I meets the joint targets among these finite protocols; other external-readout protocols or learning rules remain open |

None of these readings names a cause inside the learner. In particular, a
failure of O+ and IB does not show that cross-block coordination is the missing
ingredient. That needs a rule comparison under this frozen supervision.

## 10. Proofs and tests

- **Proofs:**
  - `B6x`/`B7x` are projections of B6/B7, and B1 is common to all three
    conditions. Equality, and containment of teacher sets, are preserved under
    projection. Hence `V_I ⊆ V_IB ⊆ V_O` for full and for positive-only filtering,
    for the same prefix. Strict inclusion is not guaranteed.
  - O+ and O run the same deterministic learner on the same first 15
    experiences, so their states agree through step 15.
  - Further clean O+ observations only eliminate version-space members, so the
    final O+ V is a subset of its step-15 V for both filtering rules. No such
    monotone accuracy guarantee follows for the local learner.
  - The core-sign mirror maps every B6x/B7x atom to itself, as for B6/B7, so it
    is in every IB ceiling class.
- **Tests:**
  - projection against the oracle; IB keys contain no input coordinates;
  - distinct internal triples projecting to the same mode are deduplicated before
    IB feedback, loss and cost calculation; I and IB need not have equal losses;
  - coverage subset: 3 per evidence combination, disjoint from `mixed24`,
    deterministic, independent of teacher;
  - O+ prefix identity and order rule;
  - per-item timeline categories on hand-built sequences (never acquired, held,
    recovered, forgotten), with acquisition and forgetting counted separately;
  - acquisition latency, never-acquired censoring, multiple loss/reacquisition
    cycles, correctness before presentation and a break at presentation;
  - O+ axis baseline at step 15, including an initially wrong item learned during
    coverage and an initially correct item lost then recovered;
  - supervised-slot and clamp-run accounting per key type;
  - reproduction of v1 rows for O and I;
  - completion status requires passing tests;
  - learner simulation counters are unaffected by extra evaluator/timeline calls;
  - null denominators and pooled raw-count target calculations;
  - deterministic regeneration and unchanged prior artifact hashes.

## 11. Deliverables and stopping rule

- Code: `src/local_supervision.py`, `experiments/local_supervision.py`,
  `tests/test_local_supervision.py`. Import v1 code; add nothing to it.
- Reports, prefix `local-learning-supervision-v1`: `-manifest.json` (contract
  hash, v1 battery and split hashes, coverage subsets, O+ orders, checkpoints,
  cost definitions, rules, observation-key schemas, teacher vectors and frozen
  v1 learning-report hash), `-ceiling.json` (IB, plus reproduced O and I),
  `-learning.json` (rows, per-item timelines, cost, targets), `-runtime.json`.
- Results: `docs/local-learning-supervision-v1-results.md`. README and
  open-problems entries after validation.

Order: freeze the manifest; verify battery hashes; run projection oracle checks;
write ceilings; run O and I and check reproduction; run IB and O+. Never continue
through a failed gate. Run every condition in full regardless of ceilings. Stop
after this grid. Do not change the learner, add conditions, resize the coverage
subset, add teachers or change thresholds. A complete run with negative results
is a completed study; report completion separately from each target.

Use standard-library code and unittest conventions. Run targeted tests and the
full existing suite; include the suite result in the new completion gate without
changing the frozen v1 runner. Keep deterministic scientific data separate from
timestamps, git metadata and runtime measurements. Record source/interpreter and
prior-artifact hashes. Start total timing before battery loading/reconstruction;
report learner, battery, evaluator and test costs separately. Report main-process
and worker memory measurements with their method; a main-process peak alone is
not the peak memory of all concurrent workers.

Provide `--out-dir` and `--workers` for reproducibility. Record the exact command
and compare the three deterministic reports (manifest, ceiling, learning) after
a temporary-directory rerun. Preserve the old artifacts even if the new study
exposes a reporting limitation; document it in the new results.

## 12. Scope

This study holds one local rule fixed and varies supervision. It shows which
supervision suffices for that rule on four hand-chosen teachers, with
realizable, noise-free, complete-set feedback, and a finite battery. It does not
show that other rules need the same supervision, that coverage beyond these 12
queries would or would not help, or anything about autonomous discovery,
plasticity laws, robustness or semantics. IB and I differ in readout and in
supervision volume together; that confound is reported, not resolved.
