# Local learning supervision v1 — what makes local mechanism learning succeed

**Status:** prospective, not implemented. Draft of 2026-10-23. Preserve every
earlier contract, code file and report unchanged. That includes the
learned-input-mediation v1 contract, manifest, battery summary, ceilings and
learning report, and the frozen v1 split manifest. Freeze this contract's hash
in the new manifest before any computation.

## 1. Question

> With the local update rule held fixed, which supervision makes the learned
> mechanism recoverable reliably, and does it preserve earlier behaviour?

In [learned input mediation v1](learned-input-mediation-v1-results.md) the block
learner recovered C2's behaviour in 9/25 runs from canonical outputs (arm O) and
in 25/25 runs from satellite clamps with internal input readouts (arm I). Arm I
differs from arm O in three ways at once: it intervenes, it reads internal input
coordinates, and it supplies more supervised terms per experience. This study
separates those factors. It does not change the learner.

Three supervision conditions are compared with two reference conditions taken
from v1:

| Condition | Training feedback per experience | Experiences | Role |
| --- | --- | --- | --- |
| **O** | B1 | 15 axis | Reference; reproduces v1 arm O |
| **O+** | B1 | 15 axis, then 12 mixed-evidence | 1. More observational coverage, no intervention |
| **IB** | B1, plus B6/B7 projected to the other satellite's mode only | 15 axis | 2. Intervention without internal readout |
| **I** | B1, B6, B7 | 15 axis | 3. Intervention with internal readout; reproduces v1 arm I |

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

**Checkpoints:** 0, 5, 10, 15 for every condition; O+ adds 18, 21, 24, 27.

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
directly; the learner must never see the unprojected B6/B7 sets or any `i1`,
`i2` value. Memory answers only keys it was given.

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

### 6.2 Acquisition and forgetting (per item)

For each stream, each training item `s` presented at step `t_s`, and each step
`t ≥ t_s` after that step's update, record whether the current model reproduces
`s`. Do this twice:

- **own map**: every key the condition supplied for `s`;
- **B1 only**: the canonical output, which is comparable across conditions.

From that timeline report, per item and pooled:

- **acquired at presentation**: correct immediately after its own update;
- **ever acquired**, and acquisition latency in steps;
- **loss events**: correct at `t−1` and wrong at `t`; **reacquisitions**;
- **final category** at the last checkpoint: *never acquired*; *held* (acquired
  and never lost); *recovered* (lost, then correct at the end); *forgotten*
  (acquired, wrong at the end).

Derived rates, each with its denominator:

- acquisition rate = ever acquired / presented;
- forgetting rate = forgotten / ever acquired;
- final retention = correct at the last checkpoint / presented.

For O+, also report these for the 15 axis items over the coverage phase
(steps 16–27) alone: how much axis behaviour coverage training breaks.

Also report the v1 pooled retention metric, labelled as such, so v1 numbers can
be compared. It is not a forgetting rate.

### 6.3 Mechanism recovery

As v1: behavioural recovery (every positive teacher fixed-witness tuple
reproduced, and the teacher-specific full gate passed), battery accuracy per
item type, false B8 atoms, and convergence to the teacher versus its core-sign
mirror. For S2 and T1, the false-forward-attribution rate of local models and of
version-space survivors, with SF, CF and their union separate.

## 7. Identification ceilings (before learning)

O and O+ share the v1 arm-O ceiling (B1 on all 81 queries), and I shares the v1
arm-I ceiling; reproduce both. Compute the IB ceiling: the partition of the
2,187 hypotheses by B1, B6x and B7x on all 81 queries. For each teacher report
the class, its size, whether it is mechanism-homogeneous, and any false-forward
member, as v1 §5. Write the ceiling report before any learning.

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
- for every prefix and filtering rule, `V_I ⊆ V_IB ⊆ V_O`, and `V_O+` at step 15
  equals `V_O`;
- IB learner isolation: no `i1`/`i2` value and no unprojected B6/B7 set reaches
  the learner or its simulation service;
- coverage subset and O+ order match the manifest, and no coverage query is in
  any O+ held-out partition;
- **completion** is `complete` only if every gate and the test suite pass
  (`stopped_failed_correctness_gate` or `stopped_failed_tests` otherwise);
- regeneration into a temporary directory with a different worker count is
  byte-identical for manifest, ceilings and learning report.

## 9. Prospective targets (no retrospective changes)

For S1 and C2, block-local learner, at each condition's final checkpoint,
over its 25 streams, reported per split:

- **L1 recovery:** behavioural recovery in at least 23 of 25 runs, with at least
  4 runs in every split.
- **L2 outputs:** at least 80% exact-set accuracy on `within` and on `mixed24`,
  and a strict improvement over zero coupling on each.
- **L3 preservation (own map):** final retention at least 95%, and forgetting
  rate at most 5%.

L1's threshold is stricter than v1's 13/25, because the question here is
reliability. v1 arm I would pass it (25/25). Reaching 13/25 is reported as well.

A condition **suffices** for a teacher when it meets L1–L3 for that teacher. The
study's primary answer is the set of conditions that suffice for both S1 and C2,
with their supervision cost.

For the version spaces (secondary), at the final checkpoint:

- **V1 (S2 coverage):** zero false-forward survivors for S2 in the full-feedback
  reference, reported separately for O+ and IB, in every split. If the IB
  ceiling contains a false-forward member, mark V1 for IB unattainable by
  identifiability rather than failed.
- **V2 (no regression):** S1 and C2 survivors stay mechanism-equivalent to the
  teacher in all 5 splits in O+ and IB.

For S2 and T1 local learners, report false-attribution rates; no threshold.

### Reading the results (declared before computation)

| Outcome | Reading |
| --- | --- |
| IB suffices | Intervention with external readout is enough; internal readout is not needed by this rule |
| IB fails, I suffices | Internal readout, or I's larger supervision volume, is needed; the two are not separated here |
| O+ suffices | Observations suffice with mixed coverage; clamps are not needed by this rule |
| O+ recovers but fails L3 | Coverage teaches the mechanism but breaks axis behaviour: an acquisition/forgetting trade-off |
| No condition except I suffices | Less privileged feedback is insufficient for this rule; a rule change is the next variable |

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
  - The core-sign mirror maps every B6x/B7x atom to itself, as for B6/B7, so it
    is in every IB ceiling class.
- **Tests:**
  - projection against the oracle; IB keys contain no input coordinates;
  - coverage subset: 3 per evidence combination, disjoint from `mixed24`,
    deterministic, independent of teacher;
  - O+ prefix identity and order rule;
  - per-item timeline categories on hand-built sequences (never acquired, held,
    recovered, forgotten), with acquisition and forgetting counted separately;
  - supervised-slot and clamp-run accounting per key type;
  - reproduction of v1 rows for O and I;
  - completion status requires passing tests;
  - deterministic regeneration and unchanged prior artifact hashes.

## 11. Deliverables and stopping rule

- Code: `src/local_supervision.py`, `experiments/local_supervision.py`,
  `tests/test_local_supervision.py`. Import v1 code; add nothing to it.
- Reports, prefix `local-learning-supervision-v1`: `-manifest.json` (contract
  hash, v1 battery and split hashes, coverage subsets, O+ orders, checkpoints,
  cost definitions, rules), `-ceiling.json` (IB, plus reproduced O and I),
  `-learning.json` (rows, per-item timelines, cost, targets), `-runtime.json`.
- Results: `docs/local-learning-supervision-v1-results.md`. README and
  open-problems entries after validation.

Order: freeze the manifest; verify battery hashes; run projection oracle checks;
write ceilings; run O and I and check reproduction; run IB and O+. Never continue
through a failed gate. Run every condition in full regardless of ceilings. Stop
after this grid. Do not change the learner, add conditions, resize the coverage
subset, add teachers or change thresholds. A complete run with negative results
is a completed study; report completion separately from each target.

## 12. Scope

This study holds one local rule fixed and varies supervision. It shows which
supervision suffices for that rule on four hand-chosen teachers, with
realizable, noise-free, complete-set feedback, and a finite battery. It does not
show that other rules need the same supervision, that coverage beyond these 12
queries would or would not help, or anything about autonomous discovery,
plasticity laws, robustness or semantics. IB and I differ in readout and in
supervision volume together; that confound is reported, not resolved.
