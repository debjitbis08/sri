# Local learning supervision v1 — less privileged feedback identifies the mechanism, but the fixed local rule does not learn it

**Completion:** `complete`. Every correctness gate passed. The full declared grid
ran: 4 teachers × 4 conditions × 5 splits × 5 orders, with 4 checkpoints
(8 for O+). Conditions O and I reproduce v1 arms O and I exactly.

**Headline:**

- **Only I suffices.** With the block-local rule fixed, only condition I (clamps
  with internal readouts) meets L1–L3 for S1 and C2. IB (clamps, partner-mode
  readout only) and O+ (15 axis queries, then 12 mixed-evidence queries) fail all
  three targets for both teachers.
- **The feedback is sufficient; the rule does not use it.** On the same streams,
  the exact full-feedback learner ends with only the teacher and its core-sign
  mirror in every split under IB and under O+, for every teacher. Both
  conditions remove S2's false-chain survivors (V1 passes for both). So the 15 IB
  experiences, and the 27 O+ experiences, exclude every hypothesis that is not
  mechanism-equivalent. The local learner's failure under IB and O+ is a failure
  to use information the stream contains. Which part of the rule causes it was
  not tested.
- **IB does not meet the reliability targets.** Relative to O it improves S1
  recovery from 0 to 5/25 runs, and C2 goes from 9 to 7/25. It has the same 70.2 clamp runs
  per stream as I and 17% of I's supervised slots.
- **O+ helps C2 slightly and breaks S1's axis behaviour.** C2 goes from 9/25 at
  step 15 to 13/25 at step 27. S1 goes from 0 to 1/25. During coverage S1 loses
  more axis items than it gains (178 correct at step 15 → 148 at step 27); C2
  gains net (241 → 273).
- **v1's 77.2% C2 retention was delayed acquisition plus temporary loss, with no
  end-of-stream forgetting.** Under I, all 375 presented items are acquired and
  none is wrong at the end. The losses were real but temporary: 18 items were
  lost and later recovered (19 loss events).

[Contract](local-learning-supervision-v1-contract.md) (sha256 `4b4fa3e6…`, frozen
and committed before the manifest; not edited afterwards) ·
[manifest](../reports/local-learning-supervision-v1-manifest.json) (committed
before computation) · [ceilings](../reports/local-learning-supervision-v1-ceiling.json) ·
[learning](../reports/local-learning-supervision-v1-learning.json) ·
[runtime](../reports/local-learning-supervision-v1-runtime.json). The run takes
180 s on 8 workers: 51 s battery, 11 s oracle, projection and v1 reproduction,
1 s ceilings, 69 s learning and evaluation, 43 s v1 reproduction, targets and
local gates, 4 s tests. Peak resident memory is 566 MiB in the main process
(`getrusage`; the largest single worker peaked at 29 MiB; this is not a sum over
concurrent workers).

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Battery, atom schema and all 2,187 per-hypothesis hashes equal v1's | yes |
| v1 independent-oracle check (54 hypotheses) and frozen shared-input reproduction | pass |
| B6x/B7x: independent oracle terminals = projected battery masks = IB simulation service, 54 hypotheses, every clamp | 20,412 / 20,412 equal |
| v1 ceilings reproduced for O and I (new class code and v1 code against the frozen report) | all equal |
| v1 learning reproduced for O and I: all 14,400 rows (`mixed36` ↔ v1 `mixed`), every run, every learner and simulator counter, pooled retention | 0 mismatches |
| `V_I ⊆ V_IB ⊆ V_O` after every step 0–15, full and positive-only (3,200 checks) | 0 violations |
| O+ learner state (V, failure state, coefficients, memory) equals O through step 15 (1,600 checks) | 0 mismatches |
| O+ version spaces only shrink during coverage (2,400 checks) | 0 violations |
| IB boundary: no internal coordinate or unprojected B6/B7 set in any IB feedback map | 0 violations |
| Coverage subsets and O+ orders equal the manifest; coverage disjoint from `mixed24` | yes |
| Teacher eliminated / retention violations / containment violations / order-dependent final V / `inconsistent_feedback` | 0 / 0 / 0 / 0 / 0 |
| Byte-identical regeneration of manifest, ceilings and learning report, temporary directory, 4 workers instead of 8 | yes |
| Tests | 19 new; full suite 209, all passing |

**Defect found before any scored run.** A new unit test failed on its first run:
`axis_phase` counted two cells of the O+ coverage-phase table twice. It was fixed
before the manifest was frozen and before any computation.

**Contract clarifications at freeze.** Before the contract was frozen, two points
were made explicit: v1's `evaluations` counter keeps its meaning (one per
candidate × key), with a new `candidate_losses` counter added; and the subset
gate applies after every step. Neither changed a condition, threshold or rule.

## Identification ceilings (computed before learning)

For every teacher, the complete-information class under O, I and IB has exactly
two members: the teacher and its core-sign mirror. Each class is
mechanism-homogeneous. No class member has a forward witness for S2 or T1. So
the IB class does not obstruct V1 for IB, and O+ shares O's class.

## Targets (S1 and C2; block-local learner, final checkpoint, 25 streams)

| Condition | L1 recovery (worst split) | L2 within / mixed24 (zero baseline) | L3 final retention; forgetting rate (own map) | Suffices |
| --- | --- | --- | --- | --- |
| **S1, O** | 0/25 (0) | 313 / 195 (275 / 140) of 750 / 600 | 47.5%; 38.6% | no |
| **S1, I** | **25/25** (5) | **750 / 600** | **100%; 0%** | **yes** |
| **S1, IB** | 5/25 (1) | 411 / 288 | 37.1%; 46.1% | no |
| **S1, O+** | 1/25 (0) | 291 / 263 | 41.6%; 47.1% | no |
| **C2, O** | 9/25 (0) | 468 / 315 (210 / 60) | 64.3%; 15.4% | no |
| **C2, I** | **25/25** (5) | **750 / 600** | **100%; 0%** | **yes** |
| **C2, IB** | 7/25 (0) | 455 / 290 | 59.2%; 19.0% | no |
| **C2, O+** | 13/25 (1) | 526 / 421 | 73.2%; 15.3% | no |

Thresholds: L1 at least 23/25 with at least 4 in every split; L2 at least 80% on
each partition and above zero coupling; L3 at least 95% final retention and at
most 5% forgetting. All are scored on raw counts. C2 under O+ reaches v1's old
13/25 level, but not this study's 23/25.

**Primary answer:** the only condition that suffices for both S1 and C2 is I.

Details behind the table:

- **Convergence.** Under I the learner reaches the mirror (S1 25/25; C2 24/25,
  plus 1 run on the teacher), as in v1. Under IB it ends on 20 distinct vectors
  for S1 (4 mirror, 1 teacher) and 10 for C2 (7 mirror). Under O+ it ends on 13
  (S1) and 11 (C2) distinct vectors.
- **IB fits outputs better than clamp readouts.** For S1 under IB, final
  retention is 37.1% on the own map but 57.6% on B1 alone. The learner often
  reproduces the canonical output while missing the projected clamp result.
- **Recovery along the stream.** C2 under O+ recovers in 9, 9, 11, 11 and 13 runs
  at steps 15, 18, 21, 24 and 27. Under I, recovery is 21/25 (S1) and 5/25 (C2)
  at step 5, then 25 and 21 at step 10.

## Acquisition and forgetting (own map; S1 and C2; 375 item instances, 675 for O+)

| Condition | Never acquired | Held | Recovered | Forgotten | Loss events | Acquisition rate |
| --- | --- | --- | --- | --- | --- | --- |
| S1, O | 85 | 138 | 40 | 112 | 190 | 77.3% |
| S1, I | 0 | 364 | 11 | 0 | 12 | 100% |
| S1, IB | 117 | 84 | 55 | 119 | 250 | 68.8% |
| S1, O+ | 144 | 159 | 122 | 250 | 652 | 78.7% |
| C2, O | 90 | 212 | 29 | 44 | 105 | 76.0% |
| C2, I | 0 | 357 | 18 | 0 | 19 | 100% |
| C2, IB | 101 | 153 | 69 | 52 | 209 | 73.1% |
| C2, O+ | 92 | 391 | 103 | 89 | 314 | 86.4% |

"Held" means acquired and never lost; "recovered" means lost and correct at the
end; "forgotten" means acquired and wrong at the end. No item was broken by its
own update in any condition.

- **The failing conditions show both problems.** They leave 14–31% of items never
  acquired and forget 15–47% of the items they acquire. Neither alone accounts
  for the failure.
- **I passes L3 despite disruption along the way.** Its pass is end-of-stream
  preservation, not uninterrupted retention. Mean acquisition latency is 1.97
  steps for C2 (0.29 for S1), and two C2 items took 13 steps. The declared
  minimum seen-item accuracy per stream is 0 in every condition, because it
  includes step 1. As a post-hoc descriptive only, the minimum from step 5
  onward has median 1.0 (S1, I) and 0.0 (C2, I), 0.31/0.33 (O), 0.0/0.17 (IB)
  and 0.21/0.24 (O+). For C2 under I those zeros come almost entirely from
  items not yet acquired: of 370 wrong item-steps at those minima, 363 are items
  never yet correct and 7 are losses.
- **v1 pooled retention** (the throughout-training metric) is 30.9% for S1 and
  50.0% for C2 under IB, and 43.7% and 63.4% under O+. It is reported for
  comparison and is not a forgetting rate.

### O+ coverage phase (axis items, step 15 → step 27)

| Teacher | Correct at 15 | of which ever lost / wrong at 27 | Wrong at 15 | of which acquired during coverage / correct at 27 | Axis correct at 27 |
| --- | --- | --- | --- | --- | --- |
| S1 | 178 | 93 / 60 | 197 | 68 / 30 | 148 |
| C2 | 241 | 44 / 17 | 134 | 65 / 49 | 273 |

Own-map and B1 views coincide for O+. This compares the whole extra-training
phase; it cannot separate the effect of mixed evidence from the effect of 12 more
updates.

## Supervision cost (mean per stream at the final checkpoint)

| Condition | Experiences | Entries | Supervised slots | Positive tuples, S1 / C2 | Clamp runs | Candidate predictions | Recovery, S1 / C2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O | 15 | 15 | 135 | 23.8 / 29.6 | 0 | 405 | 0 / 9 |
| O+ | 27 | 27 | 243 | 38.4 / 47.8 | 0 | 729 | 1 / 13 |
| IB | 15 | 85.2 | 345.6 | 119.8 / 125.6 | 70.2 | 2,300 | 5 / 7 |
| I | 15 | 85.2 | 2,030.4 | 119.8 / 133.8 | 70.2 | 2,300 | 25 / 25 |

Slots are nominal alphabet sizes, not bits of information. For S1, IB and I
supply the same number of positive tuples; they differ in the internal input
coordinates of those tuples and in the 27-way versus 3-way alphabet against which
mismatches count. So the IB–I comparison still confounds readout detail with
tuple-level loss volume.

## Version spaces (secondary)

| Teacher | Condition | Final full-feedback survivors per split | Mechanism-equivalent | Forward-witness survivors | Status |
| --- | --- | --- | --- | --- | --- |
| S1, C2 | O, I, IB, O+ | 2 | 2 | 2 (correct: the teacher's own chain) | V2 pass (O+, IB) |
| S2 | O | 7, 7, 7, 7, 15 | 2 | 5 | reference (as v1) |
| S2 | I | 2 | 2 | 0 | reference |
| S2 | IB | 2 | 2 | 0 | **V1 pass** |
| S2 | O+ | 2 | 2 | 0 | **V1 pass** |
| T1 | all | 2 | 2 | 0 | — |

For S1 and C2 the forward witnesses are correct: the teacher has that chain. For
the negative controls S2 and T1, every forward-witness survivor is false
attribution. Certified output coverage is 100% on `within` and `mixed24` for
every S1/C2 condition, with no incorrect certificate. V2 is a consistency check:
it follows from the subset relations once v1 is reproduced.

The 12 coverage queries are enough to separate S2's false survivors. That was
not known at declaration: v1 had shown only that all 36 mixed queries separate
them. Clamps with partner-mode readout separate them as well. Internal readouts
are therefore not needed to remove S2's false attribution in this grid.

## Negative controls (local learner; no threshold)

| Teacher | Runs with a forward witness, O / I / IB / O+ |
| --- | --- |
| S2 | 9 / 0 / 6 / 16 of 25 |
| T1 | 9 / 0 / 6 / 12 of 25 |

O+ raises the local learner's false-attribution rate for both negative controls,
even though the exact learner under O+ has none.

## Reading the results (as declared)

The outcome matches the declared row *"No condition except I suffices: only I
meets the joint targets among these finite protocols; other external-readout
protocols or learning rules remain open."* The "IB fails, I suffices" reading
also applies: readout detail and tuple-level supervision volume remain
confounded.

What the version-space results add is measured, not inferred. The IB and O+
streams contain enough information to identify the mechanism up to the mirror
and to rule out S2's false chains. The fixed local rule does not reach that
class from them. So the next variable is the rule, not more privileged feedback.
This study does not show *which* property of the rule matters: one block edit
per experience, strict-improvement commits, the single pass without replay, or
the global set loss. Cross-block coordination is one candidate among these.

## What this does not establish

- **No cause inside the learner.** The rule was held fixed by design.
- **Confounds.** O+ adds both mixed evidence and 12 more updates. IB and I differ
  in readout detail and in tuple-level loss volume. O and IB differ in
  intervention access and supervision volume. No control separates these.
- **End-of-stream preservation only.** L3 lets an item be lost and recovered.
  I's pass coexists with delayed acquisition and 12–19 temporary loss events.
- **Finite and easy supervision.** Four hand-chosen teachers whose behaviour the
  researchers knew, realizable noise-free complete-set feedback, one coverage
  subset of 12, and a finite battery. Agreement on the battery is not
  equivalence under every intervention.
- **No broader claims.** Nothing here concerns autonomous discovery, plasticity
  laws, robustness or semantics.

## Next obligations (not run)

- **Primary supervision: IB, held fixed.** It supplies sufficient information
  (the exact learner identifies every teacher from it), avoids internal
  readouts, and keeps the 15-experience length of the original baseline.
- **Diagnose failed updates before choosing a replacement rule.** For each IB
  experience where the block learner does not reach or keep the right model, ask:
  1. *Restricted edits:* does no permitted single-block edit improve the current
     example?
  2. *Preservation failure:* does an improving edit exist but break previously
     correct items?
  3. *Short-sighted improvement:* would more updates on the current example
     help, or does progress need moves that are temporarily neutral or worse?
  This separates restricted edits, short-sighted improvement and preservation
  failures, instead of assuming that cross-block coordination is the cause.
- **Then vary one rule property at a time,** with the extra candidate
  evaluations each variant costs counted and reported.
- **Replay is a diagnostic comparator only.** It can show whether access to
  earlier feedback would fix preservation failures. It is not a proposed SRI
  mechanism: the target architecture keeps retention within learned structure,
  without an external experience store.
- **Separate O+'s confound (lower priority).** An extra-axis or repeated-axis
  control would show whether C2's gain under O+ comes from mixed evidence or from
  more updates.

These need a new contract.

Reproduce:

```bash
.venv/bin/python -m unittest tests.test_local_supervision -v
.venv/bin/python -m experiments.local_supervision
.venv/bin/python -m experiments.local_supervision --out-dir /tmp/lls-regen --workers 4
for f in manifest ceiling learning; do cmp reports/local-learning-supervision-v1-$f.json /tmp/lls-regen/local-learning-supervision-v1-$f.json; done
.venv/bin/python -m unittest discover -s tests -v
```
