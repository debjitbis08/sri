# Local update diagnosis v1 — the block learner's failures are mixed; no single-property variant suffices for both teachers

**Completion:** `complete`. Every correctness gate passed. Phase A diagnosed all
1,500 IB experiences of the reproduced block learner (B0). Phase B ran all
three declared variants in full: 4 teachers × 25 streams × 15 experiences each.

**Headline:**

- **No single failure type dominates.** Among B0's failing experiences
  (S1: 255 of 375; C2: 197 of 375), the largest class is an improving,
  non-breaking edit that leaves the current example inexact (E2a: 40% S1, 32%
  C2). Next come preservation conflicts, where every improving single-block
  edit breaks an earlier item (E2c: 22%, 35%). Stuck states that a two-block
  edit would improve are a minority (E1a: 21%, 18%), as is selection that
  ignores an available preserving edit (E2b: 12%, 11%).
- **Most forgetting comes from preservation conflicts.** Of B0's loss events,
  76% (S1) and 86% (C2) occur at E2c decisions; the rest occur at E2b.
- **Under a frozen example, the teacher class usually needs a plateau move.**
  From 72% (S1) and 64% (C2) of the states visited outside the teacher class,
  the class cannot be reached by strict descent on the current example. It can
  be reached by non-increasing moves. An uphill move is needed in 4% and 10%.
- **More updates on the same example help only sometimes.** Repetition makes a
  failing example exact without breaking an earlier item in 21% (S1) and 15%
  (C2) of failing experiences.
- **No variant meets L1–L3 for both S1 and C2.** REPLAY meets all three for C2
  (23/25 recovered, 96.8% final retention, 0.5% forgetting). It is the
  diagnostic comparator with an external experience store, and it spends
  about 7× B0's current-example predictions on the stored maps. SCOPE recovers
  S1 in 19/25 runs at 10× B0's candidate evaluations. REPEAT recovers 15/25 for
  both teachers at about 1.5×.

[Contract](local-update-diagnosis-v1-contract.md) (sha256 `cbcfbf89…`, frozen
and committed before the manifest; not edited afterwards) ·
[manifest](../reports/local-update-diagnosis-v1-manifest.json) (committed before
computation) · [diagnosis](../reports/local-update-diagnosis-v1-diagnosis.json)
(phase A) · [variants](../reports/local-update-diagnosis-v1-variants.json)
(phase B) · [runtime](../reports/local-update-diagnosis-v1-runtime.json).

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Battery and all 2,187 per-hypothesis hashes equal v1's; prior artifact hashes unchanged | yes |
| Teacher class equals the frozen IB complete-information class (teacher + mirror) for all four teachers | yes |
| Teacher class has zero loss on every current and cumulative IB objective | yes |
| B0 lockstep against the unchanged v1 `BlockLearner`: state, evaluations and commits after every experience, 100 streams | 0 mismatches |
| B0 against the frozen local-supervision report: timelines, checkpoint and final vectors (100 streams), 4,800 local/zero/teacher rows, learner work counters | 0 mismatches |
| Battery-derived losses equal the learner-facing service for every evaluated (candidate, query, key): B0 94,701; SCOPE 356,422; REPEAT 103,568; REPLAY 132,151 | 0 mismatches |
| `N1`/`N2` enumerations (26 / 240 distinct, disjoint); unrestricted N1 graph connected; `G↓ ⊆ G≤` for every frozen example | yes |
| All 6,694 decision labels in E0–E2c; no E1a for SCOPE; every REPEAT experience ends with a non-commit check; REPLAY commits exactly when B0 would at the same state | yes |
| Byte-identical regeneration (manifest, diagnosis, variants), temporary directory, 4 workers instead of 8 | yes |
| Tests | 22 new; full suite 231, all passing |

**Defect found before results.** The first scored launch crashed at its first
experience with a `TypeError`: the runner called the previous study's
`feedback()` with an extra argument. No experience had been processed and no
report was written. The call was fixed and the run restarted from the frozen
manifest.

**Provenance.** The scored run preceded the commit of its source, so the runtime
record marks the tree dirty. The source hashes recorded in the reports equal the
committed `src/update_diagnosis.py`, `experiments/update_diagnosis.py` and
`tests/test_update_diagnosis.py` exactly.

## Phase A — why B0 fails (IB, S1 and C2)

### A.1 Per-experience classes

| Class | S1: all / failing | C2: all / failing | Meaning |
| --- | --- | --- | --- |
| E0 already exact | 69 / 0 | 132 / 0 | 47 (S1) and 92 (C2) of these are outside the teacher class |
| E1a stuck; a two-block edit improves | 53 / 53 (20.8%) | 36 / 36 (18.3%) | restricted edit scope |
| E1b stuck; needs ≥ 3 blocks | 15 / 15 (5.9%) | 8 / 8 (4.1%) | 3 blocks: 12 and 4; 4 blocks: 3 and 4 |
| E2a improving, breaks nothing | 153 / 102 (40.0%) | 109 / 63 (32.0%) | failing only because the example stays inexact |
| E2b breaks; a preserving improving edit existed | 30 / 30 (11.8%) | 21 / 21 (10.7%) | tie-selection 16 / 11; current-loss trade-off 14 / 10 |
| E2c every improving edit breaks | 55 / 55 (21.6%) | 69 / 69 (35.0%) | preservation conflict |
| **Failing experiences** | **255 of 375** | **197 of 375** | |

In no failing E2a experience did the committed edit make the example exact. The
single commit always left loss behind.

### A.2 Under-updating (repetition probe)

| | S1 | C2 |
| --- | --- | --- |
| Failing experiences where repetition makes the example exact with no earlier item wrong at the end | 54 / 255 (21.2%) | 30 / 197 (15.2%) |
| … of which failing E2a | 36 / 102 | 23 / 63 |
| Experiences with at least one extra commit | 87 / 375 | 54 / 375 |
| Probes that lose an earlier item at some commit and restore it within the experience | 20 | 18 |

So one commit per experience is too few in some cases, mainly at E2a. But in
most E2a failures repetition also stops short: single-block descent on the
example stalls before it is exact.

### A.3 Progress toward the teacher class

States outside the teacher class: 353 (S1) and 335 (C2).

| | S1 | C2 |
| --- | --- | --- |
| Actual update: d decreases / same / increases | 70 / 237 / 46 | 63 / 252 / 20 |
| Best teacher-ward N1 edit is improving / neutral / worse | 189 / 138 / 26 | 168 / 129 / 38 |
| **Frozen-example graph: reachable by strict descent (G↓)** | 86 (24.4%) | 88 (26.3%) |
| **Only by non-increasing moves (G≤, plateau needed)** | 254 (72.0%) | 214 (63.9%) |
| **Not by non-increasing moves (uphill needed)** | 13 (3.7%) | 33 (9.9%) |
| G↓-reachable states where repetition reaches the class | 19 / 86 | 17 / 88 |

Every failing E1a and E1b state is in G≤ or needs an uphill move. None is in G↓.
Where a strict-descent route exists, greedy repetition finds the class in only
about a fifth of states. Shortest witness paths have 1–8 N1 edges, mostly 2–4. These
are bounds for one example held fixed. Later examples change the objective and
can open a route blocked here.

### A.4 Stream-end states

| | S1 | C2 |
| --- | --- | --- |
| End in the teacher class | 5 / 25 | 7 / 25 |
| Positive cumulative training loss | 20 | 18 |
| … and some single-block edit lowers it (not a local minimum) | 16 | 10 |
| … single-block but not two-block local minimum | 3 | 8 |
| … two-block local minimum | 1 | 0 |

Most failed end states are not local minima of the cumulative loss. A
single-block edit would lower it, but B0 never sees the cumulative loss: it
scores only the current example.

### A.5 Where items are lost

| | S1 | C2 |
| --- | --- | --- |
| Loss events caused by E2b / E2c decisions | 59 / 191 | 30 / 179 |
| Forgotten items whose last loss was E2b / E2c | 22 / 97 | 0 / 52 |
| Never-acquired items, presentation class E2a / E2c / E1a / E2b / E1b | 49 / 22 / 19 / 15 / 12 | 35 / 34 / 17 / 7 / 8 |

By definition only E2b and E2c decisions break items. The split shows that
conflicts (E2c), not ignored alternatives (E2b), account for most losses.

### A.6 Declared readings

All the declared patterns co-occur; none accounts for most failures.

- **E1a (restricted edits):** present in about a fifth of failing experiences.
- **E1b:** rare.
- **E2b (selection ignores preservation):** about a tenth, split evenly between
  ties and trade-offs.
- **E2c (preservation conflict):** a fifth to a third of failing experiences,
  and the source of most forgetting.
- **Under-updating:** repetition fixes 15–21% of failing experiences.
- **Plateau needed under a frozen example:** the most common graph category, at
  two-thirds or more of visited states.
- **End states:** mostly not local minima of the cumulative loss.

The diagnosis does not support a single-cause story, and it does not show that
cross-block coordination is the cause. Two-block edits would unstick about a
fifth of failures (E1a). The frozen-example graphs usually require at least one
plateau move; how many plateau moves a route needs was not measured.

## Phase B — single-property variants

| Variant | S1 recovered (worst split) | C2 recovered (worst split) | L2 within / mixed24, S1; C2 | L3 final retention; forgetting, S1; C2 | Candidates / stream | Predictions / stream (history) |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | 5 (1) | 7 (0) | 411/288; 455/290 | 37.1%; 46.1% · 59.2%; 19.0% | 405 | 2,300 (0) |
| SCOPE | **19** (3) | 12 (1) | **633/500** ✓; 536/353 | 79.5%; 14.9% · 69.1%; 18.6% | 4,005 | 22,748 (0) |
| REPEAT | 15 (2) | 15 (2) | 596/437; 588/428 | 73.1%; 14.9% · 76.8%; 11.4% | 647 / 606 | 3,707 / 3,473 (0) |
| REPLAY | 16 (2) | **23 (4)** ✓ | **601/485** ✓; **724/576** ✓ | 71.7%; 10.6% · **96.8%; 0.5%** ✓ | 405 | 2,300 (16,043) |

L2 counts are out of 750 / 600 (zero baseline: S1 275/140, C2 210/60).
Thresholds: L1 ≥ 23/25 with ≥ 4 per split; L2 ≥ 80% and above zero coupling; L3
≥ 95% retention and ≤ 5% forgetting.

**No variant suffices for both teachers.** REPLAY suffices for C2 only: "access
to earlier feedback suffices for this selection rule, for C2". It is not a
solution to retention without an experience store. SCOPE passes L2 for S1 only.

What each variant changed at the decision level (actual first decisions, versus
B0 counterfactually at the same states):

- **SCOPE** removes E1a by construction. Its E2c decisions fall sharply (S1 10,
  C2 28), but E2b rises (S1 49, C2 100). The larger neighbourhood usually
  contains a preserving improving edit, and current-loss selection usually
  passes it over. So the extra scope turns many conflicts into selection
  failures.
- **REPLAY** reduces E2b without eliminating it (S1 10 versus 19, C2 7 versus 15
  counterfactual at the same states). That is consistent with the contract's
  counterexample: minimizing summed past loss is not the same as preserving
  every correct item. Its trajectories also visit fewer E2c states.
- **REPEAT** uses 1.5–1.6× as many decisions as experiences (S1 599, C2 561 for
  375 experiences).

End states in the teacher class: B0 5 / 7, SCOPE 19 / 12, REPEAT 15 / 15,
REPLAY 16 / 23 (S1 / C2).

**Negative controls** (local runs with a forward witness, S2 / T1): B0 6 / 6,
SCOPE 15 / 11, REPEAT 2 / 0, REPLAY 0 / 0. SCOPE raises false attribution for
both controls. Under SCOPE, T1 ends outside its class in every run, with 157
items forgotten, against 17 under B0.

**Cost.** SCOPE's gain for S1 comes with 10× the candidate evaluations. REPLAY's
gain for C2 adds 16,043 history predictions per stream, 7× its current-example
predictions, and stores every earlier map. REPEAT's cost is the smallest
increase, at about 1.5×. No comparison here is compute-matched.

## What this does not establish

- **No unique cause.** The diagnosis describes states, objectives and
  neighbourhoods of this rule, and the patterns co-occur.
- **Frozen-example bounds.** The graph categories hold one example fixed. They
  do not show that a sequential learner must accept neutral or worse moves.
- **Variants are not compute-matched,** and each changes one property only. No
  combination was run.
- **REPLAY is outside the target architecture.** It uses an external store of
  earlier feedback.
- **Finite and easy setting.** Four hand-chosen teachers, IB supervision,
  noise-free complete-set feedback, 25 streams that share 5 splits, and a
  finite battery.

## Next obligations (not run)

- **Retention within learned structure.** E2c conflicts cause most losses, and
  REPLAY's success for C2 shows what access to earlier feedback buys. The open
  question is what mechanism inside the learned state, without an experience
  store, could supply some of that preservation signal.
- **Plateau moves.** Frozen-example graphs usually need a neutral move. A
  variant that may commit an equal-loss move would test this directly, with a
  cycle guard declared in advance.
- **Selection in a larger neighbourhood.** SCOPE turned conflicts into E2b. A
  study pairing scope with preservation-aware selection would combine two
  properties and needs its own contract and cost accounting.

These need a new contract.

Reproduce:

```bash
.venv/bin/python -m unittest tests.test_update_diagnosis -v
.venv/bin/python -m experiments.update_diagnosis
.venv/bin/python -m experiments.update_diagnosis --out-dir /tmp/lud-regen --workers 4
for f in manifest diagnosis variants; do cmp reports/local-update-diagnosis-v1-$f.json /tmp/lud-regen/local-update-diagnosis-v1-$f.json; done
.venv/bin/python -m unittest discover -s tests -v
```
