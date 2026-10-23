# Learned reciprocal participation v1 — mechanism refuted in the declared family; no learning run

**Status:** `mechanism_refuted_in_declared_family`. No teacher among T1–T3 passes the
phase-A gate. Each of the three passes six of the seven conditions and fails the
same one: **feedback necessity** (condition 3). Input revision does happen, on 16
of 81 queries for each of these teachers. But on all 81 queries the response set
equals the response set obtained with the input frozen at its evidence. Under the
contract's stop rule (§5), phase B was **not run**: no learning campaign, no
learning metrics, and no phase-B report. Nothing in the model, the teachers or the
gate was changed.

[Contract](learned-reciprocal-participation-v1-contract.md) ·
[phase-A report](../reports/learned-reciprocal-participation-v1-phase-a.json) ·
[split manifest](../reports/learned-reciprocal-participation-v1-splits.json) (frozen
before phase A; sha256 recorded in the phase-A report) ·
[runtime](../reports/learned-reciprocal-participation-v1-runtime.json). Phase A
itself takes about 0.6 s. A full run, including the in-process tests and the
post-gate diagnostic, takes about 19 s.

## Verification before the gate

- **Independent oracle.** Production dynamics were compared with a depth-first
  traversal written from the contract's energy formula. The oracle uses no
  production energy, delta, field or traversal helper. Reachable sets, full
  terminal configurations and Y agree on all 59,049 (hypothesis, query) cases,
  and every graph is acyclic.
- **Controlled tests.** These cover ties, truncation (zero cap, exact boundary,
  cap − 1), simultaneous opposing contributions (`w_1A·i1` and `w_KA·k` cancelling
  to a zero field), neighbor dependence of every local rule for all 729
  hypotheses, core clamps, control invariants, fixed-input factorization for all
  729 core-disconnected hypotheses, label-free splits, contradictory feedback, and
  learner API and update rules.
- **Ties.** Zero-change candidates occur in 1,152 of the 59,049 canonical cases,
  none of them under the four teachers. No tie ever becomes an edge.
- **Test totals.** 25 new tests; the full suite has 160 tests, all passing.
- **Determinism.** A regeneration into a temporary directory produced
  byte-identical phase-A and manifest files.

The learner classes (version space, positive-only, block learner, memory) exist
and are unit-tested, including on declared training streams, as §10 requires.
Those tests check correctness properties only. They produce no learning metrics,
and no learning campaign was run.

## Phase-A gate (all 81 queries, every teacher)

| Condition | T0 (independent control) | T1 | T2 | T3 |
| --- | --- | --- | --- | --- |
| 1. State-conditioned influence (co-reachable pairs) | **fail**: no co-reachable pair | pass | pass | pass |
| 2. Realized input revision: possible / at every terminal | **fail**: 0 / 0 | 16 / 16 | 16 / 16 | 16 / 16 |
| 3. Y ≠ input-frozen Y | **fail**: 0 | **fail**: 0 | **fail**: 0 | **fail**: 0 |
| 4. Core contribution (Y and participation differ; no fixed k reproduces all) | **fail**: 0; every fixed k reproduces 81 | pass: 24, 24; fixed k reproduces 33 / 57 / 33 | same as T1 | same as T1 |
| 5. Selective participation: unique nonempty signatures | pass: A, B, AB | pass: A, B, AB; 48 queries | pass | pass |
| 6. Local exclusions preserved (domain violations / evidence mutations) | pass: 0 / 0 | pass: 0 / 0 | pass: 0 / 0 | pass: 0 / 0 |
| 7. Core-mediated clamp influence, B→A / A→B | **fail**: 0 / 0 | pass: 69 / 69 | 69 / 69 | 69 / 69 |
| **Gate** | not a gate teacher | **fail (3)** | **fail (3)** | **fail (3)** |

Details:

- **Condition 1.** For T1–T3, influence on satellite moves comes only through the
  core: 24 queries where `k` changes whether a satellite move decreases energy.
  No co-reachable pair shows the input changing a satellite move (0 queries).
  The reverse direction, a satellite changing an input move, occurs on 16
  queries, and the core responds to satellites on 56. For T0 the input never
  moves, so no co-reachable pair differs in I. Influence exists in T0's
  unrestricted state space (158 query-direction instances), but it does not
  count for the gate.
- **Condition 5.** Mixed-sign joint outputs occur on 24 queries for T1 and T3, on
  8 for T2, and on 8 for T0.
- **Condition 6.** Cross-satellite (non-rectangular) exclusions occur on 8 queries
  for each of T1–T3. These are dynamical exclusions, not a global incompatibility
  theorem.
- **Shared aggregates.** T1–T3 have identical aggregate counts in every row except
  the mixed-sign count. Their per-query sets differ.

### Failure certificate for condition 3

For each of T1, T2 and T3, Y equals the input-frozen response set on 81 of 81
queries, while input revision occurs on 16. The participation-change count and
the sign-only-change count are both 0. An independent oracle confirms this,
including with the input coordinates frozen, in
`test_declared_teachers_have_no_feedback_effect_on_responses`. Input revision is
real but has no effect on the response. A typical trajectory, T1 with evidence
(−1, 0) and both domains D0:

1. `a: 0→−1`, driven by i1.
2. `k: 0→−1`, following A.
3. `b: 0→1`, recruited through `w_KB`.
4. `i2: 0→−1`, revised toward B.

The revision comes after B is already committed and only reinforces it. The
terminal outputs are the same with I frozen.

**Structural explanation (exhaustive, not a general proof).** In all four teachers,
each input coordinate couples to exactly one satellite: i1 to A and i2 to B in
T0–T2, and the crossed assignment in T3. Of the 729 coefficient vectors, 225 have
this property, and **none** of them shows a feedback effect on any query. This
was verified exhaustively in `test_single_satellite_input_couplings_never_change_responses`.
I have not found a short analytic proof. The intuition is that an input coordinate
can move only toward the sign already held by its single satellite, which
reinforces a commitment that already exists. For the declared teachers,
condition 3 therefore fails by construction of their coupling pattern, not by
chance.

**Scope of the refutation.** The post-gate diagnostic in the phase-A report finds
312 of the 729 hypotheses with Y ≠ input-frozen Y on some query. Every one of them
has an input coordinate coupled to both satellites. None was evaluated against
the gate or used in any way. The contract forbids adding teachers, and this is a
scope statement, not a search. "Refuted in the declared family" means these four
teachers. It does not say that the energy candidate cannot realize feedback
necessity.

### What did work (scoped, T1–T3)

- **Core-dependent participation.** Removing the core couplings changes
  participation on 24 queries, all of them participation changes rather than
  sign-only changes. In the example above, with the core disconnected the output
  is `{(−1, 0)}` instead of `{(−1, 1)}`: the core recruits B. No single fixed `k`
  reproduces the full responses (33, 57 and 33 of 81 queries for k = −1, 0, 1).
  The fixed-input independent product mismatches Y on 24 queries, which refutes
  only that scoped control.
- **Clamp interventions.** In both directions, B→A and A→B, 69 clamp pairs change
  the other satellite's terminal set, and every one of those differences
  disappears when the core couplings are removed (0 not core-mediated). These are
  initialization interventions, not canonical-start trajectories.
- **Selectivity without equality.** The unique nonempty signatures include A, B
  and AB. Mixed-sign joint outputs occur. There are 8 queries with
  non-rectangular cross-satellite exclusions; for example,
  `{(−1,−1), (−1,0), (0,−1)}` excludes `(0,0)`.
- **Terminal multiplicity.** 73 queries have one terminal and 8 have three. The
  primary prediction is the full set Y; no terminal is selected.

T0, the independent control, behaves as an independent control should. The core
never moves, the input never revises, every fixed `k` reproduces it, and it
matches the independent product exactly.

## Controls

| Control | Queries with Y ≠ full, T1–T3 (each) | T0 |
| --- | --- | --- |
| Uncoupled (all zero) | 56 | 56 |
| Core disconnected | 24 | 0 |
| Input frozen | **0** | 0 |
| Input feedback removed | 0, identical to input frozen as expected | 0 |
| Core fixed at −1 / 0 / 1 (initialization intervention) | 48 / 24 / 48 | 0 / 0 / 0 |
| Independent satellite product (fixed input, no core) | 24 | 0 |

## Proofs (scoped)

1. **Termination and acyclicity.** Every edge strictly lowers an integer energy,
   and the reachable space is finite (at most 243 states). So the graph has no
   cycle, and every sequence of actual updates terminates. No claim is made about
   schedulers that refuse available moves. The input-feedback-removed control is
   not a single-potential descent. From the canonical start, an I move would
   raise the evidence-only penalty from 0, so I never moves and the control
   coincides with input-frozen (tested). Acyclicity is also checked by DFS on
   every graph.
2. **Reachable-terminal completeness.** Uncapped BFS examines every
   (coordinate, value) candidate at every dequeued state, so every reachable state
   is visited. This is shown by induction on path length, as in the earlier study.
   A terminal is a fully examined state with no decreasing candidate, and Y is
   their projection. The oracle agrees on 59,049 cases. A capped run returns
   `truncated` with no Y, and lists only fully examined terminals as discoveries.
3. **Constraint preservation.** Candidates are drawn only from the coordinate's
   allowed values, and the query is frozen and never written. So every
   intermediate state respects the domains and the evidence (checked over every
   reachable state).
4. **Reciprocal local-field dependence.** For a satellite move `a→a'`,
   `ΔE = 2(a'²−a²) − h_A·(a'−a)` with `h_A = w_1A·i1 + w_2A·i2 + w_KA·k`. Whether
   it decreases energy depends on k exactly when `w_KA ≠ 0`, and on i1 exactly
   when `w_1A ≠ 0` (tested for all 729 hypotheses). Symmetrically, an i1 move has
   field `w_1A·a + w_1B·b`, and k has `w_KA·a + w_KB·b`.
5. **Fixed-input, core-disconnected factorization.** With I fixed at e and
   `w_KA = w_KB = 0`, the energy is
   `E = c(e) + k² + [2a² − h_A a] + [2b² − h_B b]`, where `h_A` and `h_B` are
   constants. The core starts at 0, and any k move raises `k²`, so k never moves.
   An A move changes only the A bracket, and its decrease condition depends only
   on a, and likewise for B. The transition graph is therefore the Cartesian
   product of the A and B graphs, reachable states are `R_A × R_B`, and
   `Y = T_A × T_B` (tested for all 729 disconnected hypotheses).
6. **Learning propositions (implemented and unit-tested, not exercised by a
   campaign).** Clean realizable feedback retains the teacher. V is an
   intersection of per-item filters, so it is order-independent and every
   survivor reproduces every observed set. With the teacher in V,
   `certain ⊆ Y_T ⊆ possible`, and a certified set (all survivors agree) equals
   `Y_T`. These hold for the classes in `src/reciprocal_participation.py`.

## What this does not establish

- No learning result of any kind. Phase B was not run.
- No claim about the 312 untested hypotheses. Whether any of them satisfies all
  seven conditions is unknown, and the contract did not authorize searching for
  one.
- The core's participation effects are dynamical facts about this energy
  candidate. They are not evidence that numerical energy descent is SRI's
  foundation. A general transition table could reproduce these finite dynamics.
- There is no recovery, schedule-independence or convergence claim beyond the
  termination of actual updates.

## Next obligation (for a new contract, not run here)

The failure points at one structural requirement: for input revision to matter,
an input coordinate must couple to more than one satellite, so that revision
driven by one satellite can change another's options. A follow-up contract could
declare, before any evaluation, teachers with such shared input couplings. It
should keep this gate unchanged and report whether feedback necessity then
coexists with conditions 1, 2 and 4–7. The current contract does not allow that
change here.

Reproduce:

```bash
.venv/bin/python -m unittest tests.test_reciprocal_participation -v
.venv/bin/python -m experiments.reciprocal_participation
.venv/bin/python -m experiments.reciprocal_participation --out-dir /tmp/lrp-regen
cmp reports/learned-reciprocal-participation-v1-phase-a.json /tmp/lrp-regen/learned-reciprocal-participation-v1-phase-a.json
.venv/bin/python -m unittest discover -s tests -v
```
