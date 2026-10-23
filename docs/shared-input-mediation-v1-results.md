# Shared input interpretation v1 — a feedback chain through the input exists for two declared teachers

**Status:** `mechanism_witness_found_for_declared_teachers`. Two of the six
declared teachers pass the full gate (v1 conditions 1–7, plus 3′ and 8): **S1**
(shared input coordinate) and **C2** (internally coupled input). S2, S3, C1 and
C3 fail, and every failed condition is reported below. As the contract requires,
the study stops here: no learning was run and no teacher was searched for.

[Contract](shared-input-mediation-v1-contract.md) (sha256 `a9163a13…`, recorded in
the manifest before scoring; the contract file was not edited afterwards) ·
[manifest](../reports/shared-input-mediation-v1-manifest.json) ·
[gate report](../reports/shared-input-mediation-v1-gate.json) ·
[runtime](../reports/shared-input-mediation-v1-runtime.json). A full run takes
about 35 s and makes 44,386 recorded scoring explorations; the exact count
is in `scoring_runs` in the gate report.

## Verification

- **Oracle, family C.** A traversal written from the contract formulas matches
  production on all 81 queries × 2,187 coefficient vectors. Reachable states,
  full terminals and Y all agree, and every graph is acyclic. Those 177,147 cases
  include the 59,049 family-S cases, which are not counted as extra coverage.
- **Reduction to v1.** With `w_12 = 0`, the new module equals the frozen v1
  implementation on all 59,049 v1 cases: reachable sets, terminals, Y, and v1's
  legacy feedback-removed control.
- **Every scoring run rechecked.** Each recorded exploration used to score 3′ and
  8 was re-checked against the oracle, including NF's directed rule, every clamp
  and every edge ablation. In each clamped run the clamped coordinate never
  moves and stays in its domain.
- **Regression.** v1 teachers T0–T3, rerun in the new module, reproduce the frozen
  v1 phase-A conditions 1–7 exactly. All four v1 source and contract hashes still
  match the v1 report.
- **Tests.** 14 new tests; the full suite has 174, all passing.
- **Determinism.** Regenerating into a temporary directory gives byte-identical
  gate and manifest files.
- **Disclosure.** Three fixture tests used the S1 or C1 vectors. They checked
  nonempty control outputs, the existence of a multi-terminal query, fresh
  intervention runs and clamp domains. A truncation fixture used a vector that
  differs from C1 only in `w_1A`. They ran after the manifest was frozen and before the gate was
  scored. They checked no gate condition, and no protocol choice depended on them.

## Gate

| Condition | S1 | S2 | S3 | C1 | C2 | C3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 state-conditioned influence | pass | pass | pass | pass | pass | pass |
| 2 input revision: possible / every terminal | 24 / 8 | 32 / 32 | 32 / 32 | 54 / 46 | 46 / 30 | 54 / 46 |
| 3 Y ≠ input-frozen (queries) | **12** | **0 ✗** | **0 ✗** | 22 | **40** | 22 |
| 4 core contribution (Y, participation; fixed k −1/0/1) | 36, 36; 27/45/27 | 8, 8; 39/73/39 | same as S2 | 14, 14; 29/67/29 | 40, 40; 22/41/22 | as C1 |
| 5 unique signatures A, B, AB; mixed-sign queries | yes; 16 | yes; 0 | yes; 32 | yes; 38 | yes; 24 | yes; 38 |
| 6 domain / evidence violations | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7 core-mediated clamp influence, B→A / A→B | 26 / 36 | 34 / 18 | 34 / 18 | 40 / 40 | 56 / 56 | 40 / 40 |
| 3′ B's projection differs under NF (activity / sign-only) | **12 (12/0)** | **0 ✗** | **0 ✗** | 4 (4/0) | **8 (4/4)** | 4 (4/0) |
| 8 qualifying (query, v, v′, r, r′) tuples | **24** | **0 ✗** | **0 ✗** | **0 ✗** | **8** | **0 ✗** |
| **Gate** | **pass** | fail | fail | fail | **pass** | fail |
| Reverse chain B → input → A (reported only) | 8 | 6 | 6 | 0 | 8 | 0 |

S2 and S3, and likewise C1 and C3, give identical aggregate counts. Their
per-query results differ. Mixed-sign counts differ within each pair.

## The passing witnesses

**S1** (`w = (3,3,3,0,3,−3)`, i1 shared). The first witness is query 28: evidence
(0, −1), `D_A = D0`, `D_B = D1`. Under `do(A=0)`, every terminal has i1 = 0 and
B's set is {0}. Under `do(A=1)`, the i1 terminal set becomes {0, 1}, and B's set
becomes {0, 1}. Clamping i1 at 0 versus 1, with A fixed at 0, gives B sets {0}
versus {1}. Under NF, or with the input clamped at evidence, the contrast
disappears (B = {0} in both runs). It also disappears when `w_1A` or `w_1B`
alone is removed. There are 24 qualifying tuples on 12 queries. Six of those
queries are also condition-3′ queries.

**C2** (T1 with `w_12 = −3`). The first witness is query 37: evidence (0, 0),
`D_A = D0`, `D_B = D1`. Under `do(A=1)`, A pushes i1 to 1. The internal coupling
then pushes i2 to −1, and `w_2B = −3` makes b = 1 available: B's set is {0, 1}
against {0} under `do(A=0)`. The recipient clamps i2 = 0 versus −1 give B sets
{0} versus {1}. The contrast disappears under NF, with the input clamped at e,
and when any one of `w_1A`, `w_12` or `w_2B` is removed. There are 8 qualifying
tuples on 4 queries. **None of them is a condition-3′ query.** C2's
output-level feedback effects on canonical starts (8 queries) and its
interventional chain lie on different queries.

**What the witnesses do and do not show.**

- **Available, not forced.** In every first witness, A's change makes b = 1
  *available*: B's set grows from {0} to {0, 1}, depending on the schedule. It
  does not force B to activate. This matches the contract's "available
  responses", not deterministic recruitment.
- **The chain does not need the core.** For every qualifying S1 and C2 pair, the
  A→B contrast persists with the core disconnected and with each off-chain
  coefficient removed. Core contribution (condition 4) and the input chain are
  separate facts about the same teacher. This is not a core-mediated input chain.
- **Other input values allow a bypass.** With the input clamped at u = e there is
  no contrast, as (c) requires. At other common input vectors there is one: S1
  shows a contrast at every u with i1 equal to the satellite-driven value
  (u1 = 1 or −1), and C2 at u2 = −1 or 1. With the input pinned there, A still
  changes B through another route, which the core-disconnected run attributes to
  the core. Condition 8 is an interventional set-level witness at e, not proof of
  a unique route.

## Failure certificates

- **S2 and S3 fail 3, 3′ and 8.** Freezing the input, or removing satellite
  feedback, never changes Y on any of the 81 queries, although input revision
  occurs on 32. For condition 8, 36 A-clamp pairs survive (c) and (a). In all 36,
  removing `w_1B`, the i1–B link, leaves the contrast intact, so (d) fails. A
  parallel route, carried by the core, makes the input route redundant. That is
  a bypass, not an absence of input influence; with matching core signs (S2)
  this was flagged as a risk before implementation. The reverse chain does pass
  for both (6 tuples).
- **C1 and C3 fail 8.** Of 150 A-clamp pairs with a B contrast, 12 lose the
  contrast under NF and 36 with the input clamped at e, but **none under both**.
  Every A→B contrast is therefore carried at least partly by something other than
  satellite feedback into the input. 70 of the 150 survive core disconnection.
  The reverse chain fails the same way.
- **Archive.** Subcondition masks for every query, ordered clamp pair and
  recipient-value pair, plus the joint counts, are in the gate report
  (`condition_8_archive`, `joint_pair_counts`). Any failure can be checked
  independently from them.

## Ablations (queries whose Y changes; full table in the report)

| Removed | S1 | C2 |
| --- | --- | --- |
| A–input link (`w_1A`) | 40 | 60 |
| Input–B link (`w_1B` for S1, `w_2B` for C2) | 56 | 60 |
| `w_12` | — | 40 |
| Core link, `w_KA` / `w_KB` | 36 / 36 | 40 / 40 |
| Input frozen | 12 | 40 |
| NF (Y / A projection / B projection / signatures) | 12 / 0 / 12 / 12 | 16 / 8 / 8 / 8 |

For C2, NF and full input freezing differ: 16 queries change Y under NF against
40 under freezing. That difference is the input-internal processing that NF
preserves and freezing removes. It is the confound that 3′ and 8 were designed to
separate.

## Proofs (scoped)

1. **Termination and acyclicity under `E_C`.** Every edge lowers an integer
   energy on at most 243 states.
2. **NF terminates.** An input update strictly lowers `E_I`. K, A and B updates
   leave `E_I` unchanged (it depends only on i1 and i2) and strictly lower
   `E_C`. So `(E_I, E_C)` decreases lexicographically on every actual update.
   That holds on a finite space and under any clamps, because frozen coordinates
   never update.
3. **Reduction at `w_12 = 0`.** The extra term vanishes, so `E_C = E_v1`. The
   NF input rule then reduces to v1's evidence-only rule. This holds identically
   and was checked on 59,049 cases.
4. **Local fields.** An i1 move has field `w_1A·a + w_1B·b + w_12·i2`, and i2 has
   `w_2A·a + w_2B·b + w_12·i1`. Under NF, only the `w_12` term remains. So each
   declared chain link appears in the corresponding local difference
   (fixture-tested for both signs of `w_12`).
5. **Three levels of dependence.** Algebraic dependence does not imply that a
   reachable move crosses its threshold, and that does not imply an output
   effect. These are reported separately: condition 1 for reachable witnesses,
   and the ablation table and conditions 3, 3′ and 8 for outputs.
6. **Completeness and preservation.** Reachable-terminal completeness and
   domain/evidence preservation follow as in v1. Moves are drawn only from
   allowed values, and the query is immutable. Every complete run has at least
   one terminal; this is asserted in code.

## Scope

- **Where it holds.** The result holds inside this signed-energy candidate, at
  mode level, on 81 queries.
- **Families.** Two passes among six hand-declared teachers, one per family. This
  cannot rank family S against family C.
- **Not established.** Learnability, robustness, recovery, semantics, a unique
  causal route, or natural mediation effects.
- **The core.** The passing chains do not require the core, so they are not
  evidence for core-mediated input interpretation.
- **Not universal.** A general transition table could reproduce any of these
  finite results.
- **Learning.** Under the contract, any later learning study must be a new
  contract using S1 and/or C2. This result authorizes no campaign by itself.
