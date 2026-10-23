# Shared core abstraction v3 — no complementarity exists in the declared pool: every source's constraints form a chain

**Completion:** `complete`. Every correctness gate passed. The full census ran:
655,340 Stage A tasks, 1,310,680 Stage B tasks, and the unmatched envelope
(`2^30` source-subset pairs per teacher/split, 20 teacher/splits).

**Headline:**

- **The two-teacher target is not met, and no teacher passes at any level.**
  No structural complementarity occurs in any Stage A task, for any teacher or
  split. Behavioral complementarity, full witnesses, executable successes, and
  compensating passes are therefore all zero as well.
- **The unmatched envelope is empty too.** No allocation of the allowed
  observations works, whether the two sources' query sets are identical,
  disjoint, or arbitrary. So the absence is not a matched-presentation effect.
- **Exact certificate.** For every teacher and split, the masks reachable by A
  together with those reachable by B form a **chain** under inclusion. Any
  intersection `C_A ∩ C_B` is then just the smaller of the two masks. Two
  ambiguous sources can never combine into an identification.
- **S1 and C2.** Every single axis observation either leaves all five sign
  classes (uninformative) or pins the opposing class `±(3,-3)` (identifies).
  Neither satellite ever holds a partial constraint. This is v2's redundancy,
  now shown to be exhaustive over all subsets of the pool.
- **S2 and T1.** Their partial constraints exist but coincide. Each one
  excludes the same class in both satellites: opposing for S2, matching for T1.
  So combining the two sources adds nothing.
- **The obstruction is the allowed observations under this relation family.**
  It is not the presentation (the envelope is empty) and not executable
  private acquisition (the true-block ceiling is already empty).

[Contract](shared-core-abstraction-v3-contract.md) (sha256 `e8b27809…`) ·
[manifest](../reports/shared-core-abstraction-v3-manifest.json) ·
[ceiling](../reports/shared-core-abstraction-v3-ceiling.json) (Stage A and envelope) ·
[selector](../reports/shared-core-abstraction-v3-selector.json) (Stage B) ·
[runtime](../reports/shared-core-abstraction-v3-runtime.json).

**Provenance.** The reviewed contract, manifest, code, and fixture tests were
committed in `3adea62` before the scored census. Fixture tests used only
hand-built masks and random non-teacher vectors. The runtime file marks the
worktree dirty only because the run wrote new report files.

## Pre-implementation review

The draft was reviewed before freezing. Five changes were made, all recorded
in the contract header:

1. The draft's 30-marginal envelope bound was vacuous. It was replaced by the
   full pair space, with identical, disjoint, and any-overlap counts.
2. Analytic consequences were disclosed. Structural complementarity on the true
   branch makes the installed model equal the teacher, so several section 7
   conditions are automatic. Condition 6 equals the behavioral clause.
3. Stage A is independent of the assignment, and its true-branch masks are
   reused in Stage B. Executable success therefore equals level 3 plus
   selection of the true branch.
4. Exact definitions were added for non-true branches, source omission, fixed
   cores, witness order, cross-context targets, and failure labels.
5. Two ambiguous sentences were rewritten.

## Correctness gates (all passed)

| Gate | Result |
| --- | --- |
| Prior-artifact hashes (v1, v2 code, contracts, results and reports; splits) and contract hash | yes |
| Incremental subset masks equal direct reduction (all `2^15` subsets, both sources, 20 teacher/splits) | yes |
| Teacher core survives every atomic mask on the true branch; atoms closed under `c ↦ -c` | yes |
| Flat centralized single-mask reference equals `C_A ∩ C_B` on every task | yes |
| Repeated paired pass and duplicate source leave masks unchanged | yes |
| Envelope: convolution equals the four-choice joint DP; totals `4^15`, `3^15` (disjoint), `2^15` per source; envelope contains every matched count | yes |
| Envelope DP and convolution match brute force on hand-built examples (tests) | yes |
| v2 private masks reproduced (40 runs); v2 assigned-query core masks rebuilt from v3 atoms on every branch | yes |
| Sequential v2 `CoreFilter` over the five frozen orders equals the full-set mask on every branch | yes |
| Stage B true-branch masks equal Stage A on every task | yes |
| Private blocks unchanged by installation; serialization round trip of every installed vector (all 81 response sets) | yes |
| Every response set used (7,238) against the independent oracle | 0 mismatches |
| Byte-identical regeneration (manifest, ceiling, selector) in a temporary directory | yes |
| Tests | 14 new; full suite 271, all passing |

## Stage A — complementarity ceiling on true private blocks

### Atomic observations

Each training query gives each satellite one atomic mask. Classes are named by
their minimal-change representative. `opp` = `±(3,-3)`, `match` = `±(3,3)`,
`A` = `±(3,0)`, `B` = `±(0,3)`, `none` = `c0`.

| Teacher | Atomic masks that occur (either satellite) | Informative atoms per split, A / B |
| --- | --- | --- |
| S1 | all 5 classes; `{opp}` | 1–3 / 5–6 |
| C2 | all 5 classes; `{opp}` | 5–6 / 5–8 |
| S2 | all 5 classes; `{none, A, B, match}` (opp excluded) | 1–3 / 5–6 |
| T1 | all 5 classes; `{opp}`; `{none, A, B, opp}` (match excluded) | 3–4 / 3–4 |

The reachable masks per source are `{all, opp}` for S1 and C2,
`{all, all−opp}` for S2, and a subset of `{all, all−match, opp}` for T1. For
S1 and C2, every informative atom already identifies the relation. That is the
contract's section 5 implication case.

### Task patterns (all 32,767 subsets × 5 splits per teacher)

Patterns are `(state of C_A / state of C_B / state of C_AB)`.

| Teacher | Pattern | Tasks |
| --- | --- | --- |
| S1 | identifies / identifies / two_link | 118,784 |
| S1 | uninformative / identifies / two_link | 41,984 |
| S1 | uninformative / uninformative / ambiguous | 3,067 |
| C2 | identifies / identifies / two_link | 159,232 |
| C2 | identifies / uninformative, or uninformative / identifies | 3,712 |
| C2 | uninformative / uninformative / ambiguous | 891 |
| S2 | ambiguous / ambiguous / ambiguous | 118,784 |
| S2 | uninformative / ambiguous / ambiguous | 41,984 |
| S2 | uninformative / uninformative / ambiguous | 3,067 |
| T1 | identifies on at least one side / … / two_link | 158,464 |
| T1 | ambiguous or uninformative on both sides / ambiguous | 5,371 |

Structural complementarity needs the pattern `ambiguous / ambiguous /
two_link`. It never occurs. The closest cases fail in the two ways the
contract anticipated:

- **v2 donor pattern** (uninformative + identifying): 41,984 S1 tasks and
  3,712 C2 tasks.
- **v2 redundancy pattern** (both identify): the majority of tasks for S1,
  C2 and T1.

S2's and T1's `ambiguous / ambiguous` tasks have the *same* ambiguous mask on
both sides, so their intersection stays ambiguous.

The full 15-event anchor is `identifies / identifies / two_link` for S1, C2 and
T1 in every split. For S2 it is `ambiguous / ambiguous / ambiguous`, keeping
`c0`, as in v2.

### Envelope (unmatched allocations)

Counts are zero for structural, behavioral, level-3, and route predicates, at
every total budget from 0 to 30 and at all three overlap levels, in all 20
teacher/splits. The closure certificate is exact. Each source's closure has
two or three masks, and their union is totally ordered by inclusion in every
case. So for any `(J_A, J_B)`, `C_A ∩ C_B ∈ {C_A, C_B}`. The predicate would
need `|Q_A|, |Q_B| ≥ 2` and `|Q_AB| = 1`, and these are then incompatible.

### Controls on the true branch

- **Fixed cores.** Only the teacher's own class passes v2's `CP`, `TR`, and
  core-dependence checks: `±(3,-3)` for S1, C2 and T1, and `±(3,3)` for S2.
  This is unchanged from v2.
- **Other controls.** There are no level-3 tasks, so the single-source-choice
  and fixed-core-at-witness tallies are empty (denominator 0). Memorization
  answers no mixed query.

## Stage B — executable census

| Teacher | Tasks | True branch selected | Wrong branch (runs) | Inconsistent | Structural / full / executable / compensating |
| --- | --- | --- | --- | --- | --- |
| S1 | 327,670 | 323,575 | 4,095 (1 run: split 4 base) | 0 | 0 / 0 / 0 / 0 |
| C2 | 327,670 | 316,928 | 10,742 (all 10 runs) | 0 | 0 / 0 / 0 / 0 |
| S2 | 327,670 | 323,575 | 4,095 (1 run: split 4 base) | 0 | 0 / 0 / 0 / 0 |
| T1 | 327,670 | 327,639 | 31 (1 run: split 0 base) | 0 | 0 / 0 / 0 / 0 |

- **Every task is labeled `not_structural`.** This follows from Stage A,
  because the true-branch masks are identical. On the wrong branches, `C_AB`
  was never a single two-link class with two ambiguous sources either.
- **C2 branch selection.** For small `J`, C2 selects the wrong branch
  `((0,-3),(0,-3))`. Most of those tasks install `(-3,3)`, the rest `c0`.
  This is the compensation risk disclosed in contract section 4.1. On the full
  15-event set, every run of every teacher selects the true branch.
- **Installed cores.** S1, C2 and T1 install `(-3,3)` on the true branch in
  most tasks, and `c0` when neither source identifies the relation. S2 always
  installs `c0`.
- **Source omission, end to end.** Omitting a source changes the installed core
  or branch in many tasks. Examples: S1 omit-B changes the core in 83,968
  tasks, and C2 omit-A changes the branch in 38,400. These are changes from
  losing an *identifying* source, not from losing a partial constraint. No
  executable task exists to score.

## Readings

1. **The negative is complete and exact, not a search failure.** Every subset
   and every unmatched allocation was enumerated. The chain certificate shows
   why no allocation can work.
2. **Matched presentation is not the obstruction.** The disjoint and
   any-overlap envelopes are empty as well.
3. **Executable private acquisition is not the obstruction.** The
   oracle-supplied true blocks already fail.
4. **What the pool lacks.** Under the nine-core family, axis observations
   split the classes in only one nested way per teacher. Complementarity would
   need two sources whose informative atoms exclude *different* classes, with
   only the teacher class in common, such as `{opp, match}` against
   `{opp, A}`. No axis query in any split produces such a crossing partition
   for any teacher.
5. **This is the contract's "allowed observations" category,** relative to
   the candidate relation family. Whether other query types or another core
   grammar would produce crossing partitions is outside this contract. Section
   11 forbids expanding the pool here.

## What this does not establish

- An impossibility result for other queries, readouts, core families, or SRI
  architectures. The certificate covers the 15-query axis pools of these five
  splits, the nine-core family, and these four teachers.
- Any claim about dreaming, learned alignment, or preservation under later
  interference. None was tested.
- A defect in v2. v2's S1 donor witnesses are exactly this study's
  "uninformative + identifying" tasks under that study's assignment. v3's
  stricter criterion excludes them by design.

## Accounting

| Item | Value |
| --- | --- |
| Atomic filter evaluations: Stage A / Stage B (all branches) | 5,400 / 33,480 |
| Isolated filter evaluations (Stage B, 40 runs, 5 orders + repetition) | 15,475 |
| Subset mask updates (Stage B, all branches) | 8,126,216 |
| Simulation requests / unique response sets | 357,790 / 7,238 |
| Logical mask state per task (max, C2: 9 branches) | 180 bits |
| Logical per-branch subset table (max) | 5,308,416 bits |
| Persisted after installation | 7 coefficients |
| Wall time / peak RSS (whole process) | ≈25 s / 65 MiB |

## Next obligations (not run)

1. The stopping rule applies. Do not add queries, readouts, teachers, or core
   capacity inside v3.
2. Any follow-up must first declare an observation family and show, before
   scoring and by atomic masks alone, that the two sources' informative atoms
   are not nested. Without that, complementarity is ruled out by the chain
   argument above. Such a study needs a new contract.
3. The v2 exact-mask reference remains the upper comparator for any later
   endogenous rule. Dreaming and the deferred learning-rule studies stay out of
   scope.
