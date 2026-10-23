# Selective satellite mediation v1 — exact gated repair transfers; this grid barely tests exclusion learning

**Result:** the correctness gate passes and all 8 primary transfer targets pass.
The exact version-space learner repairs held-out conflicts by executing learned,
context-gated transformations. It never makes a domain, gate, path or evidence
violation. It never predicts a repair the teacher forbids (0 false-feasible
predictions in 100 runs). It transfers to the unseen context `11` exactly for T1
and T2, and at 0.960 for T0.

Several matched controls weaken what this pass means. They are reported here as
findings, not as footnotes:

- **The positive-only ablation is indistinguishable in execution.** In all 400 run
  checkpoints it selects the same hypothesis as full feedback, so its executed
  responses are identical. Exclusions shrink the version space a lot: at
  checkpoint 48, V has 1–2 survivors with full feedback against 16–256 without
  exclusions. They also decide what can be *certified*. They do not change what
  is *executed*. The smallest-id selection rule prefers `never` gates, and that
  conservative prior does the work the exclusions were meant to isolate.
- **No mediation already meets 2 of 3 conditions in several cells.** It scores
  0.916 on T0 within-context, which meets the 90% bar, and it is never
  false-feasible. Only the repair condition separates it from the learner.
  Exact-query memory repairs 0 held-out conflicts by construction, because its
  fallback for unseen queries is no mediation. So "strictly more repairs than
  memory" reduces to "at least one repair".
- **T3 passes before any learning.** T3 is hypothesis id 0. That is the initial
  selection, and the no-mediation control. Its negative-control pass comes from
  the selection rule.
- **T1's unseen context does not test selectivity.** In context `11`, T1 opens all
  four gates. Its unseen partition therefore has no teacher-permission
  exclusions, and the permissive control is also exact there (1.000).
- **The core is not distinguished.** The core-free control reproduces every
  version space and every response, as §7 of the contract predicted (0
  mismatches).

[Contract](selective-satellite-mediation-v1-contract.md) ·
[report](../reports/selective-satellite-mediation-v1.json) ·
[split manifest](../reports/selective-satellite-mediation-v1-splits.json) (sha256
`b79f639b…0224c`, frozen before any learner code existed) ·
[runtime](../reports/selective-satellite-mediation-v1-runtime.json). There are 700
runs (7 learners/controls × 4 teachers × 5 splits × 5 orders), 4 checkpoints each,
and about 45 s in total. Scientific output regenerates byte-identically; this was
checked by two consecutive runs.

## Correctness gate (all zero, tests 31/31)

| Check | Value |
| --- | --- |
| Hard-constraint / path violations (domain, gate, step, evidence; every executed prediction of every executor) | 0 |
| Observed-item retention failures, full feedback (16,800 checks, *every survivor* must reproduce each observed set) | 0 |
| Teacher eliminated (core, core-free, positive-only, all 4,800 updates) | 0 |
| Core-free V mismatches / response mismatches | 0 / 0 |
| Final V differs across the 5 orders of the same split (full and positive-only) | 0 |
| `certain ⊆ R_teacher ⊆ possible` violations | 0 |
| `inconsistent_feedback` events | 0 |
| BFS vs independent Warshall-closure oracle | 147,456 / 147,456 equal |

## Transfer targets at checkpoint 48 (pooled over 25 split/order runs)

Exact-set accuracy, false-feasible / teacher-empty queries, and exact repairs /
resolvable queries. Memory repairs are 0 in every cell.

| Teacher, partition | Core learner | No mediation | Permissive | Target |
| --- | --- | --- | --- | --- |
| T0 within | 0.990, 0/6275, 715/810 | 0.916, 0, 0 | 0.383, 4390, 300 | pass |
| T0 unseen | 0.960, 0/1950, 605/700 | 0.736, 0, 0 | 0.528, 1250, 500 | pass |
| T1 within | 1.000, 0/5010, 2075/2075 | 0.784, 0, 0 | 0.418, 3125, 640 | pass |
| T1 unseen | 1.000, 0/700, 1950/1950 | 0.347, 0, 0 | **1.000**, 0, 1950 | pass |
| T2 within | 1.000, 0/5220, 1865/1865 | 0.806, 0, 0 | 0.417, 3335, 625 | pass |
| T2 unseen | 1.000, 0/1475, 1175/1175 | 0.674, 0, 0 | 0.451, 775, 375 | pass |
| T3 within (negative control) | 1.000, 0/7085, — | 1.000, 0, — | 0.352, 5200, — | pass (zero false-feasible) |
| T3 unseen (negative control) | 1.000, 0/2650, — | 1.000, 0, — | 0.347, 1950, — | pass (zero false-feasible) |

The positive-only and core-free learners give the same numbers as the core
learner in every cell. The report lists each condition separately for every
learner and control. The controls were added to the target table after the first
run, as a reporting addition only. No learner, split, threshold or generator
changed, and the frozen manifest was verified on each run.

**Worst runs.** T0 split 3 has the worst run in every order: 0.951 within and
0.799 unseen. Its 16 context-`10` training queries never expose `A_low`, so two
hypotheses survive: T0 and `(never, never, c1, never)`. The smallest-id rule picks
the second. The cost is 95 + 95 missed repairs and 50 missed already-compatible
alternatives. All of them are under-generalization, and none is false-feasible.
Every T1 and T2 run is exact.

**Errors are always flagged.** No held-out error occurs where all survivors agree.
In the report, `wrong_with_survivor_agreement` is 0 in every cell. The learner's
own uncertainty marks every query it gets wrong (proposition 7 below). On T0
within-context, 235 pooled queries have an empty `certain` set and a nonempty
`possible` set. These are uncertain cases, not certified infeasibility.

## Learning curve (core learner, pooled accuracy; held-out changes vs checkpoint 0)

| Teacher, partition | cp 0 | cp 12 | cp 24 | cp 48 | right→wrong / wrong→right |
| --- | --- | --- | --- | --- | --- |
| T0 within | 0.916 | 0.958 | 0.976 | 0.990 | 0 / 715 |
| T0 unseen | 0.736 | 0.829 | 0.906 | 0.960 | 0 / 805 |
| T1 within | 0.784 | 0.922 | 0.965 | 1.000 | 0 / 2075 |
| T1 unseen | 0.347 | 0.646 | 0.830 | 1.000 | 0 / 2350 |
| T2 within | 0.806 | 0.943 | 0.983 | 1.000 | 0 / 1865 |
| T2 unseen | 0.674 | 0.930 | 0.977 | 1.000 | 0 / 1175 |

Checkpoint 0 equals no mediation, because hypothesis 0 is selected first. Sequential
exposure never turned a correct held-out prediction into a wrong one in this grid.
That is an observation. Earlier unobserved predictions *can* change; for T0 unseen,
50 changed from one wrong set to another. Version-space size at checkpoint 48 is
1–2 for T0, 1 for T1 and T2, and 1–3 for T3 (full feedback). For positive-only it
is 64–128, 16, 32 and 256.

**Witness shape.** Mean repair path length at checkpoint 48 is 1.0 for T0 and
1.20 / 1.49 for T1 (within / unseen). For T2 it is 1.11 / 1.19. Repairs that
change both satellites occur for T1 (410 within, 400 unseen) and T2 (200, 225),
and never for T0. No committed witness uses a gate the teacher would close
(`teacher_impermissible_steps` = 0). The witnesses section of the report has
example certificates.

**Selectivity versus domain safety.** The permissive control respects every
domain by construction, yet it is false-feasible on every teacher-permission
exclusion: 4,390 pooled for T0 within, and 5,200 for T3 within. The learner's 0
here is the selectivity the contract asks about. Treat it with the selection-prior
caveat above: it is partly inherited from the bias toward `never`.

## Incompatibility classes

Over all 576 queries the classes are 152 already compatible and 112 disjoint-domain
obstructions, for every teacher. Resolvable conflicts are 66 / 172 / 132 / 0 for
T0–T3, and teacher-permission obstructions are 246 / 140 / 180 / 312. The
**all-enabled obstruction class is empty** for every teacher (proposition 8). Per
split and partition counts are in `class_counts`.

## Proofs (scoped to the declared model)

1. **BFS soundness, termination, completeness.** States lie in `D_A × D_B`, at most
   16. A state is enqueued only once, when it is first discovered, so there are at
   most 16 dequeues and 64 primitive examinations. Soundness: a state is added only
   as an executable successor of a visited state, so by induction every visited
   state is reachable. Completeness when uncapped: by induction on the length of a
   reachable path, its predecessor is dequeued and all four primitives are examined
   there, so the next state is visited. `R_h(x)` is the set of visited `(t,t)`
   states, including successful states that are later expanded. BFS discovers
   states in nondecreasing distance, so the parent chain to the smallest `t` is a
   shortest path. Oracle test: all 147,456 cases agree, and path lengths match an
   independent layer computation.
2. **Committed paths preserve constraints and evidence.** A state is enqueued only
   if its gate is open in `c` and the destination lies in both domains. The start
   state is validated at query construction. `Query` is frozen, and inference
   never writes to it. The equality requirement is a fixed predicate. Invalid
   destinations are kept only in `invalid_attempts`. In the grid, `verify_path`
   re-checks every executed prediction independently of BFS.
3. **The teacher is retained.** Clean feedback is `R_T(x)`, so T satisfies every
   filter. The same holds for positive-only, because `R_T(x) ⊆ R_T(x)`.
4. **Observed sets are preserved, and the order does not matter.** After items
   `x_1..x_k`, `V = All ∩ ⋂_i {h : R_h(x_i) = f_i}`. Every survivor satisfies every
   earlier equality. Intersection is commutative and associative, so the final V
   depends only on the set of items. Positive-only is the same with `⊇`.
5. **`certain ⊆ R_T ⊆ possible`.** This holds because `T ∈ V` (proposition 3),
   given the realizable class and clean feedback.
6. **Core-free equivalence.** `direct_response` computes the same reachable set as
   BFS (both equal the closure; tested on all cases). The filter and the
   smallest-id selection are the same, so V and the selected model are identical
   at every step. The certificate is a by-product, not a computational resource.
7. **Agreement implies correctness.** If every survivor predicts the same set and
   `T ∈ V`, that set is `R_T(x)`. So every error of the selected model occurs on a
   query where survivors disagree.
8. **The all-enabled class is empty.** Each declared domain is connected under the
   XOR moves it permits. `{0,1}` is connected by `xor 1`, `{0,2}` by `xor 2`, and
   the full set by both. With every gate open, the reachable set is therefore
   `D_A × D_B`, and a shared mode is reachable exactly when the domains intersect.
   So "obstructed even with all gates enabled" coincides with disjoint domains,
   which are tested first.

The coincidence of the positive-only and full-feedback selections is **not
proved**. It was observed in all 400 checkpoints. A larger hypothesis order or a
different selection rule could separate them.

## Cost

- **Persistent learner state:** 256 bits plus schema metadata. With
  `sys.getsizeof`, the membership int is 60 bytes and the learner object about
  440 bytes (see the runtime file).
- **Learner update work (all 100 runs):** 103,252 hypothesis evaluations and
  847,628 edge checks with full feedback, against 217,200 and 2,308,904 for
  positive-only, which keeps more survivors. These counts are logical: each
  requested evaluation counts even when the pure inference cache served it.
- **Learner inference:** 219,600 executed predictions, with 1,310,464 edge checks.
- **Evaluator costs, kept separate:** a 147,456-inference evaluation cache and
  60,176,064 survivor-response lookups for `certain` / `possible`. The teacher
  feedback tables cost 2,304–6,304 edge checks per teacher.

## What this does not establish

- **No distinct-core advantage.** The core is a rule controller with a
  certificate, not an autonomous recurrent core. The core-free control is
  equivalent by proposition 6.
- **Agreement only.** The success predicate is `a == b`. A pass does not
  distinguish general compatibility mediation from constrained agreement.
- **The selection rule supplies a prior.** The zero false-feasible results and
  the equality with positive-only rest partly on smallest-id selection. This grid
  cannot separate learned selectivity from that prior.
- **Strong supervision.** Feedback is the complete response set, from a
  realizable teacher, without noise. The hypothesis language and generators are
  hand-designed, and the learner is a centralized exact version space, not a
  local plasticity law.
- **Weak transfer thresholds.** The 90% bar is close to the no-mediation baseline
  on T0 within-context. The memory comparison is satisfied by any single repair.
- **No dynamics.** This is centralized BFS, so no fairness, fixed-point,
  disturbance-recovery or local-convergence claim follows.
- "Unresolvable" is relative to these four generators and the teacher
  permissions.

## Next obligations (not run here)

As the contract directs, the next experiments should keep these exclusions and
transfer tests. They should either test a non-equality compatibility relation or
realize the learned transformations inside graded satellite dynamics. Two design
lessons from this grid:

- Use a selection rule, or an evaluation, that separates exclusion information
  from a conservative prior. Candidates are survivor-agreement abstention, or
  reporting `certain` / `possible` as the primary output.
- Include teachers whose unseen context still contains permission exclusions.

Reproduce:

```bash
.venv/bin/python -m unittest tests.test_selective_mediation -v
.venv/bin/python -m experiments.selective_mediation
.venv/bin/python -m unittest discover -s tests -v
```
