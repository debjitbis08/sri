# Conflict search diagnostics v1

**Status: diagnostic study on frozen checkpoints, no training.** Requested by the user on 2026-11-06 after the review
of [conflict search v1 results](conflict-search-v1-results.md). Its job is to find out what actually needs learning
before any new contract. Runner `experiments/conflict_search_diag.py`; reports
`reports/conflict-search-diagnostics-v1{,-runtime}.json`. Inputs: the four phase-1 readers and the twelve phase-2
search controllers of conflict search v1 (`runs/conflict-search-v1/`), unchanged.

The exhaustive and oracle searches below are measuring tools only, used with the user's consent. They are not part of
the system, which stays purely neural.

## Questions

1. Which found paths are proofs? Three reads, from loosest to strictest.
2. With hops restricted to permitted relations, what does the trained controller find?
3. Which part fails: choosing the relation, ranking the entities, the search budget, or the reader graph itself?
4. How much of logic's score is "not found, so false"? And what do the trivial baselines score?

## Definitions

**Permitted hop relations (33; first written as 32 and corrected 2026-11-06, the runner always used 33):** relations with taught individual facts, minus the nine withheld relations
(`grandparent-of`, `ancestor-of`, `kind-of`, `bigger-than`, `smaller-than`, `named-can`, `named-has`, `named-kind`,
`same-colour-as`), the grammar relations (`good-phrase`, `makes-sense`) and the connectives (`not`, `and`, `or`).

**Proof reads**, on a found path. The path is read as an ordered chain a → … → b of facts:

- *factual:* every hop is a true individual fact in the world, under any relation (this is v1's "true-path rate");
- *permitted:* every hop's relation is permitted;
- *valid proof:* every hop is a true fact, and the sequence of relations and directions matches the asked relation's
  definition. `child-of` reversed counts as `parent-of`. The definitions:

  | Relation | Proof |
  |---|---|
  | grandparent-of | `parent-of>` `parent-of>` |
  | ancestor-of | `parent-of>`+ |
  | kind-of | `is-a>` `subkind-of>`* |
  | bigger-than | `just-bigger-than>`+ |
  | smaller-than | `just-bigger-than<`+ |
  | named-can / named-has | `names>` `can>` / `names>` `has>` |
  | named-kind | `names>` `is-a>` `subkind-of>`* |
  | same-colour-as | `colour-of>` `colour-of<` |

  Proof validity is an evaluation read only. No controller sees it.

**World ceiling:** the share of true rows that have a valid proof of at most D_max = 5 hops in the world's true facts.
Size distances above 5 cannot be proved within the walk's depth limit.

**Frozen reader graph:** the confirmed facts (x, ρ, y): logic's s⁺(x ρ y) > 0.5 and the two routes agree
(disagreement ≤ τ), exactly the search's hop check. The candidate pool at a node is logic's s⁺ > 0.5 edges, as in the
search.

## Controls

On every row of the withheld-relation tests (P6-grandparent, P6-colour, L1-kind, L1-anc, ancestor depth 4–5, L1-size,
L2-named) and on the withheld training rows, per seed:

| Control | Relation and direction | Entity order | Budget | Graph |
|---|---|---|---|---|
| `trained` | trained controller, as run | trained Q | B = 24 | reader pool, both-route check |
| `trained-restricted` | trained controller, masked to permitted relations | trained Q | B = 24 | as above |
| `trained-base` | trained controller, masked to the question's base relations (order not given) | trained Q | B = 24 | as above |
| `oracle-random` | oracle (the proof definition's next step) | random among the pool, 3 repeats | B = 24, k = 3 / 2 | as above |
| `oracle-exhaustive` | oracle | all | unlimited, depth ≤ 5 | confirmed graph |
| `bfs-permitted` | any permitted, either direction | all | unlimited, depth ≤ 5 | confirmed graph |
| `bfs-all` | any of the 47 relations (v1's action space) | all | unlimited, depth ≤ 5 | confirmed graph |

Each `trained` control runs for `bt-forward`, `bt-bidir` and `bt-forward-notrace`. The three oracle and BFS controls
use only the phase-1 reader. Reading the differences:

- `oracle-exhaustive` against the world ceiling gives the reader graph's misses;
- `oracle-random` against `oracle-exhaustive` gives the budget and ranking cost when the relation is known;
- `trained-base` and `trained-restricted` against `oracle-random` give the cost of choosing the relation;
- `bfs-*` shows what an unguided search finds, and how often it "proves" false rows.

**Answer reads,** for every control: found → T. Exhausted → F (rule A) or exhausted → U (rule B, the exhaustion → U
control). Reported: T recall, F recall, U rate and balanced accuracy (`tf_bal`, U counted wrong) under each rule;
valid T recall (true rows with a valid proof found); the false-proof rate (false rows with a path found); and the
three proof reads on found paths, for true and false rows separately. **Baselines:** always-F (balanced accuracy
0.50, T recall 0), intuition alone, and v1's logic-final and arbiter as run (T recall, F recall, U rate).

## Prior work and expectations

Exhaustive bounded search over a knowledge graph is the classical baseline that learned walkers (MINERVA, DeepPath;
see [prior work](prior-work.md#backtracking-search-over-middle-things-added-2026-11-06-for-the-conflict-search-v1-draft))
are measured against: with a correct graph and a known rule, it finds every proof within its depth. Path-ranking work
(Lao and Cohen 2010) shows that unguided path enumeration finds many spurious paths between related entities. That
predicts `bfs-*` finds paths on many false rows. Oracle controls that give the program separate learning the program
from executing it (as in Neural LP and NTP analyses). Expectations, not thresholds:

- the world ceiling is 1.0 except for size rows at distances 6–11;
- `oracle-exhaustive` is close to the ceiling wherever the per-hop check is right (1.00 / 0.00 on base facts);
- `bfs-*` finds paths on many false rows;
- under rule B, the trained controllers' balanced accuracy falls well below 0.50 on every test except ancestor.

## Computation (declared)

- **Graph construction per seed:** logic's edge table, n_e² · n_r = 246² · 47 ≈ 2.8 M reader evaluations (batched,
  as in the v1 runner), plus intuition on the pool-true triples (about 10⁴–10⁵). Storage: 2.8 M booleans, plus the
  confirmed adjacency lists.
- **Exhaustive searches:** depth-limited BFS over (node, proof state). Time O(rows × (V · S + E · S)) with V = 246
  nodes, S ≤ 3 proof states and E confirmed edges, shared across rows with the same start. Memory O(V · S).
- **Budgeted searches:** at most B = 24 checks per row; the trained controls run the v1 `search` on all rows.
- **Estimate:** 20–40 CPU min in total, 4 seeds on 4 workers. Measured CPU time, peak RSS, reader-call and check
  counts per control are reported in the runtime JSON. No selection or tuning: every constant is v1's.

## Results (2026-11-06)

All four seeds, 49 s wall. Numbers are means over seeds; for `oracle-random`, also over its 3 repeats. "Valid T recall"
is the share of true rows with a valid proof found. Every test row counts once (v1's reads counted some twice).

### Where proofs are lost

Withheld training rows (257 true, 2,864 false per seed):

| Control | T recall | valid T recall | path found on F rows | valid / found on T rows |
|---|---|---|---|---|
| world ceiling | 1.00 | 1.00 | 0.00 | 1.00 |
| `oracle-exhaustive` (frozen reader graph) | 0.93 | **0.77** | 0.04 | 0.83 |
| `trained-base` forward / notrace / bidir | 0.39 / 0.51 / 0.26 | **0.25 / 0.36 / 0.09** | 0.03 / 0.02 / 0.04 | 0.64 / 0.73 / 0.32 |
| `trained-restricted` forward / notrace / bidir | 0.37 / 0.43 / 0.40 | 0.13 / 0.19 / 0.01 | 0.03 / 0.02 / 0.14 | 0.26 / 0.43 / 0.03 |
| `trained` as run, forward / notrace / bidir | 0.39 / 0.42 / 0.41 | 0.13 / 0.18 / 0.01 | 0.03 / 0.02 / 0.22 | 0.26 / 0.41 / 0.03 |
| `oracle-random` (v1 budget) | 0.15 | 0.13 | 0.00 | 0.89 |
| `bfs-permitted` / `bfs-all` | 1.00 / 1.00 | 0.04 / 0.03 | 0.97 / 0.97 | 0.04 / 0.03 |

Read down the table:

1. **The reader graph loses about a quarter.** With the right program and no budget, valid proofs are found for 0.77
   of true rows. The rest have a hop that logic does not propose or the routes do not confirm. Wrong confirmed facts
   also give false proofs: on test rows, kind 0.28 and colour 0.19 of false rows. The loss depends on the seed:
   ancestor depth 4–5 is provable in the reader graph for 3 / 9 true rows in seed 450, 4 in 451, 7 in 452 and 9 in 453.
   Seed 450's ancestor failure is in the reader, not the controller.
2. **Choosing the relation is the largest learned loss.** Restricting hops to permitted relations changes nothing for
   the forward walkers (0.13 → 0.13, 0.18 → 0.19). The bidirectional walker switches to spurious permitted paths
   (valid 0.01). Masking to the question's base relations doubles valid recall (forward 0.13 → 0.25, notrace 0.18 →
   0.36), and the relation order is still not given.
3. **The budget and ranking lose about half of what remains.** With base relations given, the trained walkers reach
   0.25–0.36 against the exhaustive 0.77. The learned entity ranking does matter: under v1's budget and branching
   (B = 24, k = 3 then 2), a random order with the oracle's relations finds only 0.13 here and 0.00–0.01 on ancestor
   test rows, against 0.28–0.51 for the trained walkers.
4. **Unguided search proves nothing.** Within 5 hops, almost every pair is connected by permitted confirmed facts:
   paths on 0.97 of false rows, with valid proofs on 0.03–0.04 of true rows.

### With the program known, the reader's knowledge answers the tests

Rule A (found → T, exhausted → F), test rows:

| Test | world ceiling (T recall) | `oracle-exhaustive` bal. / F rows with path | best trained (as run) bal. | intuition bal. | v1 arbiter bal. |
|---|---|---|---|---|---|
| P6-grandparent (4 T, 20 F) | 1.00 | 0.99 / 0.03 | 0.50 | 0.97 | 0.74–0.81 |
| P6-colour (20 T, 204 F) | 1.00 | 0.88 / 0.19 | 0.57 | 0.57 | 0.34–0.50 |
| L1-kind (70 T, 24 F) | 1.00 | 0.73 / 0.28 | 0.62 | 0.66 | 0.39–0.55 |
| L1-anc (18 T, 18 F) | 1.00 | 0.91 / 0.00 | 0.74 | 0.72 | 0.67–0.76 |
| L1-anc depth 4–5 (9 T, 9 F) | 1.00 | 0.89 / 0.00 | 0.67 | 0.54 | 0.53–0.64 |
| L1-size (90 T, 90 F) | **0.53** | 0.72 / 0.00 | 0.53 | 0.96 | 0.56–0.65 |
| L1-size, distance ≤ 5 (48 T, 48 F) | 1.00 | 0.91 / 0.00 | 0.54 | 0.94 | 0.53–0.65 |
| L2-named (52 T, 104 F) | 1.00 | 0.94 / 0.01 | 0.62 | 0.88 | 0.50–0.59 |

- **Size distances 6–11 cannot be proved within D_max = 5.** That is a design ceiling of v1, not a learning failure.
- **The trained walkers' valid T recall on test rows is near zero except on ancestor.** Grandparent 0.00 (bidir 0.00,
  notrace 0.00), colour 0.00, size 0.00–0.04, named 0.00, kind 0.01–0.15. On ancestor it is 0.38 / 0.44 (forward /
  notrace), and depth 4–5 is 0.28 / 0.33. Per seed, depth 4–5 valid recall is 0 / 0.67 / 0 / 0.44 (forward) and
  0 / 0.56 / 0 / 0.78 (notrace).
- **Exhaustion → U.** Under rule B, the trained walkers' balanced accuracy is at most 0.34 on every test (kind, bidir).
  Most is from F rows answered by the give-up vote. v1's logic-final as run has T recall 0.00 on grandparent, 0.03–0.36
  on colour and 0.14–0.37 on size.
- **Intuition alone beats the v1 arbiter on every withheld test except ancestor** (notrace arbiter 0.76 against 0.72).
  On size and named, the arbiter keeps T recall (0.81–0.95) but answers U or T on false rows (F recall 0.19–0.43).

### What needs learning

In order of size, on the withheld training rows:

1. **The relation program.** Choosing relation and direction costs about half of the valid proofs that the given base
   relations reach. Restricting the action space does not substitute for it.
2. **Entity ranking under the budget.** It costs about half of what remains (0.25–0.36 against 0.77). The ranking is
   learned and beats random by far, so this is a matter of degree: a larger budget, or better ranking.
3. **The reader graph.** It misses or wrongly confirms about a quarter of the proofs, unevenly across seeds. It caps
   anything built on it, and it should be measured per seed before any controller is judged.
4. **The arbiter.** It should keep intuition where logic has no proof. v1's arbiter does not.

Two things the next contract has to fix: D_max must cover the test distances, or the size test must be limited to
distances ≤ D_max; and it must report proof validity and T recall, not balanced accuracy over a give-up vote.

### Computation (measured)

`reports/conflict-search-diagnostics-v1-runtime.json`; CPU, float32, 4 workers, 1 thread each. Per seed:

| Stage | CPU s | Work |
|---|---|---|
| reader graph construction | 9.1 | 2.84 M logic evaluations (edge table), 0.46 M route evaluations on 229 k pool triples; 58 k confirmed facts |
| `oracle-exhaustive` | 0.3 | 202 BFS over (node, proof state), shared across rows by start and relation |
| `bfs-permitted` / `bfs-all` | 0.5 / 0.9 | 116 BFS, shared by start |
| `oracle-random`, 3 repeats | 0.6 | 47.6 k checks (2 reader calls each) |
| each trained control | 1.6–3.3 | 3,831 rows; 18–35 k checks |
| v1 answers as run, all arms | rest | |

Total 178 CPU s (3.0 CPU min) for 4 seeds, 49 s wall; peak RSS 0.66 GB per worker. This was well below the declared
20–40 CPU min, because the exhaustive searches share BFS by start node. No selection or tuning.

## Correction (2026-11-06, from the composition critic check)

"The failure is not in the per-hop check" holds only for labelled facts. 97% of the confirmed graph under permitted
relations (40,382 of 41,687 facts, seed 450) are relation–entity combinations never taught, such as
`cow parent-of moo`, and both routes confirm them. Unguided BFS connecting 0.97 of false pairs is largely due to these
edges. See [composition critic check v1](composition-critic-check-v1.md).
