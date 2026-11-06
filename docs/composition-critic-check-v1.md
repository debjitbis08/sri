# Composition critic check v1

**Status: diagnostic on frozen readers, no walker training.** User direction, 2026-11-06: proof feedback needs three
levels (hop, composition, answer), and composition is the missing one. The consistency critic is the network to check
it, locally, step by step rather than over the whole chain. The question left open is whether a step can depend on
more than the previous one. Runner `experiments/composition_critic_check.py`; reports
`reports/composition-critic-check-v1{,-runtime}.json`. Inputs: the four phase-1 readers of conflict search v1.

## Question

Can a critic learned only from the true/false labels of questions tell valid proof programs from spurious ones? Which
form of memory does it need: the previous step only, a carried summary, or a look-back over earlier steps?

## What the critic sees

A **program** is a found path's sequence of steps, each a relation and a direction, between START and END markers,
together with the question's relation q. Hop truth is the existing two-route check and is not repeated. Every candidate
path runs over facts the frozen reader confirms, using permitted relations only (the 32 of
[the diagnostics](conflict-search-diagnostics-v1.md)). Relation and direction embeddings belong to the critic and are
learned. No proof definition is given to it; those are used only to score it.

## Three variants (same size class, 16-dim embeddings, 32 hidden)

At each step t (the END marker counts as a step), a local check gives c_t = σ(·). A program's score is ∏ c_t, so it is
checked as it grows and can be pruned at the first failing step.

- **`pairwise`:** c_t from [q, step t−1, step t].
- **`summary`:** c_t from [q, h_t, step t], where h_t = GRU over the earlier steps (starting from START).
- **`lookback`:** `pairwise` plus attention over steps 0…t−2, keyed by learned projections and queried from [q, step t].

## Training (multiple instance, labels of questions only)

- **Rows:** withheld-relation training questions (257 T, about 2,864 F per seed).
- **Candidates per row:** every distinct program of at most 2 steps that connects a to b in the confirmed graph (about
  470 per row). This is exhaustive at that length.
- **Loss:** a question's score is the maximum program score among its candidates, with binary cross-entropy against its
  label. On false questions every candidate is a negative. On true questions only the best has to pass. Each step
  takes 128 T and 128 F rows; 1,500 Adam steps, lr 3e-3. Seed per run.
- **Known gap:** some training questions need 3 steps (named-kind through `subkind-of`), so their valid program is not
  among the candidates and the maximum falls on a spurious one. This is recorded, not fixed.

## Reads

On the test questions of the withheld relations (as in the diagnostics), candidates are:

- every program of at most 2 steps;
- 30 sampled random paths of each length 3, 4 and 5 that end at b;
- the shortest valid proof in the confirmed graph, when there is one (the evaluation answers "if the walker proposed
  it, would the critic accept it?").

1. **Composition (program level):** AUROC of the critic's score, valid programs against spurious ones, by relation and
   by length; acceptance rates (score > 0.5) for each.
2. **Answer (row level):** question score = maximum over candidates. T recall, F recall and balanced accuracy at 0.5,
   and AUROC; with and without the valid proof among the candidates.
3. **Length:** ancestor programs `parent-of>`×k for k = 3–5 (trained on k ≤ 2), and kind-of `is-a>` `subkind-of>`×k.
4. **Counting probe:** scores of `parent-of>`×k, k = 1–5, under q = grandparent (valid only at k = 2) and q = ancestor
  (valid for all k).

## Expectations

- **[derived]** `pairwise` generalizes to ancestor 3–5 and to longer kind chains. `summary` may not, unless its state
  settles. `lookback` behaves like `pairwise`.
- **[derived]** On the counting probe, `pairwise` accepts grandparent for every k ≥ 2, because it cannot count.
  `summary` and `lookback` can reject k = 3, but only if training contains a negative that teaches it. The grandparent
  training rows contain no 3-step `parent-of` path, so all three may accept k = 3. That would be a data limit, not
  evidence about the architecture.
- **[familiar]** Learning acceptance from positive and negative strings is grammatical inference. First-order (bigram)
  constraints are identifiable from short strings and extend to any length; counting is not (Delétang et al. 2023).

## Computation (declared)

Candidate enumeration per row: one 66 × 246 × 66 product for 2-step programs (about 0.1 ms) plus 90 sampled walks;
storage of about 1.5 M short programs per seed. Training: 1,500 steps × about 120 k programs × a 32-unit network per
variant; estimated 1–2 CPU min per variant and seed. Total 12 runs (3 variants × 4 seeds) on 4 workers: about 5–10 min
wall. Measured times go in the runtime JSON. No tuning: the constants above are fixed before the run.

## Results (2026-11-06)

Four seeds, means unless stated. "With valid" means the shortest valid proof in the confirmed graph is among a
question's candidates; "without" uses only enumerated programs of at most 2 steps and sampled 3–5-step paths.

### Ancestor: pairwise checking generalizes in length

| Variant | valid `parent-of`×3 / ×4 / ×5 accepted | spurious 3–5 accepted | program AUROC | question bal. with valid (depth 4–5) | without valid |
|---|---|---|---|---|---|
| `pairwise` | 0.96 / 1.00 / 1.00 | 0.00 | 1.00 | 0.89 (0.88) | 0.55 |
| `summary` | 0.71 / 0.75 / 0.75 | 0.00 | 0.80 | 0.78 (0.76) | 0.56 |
| `lookback` | 0.25 / 0.25 / 0.25 | 0.00 | 0.64 | 0.69 (0.68) | 0.57 |

Trained on chains of at most 2 steps, the pairwise critic accepts every valid chain up to 5 steps in all four seeds
(probe `ancestor-of | parent-of×k`: 0.998–1.000 for k = 1–5) and rejects every spurious long path. `summary` fails
completely in seed 452 (all `parent-of` chains below 10⁻⁴); otherwise its state settles. `lookback` learns the chain in
1 seed of 4. Without the valid proof among the candidates, question accuracy falls to 0.55–0.57: the critic can judge
a proof but does not find one. The proposer is still needed.

### Counting: no variant counts, and the data cannot teach it

Probe `grandparent-of | parent-of×k`, k = 1…5, accept if > 0.5:

| Variant | k = 1 | k = 2 | k = 3–5 |
|---|---|---|---|
| `pairwise` | 0.65 (3 seeds 0.86–0.88, one 0.005) | the same | the same, flat in k |
| `summary` | 0.22 (rejects in 3 seeds) | 0.72 | 0.67–0.72 |
| `lookback` | 0.01 | 0.26 | 0.26 |

As expected, `pairwise` cannot tell one, two or five steps apart. `summary` learns to reject a single step but accepts
3–5. The training data contains no 3-step `parent-of` path for any grandparent question, so nothing teaches that
negative. Counting stays untested here.

### Everywhere else, valid programs are rejected

Pairwise critic, seed 450, trained scores of the valid programs: kind-of `>is-a` 0.00 and `>is-a >subkind-of` 0.00
(72 true memberships, 4 false); named-can `>names >can` 0.00; named-has 0.00; named-kind `>names >is-a` 0.00;
same-colour `>colour-of <colour-of` 0.00; bigger-than `>just-bigger-than` 1.00 but 2 steps 0.00; smaller-than 0.01.
The highest-scoring programs are nonsense: kind-of `>parent-of`, `>opposite`, `>lives-in` (all 1.00); named-can
`>parent-of >opposite`, `>lives-in`; same-colour `>sibling-of`, `>has-baby`. On test questions, the valid programs of
kind, size, named and colour are accepted at 0.00–0.25 at every length (program AUROC for size 0.19–0.37, below
chance). Question-level scores are still high (named 0.86–0.88, grandparent 0.81–0.95), but through these nonsense
programs.

### Why: the confirmed graph is mostly facts the reader was never taught

Of the confirmed facts under permitted relations, seed 450: **40,382 unlabelled, 1,251 true, 54 false** (seed 453:
42,335 / 1,302 / 41). "Unlabelled" means a relation–entity combination never taught either way, for example
`cow parent-of moo`, `legs4 parent-of animal`, `sparrow parent-of fly`, `penguin parent-of bird`. Both routes confirm
them. The per-hop accuracy of v1 and the diagnostics (1.00 / 0.00) was measured on labelled base facts, which are 3% of
the graph.

These edges follow associations (cow–moo, penguin–bird), so they correlate with the answers. Each true question needs
only one accepted program, so the critic picks such edges as its witnesses: they reach b on true questions and seldom
on false ones. A valid program loses as soon as one of its steps is shared with programs that occur on false questions.
For example, the step START → `is-a` also begins `>is-a >X` paths on false kind-of questions. For same-colour the
valid program occurs on more false questions than true ones (101 against 35 memberships), through wrongly confirmed
`colour-of` facts. That last account is a reading of the scores, not a separate test.

Ancestor works because `parent-of` between people is fully labelled (every pair of people is taught true or false), so
there are no hallucinated people-to-people `parent-of` edges to exploit.

### What this means

1. **The hop level is not solved.** The two-route check confirms almost any fact outside the taught combinations. Any
   composition learner trained from question labels will use those edges. The reader needs to answer unknown (or
   false) for relation–entity combinations it was never taught evidence for. That is a reader property, not a search
   property.
2. **Where the hops are sound, step-local composition works and generalizes in length** (ancestor, pairwise).
3. **Max-witness training needs a constraint.** With one witness per true question, the critic is free to choose
   idiosyncratic witnesses. A valid program should explain many true questions of its relation and few false ones
   (support and confidence, as in rule mining). That would also make a sound graph easier to exploit correctly.
4. **Counting remains open,** and needs data with the negative cases.

### Computation (measured)

`reports/composition-critic-check-v1-runtime.json`; 4 workers, 1 thread each. Per seed: graph 9.4 CPU s; candidates
73.7 CPU s (1.08 M training memberships over 31.8 k unique (q, program) pairs; 0.69 M evaluation memberships over
231 k pairs); training 296 / 1,131 / 573 CPU s (`pairwise` / `summary` / `lookback`, 3,457 / 9,313 / 5,073 parameters);
evaluation 2–6 CPU s. Total 8,380 CPU s (2.3 CPU h), 36 min wall. The declared 5–10 min wall was too low by about 4×:
the single-seed timing was taken at 100 steps with one process, and four parallel workers ran about 2× slower per step.
The inspection rerun (pairwise, seed 450, 4 threads) and the graph-label count added about 6 CPU min. No tuning.

### Follow-up: type-level reading of the untaught facts

For each permitted relation, the fixture labels its whole domain × range: every pair of entities that ever appears as
subject and object of the relation is taught true or false. So **every untaught fact is outside the relation's types**
(the subject was never a subject of that relation, or the object never its object). Confirmation rates of the
both-route check, seed 450 (seed 453 alike):

| Facts | Confirmed |
|---|---|
| taught true (train and test) | 0.740 (1,251 / 1,690) |
| taught false | 0.004 (54 / 13,101) |
| untaught, outside the relation's types | 0.020 (40,210 / 1,974,490) |

The reader rejects 98% of type-violating facts, so it is not blind to the combinations. But it confirms them five times
as often as taught false facts. They are 130 times as many as the taught pairs, so the 2% outnumbers the true facts 32
to 1. Leakage is highest for relations whose domain spans many kinds of entity: `lives-with` 0.14, `brighter-than` 0.07,
`lighter-than` 0.055. A search selects exactly these edges, because it looks for any edge that reaches b.

Leakage does not depend on how much the reader knows about an entity. The Spearman correlation between an entity's
confirmed untaught facts and the number of relations it was taught in is −0.007 (seed 450). Entities taught in one
relation average 339 leaked facts; those taught in 9–47 relations average 477. The most leaking entities are `tail`
(2 relations), `animal`, `vertebrate`, then `whale`, `horse` and `cheetah` (17–23 relations). `moo` is taught only as
the object of `says` (1 true, 17 false). More facts of the current kind do not stop the leak: the reader does not use
what it knows about an entity to decide whether a relation can apply to it.
