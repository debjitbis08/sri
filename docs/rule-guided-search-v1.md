# Rule-guided search v1: told rules steer the walker and the composition critic

**Contract, frozen 2026-11-07 before the run** (user: "yes build"), after one external review. User direction, 2026-11-07: rule
application v1 replaced the component structure we were developing with a new architecture, and that should not have
happened. The architecture is built part by part: take smaller established components from the literature and
integrate them with ours; build only what is missing. This study extends the relational components we have, on the
dense testbed (a standing direction: integration comes later). The rules are the agreed base: `p → q` with variables,
modus ponens, each relation rule an implication the system is told. Train on short problems; test on longer ones, on
different relations and on relations taught earlier.

The review (pasted by the user, 2026-11-07) raised seven points, all confirmed and addressed here:

1. The readers were trained on `ancestor-of`, `grandparent-of`, `bigger-than`, `smaller-than` and `is-a`, so the
   graph contained the answer relations. There is now a hop vocabulary for each task.
2. A conjunction rule cannot be followed as a path. The pilot is restricted to a path fragment, with conjunction as a
   diagnostic.
3. Two derivation rounds can need 4 hops. Proof length is now counted in hops, and training questions are chosen by it.
4. Multiplying supports rejects long valid proofs (0.9¹¹ = 0.31). Acceptance is now local and separate from ranking.
5. The beam can lose the correct branch. There is an exhaustive reference, supplied-proof evaluation, and a failure
   taxonomy.
6. The graph used for training and for evaluation was unspecified. Both are now fixed, and the closure is recomputed on
   the new confirmed graph.
7. Quotas and costs are now complete.

Planned runner `experiments/rule_guided_search.py`; reports `reports/rule-guided-search-v1{,-summary,-runtime}.json`.

## Components

| Component | Source | Role here | Status |
|---|---|---|---|
| logic and intuition readers, applicability head | ours: dual-route truth, type meanings, [amortized introduction v1](amortized-introduction-v1.md) | check every hop; mark meaningless hops | **reused frozen**: the `virtual` checkpoints of seeds 450–453 |
| hop check `ask(x, ρ, y)` | ours: [conflict search v1](conflict-search-v1.md) | a hop is confirmed when both routes agree and logic supports it | **extended:** also requires applicability > 0.5 |
| committed walk, inhibition of return, budget | ours: conflict search v1 | the search discipline | **reused** |
| pairwise composition critic over steps, multiple-instance training on question labels | ours: [composition critic check v1](composition-critic-check-v1.md) | decides which step may follow which | **extended:** context is the told rules instead of the question's relation code |
| rule encoder (atoms of relation, variable and role, summed per rule) | ours: [rule application v1](rule-application-v1.md) | turns told formulas into the critic's context | **reused**, with the readers' relation codes as symbols |
| beam search over a knowledge graph with a learned step scorer | literature: MINERVA's test-time beam (Das et al. 2018) | replaces the learned entity proposer, conflict search's main loss | **adopted** |
| rule execution as following relation sequences | literature: TensorLog (Cohen 2016), Neural LP (Yang et al. 2017) | what a valid proof of a chain rule is | **adopted** as the definition; the symbolic execution is the oracle only |
| type constraints on arguments | literature: type-constrained KG embeddings (Krompaß et al. 2015) | cuts hallucinated edges | **adopted** through our applicability head |

The only new parts are the critic's conditioning on rules and the joining of the pieces. The literature supports the
pieces, not the combination: MINERVA learns navigation and TensorLog compiles inference, and neither establishes that a
critic seeing only the last step implements rule application. That is what this study tests.

## Supported fragment

A proof the walk can represent is a sequence of hops a → … → b, each a base relation in either direction. The pilot
uses rules whose every proof has that form:

- `sub`, `inv`;
- `chain1`–`chain4` (two atoms sharing a middle variable, any orientation);
- recursive `lin-l`, `lin-r`, `trans`, `sym`.

Excluded from the main pilot:

- **`conj`** (A(x, y) ∧ B(x, y) → H(x, y)): checks two facts about the same endpoints. It is a diagnostic with 50 rule
  sets per seed, expected to fail.
- **Questions with a = b.**
- **Proofs that must revisit an entity.** The walk forbids returns. The exhaustive reference, which allows returns,
  counts the T questions this costs.
- **Counting rules** such as `grandparent-of` = `parent-of` ∘ `parent-of` are in the fragment, but a pairwise check
  cannot count. They are tested and expected to over-accept.

## Hop vocabulary per task (what the walk may step on)

Base relations are reader-trained relations minus an excluded set. The asked relation, intermediate derived relations
and relations defined from them are never hops:

| Task | Excluded from hops |
|---|---|
| family (`ancestor-of`, `grandparent-of`, `child-of`) | `ancestor-of`, `grandparent-of`, `child-of` |
| size (`bigger-than`, `smaller-than`) | `bigger-than`, `smaller-than` |
| is-a | `is-a`; `kind-of` is replaced by its immediate restriction (below) |
| training and synthetic rule sets | all of the above, plus the ρ tokens (never trained) |

In every task, the derived relations of the rules are fresh symbols, never graph relations.

**Immediate `kind-of`:** in the confirmed graph, an edge kind-of(x, d) is dropped when another confirmed kind-of(x, c)
reaches d through the confirmed `subkind-of` closure. The oracle graph uses the world's immediate memberships
(dog → mammal), as in rule application v1.

## Graphs, labels and proof length

- **Confirmed graph** (the system's knowledge): for each base relation, direction and entity pair, `ask` is computed once
  per seed. Both routes agree, logic support > 0.5, applicability > 0.5.
- **Oracle graph:** the world's true base facts.
- **Labels** are always the world truth: the oracle graph closed under the rules (rule application v1's generator,
  data only).
- **Proof length:** the minimum number of hops of a valid proof, computed exactly by a min-plus closure in which a base
  fact costs 1 hop, a rule application sums its body's hops, and the minimum is kept over derivations. It is computed
  on the oracle graph (for reads) and on the confirmed graph (for coverage). Derivation depth (rounds) is reported
  beside it. The taught chains have equal depth and hop count except is-a (one `kind-of` hop plus `subkind-of` hops).
- **Confirmed closure:** the confirmed graph closed under the rules, recomputed for this study's gated graph. It is the
  knowledge ceiling: what perfect application would give with the system's facts.
- **Validity of a program:** a walk is a valid proof iff the asked relation holds between its ends when the rules are
  closed over only the walk's own edges. This is exact, and it is an evaluation read only.

## Training (critic trained on the confirmed graph)

- **Rule sets:** 300 per seed, each defining 1–2 derived relations. Forms are drawn from the fragment, uniformly over
  the 6 seed forms, plus one recursive form with probability 0.5 per derived relation. Body relations are sampled from
  the training hop vocabulary (real and synthetic).
- **Questions per rule set:** up to 20 T and 20 F, entities of the body relations' categories.
  - **T questions** have oracle proof length ≤ 2 **and** confirmed proof length ≤ 2, so a valid candidate exists in the
    training graph. Coverage before this filter (the share of oracle-T, length ≤ 2 questions with a confirmed proof of
    length ≤ 2) is reported.
  - **F questions** are oracle-F. Those that the confirmed closure makes T, through the reader's false facts, are
    excluded and counted, so the critic is never taught to reject a valid composition.
  - Half of the F questions are connected negatives: the ends are joined by a confirmed path of ≤ 2 hops.
- **Candidates:** every walk of 1–2 hops from a to b in the confirmed graph, over the training hop vocabulary,
  enumerated by boolean products. This exhaustive enumeration disables the beam, top-k and any learned pruning; it is
  a training-time candidate generator, as in the critic check.
- **Loss:** a question's score is the maximum over its candidates of ∏ c_t (at most 3 checks including END), with
  binary cross-entropy against the label. 1,500 Adam steps, lr 3e-3, 256 T and 256 F questions per step.
- **Critic:** c_t = σ(MLP([rule-set context attended by the step; step t−1; step t])), 64 hidden. A step is [relation
  code R_L (frozen), direction, START/END flags]. The rule tokens follow rule application v1: base relations given by
  their R_L codes, derived relations by fresh symbols, the asked relation flagged.

## Search at test

```
beam = [(a, START)]                                  ranking score: Σ log c_t (critic only)
for hop in 1..D_max (= 12):
    for each path (node, last) in beam:
        for each step σ = (ρ, direction), ρ in the task's hop vocabulary:
            c = critic(rules, last, σ);  if c ≤ 0.5: skip
            for m in confirmed(node, σ), m not on the path:
                extend; if m == b: check critic(rules, σ, END)
    beam = best W extensions by ranking score (ties: fixed random order per seed)
answer T iff some path to b has every c_t > 0.5 (including END)
```

- **Acceptance is local:** a proof is accepted when every step and END pass the critic. Hops are already thresholded by
  `ask`. Supports never multiply into acceptance, so length alone cannot reject a proof. The product policy (∏ c_t ×
  supports > 0.5) is reported as a diagnostic, with the ceiling it imposes on valid confirmed proofs by length.
- **Entities:** every confirmed successor. There is no top-k; the beam (W = 16) is the only pruning.
- **Exhaustive reference:** the same search with an unlimited beam (critic-pruned, all successors), up to 12 hops on the
  taught tasks and up to 4 hops on synthetic rule sets. The same reference with returns allowed counts what no-return
  costs.
- **Supplied proof:** for each T question with a valid proof in the graph, the critic's verdict on the shortest one.

## Test sets (per seed)

- **Taught rules,** every labelled pair of the entities involved:
  - `ancestor-of`: `parent-of` hops, lengths 1–5;
  - `bigger-than`: `just-bigger-than`, 1–11;
  - `smaller-than`: through `bigger-than` reversed, 1–11;
  - is-a: immediate `kind-of` plus `subkind-of`, 1–4;
  - `grandparent-of`: 2;
  - `child-of`: 1.
- **Synthetic:** 100 fresh rule sets. Quotas of oracle T questions by proof length 1, 2, 3, 4, 5–6: 100 each (pooled
  over rule sets), and as many F questions, half of them connected negatives. Shortfalls are reported, and buckets
  under 30 questions are marked uninformative.
- **Rule edits** (taught rules, and the first 300 synthetic questions):
  - relevant: swap a body atom's arguments, replace a body relation with another hop relation, swap the head. Taught
    examples: `ancestor-of` from `child-of`, `bigger-than` from `just-bigger-than` reversed;
  - irrelevant: fresh symbols for derived relations, a dead rule (body relation with no facts), an unused extra rule.
- **`conj` diagnostic:** 50 rule sets.

## Arms (seeds 450–453)

| Arm | Hop check | Critic context | Critic trained on |
|---|---|---|---|
| `rules` | both routes + applicability | told rules | training rule sets, confirmed graph |
| `rules-nogate` | both routes (conflict search's check) | told rules | the same, on its own ungated graph |
| `question` | both routes + applicability | the question's relation code, as in the critic check | the taught relations' training-split questions, 1–2 hops (learned from examples, not told) |

Every arm is evaluated on its confirmed graph and on the oracle graph, with the same trained critic. Only the
evaluation graph changes.

## Reads (four seeds; by proof length)

1. **Answer:** balanced accuracy and T recall per proof length; F recall overall and on connected negatives.
2. **Failure taxonomy** for each T question:
   - no valid proof in the graph;
   - a proof exists but the beam lost it (the exhaustive reference finds an accepted one);
   - a found or supplied proof is rejected by the critic.

   The shares by proof length.
3. **Composition:** critic acceptance of supplied valid proofs and of spurious programs, by length. Program AUROC.
4. **Proof validity:** the share of accepted paths on T questions that are valid proofs.
5. **Hop:** the confirmed graph's composition (labelled true, labelled false, unlabelled) with and without the gate.
   Coverage of training T questions. The confirmed closure against the oracle (knowledge ceiling) by length.
6. **Rule edits:** flipped answers correct; answers unchanged under irrelevant edits.
7. **Diagnostics:**
   - the product-policy ceiling;
   - no-return cost;
   - the `conj` sets;
   - counting (`grandparent-of` on 3–4-hop `parent-of` pairs).
8. **Cost:** expansions and critic calls per question by length, beam and exhaustive.
9. **Reference:** intuition's one-pass verdict on the taught relations it was trained on.

## Predictions (all hypotheses)

1. `rules`, oracle graph: `ancestor-of` and `bigger-than` ≥ 0.9 balanced accuracy at every length; `child-of` ≥ 0.9.
   Supplied valid proofs accepted ≥ 0.9 at every length (the critic check's pattern, now with told rules).
2. `grandparent-of` over-accepts: F recall < 0.5 on 3–4-hop `parent-of` pairs.
3. `smaller-than` and is-a ≥ 0.8 on the oracle graph.
4. Confirmed graph: within 0.1 of the confirmed closure for `ancestor-of` and `bigger-than`. The remaining gap falls
   mostly in "no proof in the graph", not "lost" or "rejected".
5. The gate cuts unlabelled confirmed facts ≥ 3× and raises proof validity ≥ 0.2 over `rules-nogate`.
6. Rule edits: flipped answers ≥ 0.8 correct on the taught linear chains; irrelevant edits leave ≥ 0.95 unchanged.
7. `question` is within 0.05 of `rules` on taught relations at lengths 1–2, but follows edits no better than chance
   (flipped ≤ 0.6).
8. Synthetic, oracle graph: lengths 3–6 within 0.1 of lengths 1–2 for linear recursive rules. The beam loses ≤ 0.05
   of what the exhaustive reference finds.
9. `conj` rule sets near chance, as expected.

## Prior work

Searches of 2026-11-07 are recorded in [rule application v1](rule-application-v1.md) and conflict search. Added here,
all **[familiar]**, except the two papers the review linked, which are cited for their scope only:

- MINERVA ([Das et al. 2018](https://arxiv.org/abs/1711.05851)): a learned walker with a test-time beam;
- TensorLog ([Cohen 2016](https://arxiv.org/abs/1605.06523)): compiles logical inference into differentiable
  operations;
- Neural LP;
- type-constrained KG embeddings (Krompaß et al. 2015);
- grammar-constrained decoding.

- **Predicted by theory:** pairwise (bigram) constraints extend to any length and cannot count. The critic check
  confirmed both. Path-following covers exactly chain-like rules, not conjunctions over the same endpoints.
- **Known failures:**
  - hallucinated confirmed edges (gate; read 5);
  - lost branches (beam against exhaustive; read 2);
  - nonsense witnesses (proof validity, connected negatives);
  - shortcut hop relations (excluded per task);
  - multiplicative length penalty (local acceptance; the product policy as diagnostic);
  - counting and conjunction (diagnostics).
- **Excluded alternative:** symbolic execution of the rules is the oracle, not a component (purely neural).
- **What this adds:** rule-conditioned local composition checks over the system's own neural knowledge, and whether
  told rules make answers follow the rule.

## Scale

| Part | Time | Storage |
|---|---|---|
| confirmed graph, built once | O(R · N²) reader checks | R · N² · 2 directions bits + float32 supports: about 26 MB here |
| candidate enumeration (training) | O(Q · R · N) per question by boolean products | O(candidates) |
| search per question | O(D_max · W · S) critic calls (S = hop relations × 2 directions) + successor lookups | O(W · D_max) paths |
| critic and rule encoder | size independent of N, R and length | |

The rule encoding costs O(atoms) per question and is cached per rule set. At scale, the graph is computed lazily per
hop (O(N) checks per expansion) or retrieved from memory. Counting and conjunction need state beyond the last step;
that is later work.

## Computation (declared, estimates)

- **Confirmed graph:** about 100 relations × 256² × 3 passes ≈ 20 M reader evaluations at about 25 k MAC: 2–5 CPU min
  per seed, per gate setting.
- **Supports and graph storage:** about 26 MB per graph.
- **Candidates:** about 12 k training questions per seed, about 500 programs each: about 6 M short programs (about
  100 MB as int16).
- **Critic:** about 30 k parameters, plus Adam state (2× parameters); 1,500 steps, 2–5 CPU min.
- **Test:**
  - beam: about 4 k questions × 12 hops × 16 paths × about 200 steps scored;
  - exhaustive reference on the taught and short synthetic questions;
  - 15–40 CPU min per run.
- **Totals:** 3 arms × 4 seeds, each with confirmed and oracle evaluation, about 1.5–3 h wall on 6 workers.
- **Tuning:** none.
- **Runtime JSON:** by phase (graph construction, closures, candidates, critic training, each test family, the
  exhaustive reference).
