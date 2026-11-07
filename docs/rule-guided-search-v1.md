# Rule-guided search v1: told rules steer the walker and the composition critic

**Contract draft, 2026-11-07, not frozen.** User direction, 2026-11-07: rule application v1 replaced the component
structure we were developing with a new architecture, and that should not have happened. The architecture is built
part by part: take smaller established components from the literature and integrate them with ours, build only what
is missing. This study extends the relational components we have. It stays on the dense testbed until stable
(a standing direction; integration comes later). The rules are the agreed
base: `p → q` with variables, modus ponens, each relation rule an implication the system is told. Train on 1–2 step
problems; test on longer ones, on different relations and on relations taught earlier. Planned runner
`experiments/rule_guided_search.py`; reports `reports/rule-guided-search-v1{,-summary,-runtime}.json`.

## Components

| Component | Source | Role here | Status |
|---|---|---|---|
| logic and intuition readers, applicability head | ours: dual-route truth, type meanings, [amortized introduction v1](amortized-introduction-v1.md) | check every hop; mark meaningless hops | **reused frozen**: the `virtual` checkpoints of seeds 450–453, the same readers whose knowledge damage rule application v1 measured |
| hop check `ask(x, ρ, y)` | ours: [conflict search v1](conflict-search-v1.md) | a hop is confirmed when both routes agree and logic supports it | **extended:** also requires applicability > 0.5 (type constraint) |
| committed walk, one middle at a time, inhibition of return, budget | ours: conflict search v1 | the search discipline | **reused** |
| pairwise composition critic c_t = f(context, step t−1, step t), program score ∏ c_t, multiple-instance training on question labels | ours: [composition critic check v1](composition-critic-check-v1.md), where it generalized ancestor from ≤ 2 to 3–5 steps | decides which step may follow which | **extended:** context is the told rules instead of the question's relation code |
| rule encoder (atoms of relation, variable and role, summed per rule) | ours: [rule application v1](rule-application-v1.md) | turns told formulas into the critic's context | **reused**, with the readers' relation codes as symbols |
| beam search over a knowledge graph with a learned step scorer | literature: MINERVA's test-time beam (Das et al. 2018); path following for rules in TensorLog / Neural LP (Cohen 2016; Yang et al. 2017) | replaces the learned entity proposer, which was conflict search's main loss (recall 0.13–0.36 against 0.77 for the right program run exhaustively) | **adopted** |
| type constraints on relation arguments | literature: type-constrained KG embeddings (Krompaß et al. 2015) | cuts the hallucinated edges (`cow parent-of moo`) that made 97% of the confirmed graph untaught | **adopted**, through our own applicability head |

Nothing here is new except the conditioning of the critic on rules and the joining of the pieces.

## Why this should iterate where rule application v1 did not

The critic only judges whether step t may follow step t−1 under the told rules. A chain of length k is the same check
k times. That is how the composition critic generalized ancestor from 2 to 5 steps. Depth is not something the network
must learn to repeat: the walk repeats, and each step is checked locally (bigram constraints, which carry to any
length; [prior work](prior-work.md), grammatical inference). Rule application v1 had no such structure and learned a
fixed depth.

## World and questions

- **World:** amortized introduction v1's world (`relations_as_objects.World`): the everyday world with category facts,
  62 synthetic relations and the same splits.
- **Base relations** (hops) are relations the readers were trained on. The search walks only base relations, in either
  direction.
- **Derived relations** are defined only by told rules. Their truth is the closure of the world's true base facts under
  the rules (the generator of rule application v1, as data and evaluation oracle only).
- **Training rules:** 300 rule sets per seed, each defining 1–2 derived relations with fresh symbols. Rules are drawn
  from rule application v1's forms, with body relations sampled from the trained base relations, real and synthetic.
  Five rules are excluded: the exact taught rules of the test (`ancestor-of`, `bigger-than`, `smaller-than`, is-a,
  `grandparent-of`, `child-of`).
- **Training questions:** for each rule set, up to 20 T pairs whose shortest derivation is ≤ 2 steps, and the same
  number of F pairs. Half of the F pairs are connected negatives: the entities are joined by a path of confirmed facts,
  but not through the rules.
- **Test questions:**
  - **taught rules:** `ancestor-of` (via `parent-of`, depths 1–5), `bigger-than` (via `just-bigger-than`, 1–11),
    `smaller-than` (through `bigger-than` reversed, 1–11), is-a (immediate `kind-of` plus `subkind-of`, 1–4),
    `grandparent-of` (2), `child-of` (1). Every labelled pair of the entities involved;
  - **synthetic:** 100 fresh rule sets per seed, depths 1–6 where the closure has them;
  - **rule edits:** as in rule application v1. Swap a body atom's arguments, replace a body relation, swap the head;
    irrelevant edits are fresh symbols, a dead rule and an unused relation.

## Search (frozen readers, learned critic)

```
search(a, b, rules):                         beam of committed paths, each from a
    beam = [(a, START, score 1)]
    for depth in 1..D_max:
        for each path (node, last, s) in beam:
            for each step σ = (relation ρ, direction):         ρ over base relations
                c = critic(rules, last, σ)
                if c < 0.5: skip
                for m in confirmed(node, σ), not yet on the path, top k by logic support:
                    extend: (m, σ, s · c · support)
        if an extension reaches b: score it with critic(rules, σ, END); keep the best found
        beam = best W extensions
    answer T iff best found score > 0.5
```

- **Confirmed graph:** for each base relation, direction and entity pair, `ask` is computed once per seed. Both routes
  agree, logic support > 0.5, applicability > 0.5. The graph is fixed during search: the readers never change.
- **Constants:** W = 8 paths, k = 4 entities per step, D_max = 12. Budget = expansions scored, reported.
- **No learned entity proposer.** Entities come from the confirmed graph, ranked by the readers' support. The only
  learned choice is which step may follow, and that is the critic's.

## Critic training

- **Candidates:** for each training question, every program of at most 2 steps connecting a to b in the confirmed
  graph. This is the beam search with an unlimited beam at depth 2. It is a training-time candidate generator, as in
  the critic check.
- **Loss:** question score = maximum program score; binary cross-entropy against the label (multiple-instance).
- **Training:** 1,500 Adam steps, lr 3e-3, 256 T and 256 F questions per step.
- **Critic:** c_t = σ(MLP([rule-set encoding attended by the step's relation code; step t−1; step t])), 64 hidden. A
  step is [relation code R_L (frozen), direction, START/END flags]. The rule-set encoding: rule tokens as in rule
  application v1, with base relations given by their R_L codes, derived relations by fresh symbols, and the asked
  relation flagged.

## Arms (seeds 450–453, matching the reader checkpoints)

| Arm | Hop check | Critic context |
|---|---|---|
| `rules` | both routes + applicability | told rules |
| `rules-nogate` | both routes only (conflict search's check) | told rules |
| `question` | both routes + applicability | the question's relation code only, as in the critic check; trained on the taught relations' training questions instead of rules (learned from examples, not told) |

Each arm is also run with the **oracle graph** in place of the confirmed graph. These are the world's true base facts,
which separates applying the rules from knowing the facts, as `oracle` and `reader-closure` did in rule application v1.

## Reads (four seeds; by depth wherever depth exists)

1. **Answer:** balanced accuracy per depth; T recall per depth; F recall overall and on connected negatives.
2. **Composition:** critic acceptance of the valid program and of spurious programs, by length. Program AUROC.
3. **Proof validity:** the share of found best paths on T questions that are valid derivations under the rules on true
   facts. This is the oracle check, an evaluation read only.
4. **Hop:** the confirmed graph's composition (labelled true, labelled false, unlabelled) with and without the
   applicability gate. Conflict search found 40 k unlabelled against 1.3 k true.
5. **Rule edits:** flipped answers correct, and answers unchanged under irrelevant edits.
6. **Cost:** expansions per question by depth.
7. **Reference:** intuition's one-pass verdict on the taught relations it was trained on.

## Predictions

1. **[derived from the critic check]** `rules`, oracle graph: `ancestor-of` ≥ 0.9 at depths 1–5, `bigger-than` ≥ 0.9
   at 1–11, `child-of` ≥ 0.9. All of these are linear chains a pairwise check covers.
2. **[derived]** `grandparent-of` over-accepts. A pairwise check cannot count, so great-grandparents pass (the critic
   check's counting result). F recall on 3–4-step `parent-of` pairs < 0.5. This is a known limit, not a target.
3. **[hypothesis]** `smaller-than` and is-a ≥ 0.8 on the oracle graph. They need a reversed direction (`smaller-than`)
   or a change of relation after the first step (is-a), both still bigram patterns.
4. **[hypothesis]** On the confirmed graph, answers fall to about the knowledge damage of rule application v1:
   `ancestor-of` and `bigger-than` within 0.1 of their `reader-closure`, is-a poor.
5. **[hypothesis]** The applicability gate cuts unlabelled confirmed facts ≥ 3× (type meanings v1: 3.6×) and raises
   proof validity ≥ 0.2 over `rules-nogate`.
6. **[hypothesis]** Rule edits: flipped answers ≥ 0.8 correct on linear-chain rules; ≥ 0.95 unchanged under
   irrelevant edits.
7. **[hypothesis]** `question` matches `rules` on taught relations it was trained on, but cannot follow edits (flipped
   ≈ unchanged). Told rules are what make the answer follow the rule.
8. **[hypothesis]** Synthetic fresh rule sets, oracle graph: depth 3–6 within 0.1 of depth 1–2 for linear-chain rules.

## Prior work

Searches of 2026-11-07 recorded in [rule application v1](rule-application-v1.md) and the conflict-search contract.
Added here, all **[familiar]**: MINERVA's test-time beam (Das et al. 2018); TensorLog (Cohen 2016) and Neural LP (Yang
et al. 2017), which execute chain rules by following relation sequences over the knowledge base; type-constrained KG
embeddings (Krompaß et al. 2015); grammar-constrained decoding, where a transition model limits which token may follow
which.

- **Correspondence:** executing a told chain rule is following its relation sequence. TensorLog does that with matrix
  products over a symbolic KB. Here, hops are neural reader checks, and which relation may follow which is a learned,
  rule-conditioned local check.
- **Predicted by theory:** bigram (pairwise) constraints extend to any length and cannot count. The critic check
  confirmed both (ancestor 3–5 accepted; no counting).
- **Known failures:**
  - hallucinated confirmed edges, which the gate addresses and read 4 measures;
  - a proposer missing true middles, avoided by taking entities from the confirmed graph;
  - critics that win on true questions through nonsense witnesses (the critic check outside ancestor), measured by
    proof validity (read 3) and connected negatives;
  - counting rules (prediction 2).
- **Excluded alternative:** symbolic execution of the rules, TensorLog-style exact matrix chaining over true facts. It
  is the oracle, not a component (purely neural).
- **What this adds:** rule-conditioned composition checking over the system's own neural knowledge, and whether told
  rules (not examples) make answers follow the rule.

## Scale

- **Confirmed graph:** O(R · N²) reader checks once (R base relations, N entities): 100 × 256² ≈ 6.5 M here. At scale it
  is computed lazily per hop: O(N) checks per expansion, or retrieval by attention from memory.
- **Search per question:** O(D_max · W · S · k), with S the steps scored per expansion (base relations × 2 directions).
  Grows linearly with depth.
- **Critic:** size independent of N, R and depth.
- **Counting rules** need a state beyond the last step. That is later work.

## Computation (declared, estimates)

- **Confirmed graph:** about 6.5 M × 3 reader passes at about 25 k MAC each, about 0.5 TMAC. 2–5 CPU min per seed.
- **Candidates:** 2-step programs for about 12 k training questions, by boolean products. A few CPU min.
- **Critic training:** 1,500 steps. 2–5 CPU min.
- **Test:** about 5 k questions × up to 12 × 8 × 200 critic calls. 10–30 CPU min per run.
- **Totals:** 3 arms × 2 graphs × 4 seeds; the confirmed graph is shared per seed. About 1–2 h wall on 6 workers.
- **Tuning:** none.
- **Runtime JSON:** by phase.
