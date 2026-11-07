# Rule application v1: learning modus ponens with variables on short chains, testing on long ones

**Contract draft, 2026-11-07, not frozen.** The user's direction (2026-11-07): teach first-order rules as meta rules
instead of inventing relation tokens. The base is `p → q` and modus ponens with variables, and every relation rule is
an ordinary implication the system is told. Train on 1–2 step applications; test on problems needing more than 2
steps, with different relations and with relations taught earlier ("remember our original problem"). The design is
mine, not externally reviewed. Planned runner `experiments/rule_application.py`; reports
`reports/rule-application-v1{,-summary,-runtime}.json`.

## Why, and what carries over

[Amortized introduction v1](amortized-introduction-v1.md) showed that practising the use of what is said works when
the use is a pass-through: copying a described partner generalized. Reversing did not: an inverse needed a
transform of a relation code, and the encoder learned it only for practised partners. At the level of facts, the
same rule is a variable binding: to answer C(a, b), check P(b, a). This study applies rules to facts and does not
compress them into a relation code.

Carried over from that design:

- training on use, scored on the consequences, and never on reproducing stored codes;
- reading told statements by attention;
- fresh, never-practised items at test as the memorization check;
- applicability and category statements are left out here: that part worked and is not under test.

## Question

Told a set of facts and rules written as formulas with variables, does a network trained only on problems that need
1–2 rule applications answer questions that need 3–11? Does that hold for relation symbols it has never seen, for
larger problems, and for relations taught earlier (`ancestor-of`, `bigger-than`, is-a)? Does it hold when the base
facts come from the system's own trained reader rather than being given?

## Problems

A problem is a set of entities, base facts, rules and questions. Everything in it is given as context: a working set,
which in the target architecture would be retrieved from memory.

- **Entities:** each gets a random unit identity vector (32), fresh in every problem. Nothing about an entity can be
  memorized.
- **Relation symbols:** each relation in a problem gets a random unit symbol vector (16), fresh in every problem.
  "Different relations" at test is therefore automatic: every problem's relations are new symbols, and only the rules
  say what they mean.
- **Rules** are implications with variables x, y, z. Each has one head atom and one or two body atoms, and each atom
  is (relation symbol, variable, variable). The forms cover all argument orders:
  - one body atom: B(x, y) → H(x, y) (sub-relation); B(y, x) → H(x, y) (inverse); H(y, x) → H(x, y) (symmetry);
  - two body atoms sharing a variable: A(·, ·) ∧ B(·, ·) → H(x, z) in each orientation of the two atoms, e.g.
    A(x, y) ∧ B(y, z) → H(x, z) (composition) and A(y, x) ∧ B(y, z) → H(x, z);
  - recursive: H(x, y) ∧ H(y, z) → H(x, z) (transitivity); A(x, y) ∧ H(y, z) → H(x, z) (linear recursion);
  - conjunction on one pair: A(x, y) ∧ B(x, y) → H(x, y).

  One orientation, A(y, x) ∧ B(z, y) → H(x, z), is held out of training and tested.
- **Labels** come from forward chaining over the given facts and rules, under the closed-world assumption: an atom is T
  if it can be derived and F otherwise. Its **depth** is the forward-chaining round in which it first appears, with all
  rules applied in parallel each round. Base facts have depth 0. The generator is the data source and the evaluation
  oracle only, like v1's ρ truth function. It is never part of the system.
- **Training problems:** 6–10 entities, 2–3 base relations, 1–3 derived relations with 1–2 rules each, base facts at
  density 0.15, plus planted chains so that depth-2 atoms occur. Problems with any atom deeper than 2 are rejected.
  They are generated fresh every step.
- **Synthetic test problems:** 10–16 entities, the same rule forms, with planted chains that give depths 3–6. A fixed
  set of 2,000 per seed, drawn once.

## Taught-relation test problems (the original problem)

These problems use the everyday world's entities and its taught relations. The rules are the true definitions, told
as formulas. Each base relation also has a symbol, as in training:

| Derived relation | Base facts | Rules told | Depths |
|---|---|---|---|
| `ancestor-of` | `parent-of` (39 facts, 38 people) | parent(x, y) → anc(x, y); parent(x, y) ∧ anc(y, z) → anc(x, z) | 1–5 |
| `bigger-than` | `just-bigger-than` (an 11-step chain) | jb(x, y) → big(x, y); jb(x, y) ∧ big(y, z) → big(x, z) | 1–11 |
| `smaller-than` | as above | the two above, plus big(y, x) → small(x, y) | 2–12 |
| is-a | `kind-of`, `subkind-of` (45 entities) | kind(x, c) → isa(x, c); isa(x, c) ∧ sub(c, d) → isa(x, d) | 1–4 |
| `grandparent-of`, `child-of` | `parent-of` | parent ∘ parent; inverse of parent | 1–2 |

Each derived relation is scored against the oracle's closure and against the world's own labels for the relation
where they exist (`ancestor-of`, `bigger-than`, `smaller-than`, is-a, `grandparent-of`, `child-of`). Disagreements
between the two are reported. These problems are also larger than anything trained: 12–50 entities.

Two sources of base facts:

- **`oracle`:** the world's true base facts are given;
- **`reader`:** the base facts are the T answers of the trained logic reader, the stored-code path of amortized
  introduction v1's `virtual` checkpoint of the same seed index, over every entity pair of the problem. This is the
  combined case: taught knowledge plus told rules. Its base-fact errors are reported as the hop level.

## Model: a pair-state network looped with shared weights

Every ordered entity pair (x, y), including x = y, holds a state vector h(x, y) of 64. A problem with n entities has n²
states.

- **Initial state:** an MLP of [Σ of the symbols of base facts on (x, y); Σ of the symbols of base facts on (y, x);
  id(x); id(y); x = y].
- **Rule tokens:** each atom is encoded by an MLP of [symbol; variable embedding of its first argument; of its second
  (x, y, z are learned 8-dim embeddings); role (head, body 1, body 2)]. The atoms of a rule are summed and passed
  through an MLP, giving one token per rule.
- **Step (shared across loops):**
  1. **triangular attention** (Edge Transformer): pair (x, z) attends over every middle entity y. The query comes
     from h(x, z), keys from h(x, y) and h(y, z), and values from the product of projections of h(x, y) and h(y, z).
     4 heads, residual, layer norm;
  2. **reverse mixing:** an MLP of [h(x, z); h(z, x)], residual;
  3. **rule attention:** pair (x, z) attends over the rule tokens, residual, layer norm;
  4. an MLP with residual and layer norm.
- **Readout:** atom r(x, y) is T with probability σ(sym_rᵀ U h_T(x, y) + b).

Nothing in the model encodes modus ponens, which pairs a rule's variables bind to, or what a head does. The triangle
supplies only which pairs can be combined (every middle entity is a candidate, as attention over candidates). Variable
binding, matching symbols to facts, conjunction and writing the head are all learned from the consequences.

## Arms (seeds 460–463)

| Arm | Step | Depth at test |
|---|---|---|
| `edge-loop` | the step above, shared weights, looped | T loops, any T |
| `edge-fixed` | 4 unshared copies of the step, run once | fixed at 4 |
| `pair-loop` | as `edge-loop`, but triangular attention replaced by ordinary self-attention over all n² pair states | T loops |

`edge-fixed` tests whether looping matters. `pair-loop` tests whether the triangle's candidate structure matters, or
generic attention suffices.

## Training

- 6,000 steps, 32 fresh problems per step, Adam lr 1e-3, no weight decay.
- **Loops:** T is drawn from {2, 3, 4} per step, so the step must be stable past the needed depth. `edge-fixed` always
  runs its 4 layers.
- **Loss:** binary cross-entropy on every atom of every relation in the problem, base and derived. T and F atoms are
  weighted equally within each problem.
- **Supervision is on consequences only:** no proofs, no intermediate steps.

## Test

`edge-loop` and `pair-loop` run with T = 16 loops, fixed whatever the depth, since a real system does not know the
depth in advance. They are also read at T = 4 and T = 8 (taught relations up to T = 16 only). `edge-fixed` runs its 4
layers.

## Reads (balanced accuracy unless stated; means over seeds)

1. **By depth:** T recall of atoms of each depth (0–6 synthetic, 1–12 taught) and F recall, combined into balanced
   accuracy per depth bucket.
2. **Composition (connected negatives):** F recall on non-derived atoms whose entities are joined by a path of base
   facts of length ≤ 3, so a path exists but not through the rules. This catches the shortcut "connected means
   true", the composition level of the three-level feedback.
3. **Hop level (`reader`):** precision and recall of the reader's base facts against the world.
4. **Answer level:** per derived taught relation and source, by depth, against the oracle and the world.
5. **Size:** synthetic depth ≤ 2 at 6–10 against 12–16 entities.
6. **Stability:** depth ≤ 2 at T = 4, 8 and 16.
7. **Held-out orientation:** atoms derived through A(y, x) ∧ B(z, y) → H(x, z).
8. **In distribution:** fresh depth ≤ 2 problems like training.

## Predictions

1. **[hypothesis, main]** `edge-loop` at T = 16 on synthetic depths 3–4 ≥ 0.9 and depths 5–6 ≥ 0.8. Connected
   negatives F recall ≥ 0.85.
2. **[derived]** `edge-fixed` ≤ 0.7 at depths 5–6. With one application per layer, 4 layers cannot reach depth 5
   through a linear chain.
3. **[hypothesis]** `pair-loop` ≥ 0.1 below `edge-loop` at depths 3–6.
4. **[hypothesis, the original problem]** `edge-loop`, `oracle` facts: `ancestor-of` ≥ 0.9 at every distance;
   `bigger-than` ≥ 0.9 at distances 1–6 and ≥ 0.8 at 7–11; is-a, `grandparent-of` and `child-of` ≥ 0.9. All of this on
   problems 2–5× larger than any trained.
5. **[derived]** `reader` facts: answers fall below `oracle` by about what the hop errors imply. A wrong base fact
   propagates along every chain through it. This read separates application from knowledge.
6. **[hypothesis]** Stability: depth ≤ 2 at T = 16 within 0.03 of T = 4.
7. **[hypothesis]** Held-out orientation ≥ 0.85 at depth ≤ 2.
8. **[derived]** In distribution ≥ 0.97. Symbols and entities are fresh in every problem, so training accuracy is
   already accuracy on new relations.

## Prior work

Search 2026-11-07. Queries: "Edge Transformer systematic generalization Bergen O'Donnell Bahdanau 2021 triangular
attention CLUTRR"; "RuleTaker Transformers as soft reasoners over language depth generalization"; "Looped
transformers for length generalization Fan 2024"; "Neural Logic Machines Dong 2019 generalize larger problem sizes".

- **Edge Transformer** ([Bergen et al. 2021](https://arxiv.org/abs/2112.00578)) **[abstract]**: vector states on every
  pair of nodes, updated by triangular attention inspired by unification. It generalized more systematically than
  graph attention networks and Universal/relation-aware Transformers on graph CLUTRR (kinship composition) and on
  compositional parsing. Its step is used here. Differences: there, relations are learned from examples with fixed
  vocabularies, the model is not looped, and rules are not told; here, symbols are fresh per problem, rules are given
  as formulas, and the step is looped.
- **RuleTaker** ([Clark et al. 2020](https://arxiv.org/abs/2002.05867)) **[abstract]** and ProofWriter (Tafjord et al.
  2021) **[familiar]**: transformers given facts and rules in English, trained by maximum proof depth, reported at 95%+
  on chains deeper than trained. That is the task here, in natural language and with a pretrained language model.
  Per-depth numbers were not checked; the 95% figure is from the abstract. This study uses formulas, a small network
  without pretraining, fresh symbols, and taught world relations.
- **Looped Transformers** ([Fan et al. 2024](https://arxiv.org/abs/2409.15647)) **[abstract]**: weight-tied loops with
  an adaptive step count generalize to longer inputs than trained, beating fixed-depth models, on tasks with a known
  iterative solution. They predict `edge-loop` over `edge-fixed` **[derived]**. Their loss is applied after the
  required number of steps; here T is random in 2–4, at or above the training depth.
- **Neural Logic Machines** ([Dong et al. 2019](https://arxiv.org/abs/1904.11694)) **[abstract]**: tensors over objects
  and pairs with MLPs and quantifier reductions, trained small and reported to generalize perfectly to larger
  instances on family-tree and graph reasoning. They learn the rules from examples; here the rules are told and
  application is learned. Recurrent relational networks (Palm et al. 2018) **[familiar]** run more steps at test on
  harder problems.
- **∂ILP** (Evans and Grefenstette 2018), **Neural Theorem Provers** (Rocktäschel and Riedel 2017) and **Neural LP**
  **[familiar]**: differentiable rule learning or proving with the inference structure built in (forward chaining or
  backward chaining with soft unification). That is the excluded alternative: modus ponens supplied, not learned.
- **Known failures:**
  - looped models diverging beyond trained step counts (read 6; random T in training);
  - shortcuts from graph statistics, such as connected means true (read 2) and depth from distance alone (the
    wrong-relation paths in read 2);
  - symbol or entity memorization (fresh per problem);
  - one-pass networks failing length generalization (Anil et al. 2022 **[familiar]**; `edge-fixed` checks it);
  - the gap between the reader's facts and the world (read 3; the hop level found in conflict search, where 97% of the
    confirmed graph was untaught facts).
- **Cheaper alternative:** symbolic forward chaining is exact and is used as the label oracle. It is excluded from the
  system (purely neural, user direction).
- **What this adds:** whether modus ponens with variables, learned from consequences of told formula rules on 1–2 step
  problems, carries to long chains and to the system's own taught relations, including with its own imperfect facts.
- **Left for later:** the truth table of → as a judgement (does p → q hold, false antecedent, unknowns). Here rules are
  always told true, and only their use is taught. Judging rules belongs with the two-scalar truth currency
  ([dual-route truth](dual-route-truth-v1-results.md)) and with rules that can fail or have exceptions.

## Scale

A step costs O(n³·d) for the triangle (every pair over every middle entity), O(n²·d²) for the projections and MLPs,
and O(n²·r·d) for rule attention with r rules. A problem costs T times that, so depth costs loops linearly.

At scale, n is the working set (entities relevant to a question, retrieved from memory by attention), not the whole
world. That bound is what keeps n³ affordable, and it is an assumption, not a result of this study. Long chains whose
middle entities are not in the working set would need retrieval inside the loop, which is not tested here.

Storage per problem: n² states × 64 floats, plus rule tokens. The network's size does not grow with n, r or T.

## Computation (declared)

- **Per step (estimate):** about 3.5 M MAC per loop per problem at n = 10. With T ≈ 3 loops and 32 problems, about
  0.34 GMAC forward, about 1 GMAC with backward.
- **Training estimate:** about 0.2–0.5 s per step on one CPU thread, 20–50 min per run. To be checked in the smoke
  stage before the main run.
- **Test:**
  - synthetic: 2,000 problems at T = 16 (n ≤ 16), plus reads at T = 4 and 8;
  - taught: 6 derived relations × 2 sources at n ≤ 50, plus 4 reader passes over at most 2,500 pairs × 4 base
    relations;
  - estimate 2–5 CPU min per run.
- **Totals:** 12 runs (3 arms × 4 seeds) on 6 workers, about 1–2 h wall.
- **Parameters:** about 80–120 k (counted at the smoke stage). One checkpoint per run, not tracked.
- **Tuning:** none. Every constant is fixed here.
- **Resumability:** per run, with `experiments/durable.py`. The runtime JSON follows `computation-reporting.md`
  (generation, training, test by phase). The `reader` condition depends on the untracked checkpoints
  `runs/amortized-introduction-v1/virtual-seed45{0..3}.pt`, whose sha256 are in that study's report.
