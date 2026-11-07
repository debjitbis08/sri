# Rule application v1: learning modus ponens with variables on short chains, testing on long ones

**Contract, frozen 2026-11-07 before the run** (user: "Ya freeze, implement and run"), after one external review. The user's direction (2026-11-07): teach
first-order rules as meta rules instead of inventing relation tokens. The base is `p → q` and modus ponens with
variables, and every relation rule is an ordinary implication the system is told. Train on 1–2 step applications; test
on problems needing more than 2 steps, with different relations and with relations taught earlier ("remember our
original problem"). The work stays on the dense dual-route testbed until it is stable; integration into the target
architecture comes later (user, 2026-11-07). The design is mine. One external review (pasted by the user, 2026-11-07)
found six issues, all confirmed against the world and the sources and addressed here: the is-a base facts, the
fixed-depth bound, separating reader damage from application, rule-change tests, the literature, and the generator
and cost specification. Planned runner `experiments/rule_application.py`; reports
`reports/rule-application-v1{,-summary,-runtime}.json`.

## Why, and what carries over

[Amortized introduction v1](amortized-introduction-v1.md) showed that practising the use of what is said works when
the use is a pass-through: copying a described partner generalized. Reversing did not: an inverse needed a
transform of a relation code, and the encoder learned it only for practised partners. At the level of facts, the
same rule is a variable binding: to answer C(a, b), check P(b, a). This study applies rules to facts and does not
compress them into a relation code.

Carried over from that design:

- training on use, scored on the consequences, never on reproducing stored codes;
- reading told statements by attention;
- fresh, never-practised items at test as the memorization check.

Applicability is left out here: it worked there and is not under test.

## Question

Told a set of facts and rules written as formulas with variables, does a network trained only on problems that need
1–2 rule applications answer questions that need 3–12? Does that hold for relation symbols it has never seen, for
larger problems, and for relations taught earlier (`ancestor-of`, `bigger-than`, is-a)? Does it hold when the base
facts come from the system's own trained reader? Do its answers follow the rules, so that changing a rule changes the
answer?

## Problems

A problem is a set of entities, base facts, rules and questions, all given as context: a working set, which in the
target architecture would be retrieved from memory.

- **Entities:** each gets a random unit identity vector (32), fresh in every problem.
- **Relation symbols:** each relation in a problem gets a random unit symbol vector (16), fresh in every problem.
  Every problem's relations are new symbols, and only the rules say what they mean.
- **Rules** are implications with variables x, y, z. Each has one head atom and one or two body atoms; an atom is
  (relation symbol, variable, variable). The forms:

| Group | Forms |
|---|---|
| seed (non-recursive) | `sub` B(x,y) → H(x,y); `inv` B(y,x) → H(x,y); `chain1` A(x,y) ∧ B(y,z) → H(x,z); `chain2` A(y,x) ∧ B(y,z) → H(x,z); `chain3` A(x,y) ∧ B(z,y) → H(x,z); `conj` A(x,y) ∧ B(x,y) → H(x,y) |
| recursive | `sym` H(y,x) → H(x,y); `trans` H(x,y) ∧ H(y,z) → H(x,z); `lin-l` A(x,y) ∧ H(y,z) → H(x,z); `lin-r` H(x,y) ∧ A(y,z) → H(x,z) |
| held out of training | `chain4` A(y,x) ∧ B(z,y) → H(x,z) |

- **Labels** come from forward chaining over the given facts and rules, under the closed-world assumption: an atom is T
  if derivable and F otherwise. Its **depth** is the forward-chaining round in which it first appears, with all rules
  applied in parallel each round. That is its shortest derivation, so planted chains with shorter alternative proofs
  get their true, shorter depth. Base facts have depth 0. The generator is the data source and evaluation oracle only,
  never part of the system.

### Generator (frozen)

- **Sizes:**
  - training: 6–10 entities (uniform), 2–3 base relations, 1–3 derived relations;
  - synthetic test: 10–16 entities, 2–4 base relations, 1–3 derived relations.
- **Rules per derived relation:**
  - exactly one seed rule, so no derived relation is vacuous. Its form is drawn uniformly from the 5 training seed forms
    (6 at test, with `chain4` at one-sixth);
  - with probability 0.5, one recursive rule, drawn uniformly from the 4 recursive forms;
  - body relations are base relations, or, with probability 0.3 each, an earlier derived relation. That gives depth
    across rules.
- **Base facts:** density 0.15 per relation over ordered pairs of distinct entities. Planted chains: 1–2 per problem, a
  random path of 2–4 entities (training) or 4–8 (test) whose consecutive pairs get the base facts a rule body needs.
- **Acceptance:**
  - every derived relation has at least 1 T atom, and T atoms make up 5–60% of each derived relation's atoms;
  - training: maximum depth ≤ 2, and each batch of 32 holds 16 problems whose maximum depth is exactly 2;
  - synthetic test: 2,000 problems per seed, 500 each with verified maximum depth 3, 4, 5 and 6, drawn once.
  - Each problem gets up to 200 attempts. After that its sizes are redrawn. The rejection count and the number of
    redraws are reported.
- **The held-out form** appears in 500 extra test problems per seed, with maximum depth ≤ 2 and ≤ 4 in equal numbers.

## Taught-relation test problems (the original problem)

These use the everyday world's entities and taught relations, with the true definitions told as formulas and a fresh
symbol for each relation:

| Derived relation | Base facts | Rules told | Depths |
|---|---|---|---|
| `ancestor-of` | `parent-of` (39 facts, 38 people) | parent(x, y) → anc(x, y); parent(x, y) ∧ anc(y, z) → anc(x, z) | 1–5 |
| `bigger-than` | `just-bigger-than` (an 11-step chain, 12 entities) | jb(x, y) → big(x, y); jb(x, y) ∧ big(y, z) → big(x, z) | 1–11 |
| `smaller-than` | as above | the two above, plus big(y, x) → small(x, y) | 2–12 |
| is-a | **immediate** `kind-of` plus `subkind-of` (45 entities) | kind(x, c) → isa(x, c); isa(x, c) ∧ sub(c, d) → isa(x, d) | 1–4 |
| `grandparent-of`, `child-of` | `parent-of` | parent ∘ parent; inverse of parent | 1–2 |

**Immediate memberships only.** The world's `kind-of` lists every supercategory (dog → mammal, vertebrate, animal,
living-thing), which would make every is-a answer depth 1. The base facts therefore keep kind(x, c) only when no other
given kind(x, c′) reaches c through `subkind-of`. That leaves dog → mammal, so is-a(dog, living-thing) needs 4 steps.
`subkind-of` is already immediate links only (mammal → vertebrate → animal → living-thing). The same reduction is
applied to the reader's facts, using the reader's own `subkind-of` answers.

Each derived relation is scored against the oracle closure. Where the world has labels, it is also scored against
those (`ancestor-of`, `bigger-than`, `smaller-than`, `grandparent-of`, `child-of`; is-a is labelled for only some
categories, e.g. dog against mammal, animal and the other classes, not vertebrate or living-thing). Disagreements
between closure and world are reported. These problems have 12–45 entities, larger than any trained.

Three closures per taught problem:

- **`oracle`:** the world's true base facts, closed by the generator;
- **`reader`:** the base facts are the T answers of the trained logic reader, the stored-code path of amortized
  introduction v1's `virtual` checkpoint of the same seed index, over every entity pair of the problem. The network
  is given these facts;
- **`reader-closure`:** the generator's exact closure of exactly the facts the reader supplied.

The comparisons that follow:

- network on `reader` facts against `reader-closure`: did it apply the rules correctly?
- `reader-closure` against `oracle`: what damage did imperfect knowledge cause? Base-fact precision and recall cannot
  say that, since a redundant missing edge may change nothing while one false edge can connect whole components.

## Model: a pair-state network looped with shared weights

Every ordered entity pair (x, y), including x = y, holds a state vector h(x, y) of 64; n entities give n² states.

- **Initial state** h₀: an MLP of [Σ of the symbols of base facts on (x, y); Σ of the symbols of base facts on (y, x);
  id(x); id(y); x = y].
- **Rule tokens:** each atom is encoded by an MLP of [symbol; variable embedding of its first argument; of its second
  (x, y, z are learned 8-dim embeddings); role (head, body 1, body 2)]. A rule's atoms are summed and passed through
  an MLP, giving one token per rule.
- **Step (shared across loops):**
  0. **input injection:** h ← h + W₀h₀, as in looped transformers;
  1. **triangular attention** (Edge Transformer): pair (x, z) attends over every middle entity y. Query from h(x, z);
     keys from h(x, y) and h(y, z); values from the product of projections of h(x, y) and h(y, z). 4 heads, residual,
     layer norm;
  2. **reverse mixing:** an MLP of [h(x, z); h(z, x)], residual;
  3. **rule attention:** pair (x, z) attends over the rule tokens, residual, layer norm;
  4. an MLP with residual and layer norm.
- **Readout:** atom r(x, y) is T with probability σ(sym_rᵀ U h_T(x, y) + b).

Nothing in the model encodes modus ponens, which pairs a rule's variables bind to, or what a head does. The triangle
supplies only which pairs can be combined (every middle entity is a candidate, as attention over candidates). Variable
binding, matching symbols to facts, conjunction and writing the head are learned from consequences. The network is
not limited to one rule application per step. Triangular attention can combine two pair states that each already
summarize a path, so the span it covers can double with each step. Its depth in steps and the generator's depth in
rounds are different quantities.

## Arms (seeds 460–463)

| Arm | Step | At test |
|---|---|---|
| `edge-loop` | the step above, weights tied across steps, looped | any number of steps T |
| `edge-untied` | 4 untied copies of the step, run once | 4 steps |
| `pair-loop` | as `edge-loop`, with triangular attention replaced by ordinary self-attention over all n² pair states | any T |

Separate comparisons:

- **Tying at equal depth:** `edge-loop` at T = 4 against `edge-untied`.
- **Extra iterations:** `edge-loop` at T = 8 and T = 16 against its own T = 4. This is the test that matters for depth
  generalization. The Edge Transformer ties weights by default, but its paper did not run more steps at test than in
  training.
- **Candidate structure:** `pair-loop` against `edge-loop` at the same T.

## Training

- 6,000 steps, 32 fresh problems per step, Adam lr 1e-3, no weight decay.
- **Loops:** T is drawn from {2, 3, 4} per step, at or above the training depth. Unlike Fan et al., no step count is
  supplied per example. `edge-untied` always runs its 4 layers.
- **Loss:** binary cross-entropy on every atom of every relation in the problem, base and derived. T and F atoms are
  weighted equally within each problem.
- **Supervision is on consequences only:** no proofs, no intermediate steps.

## Test

`edge-loop` and `pair-loop` are read at T = 4, 8 and 16. T = 16 is the main read: fixed, since a real system does not
know the depth. `edge-untied` runs its 4 layers.

## Reads (means over seeds)

1. **By depth:** T recall per depth (0–6 synthetic, 1–12 taught). F recall is computed over the non-derived atoms of
   the derived relations in the same problems. Balanced accuracy per depth bucket is the mean of the two.
2. **Connected negatives (composition):** F recall on non-derived atoms of derived relations whose entities are joined
   by a directed path of base facts, ignoring relation and direction, bucketed by shortest path length (1–2, 3–4, 5–6,
   7–12). The buckets match the deep positive cases. This catches "connected means true".
3. **Rule changes (does the answer follow the rule?):** 500 synthetic test problems per seed, plus each taught problem,
   with facts and queries kept fixed:
   - **relevant edits:** swap the argument order of one body atom; replace one body relation with another relation of
     the problem; swap the head's argument order. Taught examples: anc from child-of instead of parent-of; big from
     jb(y, x). Scored on atoms whose oracle label flips (accuracy of the new answer) and on atoms whose label stays
     (unchanged-answer rate);
   - **irrelevant edits that must preserve every answer:**
     - replace all symbols and entity vectors with fresh ones;
     - add a rule whose body relation has no facts and can derive none;
     - add facts of a new base relation no rule mentions.

     Scored as the share of atoms whose answer is unchanged.
4. **Hop and knowledge damage (`reader`):** reader base-fact precision and recall against the world; `reader-closure`
   against `oracle` by depth (damage); network against `reader-closure` (application).
5. **Answer level:** per taught relation and source, by depth, against the oracle and the world.
6. **Size:** synthetic depth ≤ 2 at 6–10 against 12–16 entities.
7. **Stability:** depth ≤ 2 at T = 4, 8 and 16.
8. **Held-out form:** atoms whose shortest derivations use `chain4`.
9. **In distribution:** fresh depth ≤ 2 problems like training.

## Predictions

1. **[hypothesis, main]** `edge-loop` at T = 16 on synthetic depths 3–4 ≥ 0.9 and depths 5–6 ≥ 0.8. Connected negatives
   F recall ≥ 0.85 in every path-length bucket.
2. **[hypothesis]** Extra iterations help: `edge-loop` at T = 16 is ≥ 0.1 above its T = 4 on depths 5–6. Not derived:
   doubling path spans may let 4 steps reach depth 6.
3. **[hypothesis]** Tying: `edge-loop` at T = 4 within 0.05 of `edge-untied` on depths ≤ 4.
4. **[hypothesis]** `pair-loop` ≥ 0.1 below `edge-loop` at depths 3–6.
5. **[hypothesis, the original problem]** `edge-loop`, `oracle` facts:
   - `ancestor-of` ≥ 0.9 at every distance;
   - `bigger-than` ≥ 0.9 at distances 1–6 and ≥ 0.8 at 7–11;
   - is-a, `grandparent-of` and `child-of` ≥ 0.9.

   All on problems up to 5× larger than any trained.
6. **[derived]** `reader`: the network against `reader-closure` matches its `oracle` accuracy within 0.05, since
   application does not care where facts came from. Knowledge damage shows up in `reader-closure` against `oracle`.
7. **[hypothesis]** Rule changes: flipped atoms ≥ 0.85 correct after relevant edits; ≥ 0.98 of answers unchanged after
   irrelevant edits.
8. **[hypothesis]** Stability: depth ≤ 2 at T = 16 within 0.03 of T = 4.
9. **[hypothesis]** Held-out form ≥ 0.85 at depth ≤ 2.
10. **[derived]** In distribution ≥ 0.97. Symbols and entities are fresh in every problem, so this is already accuracy
    on new relations at trained depth.

## Prior work

Search 2026-11-07. Queries: "Edge Transformer systematic generalization Bergen O'Donnell Bahdanau 2021 triangular
attention CLUTRR"; "RuleTaker Transformers as soft reasoners over language depth generalization"; "Looped
transformers for length generalization Fan 2024"; "Neural Logic Machines Dong 2019 generalize larger problem sizes".
After the review, [Edge Transformer §3](https://arxiv.org/html/2112.00578v1) and [Looped Transformers
§4](https://arxiv.org/html/2409.15647v5) were read for the two points below **[read: sections]**.

- **Edge Transformer** ([Bergen et al. 2021](https://arxiv.org/abs/2112.00578)) **[abstract; §3 read]**: vector states
  on every pair of nodes, updated by triangular attention inspired by unification. Weights are tied across layers by
  default, like the Universal Transformer, with 8 layers on CLUTRR. It generalized more systematically than graph
  attention networks and Universal/relation-aware Transformers on graph CLUTRR (kinship composition, test chains up
  to length 10) and on compositional parsing. It did not run more layers at test than in training. Its step is used
  here. Differences: there, relations come from fixed vocabularies learned from examples and rules are not told;
  here, symbols are fresh per problem, rules are given as formulas, and extra iterations are tested.
- **RuleTaker** ([Clark et al. 2020](https://arxiv.org/abs/2002.05867)) **[abstract]** and ProofWriter (Tafjord et al.
  2021) **[familiar]**: transformers given facts and rules in English, trained by maximum proof depth, reported at 95%+
  on chains deeper than trained. That is the task here, in natural language and with a pretrained language model.
  Per-depth numbers were not checked. This study uses formulas, a small network without pretraining, fresh symbols,
  and taught world relations.
- **Looped Transformers** ([Fan et al. 2024](https://arxiv.org/abs/2409.15647)) **[abstract; §4 read]**: weight-tied loops
  generalize to longer inputs than trained and beat fixed-depth models, on tasks with a known iterative solution
  (RASP-L programs). Training supplies each example's required step count, and the original input is injected at
  every loop. They motivate this experiment but do not predict its result. Here no step counts are supplied (T is
  random in 2–4), and input injection is adopted.
- **Neural Logic Machines** ([Dong et al. 2019](https://arxiv.org/abs/1904.11694)) **[abstract]**: tensors over objects
  and pairs with MLPs and quantifier reductions, trained small and reported to generalize perfectly to larger instances
  on family-tree and graph reasoning. They learn the rules from examples; here the rules are told and application is
  learned. Recurrent relational networks (Palm et al. 2018) **[familiar]** run more steps at test on harder problems.
- **∂ILP** (Evans and Grefenstette 2018), **Neural Theorem Provers** (Rocktäschel and Riedel 2017) and **Neural LP**
  **[familiar]**: differentiable rule learning or proving with the inference structure built in. That is the excluded
  alternative: modus ponens supplied, not learned.
- **Known failures:**
  - looped models diverging beyond trained step counts (read 7; random T; input injection);
  - shortcuts from graph structure (connected negatives up to path length 12, read 2) and from template statistics
    that fresh symbols do not remove (rule changes, read 3);
  - symbol or entity memorization (fresh per problem);
  - one-pass networks failing length generalization (Anil et al. 2022 **[familiar]**);
  - reader damage mixed with application errors (`reader-closure`, read 4).
- **Cheaper alternative:** symbolic forward chaining is exact. It is the label oracle and the `reader-closure`
  reference, and it is excluded from the system (purely neural, user direction).
- **What this adds:** whether modus ponens with variables, learned from consequences of told formula rules on 1–2 step
  problems, carries to long chains and to the system's own taught relations, with answers that follow the rules.
- **Left for later:** the truth table of → as a judgement (does p → q hold, false antecedent, unknowns). Here rules are
  always told true and only their use is taught. Judging rules belongs with the two-scalar truth currency
  ([dual-route truth](dual-route-truth-v1-results.md)) and with rules that have exceptions.

## Scale

Per step, for a problem with n entities and r rules, at width d:

| Part | Time | Activation memory |
|---|---|---|
| triangular attention (`edge-*`) | O(n³·d) | O(n³·heads) attention weights |
| self-attention over pairs (`pair-loop`) | O(n⁴·d) | O(n⁴·heads) |
| projections and MLPs | O(n²·d²) | O(n²·d) |
| rule attention | O(n²·r·d) | O(n²·r) |

A problem costs T times this; training stores activations for all T steps, so its memory is T times as well.

At scale, n is the working set (entities relevant to a question, retrieved from memory by attention), not the whole
world. That bound is what keeps n³ affordable, and it is an assumption, not a result of this study. `pair-loop`'s n⁴
makes it a control, not a candidate. Long chains whose middle entities are not in the working set would need retrieval
inside the loop, which is not tested here. The network's size does not grow with n, r or T.

## Computation (declared)

- **Training per step (estimate):** about 3.5 M MAC per step per problem at n = 10 for `edge-*` (the triangle is about
  0.2 M of it). With T ≈ 3 and 32 problems that is about 0.34 GMAC forward, about 1 GMAC with backward. `pair-loop`
  adds about n⁴·d ≈ 0.6 M MAC per step per problem at n = 10.
- **Training estimate:** about 0.2–0.5 s per step on one CPU thread, 20–50 min per run, to be measured in the smoke
  stage. Peak activation memory at n = 10, T = 4, batch 32 is under 100 MB.
- **Test, the largest case:** taught is-a at n = 45, T = 16:
  - `edge-*`: triangle weights 45³ × 4 × 4 B ≈ 1.5 MB per step;
  - `pair-loop`: attention weights 45⁴ × 4 × 4 B ≈ 66 MB per step, run without gradients.
- **Test totals:**
  - synthetic: 2,000 problems + 500 held-out form + 500 × 7 edited versions, each at three T values;
  - taught: 6 derived relations × (`oracle`, `reader`) plus their edits;
  - reader passes: at most 2,025 pairs × 4 base relations per seed;
  - estimate 5–15 CPU min per run, `pair-loop` the most.
- **Totals:** 12 runs (3 arms × 4 seeds) on 6 workers, about 1–2.5 h wall.
- **Parameters:** about 80–120 k (counted at the smoke stage). One checkpoint per run, not tracked.
- **Tuning:** none. Every constant is fixed here.
- **Resumability:** per run, with `experiments/durable.py`.
- **Runtime JSON** (per `computation-reporting.md`), by phase: problem generation (with rejection counts), training,
  each test family, and reader passes. The `reader` condition depends on the untracked checkpoints
  `runs/amortized-introduction-v1/virtual-seed45{0..3}.pt`, whose sha256 are in that study's report.

## Commands

    EV_WORKERS=6 .venv/bin/python -m experiments.rule_application       # rerun to resume
    .venv/bin/python -m experiments.rule_application summary

## Results (2026-11-07)

Run at revision `9b2720f`, 12 runs, 6 workers, 1.75 h wall. Means over seeds 460–463. Balanced accuracy per depth =
mean of T recall at that depth and F recall over the non-derived atoms of derived relations.

### The networks did not learn the task at the trained depth

Training-batch balanced accuracy (base atoms included) was still rising at step 6,000: 0.86 (`edge-loop`), 0.90
(`edge-untied`), 0.85 (`pair-loop`). On fresh in-distribution problems (depth ≤ 2):

| Arm (T) | depth 1 | depth 2 | T recall d1 / d2 | F recall |
|---|---|---|---|---|
| `edge-loop` (4) | 0.75 | 0.67 | | |
| `edge-loop` (16) | 0.68 | 0.61 | 0.62 / 0.49 | 0.73 |
| `edge-untied` (4 layers) | **0.83** | **0.73** | 0.88 / 0.67 | 0.79 |
| `pair-loop` (16) | 0.63 | 0.58 | 0.64 / 0.54 | 0.62 |

Synthetic test, depths 3–6, is near chance in every arm (0.53–0.58). T recall falls from 0.39–0.49 at depth 3 to
0.35–0.48 at depth 6. More entities (12–16, depth ≤ 2) cost 0.02–0.05.

### Taught relations, `oracle` facts: exactly the trained depths, nothing beyond

| Relation | arm | depth 1 | depth 2 | depths ≥ 3 | world labels |
|---|---|---|---|---|---|
| `bigger-than` (1–11) | `edge-untied` | 1.00 | 1.00 | 0.50 (T recall 0.00) | 0.66 |
| | `edge-loop` T16 | 0.94 | 0.51 | 0.44 | 0.55 |
| `ancestor-of` (1–5) | `edge-untied` | 0.98 | 0.93 | 0.48 | 0.86 |
| | `edge-loop` T16 | 0.99 | 0.49 | 0.49 | 0.72 |
| is-a (1–4) | `edge-untied` | 0.99 | 0.99 | 0.49 | 0.72 |
| | `edge-loop` T16 | 0.98 | 0.61 | 0.48 | 0.67 |
| `smaller-than` (2–12) | `edge-untied` | | 0.95 | 0.45–0.66 | 0.56 |
| `grandparent-of` (1) | `edge-untied` / `edge-loop` | 0.99 / 0.75 | | | 0.98 / 0.74 |
| `child-of` (1) | `edge-untied` / `edge-loop` | 0.99 / 0.99 | | | 0.98 / 0.98 |

`pair-loop` is like `edge-loop` or worse (`grandparent-of` 0.47). The oracle closures match the world's labels except
6 of 281 is-a pairs.

**Diagnostic after the run** (`edge-loop`, seeds 460–461, T = 1…8 on the taught chains): depth-1 recall is 1.00 at
every T ≥ 2. Depth 2 reaches at most 0.6–0.9 at T = 3–4 and 0 elsewhere. Depths ≥ 3 are 0 at every T. The tied loop
applies the one-step rule and almost never the recursive one. Extra loops do not extend its reach, and F recall falls
with T (bigger-than 1.00 at T = 2, 0.72 at T = 8).

### Answers do not follow rule changes in synthetic problems

| Arm | relevant edits: flipped atoms correct (e1 / e2 / e3) | irrelevant edits: answers unchanged (fresh vectors / dead rule / unused relation) |
|---|---|---|
| `edge-loop` | 0.50 / 0.50 / 0.44 | 0.65 / 0.90 / 0.84 |
| `edge-untied` | 0.54 / 0.52 / 0.42 | 0.71 / 0.93 / 0.88 |
| `pair-loop` | 0.53 / 0.52 / 0.48 | 0.59 / 0.87 / 0.83 |

On taught problems, `edge-untied` keeps 0.95–1.00 of answers under fresh vectors and dead rules, and follows some
edits: is-a with swapped kind arguments 0.92, `child-of` 1.00, `ancestor-of` from `child-of` 0.75. Head swaps
0.25–0.59.

### Reader: hop level and knowledge damage

| Base relation | reader precision / recall (raw) | after immediate reduction |
|---|---|---|
| `parent-of` | 0.72 / 0.97 | |
| `just-bigger-than` | 0.90 / 0.95 | |
| `kind-of` (all memberships) | 0.33 / 0.75 | 0.08 / 0.10 |
| `subkind-of` | 0.04 / 0.92 | |

Knowledge damage, exact closure of the reader's facts against the oracle (T recall by depth, F recall):

- `ancestor-of`: 0.97 → 0.83 from depth 1 to 5, F 0.98;
- `bigger-than`: 0.96 → 0.75 from depth 1 to 11, F 0.98;
- is-a: 0.43–0.46, F 0.60. The reader's false `subkind-of` links remove true immediate memberships and add false
  chains.

With perfect application, the system's own knowledge would support `ancestor-of` and `bigger-than` chains well and
is-a poorly. The network on reader facts against `reader-closure` follows its `oracle` pattern: depth 1 0.86–0.96,
deeper at chance.

### Against the predictions

1. **Failed.** `edge-loop` T16, depths 3–4: 0.55 / 0.54; 5–6: 0.54 / 0.53. Connected negatives 0.71 (path 1–2), 0.93
   (3–4). Longer-path buckets are empty: synthetic graphs with up to 16 entities have no long undirected paths.
2. **Failed.** Extra loops hurt: depths 5–6 0.537 / 0.532 at T16 against 0.540 / 0.532 at T4; depth ≤ 2 falls
   0.06–0.07.
3. **Held except depth 1.** Tied at T = 4 against untied: 0.054, 0.026, 0.011, 0.010 below on depths 1–4.
4. **Failed.** `pair-loop` equals `edge-loop` at depths 3–6 (both at chance).
5. **Failed beyond the trained depth.** `edge-untied` meets the thresholds at depths 1–2 (0.93–1.00), every arm is at
   chance beyond that.
6. **Not testable beyond depth 1.** Depth 1 on reader facts 0.86–0.96 against 0.98–0.99 on oracle facts; deeper both
   at chance.
7. **Failed.** Flipped atoms 0.42–0.54; irrelevant edits keep 0.59–0.93.
8. **Failed.** `edge-loop` depth ≤ 2 loses 0.06–0.07 from T = 4 to 16.
9. **Failed.** Held-out form, T recall 0.44 (T16) to 0.60 (T4).
10. **Failed.** In distribution 0.63–0.83 at depth 1, 0.58–0.73 at depth 2 (predicted ≥ 0.97).

### What this means

- **The question was not reached.** Depth generalization needs a network that has learned 1–2 step application. None
  had: the best is 0.83 / 0.73 in distribution, with training still improving. Every deeper read inherits that.
- **Where application was learned, it was learned as a fixed depth, not a repeatable step.** On clean taught chains,
  the untied network is exact at depths 1–2 (`bigger-than` 1.00 / 1.00, is-a 0.99 / 0.99) and says T to nothing at
  depth 3. It learned "apply the rules up to twice". That matches the length-generalization failure of fixed-depth
  networks.
- **The tied loop never became an iteration.** It applies one-step rules and not the recursion, and running it longer
  degrades it. Random T in 2–4 with loss only at the end never required step k to do the k-th application. A network
  can satisfy that loss by finishing early and then holding still, or by not iterating at all. Looped Transformers
  supply each example's step count, and this study did not. That difference is the most likely cause, but untested.
- **The synthetic problems are much harder than the taught ones.** Many interacting rules over dense random facts. The
  untied network solves clean taught chains at depth ≤ 2, but only 0.73 of synthetic depth-2 atoms.
- **The reader's knowledge is a separate limit.** Its closure supports `ancestor-of` and `bigger-than` chains to
  0.75–0.97 and is-a only to 0.43–0.46, because of false `subkind-of` links. This is the hop level found in conflict
  search, now measured as damage.
- **Candidates for the next step**, not chosen:
  - per-step supervision (atoms of depth ≤ k labelled T after loop k), which is what makes looped models iterate;
  - longer training, or a curriculum (depth 1 first);
  - simpler synthetic problems (fewer interacting rules, sparser facts);
  - a stopping signal.

### Computation (measured)

`reports/rule-application-v1-runtime.json`. Per run:

| Phase | `edge-loop` | `edge-untied` | `pair-loop` |
|---|---|---|---|
| training, CPU s | 2,657–2,663 | 2,718–3,406 | 1,667–1,678 |
| synthetic reads | 382–404 | 26–50 | 150–165 |
| edit reads | 250–255 | 34–57 | 81–115 |
| taught reads (including reader passes) | 29–34 | 9–12 | 20–22 |

- **Training includes problem generation:** about 400 CPU s per run, about 953 k attempts for 192 k problems.
- **Test generation:** 73–85 CPU s.
- **Parameters:** 138,793 / 401,257 / 130,473.
- **Peak RSS:** 0.66–0.75 GB per worker.
- **Total:** 34,736 CPU s (9.6 CPU h), 6,284 s wall, within the declared 1–2.5 h.
- **Diagnostic after the run:** about 2 CPU min. No tuning.
