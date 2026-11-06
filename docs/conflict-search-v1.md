# Conflict search v1: backtracking over middle things in the logic route

Status: **contract frozen 2026-11-06 on the user's instruction ("yes go ahead & run"), after two external reviews of
the draft.** Follows [conflict loop v1](conflict-loop-v1-results.md) and the external review of it. Decisions taken
while writing the runner are in "Implementation (recorded at freezing)".

## User direction (2026-11-06)

- "What if we allow actual loops?" The step's output should go back in as a new question, with steps not fixed in
  advance.
- "How will backtracking, or starting from the middle, work, instead of a loop: a tree of reasoning but one path at a
  time."
- Draft the contract.

## Why

Three defects of conflict loop v1, confirmed in `experiments/conflict_loop.py`:

1. **The loop's step shared the reader with step 0.** The loss on withheld facts trained G_L, S_L and E_L directly, so
   step 0 reached 0.81 on facts it was never told, and a near-identity produce head was the cheapest fit.
2. **The trigger extinguished itself.** Recomputed every step from step 0's disagreement with intuition, it fired on
   6% of withheld rows by the end (98 of 1,558).
3. **Half the rows of a relation is not the relation withheld.** The told half taught the higher-order relation code as
   a one-pass readout; the withheld half was interpolation.

The loop also had no memory of what it tried, could not revise a wrong middle, and had a fixed depth of 3.

What a language model's chain of thought has and the loop lacked **[familiar; derived]**: the intermediate result is
discrete and committed, it is fed back as an input through a reader that cannot change at question time, and the
chain is read by a check. This contract gives the logic route those properties without text: a walk over entities
following chosen relations, one path at a time with backtracking, a trace, a budget, and the readers frozen inside the
search.

## Design

### Routes, as in v1

- **Intuition:** own codes, one-pass verdict u, trained on all trained individual T/F statements.
- **Logic:** own codes (E_L, R_L, G_L, S_L), one-pass support (s⁺, s⁻), kind quantities, fill-the-tail, connectives, as
  in dual-route v1. **Logic one-pass is trained on base relations and every non-chain statement only.**

### Pressure: whole relations withheld from logic's one pass

Logic's reader never sees a statement of `grandparent-of`, `ancestor-of`, `kind-of`, `bigger-than`, `smaller-than`,
`named-can`, `named-has`, `named-kind` or `same-colour-as`. Intuition sees them all. Their base relations (`parent-of`,
`is-a`, `subkind-of`, `just-bigger-than`, `names`, `can`, `has`, `colour-of`) are told to both routes.

**The withheld relations' codes in R_L are never trained.** They stay at their random initialization and serve only as
identifiers of the question's relation at the controller's input. Nothing in phase 2 moves any reader parameter.

**Pre-registered smoke check:** what logic's frozen one-pass returns for an untrained relation code is unknown. The
smoke run reads the distribution of step-0 support on withheld-relation statements and the trigger rate where
intuition is confident. If the trigger fires on fewer than 0.8 of those rows, phase 1 adds a target of no support
(s⁺ = s⁻ = 0) on withheld-relation statements, so every such question conflicts with a confident intuition. That
tells logic which relations it was never told, and is recorded as a decision if taken.

### Phase 1: readers

Both routes trained as in v1 (full batch, Adam lr 3e-3, weight decay 1e-4, 3,000 steps), on the told statements.
Phase 1 is identical for every arm of a seed and is trained once per seed.

### Phase 2: the controller, with the readers frozen

A question (a, r, b) with a conflict between intuition's σ(u) and logic's step-0 support (d0 > τ, τ = 0.5) is
**searched**. The search walks from a over entities, one hop at a time, following relations the controller chooses,
and succeeds when the walk reaches b. It is depth-first, one path at a time, with a stack of untried candidates and a
trace of what was tried in working memory. Nothing inside the search changes the readers: G_L, S_L, E_L, R_L and
intuition are detached.

```
search(a, r, b):
    node = a; depth = 0; stack = []; trace = []; checks = 0
    while checks < B:
        state = GRU(trace; [c_node, c_b, R_L[r], depth])
        ρ = R_L[argmax W_ρ state]                        straight-through: pure code forward, softmax gradient back
        q = Q(state);  cands = top-k entities by cosine(q, E_L), not yet tried at this node
        push cands[1:] with (node, depth) on the stack;  m = cands[0]
        v = ask(node, ρ, m);  checks += 1                 one hop, through the front door
        if not v.confirmed:  trace ← (node, ρ, m, failed);      (node, depth) = pop(stack) or exhausted
        elif m == b:         return found, p = ∏ hop supports along the path, path
        elif depth + 1 < D_max:  trace ← (node, ρ, m, ok);  node = m;  depth += 1
        else:                trace ← (node, ρ, m, dead end);  (node, depth) = pop(stack) or exhausted
    return exhausted

ask(x, ρ, y):
    u = intuition(x, ρ, y);  s = logic one-pass(x, ρ, y);  d = disagreement(σ(u), s)
    confirmed iff d ≤ τ and s⁺ > 0.5;  hop support = s⁺
```

- **Every hop is a check,** by both routes. There is no unchecked read. A hop on a withheld relation code conflicts or
  reads arbitrarily and is simply not confirmed, so the controller learns to walk on told relations without being given
  the list.
- **Termination is reaching b.** Entities are committed, so this is an identity test, not a threshold.
- **Budget** B = 24 checks per question; k = 3 candidates at the first choice point and 2 at deeper ones (worst case
  3 · 2³ = 24 for four hops); D_max = 5. Budget spent or stack empty: the search returns **exhausted**. In the fixture
  every person has 1 or 2 children and 1 or 2 parents (15 and 12 of each), so a walk that only proposes real children
  needs a handful of checks at any depth. B governs cost; k governs misses: a node whose true next entity is not among
  the k proposals is a miss no budget repairs. The depth-5 concern is a proposer concern, read directly below.
- **Inhibition of return:** tried candidates are excluded from later proposals at the same node. This is search
  discipline, like the stack, not a rule about the world. It is built in; a control arm removes it along with the trace.
- **Logic's output after the search:** found → (p, 1 − p); exhausted → (0, e), where e = σ(w_e · [state, checks / B])
  is a learned give-up scalar. Exhaustion is not assumed to mean false: what it means is learned by e and by the arbiter.

### Forward and bidirectional

| Mode | Walkers | Proposer input | Termination |
|---|---|---|---|
| `forward` | one, from a | [c_node, c_b, R_L[r], state] | node == b |
| `bidir` | two, from a and from b, extended in alternation | [c_node_a, c_node_b, R_L[r], state] | the two frontier nodes meet (identity), or a confirmed candidate on either side equals a node already on the other side's path |

A backward hop proposes an entity e and checks ask(e, ρ, node_b): e is ρ-related to the b-side node. `bidir` meets in
the middle, so a chain of n hops needs about n / 2 levels on each side, and the proposer sees both ends from the start.
This is what "starting from the middle" becomes once every hop is checked. Alternation is fixed in this version: in
the trees forward and backward branching are the same (1 or 2 children, 1 or 2 parents), and the symmetric relations
are one hop each side, so no side is systematically cheaper. A learned side choice would be a third discrete decision
trained only by outcome, and is held until the first two are shown to learn.

### Controller training

Phase 2 runs the search on trained statements of the withheld relations, in batches, with the readers frozen: every T
row, and F rows subsampled each step to the number of T rows (F rows mostly exhaust and carry little signal). Only W_ρ,
Q, the GRU and w_e learn, from three signals:

1. **Outcome:** binary cross-entropy of logic's output after the search against the truth label. Differentiable through
   p into W_ρ (straight-through, with the frozen reader evaluated on pure relation codes) and into e.
2. **Ranking from the search's own checks:** at each choice point, the candidate on the found path is the positive and
   every other tried candidate, failed hop or dead end, is a negative, for Q (softmax cross-entropy over the tried
   candidates). Rows that exhaust give no ranking signal. This is the only signal for the discrete choice of entity. It
   is self-generated, so a wrong reader can train a confident wrong proposer; the final label stays the only ground
   truth, and the true-path read measures the damage.
3. **KL to a geometric prior on checks used** (p = 0.5, weight 0.01), as in v1, to keep paths short when short ones
   work.

Adam lr 3e-3, 1,000 full-batch steps. Every withheld training row is searched in training, so the trigger does not
gate the controller's training set. At test the search runs on conflict only; a diagnostic read runs it on every row.

### Arbiter

As in v1 (trained on the 5% validation statements only, routes and controller frozen, 500 steps), with 9 inputs:
σ(u), s⁺ and s⁻ after the search, d0, found, exhausted, checks / B, path length / D_max, and e. 9 → 16 → 3.

## Arms

| Arm | Phase 2 | Trace and exclusion | What it tests |
|---|---|---|---|
| `one-pass` | none; logic is step 0 | — | floor: whole relations withheld, no search; the arbiter on it |
| `soft-loop` | v1's loop (T = 3, PonderNet mixture) with the readers detached, and the produce head replaced by attention over E_L weighted by softmax of the frozen S_L(node, ρ, e) | — | the cheaper alternative: one soft step per hop, no commitment, no backtracking |
| `bt-forward` | the walk from a | yes | the design |
| `bt-bidir` | the walk from both ends | yes | meeting in the middle |
| `bt-forward-notrace` | as `bt-forward`, state from [c_node, c_b, R_L[r], depth] only; tried candidates not excluded | no | whether memory of failure matters (removes both the trace and the exclusion) |

Seeds 450–453, the same splits as v1. 4 phase-1 runs, 16 phase-2 runs. Statements as in multi-step reader v1
(`src/everyday_truth_v2.py`: 246 entities, 47 relations). Reference numbers: v1's `arbiter` arm (everything told).

## Reads

**Answers,** every T/F test of v1, as intuition / logic step 0 / logic after the search / arbiter, balanced accuracy
and AUROC. **By depth:** ancestor 3–5 (trained 1–2), kind-of 3–4, size distance 3–11, named-kind by its chain length.
Ancestor pairs by depth in the fixture: 9 at depth 3, 6 at depth 4, 3 at depth 5 (test rows are doubled by their
reverses). Depth 5 is reported as counts, and thresholds are set on depth 4 and 5 pooled (18 rows).

**Unknown:** S4a, P7, "sometimes → unknown" on new individuals (S3b), the dax, U on inferable facts.

**Is it reasoning?**
- checks used, backtracks and path length by depth, and the share of questions that exhaust the budget, on T, F and U
  facts separately;
- the index of the candidate that succeeded at each choice point (1 = the first proposal);
- **proposer recall:** on true rows, at each confirmed node, whether the true next entity on some true path was among
  the k proposals, by depth. This separates "the proposer missed" from "the budget ran out", which the exhausted share
  alone cannot;
- **backward waste** (`bt-bidir`): the share of confirmed backward hops that are not on the found path. If it is high, a
  learned side choice would pay in a later version;
- **true-path rate:** every hop of the found path is a true base fact in the world, by relation and depth;
- **the program:** the sequence of chosen relations per question relation (grandparent should read `parent-of`,
  `parent-of`; size `just-bigger-than` repeated; named relations `names` then the base; ancestor `parent-of` repeated);
- trigger rate on withheld-relation tests, and the diagnostic search-on-every-row accuracy, to separate trigger
  failures from search failures;
- per-hop accuracy of the frozen base reader, so compounding can be computed and compared with the depth curve.

**Conflict:** d0 as an error signal for intuition, as in v1.

**Cost:** reader calls and cosine scorings per question (mean and maximum), parameters, wall time per 1,000 questions
at inference (median of 5 repeats after warm-up), separately for triggered and untriggered questions.

## Predictions

1. **[v1 code; derived]** With the readers frozen inside the search and whole relations withheld, `one-pass` logic is
   near chance on the withheld relations (balanced accuracy ≤ 0.6), and its step 0 is identical before and after
   phase 2 (nothing can move it). Trigger rate on withheld-relation tests where intuition is confident ≥ 0.8, after the
   smoke check and its fallback if taken.
2. **[MINERVA, Neural LP; familiar; derived]** `bt-forward` and `bt-bidir` beat `one-pass` logic by ≥ 0.15 on
   grandparent, size and named relations, and the program read shows the base relations chosen on ≥ 0.8 of found paths.
3. **[derived]** Depth generalization: ancestor depth 4 and 5 pooled after the search ≥ 0.7 (v1 arbiter 0.81 and
   0.54 separately; the depth-5 figure there was 6 rows and its 0.54 against 0.50 is noise), with checks used growing
   with depth. `bt-bidir` ≥ `bt-forward` on the pooled rows and exhausts less often on true depth-5 facts. Proposer
   recall ≥ 0.9 at depths 1–3 on true rows; where it falls at depth 4–5, exhaustion follows it.
4. **[derived]** True-path rate ≥ 0.8 where a path is found. Every hop is checked by both routes, so a false path needs
   both routes to confirm a false base fact; if the rate is lower, the base readers agree on wrong facts and the per-hop
   read will show it.
5. **[derived]** `bt-forward-notrace` repeats failed candidates and exhausts the budget more often (≥ 1.5× the exhausted
   share of `bt-forward` on true facts), and is below it on depth ≥ 4. This control removes both the trace and the
   exclusion, so it does not separate the two.
6. **[v1; derived]** `soft-loop` gains ≤ 0.08 over `one-pass` on held-out tests and does not generalize past depth 3:
   a blend of candidates is not an entity, and the reader scores a blend unreliably. If it matches `bt-forward`, the
   commitment and backtracking were not needed.
7. **[v1; read]** The arbiter keeps intuition's chains within 0.05 and ≥ 0.5 unknown on S4a with ≤ 0.05 unknown on
   inferable facts, in every arm. Unknown after exhaustion: the arbiter says U on ≥ 0.5 of held-out unknown colours
   where the search exhausted.
8. **[taught truth v1; derived]** "Sometimes → unknown" on new individuals stays ≤ 0.3 for logic: the search checks T/F
   hops and does not read a kind's quantity. A known limit of this version, not a test it can pass.
9. **Risk [derived]:** compounding. Per-hop base accuracy around 0.9 bounds a 5-hop forward path near 0.6; the depth
   curve should not fall faster than the per-hop product predicts. If it does, the proposer, not the reader, is the
   bottleneck.
10. **Risk [MINERVA; familiar]:** sparse outcome signal. F rows and exhausted T rows give no ranking signal, and the
    outcome gradient exists only on found paths. If found paths stay below 0.3 of true training rows after 250 steps,
    the controller is not learning from the search and the run is stopped and recorded.

## Prior work

No new search was run on 2026-11-06; sources are **[familiar]** unless tagged. Shared findings are recorded in
[prior work](prior-work.md), section "Backtracking search over middle things".

- **Walking a graph with a learned policy:** MINERVA (Das et al. 2018) is the forward mode exactly: an agent at an
  entity chooses a relation and a next entity each step, trained by reinforcement on reaching the answer, with a path
  to read. DeepPath (Xiong et al. 2017) learns relation paths the same way. Known failures: sparse reward, and a beam
  needed at test. Differences here: backtracking with a trace instead of a beam, every hop checked by two learned
  routes rather than looked up in a graph, candidates proposed by a learned query over codes, and the ranking loss from
  the search's own checks as the reward shaping MINERVA lacked.
- **Differentiable backward chaining:** Neural Theorem Provers (Rocktäschel and Riedel 2017) and Conditional Theorem
  Provers (Minervini et al. 2020); exponential without k-max pruning. Here one path at a time with a budget.
- **Learned sequences of base relations:** Neural LP (Yang et al. 2017), DRUM (Sadeghian et al. 2019): relation matrix
  products over all entities. The termination rule "end when the node reads as b" is theirs; here with committed
  entities and a universal learned reader.
- **Learning the policy from search outcomes:** expert iteration (Anthony et al. 2017) and AlphaZero-style training.
  Known failure: self-generated labels amplify the evaluator's errors; the final label and the true-path read guard it.
- **Reasoning in language models [familiar]:** chain of thought, Tree of Thoughts (Yao et al. 2023), self-consistency,
  and backtracking learned by reinforcement on outcomes (DeepSeek-R1, 2025). The compositionality gap (Press et al.
  2022) and grokked two-hop composition that fails out of distribution (Wang et al. 2024) predict prediction 1's floor.
  Unfaithful chains (Turpin et al. 2023) motivate the true-path read.
- **Straight-through estimation** (Bengio et al. 2013) for the relation choice, so the frozen reader is only evaluated
  on pure relation codes.
- **Negation as failure and the closed world:** exhaustion treated as false is the Prolog convention. Here what
  exhaustion means is learned (e and the arbiter), because the world has unknowns.
- **Working memory as a stack:** neural stacks (Grefenstette et al. 2015; Joulin and Mikolov 2015). The stack
  discipline is fixed and its contents learned, as the loop's step structure was fixed in multi-step reader v1.
  Inhibition of return (Posner and Cohen 1984) is the cognitive analogue of excluding tried candidates.
- **Facts in context rather than weights:** CLUTRR (Sinha et al. 2019). Not this study; the next one, where a hop
  becomes a memory read.
- **Cheaper alternatives included:** `one-pass` and `soft-loop` (one attention step over entities scored by the frozen
  reader, no commitment). Not included: a beam over k parallel paths (MINERVA's test-time remedy); `bt-forward-notrace`
  is the nearer control.
- **What this adds:** a trace-conditioned learned walk with every hop checked by two learned routes, supervised by the
  system's own checks, on the program's dual-route reader, read for faithfulness and for depth beyond training. Neural,
  with no rule or relation form given: the controller sees codes, scores and the trace, and is not told which relations
  are base relations.

## Scale

- **Per question:** search only on conflicts. Each check costs 2 reader calls (both routes); each choice point costs one
  cosine over the entity table. Worst case B checks: 2B reader calls plus at most B cosines, independent of the number
  of relations. The cosine over entities is O(n_ent); at scale the candidate set comes from the memory module's
  addressing rather than a full table, which is the next study.
- **Depth** is bounded by D_max and B, not by the architecture: deeper chains cost checks, not parameters.
- **A new higher-order relation** costs one relation code and controller training on its examples; no new reader.
- **Online learning:** the controller's three losses need the search trace of one question; the readers are untouched.
  Keeping readers frozen inside the search is also what stops the search from moving the codes.

## Computation (declared)

- **Phase 1:** as dual-route v1, about 16 CPU min per seed; 4 runs.
- **Phase 2:** per step, a batched depth-first walk over all T withheld rows plus as many F rows: up to B = 24 checks ×
  2 reader calls on the batch plus a cosine over 246 entities per choice point, and the backward pass through the
  controller. Estimate 3–4× a v1 training step; 1,000 steps ≈ 30–50 CPU min per run; 12 search runs. `soft-loop` is
  about v1's loop cost; `one-pass` needs no phase 2.
- **Total:** about 8–13 CPU h; about 2–3 h wall on 6 workers. If the smoke run projects more than 3.5 h wall, phase-2
  steps are lowered for all arms together and recorded.
- **Storage:** parameters about 110k (float32); the trace and stack are temporary per question (≤ B entries); no raw
  samples retained. Checkpoints in `runs/` (not tracked); reports in `reports/conflict-search-v1{,-summary,-runtime}.json`
  with the runtime JSON as required by [computation reporting](computation-reporting.md).
- **No tuning:** τ, B, k, D_max, the prior and the arbiter's size are fixed here. The smoke run checks the runner and the
  pre-registered R_L check only.
- **Held for v2, not this run:** scoring all entities with the frozen logic reader at each node and letting Q order
  the reader's candidates. That makes the action space the real outgoing edges, as in MINERVA, and nearly removes
  misses at this scale, but it shrinks Q's role and is a design change. It is taken only if the smoke run shows
  exhaustion on true depth-4 training facts, and recorded if so.

## Commands (planned)

```
EV_WORKERS=6 .venv/bin/python -m experiments.conflict_search phase1    # 4 seeds, resumable
EV_WORKERS=6 .venv/bin/python -m experiments.conflict_search main      # 16 phase-2 runs, resumable
.venv/bin/python -m experiments.conflict_search summary
```

## Review record (2026-11-06)

External review of the first draft, all points accepted:
- recursion was keyed to reader conflict, so chains of depth ≥ 3 could only go deeper through a withheld relation code
  and had no way to know the last hop; replaced by the walk that ends when the node is b;
- ranking positives were candidates whose second half passed regardless of the first; now only candidates on a found
  path, with every other tried candidate negative;
- the first half was an unchecked read; now every hop goes through `ask`;
- R_L was both detached and trained; now never trained, with the pre-registered smoke check and fallback;
- soft relation mixtures reached a reader trained on pure codes; now straight-through argmax;
- MINERVA and DeepPath added; `loop-frozen` became `soft-loop` with an attention read; budget made per level (k = 3 then
  2, B = 24); F rows subsampled; the no-trace control stated to remove two things; exclusion kept built in; quantities
  left out.

Second review, both open points accepted as drafted with one read each: budget kept (branching in the trees is at
most 2, so exhaustion is a proposer miss, not a budget shortage), proposer-recall read added, depth 5 reported as
counts with the threshold on depth 4 and 5 pooled, the all-entities reader scoring held for v2 with a smoke-run
condition; alternation kept, backward-waste read added, `bidir` termination made symmetric. Tree counts verified
against `src/everyday_truth_v2.py`.

## Implementation (recorded at freezing)

- **Runner:** `experiments/conflict_search.py`. Reuses dual-route v1's `Dual` (separate codes), its one-pass logic and
  intuition, and v1's halting machinery for `soft-loop`.
- **Each hop is a relation and a direction.** The walk needs to read a relation backwards: `smaller-than` is a chain of
  `just-bigger-than` read from the larger side, and `same-colour-as` is `colour-of` then its reverse. Without a direction
  choice neither is reachable from a by any walker, forward or bidirectional. The controller chooses a direction per hop
  (2-way head, straight-through); a reversed hop checks ask(m, ρ, node). MINERVA and Neural LP include inverse edges in
  the action space the same way. This is a capability of reading, not a fact about the world.
- **Straight-through for the direction too:** logic's support is computed for both directions on each hop; the value
  is the chosen one and the gradient to the direction head is the difference. Intuition is computed for the chosen
  direction only (no gradient).
- **Fill-the-tail rows of the withheld relations are withheld too** (they would teach logic the relation through the
  code head). Logic's phase-1 losses skip them; intuition does not use fill rows.
- **Withheld relation codes are restored to their initial values after every phase-1 optimizer step,** because Adam's
  L2 weight decay would otherwise shrink the never-used rows to zero and make the question relations indistinguishable
  at the controller's input.
- **Readers frozen in phase 2** by `requires_grad_(False)` on every `Dual` parameter; the controller's gradient reaches
  the frozen readers only through its own relation and direction choices.
- **The step prior is dropped.** Checks used are a hard count in a committed search, with no halting distribution to
  take a KL against. Cost is controlled by B. Recorded as a departure from the draft's third signal.
- **Controller state:** at a choice point the heads read [h, u], where h = tanh(W [c_node, c_node_b, R_L[r], depth /
  D_max]) and u is the GRU state over the trace (zero in `bt-forward-notrace`). The GRU input per check is [c_node,
  relation code, c_m, outcome one-hot(ok, failed, dead end)].
- **Phase-2 rows:** the withheld relations have 3,287 trained individual rows per seed, of which 266 are T. Each phase-2
  step searches all T rows and a fresh sample of 266 F rows, so the declared cost estimate was high: the smoke run
  measures it.
- **`soft-loop`:** v1's loop (T = 3, PonderNet mixture over steps 1–3 on withheld rows) with produce replaced by
  attention over E_L weighted by softmax of the frozen S_L(node, ρ, e)⁺ logits at a learned temperature, readers frozen,
  relation choice straight-through as in the search arms.
- **Proposer recall** is computed against the set of entities at the next position on any true path from a to b in the
  base-fact graph restricted to the relations the question's relation unfolds to (a diagnostic using world knowledge,
  as the true-middle read in v1 did). Choice points whose node is on no true path are counted separately.
- **Bidirectional alternation:** the side to extend is (depth_a + depth_b) mod 2; total depth is bounded by D_max; one
  stack of frames tagged by side; termination when a confirmed candidate equals the other side's frontier or any node on
  the other side's path.
- **Pre-registered R_L check:** the smoke run records logic's step-0 support and the trigger rate on withheld-relation
  training rows where intuition is confident; the fallback target is a constant in the runner, off unless the check
  fails, and its setting is recorded in the results.

## Smoke-stage record (2026-11-06, after freezing; main run not started)

Seed 450 only. Nothing here is a result of the study.

1. **Pre-registered R_L check failed; the fallback was taken.** Without it, the frozen one-pass answered a confident F on
   0.93 of withheld-relation training rows, and the trigger fired on 0.13. With the no-support target, the trigger fires
   on 1.00 (logic answers U on all 3,121 rows). `WITHHELD_NO_SUPPORT = True` in the runner.
2. **The hop check is sound.** On trained base facts both routes confirm 1.00 of true and 0.00 of false; on all base
   facts 0.90 and 0.002. From a person, logic relates about 6 entities by `parent-of`, mostly non-people (never-seen
   type combinations), and both routes together confirm about 1.2 (true children 0.93).
3. **Amendment: the action space is the frozen reader's edges** (the change held for v2 in the second review, taken
   because every true training row exhausted). The relation is chosen among those under which logic relates the node to
   some entity (about 43 of 47 per entity, so this barely narrows it); candidates are the entities logic relates under
   the chosen relation and direction; Q orders them. Relation, direction and candidate are sampled in training and
   taken by argmax at test. The edge table is computed once per model, since the readers are frozen.
4. **Bug fixed: the target was excluded from the candidates.** The mask excluded both frontier nodes; in the forward
   walk the other frontier is b, so b could never be proposed and no path could be found. Now only the current side's
   node and path are excluded. Before the fix every probe found 0 paths, even with the base relations given.
5. **Stopping rule (prediction 10) fires on the probe.** `bt-forward`, 250 phase-2 steps:

   | Step | Found, true rows | Found, false rows | Ranking loss |
   |---|---|---|---|
   | 0 (untrained) | 0.04 | 0.005 | — |
   | 25 | 0.12 | 0.008 | 0.24 |
   | 100 | 0.11 | 0.027 | 0.24 |
   | 250 | 0.15 | 0.023 | 0.09 |

   With the question's base relations given (diagnostic hook `rel_oracle`, untrained controller): 0.11 of true rows,
   0.025 of false.
6. **Diagnosis.** The ranking loss trains only the entity proposer Q and falls, but found paths do not grow. The relation
   choice gets gradient only through the outcome loss, via the product of hop supports on found paths, which are near 1
   for confirmed hops. So the choice among about 43 relations is barely trained. The contract's signals do not reward
   the relation on a found path.
7. **Not run:** the main stage. Phase 1 of seed 450 with the fallback is in `runs/` (6.7 CPU min).
8. **Amendment on the user's direction ("continue to the next step after the fixes"): the relation and direction on a
   found path are positives at their choice points,** like the entity (cross-entropy on the masked relation logits and
   the direction logits). Probe, 250 steps: found on true rows 0.14 at step 25, 0.26 at 125, 0.22 at 250 (false rows
   0.004–0.039). Still below the 0.3 threshold of prediction 10, but rising from 0.04 untrained. The user directed the
   main run to proceed; the threshold miss is recorded here and the run's training logs report found rates every 100
   steps.
