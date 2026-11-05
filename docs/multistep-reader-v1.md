# Multi-step reader v1: a reader that thinks in steps

Status: **draft contract, 2026-11-06. Not frozen; awaiting user review.**

User direction (2026-11-06), after [everyday relations v3](everyday-relations-v3-results.md):
- Multi-step inference is needed, and it cannot be a 2-step patch: the readers' architecture must change. A recurrent
  ("feed-forward RNN") reader that thinks in steps.
- No memory system. Working memory is allowed: it is attention.
- Run both attention arms: the reader-scored fetch and standard dot-product attention with learned key, value and query
  projections.

## Why

In v1–v3 every remaining failure chains two or more relations through a middle entity: category from meaning (word →
animal → noun), sense of a phrase (word → animal → ability), same colour (thing → colour → thing), opposite from meaning.
The one-pass reader judges a pair it is handed. It has no way to look up the middle entity and read again.

## The reader

A **controller** (a small recurrent network over thinking steps, weight-tied) runs a loop around the existing relational
reader. **Working memory** is one slot holding a code: the "current thing".

```
question (a, r, b) ──► s0 = Init([c_a, r, c_b]);  current = c_a
each step t = 1..T:
   relation choice   ρ_t = softmax(W_ρ s_t) · R             (a soft mix of the known relation codes)
   FETCH             current ← attention over all entity codes, scored for (current, ρ_t, x)
   update            s_{t+1} = GRU(s_t, [current, ρ_t])
   halt?             λ_t = σ(w_h · s_{t+1})
   answer at t       y_t = JUDGE(current, ρ'_t, c_b),  ρ'_t = softmax(W_ρ' s_{t+1}) · R
output: PonderNet expectation  P(yes) = Σ_t p(halt at t) · σ(y_t)    (t = 0 judges with current = c_a, no fetch)
```

- **Nothing about the chain is given.** The controller is not told which relations to follow or how many steps to take.
  It is trained on final yes/no answers only, from the same facts as before.
- **JUDGE** is the existing reader, `tworel`: `S([G([c_a, c_b, ρ]), ρ])` (v2–v3's best two-space form).
- One-step facts are answered at t = 0 if the controller halts at once.

### The two fetch arms (both real attention; they differ in the score function)

| Arm | Score for entity x | Value | Established name |
|---|---|---|---|
| `loop-reader` | the judge itself: `S([G([current, c_x, ρ_t]), ρ_t]) / τ` | `c_x` | content-based attention with a learned score network (the additive-attention family, Bahdanau et al. 2015) |
| `loop-dot` | `(W_Q [s_t, current, ρ_t]) · (W_K c_x) / √d_k` | `W_V c_x` (into code space) | scaled dot-product attention (Vaswani et al. 2017) with learned Q, K, V |

- In `loop-reader`, fetching reuses the relational knowledge used to judge. "blick names owl" is known once.
- In `loop-dot`, what to fetch must be learned into W_Q and W_K, a second copy of relational knowledge. It is cheaper.

τ is a learned temperature. d_k = 32.

## The world

v3's world (`src/everyday_world_v3.py`) with chains added. The changes go in `src/everyday_world_v4.py`.

**Transitive closures of variable length** (the length test). Each has a one-step base relation that is fully trained,
and a closure relation trained only on pairs 1–2 steps apart:

| Closure | Base (one step) | Chain lengths | Trained | Tested |
|---|---|---|---|---|
| `kind-of` | `is-a` with a deeper taxonomy: animal → class → vertebrate/invertebrate → animal → living-thing | 1–4 | 1–2 | 3–4 |
| `ancestor-of` | `parent-of` in 3 new lineages of 6 generations (18 people) plus the two Hinton trees | 1–5 | 1–2 | 3–5 |
| `bigger-than` | `just-bigger-than` (adjacent in the size order) | 1–11 | 1–2 | 3–5 (and 6+ as a read beyond the step limit) |

**Compositions of different relations** (each composite has its own relation code; the loop must learn how to unfold it):

| Composite | Unfolds to | Steps |
|---|---|---|
| `named-can` (word, ability) | `names` then `can` | 2 |
| `named-has` (word, part) | `names` then `has` | 2 |
| `named-kind` (word, kind) | `names` then `kind-of` | 2–5 |
| `same-colour-as` | `colour-of` then its reverse | 2 |
| `grandparent-of` | `parent-of` twice | 2 |

Composite tests hold out a third of the noun words for all three `named-*` relations: these words have `names` trained
and nothing composite. Other held-out queries are as in v2–v3.

**Kept from v1–v3, to check that the codes stay arranged** (the thinking-loop pilot's lesson): P1 inheritance, P2
symmetry, P3 inverse, P5 opposite from meaning, P7 not inferable, G1–G4, and fast mapping (`dax`, `fep`, `blick-w`,
`zib-w`; code kept among the known codes, no exclusions given).

New fast mapping: `blick-w` given only `names(blick-w, owl)`. Read `named-can`, `named-has` and `named-kind` for it.

## Arms

| Arm | Reader |
|---|---|
| `one` | one-pass `tworel` (v3) |
| `one-flat` | one-pass `flat` (the strongest one-pass reader in v2–v3) |
| `one-wide` | one-pass `tworel` with its parameters matched to `loop-reader`'s total |
| `loop-reader` | controller + reader-scored fetch |
| `loop-dot` | controller + dot-product fetch |

Seeds 430–433, closed world (v3: few "no"s did not help), 20 runs.

**Training (all arms the same):**
- minibatches of 512 queries per step: relation uniform, label balanced, query uniform within (relation, label);
- Adam lr 3e-3, weight decay 1e-4, 6,000 steps;
- loops: T = 4 thinking steps in training. PonderNet loss: BCE on the expected answer, plus KL to a geometric halting
  prior (p = 0.5, weight 0.01).

At test, loops are read at T = 4 and at T = 6. The one-pass arms have no T.

## Reads

**Primary:**
- **Length:** closure accuracy by chain length; trained 1–2 against tested 3–5.
- **Composition:** held-out words on `named-*`; `same-colour-as`; `grandparent-of` (tree B, as v2).
- **New word:** `blick-w`'s composites.

**Is the loop doing what it seems to?**
- the halting step against true chain length;
- the top-attended entity at each step against the true middle entity (owl for blick-w; each generation on an
  `ancestor-of` path);
- the relation mix chosen at each step against the true relation;
- accuracy when the loop is forced to stop at t = 0, 1, 2, … (does it use its steps?).

**Did the codes stay arranged?** P1, P2, P3, G1–G4 and fast mapping, against the one-pass arms.

**Cost:** reader calls per query, parameters, and wall time per 1,000 queries at inference (median of 5 repeats after
warm-up). Quality is compared against compute, not only against examples.

## Predictions

1. **[local: v1–v3]** The one-pass arms fail length (3+ steps) and composition: near the v3 levels (closure at length 3+
   ≤ 0.65; `named-*` held-out words ≤ 0.65). `one-wide` is not better: capacity is not the problem.
2. **[Sukhbaatar et al. 2015; Yang et al. 2017; familiar]** Both loops learn the trained 2-step composites on held-out
   words (≥ 0.8). The fetched middle entity is the true one in most cases (top-1 ≥ 0.6).
3. **[Fan et al. 2024; Banino et al. 2021; familiar]** Length generalization is partial. Closures at length 3 are well
   above the one-pass arms (+0.15), but they fall with length. Reading at T = 6 helps length 4–5 only if halting
   follows chain length.
4. **[derived]** `loop-reader` beats `loop-dot` on composition and on `blick-w`, because fetching uses the judge's own
   knowledge. `loop-dot` is cheaper by about a factor of N in score cost.
5. **[local: thinking-loop pilot v1]** At least one loop arm loses on code arrangement: P1 inheritance or fast mapping
   below the one-pass arms by ≥ 0.05. This is the risk to watch, not a hoped-for result.
6. Halting tracks chain length on closures (Spearman ≥ 0.5) in the arm that generalizes length.

## Prior work

Searched 2026-11-06 (queries: "PonderNet learning to ponder"; "looped transformers length generalization"; "NBFNet";
"Neural LP differentiable learning of logical rules"), plus v1–v3 and the shared survey.

- **Multi-hop retrieval loops.** End-to-end memory networks (Sukhbaatar et al. 2015) **[familiar]** and the MAC cell
  (Hudson and Manning 2018) **[familiar]** run query → attention → update over several hops. The known weakness is that
  learning the hops from final answers alone is hard. Earlier memory networks used supervision of the supporting facts.
  Here the attention is over the network's own entity codes, not over stored facts or an image.
- **Soft chains of relations.** [Neural LP (Yang et al. 2017)](https://arxiv.org/abs/1702.08367) **[abstract]** uses a
  neural controller that composes differentiable relation operators to learn rule structure. Its operators are the KB's
  adjacency matrices: a built-in symbolic graph, which the program excludes. `loop-reader` replaces the adjacency
  matrices with the learned reader scored over learned codes.
- **Path-based link prediction.** [NBFNet (Zhu et al. 2021)](https://arxiv.org/abs/2106.06935) **[abstract]** scores a
  pair by aggregating paths with a generalized Bellman-Ford recursion over the fact graph. It is strong and inductive,
  but needs the graph at reading time (a memory of facts). That is excluded for now; it is the natural comparison once
  memory returns.
- **Adaptive depth and length generalization.** [PonderNet (Banino et al. 2021)](https://arxiv.org/abs/2107.05407)
  **[abstract]** learns the number of steps end-to-end and extrapolates where standard networks fail.
  [Looped Transformers (Fan et al. 2024)](https://arxiv.org/abs/2409.15647) **[abstract]** reuse one block with an
  adaptive number of loops, trained without intermediate steps, and generalize to longer inputs. ACT (Graves 2016) and the
  Universal Transformer (Dehghani et al. 2019) **[familiar]** are the earlier forms.
- **Attention forms.** Additive / learned-score attention (Bahdanau et al. 2015) and scaled dot-product attention
  (Vaswani et al. 2017) **[familiar]**. ATLAS (2025) is a test-time memory, not an attention form, and is not used.
- **Local warning.** [Thinking-loop pilot v1](think-pilot-v1-results.md) **[read: local]**: a recurrent answerer improved
  taught answers and lost placement of never-paired items. A more flexible reader arranges the codes less. Prediction 5
  and the arrangement reads exist because of it.
- **What this adds:** a weight-tied loop that fetches through the program's own relational reader, without a fact graph
  or symbolic operators, tested on chains longer than trained and on new words. The parts are established; the
  combination and the test on this core are the question.

## Scale

- **Inference cost per query:**
  - `loop-reader`: (T + 1) judge calls plus T × N fetch scores, where each fetch score is a reader call, so O(T · N ·
    reader). At N = 250 this is fine. At large N, fetch needs a cheap first stage (cosine top-k on codes, as in the
    memory line) before reader scoring.
  - `loop-dot`: O(T · N · d_k) for fetch plus (T + 1) judge calls. Keys can be cached once per entity.
  - One-pass: 1 judge call.
- A new entity adds one code and, in `loop-dot`, one cached key.
- Steps grow with chain length, not with the number of relations or entities.

## Computation (declared)

- **`loop-reader` training:** 512 queries × T = 4 × N ≈ 250 fetch scores ≈ 0.5 M reader evaluations per step, at about
  28k MAC each: about 15 GMAC forward, about 45 GFLOP per step with the backward pass. 6,000 steps ≈ 270 TFLOP per run
  (estimate). On one CPU thread that is several hours per run.
  - So `loop-reader` runs on the GTX 1650: 4 runs, one at a time.
  - Or with 4 threads on the CPU. The smoke run measures both and picks the faster.
- **`loop-dot` and the one-pass arms:** minutes per run on CPU.
- **Expected total:** about 6–10 h wall, dominated by `loop-reader`. To be measured in a smoke run and recorded before
  the main run. If the smoke run projects more than 12 h, the training steps for all arms are lowered together, and the
  change is recorded.
- **Stored:** reports; checkpoints in `runs/` (not tracked).
- **No tuning:** T = 4, prior p = 0.5, its weight 0.01 and d_k = 32 are fixed here.

## Open decisions for the user

1. **Compute.** `loop-reader` is expensive (each fetch runs the reader over every entity). Is several hours on the GPU
   acceptable? Or should its fetch use a cosine top-k first stage (say k = 16) from the start, which changes the arm
   slightly?
2. **Composites as relations.** Each composite (`named-can`, …) gets its own relation code, and the loop learns how to
   unfold it. A stricter later test would ask a composite never trained. Is per-composite training acceptable for v1?
