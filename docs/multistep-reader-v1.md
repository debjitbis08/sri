# Multi-step reader v1: a generative loop on the taught-truth reader

Status: **draft contract, 2026-11-06 (second draft). Not frozen; awaiting user review.** Stage 2 of 2 after
[taught truth v1](taught-truth-v1-results.md). The first draft (commit d77e911), with fetch by attention over entities,
was replaced on the user's direction. It is kept in git history.

User direction (2026-11-06):
- Multi-step inference must come from a change in the readers' architecture, not a 2-step patch.
- The loop is generative (`loop-gen`): each step produces the middle thing, with no search over candidates. It must be
  light enough for future online learning.
- One reader judges and predicts (g and P combined).
- Truth is taught, not built in.
- Chained questions are their own relations (higher-order relations).
- No memory system; working memory is allowed.

## Why

In v1–v3 and taught truth v1, every remaining failure chains relations through a middle entity:
- category from meaning (word → animal → noun);
- sense of a phrase (word → animal → ability);
- same colour; grandparent;
- and, in taught truth, the kind → individual link (this reptile → reptile → "sometimes swims" → unknown).

A one-pass reader cannot produce the middle entity and read again.

Taught truth v1 also left a question this study must answer. Taught truth cost some inference (order −0.23,
grandparent −0.31) against the built-in truth unit. So the same loop is also built on the built-in unit, and the two are
compared.

## The reader and the loop

**The reader** is taught truth v1's `pred`. It fills the missing slot of a statement (a, r, b, v):
`out = P([G([c_a, c_b or MASK, r]), r, slot])`, read against candidate codes by cosine.
- Value slot → the 7 learned value codes;
- tail slot → the entity codes.

**The loop** (weight-tied over thinking steps; working memory is one slot holding a code, the "current thing"):

```
question (a, r, b, slot) ──► s0 = Init([c_a, r, c_b or MASK, slot]);   m0 = c_a
each step t = 1..T:
   ρ_t   = softmax(W_ρ s_{t-1}) · R                       choose a relation to follow (a learned mix of relation codes)
   m_t   = rescale(reader.out(m_{t-1}, ρ_t, MASK, tail))  PRODUCE the middle thing (the reader filling a tail)
   s_t   = GRU(s_{t-1}, [m_t, ρ_t])
   λ_t   = σ(w_h · s_t)                                   halt?
answer at every t = 0..T:
   ρ'_t  = softmax(W_ρ' s_t) · R                          the relation to answer with
   y_t   = reader.out(m_t, ρ'_t, c_b or MASK, slot)        read against values (or entities, for a tail question)
output: PonderNet mixture  p(answer) = Σ_t p(halt at t) · softmax(y_t)
```

- **Nothing about the chain is given.** The loop is not told which relations to follow, which middle things to produce
  or how many steps to take. It learns from the final answers only.
- **One reader does everything.** It produces the middle things and gives the answer. The loop adds only the controller
  (GRU state 64, relation choices, halting).
- **t = 0 is the one-pass reader.** One-step questions are answered without a step if the loop halts at once.
- **`rescale`** sets the produced code's norm to the mean entity-code norm. The reader is trained by cosine, so a
  produced code has a direction but no meaningful norm. This is a fixed choice, recorded as such.
- **Cost per question:** about (T + 1) × 2 reader calls plus the GRU. No search over entities, so the cost does not grow
  with the number of entities.

## Arms

| Arm | Judge | Produce the middle thing | Loop |
|---|---|---|---|
| `pred-one` | taught truth (value codes) | — | none (taught truth v1's `pred`) |
| `pred-loop` | taught truth | the same reader (tail slot) | yes |
| `unit-one` | built-in truth unit (`tworel`) | — | none |
| `unit-loop` | built-in truth unit | a **separate** generator P′ (`[c_a, r] → code`, trained on the same fill-the-tail statements) | yes |

`unit-loop` is the comparison promised in the taught-truth discussion: the same loop with truth built in. It needs a
separate producer, because the unit only judges. That is the old g + P pair.

Seeds 450–453, 16 runs. Full batch; the statement groups are weighted as in taught truth v1. Adam lr 3e-3, weight decay
1e-4, 3,000 steps. T = 4 in training; read at T = 4 and T = 6. PonderNet: KL to a geometric halting prior, p = 0.5,
weight 0.01. `unit-loop` uses BCE on the mixed probability; it is read only on T/F tests.

## The world

Taught truth v1's statements (`src/everyday_truth.py`), plus chains, in `src/everyday_truth_v2.py`:

**L0: the one-fetch test from taught truth v1** (first, because it needs exactly one step): the held-out anonymous
individuals (S3b) and the dax. The step is individual → kind, then the kind's value → default.

**L1: chains of variable length** (each with a one-step base, fully trained, and a closure trained only on 1–2 steps):

| Closure | Base (one step) | Lengths | Trained | Tested |
|---|---|---|---|---|
| `kind-of` | `is-a` (animal → class) and a new `subkind-of` (class → vertebrate/invertebrate → animal → living-thing) | 1–4 | 1–2 | 3–4 |
| `ancestor-of` | `parent-of`, with 3 new lineages of 6 generations (18 people) | 1–5 | 1–2 | 3–5 |
| `bigger-than` (size only) | a new `just-bigger-than` (adjacent sizes) | 1–11 | 1–2 | 3–5 (6+ read beyond the step limit) |

For size, `smaller-than` at distance ≥ 3 is held out too, so the inverse does not give the answers. The other five scales
stay as in v3.

**L2: higher-order relations (compositions with their own relation codes):**

| Relation | Unfolds to | Held out |
|---|---|---|
| `named-can` (word, ability) | `names`, then `can` | a third of the noun words, for all three `named-*` (their `names` trained) |
| `named-has` (word, part) | `names`, then `has` | as above |
| `named-kind` (word, kind) | `names`, then `kind-of` (2–5 steps) | as above |
| `same-colour-as` | `colour-of`, then its reverse | v3's P6 |
| `grandparent-of` | `parent-of` twice | v3's P6 (tree B) |

New word: `blick-w`, given only `names(blick-w, owl)`; read `named-can`, `named-has`, `named-kind`.

**Kept, to check that the codes stay arranged:** taught truth's S1 (P1–P5, G1–G4), S2, S4 and the connectives.

## Reads

**Primary:**
- **L0:** S3b and the dax by kind value. Does "sometimes → unknown" appear?
- **L1:** accuracy by chain length; trained 1–2 against tested 3–5.
- **L2:** held-out words, P6 tests and `blick-w`.
- **Taught against built-in truth** on T/F chains: `pred-loop` against `unit-loop`.

**Is the loop doing what it seems to?**
- the produced middle code's nearest entity against the true middle thing (reptile; owl; each ancestor);
- the chosen relation mix against the true relation;
- the halting step against the true chain length;
- accuracy when stopped at t = 0, 1, 2, …

**Arrangement:** S1, S2 and S4 against the one-pass arms.

**Cost:** reader calls per question, parameters, and wall time per 1,000 questions at inference (median of 5 repeats
after warm-up).

## Predictions

1. **[taught truth v1; derived]** L0 is solved by the loop: `pred-loop` "sometimes → unknown" on new individuals ≥ 0.5
   (against 0.00–0.03 one-pass), and S3b ≥ 0.7. The produced middle code is nearest the right kind ≥ 0.6.
2. **[Sukhbaatar et al. 2015; Yang et al. 2017; familiar]** L2 held-out words: `pred-loop` ≥ 0.75 against one-pass
   ≤ 0.65.
3. **[Fan et al. 2024; Banino et al. 2021]** L1 length generalization is partial: length 3 above one-pass by ≥ 0.15,
   falling with length; T = 6 helps only if halting follows length.
4. **[the taught-truth question]** On T/F chains, `pred-loop` is within 0.05 of `unit-loop`. If `unit-loop` is ahead by
   more than 0.10, taught truth's cost persists in reasoning, and the direction needs rethinking.
5. **[local: thinking-loop pilot v1]** Risk: the loop arms lose arrangement (S1 inheritance or the dax) by ≥ 0.05
   against one-pass.
6. Halting follows chain length (Spearman ≥ 0.5) where length generalizes.

## Prior work

See the [first draft's survey](multistep-reader-v1.md) in git history (commit d77e911) and taught truth v1. Added for
the generative loop:
- **Latent reasoning loops:** feeding a model's own hidden state back as the next step's input, instead of writing out
  intermediate tokens (continuous chain-of-thought, Hao et al. 2024, "Coconut") **[familiar]**. `pred-loop` does the same
  with one code of working memory and a weight-tied step.
- **Producing instead of retrieving:** a produced code need not be any real entity. That is the v1 dax problem, a code
  outside the region the reader knows. The nearest-entity read measures it. The program's number-construction studies
  imagined results the same way ([multi-step](number-construction-v1-multistep-results.md), **[read: local]**).
- **Separate judge and producer (`unit-loop`):** the program's earlier g + P pair. The relational-expression design
  ([note](relational-expression-core-design.md)) records that they disagree. `pred-loop` combines them as the user
  directed.
- **Adaptive depth and length generalization:** PonderNet, Looped Transformers (first draft).
- **What this adds:** a weight-tied generative loop over the program's taught-truth reader, tested on the one-fetch
  default link, longer chains and higher-order relations, against the same loop with built-in truth.

## Scale

- Each step costs one reader call to produce and one to answer, with no search. Inference is O(T × reader) per
  question, independent of the number of entities.
- An online update is the same: one backward pass through T steps.
- A new higher-order relation costs one relation code.

## Computation (declared)

- Taught truth v1 took 9.5 CPU min per run. The loop arms run the reader about 10 times per statement (T = 4, produce
  and answer): an estimated 45–60 CPU min per run.
- 8 loop runs plus 8 one-pass runs ≈ 8–10 CPU h, about 1–1.5 h wall on 8 workers.
- If the smoke run projects more than 2 h wall, training steps are lowered together for all arms and recorded.
- Stored: reports; checkpoints in `runs/` (not tracked). No tuning: T, the halting prior and the rescale are fixed here.

## Open decisions for the user

1. **The rescale of a produced code** (to the mean entity norm): a small fixed choice, stated above. Acceptable, or
   should the loop learn its own scale?
2. **`unit-loop` with a separate producer P′:** this is the fair "built-in truth" comparison, but it is two networks
   against one. Keep it as the comparison?
