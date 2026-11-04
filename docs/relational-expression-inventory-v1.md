# Relational expression interface: inventory and finite evaluator (stage 1)

Status: **stage 1 of the [design note](relational-expression-core-design.md), 2026-11-04. No training, no perceptual read, no learning claim.** Stage 2 (read-only checkpoint baseline) is described at the end and has not started.

Code:
- `src/relational_expression.py`: the finite expression syntax and evaluators;
- `tests/test_relational_expression.py`: the truth-table fixture, 17 checks.

## Checkpoints for stage 2

The step 8 `sparse48` core with the `joined-mlp-calibrated` image encoder (`runs/sense-core-v1/joined-mlp-calibrated/checkpoints/`). Each file holds `image`, `sequence`, `sri`, `arm`, `seed` and `withheld`.

| Seed | `delay` (before relearning; update 0 of every relearning diagnostic) | `relearn` (after the unchanged 200-update relearning) |
|---|---|---|
| 300 | `sparse48-seed300-delay.pt`, sha256 `e2bf9fe56611085f…` | `sparse48-seed300-relearn.pt`, `8b4b8f1a4aec5700…` |
| 301 | `sparse48-seed301-delay.pt`, `47e0bde62a2c450a…` | `sparse48-seed301-relearn.pt`, `8889d87995d4c198…` |
| 302 | `sparse48-seed302-delay.pt`, `1588cc2478e2386e…` | `sparse48-seed302-relearn.pt`, `0cda4055d42c7868…` |
| 303 | `sparse48-seed303-delay.pt`, `0a45e47b798cb2f7…` | `sparse48-seed303-relearn.pt`, `0d2101a7e92570bd…` |

The relearning diagnostics' captured state reproduces `delay` exactly (`delay_state_max_abs_difference_from_step8` is 0 in every report). Intermediate updates (10, 20) are not stored. Reading them would need the capture (about 3 minutes per seed) and the unchanged arm.

## What each loss trains

**[read: local code]** One relearning update, `relearning_mechanism.forward`, equal to `relearning_dynamics` and `sequence_pilot` apart from the optional stops. The batch has 16 episodes: 12 new-script (English) episodes and 4 rehearsed episodes of earlier scripts, plus 4 context episodes (taps with collections).

| Loss | Scorer and argument order | Query kinds | Candidates or pairs | Positives / negatives | Form participation |
|---|---|---|---|---|---|
| Matching | `g(query, collection; same)`, 9-way cross-entropy | forms (form episodes), taps (context episodes) | 9 collections: the batch's paired collection in the correct count's slot, and one train-split collection for each other count, the same 8 for every row of the update | one positive per row; the other 8 counts are negatives only through the softmax | forms as left argument |
| Succession | `g(a, b; next)`, binary cross-entropy | within a kind: form → form, collection → collection, taps → taps | consecutive episode steps, always oriented (n, n+1) (descending episodes reversed) | one negative per positive: a same-kind pair whose second count is not the first's successor; it can be reversed, equal-count or the same item | form → form positives and form negatives |
| Comparison | `g(a, b; more)`, binary cross-entropy | collections and taps of the batch, mixed | 64 random pairs with unequal counts | label count(a) > count(b) | none |
| Sequence prediction | `P(prev, cur)`, logits −exp(scale)·‖P − cand‖², 9-way cross-entropy | (prev, cur) of one kind: forms or collections in form episodes; taps or collections in context episodes | the matching update's 9 collections, with the target step's collection in the correct slot | next step's collection; direction (up or down) is not supplied | form rows, see below |

**Form rows** in the relearning diagnostics (`form.append(e < n_form)`) are every sequence row from a form episode. That includes those episodes' collection → collection rows and the 4 rehearsed earlier-script episodes, not only rows whose inputs are forms. The earlier documents do not state this. Their results are unchanged, but "form rows" should be read with this definition. The naming-only contract has to choose its removal at the level of the row's kind, not the episode.

## What each existing read asks

| Read (function) | Scorer, argument order | Candidates | As an expression | Trained support |
|---|---|---|---|---|
| Script matching (`limited_memory.matching`) | `g(form, collection; same)` | 9 evaluation collections | `same_quantity(w, ?x)`, collections | Trained order and roles |
| Taps → collections (`collections_taps_diagnosis.read`, `number_construction_part_d.evaluate_senses`) | `g(taps, collection; same)` | 9 evaluation collections | `same_quantity(t, ?x)`, collections | Trained order and roles |
| Collections → taps, evaluated (`collections_taps_diagnosis.read`) | `g(collection, taps; same)` | 9 evaluation taps | `same_quantity(d, ?x)`, taps | Untrained: collection never left, taps never right |
| Collections → taps, swapped order (same) | `g(taps, collection; same)`, ranked over taps | 9 evaluation taps | `same_quantity(?x, d)`, taps | Trained argument order; ranking over the left argument untrained |
| Taps ↔ forms (`evaluate_senses` cross-sense) | `g(taps, form)`, `g(form, taps)` | 9 forms or 9 taps | `same_quantity(t, ?x)` forms; `same_quantity(w, ?x)` taps | Untrained pairings |
| Memory symbol → dots, r2 (`sense_memory_core.score`) | `g(value, collection; same)`, value a retrieved dot trace | 9 collections | retrieve v; `same_quantity(v, ?x)`, collections | Untrained: collection → collection |
| Memory symbol → taps, r3 | `g(value, taps; same)` | 9 taps | retrieve v; `same_quantity(v, ?x)`, taps | As collections → taps, evaluated |
| Memory comparison, r4 | `g(v1, v2; more) > 0` | partner query's value | retrieve v1, v2; `more_than(v1, v2)` truth | Trained kinds (collections) |
| Memory taps → symbol, r3' | `g(taps, value; same)` over 9 retrieved values | one value per count, from retrieval | `same_quantity(t, ?v)` over retrieved values | Trained order |
| Comparison (`evaluate_senses`) | `g(a, b; more) > 0` within a kind | all ordered pairs | `more_than(a, b)` truth | Collections, taps trained; forms untrained |
| Coherence AUROC (`generic_relations.generic_coherence`) | summed `g(c_k, c_k+1; next)` along 5-step runs | one step replaced | `successor` truth along a chain | Within-kind, trained |
| Sequence and composition (`sequence_pilot.composition`) | k rollouts of P; read by distance and by `g(·, collection; same)` | 9 evaluation collections at the query's rendering index | continuation, chain length k, direction inferred | Within the trained protocol for k = 1; for k ≥ 2 P's input is its own output, never trained |
| `same` diagnostics (`same_relation_diagnosis.measure`) | `g` in both orders | — | diagnostics | — |

Memory reads abstain when retrieval scores below τ, and an abstention counts as wrong. The bypass read substitutes the correct stored value and is a diagnostic only.

## Mismatches the stage 2 baseline must report

1. **Composition gets intermediate candidates.** P's rollout carries a predicted code and never chooses an intermediate. A `g` chain `Exists y: successor(a, y) AND successor(y, ?z)` enumerates y over a declared domain, which is extra task information. The domain's kinds are part of each read's label.
2. **Direction needs supplied algebra.** There is no `predecessor` token. Continuation `Exists r in {successor, predecessor}: r(prev, cur) AND r(cur, ?x)` uses `predecessor := inverse(successor)`. Direction is chosen by learned `next` scores, but the inverse is supplied.
3. **Cross-kind succession is untrained.** `next` saw only within-kind pairs. For taps or form starts with collection answers, the second atom `successor(cur, ?x)` is untrained in every such read. Stage 2 reads both: **within-kind** answers (taps candidates for taps starts, matching `g`'s support) and **cross-kind** collection answers (matching P's protocol), labelled separately and never pooled.
4. **`same` has no absolute threshold.** It was trained only by ranking. A truth judgment `same_quantity(a, b)` therefore has no trained decision point. Report discrimination (AUROC) or a declared threshold labelled uncalibrated. `next` and `more` have logit 0 as their trained boundary.
5. **Scores are not comparable across relations.** Relation identification `?r(a, b)` mixes a ranking-trained relation with binary ones. Report it per relation, without a pooled argmax, unless calibration data is declared.
6. **Memory r2 is already an untrained pairing** (collection → collection `same`). It stays as it is for historical comparison and is labelled.
7. **Multiple answers.** Every current candidate set has exactly one rendering per count. The multi-answer read needs a domain of several renderings per count, every same-count rendering labelled positive, scored by precision and recall at a declared cut.

## The finite evaluator

`src/relational_expression.py`:
- **Syntax:**
  - terms are opaque entity references, relation references and named variables;
  - atoms of any arity, conjunction, negation, existentials over declared domains;
  - a query names its requested variables, and any other free variable is an error.
- **Exact semantics:** three-valued.
  - A missing fact is unknown unless its relation is declared closed.
  - An existential is false only when every assignment is false and its domains are declared complete.
  - Empty and incomplete domains therefore give unknown, not false.
- **Soft semantics:** max-min. An atom is a score, And is min, Exists is max, Not is negation, and the argmax witness is recorded. This is declared here, before any perceptual read. It is one fuzzy semantics, not a probability.
- **Supplied algebra:** only `Inverse` (argument order reversed). Every result lists the definitions it used.
- **Costs:** atoms are cached per query and counted. One missing slot is N evaluations. A two-step chain from a fixed start is at most N + N² distinct atoms.

**The fixture checks** (oracle counts generate the table and are never visible to the evaluator):
- true and false statements, and negation;
- unknown for missing facts, and false for closed relations;
- several answers across renderings and kinds, with precision and recall;
- argument reversal through the same relation;
- a repeated variable keeps one binding;
- relation identification returning every holding relation (`successor` and the supplied `less_than`);
- two-step composition with all witnesses;
- bound-variable renaming and shadowing;
- inferred-direction continuation with the supplied `predecessor` as witness;
- out-of-range continuation (false on a complete domain, unknown otherwise);
- empty domains;
- max-min ranking by the weakest link with its witness;
- the inverse passing reversed arguments to the scorer.

**Supplied versus learned, for stage 2:**

| Supplied | Learned (read from the checkpoint) |
|---|---|
| Expression syntax, variable binding, conjunction, existential enumeration | `g`'s three relation scores on core codes |
| `predecessor := inverse(successor)`, `less_than := inverse(more_than)` | The senses, the image head and the core projections |
| Max-min aggregation; candidate and witness domains | Memory retrieval and its stored traces |
| Thresholds: logit 0 for `next` and `more`; none trained for `same` | — |

Nothing in the evaluator supplies symmetry, equality substitution or composition knowledge. A result that differs from the historical reader only by this wrapper is a wrapper baseline.

## Stage 2, proposed (not started)

At `delay` and `relearn` for seeds 300–303, with the historical readers recomputed on the same candidates:
- **completion through `g`:**
  - all matching reads in both slots;
  - `successor(a, ?x)` and `successor(?x, b)`, within-kind and cross-kind;
  - `more_than` truth;
- **continuation with inferred direction,** k = 1, 2, 3, through max-min over enumerated intermediates. Each intermediate domain is labelled within-kind or collections. These are compared with P's composition on the same queries and final candidates;
- **memory present,** with actual retrieval and abstention: r2, r3, r4 and r3' re-expressed, retrieval unchanged;
- **a multi-answer read** over several renderings per count.

**Cost:** about N = 9 candidates per slot, at most 2·(N + N²) atoms per two-step continuation query, and a few thousand queries per checkpoint. That is seconds on the CPU after the codes are computed. Measured wall, CPU, peak RSS and atom counts will be reported separately from the results.

## Computation

- Stage 1 tests: 17 checks in 5 ms, peak RSS 15.7 MB, CPU.
- No perceptual or neural evaluation.
- Selection and tuning: none.
