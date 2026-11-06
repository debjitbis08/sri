# Type meanings v1: categories and a "meaningless" output

**Contract, written 2026-11-06 before the run.** User direction, the same day: the reader confirms nonsense such as
`cow parent-of moo` because it does not know what `moo` is. Meanings are is-a relations ("moo is-a sound"). Invalid
type combinations are not false, they are meaningless, and no existing value says that. Decisions taken on the user's
word: meaninglessness is a separate **applicability** output of the logic route, taught by example, with intuition
never trained on meaningless examples; whether it is infectious through connectives is measured, not taught.
Follows [composition critic check v1](composition-critic-check-v1.md), which found that 97% of the confirmed graph is
untaught facts, all of them outside their relation's types. Runner `experiments/type_meanings.py`; reports
`reports/type-meanings-v1{,-summary,-runtime}.json`.

## Question

1. Does teaching what things are (category is-a facts) stop the reader confirming type-violating facts, by itself?
2. Can a learned applicability output, taught by example, recognize meaningless facts for a category and for relations
   it was never given a meaningless example of?
3. Does knowing the categories make that carry over?

## Categories (the user's list, approved 2026-11-06)

New entities, each with an is-a fact for every member, true for its own category and false for the others (and false
for `animal`):

| Category | Members |
|---|---|
| `person` | the 42 people of both family trees and the lineages |
| `sound` | the 16 animal sounds |
| `colour` | the 11 colours |
| `body-part` | fur, feathers, scales, wings, fins, beak, tail, legs0–legs6 |
| `property` | the adjectives (big, small, fast, slow, heavy, light, loud, quiet, hot, cold, bright, dark) and warm, cool, male, female |
| `ability` | fly, swim, run, climb, sing |
| `food` | meat, plants, seeds, bugs, nectar |
| `habitat` | water, land, trees, farm |
| `substance` | water, fire, soup, tea, ice, milk |
| `object` | the remaining things: fruits, sun, snow, coal, sky, grass, chocolate, feather, thunder, night-sky, … |

Young animals (puppy, kitten, …) get is-a `animal`. `water` is both a habitat and a substance. Animals keep their
existing is-a facts. Kinds (mammal … living-thing), words (already is-a noun, verb or adjective), truth values,
quantity words and unnamed individuals get no category.

## Meaningless facts

A relation's **types** are the categories of the subjects and objects of all its taught facts, true or false.
Animals count as `animal`, words as `word`, kinds as `kind`. A triple (x, r, y) is **meaningless** if no category of x
is among r's subject types, or no category of y among its object types. In this world the taught facts cover each
relation's whole domain × range, so every untaught triple is meaningless and every taught one is meaningful.

- **Held-out category:** `sound`. No meaningless training example contains a sound.
- **Held-out relations (6 of the 33 permitted):** `lives-with`, `brighter-than`, `slower-than`, `eats`,
  `sibling-of`, `mixed-from`. No meaningless training examples for these relations.
- **Training sample:** 100 meaningless triples per remaining relation (27 × 100), drawn once per seed, excluding sounds.

## Model and training

Phase 1 of conflict search v1 unchanged (separate-code dual route, R_L of withheld relations frozen, the no-support
fallback, 3,000 steps), plus:

- **Applicability head:** a(x, r, y) = σ(A_L([G_L(x, y, r), r])), on logic's own features beside the support head
  S_L (one MLP, 64 hidden).
- **Applicability targets:** 1 on every taught individual fact of the training split (true or false, withheld
  relations included); 0 on the meaningless sample. The two groups weigh equally, and the term weighs the same as
  logic's support term. Meaningless examples give no support target and are never seen by intuition.
- **Answer read:** a < 0.5 → meaningless (M), whatever the support; otherwise logic's T, F, U or D as before. The hop
  check becomes "both routes confirm, and a ≥ 0.5".

## Arms (4 seeds each, 450–453, splits as v1)

| Arm | Category is-a facts | Meaningless examples |
|---|---|---|
| `base` | — | — (a trained on targets 1 only) |
| `isa` | yes | — |
| `meaningless` | — | yes |
| `isa+meaningless` | yes | yes |

## Reads

1. **Applicability:** the share recognized as meaningless (a < 0.5) on three sets, each up to 3,000 sampled triples:
   **M-in** (training relations, no sound, not in the training sample), **M-cat** (contains a sound, training
   relations), **M-rel** (held-out relations, no sound). The share kept as meaningful (a ≥ 0.5) on all taught facts,
   train and test.
2. **Leakage:** the share of each M set that both routes confirm, without and with the applicability gate; and the
   full count of confirmed meaningless triples under permitted relations, with and without the gate (the measure of
   the composition check: 40,210 in seed 450).
3. **Truth kept:** intuition and logic balanced accuracy on v1's T/F tests (P1 … L2-named), as change from `base`.
4. **Categories learned:** logic's support on is-a facts for sounds (taught in the `isa` arms).
5. **Infection:** not testable in this version. The connective operator reads support pairs only and has no
   applicability input, so "not M" has no way to come out as M. This is recorded as a limit.
6. If the gate cuts leakage by at least 10×, **the composition critic check is rerun** (pairwise variant) on the
   gated graph of `isa+meaningless`, as a separate run.

## Predictions

1. **[read: composition check]** `isa` alone leaves type-violating confirmations within a factor of 2 of `base`.
   Nothing makes the support head use the category. Its applicability is trivially high.
2. **[outlier exposure; derived]** Both meaningless arms recognize M-in at ≥ 0.9 and keep ≥ 0.95 of taught facts
   meaningful.
3. **[outlier exposure; derived]** M-rel, held-out relations: partial transfer, 0.5–0.9. The head has to judge "unlike
   this relation's taught arguments" for relations whose arguments it saw only as positives.
4. **[derived; the user's hypothesis]** M-cat, sounds: `isa+meaningless` beats `meaningless` by ≥ 0.15. Without
   categories, a sound's code is shaped only by `says`, so it is not yet "a sound". This could fail in two ways: both
   arms high (sounds are already distinct through `says`) or both low.
5. **[derived]** Truth on v1's tests stays within 0.02 of `base` in every arm.
6. **[derived]** In `isa+meaningless`, the gate cuts the full count of confirmed meaningless triples by at least 10×.

## Prior work

Search 2026-11-06, from memory **[familiar]**; to be checked against sources before the results are written up.

- **Meaningless as a truth value:** Bochvar's internal three-valued logic (1938) has a "meaningless" value that
  infects every compound. Kleene's unknown does not. Ryle's category mistakes (1949) and Strawson's presupposition
  failure (1950): a sentence whose presupposition fails ("the king of France is bald") is neither true nor false.
  Implication: M is distinct from F (negation does not flip it) and from U (no evidence resolves it). This design
  makes it a separate output, not a third support value.
- **Type constraints in knowledge-graph embedding:** domain and range constraints (Krompaß, Baier and Tresp 2015)
  improve link prediction mainly by restricting negative sampling. Type-embodied and ontology-joint embeddings (TKRL,
  Xie et al. 2016; JOIE, Hao et al. 2019) learn types beside instances. These treat violations as false or simply
  exclude them; none has a meaningless output. Selectional preferences (Resnik 1996; distributional, Erk 2007) model
  which arguments fit a predicate as graded plausibility.
- **Out-of-distribution detection:** discriminative networks are confidently wrong off their training distribution
  (Nguyen et al. 2015; Hendrycks and Gimpel 2017), so two routes trained on the same data agreeing is not evidence.
  **Outlier exposure** (Hendrycks et al. 2019) trains a detector on auxiliary out-of-distribution examples. It
  generalizes to unseen out-of-distribution sets, but less to ones unlike the auxiliary set. This is the closest
  established method to the meaningless arms, and it predicts partial transfer to held-out relations and categories
  (predictions 2–4). Rejection options (Chow 1970) are the classical form of abstaining.
- **Cheaper alternatives:** a fixed domain/range table per relation would detect every violation exactly. It is
  excluded because it is a built-in rule ([purely neural system](prior-work.md)); it is the upper bound that the
  M sets are scored against. A density estimate on codes (familiarity) is an unsupervised alternative not tested here.
- **What this adds:** a learned applicability judgement that is separate from truth, inside a two-route reasoner, with
  transfer measured to a held-out category and held-out relations. It also tests the user's hypothesis that category
  meaning is what makes the transfer.

## Scale

The applicability head costs one MLP evaluation per fact checked, beside the support head. Meaningless examples grow
with the number of relations × categories sampled, not with entities². Held-out categories and relations test whether
it scales to unseen types without new examples.

## Computation (declared)

Phase 1: 3,000 steps as v1, about 10 CPU min per run at one thread (v1 measured 9.7); the extra head and about 2.6 k
meaningless rows per step add about 10%. 16 runs, about 3 CPU h; 6 workers, about 45–60 min wall (parallel runs were
about 2× slower per step in the composition check). Reads: the full gated graph per run, about 3 M reader evaluations,
about 15 CPU s. Storage: one checkpoint per run (about 0.4 MB), not tracked. No tuning: every constant is fixed here.
Resumable per run with `experiments/durable.py`.

## Commands

    EV_WORKERS=6 .venv/bin/python -m experiments.type_meanings          # rerun to resume
    .venv/bin/python -m experiments.type_meanings summary
