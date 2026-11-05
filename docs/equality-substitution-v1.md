# Equality as substitutability v1

Contract written 2026-11-05, before the run. User direction (2026-11-05), after the
[relation codes check](relation-codes-check-v1.md): objects sit as clouds in code space and relations in another space; what has
to be captured is equality as in logic. If a = b, then φ(a) ⇒ φ(b) for every well-formed formula φ, so every operation on
supposedly equal elements must give the same result. And a and b are equal only if they contain the same elements, wherever they
are. The user also asked for cross-language equality examples in training.

## What this changes in the test

Earlier reads judged equality by location (nearest collection-count mean) or by the one relation family the forms were trained on.
Substitutability asks for the same result under **every** operator, including operators never trained on the forms. Location
does not matter: two clouds may sit apart if every operator treats them alike.

Extensionality ("same elements") applies to collections directly: two collections have the same count when their elements
correspond one to one (Hume's principle). A glyph has no elements to see, so its equality can only be inherited, by
substituting for a collection or for another script's form.

## Questions

1. Do operators learned **only on collections**, after the forms were taught, give the same result on a script's forms as on
   same-count collections (substitution), and does that hold for an old script after a new one is taught?
2. Do cross-language equality pairs, trained by substitution and without count labels, improve that over replaying the same
   inputs with count labels?
3. Is equality transitive: after Chinese ≡ English and Devanagari ≡ English, does Chinese ≡ Devanagari hold untrained?

## Design

**System.** The Part C network ([contract](relation-codes-check-v1.md#part-c-contract-one-code-space-or-two-trained-pilot-written-2026-11-05-before-its-run)):
frozen 768-dim sense codes, object encoder E (768 → 128 → 16), relation codes for the 17 differences. Two bases: `count` (best
retention in Part C) and `two-ctx` (best relation arm).

**Curriculum, as Part C.** Block 1: collections and English. Block 2: Chinese. Block 3: Devanagari. Collections in every block.
1,500 Adam steps per block, learning rate 1e-3, seeds 300–303, withheld counts as Part C. Thai never taught.

**Operators learned on collections only.** In block 3, two binary operators are trained on collection pairs only: `more`
(b > a) and `sum10` (a + b = 10). One shared readout S([pair feature, operator code]) with a learned 8-dim code per operator
(a single readout for all relations). The pair feature is [c_a, c_b] for `count` arms and the relation-space code G([c_a, c_b])
for `two-ctx` arms. `more` depends only on order; `sum10` needs each argument's own count. No form ever gets an operator label.

**Buffers.** Stored frozen sense codes, not images, with the same contents for every arm of a seed: 4 English items per count
(stored after block 1) and 4 Chinese items per taught count (stored after block 2). Replayed 2 per count per step. Replaying
stored codes is rehearsal; every buffer arm has a control with the same inputs.

**Cross-language equality pairs.** Each buffer item is paired with every current-script form of the same count in the batch.
The substitution loss is the symmetric KL between the outputs of every operator present (the difference relation; in block 3
also `more` and `sum10`) on (x, m) and (y, m), and on (m, x) and (m, y), over all collections m in the batch. No count label
enters this loss; only "x equals y" does.

**Arms.**

| Arm | English buffer, blocks 2–3 | Chinese buffer, block 3 | Role |
|---|---|---|---|
| `count` | none | none | Part C baseline |
| `count-enbuf` | count labels | none | rehearsal control for `eq-en` |
| `eq-en` | equality pairs with the current script | none | (a): equality through English only; Chinese inputs never return |
| `two-ctx` | none | none | Part C baseline |
| `two-ctx-enbuf` | labelled relations | none | control for `two-ctx-eq-en` |
| `two-ctx-eq-en` | equality pairs | none | (a) on the two-space base |
| `count-enzhbuf` | count labels | count labels | control for `eq-en-zh` |
| `eq-en-zh` | equality pairs | Chinese ≡ Devanagari pairs | (b) |

Chinese ≡ Devanagari is trained only in `eq-en-zh`; elsewhere it tests transitivity through English and collections.

A label-free consistency pressure (path consistency of relations) was considered and left out: it acts only on items present
in a batch, and every item present already carries labels, so it would add nothing here.

## Reads (after each block; operators after block 3)

- **Substitution agreement** (primary). For a kind k and its paired reference (same count, same item index): the agreement of
  each operator's output on (x, m) with the output on (ref, m), and on (m, x) with (m, ref), over all 270 collection evaluation
  items m. Agreement is balanced over the reference's output (true and false weighted equally). Pairs: zh~coll, dev~coll,
  en~coll, thai~coll, zh~dev. Two bounds: a **null** where the reference has a different count (n + 3 mod 9), and a
  **ceiling** coll~coll (another collection item of the same count). Also the difference relation's argmax agreement. Split
  into all counts and counts taught to both kinds.
- **Operator accuracy**, balanced, on pairs of kinds: coll×coll, zh×coll, coll×zh, zh×zh, zh×dev, dev×coll, dev×dev, en×coll,
  en×en, thai×coll.
- **Identity** (diagnostic), nearest and relational, as Part C; script relations zh×dev, own read and via identity.

**Primary measure:** zh~coll substitution after block 3 for `more` and `sum10`, against its null and ceiling, in `eq-en` against
`count-enbuf` and `count`, and in `eq-en-zh` against `count-enzhbuf`.

## Predictions

1. **[functional regularization and Part C]** Without Chinese replay, equality through English does not protect Chinese:
   `eq-en` and `two-ctx-eq-en` are within 0.05 of their controls on Chinese identity and zh~coll substitution. Both Chinese
   buffer arms beat `count` on Chinese.
2. **[pivot-based zero-shot alignment]** Transitivity is partial: zh~dev substitution in `eq-en` is above `count` but below the
   coll~coll ceiling.
3. **[from part A: order kept, scale lost]** `sum10` transfers to forms worse than `more`. Operator accuracy on a script tracks
   its substitution agreement.
4. `eq-en-zh` ≈ `count-enzhbuf` on Chinese identity (same inputs), and `eq-en-zh` > `count-enzhbuf` on zh~dev substitution
   (trained directly).
5. thai~coll substitution is at its null in every arm.

## Prior work

Searched 2026-11-05 (queries: "zero-shot translation pivot transitivity multilingual alignment"; "learning without forgetting
functional regularization output distillation"), plus the shared survey in [prior work](prior-work.md).

- **Equality.** The substitution loss is Leibniz's law (indiscernibility of identicals) made a training signal; reading equality
  as "same relations to everything" is the converse (identity of indiscernibles) and, in category theory, the Yoneda lemma: an
  object is fixed up to isomorphism by its relations to all others. Count equality for collections is Hume's principle
  (equinumerosity). [established, theory]
- **Substitution as output matching.** Matching an operator's outputs across two inputs is functional regularization and
  distillation (Learning without Forgetting, Li and Hoiem 2017; Hinton et al. 2015), here across equal inputs rather than across
  model versions. Known failure: it constrains only the outputs and inputs present, so it protects nothing whose inputs are
  absent. That is why prediction 1 expects no protection for Chinese without Chinese replay. [established]
- **Cross-language equality and transitivity.** Multilingual translation trained through a pivot language translates unseen
  pairs zero-shot but trails pivoting (Johnson et al. 2017). Auxiliary losses that align encoder representations across
  languages close much of the gap (Arivazhagan et al. 2019, "The Missing Ingredient in Zero-Shot NMT"). Predicts partial
  transitivity. [established, other domain]
- **Rehearsal.** Replaying stored codes is latent replay (van de Ven et al. 2020); each buffer arm has a same-input control.
- **Cheaper alternative:** count labels on the same replayed inputs (the control arms). The study adds a test the earlier
  reads lacked: operators never trained on forms, read by substitution against a null and a ceiling.

## Scale

Substitution pairs grow with buffer items × same-count current items × collections per batch, linear in each. Operator codes
are one per operator; the readout is shared. Buffers grow with scripts × counts × stored items; at scale they would come from a
limited, decaying memory ([limited memory](limited-memory-drives-consolidation.md) direction), not a fixed store.

## Computation (declared)

Parameters: E 100,496; G 3,152 (two-space arms); operator readout 2,689 (count arms) or 1,665 (two-space arms), plus 16 for the
operator codes. Per step: 72 collections, 56–72 current forms and 18–32 buffer items encoded; up to ≈ 17k labelled pairs; about
100–150 equality pairs × 72 collections × 2 positions; in block 3, 5,184 collection pairs for operators. Evaluation: 72,900 pairs
per kind pair and substitution read. 32 runs (8 arms × 4 seeds) × 4,500 steps on CPU. Expected about the cost of Part C per
run, plus about half again for the operators and pairs. Stored: reports only; no weights are kept.

## Commands

```
.venv/bin/python -m unittest tests.test_equality_net
EQ_SMOKE=1 .venv/bin/python -m experiments.equality_substitution_pilot
.venv/bin/python -m experiments.equality_substitution_pilot            # EQ_WORKERS=8
.venv/bin/python -m experiments.equality_substitution_pilot summary
```

## Results (run 2026-11-05)

Means over seeds 300–303 (range in brackets where it matters), after block 3 unless stated. Report:
`reports/equality-substitution-v1.json`; `summary` prints every table.

**Substitution under operators learned on collections only** (balanced agreement, all counts; null = reference of a
different count; ceiling coll~coll is 0.97–0.99 for `more`, 0.90–0.96 for `sum10`):

| Arm | zh~coll more / sum10 | null | dev~coll more / sum10 | en~coll more / sum10 | zh~dev more / sum10 | null |
|---|---|---|---|---|---|---|
| `count` | 0.84 / 0.61 | 0.59 / 0.47 | 0.82 / 0.60 | 0.58 / 0.51 | 0.79 / 0.58 | 0.58 / 0.47 |
| `count-enbuf` | 0.87 / 0.64 | 0.62 / 0.46 | 0.87 / 0.65 | 0.84 / 0.61 | 0.82 / 0.62 | 0.65 / 0.46 |
| `eq-en` | 0.75 / 0.50 | 0.74 / 0.50 | 0.74 / 0.50 | 0.74 / 0.50 | (degenerate) | |
| `two-ctx` | 0.90 / 0.71 | 0.56 / 0.44 | 0.94 / 0.85 | 0.81 / 0.62 | 0.88 / 0.69 | 0.57 / 0.44 |
| `two-ctx-enbuf` | 0.91 / 0.71 | 0.57 / 0.44 | 0.94 / 0.85 | 0.93 / 0.78 | 0.88 / 0.68 | 0.58 / 0.45 |
| `two-ctx-eq-en` | 0.91 / **0.75** | 0.57 / 0.44 | 0.96 / 0.86 | 0.92 / 0.79 | 0.89 / **0.73** | 0.59 / 0.45 |
| `count-enzhbuf` | 0.89 / 0.70 | 0.62 / 0.44 | 0.85 / 0.63 | 0.84 / 0.62 | 0.83 / 0.61 | 0.64 / 0.46 |
| `eq-en-zh` | 0.74 / 0.50 | 0.74 / 0.50 | 0.74 / 0.50 | 0.74 / 0.50 | (degenerate) | |

**Identity (diagnostic, best of nearest and relational) and operator accuracy:**

| Arm | zh block 2 → 3 | dev | en | zh×coll more / sum10 accuracy |
|---|---|---|---|---|
| `count` | 0.80 → 0.74 (0.62–0.83) | 0.79 | 0.16 | 0.86 / 0.64 |
| `count-enbuf` | 0.80 → 0.74 | 0.80 | 0.74 | 0.89 / 0.65 |
| `eq-en` | 0.81 → 0.58 (0.41–0.70) | 0.78 | 0.34 | 0.75 / 0.50 |
| `two-ctx` | 0.86 → 0.59 | 0.85 | 0.38 | 0.88 / 0.70 |
| `two-ctx-enbuf` | 0.84 → 0.57 | 0.84 | 0.69 | 0.90 / 0.70 |
| `two-ctx-eq-en` | 0.86 → 0.61 | 0.85 | 0.69 | 0.90 / 0.76 |
| `count-enzhbuf` | 0.80 → 0.79 | 0.75 | 0.74 | 0.88 / 0.72 |
| `eq-en-zh` | 0.81 → 0.61 (0.47–0.80) | 0.78 | 0.35 | 0.74 / 0.50 |

Thai identity 0.13–0.18 everywhere; thai~coll substitution at its null (`more` 0.64–0.73 against 0.66–0.74).

### What the results say

1. **Substitution on operators that read object codes directly has a degenerate solution, and the network found it.** In
   `eq-en` and `eq-en-zh`, operator outputs no longer depend on a form at all: every script, including the current one and
   never-taught Thai, gives the same agreement (0.74 / 0.50), equal to the null; `sum10` accuracy is exactly 0.50 on every
   form pair, Devanagari × Devanagari included (`more` 0.51–0.57). Forms and collections are two clouds here, and the
   operator readout, trained only on collections, can satisfy "equal pairs give equal outputs" by making every output on the
   form cloud constant. That costs nothing on collections and meets the loss exactly. Leibniz's law holds trivially if
   operators cannot tell anything apart; equality as substitutability needs the converse as well, that **unequal** things give
   **different** outputs (identity of indiscernibles). The design had no such term. This is inferred from the reads (identical
   values across all scripts; no outputs on forms vary), not from inspecting weights, which are not kept.
2. **Where operators read a relation space that the forms are also trained in, equality pairs help a little and transitivity
   holds partly.** `two-ctx-eq-en` against its same-input control `two-ctx-enbuf`: zh~coll `sum10` 0.75 against 0.71,
   zh×coll `sum10` accuracy 0.76 against 0.70, zh~dev 0.89 / 0.73 against 0.88 / 0.68 (null 0.58 / 0.45, ceiling 0.99 /
   0.96). Chinese ≡ Devanagari was never trained there, so this is transitivity through English and collections, partial
   (prediction 2 holds for this base).
3. **Operators learned only on collections do reach forms, by order more than by count.** In every non-degenerate arm,
   `more` substitutes well for Chinese (0.84–0.91, null ≈ 0.6) and `sum10` much less (0.61–0.75, null ≈ 0.45); operator
   accuracy tracks substitution (prediction 3 holds). The two-space base carries both further than the count base (`sum10`
   0.71 against 0.61), although its Chinese identity is lower (0.59 against 0.74): the relation space keeps what operators
   need better than nearest-mean location does.
4. **No equality arm protects Chinese identity.** Without Chinese replay, `two-ctx-eq-en` ≈ its control (0.61 against 0.57);
   `eq-en` loses 0.16 against `count` because of the collapse. The label-replayed Chinese buffer gives the best Chinese
   (0.79), the equality-paired one does not (0.61). Prediction 1 holds for the two-space base, fails for `eq-en`; prediction 4
   fails (the zh~dev advantage of `eq-en-zh` is the degenerate one).
5. **English through equality pairs:** in the two-space base, equality pairs keep English as well as labels do (0.69 against
   0.69); in the count base they do not (0.34 against 0.74), again the collapse.

### Next step this points to

Train substitution together with its converse: equal pairs must agree under every operator, and unequal pairs (different
counts, same or different script) must disagree under some operator, with operators reading a relation space rather than raw
object codes. That is a contrastive form of equality (the standard fix for collapse in siamese and self-supervised learning,
e.g. contrastive negatives or variance terms as in VICReg). Not run; the user's decision.

### Computation (measured)

- 32 runs (8 arms × 4 seeds) × 4,500 steps on 8 CPU workers, 1 thread each: 6,172 s wall, 47,022 s aggregate worker CPU
  (≈ 1,470 s per run, about four times Part C's ≈ 370 s; the declared estimate of one and a half times was low, mostly from
  the substitution pairs × collections and the block-3 operator pairs), 1.9 GB peak RSS per worker. Data preparation 28 s.
  The two 20-step smoke runs cost about 2 minutes each.
- No weights are retained.
