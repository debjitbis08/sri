# Equality as a learned relation v2

Contract written 2026-11-05, before the run. Follows [equality as substitutability v1](equality-substitution-v1.md), where a
substitution loss on its own collapsed: operators went blind to every form, so equal pairs agreed trivially.

User direction (2026-11-05): equality's symmetry must be learned, not built in. And "be careful with other operators — all
operators have different properties."

## What changes from v1

- **Equality is an operator**, learned from examples like any other, with positives (same count) and negatives (different
  count). Negatives remove v1's degenerate solution: an operator blind to forms now fails the negatives.
- **No property is built into a loss.** v1's symmetric KL in both argument positions is gone. Substitution is no longer
  trained; it becomes a derived read.
- **Each operator has its own properties, read separately:**

  | Operator | Meaning | True properties |
  |---|---|---|
  | `eq` | a = b | symmetric, reflexive |
  | `more` | b > a | antisymmetric, irreflexive |
  | `sum10` | a + b = 10 | symmetric (neither reflexive nor irreflexive) |

  The difference relation (d = b − a, from Part C) stays and is antisymmetric.

  Learning "equality is symmetric" must not make `more` symmetric: that would be overgeneralization.
- **Order is what tests learning.** Collection pairs are trained in both orders for every operator. Cross-script pairs are
  trained in one order only, current script first: (Chinese, English) in block 2, (Devanagari, English) in block 3. The
  reversed order is never trained. A property is learned if the reversed order follows the operator's own property: `eq` the
  same, `more` flipped.

## Design

**System and curriculum as v1** (Part C network on frozen sense codes; collections and English, then Chinese, then Devanagari;
1,500 Adam steps per block; seeds 300–303; withheld counts as Part C; Thai never taught). Same stored buffers as v1: 4 items
per count of English (from block 1), 4 per taught count of Chinese (from block 2), replayed 2 per count per step.

**Operators.** One shared readout S([pair feature, operator code]); pair feature G([c_a, c_b]) on the `two-ctx` base,
[c_a, c_b] on the `count` base.
- `eq` and `more` are trained on all ordered collection pairs from block 1.
- `sum10` is added in block 3, on collections only (a new operator after the forms, as in v1).
- Losses are balanced binary cross-entropy.

**Arms.**

| Arm | Base | English buffer, blocks 2–3 | Chinese buffer, block 3 |
|---|---|---|---|
| `two-ctx-enbuf` | two-ctx | labelled relations to collections (control) | none |
| `rel-en` | two-ctx | `eq`, `more` pairs with the current script, current script first only | none |
| `rel-en-both` | two-ctx | as `rel-en`, both orders (upper bound for the reversed order) | none |
| `two-ctx-enzhbuf` | two-ctx | labelled relations (control) | labelled relations (control) |
| `rel-en-zh` | two-ctx | as `rel-en` | `eq`, `more` pairs with Devanagari, Devanagari first only |
| `count-enbuf` | count | count matching (control) | none |
| `count-rel-en` | count | as `rel-en` | none |

Chinese × Devanagari is trained only in `rel-en-zh` (one order); elsewhere it tests transitivity. No form is ever paired with
another form of its own script, so within-script equality (reflexivity across items) is untrained everywhere.

## Reads (after every block)

- **Balanced accuracy per operator and ordered kind pair:**
  - coll×coll
  - zh×en (trained direction) and en×zh (reversed)
  - dev×en and en×dev
  - zh×dev and dev×zh
  - zh×zh, dev×dev and en×en
  - zh×coll and coll×zh
  - dev×coll and coll×dev
  - en×coll
  - thai×coll
- **Properties per operator,** comparing out(a, b) with out(b, a) over all item pairs:
  - `eq` and `sum10`: symmetric agreement.
  - `more` on unequal counts: antisymmetry, exactly one order true.
  - `more` on equal counts: irreflexivity, both orders false.

  Pairs read: coll|coll, zh|en, dev|en, zh|dev, zh|zh, dev|dev, zh|coll, dev|coll, thai|coll. Agreement alone is satisfied
  by a constant output, so each property is read together with accuracy.
- **Identity by equality:** for each form, the count of the collection item with the highest `eq` logit. Nearest and
  relational identity as before (diagnostic).
- **Substitution (derived, never trained):** agreement of `more` and `sum10` on (x, m) and (m, x), over all collection items m,
  with the same outputs for a reference. The reference is the same-count collection item (true), a different-count item
  (null), or the collection item the system itself judges equal (judged).

**Primary measures (block 3):**
1. Reversed-order accuracy: en×zh and en×dev for `eq` and `more`, against the trained direction and `rel-en-both`.
2. Property agreement against the truth for each operator. Is `more` kept antisymmetric while `eq` is symmetric?
3. Chinese: the collapse is gone in `count-rel-en` (form accuracy above 0.5), and Chinese identity and substitution against the
   controls.

## Predictions

1. **[contrastive learning; v1]** Negatives remove the collapse: `count-rel-en` forms get `eq` and `sum10` accuracy well above
   0.5, unlike v1's `eq-en`.
2. **[relation networks generalize to an untrained argument order only through aligned inputs]** Reversed-order accuracy is
   above chance but below the trained direction. `rel-en-both` closes most of the gap. The reversed-order gap is smaller for
   the current script than for Chinese (Chinese codes have drifted from collections).
3. **[the readout keeps operators apart by code]** Properties follow each operator: `more` stays antisymmetric on cross-script
   pairs (> 0.8 where its accuracy is > 0.8), and is not symmetrized by learning `eq`.
4. **[v1, pivot alignment]** zh×dev `eq` (untrained) is above chance and below trained pairs; `rel-en-zh` raises it.
5. **[v1; functional regularization does not reach absent inputs]** Without Chinese replay, Chinese identity is within 0.05 of
   `two-ctx-enbuf`. With Chinese replay, `rel-en-zh` ≈ `two-ctx-enzhbuf`.
6. `eq` identity tracks nearest identity. Judged substitution is close to true substitution where `eq` identity is high.
7. Thai at chance.

## Prior work

Searched 2026-11-05 (query: "RotatE knowledge graph embedding symmetry antisymmetry inversion composition"), plus v1's search
and the shared survey in [prior work](prior-work.md).

- **Relation properties.** Knowledge-graph embeddings build relation properties into their geometry. TransE cannot represent
  symmetric relations; DistMult represents only symmetric ones; RotatE (Sun et al. 2019) represents symmetry, antisymmetry,
  inversion and composition through a phase per relation. Here no property is built in: one readout, operator codes, and the
  property must come from examples. This is the program's universal-relational-component constraint. [established]
- **Equality with negatives.** Learning a same/different judgement with negatives is metric learning and contrastive learning
  (siamese networks; InfoNCE; VICReg's variance term against collapse). The v1 collapse is the textbook failure of positive-only
  alignment. [established]
- **Generalizing to an untrained argument order.** Networks generalize poorly to argument combinations absent from training
  unless the inputs are aligned with trained ones (systematicity: Lake and Baroni 2018). That is why prediction 2 ties the
  reversed-order gap to alignment. [established, related domain]
- **Same/different in relational networks.** Same/different is a classic hard case for feedforward networks out of
  distribution (Kim, Ricci and Serre 2018); relational-bottleneck models (ESBN, CoRelNet) do better by comparing codes. The
  `two-ctx` pair space is in this family. [established]
- **Cheaper alternative and controls:** the label-replay arms, with the same inputs. What is added: each operator's property
  is read on untrained orders, so one operator's property cannot be credited from another's.

## Scale

Cross-script pairs per step: current forms × buffer items (≈ 1,000 ordered pairs) × 2 operators. Collection pairs: 5,184 × 3
operators. Operators are one code each; the readout is shared, so a new operator costs 8 parameters and its examples.

## Computation (declared)

Parameters: E 100,496; G 3,152 (two-space arms); readout 1,665 (two-space) or 2,689 (count), plus 24 for the operator codes.
28 runs (7 arms × 4 seeds) × 4,500 steps on 8 CPU workers. v1 cost ≈ 1,470 CPU s per run; v2 has no substitution pairs, but it
has operator training in every block and operator reads after every block. Expected ≈ 1,000–1,500 CPU s per run, about
1–1.5 h wall. Stored: reports only; no weights.

## Commands

```
.venv/bin/python -m unittest tests.test_equality_net_v2
EQ_SMOKE=1 .venv/bin/python -m experiments.equality_relations_v2
.venv/bin/python -m experiments.equality_relations_v2            # EQ_WORKERS=8
.venv/bin/python -m experiments.equality_relations_v2 summary
```
