# Equality diagnostics on v2 snapshots

Contract written 2026-11-05, before the diagnostics run. It follows [equality v2](equality-relations-v2.md) and the external
review in [equality goal analysis](equality-goal-analysis.md). The user's decision: do the diagnostics first, then decide on
a consistency critic.

## Question

Is the v2 failure of exactness a representation failure, where count meaning is poorly present in the codes? Or is it a
reader failure, where the `eq` reader cannot extract meaning that is there? Is the label-free conflict signal a critic would
learn from present, and does it separate a drifted script from a fresh one?

## Snapshots

A rerun of three v2 arms with weights kept after every block: `two-ctx-enbuf`, `rel-en`, `count-rel-en`, seeds 300–303,
unchanged v2 runner (`EQ_SNAP=1`). Reproduction check: the rerun's report must equal the v2 report for these arms exactly. If
it does not, the diagnostics are reported against the rerun only, and the difference is stated.

## Diagnostics (after blocks 2 and 3; frozen weights; nothing in the system changes)

Evaluation items are those of v2: 30 per count and kind. Items 0–14 of each count are the **fit half**, used only by the
refits; items 15–29 are the **test half**, on which everything is reported. Collection references are all 270 collection
evaluation items.

**1. Identity by six reads on the same codes.** Each read is split into taught and withheld counts. Each reports accuracy,
within one, the share of errors that are adjacent (|error| = 1), and the score margin (top-1 minus top-2) where a score
exists.
- Nearest collection-count mean (as v2).
- `eq` by count: argmax_n of the mean `eq` logit against collection items of count n. Read with the form first (the
  trained order) and with the form second.
- `eq` by item: the v2 forced-choice argmax item.
- Difference reader, zero class: argmax_n of the mean log p(d = 0 | x, c) over collection items c of count n.
- Relational identity (Part C).
- Linear count probe on the codes. One probe is fit on collections and applied to the kind, which measures transfer of
  collection count structure. One is fit on the kind's fit half, which is an upper bound for linearly available count.

**2. Reader refits.** A fresh `eq` reader with the same architecture, on the same frozen pair features (G for `two-ctx`,
[c_a, c_b] for `count`), trained 500 Adam steps on minibatches of 8,192 pairs. One is trained on collection pairs only, as the original. One is trained
on collection pairs plus kind × collection pairs from the fit half, an upper bound. Both are read as `eq` by count on the
test half. If the refit on collections only beats the original reader, the original reader is under-using the codes. If
only the upper bound is high, the information is present but not in collection form.

**3. `eq` as a judgement** (form × collection pairs, test half):
- Positive recall: P(eq > 0 | same count).
- False acceptance at |Δ| = 1 and at |Δ| ≥ 2.
- Acceptance: the share of items whose best count score is > 0. `eq`-by-count accuracy among accepted and among rejected
  items.
- Symmetry split by sign: P(eq(c, x) > 0 | eq(x, c) > 0) and P(eq(c, x) ≤ 0 | eq(x, c) ≤ 0).
- **Triples:** zh → coll → dev. Among (zh, dev) pairs joined by an accepted path through some collection item, the share
  with eq(zh, dev) ≤ 0 (violations). Among pairs with no path, the share with eq(zh, dev) > 0. Each is reported with the
  true same-count rate in that set.

**4. Substitution against truth.** For each item, its own `more` and `sum10` outputs against every collection item, in both
orders, are scored against two targets. The first comes from the item's true count (correctness). The second comes from the
count its own `eq` judges (consistency). Both are split by accepted-correct, accepted-wrong and rejected items. This replaces
v2's forced-choice reference read.

**5. Conflict rates (label-free, no critic trained):**
- **eq–more conflict:** over kind × collection pairs, eq(x, c) > 0 together with more true in either order.
- **More both ways:** more true in both orders.
- **Kinds:** coll×coll, zh, dev, en and thai.

This shows whether the signal a critic would learn from exists and ranks kinds. Expected order if it is a usable trigger:
coll < dev (just taught) < zh (drifted) at block 3.

## Predictions

1. **[v2 reads]** On the `count` base, nearest identity exceeds `eq` by count by more than 0.2, and the collection-only refit
   closes most of that gap. That would make it a reader failure. On `two-ctx` the reads are within 0.05 of each other.
2. **[v2 irreflexivity failure]** Most `eq` errors on forms are adjacent-count errors. False acceptance at |Δ| = 1 is several
   times that at |Δ| ≥ 2.
3. The kind-fit linear probe is well above the collection-fit probe for Chinese at block 3. Count is present in Chinese codes,
   but not in collection form.
4. Among accepted equalities, consistency exceeds correctness on wrong acceptances (the system follows its own judgement), and
   correctness on accepted-correct items is high (> 0.9 for `more`).
5. Conflict rates order coll < dev < zh at block 3 in the two-space arms.

## Prior work

The review's search on 2026-11-05 covered supervised contrastive learning (Khosla et al. 2020); see the
[analysis](equality-goal-analysis.md). The methods here are standard diagnostics:
- **Linear probes** (Alain and Bengio 2016) measure what is linearly decodable.
- **Reader refits on frozen features** separate representation from readout, as in linear evaluation of self-supervised
  features.
- **Error-distance analysis** follows the numerical distance effect (Moyer and Landauer 1967).

Known limit: a probe fit on the same kind can find count information that no reader in the system uses. That is why it is
an upper bound only. Nothing new is claimed; this is an internal diagnostic.

## Computation (declared)

- **Rerun:** 12 runs × 4,500 steps, ≈ 1,275 CPU s each (v2's measured cost), on 12 workers.
- **Weights:** ≈ 0.45 MB per snapshot × 3 blocks × 12 runs ≈ 16 MB under `runs/`, not committed.
- **Diagnostics:** per snapshot (24: blocks 2 and 3), the following.
  - **Pairs:** 270 × 270 pair reads per kind and order.
  - **Refits:** two reader refits of 500 steps on ≤ 73k pairs each.
  - **Probes:** linear probes of 16 × 9 parameters.
  - **Triples:** 135 × 270 × 135 by matrix products.
  - **Cost:** expected under a minute of CPU each.

## Commands

```
EQ_SNAP=1 EQ_WORKERS=12 .venv/bin/python -m experiments.equality_relations_v2
.venv/bin/python -m experiments.equality_diagnostics_v2
.venv/bin/python -m experiments.equality_diagnostics_v2 summary
```
