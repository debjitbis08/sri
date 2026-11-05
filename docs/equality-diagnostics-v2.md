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

## Results (run 2026-11-05)

Means over seeds 300–303, test half of the evaluation items. Report: `reports/equality-diagnostics-v2.json`; `summary` and
`conflict` print every table. **Reproduction:** the snapshot rerun's report equals the v2 report exactly for all 12 runs.

### 1. Identity by read, Chinese at block 3 (block 2 in brackets)

| Read | `two-ctx-enbuf` | `rel-en` | `count-rel-en` |
|---|---|---|---|
| nearest | 0.57 (0.80) | 0.64 (0.81) | 0.64 (0.82) |
| `eq` by count, form first | 0.54 (0.83) | 0.58 (0.84) | **0.33** (0.57) |
| `eq` by count, form second | 0.55 (0.83) | 0.61 (0.81) | 0.46 (0.57) |
| difference reader, zero class | 0.57 (0.85) | 0.58 (0.84) | 0.15 (0.46) |
| relational | 0.56 (0.86) | 0.59 (0.85) | 0.19 (0.49) |
| refit `eq`, collections only | 0.57 (0.84) | 0.59 (0.83) | **0.37** (0.60) |
| refit `eq`, with Chinese fit half (upper bound) | 0.66 (0.91) | 0.65 (0.90) | **0.92** (0.91) |
| linear probe fit on collections | 0.57 (0.80) | 0.56 (0.79) | 0.54 (0.74) |
| linear probe fit on Chinese | 0.99 (1.00) | 0.99 (1.00) | 0.99 (1.00) |

Same pattern for Devanagari at block 3: `two-ctx` reads 0.83–0.85 everywhere; `count-rel-en` nearest 0.84, `eq` 0.40, refit on
collections 0.44, upper bound 0.97. Errors are mostly adjacent: 0.42–0.77 of identity errors are off by one for Chinese and
Devanagari, against about 0.2 by chance.

**Control for the probe:** a probe fit on Thai, never taught, reaches 0.84–0.87. The kind-fit probe therefore measures how
separable the glyph classes are, not count meaning; frozen senses separate digit glyphs on their own.

### 2. `eq` as a judgement (form × collection, block 3)

| | `two-ctx-enbuf` zh / dev | `rel-en` zh / dev | `count-rel-en` zh / dev | collections (all arms) |
|---|---|---|---|---|
| positive recall | 0.56 / 0.80 | 0.61 / 0.82 | 0.35 / 0.35 | 0.94–0.96 |
| false accept, \|Δ\| = 1 | 0.20 / 0.09 | 0.21 / 0.09 | 0.25 / 0.18 | 0.08–0.17 |
| false accept, \|Δ\| ≥ 2 | 0.02 / 0.01 | 0.03 / 0.03 | 0.03 / 0.02 | 0.00 |
| symmetry given accepted, P(eq(c, x) > 0 \| eq(x, c) > 0) | 0.78 / 0.85 | 0.74 / 0.81 | 0.40 / 0.48 | 0.92–0.97 |
| symmetry given rejected | 0.97 / 0.98 | 0.97 / 0.98 | 0.90 / 0.88 | 0.99–1.00 |

**Triples** zh → coll → dev (block 3, two-space arms): 20–21% of (zh, dev) pairs are joined by an accepted path through a
collection item, but only 37% of those are truly the same count, and 43–44% of them have eq(zh, dev) ≤ 0. Without a path,
eq(zh, dev) > 0 only 1% of the time.

### 3. Substitution against truth and against the system's own judgement (block 3, `more` / `sum10`)

| Chinese items | `two-ctx-enbuf` truth / own | `rel-en` truth / own |
|---|---|---|
| accepted and correct (n ≈ 68–78 of 135) | 0.94 / 0.94, 0.87 / 0.87 | 0.97 / 0.97, 0.85 / 0.85 |
| accepted and wrong (n ≈ 46–47) | 0.87 / 0.91, **0.55 / 0.87** | 0.84 / 0.89, **0.52 / 0.81** |
| rejected (n ≈ 10–21) | 0.86 / 0.87, 0.61 / 0.70 | 0.81 / 0.86, 0.60 / 0.69 |

Thai accepted-and-wrong: truth 0.58–0.59 / 0.44–0.47, own 0.92–0.94 / 0.79–0.85.

### 4. Conflict signal (label-free; `eq` accepted together with `more` true in either order)

Rate among accepted pairs, block 3: collections 0.10 < Devanagari 0.21–0.22 < Chinese 0.29–0.30 (two-space arms), Thai
0.35–0.42. On the count base, 0.63–0.79 for every form kind.

**Added after the run** (not in the contract; labels used only to score it), per pair among accepted form × collection pairs,
block 3:

| | conflict when truly equal | conflict when not equal | share of accepted that are false | share of conflicts that are false |
|---|---|---|---|---|
| `two-ctx-enbuf` zh / dev | 0.18 / 0.11 | 0.43 / 0.56 | 0.46 / 0.23 | 0.69 / 0.61 |
| `rel-en` zh / dev | 0.14 / 0.08 | 0.49 / 0.59 | 0.47 / 0.28 | 0.74 / 0.75 |
| `count-rel-en` zh / dev | 0.78 / 0.65 | 0.77 / 0.60 | 0.64 / 0.54 | 0.64 / 0.50 |
| Thai, two-space arms | 0.41–0.53 | 0.34–0.40 | 0.86 | 0.82–0.84 |

### What the diagnostics say

1. **On the two-space base the reader is not the bottleneck.** Every read of the same codes agrees within about 0.07: nearest,
   `eq`, zero class, relational, and a fresh reader. Only a reader that sees Chinese labels does better, and by just 0.07–0.09.
   What limits Chinese is how its codes sit against collections (prediction 1 holds for this base).
2. **On the count base the reader is the bottleneck, and a refit on collections does not fix it.** `eq` reads 0.33 where
   nearest reads 0.64. A fresh reader trained on collections reads 0.37; one that also sees Chinese reads 0.92. A reader on
   concatenated raw codes learns collection-specific features and does not generalize to codes off the collection manifold.
   Prediction 1's "refit closes the gap" fails.
3. **Errors are adjacent-count errors.** False acceptance at |Δ| = 1 is 7–10 times that at |Δ| ≥ 2 (prediction 2). The learned
   `eq` behaves as a tolerance relation, "within about one": symmetric on most pairs, but not transitive. Paths through
   collections chain adjacent acceptances, so 63% of them join different counts.
4. **Symmetry of accepted equalities is weaker than v2 reported.** v2's 0.91–0.94 was dominated by rejected pairs. Given
   acceptance it is 0.74–0.85 for forms on the two-space base and 0.40–0.48 on the count base (collections 0.92–0.97). This
   corrects v2 result 4.
5. **Substitution follows the system's own judgement** (prediction 4). Where it accepts a wrong equality, `sum10` agrees with
   the judged count (0.81–0.87) and not with the truth (0.52–0.55). Correctness is high only where the judgement is right.
6. **The label-free conflict signal exists and is informative per pair on the two-space base for taught scripts.** A false
   acceptance conflicts 3–7 times as often as a true one (Chinese 0.43–0.49 against 0.14–0.18). Flagged pairs are false
   61–75% of the time, against a base rate of 23–47%. It is **not** informative on the count base (0.77 against 0.78), and
   **not** for Thai, which was never taught (true equalities conflict as often as false ones). At the kind level it ranks
   collections < Devanagari < Chinese (prediction 5).
7. **Linear decodability is not evidence of kept meaning here.** Glyph classes are separable even for Thai, so "information
   present" in the probe sense says nothing about count. Prediction 3's premise was wrong.

### Bearing on the critic

On the two-space base, a consistency signal of the kind the critic would learn picks out wrong equalities on taught scripts
with useful precision. It cannot help a script with no taught relations (Thai), and it cannot help where the reader itself
fails (the count base). Errors are adjacent, so a repair would mostly be pushing a form from count n ± 1 to n. The
counterweight against drifting to "equal to nothing" is still needed: rejected items are a minority, but they exist
(n ≈ 10–21 of 135).

### Computation (measured)

- Snapshot rerun: 12 runs, 12 workers, partly sharing the CPU with smoke checks. Its timing is in
  `reports/equality-relations-v2-snap-runtime.json`. It reproduced v2 exactly.
- Weights: 36 snapshots, about 16 MB under `runs/equality-relations-v2-snap/weights`, not committed.
- Diagnostics: 24 snapshots on 12 workers, 221 s wall, 2,181 s aggregate CPU.
- Added conflict read: 33 s wall.
- Smoke checks: about 4 minutes for one snapshot, under full CPU contention.
