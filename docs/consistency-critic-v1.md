# Consistency critic with learned probe selection v1

Contract written 2026-11-05, before the run. It follows the [critic notes](learned-consistency-critic-notes.md) (review
response, asking, learned probe selection) and the [equality diagnostics](equality-diagnostics-v2.md). User direction
(2026-11-05): the critic must not check every dependency; a learned filter chooses a small set. The number of candidates
may be reduced.

This is stages 1–2 of the staging in the notes, plus probe selection, on frozen snapshots. **No repair and no asking**: the
system does not change. Repair (with output preservation) and the asking test are later contracts.

## Question

1. Can a critic that sees only operator outputs, trained only on collection teaching labels, detect wrong operator outputs on
   forms? Does it add to the reader's own margin, the established cheap detector?
2. Can a learned selector keep most of the full critic's detection with k of 18–27 probe units?
3. Does the selection pattern differ by query operator in the way substitutability predicts?

## System (unchanged)

The 24 v2 snapshots of the diagnostics: `two-ctx-enbuf`, `rel-en`, `count-rel-en`; seeds 300–303; after blocks 2 and 3.
Operators `eq`, `more` (both blocks) and `sum10` (block 3 only). Items as the diagnostics: 30 per count and kind; the
collection **fit half** (135 items, 15 per count) is the memory of reference items `M`; reports are on each kind's **test
half** (collections, Chinese, Devanagari, English, Thai).

## Critic

- **Query:** operator q on (a, b), with a an item of some kind and b ∈ M. The query's own output is masked.
- **Units** (R operators, K = 8 contexts):
  - same-pair unit for each r: [r(a,b), r(b,a)], with r(a,b) masked when r = q;
  - context unit for each r and context m: [r(a,m), r(m,a), r(b,m), r(m,b)].

  That is R + R·K units: 18 at block 2, 27 at block 3. Values are the reader's binary outputs (logit > 0 as ±1). Grouping
  the four context outputs into one unit is the only structure given; which combinations are consistent is learned.
- **Candidate stage (reduced):** 3 nearest items of M to a by cosine on codes, 3 nearest to b, 2 random; distinct, excluding
  a and b.
- **Network:** DeepSets: a token MLP over [values, mask, operator embedding, unit-type embedding, query embedding], mean pool,
  head MLP to a logit. 64 hidden units.
- **Training:** collection queries only (a, b ∈ M, a ≠ b, q uniform); inputs are the reader's outputs; the target is the
  **teaching label** of q(a, b), with positive weight per operator. Each unit is kept with probability p ~ U(0.1, 1) per query,
  so the critic works from any subset. 2,000 Adam steps, batch 256, lr 1e-3. One critic per snapshot.
- **Conflict score:** the critic's probability that the reader's output on the query is wrong.

## Selector

- **Score per unit,** from the query alone before any unit is evaluated: query–key matching with learned projections. One
  16-dim query vector per (operator r, unit type) from [c_a, c_b, q]; dotted with K(c_m) for context units (K precomputable per
  memory item) or with a learned key per operator for same-pair units; plus a bias per (q, r, type). No fixed address.
- **Training target: distil the full check.** For 4,096 queries, the leave-one-out influence of each unit on the full
  critic: |σ(all units) − σ(all units without u)|, normalized per query; listwise softmax cross-entropy, 1,000 Adam steps,
  batch 256. Labels are not used.
- **Two selectors:**
  - `sel-coll`: queries from collections only.
  - `sel-seen`: one quarter each from collections and the **fit halves** of Chinese, Devanagari and English, label-free (the
    system sees its own outputs on these at consolidation). Thai is excluded.
- Online, only the top k units are evaluated, k ∈ {1, 2, 3, 4, 6, 9}.

## Reads (test halves; 3,000 sampled queries per kind and operator, a from the kind, b ∈ M)

- **AUROC for reader error** (reader output ≠ truth; labels used only to score):
  - full critic (all units);
  - **margin baseline:** −|query logit|, the reader's own uncertainty;
  - **margin + critic:** the mean of the two ranks, no fitting;
  - learned selection (`sel-coll`, `sel-seen`), random selection and **oracle** selection (top-k by true influence, which needs
    the full check; an upper bound for the selector), each at every k.
- **`eq` acceptances:** among accepted queries (eq > 0), the false share; the precision of the diagnostics' hand eq–more flag
  (a diagnostic reference, not part of the system); the critic's precision when it flags the same number of acceptances (full,
  and k = 3 for both selectors).
- **Selection pattern at k = 3,** per query operator: mean number of chosen units by type (same-pair, context) and operator.
- **Costs per query:** reader S calls and pair features for the full check and for k units; selector cost.

## Predictions

1. **[read: diagnostics]** On the two-space arms, the full critic's AUROC for `eq` errors on Chinese and Devanagari is above
   0.6. On `count-rel-en` it is near 0.5 for forms. On Thai it is near 0.5 for every arm.
2. **[derived]** The margin baseline is strong (adjacent-count errors sit near the decision boundary). The critic alone may
   not beat margin; the useful claim is margin + critic > margin by at least 0.02 on Chinese and Devanagari `eq` in the
   two-space arms.
3. **[derived]** Learned selection at k = 3 keeps at least 80% of the full critic's AUROC gain over 0.5, and beats random k =
   3, averaged over taught kinds. The oracle at k = 3 is at or above the full critic.
4. **[derived] Substitution pattern:** for `eq` queries, chosen units are mostly context units across operators; for `more`
   queries, the same-pair `eq` unit is chosen more often than for `eq` queries' context share would suggest. This is read, not
   a pass/fail criterion; a pattern that is the same for every query operator is evidence that the selector learned unit
   frequency, not query-specific use.
5. **[derived]** `sel-seen` transfers to forms at least as well as `sel-coll`. Selection trained on collections alone sees a
   distribution where most outputs are correct.

Failure of 1 on the two-space arms says the learned critic does not reproduce the hand-flag signal. Failure of 2 says
conflict adds nothing to the reader's margin, which makes the asking and repair stages hard to justify. Failure of 3 says
selection loses the critic's information at this budget.

## Prior work

See [prior work: consistency critic, probe selection and asking](prior-work.md#consistency-critic-probe-selection-and-asking).
- The critic is a **dependency network** (masked conditional prediction, pseudo-likelihood) over operator outputs. The
  selector is a **top-k learned gate** (as in sparse mixture-of-experts) distilled from leave-one-out influence; **active
  feature acquisition** with a sequential policy is the stronger known alternative, not included.
- **Established baseline:** the reader's own margin (uncertainty sampling in active-learning terms), included. Random
  selection is included; the oracle is an upper bound.
- **Known failure modes:** conditional frequencies taken as laws (the critic's inputs exclude codes and confidences, but it
  can still learn base rates); selection learning unit frequency rather than query-specific use (read 4); consistent wrong
  outputs are invisible (Thai, read 1). Repair-side failures (confirmation bias, anchor drift) do not arise here.
- **What it adds:** BeliefBank and ConCoRD use symbolic solvers over given relations; here dependencies are learned from one
  domain's examples and applied to another through shared operator outputs, and the check is sparse by a learned filter.
  This is a needed internal test, not a novelty claim.

## Scaling

- **Full check per query:** 2R + 4RK reader S calls (102 at block 3) and 2 + 4K pair features; critic O((R + RK)·H²).
- **Selected check:** at most 4k S calls (12 at k = 3) and the critic on k tokens. Selector: one query MLP per query, then
  R(K + 1) dot products of 16 dims; keys K(c_m) are computed once per memory item. The candidate stage is O(|M| d) cosine
  here; an approximate nearest-neighbour index makes it sublinear in memory size.
- **Growth:** units grow as R(K + 1) with the operator inventory; the selector's cost grows the same way but with a 16-dim
  dot product per unit, so the saving grows with R. The full check at consolidation is paid on a sample of queries.
- **Storage:** critic 10,553 parameters, selector 10,738 (block 3; float32), plus 16 floats per memory item for keys; nothing raw retained
  beyond the frozen v2 evaluation items already used.

## Computation (to be measured)

Per snapshot: table construction, critic training, influence and selector training, evaluation (including oracle influence on
evaluation queries), wall and CPU seconds, peak RSS; worker count and threads. Runtime in
`reports/consistency-critic-v1-runtime.json`, results in `reports/consistency-critic-v1.json`.

## Results

Run 2026-11-05 at revision `4ce8bae` (runner and contract committed before the run; the tree was dirty only through
unrelated untracked files). 24 snapshots, 8 workers. Results in `reports/consistency-critic-v1.json`; summary with
`.venv/bin/python -m experiments.consistency_critic_v1 summary`. Means over seeds unless a range is given. Devanagari is
taught in block 3, so at block 2 it is untaught, like Thai.

**AUROC for reader error at block 3** (full critic / margin / margin + full critic):

| Arm | `eq` zh | `eq` dev | `eq` thai | `more` zh | `more` dev | `more` en |
|---|---|---|---|---|---|---|
| `two-ctx-enbuf` | 0.65 / 0.83 / 0.78 | 0.73 / 0.84 / 0.84 | 0.47 / 0.70 / 0.60 | 0.85 / 0.82 / 0.88 | 0.88 / 0.77 / 0.85 | 0.88 / 0.79 / 0.87 |
| `rel-en` | 0.70 / 0.84 / 0.82 | 0.74 / 0.85 / 0.83 | 0.51 / 0.72 / 0.64 | 0.84 / 0.78 / 0.85 | 0.81 / 0.73 / 0.79 | 0.85 / 0.80 / 0.86 |
| `count-rel-en` | 0.83 / 0.76 / 0.84 | 0.80 / 0.78 / 0.84 | 0.59 / 0.62 / 0.62 | 0.80 / 0.81 / 0.84 | 0.79 / 0.86 / 0.86 | 0.80 / 0.74 / 0.81 |

**Selection, block 3, mean over collections, Chinese, Devanagari and English** (AUROC at k = 3: `sel-coll` / `sel-seen` /
random / oracle; full critic):

| Arm | `eq` | `more` | `sum10` |
|---|---|---|---|
| `two-ctx-enbuf` | 0.69 / 0.70 / 0.64 / 0.62; 0.70 | 0.82 / 0.83 / 0.70 / 0.83; 0.89 | 0.58 / 0.55 / 0.58 / 0.55; 0.66 |
| `rel-en` | 0.66 / 0.68 / 0.66 / 0.60; 0.72 | 0.81 / 0.81 / 0.67 / 0.80; 0.86 | 0.66 / 0.64 / 0.64 / 0.61; 0.74 |
| `count-rel-en` | 0.71 / 0.79 / 0.71 / 0.72; 0.81 | 0.77 / 0.80 / 0.64 / 0.81; 0.84 | 0.76 / 0.77 / 0.66 / 0.77; 0.81 |

**Units chosen at k = 3, block 3, Chinese queries** (`sel-seen`; same-pair units below 0.15 omitted unless noted):

| Arm | `eq` queries | `more` queries | `sum10` queries |
|---|---|---|---|
| `two-ctx-enbuf` | ctx `eq` 0.45, ctx `more` 1.09, ctx `sum10` 1.30 | ctx `more` 2.94 | ctx `sum10` 1.98, same `sum10` 1.00 |
| `rel-en` | ctx `eq` 1.23, ctx `more` 0.47, ctx `sum10` 1.14 | ctx `more` 2.92 | ctx `sum10` 1.98, same `sum10` 1.00 |
| `count-rel-en` | ctx `eq` 0.84, ctx `more` 1.02, same `eq` 0.66, same `more` 0.40 | ctx `more` 2.91 | ctx `sum10` 1.98, same `sum10` 1.00 |

The oracle and `sel-coll` choose almost the same units as `sel-seen`.

**`eq` acceptances, block 3, two-space arms** (precision of flags; hand eq–more flag / full critic at the same flag count):
Chinese 0.68 / 0.66 and 0.75 / 0.75; Devanagari 0.62 / 0.71 and 0.77 / 0.83; English 0.57 / 0.72 and 0.69 / 0.74
(`two-ctx-enbuf` and `rel-en`). Base rates (false share of acceptances): Chinese 0.45–0.46, Devanagari 0.25–0.28.

### Against the predictions

1. **Partly holds.** On the two-space arms the full critic detects `eq` errors on Chinese (0.65–0.70; seeds 0.53–0.83) and
   Devanagari (0.73–0.74) at block 3, and is near 0.5 on Thai (0.47–0.51) and on block-2 Devanagari, which was untaught then
   (0.43). **The count-base prediction fails:** there the critic is the best detector (Chinese 0.83, Devanagari 0.80), although
   the diagnostics' hand eq–more flag carried no information on that base. The learned critic uses more than that one
   dependency.
2. **Fails for `eq` on the two-space arms; holds for `more` and on the count base.** The reader's margin is the stronger `eq`
   detector (0.83–0.85), and the unfitted rank mean with the critic lowers it (0.78–0.84; per seed −0.11 to +0.05). For `more`
   on forms the critic beats margin alone (Devanagari 0.88 against 0.77) and the mean adds 0.03–0.13 per seed on the
   two-space arms. On the count base the mean adds to `eq` in every seed (Chinese +0.05 to +0.11). Which errors the critic
   catches that margin misses was not read.
3. **Mixed.** Learned k = 3 keeps at least 80% of the full critic's gain over 0.5 for `more` (all arms) and for `eq` on
   `two-ctx-enbuf` and with `sel-seen` on `count-rel-en`; it beats random by 0.12–0.16 for `more`. For `eq` on `rel-en` (73%)
   and for `sum10` on the two-space arms (25–50%, no better than random) it fails. **The oracle fails for `eq`** (0.60–0.72,
   below the full critic): leave-one-out influence underrates units whose evidence is shared with others. At block 2 learned
   k = 3 beats the full critic for `eq` (0.69–0.71 against 0.62); the mean pool dilutes a few informative units among many.
4. **Query-specific, but not as predicted.** The pattern differs by query operator: `eq` queries take context units of *other*
   operators (`more`, `sum10`), which is the substitution check (does `R(a,m)` match `R(b,m)`?); `more` queries take almost only
   `more` context units (transitivity through m); `sum10` queries take their own reversed same-pair unit (symmetry) plus
   `sum10` context units. The predicted same-pair `eq` unit for `more` queries is not chosen (0.00–0.02). Same-pair units
   matter for `eq` only on the count base.
5. **Holds.** `sel-seen` is equal to or better than `sel-coll`, most on `count-rel-en` `eq` (0.79 against 0.71).

**Hand flag against learned critic:** at the same number of flags, the full critic matches the hand eq–more flag on Chinese
and exceeds it on Devanagari and English.

### Bearing on the design

- A critic trained on collection teaching labels does transfer to forms through shared operator outputs, and a learned
  selector finds operator-specific checks without being given any. Its `eq` choices read as substitution checks.
- **For `eq` errors on the two-space base, the reader's own margin is the better trigger.** Those errors are adjacent counts,
  near the decision boundary, where margin is informative. A critic adds detection where margin is weak: `more`, and the count
  base. Before asking or repair, the next read is whether the critic catches **confident** errors that margin misses.
- Leave-one-out influence is a weak teacher for selection when evidence is redundant. Better targets are the drop in critic
  accuracy for a subset (greedy or sampled subsets) or a sequential acquisition policy.
- The full critic is not an upper bound either: a mean pool over all units can be worse than a good subset. An
  attention pool (Set Transformer) is the established alternative.

## Computation (measured)

- 24 tasks on 8 workers, one thread each, CPU (Ryzen 5 3600): 741 s wall, 5,183 s aggregate worker CPU. Median per snapshot:
  tables 1.4 s, critic training 51 s, influence and two selectors 25 s, evaluation 89 s (including oracle influence on
  evaluation queries). Peak RSS per worker 1.8 GB, mostly the shared v2 data loaded before the fork.
- Training: critic 2,000 steps × 256 queries; each selector 4,096 queries for influence (U + 1 critic passes each), then
  1,000 steps × 256. Evaluation: 3,000 queries per kind and operator; 15 (block 2) or 21 (block 3) kind–operator cells.
- Per query, the full check uses 68 reader S calls and 34 pair features at block 2, and 102 and 34 at block 3. At k = 3, at most
  12 S calls and 12 pair features: about an 8× saving at block 3. The selector costs one query MLP plus R(K + 1) 16-dim dot
  products. Critic 10,473 (block 2) and 10,553 (block 3) parameters; selector 8,568 and 10,738.
- Deterministic results were not checked by a rerun.
