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
