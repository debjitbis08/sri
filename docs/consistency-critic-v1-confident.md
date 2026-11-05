# Consistency critic v1 follow-up: confident errors and combination with margin

Contract written 2026-11-05, before the run. It follows [consistency critic v1](consistency-critic-v1.md), whose result was
that the reader's own margin is the better `eq` error detector on the two-space base, while the critic adds for `more` and on
the count base. User decision (2026-11-05): run the next step, which is whether the critic catches the errors margin misses.

Nothing in the system or in v1's critic and selectors changes. v1 kept no critic weights, so each snapshot's critic and
`sel-seen` selector are retrained by the same code and seeds. The same evaluation queries are drawn.

## Question

Does the critic detect **confident** reader errors, where margin is weak by construction? At a fixed flagging budget, does
splitting the budget between margin and critic catch more errors than margin alone? Does a combination fitted on collections
(where teaching labels exist) add to margin on forms?

## Reads (per snapshot, kind and operator; the 3,000 queries of v1)

- **Reproduction:** full-critic and margin AUROC must equal v1's report. If they do not, results are reported against this
  run, and the difference is stated.
- **Confident stratum (primary):** queries whose |reader logit| exceeds the median |logit| of the cell's errors. It holds the
  more confident half of the errors, with the correct queries at the same confidence. Labels define the stratum only for
  scoring. AUROC for error within it of the full critic, of `sel-seen` at k = 3 and of margin.
- **Secondary strata:** the third of queries with the largest |logit|, and |logit| > 2 (reader probability above 0.88). A
  check on one snapshot before the run showed that the top third can hold almost no `eq` errors (2 of 260), so it is
  reported but not used for the predictions.
- **Budget split:** flag 10% of the cell's queries.
  - Margin alone: the 10% with the smallest |logit|.
  - Critic alone: the 10% with the highest conflict.
  - Split: the union of the top 5% by each (at most 10%; its size is reported).

  Report the share of the cell's errors caught by each, and the errors caught only by the critic's half.
- **Fitted combination:** logistic regression on two standardized features, −|logit| and the logit of the critic's conflict
  probability. It is fitted per snapshot and operator on 3,000 separate collection queries (a from the collection test half,
  b ∈ M; teaching labels) and applied unchanged to every kind. AUROC for error, against margin. The collection cell is not
  held out for this read.

## Predictions

1. **[derived]** Errors are fewer in the confident stratum. On taught forms in the two-space arms (Chinese, Devanagari at block
   3), the full critic's AUROC within the stratum is above 0.6 for `eq` and `more`, and above margin's.
2. **[derived from v1]** At a 10% budget the split catches more errors than margin alone for `more` on Chinese, Devanagari and
   English, in every two-space arm. For `eq` on the two-space arms it does not.
3. **[derived]** The fitted combination is at least margin's AUROC (to within 0.01) in every cell. It exceeds margin by at least
   0.02 for `more` on forms and for `eq` on the count base.
4. **[read: v1]** On Thai the critic stays near 0.5 in the confident stratum: consistent wrong outputs are invisible.

If 1 fails, the critic and margin catch the same errors, and the critic's value as a trigger rests only on `more` and the
count base. If 3 fails where the rank mean also failed, the critic carries no usable extra information for that operator.

## Prior work

See [prior work: consistency critic](prior-work.md#consistency-critic-probe-selection-and-asking); additions there.
- **Margin is the maximum-softmax-probability baseline** for misclassification detection (Hendrycks and Gimpel 2017). It is
  the established detector this read must beat.
- **Agreement with a second predictor as a trust signal:** the trust score (Jiang et al. 2018) compares the classifier with a
  nearest-neighbour classifier and outperforms the classifier's confidence in their tests. The critic is a second predictor of
  the same output from other outputs, not from inputs.
- **Confident errors are "unknown unknowns"** (Lakkaraju et al. 2017): errors made with high confidence, which confidence-based
  active learning cannot find. Their remedy explores partitions of the data under an oracle budget. Here the question is
  whether a consistency signal finds some of them without any oracle.
- **What it adds:** a needed internal read before the asking and repair stages. No novelty claim.

## Scaling and computation

Same per-query cost as v1 for the critic, plus margin, which is free given the query's own output. The logistic fit is two
features and negligible. Retraining the critic and one selector repeats v1's training cost. Runtime in
`reports/consistency-critic-v1-confident-runtime.json`, results in `reports/consistency-critic-v1-confident.json`.
