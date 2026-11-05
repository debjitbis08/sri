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

## Results

Run 2026-11-05 at revision `c4a77c5`, 24 snapshots, 8 workers. Results in `reports/consistency-critic-v1-confident.json`;
summary with `.venv/bin/python -m experiments.consistency_critic_v1_confident summary`. **Reproduction is exact:** every
full-critic and margin AUROC equals v1's, so v1's critics and queries are retrained deterministically. Means over four seeds.

**Block 3, two-space arms** (`two-ctx-enbuf` / `rel-en`):

| Read | `eq` zh | `eq` dev | `more` zh | `more` dev | `more` en |
|---|---|---|---|---|---|
| Confident half of errors: critic AUROC | 0.56 / 0.63 | 0.59 / 0.58 | 0.81 / 0.81 | 0.79 / 0.71 | 0.81 / 0.81 |
| Confident half of errors: margin AUROC | 0.79 / 0.78 | 0.76 / 0.78 | 0.77 / 0.70 | 0.62 / 0.58 | 0.70 / 0.70 |
| 10% budget, share of errors: margin | 0.36 / 0.40 | 0.52 / 0.53 | 0.36 / 0.41 | 0.49 / 0.42 | 0.42 / 0.44 |
| 10% budget: critic alone | 0.39 / 0.45 | 0.49 / 0.56 | 0.50 / 0.46 | 0.65 / 0.46 | 0.61 / 0.52 |
| 10% budget: split | 0.37 / 0.38 | 0.50 / 0.55 | 0.42 / 0.44 | 0.57 / 0.46 | 0.49 / 0.45 |
| Fitted combination / margin | 0.84 / 0.83, 0.85 / 0.84 | 0.86 / 0.84, 0.87 / 0.85 | 0.87 / 0.82, 0.84 / 0.78 | 0.82 / 0.77, 0.79 / 0.73 | 0.85 / 0.79, 0.87 / 0.80 |

**Count base, block 3** (`count-rel-en`): in the confident half of errors, the critic beats margin for `eq` on Chinese (0.80
against 0.71) and Devanagari (0.77 against 0.72). The fitted combination adds to margin for `eq` (Chinese 0.83 against 0.76,
Devanagari 0.82 against 0.78, English 0.79 against 0.74) and for `sum10`.

**The secondary strata are uninformative for `eq`:** the top third by |logit| holds under 5% of `eq` errors on taught forms, and
the |logit| > 2 stratum holds 56–77% of all errors, so it is not a confident subset.

### Against the predictions

1. **Fails for `eq`, holds for `more` on the two-space arms.** On the confident half of `eq` errors the critic is near chance
   (0.56–0.63) and margin still ranks better (0.76–0.79): the critic does not find the `eq` errors margin is confident about.
   For `more` the critic is above 0.7 and above margin on every taught kind (Devanagari 0.79 against 0.62).
2. **Holds.** At a 10% budget the split catches more `more` errors than margin alone on Chinese, Devanagari and English in both
   two-space arms (by 0.01–0.08 in the mean; 4 of 24 seed–kind cells are lower, by at most 0.05). For `eq` it does not differ
   (−0.02 to +0.02). For `more`, giving the whole budget to the critic does better still (Devanagari 0.65 against 0.49).
3. **Mostly holds.** The fitted combination is within 0.01 of margin or better in every mean cell except `two-ctx-enbuf` Thai
   `eq` (−0.011). Per seed the worst cell is −0.04, on block-2 Devanagari, which was untaught then. It exceeds margin by at
   least 0.02 for `more` on forms in both two-space arms (+0.05 to +0.07) and for `eq` on the count base (+0.04 to +0.07); not
   for `more` on count-base Devanagari (0.86 against 0.86). On the two-space `eq` cells it adds 0.01–0.02.
4. **Holds for `eq`** (Thai, confident half: 0.47–0.51). For `more` the critic is slightly above chance on Thai (0.57).

### Bearing on the design

- **`eq` on the two-space base:** the critic and margin find the same errors, and margin finds them better, including
  confident ones. Equality errors here are adjacent counts that the consistency checks available (operators against nearby
  collection items) do not contradict, because neighbouring counts give nearly the same `more` and `sum10` pattern against
  most context items. So margin is the trigger for asking about `eq`; the critic adds nothing there.
- **`more`, `sum10` and the count base:** the critic catches errors margin misses, including confident ones, and a two-feature
  combination fitted on collections transfers to forms without harm. That is the combined trigger to use for asking.
- Fitting matters: v1's unfitted rank mean lowered `eq` detection, the fitted combination does not.
- Untaught kinds (Thai, block-2 Devanagari) remain invisible to both.

## Computation (measured)

24 tasks on 8 workers, one thread each, CPU: 262 s wall, 1,730 s aggregate worker CPU. Median per snapshot: 71 s to retrain the
critic and both selectors (as v1), 3.7 s for all reads, including the logistic fits (LBFGS, up to 500 iterations, two features)
on 3,000 collection queries per operator. Peak RSS per worker 1.7 GB. Margin needs no extra computation; per-query critic cost
is as in v1.
