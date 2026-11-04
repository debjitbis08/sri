# What still breaks after the candidate path is stopped: margin attribution, updates 10–20

Status: **diagnostic, 2026-11-04, seeds 300–303. Not a contract; no new learning policy.**

The request (2026-11-04) was to reuse the [margin attribution](relearning-margins-v1.md) on the trajectory of the ordinary form-candidate stop-gradient, concentrating on updates 10–20, in order to:
- identify which remaining loss paths lower the correct-answer margins of collections → taps and of collection composition;
- separate decisions that become wrong from those already wrong at update 10;
- check whether the harmful changes lie in collection representations, the comparator or the predictor;
- propose one targeted intervention with a stated prediction, with no new loss-function sweep.

Code: `experiments/relearning_margins.py` with `MG_ARM=form-cand-stop`; tables from `experiments/relearning_margins_summary.py` (`window`). Report: `reports/relearning-margins-stop-v1.json.gz`. The per-query attributions are in `runs/relearning-margins-stop-v1/` (not committed).

## Method

The method is the same as the [margin decomposition](relearning-margins-v1.md): the same reads (memory present included), paths, groups, two-point path integral and checks. Only the trajectory differs.

**Trajectory:** the ordinary stop-gradient of form rows' sequence gradient through the candidate collections. That is the [path-removal](relearning-path-removal-v1.md) run's `form-cand-stop` and the [mechanism](relearning-mechanism-v1.md)'s `seq-no-cand`. That path's part is zero; the other seven are as before.

**Checks:**
- The gradient split is exact to 1.9·10⁻⁷, and the displacement reconstruction to 1.3·10⁻⁵.
- Per update, the path integral's median relative error on the margins is 0.02–0.05, with the worst update at 0.29.
- Summed over 10–19, the mean change of every decision set is reproduced within 1–7%, apart from one small set (taps → collections, recovered, 0.32).

As before, the instrumented trajectory departs slightly from the uninstrumented arm through restoring rounding. Collections → taps reads 0.74 / 0.67 here at 10 / 20, against 0.75 / 0.69 in the path-removal run.

**Decision sets in updates 10–19,** among decisions correct at update 0:
- **lost:** correct at 10, wrong at 20;
- **already wrong:** wrong at 10, split by whether they stay wrong or recover by 20;
- **kept:** correct at both.

## Accuracy on this trajectory

| Read | 0 | 5 | 10 | 15 | 20 |
|---|---|---|---|---|---|
| Collections → taps, fresh | 0.89 | 0.82 | 0.74 | 0.62 | 0.67 |
| Memory A, symbol → taps | 0.76 | 0.73 | 0.69 | 0.59 | 0.62 |
| Composition, collections up 2 | 0.71 | 0.68 | 0.69 | 0.60 | 0.53 |
| Composition, taps up 2 | 0.76 | 0.79 | 0.82 | 0.73 | 0.64 |
| Sequence, collections down 1 | 0.91 | 0.89 | 0.84 | 0.71 | 0.64 |

Collection spacing and the senses' alignment stay near their update-0 values: 23.1 and 0.36 at 20 in the uninstrumented run.

## Findings, updates 10–19

**Mean margin change by group and path,** with the largest cell:

| Read, decision set (n over seeds) | Actual Δm | Image head + projection | Taps encoder + projection | Comparator | Predictor | Largest paths | Largest cell |
|---|---|---|---|---|---|---|---|
| Collections → taps, lost (5–65) | −7.8 | −5.3 | −0.9 | −1.6 | — | match form −6.5, form-seq prediction −3.0, match context +2.6 | match form × comparator −4.9 |
| Collections → taps, already wrong, still wrong (13–34) | −3.0 | −2.4 | −1.0 | +0.4 | — | match form −2.9, form-seq prediction −2.0, match context +2.4 | match form × comparator −3.4 |
| Collections → taps, stored, lost (16–220) | −8.1 | −5.0 | −1.1 | −1.9 | — | match form −6.9, form-seq prediction −2.9, match context +2.6 | match form × comparator −5.4 |
| Memory A, symbol → taps, lost (7–53) | −9.4 | −6.2 | −1.0 | −2.2 | — | match form −8.8, form-seq prediction −3.6, match context +3.7 | match form × comparator −6.7 |
| Composition, collections up 2, lost (22–80) | −34.4 | +0.8 | — | — | **−35.2** | form-seq prediction −36.7 | form-seq prediction × predictor **−36.7** |
| Composition, collections up 2, already wrong, still wrong (3–27) | −42.2 | +1.4 | — | — | **−43.8** | form-seq prediction −46.2 | form-seq prediction × predictor **−46.7** |
| Composition, collections up 3, lost (22–85) | −39.6 | +1.7 | — | — | **−41.4** | form-seq prediction −43.3 | form-seq prediction × predictor **−43.4** |
| Sequence, collections up 1, lost (7–38) | −35.6 | −5.6 | — | — | **−30.0** | form-seq prediction −36.3 | form-seq prediction × predictor **−30.9** |

1. **Collection composition and sequence prediction fail through the predictor.**
   - Over updates 10–19, the decisions that become wrong lose almost all their margin through form rows' sequence prediction acting on the predictor P: 84–105% of the loss, by the predictor group.
   - The decisions already wrong at 10 lose margin the same way and fall further: −42 to −52 for those that stay wrong.
   - The collection representations are stable here. The image groups' contribution is small and often positive for composition, and spacing is kept.
   - This is the request's third case: the predictor's updates damage composition despite stable candidate representations. Forgotten forms' prediction errors retrain P in a way that breaks its use on collections, on taps and on descending sequences.
2. **The remaining collections → taps loss is smaller, and dominated by form matching.**
   - Here accuracy goes from 0.74 to 0.67, and memory from 0.69 to 0.62.
   - For decisions that become wrong, the image side carries −5.3 of −7.8 in all, from several paths. The comparator carries −1.6 net. The largest single cell is form matching into the comparator (−4.9), partly offset by context matching (+2.6 over all groups).
   - Form matching mostly raises wrong candidates' scores (max wrong score +4.0, of which form matching +5.1), whereas the correct score falls a little through several paths (−3.8).
   - Decisions already wrong at 10 fall further for the same reasons. The small sets that recover do so through context matching (+3.1 to +9.9).
   - Form matching contributes −6.5 of the −7.8 lost margin (memory A: −8.8 of −9.4), mostly through the comparator. Its largest harmful path also supports acquisition, so deleting it is not an obviously useful remedy.
3. **The two reads now have different mechanisms.**
   - The early collapse (updates 0–10, previous study) was the representation route.
   - What remains for the interpreter is diffuse comparator and representation change, driven mostly by form matching.
   - Composition's 10–20 decline is the predictor.

**Linear counterfactuals over updates 10–19,** on this trajectory, with the same limitations as before. Accuracy at 20, actual → without the stated part:

| Read | Form-seq prediction × predictor | Form matching × comparator | Form matching, all groups |
|---|---|---|---|
| Collections → taps, fresh | 0.67 → 0.67 | 0.67 → 0.71 | 0.67 → 0.72 |
| Memory A, symbol → taps | 0.62 → 0.62 | 0.62 → 0.65 | 0.62 → 0.65 |
| Composition, collections up 2 | 0.53 → **0.71** | 0.53 → 0.53 | 0.53 → 0.53 |
| Composition, collections up 3 | 0.45 → **0.66** | 0.45 → 0.45 | 0.45 → 0.45 |
| Composition, taps up 2 | 0.64 → **0.87** | 0.64 → 0.64 | 0.64 → 0.63 |
| Sequence, collections down 1 | 0.64 → **0.82** | 0.64 → 0.64 | 0.64 → 0.66 |
| Sequence, collections up 1 | 0.80 → 0.86 | 0.80 → 0.80 | 0.80 → 0.79 |

## Proposed intervention, with its prediction

Revised 2026-11-04 after review, before the run. Run: [predictor stop](relearning-predictor-stop-v1.md).

**Intervention.** From update 10 of the form-candidate stop trajectory, run updates 10–20 with the ordinary form-candidate stop-gradient kept, and also stop form rows' sequence-prediction gradient into the predictor's parameters:
- form rows' predictions are computed with P's parameters detached;
- their gradient into the form and collection codes they are made from is kept.

Context rows' sequence gradients (all parameters) and every other loss are unchanged.

**References:**
- the same trajectory continued unchanged from update 10;
- `unchanged`.

All arms see identical batches and draws.

**Optimizer state is preserved.** The arms share Adam's state at update 10, which already holds earlier form-driven contributions in its first moment. Those keep decaying into later steps. This tests stopping **new** form-row gradients into P, not removing all form-attributed movement of P after update 10.

**Form-sequence learning is not stopped by construction.** Form codes can still move, through the prediction path's gradient into the image head and projection, so that the existing predictor predicts them better. For new senses and forms this is the desirable route: they reach shared relational knowledge without rewriting it.

This is measured directly at updates 15 and 20, for English, Chinese and Devanagari forms, up and down one step:
- English form sequence prediction as read;
- the same with the **current form codes and the predictor as it was at update 10**: gains here come from representations adapting to the existing predictor;
- the same with the **update-10 codes and the current predictor**: gains here come from the predictor's change.

**Approximate expectations at update 20,** from the 10–19 counterfactual. They are not thresholds.

| Read | Without the intervention | Expected |
|---|---|---|
| Composition, collections up 2 | 0.53 | about 0.71 |
| Composition, collections up 3 | 0.45 | about 0.66 |
| Composition, taps up 2 | 0.64 | about 0.87 |
| Sequence, collections down 1 | 0.64 | about 0.82 |
| Sequence, collections up 1 | 0.80 | about 0.86 |

- **No improvement expected** on collections → taps (about 0.67) or memory-present symbol → taps (about 0.62 and 0.65). This intervention does not address their remaining loss. Memory-present reads stay central to the judgment.
- **English matching** is predicted to change little. That is a prediction, not a guarantee: its direct loss path remains, but the shared representations and the optimizer's scaling can still change.

**How it would be read:**
- **If composition survives and form-sequence learning continues,** the result supports adapting representations to the existing predictor in this setting. That continuation means English form sequence prediction still rises, with the gain visible under the update-10 predictor.
- **If composition survives and form-sequence acquisition weakens,** that is a tradeoff under this intervention and over this window. It does not show that separate predictors or an architectural redesign are necessary.
- **If composition is not protected,** the effects the linear account omits matter again, including the decaying optimizer history.

**The remaining interpreter loss is not addressed.** Its largest harmful path, form matching, also supports acquisition, so deleting it is not an obviously useful remedy.

## Computation

- **GPU** (GTX 1650), three workers, float32, deterministic algorithms.
- **Per seed:** capture 158–205 s; instrumented 20 updates 48–53 s.
- **Totals:** 8 min 25 s wall, 1,120 s aggregate worker CPU, peak RSS 3.7 GB.
- **Storage:** about 20 MB of per-query attributions per seed, not committed.
- **Selection and tuning:** none.
