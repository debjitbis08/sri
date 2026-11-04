# Which updates make old decisions wrong: margin decomposition of early relearning

Status: **diagnostic, 2026-11-04, seeds 300–303. Not a contract; no new learning policy.** The request (2026-11-04) was to explain the first lost decisions by decomposing their margin changes across parameter groups and loss paths, verify the decomposition against the actual changes, then propose one targeted intervention with a predicted outcome, with no new remedy screen.

Code: `experiments/relearning_margins.py`, with tables from `experiments/relearning_margins_summary.py` (`counterfactual` for the last table). Report: `reports/relearning-margins-v1.json.gz`. The per-query attributions are in `runs/relearning-margins-v1/margins-seed*.npz`, which is not committed (about 80 MB; reproducible by rerunning).

## What is decomposed

**Decision margins on fixed evaluation examples,** used for diagnosis only. For each query, m = score of the correct candidate − the maximum over the eight wrong candidates; the answer is correct when m > 0. Correct and wrong scores are tracked separately.

| Read | Query → nine candidates | Score | Queries per seed |
|---|---|---|---|
| Collections → taps, fresh | evaluation collection → evaluation taps | g(collection, taps), the order memory's recalled values need | 270 |
| Collections → taps, stored | memory's stored dot values → taps | same | 852 |
| Taps → collections | evaluation taps → evaluation collections | g(taps, collection), the trained order | 270 |
| Sequence and composition | the code predicted 1, 2 or 3 steps ahead → the nine collections | −‖prediction − collection‖² | 150–210 per read |

Sequence and composition cover collections up and down one step, taps up one step, collections up two and three steps, and taps up two steps. The draws are those of stage A's reads, so accuracy at update 0 is identical.

**The trajectory.** From the captured pre-relearning state, the original relearning (stage A's `unchanged`) is rerun for 20 updates, instrumented. The captured state equals step 8's delay checkpoint exactly.

**Exact split of each update by loss path.** Adam's step is −lr/bc₁ · m_t / den_t, where m_t = β₁m_{t−1} + (1 − β₁)·clip·Σₖ gₖ. Given the actual denominator, the step is exactly the sum of one part per loss path, each with its own first moment. The history part is Adam's moment at the captured state, decaying. There are eight paths:
- history;
- matching with form queries;
- matching with context queries;
- succession;
- comparison;
- sequence prediction on form rows through the candidates;
- sequence prediction on form rows through the prediction, with the temperature;
- sequence prediction on context rows.

Each part is further split by parameter group:
- the image head (the trained MLP on the frozen DINOv2 + TrOCR codes);
- the image projection `proj['image']`;
- the taps encoder;
- the taps projection;
- the comparator (g and its relation tokens);
- the predictor P;
- the temperature.

**The checks:**
- The path gradients sum to the actual clipped gradient: largest relative error 2·10⁻⁷.
- The parts sum to the actual parameter displacement: 1.2·10⁻⁵.

**Attribution.** Each output's change is attributed to each (path, group) part as a Jacobian–vector product computed by double backward. Two estimates are taken:
- first order, the gradient at θ_t;
- a two-point Gauss–Legendre path integral along the actual step.

## Validation: the decomposition explains the observed changes

The path integral reproduces the actual change of every output over each update. The first-order estimate is too coarse to use:

| Estimate | Margin change, median relative error | Worst update |
|---|---|---|
| First order (gradient at θ_t) | 0.21–0.32 | up to 1.26 |
| Path integral | 0.05–0.07 | ≤ 0.29 |

For the correct scores alone, the path integral's median error is 0.003–0.025. Summed over updates 0–9 and 0–19, the mean change of the lost decisions is reproduced within 1–7% for every read and score, for example:
- collections → taps margin, actual −23.47, estimated −23.50;
- descending collection prediction, actual −162.8, estimated −163.4.

All attributions below use the path integral.

## Findings

Accuracy falls from update 0 to update 20 (unchanged run):

| Read | At 0 | At 10 | At 20 |
|---|---|---|---|
| Collections → taps, fresh | 0.89 | 0.51 | 0.25 |
| Collections → taps, stored | 0.94 | 0.54 | 0.26 |
| Taps → collections | 0.97 | 0.59 | 0.47 |
| Sequence, collections up 1 | 0.92 | 0.71 | 0.52 |
| Sequence, collections down 1 | 0.91 | 0.72 | 0.43 |
| Composition, collections up 2 | 0.71 | 0.46 | 0.44 |
| Composition, taps up 2 | 0.76 | 0.80 | 0.53 |

**Lost decisions** were correct at update 0 and wrong at update 10. Their mean margin change over updates 0–9, by group and by path, means over seeds:

| Read (actual Δm) | Image head | Image projection | Comparator | Predictor | Form-seq via candidates | Form-seq via prediction | Match form | History |
|---|---|---|---|---|---|---|---|---|
| Collections → taps, fresh (−23.5) | **−21.1** | −6.1 | +3.7 | — | **−19.6** | −5.2 | +3.0 | −2.6 |
| Collections → taps, stored (−24.0) | **−20.9** | −6.7 | +3.6 | — | **−20.8** | −4.6 | +2.6 | −2.3 |
| Taps → collections (−24.9) | **−17.6** | −4.6 | −3.1 | — | **−15.5** | −5.1 | −5.0 | −1.2 |
| Sequence, collections up 1 (−43.2) | **−34.0** | −5.0 | — | −4.2 | **−20.7** | −18.2 | −1.4 | −2.8 |
| Sequence, collections down 1 (−162.8) | −86.5 | −20.2 | — | −56.7 | −56.9 | **−116.7** | +1.4 | +10.7 |
| Sequence, taps up 1 (−38.8) | **−30.2** | −8.6 | — | −1.3 | **−28.6** | −8.2 | −2.2 | −4.2 |
| Composition, collections up 2 (−26.6) | **−23.8** | −2.6 | — | −0.2 | **−22.2** | −2.9 | −2.0 | −0.1 |
| Composition, collections up 3 (−35.7) | **−32.6** | −5.3 | — | +2.5 | **−33.1** | +1.0 | −3.0 | −1.8 |
| Composition, taps up 2 (−23.8) | −1.6 | −3.6 | — | **−18.3** | −7.1 | **−19.0** | +0.3 | +2.7 |

The taps encoder and taps projection contribute less than ±1.4 in every row. The context paths (matching, succession, comparison, sequence on context rows) together contribute −1.8 to +4.8.

1. **The interpreter's lost decisions are representation changes, carried by one loss path.**
   - For collections → taps (fresh and stored) and for taps → collections, 89–116% of the lost margin comes from the image side: the image head and the image projection.
   - Within it, form rows' sequence prediction through its candidate collections carries 62–87% of the loss.
   - The largest single cell is that path into the image head: −16.1 of −23.5 on collections → taps, and −13.2 of −24.9 on taps → collections.
   - Correct scores fall and wrong scores rise by similar amounts (collections → taps: −8.7 and +14.8). The image change both weakens the right pairing and strengthens wrong ones.
2. **The comparator first compensates, then adds a smaller loss.** Over updates 0–9 its change raises the lost collections → taps margins (+3.7). Over 0–19 it lowers them (−3.0 for fresh, −4.0 for stored), mostly through form matching (−3.1 to −4.1). For taps → collections it is negative already in 0–9 (−3.1).
3. **The image head carries about three times the projection's share.** Earlier write-ups located the contraction in `proj['image']`; the head on the frozen codes moves more.
   - This also explains why freezing the head ([branches](relearning-branches-v1.md), `frozen-image`) did not prevent the collapse: the same loss path then acts through the projection.
   - So the unit to intervene on is the loss path, not a parameter group.
4. **Sequence prediction and composition have two mechanisms.**
   - **Collection-based reads** (one step up, two and three steps up, and taps one step up, which is scored against collections) lose margin mainly through the same candidate path, almost all of it into the image head (48–93% of the lost margin; the prediction path adds most of the rest for collections one step up).
   - **Taps composition and descending collections** lose it mainly through form rows' **prediction path into the predictor**: −20.8 of −23.8 for taps two steps up, and −65.0 of −162.8 for collections one step down. Forgotten forms' prediction errors retrain P in a way that damages its use on taps and on descending sequences.
5. **Context rows' sequence prediction does not harm these decisions** (−1.0 to +1.9 in 0–9). Stage A's stop-gradient stopped it too, which may explain part of that arm's weaker composition at the deadline. That was not tested here.

**A linear counterfactual: the same steps with one path's part removed.** Every other part of each actual step is held fixed: the same clipping, the same Adam normalization and the same trajectory for the remaining parts. Accuracy at update 10 / 20:

| Read | Unchanged | Without form-seq via candidates | Without form-seq via prediction | Without both | Without form matching |
|---|---|---|---|---|---|
| Collections → taps, fresh | 0.51 / 0.25 | **0.85 / 0.75** | 0.60 / 0.30 | 0.88 / 0.79 | 0.48 / 0.41 |
| Collections → taps, stored | 0.54 / 0.26 | **0.92 / 0.80** | 0.62 / 0.28 | 0.94 / 0.84 | 0.52 / 0.40 |
| Taps → collections | 0.59 / 0.47 | **0.91 / 0.91** | 0.65 / 0.62 | 0.94 / 0.94 | 0.69 / 0.67 |
| Sequence, collections up 1 | 0.71 / 0.52 | 0.89 / 0.92 | 0.74 / 0.59 | 0.90 / 0.92 | 0.76 / 0.60 |
| Sequence, collections down 1 | 0.72 / 0.43 | 0.88 / 0.82 | 0.74 / 0.60 | 0.90 / 0.91 | 0.73 / 0.50 |
| Composition, collections up 2 | 0.46 / 0.44 | 0.67 / 0.74 | 0.53 / 0.44 | 0.72 / 0.78 | 0.50 / 0.53 |
| Composition, collections up 3 | 0.38 / 0.32 | 0.59 / 0.69 | 0.41 / 0.39 | 0.63 / 0.68 | 0.39 / 0.40 |
| Composition, taps up 2 | 0.80 / 0.53 | 0.76 / 0.74 | 0.78 / 0.76 | 0.82 / 0.88 | 0.81 / 0.55 |

The counterfactual is linear: it ignores how the trajectory itself would change. A real gradient stop also changes the step sizes of the other paths, because clipping (factor 0.002–0.11 over these updates, set mostly by this gradient) and Adam renormalize.

The [mechanism](relearning-mechanism-v1.md) branch `seq-no-cand`, an actual stop of this path on form rows that is otherwise unchanged, read collections → taps 0.75 / 0.69 at 10 / 20. That is below this counterfactual's 0.85 / 0.75. The decomposition does not explain the gap; reallocation through clipping and Adam is the candidate explanation.

## Proposed intervention, with its prediction (not run)

Of the request's four cases, this is the second: **representation updates dominate, and comparator changes partly compensate.** The single intervention is on the responsible path and loss, not on all image learning.

**Intervention:** for the first 20 relearning updates, remove form rows' sequence-prediction gradient through the candidate collections. That gradient reaches only the image head and the image projection.
- Hold the step scale as in the original run: compute the clip factor and Adam's second moment from the full gradient, and remove the path from the applied first moment. The removed part is then not replaced by larger steps of the other paths, so the experiment tests the decomposition directly.
- Everything else is unchanged: the context rows' candidate path, form rows' prediction path, every other loss and the same draws.
- The reference is stage A's `unchanged`.
- The plain stop-gradient result from the mechanism branch is a second reference: it measures how much the reallocation costs.

**Prediction, stated before any run,** at updates 10 and 20, means over seeds:
- **Collections → taps:** fresh about 0.85 / 0.75 (unchanged 0.51 / 0.25); stored about 0.92 / 0.80.
- **Taps → collections:** about 0.91 / 0.91.
- **Memory-present symbol → taps:** close to the stored-values read, which is its interpreter. It is not decomposed here.
- **Collection sequence and composition:** up 1 about 0.89 / 0.92; up 2 about 0.67 / 0.74; up 3 about 0.59 / 0.69.
- **Reads this does not protect:** taps two steps up only partly (about 0.76 / 0.74), and collections down one step partly (0.88 / 0.82). Their second mechanism, the predictor path, remains.
- **English matching:** within about 0.03 of unchanged at 20. This rests on the plain branch (0.21 against 0.23 at 20, 0.30 against 0.29 at 50), not on this decomposition, which did not include English.

**How the result would be read:**
- **The prediction holds** if each listed read is within 0.10 of its predicted value at 10 and at 20. That means the decomposition's account of the early lost decisions is causal, and the remaining loss in the plain stop comes from reallocation.
- **If the reads fall well short,** effects the linear account omits (the trajectory's own change, and interactions between parts) matter, and they need examining before any intervention is chosen.
- **Either way, it is not a remedy.** Holding the step scale is a diagnostic device. What follows depends on whether the late phase (comparator recalibration through form matching, and the predictor path) needs its own treatment.

## What this does not establish

- **Beyond update 20.** The window covers the abrupt early collapse only, not the partial recovery or the late sag of the reverse read.
- **Why the image head responds more than the projection.** Its larger share may reflect Adam's per-coordinate normalization of a larger, previously quiet parameter set; this was not tested.
- **English acquisition and form sequence prediction.** They were not among the decomposed reads.
- **Memory-present reads.** These were not decomposed separately. The stored-values read is their interpreter step, and retrieval is unaffected by training.

## Computation

- **GPU** (GTX 1650), three workers, float32, deterministic algorithms.
- **Per seed:**
  - capture, about 155–204 s;
  - instrumented 20 updates, 47–50 s, which includes per path: 8 gradient evaluations per update, and 3 × 56 Jacobian–vector products per update by double backward over about 2,500 evaluation queries;
  - 265 s wall in all, 204 s for seed 303.
- **Totals:** 8 min 18 s wall, 1,105 s aggregate worker CPU, peak RSS 3.7 GB.
- **Storage:** per-query attributions of 20 updates × 8 paths × 7 groups × about 5,000 outputs, in two estimates, about 20 MB compressed per seed.
- **Complexity:** per update, O(K·G) JVPs for K paths and G groups, each costing about two backward passes over the evaluation graph, plus K gradient evaluations on the training batch. Linear in the number of evaluation queries.
- **Selection and tuning:** none. The window (20 updates) and the quadrature (two Gauss points, chosen against the first-order estimate in this run) are the only choices.
