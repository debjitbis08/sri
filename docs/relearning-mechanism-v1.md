# The precise route of the relearning disruption

Status: **diagnostics, 2026-11-04, seeds 300–303. Not a contract.** Follows [relearning dynamics](relearning-dynamics-v1.md), from the same captured pre-relearning state per seed (reproduced exactly; see the [branches](relearning-branches-v1.md)).

Code:
- `experiments/relearning_mechanism.py`: radial tests, stop-gradient and frozen-temperature branches, movement decomposition;
- `experiments/relearning_targets.py`: geometry of the predictions and the pull on each candidate.

Reports: `reports/relearning-mechanism-v1.json.gz` and `reports/relearning-targets-v1.json`.

**The question (user, 2026-11-04):** pursue the precise reason ordinary relearning collapses the code space that memory's interpreter depends on.

## Two facts about the core that set up the mechanism

- **Sequence prediction is scored by squared distances.** Its logits are −exp(scale)·‖pred − cand‖² (`transition_pilot.predict_logits`). The prediction `pred = P(prev, cur)` is ranked against nine collection codes, with the target at its count's slot. The loss has no ceiling: it grows with the square of how far a prediction is from its target.
- **Forms and collections share one linear image projection.** A code is `proj[medium](front end)`: forms and collections pass through `proj['image']`, taps through `proj['sequence']` (`src/standard_net.py`).

## Measurements

**1. At the first relearning batch,** with no update taken, each loss's pull on `proj['image']`:
- **Shrink derivative:** dL/d log α for every image code scaled by α. Exact, since the projection is linear; positive means shrinking all image codes lowers the loss.
- **Radial cosine:** the cosine between the gradient and the projection's own weights; +1 means descent is a pure uniform shrink.

The same is measured on a delay batch from the same state for reference.

| Batch, loss | Loss | Gradient norm on `proj['image']` | Radial cosine | Shrink derivative |
|---|---|---|---|---|
| Relearning, sequence prediction | 94 (45–164) | 343 (127–701) | 0.33 (0.28–0.36) | +169 (73–318) |
| Relearning, matching | 11.9 (7.3–15) | 31 (24–38) | 0.06 (−0.04–0.13) | +2.4 (−2.3–5.1) |
| Relearning, succession | 2.6 | 5.7 | 0.16 | +1.4 |
| Delay, sequence prediction | 0.04 | 2.1 | −0.19 | −1.2 |
| Delay, matching | 0.008 | 0.27 | 0.04 | −0.01 |

**2. Where the forgotten forms' predictions sit, and which way the loss pulls the collections.** At the same batch, for the sequence rows of form episodes (120–136 per seed) and of context episodes (26–38):

| | Form rows | Context rows |
|---|---|---|
| Share of all sequence-loss gradient on collection codes | 0.988–1.000 | 0.000–0.012 |
| Mean target probability | 0.53–0.61 | 0.90–1.00 |
| Distance to target / to the nearest candidate | 9.2–17.5 / 4.4–9.7 | 4.3–5.3 / 4.2–5.3 |
| Norm: prediction / target / other candidates | 11–21 / 10–13 / 17–22 | 11–13 / 9–12 / 17–23 |
| Descent on the target: cosine with the direction to the prediction | 1.00 | 1.00 |
| Descent on the target: cosine with the direction to the origin | 0.27–0.85 | −0.57–0.45 |
| Descent on other candidates: cosine with the direction to the prediction | −1.00 | −1.00 |

Gradient terms are weighted by their magnitude.

With exp(scale) ≈ 0.4, a target 17 away against a nearest candidate 10 away gives a row loss of about 0.4·(17² − 10²) ≈ 80. The loss depends on relative distances: a row is costly when a wrong candidate is substantially closer to the prediction than the target is. A distant target alone costs little if every candidate is equally distant. The mean target probability of 0.5–0.6 hides a tail of rows where a wrong candidate is much closer, and those rows dominate the loss and its gradient.

**3. Branches of 50 updates from the same state,** one change each:

| | Collection spacing (26.7 at 0) | Collections' best uniform scale vs. update 0, at 20 | Share of their movement a uniform scale leaves unexplained, at 20 | Taps' uniform scale, at 20 | Same-count c–t distance / spacing (0.28 at 0), at 50 | Collections → taps, at 50 | Stored values, at 50 | Memory: symbol → taps, at 50 | English, at 50 |
|---|---|---|---|---|---|---|---|---|---|
| As run | 6.5 | 0.23 (0.13–0.35) | 0.29 | 0.85 | 1.06 | 0.42 | 0.44 | 0.36 | 0.29 |
| Form rows' sequence gradient into the **candidate collections** stopped | **22.0** | **0.83 (0.81–0.88)** | 0.60 | 0.96 | **0.35** | **0.69** | **0.69** | **0.61** | 0.30 |
| Form rows' sequence gradient into the **prediction's inputs** stopped | 4.5 | 0.22 | 0.30 | 0.91 | 1.81 | 0.23 | 0.26 | 0.20 | 0.26 |
| Temperature (`scale`) frozen | 6.2 | 0.22 | 0.30 | 0.85 | 1.13 | 0.42 | 0.43 | 0.35 | 0.23 |

Each branch's first 20 updates, on `proj['image']`:
- **Radial cosine of the actual update** (cosine of each Adam step with −weights): as run 0.14; candidate-stopped 0.07; input-stopped 0.14; temperature frozen 0.14.
- **Norm after 20 updates:** as run ×0.951; candidate-stopped ×0.991; input-stopped ×0.954; temperature frozen ×0.951.
- **Radial cosine of the sequence gradient,** per update: as run 0.23; candidate-stopped 0.06. The matching gradient's is about 0.

## The mechanism

1. **Reintroduced forms make confidently wrong predictions.** The forgotten forms' sequence predictions land 9–17 code units from their target collections, against 4–10 from the nearest candidate. Squared distances turn that relative gap into very large differences between logits, so some rows cost hundreds. The sequence loss is about 94 and its gradient about 100 times the delay's.
2. **That loss pulls each target collection straight at the wrong prediction.** On the candidate side, descent moves every target collection directly toward its row's prediction (cosine 1.00) and pushes the other candidates away. Form rows carry 99–100% of this gradient on collection codes.
3. **The pull has an inward component, and summed over rows it is largely a contraction.**
   - The descent on target collections has a component toward the origin (cosine 0.27–0.85). This holds although prediction norms are not consistently smaller than target norms (11–21 against 10–13): a prediction can pull inward without having a smaller norm overall.
   - Through the shared linear projection, the summed gradient has a substantial radial component (cosine with the weights 0.23–0.33).
   - The shrink derivative is +169, so contraction locally lowers this loss. That does not make it the steepest direction: the radial cosine is well below 1, and most of the gradient points elsewhere.
   - After clipping and Adam, `proj['image']` loses 5% of its norm in 20 updates.
   - The collection codes shrink to 0.13–0.35 of their scale, and about 70% of their movement is that uniform scaling.
4. **Taps do not contract with them.** Taps are never candidates in these rows and have their own projection. Their scale stays at 0.79–0.96 after 20 updates. The two senses' count lines separate (same-count distance from 0.28 to 1.46 of the spacing).
5. **The comparator then re-adapts only in the order it is trained.** As the [dynamics](relearning-dynamics-v1.md) showed, the trained taps → collection read recovers within 50 updates. The reverse read, which memory's recalled collections need, recovers only partly.

**The candidate-side gradient is the dominant identified route of the early collapse under these conditions.** Blocking it greatly reduces the contraction, but does not restore the original performance: collections → taps reads 0.69 at 50 updates, against 0.89 before relearning.
- Stopping the form rows' sequence gradient into the candidates keeps the collections at 0.81–0.88 of their scale and the senses aligned (0.35). Collections → taps reads 0.69 at 50 updates against 0.42, and memory-present symbol → taps 0.61 against 0.36.
- English reacquisition is unchanged (0.30 against 0.29 at 50).
- Stopping the gradient into the prediction's inputs instead leaves the collapse as large or larger.
- Freezing the temperature changes nothing, so the loss does not lower its temperature in place of shrinking the codes.

**What this does not establish:**
- **Why the forgotten forms' predictions pull inward.** The predictor's response to out-of-practice form codes is not examined.
- **What else contributes to the failure.** Blocking the candidate route leaves collections → taps at 0.69, against 0.89 before. Form matching updates, other code movement and the comparator's re-adaptation remain.
- **Whether Adam's history enlarges the steps.** It is still untested; the radial direction comes from the gradient itself.
- **The smaller symmetry loss that form matching causes** ([dynamics](relearning-dynamics-v1.md)), or the late sag of the reverse read after 300 updates.
- **Remedies.** The stop-gradient branch is a diagnostic. It shows which route carries the collapse, and is not proposed as a fix.

## The architectural lesson

The desired relation need not be wrong for its training updates to destroy a useful representation. Here the sequence-prediction objective lets errors on forgotten forms move the collection codes that working skills depend on: taps ↔ collections, and the interpretation of recalled values.

Memory cannot compensate when its interpreter changes underneath it. The stored values were intact throughout; the core's reading of them was not.

## Remedies

Candidates, with the prior work they draw on, are in the [remedies plan](relearning-remedies-plan.md), for review. Nothing is run.

## Computation

- **GPU** (GTX 1650), three workers. The capture reruns each seed to the first relearning update, about 170–230 s per seed.
- **Mechanism run:** four branches of 50 updates with reads at 0, 10, 20 and 50, plus the radial tests; 7 min 8 s wall, 970 s aggregate worker CPU, peak RSS 3.7 GB.
- **Target geometry:** a single gradient evaluation per seed after the capture, no training; 6 min 31 s wall.
