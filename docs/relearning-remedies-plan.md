# Remedies for the relearning collapse: candidates and prior work (plan for review)

Status: **draft plan, 2026-11-04, for review. Nothing run.** Follows the [relearning mechanism](relearning-mechanism-v1.md), the [dynamics](relearning-dynamics-v1.md) and the [branches](relearning-branches-v1.md).

Prior work below is cited from familiar results and was not re-read for this draft. Exact formulations should be checked before any contract is frozen. The shared `docs/prior-work.md` has uncommitted edits by the user, so entries go there once those land.

## The problem in one paragraph

When the core relearns forms it had forgotten (English plus rehearsal of earlier scripts), its sequence-prediction loss scores each prediction by −exp(scale)·‖pred − cand‖² over nine candidate collections.

Rows where a wrong candidate is much closer than the target cost hundreds. Their gradient pulls each target collection straight at a wrong prediction, with an inward component. Through the image projection shared by forms and collections, that sums to a largely uniform contraction of the image codes. Taps keep their own projection, so they do not contract with them.

The two senses separate, and the core's collections → taps reading, which memory's recalled values depend on, collapses early and recovers only partly. Blocking the candidate-side gradient alone keeps most of the code space; it is the dominant identified route of the early collapse.

**The lesson:** a correct objective can still destroy a representation that other skills use, and memory cannot compensate when its interpreter moves.

## Three properties of the loss that produce it

1. **Unbounded, quadratic scores.** Confidently wrong rows dominate the loss and its gradient without limit.
2. **Scale sensitivity.** A uniform shrink of all codes lowers the loss; contraction is always a descent direction.
3. **Gradient into the targets.** A prediction objective moves the very codes it predicts, and those codes are shared with working skills.

## Candidate remedies

Each candidate changes the sequence-prediction loss or adds one term, and nothing else.

| # | Candidate | Removes | Expected effect | Risks |
|---|---|---|---|---|
| 1 | **Batch-relative distance logits:** each squared distance divided by that row's candidate spread, e.g. the mean squared distance of the nine candidates from their centroid, with gradient through both | 2, exactly: a uniform rescaling of all codes leaves every logit unchanged | no contraction incentive; keeps the number-line geometry, unlike per-item normalization | non-uniform contractions remain possible; the temperature must carry all confidence |
| 2 | **Stop-gradient into the candidates** of sequence prediction: it trains the predictor and the codes predicted from, while targets are fixed anchors | 3 | keeps the code space (diagnostic branch: spacing 22.0 against 6.5; collections → taps 0.69 against 0.42 at 50 updates) | collection codes no longer learn from sequence structure; succession and composition must be checked |
| 3 | **Bounded per-row influence:** plain distance instead of squared, or a margin cap on each row's logit gap | 1 | a confidently wrong row can no longer dominate | does not remove the contraction incentive alone |
| 4 | **Per-sense scale anchor:** a variance floor per sense and dimension, or per-sense whitening of codes | 2, through a separate term | collections cannot contract relative to taps | an extra term with its own weight; whitening costs O(d²) per batch for d = 16 |
| 5 | **Memory-anchored constraint:** keep the core's codes of memory's stored dot values stable relative to taps, up to scale, by a penalty or by projecting each update | the consequence for memory, not the cause | protects the interpreter memory depends on, using memory itself; does not train the failing read directly | which constraint to use; cost grows with the stored values used per update |
| 1 + 2 | combined | 2 and 3 | most likely to keep memory's interpreter stable | as 1 and 2 |

**Excluded or separate:**
- **Explicit collections → taps training** builds the answer into training. It stays excluded by user decision.
- **A symmetric comparator,** g(a, b) + g(b, a), addresses the fragility of the reverse read, not the collapse. It is a separate question.
- **Optimizer changes,** such as a warm-up after quiet phases, per-loss gradient normalization or Adam resets, treat a symptom. Adam's role is still untested. One diagnostic continuation with fresh optimizer moments could test it, but it is not a remedy.

## Prior work we can reuse

**Scale doubling as temperature (property 2).**
- **The identity:** with logits −β‖p − c‖², scaling every code by α is the same as replacing β with α²β. Code scale and temperature are redundant parameterizations, which is why shrinking substitutes for cooling.
- **NormFace** (Wang, Xiang, Cheng and Yuille 2017) analyses softmax over L2-normalized features and shows that a separate scale parameter is then needed. CLIP's learned logit scale is the same design. Candidate 1 follows it, with a batch-relative spread in place of per-item normalization.
- **Neighbourhood components analysis** (Goldberger, Roweis, Hinton and Salakhutdinov 2004) and **prototypical networks** (Snell, Swersky and Zemel 2017) use softmax over squared Euclidean distances. Their analyses cover why squared distance suits class means, and where its scale freedom comes from.
- **VICReg** (Bardes, Ponce and LeCun 2022) uses a variance term to prevent collapse, and **W-MSE** (Ermolov, Siarohin, Sangineto and Sebe 2021) whitens embeddings. They are the references for candidate 4.

**Gradient into the targets (property 3).**
- **BYOL** (Grill et al. 2020), **SimSiam** (Chen and He 2021), **mean teacher** (Tarvainen and Valpola 2017) and **I-JEPA** (Assran et al. 2023) stop gradients into the target side, or move it slowly, to prevent collapse in predictive objectives. They are the references for candidate 2.
- **Theory:** Tian, Chen and Ganguli (2021), *Understanding self-supervised learning dynamics without contrastive pairs*, analyse a linear predictor with a stop-gradient. They show how the predictor and the stop-gradient avoid collapse, and how the predictor's eigenspaces align with the representation's. Our predictor and our linear projection are close to their model, so their analysis can predict candidate 2's effect before it is run.
- **Measuring collapse:** Jing, Vincent, LeCun and Tian (2022) on dimensional collapse, and Wang and Isola (2020) on alignment and uniformity. Both give diagnostics beyond our uniform-scale measure.

**Unbounded row influence (property 1).**
- **Bounded losses from noisy-label learning:** mean absolute error (Ghosh, Kumar and Sastry 2017) and generalized cross-entropy (Zhang and Sabuncu 2018).
- **Margin-based cosine softmaxes:** CosFace and ArcFace. They are the references for candidate 3.
- **On the optimizer:** RAdam (Liu et al. 2020) analyses Adam's oversized steps when its second-moment estimates are unreliable. It is relevant to the untested Adam-history hypothesis after the quiet delay.

**Constraining updates with stored examples (candidate 5).**
- **Projection with memory:** GEM (Lopez-Paz and Ranzato 2017) and A-GEM (Chaudhry et al. 2019) project each update so that the loss on memory examples does not increase.
- **Orthogonal gradient descent** (Farajtabar et al. 2020) projects updates away from directions that change earlier outputs.
- **Penalties:** Learning without Forgetting (Li and Hoiem 2016) penalizes changes to earlier outputs on new inputs; elastic weight consolidation (Kirkpatrick et al. 2017) penalizes changes to important weights.
- Here the constraint would act on the core's codes of stored values, which memory already holds, rather than on task losses.

**Comparing representations across checkpoints.** CKA (Kornblith, Norouzi, Lee and Hinton 2019) and orthogonal Procrustes compare representations without assuming their coordinates line up. They give the coordinate-compatibility check required before any cross-checkpoint swap.

## Proposed evaluation

**Stage A, screen** (about 10 minutes):
- From the captured pre-relearning states (seeds 300–303), run 200 relearning updates with only the sequence loss changed. Candidates 1, 2, 3 and 1 + 2, against the unchanged loss. Candidates 4 and 5 need a weight and a constraint form, decided in review.
- Measure every 10 updates, as in the branches:
  - collection spacing and senses' alignment;
  - collections → taps on fresh and stored values, both comparator orders;
  - taps → collections;
  - memory-present reads (symbol → taps, taps → symbol);
  - English reacquisition.
- **Limitation:** the core was trained with the old loss. The screen shows only whether a loss avoids the collapse at this transition.

**Stage B, full curriculum,** for the best one or two:
- Retrain the core from scratch (about 4 minutes per seed), then rerun relearning and the 1,000-update continuation.
- Read everything step 8 measures: matching, the transfer measures, succession and composition. Changing the loss changes what the core learns.
- Then rerun stage 1's memory-and-core reads.

**Success,** stated for review:
- the combined system keeps interpreting memory through relearning: collections → taps and memory-present symbol → taps stay within 0.05 of their pre-relearning level, or recover to it;
- English reacquisition is no slower;
- the core's step 8 measures are not worse by more than 0.05.

## Questions for review

1. Is batch-relative scaling (candidate 1) the right form of scale invariance, or should codes be normalized per sense instead?
2. Should candidate 2 apply to sequence prediction only, or to every relation trained against collection candidates?
3. Is a memory-anchored constraint (candidate 5) within this stage, or is it consolidation work deferred with sleep?
4. Are stage A's four candidates the right first screen?
