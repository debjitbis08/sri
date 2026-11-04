# Remedies for the relearning collapse: candidates, prior work and the stage A screen

Status: **plan, revised 2026-11-04 after external review (*Changes from review*, below). Stage A approved and specified here, and run: [results](relearning-remedies-a-v1.md). No arm met the conditions for stage B.** Follows the [relearning mechanism](relearning-mechanism-v1.md), the [dynamics](relearning-dynamics-v1.md) and the [branches](relearning-branches-v1.md).

Prior work below is cited from familiar results and was not re-read for this plan. The review supplied links for the two most relevant: NormFace (arXiv 1704.06369) and Tian, Chen and Ganguli (arXiv 2102.06810). The shared `docs/prior-work.md` has uncommitted edits by the user, so entries go there once those land.

## The problem in one paragraph

When the core relearns forms it had forgotten (English plus rehearsal of earlier scripts), its sequence-prediction loss scores each prediction by −exp(scale)·‖pred − cand‖² over nine candidate collections.

Rows where a wrong candidate is much closer than the target cost hundreds. Their gradient pulls each target collection straight at a wrong prediction, with an inward component. Through the image projection shared by forms and collections, that sums to a largely uniform contraction of the image codes. Taps keep their own projection.

The two senses separate, and the core's collections → taps reading, which memory's recalled values depend on, collapses early and recovers only partly. Blocking the candidate-side gradient alone keeps most of the code space; it is the dominant identified route of the early collapse.

**The lesson:** a correct objective can still destroy a representation that other skills use, and memory cannot compensate when its interpreter moves.

## Three properties of the loss that produce it

1. **Unbounded, quadratic scores.** Rows where a wrong candidate is much closer than the target dominate the loss and its gradient.
2. **Scale sensitivity on this batch.** On the disruptive first relearning batch, a uniform shrink of all image codes locally lowers the loss (shrink derivative +169). That is a measured property of that batch, not a general one: on a delay batch the derivative is −1.2, and shrinking confidently correct predictions can raise their loss.
3. **Gradient into the targets.** The prediction objective moves the codes it predicts, which are shared with working skills.

## Candidates

**In the stage A screen:**

| Arm | Change | Hypothesis tested |
|---|---|---|
| `unchanged` | none | reference |
| `seq-cand-stop` | sequence prediction's gradient into its candidates is stopped, on all sequence rows (form and context) | removing property 3 for this loss prevents the early collapse |
| `seq-relative` | batch-relative distance logits (exact form below) | reducing property 2 for this loss prevents it |
| `seq-weight-0.1` | the original sequence loss at weight 0.1 | the collapse is only a matter of this loss's magnitude |

**`seq-cand-stop`** applies to sequence prediction only: matching, succession and comparison are unchanged. It applies to all sequence rows, as a general remedy would. The form-row-only intervention of the [mechanism](relearning-mechanism-v1.md) remains diagnostic evidence.

Candidates are not permanently fixed anchors. They still move through the shared parameters and every other loss; only this loss's direct pull on them is removed.

**`seq-relative`, the exact form.** For each sequence row with prediction p and nine candidates c₁…c₉ (the target at its count's slot):

- s = (1/9) Σₖ ‖cₖ − c̄‖², the candidates' mean squared distance from their centroid c̄, with gradient through s;
- logitₖ = −exp(scale) · s₀ · ‖p − cₖ‖² / (s + ε).

The constants and policies:
- **s₀** is the mean of s over all sequence rows of the first relearning batch at the captured state. It is computed once without gradient, so the logits equal the original loss's on that batch at update 0 on average, and the switch does not itself change confidence.
- **ε = 10⁻⁶ · s₀** is a numerical floor.
- **Temperature policy:** `scale` keeps training as before.

The arm stays a hypothesis to test, for two reasons:
- Jointly rescaling predictions and candidates leaves the logits unchanged. But scaling the core's codes need not rescale the predictor's output identically, so this does not prove contraction of the learned representation has no benefit.
- Differentiating through s introduces a new incentive: the model can change candidate spread to change confidence. Spread is logged.

**`seq-weight-0.1`:** the sequence loss on all rows is multiplied by 0.1, declared before the run. It is an order-of-magnitude reduction; the first relearning gradients were about 100 times the delay's. It is a control: if lowering this loss's weight alone avoids the collapse, the formulation is not what matters at this scale.

**Deferred until the individual results show a need:**
- **Combinations,** such as stop-gradient with batch-relative logits.
- **Bounded per-row influence.** Plain distance and a margin cap are different interventions:
  - plain distance bounds the distance derivative away from its singular point, but bounds neither the loss nor the gradient through the network;
  - a hard cap can stop learning exactly on the difficult examples.
- **Per-sense variance floors and whitening.**
- **Memory-anchored constraints.** Keeping the core's codes of stored values stable is a separate memory-assisted training question. It need not involve sleep, but it widens this task beyond correcting the identified failure of the prediction loss.

**Excluded or separate:**
- **Explicit collections → taps training** stays excluded by user decision.
- **A symmetric comparator** addresses the fragility of the reverse read, not the collapse.
- **Optimizer changes** treat a symptom. Adam's role is untested.

## Prior work, with its limits

- **Scale and temperature.** With logits −β‖p − c‖², scaling every code by α is the same as replacing β with α²β, so code scale and temperature are redundant parameterizations.
  - NormFace (Wang, Xiang, Cheng and Yuille 2017) supports scoring normalized features with a separate scale parameter.
  - It does not validate this candidate-spread denominator, which is our own construction and is tested as a hypothesis.
  - Neighbourhood components analysis (Goldberger et al. 2004) and prototypical networks (Snell, Swersky and Zemel 2017) also use softmax over squared Euclidean distances.
- **Stop-gradient on the target side.** BYOL, SimSiam, mean teacher and I-JEPA stop or slow gradients into the target side to prevent collapse in predictive objectives.
  - Tian, Chen and Ganguli (2021) analyse a simplified non-contrastive setting with a linear predictor and a stop-gradient. Our objective differs: nine candidates, a nonlinear predictor and shared training losses.
  - This work motivates `seq-cand-stop`. It does not predict that it will work here.
- **Bounded losses** (Ghosh, Kumar and Sastry 2017; Zhang and Sabuncu 2018) and margin softmaxes (CosFace, ArcFace) are references for the deferred bounded-influence arm.
- **Variance and whitening** (VICReg; W-MSE) are references for the deferred scale anchors.
- **Update constraints with stored examples** (GEM, A-GEM, orthogonal gradient descent, Learning without Forgetting, EWC) are references for the deferred memory-anchored constraint.
- **Comparing representations across checkpoints:** CKA (Kornblith et al. 2019) and orthogonal Procrustes, for any coordinate-compatibility check.
- **Adam:** RAdam (Liu et al. 2020) analyses oversized steps from unreliable second moments. It is relevant to the untested Adam hypothesis.

## Stage A: the screen

**From:** the captured pre-relearning states, seeds 300–303, with exact reproduction checked by the unchanged arm.

**Run:** 200 relearning updates per arm, with only the sequence loss changed.

**Reads every 10 updates,** on identical pairs and candidates:
- **The interpreter:**
  - collections → taps on fresh and stored values, in both comparator orders;
  - taps → collections;
  - memory-present symbol → taps and taps → symbol, sets A and B;
  - collection spacing, the senses' alignment, candidate spread s.
- **The abilities the loss is meant to teach:** sequence prediction and composition, read by the existing composition measure. That is the predicted next code (and k = 2 and 3 steps) matched among nine collections, by distance and by `same`, for collections, taps and each script's forms.
- **Acquisition:** English matching.

**Two outcomes per measure:**
- the **worst transient level:** the minimum over updates 0–200, against its value at update 0;
- the **level at a fixed recovery deadline:** update 200, against its value at update 0.

**What would favour an arm:**
- it limits the worst transient loss of the interpreter;
- it holds collections → taps and memory-present symbol → taps at the deadline;
- it does not weaken sequence prediction or composition, transiently or at the deadline;
- English reacquisition is no slower.

An arm that protects interpretation by weakening prediction learning does not pass.

## Stage B, conditional

Only for an arm that preserves both the combined system's interpretation and the sequence abilities the loss teaches:
- retrain the core from scratch with it;
- rerun relearning and the 1,000-update continuation;
- read everything step 8 measures;
- rerun stage 1's memory-and-core reads.

## Changes from review (2026-11-04)

- **Batch-relative scaling is a hypothesis.** Its exact denominator, numerical floor and temperature policy are specified. The claim that it removes the contraction incentive exactly is withdrawn, as is the claim that contraction is always a descent direction.
- **The stop-gradient applies to sequence prediction's candidates only, on all sequence rows.** The candidates remain movable through other paths.
- **Bounded influence is deferred.** A lower-weight control of the original loss replaces it in the screen.
- **Deferred:** memory-anchored constraints, variance floors and whitening, and the combination.
- **Sequence prediction and composition are measured in stage A.** Success is split into the worst transient and the level at a fixed deadline.
- **Prior-work claims are narrowed:** NormFace, and the stop-gradient theory.
