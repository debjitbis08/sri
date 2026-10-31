# Forgetting mechanism v1: contract (frozen 2026-10-31)

Status: **frozen 2026-10-31**, after an external review and the prior-work reading. Arms, seeds, branches, endpoints and readings below do not change after results are seen; anything added later is labeled post hoc. Implemented in `experiments/forgetting_mechanism.py` (smoke-checked: frozen parameters bitwise unchanged, `noupdate` reproduces update-0 recall exactly, `zero-grad` moves weights, frozen-bank decoder equals the rebuilt bank at update 0).
Follows [arbitrary pairings v1](arbitrary-pairings-v1-results.md), where recall of the 45 taught pairs fell to near chance (0.12–0.15 against 0.111) by the end of block 5 in every arm, with no interference from B. Before any
further architecture comparison (random rewiring, routing, working memory) we want to know **why** the pairs are lost.

## What the finished study already shows (post hoc, descriptive; seeds 320–329, Option A, branch `P-A`)

These come from the stored curves and decoders; no new training. They set the questions, they are not readings of this contract.

- **Time course.** Recall on A (arms in order sparse48 / dense / mixed / mixed-h125 / mixed-h500 / mixed-reset) at in-block-5
  updates 0, 25, 50, 250, 1000, 2000: 0.34, 0.30, 0.28, 0.25, 0.16, 0.13 (sparse48); 0.53, 0.45, 0.42, 0.39, 0.23, 0.16 (dense);
  0.48, 0.42, 0.38, 0.34, 0.22, 0.16 (mixed); 0.46, 0.34, 0.34, 0.24, 0.16, 0.13 (mixed-h125); 0.47, 0.33, 0.33, 0.24, 0.14, 0.12
  (mixed-h500); 0.52, 0.45, 0.42, 0.36, 0.23, 0.15 (mixed-reset). A fast drop in the first 25–50 updates (about a quarter to a
  third of the excess over chance for the dense arms, more for the decaying ones), then a slow decline to the floor over about 1,500 updates.
- **Recognition outlives the association.** A 1-nearest-neighbour decoder on the item code, using the symbols' training renderings
  with their true counts as the bank, still reaches count accuracy 0.74 (dense), 0.71 (mixed), 0.74 (mixed-h125), 0.72 (mixed-reset),
  0.48 (sparse48) at the end of block 5, against 0.86, 0.80, 0.82, 0.83, 0.58 at the end of block 4; symbol-identity accuracy 0.38–0.66
  against 0.43–0.76. The network's own matching readout fell to chance. The decoder is given the labels, so this says the symbols
  stay separable in the embedding; it does not say the association is stored there. What is lost, at least first, is the mapping from
  symbol to count in the matching function.
- **Learning B is not necessary for the forgetting.** `P-A` (no B) falls as far as `P-AB`. Capacity limits and representational
  competition are not excluded: the seed-299 size pilots (N = 18, 45, 90; 8 renderings; one seed) reached chance at block 5 but were not
  designed to test them.

## Question

Which changes make an acquired association inaccessible? Candidate accounts, which the branches below narrow but do not prove:

1. **Exposure:** recall depends on pair items being in recent batches (the fast early drop); continued exposure protects it.
2. **Gradient-driven change from other material:** training on the context episodes changes the encoders or the readers so the association no longer works.
3. **Optimizer or decay movement without new information:** inherited Adam state, or explicit decay (`mixed-h125`), moves weights anyway.
4. **Hidden rather than erased:** some residual of the original assignment survives and can be reused by learning.

## Design

Common start: the end of block 4 of the arbitrary-pairings schedule, Option A (connectivity fixed during pairing; RigL stopped at update
4,500 and no resets, so structure rows are static and only gradients, Adam state and, for `mixed-h125`, explicit decay change weights),
arms `sparse48`, `dense`, `mixed`, `mixed-h125`, **fresh seeds 330–339**, N = 45, L = 2,000, 64 training renderings, GPU, symbol sets and
count assignments identical across arms within a seed. `mixed-h500` and `mixed-reset` add little here and are left out (stated, not hidden).
Each branch starts from the same stored end-of-block-4 state (model, encoders, Adam moments, random streams).

**Parameter groups.** *Encoders* = image encoder and sequence encoder. *Readers and downstream* = every parameter of the model (`sri`) that
is not an encoder: the first-layer structure and memory rows, all later layers of the readers, and the shared matching/relation layers. The two
groups partition all trainable parameters; freezing sets `requires_grad = False` and a smoke check asserts the frozen parameters are bitwise unchanged.
Where decay is explicit (`mixed-h125`) it is a third, separate factor.

Branches (2,000 updates unless stated; context episodes as in frozen block 5):

| Branch | What changes | Question |
|---|---|---|
| `noupdate` | nothing: evaluation of the end-of-block-4 state, no forward in training mode, no decay | zero reference; must reproduce update-0 recall exactly |
| `zero-grad` | `opt.step()` with **explicit zero gradients** (tensors of zeros, not absent), no decay: only the inherited Adam moments move weights | movement from optimizer state alone |
| `decay-only` (`mixed-h125` only) | explicit decay each update, no gradient steps | decay alone |
| `ctx` | context episodes, everything trainable, decay as the arm's recipe (= frozen `P-A`) | reference |
| `ctx-nodecay` (`mixed-h125` only) | `ctx` with decay disabled | decay's contribution under training |
| `ctx-enc-frozen` | context episodes; encoders frozen; readers and downstream train; decay as the arm's recipe | are reader-side updates sufficient |
| `ctx-read-frozen` | context episodes; readers and downstream frozen; encoders train; decay disabled for `mixed-h125` (comparator: `ctx-nodecay`) | are encoder-side updates sufficient |
| `hold` | context + A pairs as in block 4 | does continued exposure protect |
| `rehearse-10` | context + A pair items on every 10th update | one low rehearsal dose |

Zero gradients and absent gradients differ for Adam: with absent gradients (`None`) parameters are skipped and nothing moves; with zero
gradients the stored first and second moments are applied (first moment decays by β₁ per step, so the movement is brief). `zero-grad` uses
zero gradients; a smoke check records the weight displacement to confirm it is nonzero and shrinking. `pairs-only`, learning-rate scaling and
a second rehearsal dose from the earlier draft are dropped: they change the optimization regime and the material together and cannot isolate a cause.

**Relearning (association versus familiarity).** From the identical end of `ctx` (2,000 updates in), three continuations of 250 updates (context + pairs at the
block-4 rate), evaluated at 0, 25, 50, 100, 250: the **original** assignment of A; the **reassigned** counts for the same A symbols (the frozen study's
second assignment A′); and **B**, never trained in this branch (novelty reference). Early-window recall (0–100, trapezoid mean above chance):
original − reassigned = assignment-specific advantage (reassignment interference is part of it); reassigned − B = symbol familiarity.

**Evaluation points** in every branch: update 0, 10, 25, 50, 100, 150, 200, 250, then every 250 to 2,000: **15 points**.

## Measures

- **Recall on A**, independent and training renderings, every point; shuffled-label rescoring as a check.
- **Accuracy change** Δ(t) = recall(t) − recall(0) for each seed and arm (paired), the inferential endpoint, at t = 50, 250 and 2,000.
  Loss fraction (1 − excess(t)/excess(0)) is descriptive only, reported when the seed's excess at 0 is above 0.10 and otherwise as undefined; values below 0 or above 1 are shown, not clipped.
- **Decoders on the item code**, 1-NN symbol identity and count, in two forms: (i) **rebuilt bank**, bank and queries encoded by the current encoder (separability now), and (ii) **frozen bank**, bank encoded by the
  end-of-block-4 encoder and stored, queries by the current encoder (representation movement). The gap between (i) and (ii) is the movement hidden by rebuilding.
- **Component swap** at 50, 250 and 2,000 updates in `ctx`: recall with (current encoders, end-of-block-4 readers and downstream) and (end-of-block-4 encoders, current readers and downstream). It shows
  which side's change breaks the interface; mismatch can also come from co-adaptation, so it is descriptive.
- **Weight movement** per group (image encoder, sequence encoder, structure rows, memory rows, later reader layers, shared layers) as relative L2 change from block 4, descriptive.
- **Linear probe** (descriptive, after Davari et al. 2022): ridge-regularized linear classifier from the current item code to the count, fit on the training renderings and scored on the evaluation renderings, at the same points as the decoders.
- Structured transfer (a) at the end of each branch, descriptive.

## Readings (prespecified; margin 0.05; the inherited rule: better/worse if |mean| > margin and the paired 95% t-interval excludes 0; equivalent if the interval lies within ±margin; noninferior if the lower bound exceeds −0.05; otherwise inconclusive; per arm over ten seeds; for differences between branches, Δ(branch) − Δ(reference), so positive means less loss)

1. **Evaluation stability.** `noupdate` Δ = 0 exactly (a runner check, as `dense` B = A in the last study).
2. **Optimizer state and decay.** `zero-grad` Δ(250), Δ(2,000) against 0: equivalent → inherited Adam state does not move recall; worse → it does. For `mixed-h125`, `decay-only` against 0 likewise, and `ctx` − `ctx-nodecay`.
3. **Exposure.** `hold` − `ctx` and `rehearse-10` − `ctx` at 250 and 2,000, and each against 0 for noninferiority. Better means continued pair exposure protects recall; it does not separate "stopping exposure" from "interference that exposure counteracts", and low-dose protection does not rule out capacity limits. Equivalence of start and end under `hold` is reported as such and is not called a plateau.
4. **Which updates are needed.** `ctx-enc-frozen` − `ctx` and `ctx-read-frozen` − `ctx` (for `mixed-h125` read-frozen against `ctx-nodecay`) at 250 and 2,000. Better → updates of the unfrozen-complement group are *necessary* for the loss in this setting; equivalent → freezing that group does not prevent it, so the other group's updates are *sufficient*. These are interventions on necessity and sufficiency of updates; they do not locate storage.
5. **Savings.** Original − reassigned and reassigned − B in early-window recall. Original − reassigned above the margin → assignment-specific residual information detected; equivalent → no savings detected by this test, not demonstrated erasure.
6. **Recognition and representation movement.** Frozen-bank and rebuilt-bank 1-NN identity at the end of `ctx` against update 0: descriptive with the same intervals; no outcome label.

Everything else (weight movement, swaps, per-part drift, arm differences) is descriptive. One fixed cohort; no stopping rule; inconclusive stays inconclusive.

## Prior work (written 2026-10-31; shared survey in [prior-work.md](prior-work.md#forgetting-mechanism-design-for-the-forgetting-mechanism-contract))

Corresponds to: the layer-wise analysis of forgetting ([Ramasesh et al. 2020](https://arxiv.org/abs/2007.07400) **[abstract]**: deeper layers are the main source), representation-versus-head
forgetting measured with a linear probe ([Davari et al. 2022](https://arxiv.org/abs/2203.13381) **[abstract]**), training-regime effects ([Mirzadeh et al. 2020](https://arxiv.org/abs/2006.06958)
**[abstract]**), weight protection ([Kirkpatrick et al. 2017](https://arxiv.org/abs/1612.00796) **[abstract]**, not included), and recovery by rehearsing a subset (Hinton and Plaut 1987 **[secondary]**, a
web-search summary). Catastrophic interference, rehearsal and complementary learning systems (McCloskey and Cohen 1989, French 1999, Robins 1995, McClelland et al. 1995) remain **[familiar]** and are context only.
Expected from this work (hypotheses, not facts for this setting): reader-side freezing prevents more of the loss than encoder-side freezing; representations stay separable while the readout fails. Known failure
modes carried into the Risks: freezing changes the optimization regime, probes with labels measure separability. Addition suggested by Davari et al.: a **linear probe** trained on the current code (symbol → count, fit on
the training renderings, tested on evaluation renderings) as a descriptive measure beside the 1-NN decoders. The study adds, beyond prior work, an intervention-based account (necessity and sufficiency of updates by group) in a
reader that learns arbitrary pairs; the stored-data reanalysis is the cheaper alternative and is done. **Prerequisite status:** the arXiv items are read at abstract level; the older papers are not re-read, so no prediction rests on them.

## Computation

Notation: P parameters (dense matrices), N pairs, E evaluation renderings, R training renderings, d code dimension, T updates.

- **Training:** the masked reader multiplies dense matrices, so time is O(T · batch · P) with P the dense parameter count, not only active weights; storage O(P) per live state, with one stored
  end-of-block-4 state (model plus Adam moments, about 3P floats) per arm-seed held in memory. `zero-grad` needs no forward or backward pass, only the optimizer step; `noupdate` costs evaluation only.
- **Evaluation per point:** recall O(N·E·9) forward passes; brute-force 1-NN with N·E queries against an N·R bank: O(N²·E·R·d) distance work, excluding encoding (N·R bank images, N·E queries); the frozen-bank form adds a
  stored bank (N·R·d floats) per branch.
- **Updates per arm-seed:** blocks 1–3: 6,200; block 4: 2,000; branches with training: `ctx`, `hold`, `rehearse-10`, `ctx-enc-frozen`, `ctx-read-frozen` = 5 × 2,000 = 10,000, plus 2,000 for `ctx-nodecay` in `mixed-h125`;
  relearning 3 × 250 = 750. Total about 18,950 (20,950 for `mixed-h125`) against 16,200 in the frozen study, about 1.2×; frozen-group branches skip part of the backward pass. 40 arm-seeds at the measured 89 s effective
  per frozen arm-seed (six GPU processes): about 4,200 s before the added evaluation (15 points per branch with decoders and swaps); expect 1.5–2 h. To be replaced by measurement.
- **Measured timing run** (seed 299, `mixed-h125`, one GPU process, tag `-timing`, not part of the cohort): 321 s for the arm-seed (data 128 s and rendering 14 s once per process); branches 0–30 s each (`ctx` 30 s, `zero-grad` 6 s, `decay-only` 3 s, `noupdate` under 1 s); peak RSS 2.56 GB. Extrapolation for 40 arm-seeds with six processes at the earlier 3.6× effective speed-up: about 1 h. Replaced by the measured study cost in the results.
- Selection and tuning: none beyond the smoke checks and this timing run (frozen parameters unchanged, `noupdate` exactness, `zero-grad` displacement). Timing goes to the runtime file.
- The dense-matrix point also corrects the computation description of the frozen studies, which said "active weights"; those artifacts are not rewritten.

## Risks

- Freezing changes the optimization regime (Adam state of the frozen group stops updating); results are statements about these recipes.
- The 1-NN decoders use true labels; separability, not stored association. Rebuilt-bank scores hide encoder movement; the frozen-bank form shows it.
- The swap test mixes states of different times and can fail through co-adaptation alone.
- Reassigned-count relearning includes reassignment interference; B differs from A in feature-space similarity.
- `ctx` ends near chance, so late-block Δ is bounded; the early points carry most of the information.
- GPU: the composition-measure caveat applies to transfer (a) only.

## Decisions

Command: `NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.forgetting_mechanism pilot pairs` (defaults: the four arms, seeds 330–339, N = 45, L = 2,000, Option A; resumes if interrupted). Relearning lasts 250 updates at evaluation offsets 0, 25, 50, 100, 250; ridge probe λ = 1 on standardized codes.

Decided by the researchers (2026-10-31): four arms (`sparse48`, `dense`, `mixed`, `mixed-h125`); ten fresh seeds (330–339); structured transfer descriptive; one rehearsal dose (`rehearse-10`).
