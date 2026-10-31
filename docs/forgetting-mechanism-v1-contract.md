# Forgetting mechanism v1: contract (DRAFT, not frozen, not run)

Status: **draft 2026-10-31**. Follows [arbitrary pairings v1](arbitrary-pairings-v1-results.md), where recall of the 45 taught pairs
fell to near chance (0.12–0.15 against 0.111) by the end of block 5 in every arm, with no interference from B. Before any
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
- **Not capacity or B.** `P-A` (no B) falls as far as `P-AB`; the seed-299 pilots at N = 18, 45, 90 (8 renderings, one seed) all
  reached chance at block 5.

## Question

Why do the pairs fall to chance? Candidate causes, which the branches below separate:

1. **Cessation:** recall depends on pair items being in the recent batches; it drops when they stop (the fast early drop).
2. **Context interference:** training on the other material overwrites the association (slow decline).
3. **Drift independent of content:** weight change from the optimizer (Adam state, learning rate) erodes it whatever the material.
4. **Lost or hidden:** the association is erased from the weights, or only unreadable (relearning savings).

## Design

Common start: the end of block 4 of the arbitrary-pairings schedule, Option A (connectivity fixed during pairing), arms
`sparse48`, `dense`, `mixed`, `mixed-h125`, on **fresh seeds 330–339**, N = 45, L = 2,000, 64 training renderings, GPU, identical
symbol sets and count assignments across arms within a seed (all as in the frozen contract; code reused).
`mixed-h500` and `mixed-reset` add little to a mechanism question and are left out; this is stated, not hidden.

Branches from the end of block 4 (each 2,000 updates, same context episodes as the frozen block 5):

| Branch | Block-5 material | Separates |
|---|---|---|
| `ctx` | context episodes only (= frozen `P-A`) | reference |
| `hold` | context + A pairs as in block 4 | is block 4 at a stable plateau; cause 1 |
| `rehearse-10` | context + A pair items on every 10th update | cause 1 vs 2, dose of rehearsal |
| `rehearse-50` | context + A pair items on every 50th update | the same, at a lower dose |
| `pairs-only` | A pair items, no context | cause 2 vs 3 (no other material to interfere) |
| `ctx-lr0.1` | `ctx` with learning rate × 0.1 for the whole block | cause 3: dependence on step size |

`pairs-only` needs a batch with pair items alone; whether the training step supports that is to be checked in a smoke run before
freezing, and if not the branch is replaced by a `ctx` batch whose context content is cut to a fixed small fraction (decision 2).

Relearning, from the end of `ctx` (2,000 updates in): 250 updates of **A pairs** (block-4 rate), and for the control 250 updates of
**B pairs**, never seen in this branch; both evaluated at 0, 25, 50, 100, 250. Savings = early-window recall (0–100, trapezoid mean
above chance) on A minus the same on B. It measures how much of the association survives in a form that learning can use again.

Evaluation points in every branch: update 0, 10, 25, 50, 100, 150, 200, 250, then every 250 to 2,000 (the first 25 updates carry
the fast drop and had only the 25-update point).

## Measures

- **Recall on A**, independent and training renderings (as in the frozen study), every point; shuffled-label rescoring only as a check.
- **Decoders on the item code** at each point: 1-NN symbol identity, 1-NN count (labelled bank). Descriptive: recognition against association.
- **Weight drift** at each point: relative L2 change from the end-of-block-4 values, separately for the image encoder, sequence
  encoder, structure readers and memory rows. Descriptive; relation to recall by within-seed correlation over points.
- **Excess recall** = recall − chance; **loss fraction at t** = 1 − excess(t) / excess(0).
- Structured transfer (a): at the end of each branch, for the damage reading below.

## Readings (prespecified; margin 0.05; the earlier rule: better/worse if |mean| > margin and the paired 95% t-interval excludes 0; equivalent if the interval lies within ±margin; noninferior if the lower bound exceeds −0.05; otherwise inconclusive). Per arm over ten seeds; arms are not pooled.

1. **Plateau.** `hold` at 2,000 against its own update 0: equivalent → block 4 had reached a stable plateau, and the drop in `ctx` is caused by what changes at the boundary; worse → block 4 was not stable under its own recipe.
2. **Cessation (cause 1).** Recall at 50 and at 250 updates: `hold` − `ctx`, better → stopping pair exposure drives the early drop.
3. **Rehearsal.** Recall at 2,000: `rehearse-10` − `ctx` and `rehearse-50` − `ctx`; and each against `hold` for noninferiority. A low-rate protective effect → pair exposure, not capacity, sets retention.
4. **Context interference against drift (2 vs 3).** Recall at 2,000: `pairs-only` − `ctx`. Better and noninferior to `hold` → the other material is what erases the pairs. Equivalent to `ctx` (still falls) → drift independent of material.
5. **Step size.** Loss fraction at 250: `ctx-lr0.1` − `ctx`; better → forgetting depends on the step size.
6. **Savings (cause 4).** Savings > margin (interval excludes 0) → the association is retained in usable form; equivalent to 0 → erased as far as this measure sees.
7. **Recognition.** 1-NN identity at the end of `ctx` minus at update 0: equivalent → recognition retained while the association goes.

Anything else (drift correlations, per-part drift, arm differences) is descriptive. One fixed cohort; no stopping rule; inconclusive stays inconclusive.

## Prior work (to complete before freezing; per `docs/literature-review.md`)

Corresponds to: catastrophic interference in sequential learning (McCloskey and Cohen 1989; French 1999, **[familiar]**); savings in relearning
as a test for retained structure (Hinton and Plaut 1987, **[abstract]** in the earlier survey; Ebbinghaus savings, **[familiar]**); rehearsal and
pseudo-rehearsal (Robins 1995, **[familiar]**); weight-protection methods (Kirkpatrick et al. 2017, **[familiar]**) as the non-rehearsal
alternative; complementary learning systems (McClelland et al. 1995, **[familiar]**) as the framing for fast memory plus consolidation. Cheaper
alternatives: the reanalysis above (done); one arm (`dense`) only. All **[familiar]** items must be read and recorded in
[prior-work.md](prior-work.md) before the freeze; theory to check includes the dependence of forgetting on learning rate and on overlap of
tasks in feature space, which predict readings 4 and 5.

## Computation

Notation: P parameters, W weights per part, K evaluation points, B branches.

- **Training:** time O(updates × batch × active weights), storage O(P) per live state; branches start from one stored end-of-block-4 state per
  arm-seed (about 3P floats for model and Adam moments, held in memory, no checkpoint files).
- **Updates per arm-seed:** blocks 1–3: 6,200; block 4: 2,000; branches: 6 × 2,000 = 12,000; relearning 2 × 250 = 500. Total 20,700 against 16,200 in the frozen
  study (1.3×). 4 arms × 10 seeds = 40 arm-seeds. At the measured 89 s effective per 16,200-update arm-seed (six GPU processes), the extrapolation is 40 × 89 × 1.3 ≈ 4,600 s,
  about 1.3 h, before the added evaluation points (twelve per branch against fourteen before, with decoders and drift added; expect up to 1.5–2 h). Replaced by measurement.
- **Evaluation:** per point O(N·E·9) forward passes for recall, O(N·R·dim) for the 1-NN decoders (R = 64 training renderings), O(P) for drift.
- Selection and tuning: none beyond the smoke run for `pairs-only`; the seed-299 sizing cost (1.0 h plus diagnostics) belongs to the arbitrary-pairings study.
- Timing goes to the runtime file, apart from the scientific results.

## Risks

- `pairs-only` and `ctx-lr0.1` change the optimization regime, not only the material; a result is a statement about that recipe.
- The 1-NN decoder uses true labels; read it as separability, never as stored association.
- The relearning control assumes B is neutral in `ctx`; it was never trained there, but it may be similar to A in feature space.
- Floor effects: `ctx` ends at chance, so loss fractions at 2,000 are bounded; the early points carry the information.
- GPU: composition-measure caveat from the earlier studies applies to transfer (a) only.

## Decisions for the researchers

1. Arms: the four above (`mixed-h500`, `mixed-reset` left out), or all six at 1.5× the cost.
2. If `pairs-only` is not supported by the training step: replace it with a reduced-context branch, or skip it and read cause 2 against 3 only through `ctx-lr0.1`.
3. Ten fresh seeds (330–339), or five for a cheaper first look (larger intervals).
4. Whether to add structured-damage readings (transfer (a) under each branch) or keep this study to pairs alone; the draft keeps it descriptive.
