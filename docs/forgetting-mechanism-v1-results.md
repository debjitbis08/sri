# Forgetting mechanism v1: results

Frozen contract: [forgetting-mechanism-v1-contract.md](forgetting-mechanism-v1-contract.md) (2026-10-31). Fresh seeds 330–339, four arms
(`sparse48`, `dense`, `mixed`, `mixed-h125`), N = 45 pairs, L = 2,000 updates, Option A, GPU. Deterministic results:
`reports/forgetting-mechanism-v1-analysis.json`, from `reports/forgetting-mechanism-v1-results.json.gz`. No arm, seed, branch, endpoint or reading
was changed after the results were seen. Items marked **post hoc** were added afterwards.

Accuracy change D(t) = recall(t) − recall(0) on independent renderings, paired over ten seeds; margin 0.05; intervals are 95% paired t-intervals.
Recall at update 0 (end of block 4): sparse48 0.37, dense 0.54, mixed 0.42, mixed-h125 0.44 (chance 0.111).

## Reading 1: evaluation stability

`noupdate` is exactly constant (recall, training-rendering recall, both 1-NN identity decoders and the probe) in every arm and seed: **passes**.

## Reading 2: optimizer state and decay

**The inherited Adam moments alone lose recall, quickly, and then stop.** With explicit zero gradients and no decay, D settles at 25–50 updates and does not change afterwards:

| arm | zero-grad D(50) | D(250) | D(2,000) | `ctx` D(50) | `ctx` D(250) | `ctx` D(2,000) |
|---|---|---|---|---|---|---|
| sparse48 | −0.093 worse | −0.094 | −0.094 | −0.048 | −0.093 | −0.225 |
| dense | −0.223 worse | −0.224 | −0.224 | −0.108 | −0.124 | −0.340 |
| mixed | −0.131 worse | −0.131 | −0.131 | −0.061 | −0.120 | −0.248 |
| mixed-h125 | −0.131 worse | −0.132 | −0.132 | −0.063 | −0.174 | −0.313 |

Relative weight movement in `zero-grad` is only 0.2–2% per group, yet recall falls by up to 0.22. So inherited momentum can by itself produce an early loss as large as or larger than the early drop in `ctx`: this establishes a vulnerability to optimizer-state movement. It does **not** measure how much of `ctx`'s early drop comes from momentum: during real training new gradients change the trajectory (a zero-gradient step and a gradient step are different perturbations), and gradients from the context material may partly cancel the
momentum step. That needs a direct intervention in real training (see the follow-up). The loss in `ctx` beyond about 250 updates (down to −0.23 to −0.34 at 2,000) is not explained by momentum, which is exhausted.

Explicit decay (`mixed-h125`): `decay-only` D(250) = −0.055 and D(2,000) = −0.086, both **worse** than 0, so decay alone costs recall. Under training, `ctx` − `ctx-nodecay` is inconclusive at 50 and 250 updates (+0.025, +0.008) and **equivalent** at 2,000 (−0.010): decay adds no measurable loss on top of training.

## Reading 3: exposure

Continued pair exposure protects recall. Differences from `ctx` in D (all ≥ the margin where marked better; all arms):

| arm | `hold` − `ctx` at 250 | at 2,000 | `rehearse-10` − `ctx` at 250 | at 2,000 | `hold` vs 0 at 2,000 | `rehearse-10` vs 0 at 2,000 |
|---|---|---|---|---|---|---|
| sparse48 | +0.096 better | +0.341 better | +0.084 better | +0.185 better | +0.116 better | −0.040 inconclusive |
| dense | +0.133 better | +0.389 better | +0.092 better | +0.253 better | +0.049 inconclusive | −0.087 worse |
| mixed | +0.159 better | +0.378 better | +0.135 better | +0.203 better | +0.129 better | −0.045 inconclusive |
| mixed-h125 | +0.200 better | +0.391 better | +0.120 better | +0.241 better | +0.078 better | −0.072 worse |

At the 2,000-update point `hold` keeps recall at or above its starting value in every arm, and rises in three (the pairs keep being learned); one rehearsal every ten updates removes
about three-quarters to four-fifths of the 2,000-update loss (D −0.04 to −0.09 against −0.23 to −0.34) but is worse than no loss in `dense` and `mixed-h125`, and inconclusive in the other two. This shows that continued exposure protects performance. It does not separate stopping exposure from interference that exposure counteracts, and the low-dose protection does not rule out capacity limits.

## Reading 4: which updates are needed

Wording note: the contract's description of reading 4 says "updates of the unfrozen-complement group" for better and for equivalent. Read plainly: better than `ctx` means that **freezing a group reduced the loss**, so updates to the frozen group contribute to the loss;
equivalent would mean freezing it changed nothing, so the other group's updates suffice. The thresholds are unchanged.

| arm | `ctx-enc-frozen` − `ctx` at 2,000 | `ctx-read-frozen` − reference at 2,000 | `ctx-enc-frozen` vs 0 at 2,000 | `ctx-read-frozen` vs 0 at 2,000 |
|---|---|---|---|---|
| sparse48 | +0.041 inconclusive | +0.080 better | −0.184 worse | −0.145 worse |
| dense | +0.094 better | +0.106 better | −0.246 worse | −0.234 worse |
| mixed | +0.057 better | +0.105 better | −0.192 worse | −0.143 worse |
| mixed-h125 | +0.051 better | +0.067 better (vs `ctx-nodecay`) | −0.262 worse | −0.236 worse |

Freezing either group reduces the 2,000-update loss by roughly a quarter to a third in the dense-containing arms (only the reader-side freeze is clearly better in `sparse48`), and neither
prevents it: the frozen branches still end 0.14–0.26 below their starting recall. So the loss does not come from the encoders alone or from the readers and downstream parameters alone;
updates to either group are sufficient for substantial loss, and each group contributes part of it. At 250 updates the reader-side freeze has the clearly larger effect only in `mixed-h125` (+0.097 against +0.026).
These are statements about which updates are needed and sufficient, not about where the association is stored.

## Reading 5: savings

Relearning from the end of `ctx` (250 updates). Early-window (0–100 updates) recall above chance, original assignment minus reassigned counts for the same symbols:
sparse48 +0.117 [0.074, 0.159], dense +0.251 [0.218, 0.284], mixed +0.181 [0.146, 0.216], mixed-h125 +0.129 [0.102, 0.156]: **better in all four arms** — an assignment-specific advantage is detected.
Reassigned minus B (symbol familiarity): +0.009, −0.002, +0.009, −0.005: **equivalent** in all arms, so no familiarity advantage is detected.

**Post hoc** (not a prespecified reading): part of the original-assignment window comes from recall that is still above chance at relearning start (dense 0.199, mixed 0.174, sparse48 0.147). In `mixed-h125`, where recall at the start is at chance (0.130),
the gain over the first 100 updates is still +0.256 for the original assignment against +0.091 (reassigned) and +0.078 (B). Recall after 250 relearning updates: dense 0.548 (original) against 0.345 and 0.340; mixed-h125 0.389 against 0.300 and 0.266. The pairs are not recoverable by looking, but the network relearns them faster than new or reshuffled ones, so some assignment-specific information survives. The test cannot say in what form, and reassignment interference is part of it.

## Reading 6 and descriptive measures

- **Representation moved a lot; separability and linear decodability did not collapse.** From update 0 to 2,000 in `ctx`, 1-NN symbol identity with a **rebuilt** bank falls by 0.04–0.10; with the **frozen** end-of-block-4 bank it falls by 0.35–0.49 (to 0.14–0.28). The code of each symbol drifted far from where it was, while symbols stay separable in the current code. The ridge probe from the current code to the count (fit on the training renderings with the true counts) falls only 0.04–0.05 (dense 0.52 → 0.47) while the network's own recall fell to 0.13–0.20. Counts are still linearly decodable from the current code with labels given; the matching readout no longer uses them.
- **Component swap in `ctx` at 2,000 updates** (recall): dense — end-of-block-4 encoders with current readers 0.450, current encoders with end-of-block-4 readers 0.238 (full `ctx` 0.20, block 4 0.54); mixed 0.336 and 0.206; sparse48 0.224 and 0.197; mixed-h125 0.209 and 0.176. In `dense` and `mixed`, restoring the old encoders recovers a large part of the loss while restoring the old readers does not; in `mixed-h125` and `sparse48` neither does. Co-adaptation can cause mismatch alone, so this is descriptive.
- **Weight movement** (relative L2 from block 4 at 2,000 updates, `ctx`): the sequence encoder moves most (0.22–0.23), then the image encoder (0.12–0.14); first-layer rows and later layers 0.02–0.11. The structure rows of the mixed arms move least (0.03).
- **Structured transfer (a)** at the end of each branch is descriptive; `ctx` shows small changes, with no formal harm reading (absence of a harm finding is not demonstrated preservation): block 4 → `ctx` end: sparse48 0.717 → 0.680, dense 0.617 → 0.634, mixed 0.663 → 0.662, mixed-h125 0.672 → 0.666.
- Loss fractions (descriptive; defined for all ten seeds in every arm since the initial excess exceeds 0.10): `ctx` at 2,000 updates 0.79 (dense), 0.80 (mixed), 0.88 (sparse48), 0.94 (mixed-h125).

## What this shows and does not show

- Inherited Adam state can damage recall without any new gradient: the zero-gradient branch loses 0.09–0.22, then plateaus. Its contribution during actual context training is not measured here. The slow loss to the floor needs training on other material; it is partly reduced by freezing either encoders or readers but not removed by either, which points to an interaction between them, not one faulty component. For `mixed-h125` the encoder-frozen branch includes reader updates plus explicit decay, so it does not isolate reader updates alone.
- Some assignment-specific information survives. Recognition survives, counts stay linearly decodable with labels supplied, and relearning the original assignment is faster than learning reassigned counts for the same symbols. That does not establish that every association, or its original form, is preserved; the label-supplied probe shows recoverability with supervision, not that the original mapping remains intact. The original-assignment window includes its starting advantage; the gain analysis that corrects for it is post hoc corroboration.
- A supported hypothesis, not a complete explanation: the matching readout fails after the item code drifted from where the readers learned it (frozen-bank collapse; encoder swaps in `dense` and `mixed`). Reader-only updates also cause substantial forgetting, and restoring old encoders helps much less in `sparse48` and `mixed-h125`, so the mechanism need not be the same in every arm.
- Rehearsal at one in ten updates removes most of the loss; full-rate exposure removes it. This is not evidence against capacity or competition effects.
- Limits: one fixed cohort; one set of 45 pairs per seed; four arms; the `zero-grad` and frozen branches change the optimization regime; swap tests and the probe use labels or mix times and are descriptive; these results do not say whether the pattern holds for pairs with structure, or for a different optimizer (Adam only here).
- Not claimed: that momentum is the cause of all forgetting, that the association is stored in any given group, or any ranking of arms (arm differences are descriptive).

## Computation (measured costs)

Theoretical: training per branch O(updates · batch · P) with P the dense parameter count; evaluation per point O(N·E·9) forward passes plus O(N²·E·R·d) for the 1-NN decoders; storage O(P) per live state.
Measured (GPU GTX 1650, six worker processes, 40 arm-seeds, ten per arm): task runs 4,174 s wall (data 28 s, rendering 3 s), 4,206 s in total; peak RSS 2.61 GB per worker. The single arm-seed timing run (seed 299) took 321 s alone; branches took 0–30 s each (`ctx` 30 s, `zero-grad` 6 s, `decay-only` 3 s, `noupdate` under 1 s). Selection and tuning: none beyond the smoke checks. The analysis takes under a second. Timings are in the `-runtime.json` files.
