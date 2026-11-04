# Is the core's learned `same` relation symmetric and discriminative, and why does relearning break it?

Status: **diagnosis, 2026-11-04, on existing checkpoints only; nothing was trained.** It follows the [collections → taps diagnosis](collections-taps-diagnosis-v1.md). Code: `experiments/same_relation_diagnosis.py`. Report: `reports/same-relation-diagnosis-v1.json`.

**The core:** step 8's calibrated core (identical to stage 2's `no-replay`), seeds 300–303, at block 3, the delay and relearning.

**The question (user, 2026-11-04):** why does the core's learned relation fail to transfer, or stay valid, in reverse? Training the failing direction directly is kept only as an optional diagnostic upper bound, not as a fix.

## What the relation is, and what its training constrains

`same` is a learned comparator, `g(a, b) = g(cat(a, b, token))`, shared with the succession and comparison heads through the token. It is trained only through the matching loss: a 9-way softmax with the query in the first slot (a form or a taps sequence) and nine collections in the second.

That loss constrains only score differences among collections for a fixed query. It leaves three things free:
- **Symmetry:** nothing ties `g(a, b)` to `g(b, a)`.
- **A per-query offset:** adding any f(a) to every score of query a changes nothing. Taps candidates scored in the first slot can therefore carry offsets that make them win for every query, as hubs.
- **The second slot for anything but a collection:** taps are never ranked there.

So the relation's behaviour in reverse is whatever falls out of its trained behaviour, with no pressure of its own.

## Measurements, on identical pairs at every checkpoint

The inputs are the 270 evaluation collections, the 270 evaluation taps sequences and the English evaluation forms, 30 per count. Every collection–taps pair is scored in both orders. Rankings use nine candidates, one per count, with 20 fixed draws per query.

Values are per seed (300, 301, 302, 303).

**Pair symmetry.** Differences are in units of the scores' spread:

| | Block 3 | Delay | Relearn |
|---|---|---|---|
| Correlation of g(c, t) with g(t, c) | 0.34, 0.80, 0.80, 0.62 | 0.35, 0.80, 0.88, 0.38 | 0.48, 0.35, 0.91, −0.10 |
| Mean \|g(c, t) − g(t, c)\|, same-count pairs | 0.08, 0.19, 0.13, 0.04 | 0.14, 0.08, 0.11, 0.06 | 0.16, 0.20, 0.33, 0.10 |
| Mean \|g(c, t) − g(t, c)\|, nearer half by code distance | 0.30, 0.39, 0.29, 0.19 | 0.20, 0.28, 0.22, 0.18 | 0.36, 0.54, 0.22, 0.44 |
| Mean \|g(c, t) − g(t, c)\|, farther half | 1.32, 0.62, 0.70, 1.03 | 1.33, 0.72, 0.57, 1.39 | 1.18, 1.33, 0.47, 1.80 |
| Correlation of the violation with code distance | 0.92, 0.28, 0.37, 0.87 | 0.94, 0.67, 0.60, 0.96 | 0.76, 0.61, 0.35, 0.89 |

**Pair discrimination,** threshold-free: AUROC of same-count against different-count pairs:

| | Block 3 | Delay | Relearn |
|---|---|---|---|
| g(c, t), the evaluated order | 0.96, 0.94, 0.98, 0.99 | 0.97, 0.99, 0.98, 1.00 | 0.94, 0.90, 0.93, 0.97 |
| g(t, c), the trained order | 0.97, 0.96, 0.99, 1.00 | 0.99, 0.99, 0.99, 1.00 | 0.97, 0.96, 0.96, 0.99 |
| g(f, c), English forms (trained) | 0.58, 0.51, 0.55, 0.50 | 0.61, 0.51, 0.58, 0.50 | 0.84, 0.86, 0.87, 0.51 |
| Code distance only, no comparator | 0.87, 0.93, 0.94, 0.93 | 0.86, 0.90, 0.93, 0.94 | 0.77, 0.75, 0.85, 0.75 |

**Ranking, raw and with candidate offsets removed.** "Centred" subtracts each candidate's mean score over all queries, which cancels a per-candidate offset and leaves pair-level discrimination:

| Ranking | Delay, raw | Delay, centred | Relearn, raw | Relearn, centred |
|---|---|---|---|---|
| Collection → taps by g(c, t) (the evaluated read) | 0.88, 0.90, 0.84, 0.93 | 0.93, 0.90, 0.87, 0.94 | 0.82, 0.49, 0.67, 0.70 | 0.86, 0.77, 0.72, 0.90 |
| Collection → taps by g(t, c) | 0.96, 0.94, 0.91, 0.94 | 0.87, 0.95, 0.94, 0.91 | 0.86, 0.78, 0.87, 0.90 | 0.94, 0.91, 0.94, 0.74 |
| Taps → collection by g(t, c) (trained) | 0.98, 0.97, 0.97, 0.98 | 0.97, 0.97, 0.96, 0.93 | 0.97, 0.94, 0.98, 0.96 | 0.96, 0.95, 0.97, 0.85 |
| Taps → collection by g(c, t) | 0.92, 0.94, 0.88, 0.96 | 0.73, 0.92, 0.80, 0.95 | 0.91, 0.70, 0.85, 0.78 | 0.85, 0.77, 0.77, 0.87 |

**Geometry within each checkpoint** (codes are never compared across checkpoints):

| | Delay | Relearn |
|---|---|---|
| Mean code norm, collections / taps | 15.7–21.8 / 17.0–23.4 | 5.1–6.1 / 7.2–10.6 |
| Spacing between count classes, collections / taps | 23.8–32.0 / 26.5–35.3 | 7.1–8.6 / 11.1–14.4 |
| Collection spread within a class, over spacing | 0.05–0.06 | 0.08–0.12 |
| Same-count collection–taps distance, over collection spacing | 0.24–0.33 | 0.47–0.93 |
| Direction of the count line, collections against taps (\|cos\| of first principal axes of class means) | 0.90–0.95 | 0.75–0.91 |
| English form class means nearest the same-count collections | 0.11–0.22 | 0.44–0.67 (seed 303: 0.11) |

From block 3 to the delay, collection spacing grows (18–26 to 24–32). It shrinks only during relearning.

## What this shows

**Correction (2026-11-04, user).** An earlier version of the next point said reverse validity was "not a learned property". That went too far. The comparator does not enforce symmetry, and reverse validity was expected to emerge through learning. The alignment between the senses, which it goes together with, was itself learned. The evidence shows that reverse validity was fragile and associated with that learned alignment. It does not yet establish how it arose or why it broke.

- **The comparator does not enforce symmetry, and the learned relation is only approximately symmetric where cross-sense codes overlap.**
  - Before relearning, same-count pairs score nearly the same in both orders (0.04–0.19 spread units), while far-apart pairs differ by 0.6–1.4.
  - The violation grows with the distance between the two codes (correlation 0.28–0.96).
  - So the reverse validity that emerged is fragile: it is associated with the learned alignment that brings collections and taps of a count close together, and it weakens where that alignment is weaker. How it arose is not established.
- **The trained use of the relation survives relearning.** Taps → collection by g(t, c) stays at 0.94–0.98, and its pair AUROC at 0.96–0.99. Continued training keeps exactly what it trains.
- **Relearning breaks the overlap that reverse validity depended on.**
  - Re-teaching English forms shrinks the code space about threefold in 200 updates.
  - Collections shrink more than taps: their spacing falls to about 60% of taps', against near-equal at the delay.
  - The two senses' count lines rotate apart, and same-count collection–taps distance roughly doubles relative to the spacing between counts.
  - Collections still separate by count, but no longer sit on the taps' line.
- **The evaluated read fails through two separable faults.**
  - **Ranking offsets, the larger part in three of four seeds.** Removing per-candidate offsets lifts collection → taps at relearning from 0.49–0.82 to 0.72–0.90, most in seed 301 (0.49 to 0.77). Some taps candidates score high against every collection in the first slot: these are the hubs, including 9 taps. The training loss does not constrain such offsets.
  - **Pair-level discrimination, smaller.** Pair AUROC in the evaluated order falls from 0.97–1.00 to 0.90–0.97, and centred ranking remains below the delay's (0.72–0.90 against 0.87–0.94). The pairs themselves are less separable once the senses no longer overlap.
- **Seed 303 shows the damage does not need successful form learning.** Its English forms stay far from collections and are not relearned (AUROC 0.51), yet its codes shrink just the same, and its symmetry collapses (correlation −0.10). The damage comes from the relearning updates, not from where the forms end up.

**Why does learning new forms do this, despite continued training on the original relation?** The checkpoints show what changed, not the dynamics that changed it. The evidence points to the shared image projection and the core's projection (relative changes 0.26–0.40 and 0.23–0.29), which shrank collections' codes far more than taps'.

The taps–collection matching that continued throughout keeps only the trained comparison valid. It is invariant to everything that made the reverse work: the senses' relative scale, their alignment, and per-candidate offsets. One plausible route is that the forgotten forms' large errors are reduced by shrinking the code space, which softens confident wrong matches. Checkpoints cannot test it.

## The smallest discriminating experiment (recommended, not run)

Rerun each seed to the end of the delay, deterministically, about 4 minutes per seed, checked against these checkpoints. Save the optimizer state. Then branch the 200 relearning updates, one change at a time. In each branch, record every 10 updates:
- this note's symmetry, AUROC, raw and centred ranking, and geometry;
- English matching, for acquisition of the relearned script;
- stage 1's memory-present symbol → taps read.

The branches:
1. **Relearning as run.** It must reproduce the relearning checkpoint.
2. **No new forms:** the delay's material continued for 200 updates. Does any continued training do this, or does re-teaching forms?
3. **Forms with the image projection frozen:** does the damage go through the projection that forms and collections share? English must then be acquired through the core alone, which is measured.
4. **Deferred (user, 2026-11-04): explicit reverse-direction training,** as a diagnostic upper bound only. Branches 1–3 investigate why learned reverse validity breaks without building the desired answer into training.

**Interpreting the outcome.** No outcome is mapped to a single cause in advance:
- If branch 2 keeps the relation, the decline depends on introducing form-relearning updates.
- If branch 3 helps, updates of the shared image projection contribute, not necessarily as the sole cause.
- If branch 3 does not help, other changing components (the core's projection, the readers, the taps encoder) and their interactions remain possible. Nothing is concluded about the core's projection alone.
- The time course shows the order in which the measured properties change.

Branches 1–3 were run: [results](relearning-branches-v1.md).
