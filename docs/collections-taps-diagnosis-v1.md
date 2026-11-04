# Why relearning lowers the core's collections → taps reading

Status: **diagnosis, 2026-11-04, on existing checkpoints only; nothing was trained.** Code: `experiments/collections_taps_diagnosis.py`. Report: `reports/collections-taps-diagnosis-v1.json`.

**The core:** step 8's calibrated core, `joined-mlp-calibrated`, which is identical to stage 2's `no-replay` (1,540 of 1,540 measures). Seeds 300–303, at block 3, the delay and relearning.

**The question:** memory still holds the right dot values after relearning, so why does ordinary learning make the core worse at interpreting them? The working design is retrieval without replay ([stage 2](sense-memory-replay-v1-pilot.md), closed with a negative result).

## Identical evaluation at every checkpoint

Every checkpoint reads the same inputs, scored with the same functions as the stage 1 and stage 2 reads (`D.codes_for`, `same` relation):
- **Collections → taps, fresh:** the 270 evaluation collections, 30 per count, each against nine taps candidates. The candidates and their order are fixed per query by a seeded draw.
- **Collections → taps, stored:** memory's 711–853 stored dot values from stage 2, as means of their units, labelled by majority count, against the same kind of fixed candidates.
- **Taps → collections:** the 270 evaluation taps sequences against nine fixed collections, reported separately.

Per seed (300, 301, 302, 303):

| Read | Block 3 | Delay | Relearn |
|---|---|---|---|
| Collections → taps, fresh | 0.88, 0.91, 0.89, 0.92 | 0.86, 0.90, 0.84, 0.94 | **0.81, 0.49, 0.67, 0.71** |
| Collections → taps, stored values | 0.95, 0.96, 0.97, 0.97 | 0.88, 0.95, 0.93, 0.99 | **0.85, 0.52, 0.78, 0.76** |
| Taps → collections | 0.94, 0.94, 1.00, 0.99 | 0.99, 0.99, 0.98, 0.99 | 0.98, 0.92, 0.99, 0.97 |

**The decline is confirmed under identical conditions,** on fresh and on stored values alike, in every seed. It ranges from −0.05 (seed 300) to −0.41 (seed 301). The reverse direction barely moves.

## What relearning trains, and how it differs from this evaluation

**Relearning, 200 updates.** Each batch has:
- 16 English form episodes, of which 4 are replaced by rehearsal episodes from earlier scripts;
- 4 context episodes pairing taps with collections.

**Losses:** matching, succession, comparison of collections and taps, and sequence prediction. Adam continues on every parameter group: the image projection, the taps encoder, the core's projection, relation tokens, the comparator readers `g`, the predictor `P` and a scale. No RigL and no decay.

**The delay, 2,000 updates,** has the context episodes only, with no forms.

**Collections → taps is never trained, in any phase.** The core's `same` relation is a learned comparator, `g(cat(a, b, token))`, so the order of its arguments matters. The matching loss always calls it as `g(query, candidate)`:
- the query is a form or a taps sequence;
- the candidates are always nine collections.

The evaluation calls it as `g(collection, taps)`, ranking taps candidates. That differs from training in two ways:
- **the argument order** is reversed;
- **the candidates** are taps, never ranked in training.

Taps → collections is exactly the trained call, and it is the direction that stays stable.

## Localizing the failure

**Swapping the comparator's arguments** recovers most of the loss. This reads the same pairs, at the same checkpoint, as `g(taps, collection)`:

| Collections → taps | Delay | Relearn |
|---|---|---|
| Evaluated order, fresh | 0.86, 0.90, 0.84, 0.94 | 0.81, 0.49, 0.67, 0.71 |
| Trained order, fresh | 0.96, 0.94, 0.91, 0.95 | 0.85, 0.78, 0.87, 0.90 |
| Evaluated order, stored | 0.88, 0.95, 0.93, 0.99 | 0.85, 0.52, 0.78, 0.76 |
| Trained order, stored | 0.99, 0.98, 0.98, 0.99 | 0.93, 0.83, 0.90, 0.97 |

The untrained order accounts for most of the relearning loss: −0.03 to −0.41 in the evaluated order against −0.04 to −0.16 in the trained one. A smaller loss remains even in the trained order, because ranking taps candidates is itself never trained.

**The errors sit at the end of the number line.**
- At relearning, collections of 9 dots are read correctly 0.00–0.07 of the time in every seed. They were already weak at the delay (0.17–0.67).
- In seed 301, counts 1–3 also fail (0.00–0.07): 37% of all answers are 9 taps, so a hub has formed.
- Elsewhere, errors stay within one count (all four seeds at the delay, three at relearning).

**Geometry, within each checkpoint** (codes are never compared across checkpoints):

| | Delay | Relearn |
|---|---|---|
| Collections separable by count (nearest class mean) | 0.97–0.98 | 0.94–0.95 |
| Taps separable by count | 1.00 | 1.00 |
| Collection class mean nearest the same-count taps mean | 0.78–0.89 | 0.22–0.56 |
| Mean distance to other-count taps means, against same-count | 25–34 against 6–11 | 10–13 against 4–8 |

**Collections still separate by count after relearning, and taps perfectly.** What breaks is how the two senses' number lines line up. The code space contracts: distances between different counts shrink to about 40% of their delay value, while same-count distances barely change. Collections then land between taps classes. The trained `g(taps, collection)` order compensates, and the untrained order does not.

**What moved, delay → relearn** (relative change):

| Group | Change |
|---|---|
| Image projection (all images: collections and forms) | 0.26–0.40 |
| Core projection | 0.23–0.29 |
| Scale | 0.28–0.34 |
| Comparator readers `g` | 0.06–0.09 |
| Taps encoder | 0.07–0.08 |
| Predictor `P` | 0.05–0.06 |
| Relation tokens | 0.03–0.04 |

Codes of the same inputs change by 0.72–0.78 for collections and 0.57–0.68 for taps, with cosines of 0.77–0.87 and 0.81–0.96. Relearning's 200 updates move the image projection more than anything else. Re-teaching English forms passes large gradients through the projection that collections share.

**Hybrids, with a caveat.** Codes change by about 70%, so swapping representation groups between checkpoints mixes coordinate systems that are not compatible. Those swaps are not interpretable:
- the delay checkpoint with relearning's image projection reads 0.30–0.41 in both directions;
- with relearning's taps encoder it reads 0.22–0.42.

The groups that do not shape the codes can be read:
- **The predictor `P` and the relation tokens are not the cause.** The delay checkpoint with either taken from relearning still reads 0.84–0.94 collections → taps.
- **Swapping only `g` gives mixed results.** The relearning checkpoint with the delay's `g` reads 0.66–0.83, against 0.49–0.81; the reverse swap reads 0.36–0.92. `g` and the code space co-adapt, so `g` alone neither causes nor cures the loss.

## Diagnosis

Relearning re-teaches English forms through the image projection that collections share. In 200 updates that changes collection codes substantially and contracts the space between counts.

The training signal keeps one direction calibrated: `g(taps, collection)`, through the context episodes' matching. Nothing keeps `g(collection, taps)` calibrated, or the ranking of taps candidates. So the direction memory needs to interpret a recalled collection loses calibration first, worst at the end of the line (9 dots) and, in one seed, as a hub at 9.

Memory's values are unchanged; the interpreter has moved in a direction its training never checks.

This is a diagnosis from checkpoints, not yet a tested cause.

## The smallest discriminating experiment (superseded)

**Superseded (2026-11-04, user):** training the reverse direction is only an optional diagnostic upper bound, not a proposed fix. The question is why the learned relation fails in reverse. That is answered further, and the experiment replaced, by the [same-relation diagnosis](same-relation-diagnosis-v1.md). The original recommendation is kept below.

The delay checkpoints lack optimizer state, so honest continuations need it. Rerun each seed to the end of the delay, deterministically (about 4 minutes; the reproduction is checked against these checkpoints), and save Adam's state. Then branch relearning (200 updates, seconds each), changing one thing at a time:

1. **Baseline relearning:** must reproduce the relearning checkpoint exactly.
2. **Train the evaluated direction:** add `g(collection, taps)` matching to the context episodes' existing pairs, ranking nine taps candidates, at the same weight. This tests the objective gap.
3. **Freeze the image projection for relearning:** collections' codes then cannot move, while forms still train through the core. This tests input-side drift.

**Measured in each branch:**
- collections → taps, fresh and on stored values, in both comparator orders;
- taps → collections;
- English matching, the script being relearned, to check that acquisition continues;
- stage 1's memory-present symbol → taps read.

**What would decide it:**
- If 2 restores collections → taps and English still relearns, the failure is an untrained direction, and the remedy is in the objective.
- If 3 restores it, it is drift of the shared projection.
- If both partly restore it, both act.

No new architecture is involved.

## Computation

CPU, four workers, one per seed: 69 s wall, peak RSS 2.5 GB. Each seed read three checkpoints plus 12 hybrids, with 1,251–1,393 queries per read. No model was trained.
