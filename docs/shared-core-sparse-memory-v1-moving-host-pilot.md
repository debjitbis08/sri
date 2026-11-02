# Shared-core sparse memory v1: moving host pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, A then B only, setting locked earlier on these seeds. Runner `experiments/shared_core_moving_values.py`; reports `reports/shared-core-sparse-memory-v1-moving-values*.json`.

## Question and setup

Does A survive B while the host keeps learning? This replays the long-term memory host's cached stream through local memory: block 4 teaches A, block 5 teaches B, block 6 has no pairs, and the host learns throughout. Memory is `sense-only`, H 128, k 1, τ 0.85, on the cached frozen-encoder keys.

Keys cannot drift, and addressing is identical in every value condition (checked: addresses and counts are equal). So every difference below comes from values and reading.

Gates: cached targets match the stream, cached keys equal the stage 1 keys, the host's block-3 state is the stage 0 `sparse48`, and the frozen reference reproduces the [stage 1 pilot](shared-core-sparse-memory-v1-stage1-pilot.md) exactly (A after A 0.956, 0.951; A after both 0.956, 0.907).

## Result

**The memory does not forget; its stored values go stale.** With moving values, answers fall by half to the end of block 6, including during block 6, when memory receives no writes and its state cannot change.

Answers among candidates, Euclidean, abstentions wrong:

| Values, reading | A, after block 4 | A, after block 5 | A, after block 6 | B, after block 5 | B, after block 6 |
|---|---|---|---|---|---|
| Live when written, read with current codes | 0.48, 0.45 | 0.39, 0.52 | **0.28, 0.32** | 0.51, 0.70 | **0.29, 0.36** |
| Live when written, read with block-3 codes | 0.34, 0.46 | 0.36, 0.43 | 0.36, 0.43 | 0.29, 0.35 | 0.29, 0.35 |
| Block-3 values, read with current codes | 0.28, 0.27 | 0.21, 0.28 | 0.20, 0.22 | 0.19, 0.24 | 0.16, 0.19 |
| Block-3 values, read with block-3 codes | 0.63, 0.69 | 0.64, 0.64 | 0.64, 0.64 | 0.56, 0.62 | 0.56, 0.62 |
| Host's moving average, half-life 2,000, both | 0.30, 0.37 | 0.20, 0.23 | 0.21, 0.18 | 0.37, 0.39 | 0.24, 0.25 |
| Frozen encoder, both | 0.48, 0.47 | 0.48, 0.46 | 0.48, 0.46 | 0.43, 0.43 | 0.43, 0.43 |

For comparison, at block 6 the readers alone hold A at 0.09, 0.09 and B at 0.20, 0.19. Perfect contents on the current codes read 0.59–0.89 at every checkpoint, so reading itself has not broken.

Memory level, nearest prototype of training collections from the same checkpoint. The cache has no training collections in block 6, so the moving case has no reading there:

| Values | A, after block 4 | A, after block 5 | B, after block 5 |
|---|---|---|---|
| Live when written | 0.64, 0.56 | 0.64, 0.28 | 0.66, 0.90 |
| Block-3 values, block-3 codes | 0.96, 0.95 | 0.96, 0.91 | 0.88, 0.88 |
| Frozen encoder | 0.96, 0.97 | 0.96, 0.95 | 0.88, 0.88 |

- **Writing B costs little in itself.** With values that hold still, A's loss across block 5 is 0.00, 0.04.
- **Drift costs the rest, and teaching B may drive it.** Host prototypes lose cosine with block 3 to 0.89–0.90 by block 5. The host is itself learning B in that block, so B could still harm A through the host. No context-only control ran to separate that.
- **Stale from the start.** A value is an average over the block it was written in, so it lags the code even at the block's end: 0.64, 0.56 at the memory level against 0.96 with stable values.
- **Averaging does not rescue it.** The host's moving average is worse than live values on most reads. It was already weak in the long-term pilot.
- **Stable values work, but they cut memory off from the core.** Frozen-encoder values and keys lose at most 0.02, and the host cannot affect them at all. Their answers are limited, at 0.43–0.48. This shows memory can run beside a learning host, not that it can use one.

So goal 3, keeping this working while the host learns, is **not met** for host-coded values. Reading through a stored block-3 copy of the host keeps 0.56–0.64, and that copy is frozen and must be charged.

## Correction to the value diagnostic

The [value diagnostic](shared-core-sparse-memory-v1-stage1-pilot.md#value-diagnostic) said answers are limited by reading single candidates, and implied rendering noise. Most of that gap is a systematic shift. Training collections use `radius` rendering and evaluation collections use `area` rendering, and the host places `area` collections lower on the count line at high counts. On the line at block 3, the shift between their prototypes is 1.6–3.5 at counts 6–9, against neighbouring gaps of about 3 and a within-count spread of 0.5–1.0. Nearest training prototype for each evaluation prototype is correct on 6 and 7 of nine counts.

So the memory level's 1.00 means the right stored value returned, against prototypes from the same training rendering as the values. It is not a claim about reading new renderings. Against evaluation-collection prototypes the frozen reference reads 0.56–0.67. The cap of about 0.70 at the answer level applies to these `area` candidates. Whether candidates from the training rendering lift it is untested.

## Computation

CPU, one process, float32: 20.5 s wall, 21.4 s CPU, for both seeds, about 640,000 writes across six memories per seed, loading cached codes included. No new host fit. The cached host's cost is in the long-term memory reports. Memory bytes as in the stage 1 pilot, 32-d contents add 16 floats per unit for frozen-encoder values.
