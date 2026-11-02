# Shared-core sparse memory v1: unseen trace pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, A then B only, with no setting chosen here. Runner `experiments/shared_core_unseen_traces.py`; reports `reports/shared-core-sparse-memory-v1-unseen-traces*.json`.

## Question and setup

The [sensory trace pilot](shared-core-sparse-memory-v1-traces-pilot.md) stored collections the host had trained on. Does re-encoding at read time still work on traces it never saw? Hosts, memory, keys and writes are those of that pilot; only the stored images differ. Sources, by distance from training:

| Source | What is new |
|---|---|
| `new-radius` | Freshly rendered arrangements, in the training rendering |
| `new-area` | Freshly rendered arrangements, in the rendering of the candidates, which the host did not train on |
| `tree`, `tally`, `banana` | Count formats never taught, from the held-out renders |
| `thai_digits` | A script with no quantity in it the host could read, as a negative control |

Gate: with training images this runner reproduces the sensory trace pilot's reading (A at block 6, 0.66, 0.72).

## Result

**Traces hold on unseen images of formats the host knows, and fail on formats it does not.** The limit is what the host can read, not memory. The earlier result was also partly flattered, as suspected, though at the memory level rather than among candidates.

Answers among candidates, Euclidean, abstentions wrong:

| Trace source | One trace: A, block 6 | One trace: B, block 6 | Eight traces: A, block 5 | Eight traces: A, block 6 | Eight traces: B, block 6 |
|---|---|---|---|---|---|
| Training images | 0.66, 0.72 | 0.58, 0.71 | 0.65, 0.74 | 0.67, 0.71 | 0.57, 0.71 |
| `new-radius` | 0.57, 0.68 | 0.56, 0.68 | 0.64, 0.84 | **0.65, 0.78** | **0.62, 0.76** |
| `new-area` | 0.51, 0.64 | 0.53, 0.63 | 0.58, 0.82 | **0.59, 0.76** | **0.60, 0.72** |
| `tree` | 0.23, 0.26 | 0.12, 0.16 | 0.15, 0.13 | 0.16, 0.12 | 0.12, 0.14 |
| `tally` | 0.19, 0.13 | 0.15, 0.16 | 0.12, 0.14 | 0.17, 0.10 | 0.16, 0.12 |
| `banana` | 0.13, 0.13 | 0.13, 0.14 | 0.13, 0.12 | 0.13, 0.12 | 0.14, 0.16 |
| `thai_digits`, control | 0.18, 0.09 | 0.11, 0.07 | 0.12, 0.08 | 0.15, 0.06 | 0.12, 0.05 |

Chance is 0.11. For comparison, the stale copy read 0.28, 0.32 for A and 0.29, 0.36 for B at block 6, and the frozen block-3 copy 0.64, 0.64 and 0.56, 0.62.

Memory level, nearest training-collection prototype, eight traces, and a single trace's own count against those prototypes, at block 6:

| Trace source | A | B | Single trace to its count |
|---|---|---|---|
| Training images | 0.93, 0.92 | 0.88, 0.88 | 0.98, 1.00 |
| `new-radius` | 0.63, 0.66 | 0.70, 0.63 | 0.71, 0.77 |
| `new-area` | 0.52, 0.51 | 0.58, 0.65 | 0.64, 0.73 |
| `tree` | 0.42, 0.35 | 0.32, 0.30 | 0.22, 0.24 |
| `tally` | 0.36, 0.11 | 0.29, 0.12 | 0.18, 0.16 |
| `banana` | 0.18, 0.11 | 0.13, 0.13 | 0.17, 0.14 |
| `thai_digits`, control | 0.17, 0.22 | 0.13, 0.26 | 0.10, 0.10 |

- **Unseen images of known formats pass.** With eight traces, A and B at block 6 are at or above the frozen copy, and A's change across block 5 is +0.04, +0.03 for `new-radius` and 0.00, +0.05 for `new-area`. So nothing falls across B.
- **More traces help on unseen images.** One trace reads 0.51–0.68. On training images one trace was enough, because the host reads them almost perfectly.
- **The earlier memory-level figures were flattered.** The host reads a training image's count at 0.98–1.00, and a fresh one in the same rendering at 0.71–0.77, so it has partly memorized its training images. Memory-level reads on training images overstated what it can do. Among candidates the difference is much smaller, because those are limited by the `area` rendering anyway.
- **Unknown formats give no meaning, which is right.** `tree`, `tally` and `banana` are mostly near chance through the host, and so is the control. `tree` holds a little above chance with one trace, but not with eight. This doesn't lose the fact: a trace can still be read in sensory space. It says the host can only supply a meaning it has learned.

## What this does not establish

- **Senses stay frozen.** Addresses use the frozen encoder.
- **Whether a format could be learned later.** A trace stored before the host learns its format might gain a meaning once it does. That would be a real prediction of this design, and it is untested.
- **The choice of read.** Nothing yet decides whether a query wants a sensory read or a host read.
- **Scale and order.** One load, A then B, two seeds.

## Computation

GTX 1650, one process: 45 s wall, 25 s parent CPU, both seeds, including data preparation and rendering 5,400 new images per seed. No host fit. Storage and read cost are as in the sensory trace pilot: eight traces take about 740 KB at 118 units, and a meaningful read takes eight host passes. So the eight-trace result is the costly one.
