# Shared-core sparse memory v1: sensory trace pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, count pairs A then B only, memory setting locked earlier on these seeds. Runner `experiments/shared_core_traces.py`; reports `reports/shared-core-sparse-memory-v1-traces*.json`.

## Question and setup

The [refresh pilot](shared-core-sparse-memory-v1-refresh-pilot.md) pointed towards a table of live concepts. The user objected on two grounds: such a table is the code space and readers again, and not everything has a meaning to point to. So memory stores what was experienced, a sensory trace, rather than a host code. The particular is read in sensory space. Meaning is computed at read time, by passing the trace through the host as it is then.

Memory is `sense-only`, H 128, k 1, τ 0.85, on frozen-encoder keys, with the same writes as the moving host pilot. Hosts were rerun through blocks 1–6 with weights kept at the end of blocks 4, 5 and 6.

Gates: each host's state hash equals the cached run's at blocks 3, 4, 5 and 6 on both seeds, and cached targets match the stream.

## Result

**Traces re-encoded through the current host hold through the moving host, at the ceiling for these candidates.** This passes the stated test: A and B at block 6 are at or above the frozen copy, and A's loss across block 5 is at most 0.05. Stale copies, on the same reads, are at about half that.

Answers among candidates, Euclidean, abstentions wrong, whole query sets, so directly comparable with the moving host pilot:

| Value | A, after 4 | A, after 5 | A, after 6 | B, after 5 | B, after 6 |
|---|---|---|---|---|---|
| Stale copy of the host's code | 0.48, 0.45 | 0.39, 0.52 | 0.28, 0.32 | 0.51, 0.70 | 0.29, 0.36 |
| **First trace, re-encoded** | 0.66, 0.68 | 0.65, 0.72 | **0.66, 0.72** | 0.54, 0.68 | **0.58, 0.71** |
| **First 8 traces, re-encoded** | 0.67, 0.72 | 0.65, 0.74 | **0.67, 0.71** | 0.55, 0.67 | **0.57, 0.71** |
| All traces, re-encoded | 0.67, 0.69 | 0.66, 0.68 | 0.64, 0.63 | 0.58, 0.64 | 0.58, 0.68 |
| Sensory read: first trace | 0.36, 0.37 | 0.36, 0.37 | 0.36, 0.37 | 0.32, 0.35 | 0.32, 0.35 |
| Sensory read: all traces | 0.48, 0.47 | 0.48, 0.46 | 0.48, 0.46 | 0.43, 0.43 | 0.43, 0.43 |
| Perfect contents on current codes, ceiling | 0.69, 0.73 | 0.69, 0.77 | 0.67, 0.72 | 0.64, 0.76 | 0.64, 0.81 |

For comparison, the frozen block-3 copy read 0.64, 0.64 for A and 0.56, 0.62 for B at block 6, and the readers alone 0.09, 0.09 and 0.20, 0.19.

Memory level, nearest prototype of training collections at that checkpoint. This cache now has block 6:

| Value | A, after 4 | A, after 5 | A, after 6 | B, after 5 | B, after 6 |
|---|---|---|---|---|---|
| Stale copy | 0.64, 0.66 | 0.49, 0.19 | 0.56, 0.39 | 0.64, 0.80 | 0.66, 0.48 |
| First trace, re-encoded | 0.92, 0.72 | 0.96, 0.80 | 0.96, 0.67 | 0.89, 0.68 | 0.83, 0.71 |
| First 8 traces, re-encoded | 0.93, 0.90 | 0.93, 0.90 | 0.93, 0.92 | 0.88, 0.86 | 0.88, 0.88 |

- **Drift stops mattering for stored facts.** Host prototypes still fall to a cosine of 0.88–0.91 with block 3, but no stored code is left to go stale.
- **The forgetting goal holds at the answer level with a learning host.** A's change across block 5 is −0.01, +0.04 with one trace, and −0.02, +0.02 with eight. This also closes the question left by the moving host pilot, whether B harms A through host drift: on these reads it does not. No context-only control ran, and it is no longer needed for this question.
- **One trace is enough for answers, and eight are steadier for memory.** On seed 393, one trace reads 0.67–0.80 at the memory level, likely because units shared by symbols keep only the first symbol's trace. Eight traces read 0.86–0.92. This cause was not measured.
- **Meaning needs the host.** The same traces read directly in sensory space give 0.32–0.37. Averaging them gives 0.43–0.48.

## Facts with no meaning

Symbol to symbol: 45 cues to 45 target symbols in each of two disjoint sets, arbitrary, no counts. Both orders, nine rendered candidates, chance 0.11, cosine, on the frozen encoder. Pairs here are A then B, B then A. Cue and target symbols are all from the unseen pool.

| Seed, value | Old after old | Old after both | New after both |
|---|---|---|---|
| 392, first trace | 0.99, 0.90 | 0.99, 0.90 | 0.90, 0.93 |
| 392, mean of traces | 1.00, 0.96 | 1.00, 0.96 | 0.96, 1.00 |
| 393, first trace | 0.99, 0.91 | 0.99, 0.91 | 0.83, 0.90 |
| 393, mean of traces | 1.00, 0.97 | 0.98, 0.95 | 0.93, 0.95 |

Worst loss is 0.03, and none on first traces. These facts need no concept and no host. Read through the block-6 host instead, on first traces, they give 0.39–0.52: above chance, but much lower. So particular facts should be read in sensory space.

## What this does not establish

- **Traces here are training images.** The host has trained on these collection renderings. A trace of something it has not seen has not been tested. This matters more for the claim than any other caveat here.
- **Senses are frozen.** Keys, and the sensory read, are on the frozen encoder. No memory unit is frozen, but a sensory encoder that kept learning would bring staleness back to the keys. The host's own encoder identifies symbols only 0.26–0.27.
- **How it decides which read to use.** Nothing here chooses between the sensory read and the host read. Each was read separately.
- **The cap remains.** Answers reach the ceiling for `area` candidates; the rendering shift still limits it.
- **Order and scale.** Count pairs ran A then B only, on one load. Cross-sense recall was not tested.

## Computation

Host runs, GTX 1650, two workers: 239 s wall, 413 s aggregate worker CPU, 12,000 updates each, 84 MB GPU peak. Reads, GPU, one process: 53 s wall, 30 s parent CPU, for both seeds and including data preparation; not an inference benchmark.

Storage at one byte per pixel, per unit: 784 bytes for one trace and 6,272 for eight, as against 68 bytes in the host-code memory's 16-d contents. At 118 units that is about 93 KB or 740 KB, beside the same addresses. A meaningful read costs 1 or 8 host forward passes, which is the new cost on a CPU. A sensory read costs none. Compact traces were not tried: a frozen-encoder code cannot go through the host, since the host reads pixels.
