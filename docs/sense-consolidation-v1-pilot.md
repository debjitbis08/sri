# Consolidation alongside address rebuilding: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast, and not a contract.** Two seeds. Merge thresholds, the content gate, per-unit radius and its floor were all chosen during the study, the radius after seeing the probe. Runner `experiments/sense_consolidation.py`; merge in `src/sparse_local_memory.py`; reports `reports/sense-consolidation-v1-*.json`. Follows [step 3 of the sense-level pilot](sense-relational-v1-pilot.md), where unit growth became the main limit.

## Question

The user's distinction (2026-11-02): rebuilding asks where a stored experience lies under the current encoder. Consolidation asks whether units now represent the same thing, and whether they can be combined. So they are separate steps, tested as a 2×2, rebuilding first.

## Mechanism

No mechanism sees a symbol label. True symbols and counts only score merges and define the reference below.

- **Rebuild:** a unit's address becomes the mean direction of its eight kept traces under the current encoder.
- **Merge:** reciprocal nearest neighbours, in passes until none qualify. A pair merges when each is the other's nearest eligible unit, addresses reach `tau_m`, and mean contents reach a cosine of 0.8. Statistics add, so the result is what one unit taking both units' writes would hold. Kept traces are capped at eight, taken in turn from each side.
- **Radius:** a unit's threshold for reads and writes becomes the lowest similarity of its kept traces to its address, clipped to 0.5–0.85.

Values and reading stay on block-3 host codes, so the host is not moving. Encoder updates sample the curated background set.

## Probe on frozen keys

H 2048, A then B, one merge after each block. Before merging, about 945 units:

| Pairs of units | 1st | 10th | Median | 90th | 99th percentile |
|---|---|---|---|---|---|
| Same symbol, address cosine (seeds' range) | -0.09, -0.01 | 0.21, 0.28 | 0.60, 0.63 | 0.83, 0.84 | 0.90, 0.91 |
| Other symbol, address cosine | -0.39, -0.41 | -0.21, -0.23 | 0.02, 0.00 | 0.29, 0.28 | 0.58, 0.60 |

- **A symbol's units are not near-duplicates.** Their median cosine is about 0.6, so most could not merge at any threshold that spares other symbols. At `tau_m` 0.85, 17–18% of the qualifying pairs join different symbols (90 of 498, 93 of 543).
- **The content gate helps, within a limit.** At `tau_m` 0.85, a gate of 0.8 cuts wrong-symbol merges by about half or more (30 to 13, 38 to 15), but contents here have only nine values, so it cannot separate symbols sharing a count.

The label-using reference shows what the threshold costs. Merging only within a true symbol, so no merge is wrong, reads as follows, A then B after both blocks:

| Merge within true symbol | Units | Read τ 0.85 | Read τ 0.7 | Read τ 0.5 | Unseen answered, τ 0.85, 0.7, 0.5 |
|---|---|---|---|---|---|
| None (no merge) | 948, 939 | 0.91, 0.85; 0.87, 0.85 | 0.95, 0.91; 0.91, 0.88 | 0.95, 0.91; 0.91, 0.88 | 0.20, 0.71, 0.99; 0.14, 0.58, 1.00 |
| All, one unit per symbol | 90, 90 | 0.47, 0.31; 0.46, 0.42 | 0.84, 0.74; 0.80, 0.73 | 0.94, 0.84; 0.88, 0.85 | 0.03, 0.32, 0.84; 0.01, 0.22, 0.76 |

**A merged unit keeps the answers, and a fixed read threshold stops it from answering.** Each seed's two reads are A, then B. At read τ 0.5, 90 units read within 0.01–0.07 of about 945. A low global threshold then answers most unseen symbols, however, which motivated a radius per unit.

## With the encoder still learning

Eight traces. Rebuild and merge every 2,000 updates, unless stated. End of block 6, both orders, seeds 392, 393 in each. The first-taught set, then the second:

### Normal rate (0.001), H 2048

| Arm | First set, A then B | B then A | Second set, A then B | B then A | Worst first-set loss | Units | MB | Unseen answered |
|---|---|---|---|---|---|---|---|---|
| Neither | 0.64, 0.76 | 0.68, 0.72 | 0.84, 0.84 | 0.86, 0.88 | 0.25 | 1,444–1,674 | 0.3 | 0.08 |
| Rebuild only | 0.91, 0.93 | 0.90, 0.87 | 0.92, 0.91 | 0.94, 0.93 | 0.04 | 1,448–1,678 | 9.4 | 0.13 |
| Merge 0.8 only, radius | 0.85, 0.80 | 0.74, 0.74 | 0.80, 0.75 | 0.83, 0.87 | 0.10 | 337–351 | 2.1 | 0.28 |
| Both, 0.8, radius | 0.80, 0.77 | 0.70, 0.68 | 0.70, 0.76 | 0.81, 0.78 | 0.15 | 285–306 | 1.9 | 0.19 |
| Both, 0.85, radius | 0.86, 0.82 | 0.79, 0.78 | 0.81, 0.81 | 0.84, 0.83 | 0.07 | 489–511 | 3.2 | 0.14 |
| **Both, 0.9, radius** | **0.87, 0.88** | **0.83, 0.82** | **0.87, 0.88** | **0.89, 0.92** | 0.07 | 872–898 | 5.6 | 0.12 |
| Both, 0.8, fixed τ 0.85 | 0.60, 0.64 | 0.47, 0.52 | 0.60, 0.63 | 0.63, 0.62 | 0.23 | 287–306 | 1.9 | 0.06 |
| Both, 0.8, radius, no content gate | 0.72, 0.68 | 0.55, 0.61 | 0.56, 0.64 | 0.70, 0.76 | 0.18 | 260–276 | 1.7 | 0.19 |

Rebuild only reproduces the earlier run's figures, units and forced writes exactly in A then B. Wrong-symbol merges are 3% at 0.9 (71 of 2,736), 3% at 0.85, 4% at 0.8 and 8% at 0.8 with no gate. Answers among candidates, by cosine, read 0.49–0.56 at 0.9 and 0.40–0.51 at 0.8, against 0.52–0.57 for rebuild only.

- **It compresses, and it is not free.** At H 2048, where memory did not need to compress, merging only costs reads. Units halve for 0.05 of the first set and 0.04 of the second. A fifth of the units costs 0.16 of each.
- **Every merging arm above misses the 0.05 loss bar in its worst order and seed.** The mildest are at 0.07.
- **The radius is necessary.** At the same merges, a fixed τ reads 0.47–0.64.
- **Merging alone reads a little higher at this rate, but rejects less.** At 0.8, without rebuilding, reads are 0.04–0.05 higher on average, with 337–351 units against 285–306, and unseen symbols get an answer on 0.28 of queries against 0.19.

### Where capacity is short

| Arm, normal rate | First set, A then B | B then A | Second set, A then B | B then A | Units | Forced writes |
|---|---|---|---|---|---|---|
| Rebuild only, H 1024 | 0.89, 0.89 | 0.79, 0.86 | 0.71, 0.73 | 0.48, 0.80 | 1,024 | 2,797–10,391 |
| Both 0.8, H 1024 | 0.80, 0.77 | 0.70, 0.68 | 0.70, 0.76 | 0.81, 0.78 | 285–306 | 0 |
| Rebuild only, H 512, every 500 | 0.58, 0.60 | 0.43, 0.54 | 0.02, 0.03 | 0.02, 0.03 | 512 | 28,036–30,579 |
| **Both 0.8, H 512** | **0.80, 0.72** | **0.70, 0.64** | **0.64, 0.69** | **0.77, 0.72** | 248–250 | 2,179–2,934 |
| Both 0.8, H 256, every 500 | 0.73, 0.72 | 0.59, 0.65 | 0.65, 0.63 | 0.71, 0.72 | 194–205 | 6,352–7,937 |
| Both 0.8, H 128, every 500 | 0.53, 0.50 | 0.37, 0.40 | 0.28, 0.32 | 0.29, 0.27 | 118–125 | 16,321–19,601 |

**At fixed capacity consolidation changes the outcome.** At H 512 the second set goes from 0.02–0.03 to 0.64–0.77. Below that it degrades gradually down to H 256, and fails at H 128.

### Ten times the rate (0.01), H 2048

| Arm | First set, A then B | B then A | Second set, A then B | B then A | Units | Unseen answered |
|---|---|---|---|---|---|---|
| Rebuild only | 0.88, 0.91 | 0.81, 0.85 | 0.46, 0.51 | 0.60, 0.63 | 2,048 | 0.16 |
| Rebuild only, radius | 0.89, 0.92 | 0.84, 0.85 | 0.69, 0.68 | 0.80, 0.84 | 2,048 | 0.62 |
| Both 0.9, radius | 0.86, 0.81 | 0.80, 0.78 | 0.79, 0.82 | 0.88, 0.88 | 750–839 | 0.35 |
| Merge 0.8 only, radius | 0.74, 0.82 | 0.67, 0.74 | 0.65, 0.68 | 0.80, 0.86 | 908–1,027 | 0.86 |

- **The fill that step 3 found is much reduced:** the second set rises from 0.46–0.63 to 0.79–0.88.
- **It pays in rejection.** Unseen symbols get an answer on 0.35 of queries, against 0.16.
- **Without rebuilding, rejection collapses.** At 0.8, merge only answers 0.86 of unseen queries, against 0.35 with rebuilding. So the two steps are not interchangeable. **[hypothesis]** Radii widen to cover traces that stale addresses no longer match; not measured directly.
- **The radius itself costs rejection under fast drift.** Rebuild only with a radius, with no merging, lifts the second set to 0.68–0.84 but answers 0.62 of unseen queries, with memory full and many forced writes.
- **Forced writes still occur** (3,243–5,303), because memory fills before the first merge at the end of block 4.

## What this does not establish

- **Reaching one unit per symbol.** The best real settings leave about three to ten units per symbol, against one in the reference.
- **Richer values.** The content gate rests on nine distinct values, and a sensory trace value would behave differently.
- **Everything moving at once.** Values and reading stay on block-3 host codes.
- **Learning from a stream.** Encoder updates sample a curated set, and augmentations are hand-designed.
- **Across fonts and in the core.** Neither was tested here.
- **What "the same thing" should mean.** Address proximity and content agreement are weak evidence of it here: same-symbol units overlap other symbols' pairs at these thresholds. Other evidence, such as units answering the same later queries, is untried.

## Scaling

Writes and reads cost O(h d) per item. A merge pass costs O(h² d) time and O(h²) temporary space, brute force; about 1 s per task here over all events at H ≤ 2,048, and an approximate neighbour search would be needed at much larger h. Storage is h(d + m + 2) floats plus up to eight stored images per unit, so traces dominate: 784 bytes each, about 6 KB per unit against 200 bytes. So traces, not addresses, are the thing to shrink next. Units are capped at H. Reads degrade at H 256, about 2.8 units per symbol, and fail at H 128, about 1.4.

## Prior work

**[familiar; read before use]**, none read for this pilot, so this needs doing before any contract. Vigilance allocation is leader clustering and adaptive resonance. Merging by reciprocal nearest neighbours is agglomerative clustering (nearest-neighbour chains); threshold merging of summary statistics resembles BIRCH; a per-unit radius resembles restricted Coulomb energy networks. Chaining and order dependence are their known failure modes. This pilot claims nothing new about them.

[Sleep](prior-work.md#sleep-scheduled-consolidation-and-dreaming) uses "consolidation" for a different operation, transfer to slower parameters; see the note there.

## Computation

Probe: CPU, six workers, 49 s wall, 456 s aggregate CPU, 64 tasks. Learning: GTX 1650, three workers, one thread each, float32, 313 s wall, 1,380 s aggregate CPU, 8 tasks of 19 arms sharing one encoder path, GPU peak 87 MB, peak RSS 1.6 GB per process. Rebuild, merge and radius take 0.3–1.9 s per merging arm per task. Selection in this pilot was by inspection across all of the above, with no cost beyond it.
