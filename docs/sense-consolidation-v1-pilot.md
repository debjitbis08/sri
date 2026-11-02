# Consolidation alongside address rebuilding: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast, and not a contract.** Two seeds. Merge thresholds, the content gate, per-unit radius and its floor were all chosen during the study, the radius after seeing the probe. Runners `experiments/sense_consolidation.py`, `experiments/sense_consolidation_evidence.py`, `experiments/sense_consolidation_traces.py`; merge in `src/sparse_local_memory.py`; reports `reports/sense-consolidation-v1-*.json`. Follows [step 3 of the sense-level pilot](sense-relational-v1-pilot.md), where unit growth became the main limit.

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

## Evidence for "the same thing"

The user's framing (2026-11-02, offered tentatively): consolidation compresses the varied appearances of a thing while preserving distinctions from other things. That separates two label-free tests, added after the results above, with thresholds chosen during the study. Runner `experiments/sense_consolidation_evidence.py`. Rebuilding, radius and content gate as above.

- **Compression evidence.** `shared φ`: each unit gives the other at least φ of its traces, counting each trace's nearest other unit. `cover φ`: each address answers at least φ of the other's traces at τ 0.85. Both also need addresses at 0.6 or above.
- **Distinction test (`distinct`).** A merged unit's radius would cover its own traces. The merge is refused if that radius would take in a trace of another unit with a different content.
- **Reference set (`others`).** Refuse a merge that takes in more than 1% of 256 background images, and raise each radius to reject 99% of them.

All six reference arms reproduce the earlier run's block-6 reads exactly. Normal rate, H 2048, both orders and seeds, end of block 6, ordered by units:

| Arm | First set | Second set | Worst first-set loss | Units | Wrong-symbol merges | Unseen answered |
|---|---|---|---|---|---|---|
| Rebuild only | 0.87–0.93 | 0.91–0.94 | 0.04 | 1,448–1,678 | none | 0.13 |
| Address 0.9 | 0.82–0.88 | 0.87–0.92 | 0.07 | 872–898 | 2.6% | 0.12 |
| Cover 0.5 | 0.82–0.87 | 0.85–0.88 | 0.07 | 737–748 | 2.6% | 0.13 |
| Shared 0.5 | 0.80–0.87 | 0.85–0.90 | 0.08 | 660–703 | 1.7% | 0.17 |
| Cover 0.25, distinct | 0.81–0.85 | 0.84–0.87 | 0.08 | 635–684 | 1.3% | 0.13 |
| Cover 0.25 | 0.78–0.84 | 0.81–0.86 | 0.08 | 530–556 | 2.8% | 0.13 |
| Address 0.85 | 0.78–0.86 | 0.81–0.84 | 0.07 | 489–511 | 3.0% | 0.14 |
| Address 0.8, distinct | 0.76–0.83 | 0.81–0.83 | 0.08 | 412–454 | 1.9% | 0.16 |
| Shared 0.25, distinct | 0.75–0.80 | 0.80–0.86 | 0.10 | 357–399 | 2.1% | 0.19 |
| Address 0.7, distinct | 0.76–0.80 | 0.77–0.84 | 0.12 | 311–342 | 2.9% | 0.20 |
| Address 0.8 | 0.68–0.80 | 0.70–0.81 | 0.15 | 285–306 | 4.1% | 0.19 |
| Shared 0.25 | 0.53–0.74 | 0.58–0.78 | 0.24 | 122–158 | 7.2% | 0.31 |

- **Preserving distinctions does more than the evidence for sameness.** The distinction test cuts wrong merges by about half to two-thirds in every comparison. It also rescues the most compressive rule: shared 0.25 reaches 122–158 units but reads poorly, and with the test it holds 0.75–0.86 at 357–399.
- **The frontier moves, modestly.** Against address 0.8, address 0.7 with the test takes about 10% more units and reads 0.03 and 0.05 higher on average. Against address 0.85, shared 0.25 with the test takes about a quarter fewer units, reading 0.03 lower and 0.01 higher.
- **No arm meets the 0.05 loss bar at this rate.** The best are at 0.07.
- **Wrong merges do not account for the remaining loss.** Cover 0.25 with the test has only 1.3% wrong merges and still loses 0.08 at worst. Coverage is part of it, at 0.93 of queries against 0.98. [Trace selection](#trace-selection) below tests the eight-trace cap as the cause.
- **The reference set adds nothing here.** Address 0.8 answers unseen symbols at 0.19 with and without it, with a further 200 KB of images not counted in the MB columns. **[derived]** A query is answered if any unit accepts it, so 1% per unit is a loose limit over about 300 units; rejecting unseen symbols this way needs a far stricter per-unit limit than 256 images can measure.

Where capacity is short, normal rate, H 512, merging every 2,000 updates:

| Arm | First set | Second set | Worst loss | Units | Forced writes | Unseen answered |
|---|---|---|---|---|---|---|
| Rebuild only, every 500 | 0.43–0.60 | 0.02–0.03 | 0.44 | 512 | 28,036–30,579 | 0.07 |
| Address 0.8 | 0.64–0.80 | 0.64–0.77 | 0.16 | 248–250 | 2,179–2,934 | 0.17 |
| Address 0.7, distinct | 0.71–0.81 | 0.70–0.80 | 0.10 | 252–279 | 2,178–2,920 | 0.19 |
| Shared 0.25, distinct | 0.73–0.81 | 0.71–0.79 | 0.12 | 275–286 | 2,208–3,263 | 0.18 |

The worst read rises from 0.64 to 0.70–0.71 at about the same units.

At ten times the rate, H 2048:

| Arm | First set | Second set | Worst loss | Units | Unseen answered |
|---|---|---|---|---|---|
| Rebuild only | 0.81–0.91 | 0.46–0.63 | 0.04 | 2,048 | 0.16 |
| Address 0.9 | 0.78–0.86 | 0.79–0.88 | 0.10 | 750–839 | 0.35 |
| **Address 0.8, distinct** | **0.81–0.86** | **0.76–0.83** | **0.04** | 877–1,175 | 0.39 |
| Cover 0.25, distinct | 0.80–0.89 | 0.79–0.88 | 0.05 | 1,075–1,358 | 0.39 |

**Under fast drift, the distinction test meets the bar while keeping the second set,** which no earlier arm did. It compresses less, since drift gives it more to refuse, and it still answers unseen symbols far too often.

## Trace selection

Does the eight-trace cap cause the remaining loss? Each policy ran under the same merge rules, chosen during the study. Runner `experiments/sense_consolidation_traces.py`.

- **Turns:** the rule so far, eight taken in turn from each merged unit.
- **Spread:** eight distinct images chosen by farthest-point selection when units merge. A full unit takes no new traces.
- **Intake:** as spread, but a unit may hold 16 between merges, so new appearances can enter.
- **All:** every distinct trace of a unit. A ceiling, not a proposal. Radius from the lowest similarity.
- **All, 5th percentile:** the same, with the radius tolerating outliers.

Turns reproduce the evidence run's reads exactly in blocks 4–6. Normal rate, H 2048, rule `cover 0.25, distinct`, both orders and seeds, end of block 6. MB counts traces at the end; peak traces is the most held just before any merge:

| Traces | First set, mean | Second set, mean | Worst loss | Units | Unseen answered | MB | Peak traces |
|---|---|---|---|---|---|---|---|
| Rebuild only, no merge | 0.90 | 0.93 | 0.04 | 1,448–1,678 | 0.13 | 9.4 | 12,108 at the end |
| Turns | 0.83 | 0.86 | 0.08 | 635–684 | 0.13 | 4.2 | 9,144 |
| **Spread** | **0.86** | **0.88** | **0.049** | 610–645 | 0.15 | 2.6 | 7,937 |
| Intake | 0.87 | 0.89 | 0.07 | 621–633 | 0.17 | 2.8 | 13,034 |
| All, 5th percentile | 0.85 | 0.88 | 0.049 | 618–664 | 0.14 | 5.8 | 8,217 |
| All | 0.88 | 0.90 | 0.04 | 632–668 | 0.21 | 5.8 | 8,235 |

- **The cap was a main cause.** Keeping all traces meets the 0.05 bar and comes within 0.024 of rebuilding alone, on both sets, at about 40% of the units.
- **Choosing traces well recovers over half of that with no more than eight per unit.** Spread gains 0.028 and 0.023 of the ceiling's 0.045 and 0.042. It also meets the bar, narrowly, at 0.049, with 27% of the bytes of rebuilding alone. This setting and its margin come from looking at two seeds, so it needs confirming.
- **Trace bytes fall for a second reason.** Spread drops repeated images: units end with about five distinct traces on average, not eight.
- **Coverage trades against rejection.** With all traces and the lowest similarity, unseen symbols are answered at 0.21 and more at other rules (0.27, 0.32). The 5th percentile restores 0.14, but gives back the reads.
- **Intake is not justified as built.** It reads a little better than spread and misses the bar, and its peak exceeds keeping everything, because repeated images fill its buffer.
- **Rebuilding alone needs no more than eight traces.** Keeping all of them leaves it at 0.90 and 0.92 at this rate.

Across rules at this rate and capacity:

| Rule | Turns: reads, loss, units | Spread: reads, loss, units, unseen | All: reads, loss, units, unseen |
|---|---|---|---|
| Address 0.8, distinct | 0.80, 0.82; 0.08; 412–454 | 0.84, 0.86; 0.08; 392–458; 0.18 | 0.87, 0.89; 0.07; 483–518; 0.27 |
| Address 0.7, distinct | 0.77, 0.81; 0.12; 311–342 | 0.83, 0.85; 0.13; 315–355; 0.21 | 0.86, 0.89; 0.05; 419–452; 0.32 |

With fewer units the reads rise, but address rules still miss the bar with eight traces.

**Where capacity is short,** normal rate, H 512, `address 0.7, distinct`, means of first and second sets:

| Traces | Reads | Worst read | Worst loss | Units | Forced writes | Unseen answered | MB |
|---|---|---|---|---|---|---|---|
| Turns | 0.76, 0.77 | 0.70 | 0.10 | 252–279 | 2,178–2,920 | 0.19 | 1.7 |
| Spread | 0.80, 0.80 | 0.76 | 0.08 | 265–297 | 2,013–2,841 | 0.22 | 1.5 |
| All | 0.84, 0.87 | 0.82 | 0.07 | 357–361 | 2,176–3,352 | 0.41 | 5.8 |

No policy meets the bar here. The best is `cover 0.25, distinct` with all traces at the 5th percentile, at 0.053.

**At ten times the rate,** H 2048, `cover 0.25, distinct`:

| Traces | Reads | Worst loss | Units | Unseen answered |
|---|---|---|---|---|
| Rebuild only, no merge | 0.86, 0.55 | 0.04 | 2,048 | 0.16 |
| Rebuild only, all traces | 0.87, 0.69 | 0.04 | 2,048 | 0.17 |
| Turns | 0.85, 0.83 | 0.05 | 1,075–1,358 | 0.39 |
| Spread | 0.86, 0.85 | 0.06 | 1,071–1,337 | 0.44 |
| All | 0.88, 0.87 | 0.04 | 1,321–1,503 | 0.55 |

Trace choice matters less here, and rejection gets worse as coverage rises. At H 512 and this rate, every arm answers 0.60 or more of unseen queries. Fast drift's rejection problem remains open.

## What this does not establish

- **Reaching one unit per symbol.** The best real settings leave about three to ten units per symbol, against one in the reference. The setting that meets the bar leaves about seven.
- **That the bar is met.** Spread's 0.049 comes from choosing among many arms on two seeds.
- **Richer values.** The content gate rests on nine distinct values, and a sensory trace value would behave differently.
- **Everything moving at once.** Values and reading stay on block-3 host codes.
- **Learning from a stream.** Encoder updates sample a curated set, and augmentations are hand-designed.
- **Across fonts and in the core.** Neither was tested here.
- **Later evidence.** Evidence here comes from kept traces at merge time, not from new queries as they arrive.
- **Varied appearances beyond one font.** Compression was not tested across fonts, the harder form of the user's framing.

## Scaling

Writes and reads cost O(h d) per item. A merge pass costs O(h² d) time and O(h²) temporary space, brute force; about 1 s per task here over all events at H ≤ 2,048, and an approximate neighbour search would be needed at much larger h. Storage is h(d + m + 2) floats plus up to eight stored images per unit, so traces dominate: 784 bytes each, about 6 KB per unit against 200 bytes. So traces, not addresses, are the thing to shrink next. Units are capped at H. Reads degrade at H 256, about 2.8 units per symbol, and fail at H 128, about 1.4.

## Prior work

**[familiar; read before use]**, none read for this pilot, so this needs doing before any contract. Vigilance allocation is leader clustering and adaptive resonance. Merging by reciprocal nearest neighbours is agglomerative clustering (nearest-neighbour chains); threshold merging of summary statistics resembles BIRCH; a per-unit radius resembles restricted Coulomb energy networks. Chaining and order dependence are their known failure modes. This pilot claims nothing new about them.

[Sleep](prior-work.md#sleep-scheduled-consolidation-and-dreaming) uses "consolidation" for a different operation, transfer to slower parameters; see the note there.

## Computation

Probe: CPU, six workers, 49 s wall, 456 s aggregate CPU, 64 tasks. Learning: GTX 1650, three workers, one thread each, float32, 313 s wall, 1,380 s aggregate CPU, 8 tasks of 19 arms sharing one encoder path, GPU peak 87 MB, peak RSS 1.6 GB per process. Rebuild, merge and radius take 0.3–1.9 s per merging arm per task. Evidence: same hardware and workers, 735 s wall, 6,216 s aggregate CPU, 8 tasks of 26 arms, GPU peak 87 MB, peak RSS 2.0 GB per process; merge evidence and tests take 0.6–6.0 s per merging arm per task, the slowest being cover with the distinction test and reference set. Those tests add O(t h d) per pass for t kept traces. Trace selection: same hardware and workers, 1,711 s wall, 17,461 s aggregate CPU, 8 tasks of 32 arms, GPU peak 87 MB, peak RSS 2.1 GB per process; rebuild, merge and radius take up to 9 s per arm per task at the normal rate and up to 80 s at ten times the rate. Spread selection costs O(k t d) for a unit's t traces. Selection in this pilot was by inspection across all of the above, with no cost beyond it.
