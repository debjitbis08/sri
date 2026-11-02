# Memory with no long-term image store: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast, and not a contract.** Two seeds. The arms, the map's use, the radius rule and the change of symbol values below were chosen during the study, some after a smoke run, and no setting was confirmed on other seeds.

## Question

By the user's decision, memory stores no images ([consolidation pilot](sense-consolidation-v1-pilot.md#decision-after-this-pilot-no-stored-images)). With keys, values and host all learning, can stored codes stay usable when they are corrected only by inputs passing now? An external reviewer framed the decisive question: is drift mostly a predictable change of coordinates, or actual loss of information? And how much depends on this data: nine recurring values and an encoder whose input never changes in kind?

## Design

A unit holds an address, a value code, a count and an angular radius. Each map comes from inputs passing during the block and from copies of the networks as they were at the block's start.

**Images are not fully gone (corrected after external review).** Units store no images, but the correction buffers images: up to 4,000 key probes within each block (about 3.1 MB at a byte per pixel), re-encoded through both encoders at the block's end and then discarded, and a fixed set of 2,048 background probes held for the whole run (about 1.6 MB). So this removes the per-memory store, and not images from upkeep.

**Values, two sets:**

- **Counts:** as before, 90 facts share nine counts, in host codes, among nine candidates. A write stores the code of an unseen value image under the host at the block's start.
- **Symbols:** each fact has its own target symbol, read among 150 (chance 0.007). A target's images are seen only while its own set is taught. Codes come from the key encoder as it learns.

**Change during the study.** The plan put symbol values through a host taught those pairs. Those hosts train and block 3 reproduces, but they recall taught pairs at only 0.00–0.08 among 150, and a fresh target code finds its own symbol at only 0.05–0.15. So a host read would measure the host, and values moved to the key encoder's code. **So the symbols set does not test a moving host.** Only one encoder moves there.

**Stream:** stationary, or shifting. Shifting trains the key encoder on dot collections and tally, tree and banana images from block 5, instead of background symbols.

**Maps:** affine and fitted per block by identity-centred ridge, λ 0.01, the setting from the [translation draft](translation-memory-v1-draft.md), not tuned here. Live probes are the block's stream and taught inputs. For hosts they are drawn from the same populations, not the exact draws.

**Radius, replacing the rule that read kept traces.** A new unit starts at τ 0.85. A merged unit's ball covers both members' balls. Drift widens a unit by multiplying cosines, with a floor of 0.5. Adding angles was the first rule, and was replaced after a smoke run answered almost everything. The merge test refuses a ball that would hold another unit's address with a different value.

**Arms,** varied one at a time from values-live:

| Arm | Keys | Values | Merge |
|---|---|---|---|
| Values stale, background, live | radius widens | no map, background map, live map | no |
| Values oracle | radius widens | map fitted on the values' own images; a reference | no |
| Keys fixed | no correction | live | no |
| Keys transport | linear live map, widen by its held-out residual | live | no |
| Keys frozen | never learn | live | no |
| Merge 0.8, 0.7; transport merge 0.8 | radius or transport | live | yes |
| H 512 | radius | live | none or 0.8 |
| Images, rebuild and cover | re-encoded from stored images | re-encoded | reference |

**Gate:** with counts on a stationary stream, both image arms reproduce [Everything moving](sense-consolidation-v1-pilot.md#everything-moving) exactly on units and answers, at both rates, all 16 comparisons.

## Results

End of block 6, abstentions wrong. Ranges across two seeds and two orders; means in the text. Loss is the worst drop of a set from the end of its own block. Reads have 225 queries, SE about 0.03.

### Normal rate, stationary stream

Counts, among nine:

| Arm | First set | Second set | Worst loss | Units | KB | Unseen answered |
|---|---|---|---|---|---|---|
| Images, rebuild | 0.57–0.74 | 0.59–0.72 | 0.06 | 1,448–1,678 | 18,429 | 0.13 |
| Images, cover | 0.58–0.69 | 0.57–0.68 | 0.07 | 627–671 | 5,090 | 0.14 |
| Values stale | 0.13–0.24 | 0.35–0.36 | 0.24 | 1,387–1,581 | 303 | 0.49 |
| Values background | 0.35–0.72 | 0.40–0.60 | 0.06 | 1,387–1,581 | 303 | 0.49 |
| Values live | 0.60–0.77 | 0.64–0.74 | 0.08 | 1,387–1,581 | 303 | 0.49 |
| Values oracle | 0.62–0.76 | 0.65–0.75 | 0.07 | 1,387–1,581 | 303 | 0.49 |
| Keys fixed | 0.43–0.65 | 0.59–0.72 | 0.16 | 1,444–1,674 | 319 | 0.08 |
| **Keys transport** | **0.57–0.77** | **0.64–0.75** | **0.07** | 1,435–1,661 | 316 | **0.32** |
| Keys frozen | 0.58–0.73 | 0.60–0.71 | 0.06 | 936–948 | 192 | 0.17 |
| Merge 0.8 | 0.52–0.69 | 0.55–0.70 | 0.15 | 841–1,123 | 194 | 0.78 |
| Transport merge 0.8 | 0.51–0.78 | 0.55–0.76 | 0.13 | 793–1,133 | 195 | 0.75 |
| Values live, H 512 | 0.51–0.65 | 0.33–0.40 | 0.17 | 512 | 104 | 0.43 |
| Merge 0.8, H 512 | 0.49–0.64 | 0.47–0.65 | 0.16 | 468–495 | 98 | 0.66 |

Symbols, among 150:

| Arm | First set | Second set | Worst loss | Units | KB | Unseen answered |
|---|---|---|---|---|---|---|
| Images, rebuild | 0.61–0.71 | 0.55–0.72 | 0.04 | 1,448–1,678 | 18,429 | 0.13 |
| Images, cover | 0.60–0.70 | 0.55–0.69 | 0.05 | 844–972 | 5,856 | 0.15 |
| Values stale | 0.64–0.74 | 0.63–0.75 | 0.08 | 1,387–1,581 | 398 | 0.49 |
| Values live | 0.61–0.77 | 0.63–0.74 | 0.12 | 1,387–1,581 | 398 | 0.49 |
| Values oracle | 0.62–0.72 | 0.62–0.74 | 0.13 | 1,387–1,581 | 398 | 0.49 |
| Keys fixed | 0.52–0.56 | 0.61–0.66 | 0.22 | 1,444–1,674 | 419 | 0.08 |
| **Keys transport** | **0.62–0.78** | **0.64–0.72** | **0.10** | 1,435–1,661 | 415 | **0.32** |
| Merge 0.8 | 0.62–0.76 | 0.61–0.75 | 0.13 | 1,116–1,275 | 318 | 0.71 |

Keys transport per seed and order, first/second set, unseen answered (images rebuild in brackets):

| | 392 AB | 392 BA | 393 AB | 393 BA |
|---|---|---|---|---|
| Counts | 0.57/0.64, 0.39 (0.64/0.59, 0.14) | 0.65/0.67, 0.42 (0.57/0.60, 0.15) | 0.77/0.75, 0.22 (0.74/0.72, 0.12) | 0.66/0.73, 0.24 (0.69/0.72, 0.12) |
| Symbols | 0.78/0.68, 0.39 (0.71/0.72, 0.14) | 0.72/0.72, 0.42 (0.70/0.70, 0.15) | 0.66/0.64, 0.22 (0.62/0.55, 0.12) | 0.62/0.64, 0.24 (0.61/0.55, 0.12) |

- **Answers to known items hold with no per-memory images.** Keys transport reads as well as stored images on both sets of values, at about 2% of the reported memory payload, excluding network copies and probe buffers. It also answers more unseen symbols, so this is not comparable memory behaviour.
- **Rejection does not hold.** Unseen symbols are answered at 0.32 against 0.13. Without widening, rejection is good but the first set falls, with the worst loss at 0.16–0.22. This is now the main failure on a stationary stream.
- **For the host, drift is mostly a change of coordinates, and live inputs can follow it.** Stale reads 0.13–0.24 and the live map 0.60–0.77, about the same as the oracle. Counts recur in the host's stream, which favours this. Background images are erratic (0.35–0.72).
- **One-off values on a stationary stream need no correction.** Stale values read as well as any map, 0.64–0.74. This reflects how little the key encoder changes on its own kind of input, not a general result.
- **Merging is correct but compresses little, and costs reads.** Wrong merges are few: 4 of 3,039 and 11 of 1,284 for merge 0.8 across all conditions. But widened balls overlap, so the merge test refuses most pairs, and worst loss and rejection are both poor. Merging no longer saves much either, as units are about 200 bytes.
- **Capacity still needs merging.** At H 512 on counts, merging lifts the second set from 0.33–0.40 to 0.47–0.65.
- **Other fonts read higher through wide radii.** Keys transport reads 0.43, 0.44 on counts and 0.48, 0.44 on symbols, against 0.40, 0.39 and 0.41, 0.38 for images, largely because more is answered.

### Ten times the rate, stationary stream

| Arm | Counts: first, second | Loss | Symbols: first, second | Loss | Unseen |
|---|---|---|---|---|---|
| Images, rebuild | 0.57–0.70, 0.23–0.37 | 0.06 | 0.62–0.74, 0.38–0.48 | 0.00 | 0.16 |
| Images, cover | 0.57–0.68, 0.54–0.64 | 0.07 | 0.60–0.72, 0.54–0.64 | 0.02 | 0.43–0.44 |
| Keys fixed | 0.16–0.27, 0.23–0.35 | 0.42 | 0.20–0.26, 0.29–0.37 | 0.32 | 0.05 |
| Keys transport | 0.48–0.69, 0.50–0.61 | 0.04 | 0.44–0.53, 0.51–0.72 | 0.07 | 0.59 |
| Values live | 0.47–0.67, 0.48–0.60 | 0.16 | 0.46–0.62, 0.55–0.70 | 0.07 | 0.97 |

Transport holds both sets within or near the loss bar with the store full at 2,048 units, where rebuilding alone loses the second set to capacity. Rejection is poor. Merging barely runs: 2,022–2,040 units.

### When the stream shifts

Normal rate, from block 5:

| Arm | Counts: first, second | Loss | Symbols: first, second | Loss | Unseen |
|---|---|---|---|---|---|
| Images, rebuild | 0.53–0.73, 0.52–0.66 | 0.10 | 0.43–0.50, 0.39–0.47 | 0.27 | 0.52 |
| Keys frozen | 0.59–0.73, 0.60–0.71 | 0.06 | 0.40–0.49, 0.24–0.42 | 0.37 | 0.17 |
| Keys fixed | 0.02–0.04, 0.39–0.55 | 0.78 | 0.00, 0.16–0.30 | 0.74 | 0.15 |
| Values live | 0.13–0.24, 0.52–0.64 | 0.60 | 0.02–0.06, 0.23–0.36 | 0.76 | 0.99 |
| Keys transport | 0.34–0.57, 0.40–0.56 | 0.28 | 0.21–0.28, 0.18–0.24 | 0.59 | 0.79 |

- **The design fails here, and the stationary result does not generalize to it.** Median key movement in block 5 rises from about 0.38 to 1.01 radians at the normal rate, and the linear map leaves 0.60 on held-out probes.
- **On counts, useful distinctions remain available, and the live map does not follow them.** Re-encoding from images reads 0.53–0.73 and frozen keys 0.59–0.73, against 0.34–0.57. That shows those distinctions are still there, not that no information was lost.
- **On symbols, information is also lost.** The encoder itself identifies targets worse: fresh single target codes fall from 0.51–0.61 to 0.37–0.46, and even re-encoding from images loses 0.27.
- **Maps may overcorrect.** With shifting symbols, stale values read the second set better than the oracle map: 0.45–0.58 against 0.24–0.27. A value written late in a block is already near the block's end state, yet receives the whole block's map. This is a likely cause, and it would also affect addresses, but it was not tested separately.

At ten times the rate with a shift, image-free arms with learning keys read at most 0.50 on counts and 0.33 on symbols. Frozen keys hold counts at 0.59–0.73, and the image arms degrade too (second set 0.12–0.57).

### What rejection rests on

A unit answers when a query's cosine with its address reaches the unit's threshold. Nothing else marks a symbol as unknown. On frozen keys, with memory holding both sets, a query's best similarity (seeds 392, 393; 5th, 50th, 95th percentiles):

| Queries | 392 | 393 |
|---|---|---|
| Known symbols, new renderings | 0.85, 0.94, 0.98 | 0.86, 0.94, 0.98 |
| Unseen symbols | 0.56, 0.75, 0.90 | 0.56, 0.74, 0.89 |

The separation is good but not clean, with AUC 0.97 in both. At threshold 0.85, 0.95–0.97 of known queries and 0.14–0.20 of unseen ones are answered; at 0.75, all known and 0.48–0.49 unseen; at 0.7, 0.58–0.71 unseen. Widening by drift, a first-set unit reaches about 0.68 by block 6 at the normal rate, and transport about 0.76, which accounts for the rejection seen above. On a shift, block 5 alone takes the widening rule to its floor of 0.5. This read-only measure does not cover a moving encoder.

## Windows: one version, no image held, transport separate from thresholds

A follow-up after external review, same seeds, `experiments/sense_image_free_windows.py`. Its settings were chosen before its results, and it still informs selection, so it is exploratory too.

- **One version.** Memory reads and writes use a published copy of each network while the working copy trains. Every 250 updates, inputs passing over 16 updates go through the published copy and a candidate. A map then moves all stored codes, and the candidate is published.
- **No image held.** Only running totals of code pairs are kept. Stored images remain only in the reference arms.
- **Separate rules.** Thresholds are fixed, widened by drift, or background: each unit's running 0.999 quantile of similarity to stream inputs memory is not being taught. This last rule was proposed after the first run.
- **Matched false acceptance.** All thresholds shift until 0.13 of unseen queries are answered, the stored-image rate. This uses the unseen queries, so it scores arms and is not a mechanism.

**Gate, partly failed.** Stored-image arms reproduce the earlier run exactly in the B-then-A order (8 of 16 comparisons). In the A-then-B order, keys and units match but the full-length hosts have different state hashes, and reads differ by up to 0.04. The earlier code still reproduces its hash at full length. In a shortened 120-update run, this code and the earlier code gave identical hashes, with or without saving every window. The cause of the full-length difference was not found. All arms in this run share these hosts, so comparisons within the run are unaffected.

Normal rate, stationary stream. Raw reads as above; the last column pools both sets at matched false acceptance, as range and mean.

Counts:

| Arm | First set | Second set | Worst loss | KB | Unseen | At unseen 0.13 |
|---|---|---|---|---|---|---|
| Images, rebuild | 0.57–0.75 | 0.60–0.72 | 0.06 | 18,429 | 0.13 | 0.56–0.76 (0.65) |
| Images, cover | 0.58–0.71 | 0.58–0.69 | 0.11 | 5,088 | 0.14 | 0.57–0.72 (0.64) |
| Frozen keys | 0.63–0.70 | 0.61–0.71 | 0.06 | 192 | 0.17 | 0.60–0.71 (0.66) |
| Fixed, no transport | 0.42–0.59 | 0.60–0.71 | 0.22 | 285 | 0.07 | 0.51–0.74 (0.64) |
| Transport, fixed | 0.37–0.42 | 0.56–0.64 | 0.36 | 223 | 0.07 | 0.45–0.72 (0.59) |
| Widen | 0.53–0.67 | 0.49–0.61 | 0.16 | 133 | 0.93 | 0.43–0.64 (0.53) |
| Transport, widen | 0.58–0.68 | 0.57–0.70 | 0.10 | 167 | 0.92 | 0.33–0.64 (0.51) |
| Background | 0.56–0.67 | 0.59–0.67 | 0.12 | 131 | 0.60 | 0.42–0.55 (0.49) |
| Transport, background | 0.55–0.66 | 0.58–0.68 | 0.12 | 127 | 0.54 | 0.35–0.57 (0.49) |
| Transport, fixed, merge 0.8 | 0.50–0.63 | 0.52–0.68 | 0.19 | 125 | 0.62 | 0.21–0.48 (0.34) |
| Transport, fixed, stale values | 0.13–0.22 | 0.30–0.40 | 0.37 | 223 | 0.07 | 0.16–0.43 (0.30) |

Symbols:

| Arm | First set | Second set | Worst loss | KB | Unseen | At unseen 0.13 |
|---|---|---|---|---|---|---|
| Images, rebuild | 0.61–0.71 | 0.55–0.72 | 0.04 | 18,429 | 0.13 | 0.55–0.71 (0.64) |
| Images, cover | 0.60–0.70 | 0.55–0.69 | 0.05 | 5,856 | 0.15 | 0.55–0.67 (0.62) |
| Frozen keys | 0.57–0.69 | 0.61–0.73 | 0.14 | 253 | 0.17 | 0.57–0.71 (0.64) |
| Fixed, no transport | 0.43–0.52 | 0.62–0.68 | 0.35 | 375 | 0.07 | 0.52–0.73 (0.61) |
| Transport, fixed | 0.30–0.43 | 0.51–0.62 | 0.37 | 293 | 0.07 | 0.37–0.69 (0.57) |
| Transport, fixed, stale values | 0.33–0.48 | 0.53–0.64 | 0.33 | 293 | 0.07 | 0.41–0.72 (0.58) |
| Transport, widen | 0.57–0.71 | 0.63–0.72 | 0.14 | 219 | 0.92 | 0.39–0.57 (0.50) |
| Background | 0.57–0.66 | 0.62–0.73 | 0.18 | 172 | 0.60 | 0.46–0.61 (0.51) |
| Transport, fixed, merge 0.8 | 0.50–0.56 | 0.60–0.71 | 0.25 | 196 | 0.52 | 0.27–0.40 (0.32) |

Matched, per seed and order, first/second (392 AB, 392 BA, 393 AB, 393 BA): on counts, frozen keys 0.62/0.60, 0.62/0.67, 0.69/0.70, 0.69/0.71; images rebuild 0.62/0.60, 0.56/0.59, 0.76/0.68, 0.70/0.72; transport fixed 0.46/0.61, 0.45/0.64, 0.50/0.71, 0.67/0.72.

- **The first run's result rested largely on looser thresholds.** Keys transport there read like stored images and answered more unseen symbols. At matched false acceptance, its widening and its transport both lose.
- **Linear transport of addresses adds nothing here, and costs.** At matched false acceptance it reads 0.59 on counts against 0.64 with no transport, and 0.57 against 0.61 on symbols, mostly in the first set (0.52 against 0.58, 0.48 against 0.55). Chaining 24 maps may accumulate error, as the translation draft warned. This was not tested separately.
- **Most key movement does not accumulate.** Median movement is about 0.30 radians per window, summing to about 2.5 per block, while net movement over the first run's blocks was about 0.43. So maps may be fitting jitter. This reading is an inference from those two measures.
- **Simple threshold calibration does most of the work on a stationary stream.** With no correction, and thresholds shifted to the stored images' false acceptance, reads are 0.58 first, 0.69 second on counts, against 0.66 and 0.65 with images. That shift used unseen queries, so a live way to set it remains untested. A first-set gap of about 0.08 remains.
- **Value maps from live inputs remain the strongest positive result for counts.** Matched reads are 0.59 against 0.30 for stale values, about the same as the oracle. On symbols they add nothing, again.
- **The background threshold fails as tested.** Median thresholds fall to about 0.67 and unseen answers rise to 0.54–0.60, with matched reads of 0.49–0.51. It may be mismeasuring, since the stream's images are not memory's symbols.
- **Merging without traces costs much at matched false acceptance:** 0.34 and 0.32. Covering both members' balls makes units too wide.
- **Frozen keys, with the live value map on counts, is the best image-free arm:** 192 KB, loss 0.06, matched 0.66, and it needs no key correction at all. That is an encoder that does not learn.

**Ten times the rate, stationary, matched, first and second means:** frozen keys 0.65, 0.67 on counts; fixed 0.45, 0.43; transport fixed 0.31, 0.59; images rebuild 0.63, 0.26; images cover 0.32, 0.21. So at this acceptance stored images also fail, the merged reference most. On symbols: frozen keys 0.48, 0.63; fixed 0.36, 0.49; transport fixed 0.25, 0.54; images rebuild 0.65, 0.36; images cover 0.29, 0.16.

**Stream shift, normal rate, matched, first and second means:** on counts, frozen keys 0.65, 0.67; images rebuild 0.44, 0.45; fixed 0.02, 0.39; transport fixed 0.05, 0.36. On symbols, frozen keys 0.36, 0.47 (their values still move); images rebuild 0.33, 0.30; fixed 0.00, 0.30; transport fixed 0.01, 0.27.

- **Short windows do not rescue transport under a shift.** Raw first-set reads are 0.03–0.15 on counts. Moving-address arms stay far below re-encoding.
- **The overcorrection was real, and it was not the main failure.** Stale values no longer beat the oracle on the second set: 0.36–0.45 against 0.32–0.46.

**Storage, named separately.** No image-free arm holds an image. For transport fixed on counts and symbols at H 2048: compact payload 223–293 KB; arrays this implementation allocates 0.70–0.83 MB; working copies of the networks, published and candidate, about 1.7 MB for keys and 1.9 MB for the host; running totals about 34 KB. That totals about 4.4 MB at peak, against 5.1–18.4 MB for stored images, whose arms also use no copy. With frozen keys, the key copies go. The stored-image figure grows with experiences; this one grows only with units, up to H.

**Computation.** Hosts with weights every window: 446 s wall, 890 s aggregate CPU, 139 MB on disk. Learning: GTX 1650, three workers, 1,664 s wall, 13,028 s aggregate CPU, 16 tasks of 30 arms, GPU peak 92 MB, peak RSS 2.9 GB per process. Host-hash diagnostics took a further few minutes of one process each.

## What this does not establish

- **What transport alone contributes.** Keys transport moves addresses and widens thresholds together, and arms were not compared at matched false acceptance.
- **Linear transport under fast change.** Addresses and values are written throughout a block and all receive the start-to-end map, so stored aggregates mix versions. This defect should be fixed before concluding anything about maps.
- **That a map is the right correction.** Linear per-block maps were the only ones tested. The local shift from the read-only check was left out because it was unstable there.
- **A moving host for one-off values.** The symbols set uses the key encoder's code. The host does not identify symbols, so the moving-host question remains open for anything but counts.
- **The ceilings as bounds.** Memory averages several value images, so symbol reads exceed the single-code reference, 0.51–0.64.
- **Shift as a test in general.** It is one shift, once, at one point, without any recovery time beyond block 6.
- **Richer images.** All inputs are simple images at 28 × 28.
- **A fair test of maps under fast change.** Image arms re-encode exactly at block ends, while a map gives every write the whole block's movement, so the comparison favours images.

## Scaling

Memory costs h(32 + m + 3) floats: about 200 bytes per unit for counts and 270 for symbols, against up to eight pairs of 784-byte images. Correction keeps a copy of the key encoder (0.84 MB) and of the host's image path (about 0.9 MB) for one block. This is more than this memory at this size, but constant in facts and the networks already exist at that size. Fitting a d × d map on p probes costs O(p d² + d³) per block and per value space, and moving h units costs O(h d²). Writes and reads stay O(h d); a merge pass stays O(h² d), or O(p h d) more for the merge test on p proposed pairs. Probes are re-encoded at each block's end, a fixed number, so upkeep no longer grows with stored experiences. The reported bytes estimate a compact unit, not the arrays this implementation allocates (which keep both the address and its running sum, at the full H).

## Prior work

Read for the query only: the [drift compensation family](https://arxiv.org/abs/2004.00440) (Yu et al. 2020) and [Learnable Drift Compensation](https://arxiv.org/abs/2407.08536) (Gomez-Villa et al. 2024), and [Learning without Memorizing](https://arxiv.org/abs/1811.08051) (Dhar et al. 2019), **[abstract]** each. Linear transport here is closest to the first two in aim. Their failure mode appears here: movement on current inputs need not predict movement on old ones. The [translation pilot](translation-memory-v1-pilot.md)'s anchor map also moved accuracy rather than adding to it, though that map also rewrote current codes. Ball merging resembles BIRCH **[familiar; read before use]**. Full reading remains owed before any contract, per `docs/literature-review.md`.

## Computation

Symbol hosts: GTX 1650, three workers, 469 s wall, 948 s aggregate CPU, 4 fits of 12,000 updates. Learning: same GPU and workers, 1,932 s wall, 16,927 s aggregate CPU, 16 tasks of 28 arms over both sets of values, GPU peak 90 MB, peak RSS 2.9 GB per process. Selection was by inspection, plus two smoke runs and one read-only drift check.
