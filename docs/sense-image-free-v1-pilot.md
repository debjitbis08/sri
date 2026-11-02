# Memory with no stored images: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast, and not a contract.** Two seeds. The arms, the map's use, the radius rule and the change of symbol values below were chosen during the study, some after a smoke run, and no setting was confirmed on other seeds.

## Question

By the user's decision, memory stores no images ([consolidation pilot](sense-consolidation-v1-pilot.md#decision-after-this-pilot-no-stored-images)). With keys, values and host all learning, can stored codes stay usable when they are corrected only by inputs passing now? An external reviewer framed the decisive question: is drift mostly a predictable change of coordinates, or actual loss of information? And how much depends on this data: nine recurring values and an encoder whose input never changes in kind?

## Design

A unit holds an address, a value code, a count and an angular radius. Each map comes from inputs passing during the block, which are then discarded, and from copies of the networks as they were at the block's start.

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

- **Answers hold with no stored images.** Keys transport reads as well as stored images on both sets of values, at about 2% of the bytes.
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
- **On counts, distinctions survive and the live map cannot follow them.** Re-encoding from images reads 0.53–0.73 and frozen keys 0.59–0.73, against 0.34–0.57. So the gap is in keeping codes current, not in information.
- **On symbols, information is also lost.** The encoder itself identifies targets worse: fresh single target codes fall from 0.51–0.61 to 0.37–0.46, and even re-encoding from images loses 0.27.
- **Maps may overcorrect.** With shifting symbols, stale values read the second set better than the oracle map: 0.45–0.58 against 0.24–0.27. A value written late in a block is already near the block's end state, yet receives the whole block's map. This is a likely cause, and it would also affect addresses, but it was not tested separately.

At ten times the rate with a shift, image-free arms with learning keys read at most 0.50 on counts and 0.33 on symbols. Frozen keys hold counts at 0.59–0.73, and the image arms degrade too (second set 0.12–0.57).

## What this does not establish

- **That a map is the right correction.** Linear per-block maps were the only ones tested. The local shift from the read-only check was left out because it was unstable there.
- **A moving host for one-off values.** The symbols set uses the key encoder's code. The host does not identify symbols, so the moving-host question remains open for anything but counts.
- **The ceilings as bounds.** Memory averages several value images, so symbol reads exceed the single-code reference, 0.51–0.64.
- **Shift as a test in general.** It is one shift, once, at one point, without any recovery time beyond block 6.
- **Richer images.** All inputs are simple images at 28 × 28.
- **A fair test of maps under fast change.** Image arms re-encode exactly at block ends, while a map gives every write the whole block's movement, so the comparison favours images.

## Scaling

Memory costs h(32 + m + 3) floats: about 200 bytes per unit for counts and 270 for symbols, against up to eight pairs of 784-byte images. Correction keeps a copy of the key encoder (0.84 MB) and of the host's image path (about 0.9 MB) for one block. This is more than this memory at this size, but constant in facts and the networks already exist at that size. Fitting a d × d map on p probes costs O(p d² + d³) per block and per value space, and moving h units costs O(h d²). Writes and reads stay O(h d); a merge pass stays O(h² d), or O(p h d) more for the merge test on p proposed pairs. No image is re-encoded, so upkeep no longer grows with stored experiences.

## Prior work

Read for the query only: the [drift compensation family](https://arxiv.org/abs/2004.00440) (Yu et al. 2020) and [Learnable Drift Compensation](https://arxiv.org/abs/2407.08536) (Gomez-Villa et al. 2024), and [Learning without Memorizing](https://arxiv.org/abs/1811.08051) (Dhar et al. 2019), **[abstract]** each. Linear transport here is closest to the first two in aim. Their failure mode appears here: movement on current inputs need not predict movement on old ones. The [translation pilot](translation-memory-v1-pilot.md)'s anchor map also moved accuracy rather than adding to it, though that map also rewrote current codes. Ball merging resembles BIRCH **[familiar; read before use]**. Full reading remains owed before any contract, per `docs/literature-review.md`.

## Computation

Symbol hosts: GTX 1650, three workers, 469 s wall, 948 s aggregate CPU, 4 fits of 12,000 updates. Learning: same GPU and workers, 1,932 s wall, 16,927 s aggregate CPU, 16 tasks of 28 arms over both sets of values, GPU peak 90 MB, peak RSS 2.9 GB per process. Selection was by inspection, plus two smoke runs and one read-only drift check.
