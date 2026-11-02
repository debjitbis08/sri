# Memory through a slow average of the key encoder: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-03. Not evidence for any contrast, and not a contract.** Two seeds. The decay rates, the live threshold's target and the streams were chosen for this study, and no setting was confirmed on other seeds.

## Question

The [drift measurement](sense-drift-structure-v1.md) found that a parameter average removes a key encoder's jitter and costs nothing visible. If memory reads and writes through that average, with no map on keys and no image held, does it keep both sets? And can a threshold set from inputs passing now control false answers?

## Design

- **Encoders for memory:** the working encoder, averages with β 0.99, 0.999 and 0.9999, or frozen, plus 0.999 at H 512. Time constants are 100, 1,000 and 10,000 updates, in a run of 6,000. So **β 0.9999 moves only about 45% of the way to the working encoder here**.
- **Values:** counts in host codes, with the windows pilot's live map every 250 updates. Symbols in the arm's own encoder, among 150, with no map. Frozen arms keep symbol values frozen too, unlike the earlier pilots.
- **Reads** on the same memory: at τ 0.85, which memory also writes at; at a live threshold; and at matched false acceptance. The live threshold is a running estimate of the similarity at which 0.13 of inputs passing now, which memory is not taught, would get an answer.
- **Streams:** stationary, shift to dots and formats, or shift to natural images (5,123 CIFAR-10 images) from block 5. Rates: normal and 10×. Both orders.
- **Reference:** stored-image arms, which reproduce the windows pilot in all 64 comparisons. That pilot's A-then-B hosts remain unexplained, as noted there.

**A limit in the design.** Memory's contents are glyphs here, and the shifted streams teach the encoder nothing about them. So learning can only harm in these conditions, and frozen keys cannot lose. This pilot measures the cost of learning, not its benefit.

## Results

End of block 6. Ranges over two seeds and two orders; reads have 225 queries, SE about 0.03.

### Normal rate, stationary stream

Reads at τ, with no calibration:

| Arm | Counts: first, second | Loss | Symbols: first, second | Loss | Unseen | Units | KB, counts and symbols |
|---|---|---|---|---|---|---|---|
| Working | 0.43–0.62, 0.60–0.72 | 0.18 | 0.50–0.56, 0.62–0.68 | 0.28 | 0.08 | 1,444–1,674 | 313, 413 |
| β 0.99 | 0.45–0.61, 0.59–0.70 | 0.20 | 0.50–0.54, 0.61–0.73 | 0.28 | 0.08 | 955–984 | 195, 258 |
| β 0.999 | 0.46–0.63, 0.60–0.70 | 0.20 | 0.48–0.58, 0.55–0.72 | 0.24 | 0.07 | 885–918 | 180, 238 |
| **β 0.9999** | **0.61–0.69, 0.64–0.73** | **0.05** | **0.55–0.76, 0.52–0.76** | **0.05** | **0.13** | 914–929 | 185, 244 |
| Frozen | 0.61–0.69, 0.60–0.72 | 0.07 | 0.55–0.75, 0.54–0.72 | 0.05 | 0.17 | 936–948 | 188, 249 |
| Images, rebuild | 0.57–0.75, 0.60–0.72 | 0.06 | 0.61–0.71, 0.55–0.72 | 0.04 | 0.13 | 1,448–1,678 | 18,429 |
| Images, cover | 0.58–0.71, 0.58–0.69 | 0.11 | 0.60–0.70, 0.55–0.69 | 0.05 | 0.14 | 627–664 | 5,088, 5,856 |

- **β 0.9999 reads like stored images with no threshold shift,** at the same false-answer rate and about 1% of the memory payload. Its worst loss is 0.05, at the bar, on both sets of values, within noise and from settings seen here. Frozen keys do about the same.
- **β 0.99 and 0.999 do not help forgetting.** Halving movement does not move the first set: at τ 0.46–0.63 against 0.43–0.62, and matched means of 0.60–0.62 against 0.59. This contradicts what the drift measurement suggested.
- **Removing jitter does cut unit growth,** by about 40%. Fewer writes miss their own unit.
- **What protects memory here is being slower than the run,** not removing jitter. At this length, β 0.9999 is hard to tell apart from frozen.

Per seed and order, counts then symbols, matched, first/second:

| | 392 AB | 392 BA | 393 AB | 393 BA |
|---|---|---|---|---|
| β 0.9999 | 0.61/0.64, 0.72/0.76 | 0.63/0.68, 0.75/0.72 | 0.70/0.73, 0.56/0.70 | 0.68/0.70, 0.67/0.52 |
| Frozen | 0.61/0.60, 0.71/0.71 | 0.60/0.67, 0.74/0.69 | 0.68/0.72, 0.55/0.67 | 0.68/0.71, 0.68/0.54 |
| Images, rebuild | 0.62/0.60, 0.70/0.71 | 0.56/0.59, 0.70/0.69 | 0.76/0.68, 0.62/0.55 | 0.70/0.72, 0.61/0.55 |

### Ten times the rate, stationary stream

Matched, first and second: β 0.9999 reads 0.40–0.60, 0.61–0.71 on counts and 0.36–0.55, 0.63–0.72 on symbols, with loss 0.36 and 0.34. β 0.999 reads 0.31–0.56, 0.62–0.75 on counts. Frozen reads 0.62–0.69, 0.59–0.72. Images, rebuild reads 0.55–0.72, 0.19–0.35; cover 0.10–0.53, 0.08–0.34. So the same β is not slow enough here: what matters is how far the average travels, which rises with the rate.

### When the stream changes

Matched, first and second, normal rate:

| Arm | Shift: counts | Shift: symbols | Natural: counts | Natural: symbols |
|---|---|---|---|---|
| Working | 0.02–0.04, 0.33–0.61 | 0.00, 0.28–0.42 | 0.00–0.02, 0.12–0.16 | 0.00, 0.03–0.08 |
| β 0.999 | 0.00–0.04, 0.30–0.63 | 0.00, 0.27–0.46 | 0.01–0.02, 0.12–0.20 | 0.00, 0.03–0.08 |
| **β 0.9999** | **0.33–0.42, 0.56–0.62** | **0.33–0.39, 0.50–0.60** | **0.15–0.28, 0.33–0.52** | **0.11–0.21, 0.35–0.43** |
| Frozen | 0.61–0.68, 0.60–0.71 | 0.55–0.74, 0.54–0.71 | 0.61–0.68, 0.60–0.70 | 0.55–0.74, 0.54–0.71 |
| Images, rebuild | 0.33–0.53, 0.39–0.52 | 0.30–0.36, 0.28–0.33 | 0.26–0.32, 0.16–0.24 | 0.07–0.21, 0.06–0.16 |

- **At matched false answers, β 0.9999 matches or beats stored images** on the second set under change, and roughly matches them on the first. It still loses much of the first set, 0.40 at worst on counts.
- **Slowness buys time; it does not protect.** At 10× with a shift, its first set falls to 0.04–0.08 on counts and 0.01–0.04 on symbols.
- **New writes also suffer through a fast encoder.** The second set read at the end of its own block under a shift: 0.48 working, 0.58 β 0.999, 0.68 β 0.9999, on counts.

### The live threshold fails

It targets 0.13 of unseen symbols. It delivers:

| Stream | Working and faster averages | β 0.9999 | Frozen |
|---|---|---|---|
| Stationary, normal | 0.34–0.35 | 0.35 | 0.35 |
| Shift, normal | 0.59–0.67 | 0.32 | 0.12 |
| Natural, normal | 0.97–1.00 | 0.50 | 0.42 |

Inputs passing now are not a stand-in for unseen symbols. Their similarity to memory varies with what the stream shows, so the threshold drops when the stream moves away from glyphs. Reads rise only because more is answered. Rejected as tested. At β 0.9999 or frozen on a stationary stream, the plain τ already gives about the right rate.

### Capacity and storage

β 0.999 at H 512 falls to a second set of 0.33–0.46 on counts and 0.41–0.53 on symbols, matched. Merging and low capacity with β 0.9999 were not tested.

No image-free arm holds an image. For β 0.9999: payload 185–244 KB; arrays as allocated 0.70–0.83 MB; one average of the key encoder 0.84 MB; for counts, two copies of the host's image path 1.9 MB and map totals about 17 KB. That is about 3.4 MB at peak for counts and 1.7 MB for symbols, against 5.1–18.4 MB. Units grow with appearances and stop at H; the average and copies are constant in facts.

## What this does not establish

- **That the average's learning helps.** Memory's contents never need what the encoder learns here, so frozen keys cannot lose. This is the most important gap.
- **A slow average that is truly slow, at length.** The run is shorter than β 0.9999's time constant, so its movement would keep going.
- **Moving hosts for one-off values.** Symbols use the key encoder's own code.
- **Generality.** All images are 28 × 28 and grey, on one small architecture, with one change.

## Scaling

Writes and reads cost O(h d) per item. Each average costs one parameter update per training update, O(p) for p parameters. Comparing a batch with memory for the live threshold costs O(b h d), and it is rejected. Without the live threshold, slow-average upkeep does not depend on stored facts. Over a long life, the time constant must scale with how long memories must stay readable, which also delays learning.

## Prior work

Momentum encoders hold a queue of keys consistent in the same way (He et al. 2020), and parameter averaging is older (Polyak and Juditsky 1992) **[familiar; read before use]**. Neither was read for this pilot. Reading remains owed before any contract.

## Computation

GTX 1650, three workers, 4,589 s wall, 44,545 s aggregate CPU, 24 tasks of 16 arms over both sets of values, GPU peak 93 MB, peak RSS 2.9 GB per process. The live-threshold comparison and per-write loops account for most of this.
