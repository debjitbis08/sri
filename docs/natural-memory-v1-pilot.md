# Memory on natural images: pilot (seeds 392 and 393, fresh seeds 404 and 405)

Status: **pilot, 2026-11-03. Not evidence for any contrast, and not a contract.** All settings below were chosen for this study. The later stages were designed after seeing the earlier ones. One gate threshold was chosen after seeing its figures, and says so.

## Question

Earlier memory tests stored only glyphs, so a frozen encoder could not lose. If memory holds natural images, or several kinds of image, does the encoder's learning pay for memory, and what lets memory collect it with no stored images?

## Answer in short

- **Learning pays, on what is stored.** A frozen glyph encoder finds 0.05–0.06 of images from a strong view; after learning from the stream, the same network finds 0.72 and a deep one 0.93–0.95. PCA finds 0.24–0.29.
- **No image-free way tested collects that learning for old memories.** With stored images, reads rise as the encoder improves. Slow averages, an established drift compensation and a linear map all stay below simply freezing.
- **What holds is a frozen copy of the encoder per stretch of time, each with its own memory.** It is the only arm that keeps glyphs and natural images in both orders, and it holds across six kinds on seeds that played no part in setting it up. It costs one network copy and one forward pass per copy, and it answers unseen inputs too often, no less often than PCA.
- **One blind copy can ruin all reads,** and a simple check for blindness removes that, on fresh seeds too.
- **Stored images do not protect memory across a change of kind.** Re-encoding glyphs through an encoder that moved to natural images destroys them, 0.01–0.04 against 0.61–0.70 for frozen copies.
- **The earlier recommendation, a slow average, fails here.** It has to learn the new kind to be of use, and while it does it moves more than the working encoder.

## Data

Full CIFAR-10, 50,000 training and 10,000 test images, from the Hugging Face `uoft-cs/cifar10` parquet files. All 5,123 records of the partial official archive are present with the same labels; the order differs. The stream is the training split. Everything scored comes from the test split, which no stream shows. Other kinds: MNIST, tree, Thai digits, banana and Chinese numerals at 28 × 28, and the earlier glyphs.

Two kinds of fact:

- **Instance:** a named image and its own target image. Queries are new views: *mild* (crop of at least 70%, small brightness changes) or *strong* (crop of at least 20%, flip, large brightness and contrast changes, and in colour saturation and grey). Loads of 45 and 450 per set.
- **Class:** any image of a class and that class's own target. Four classes per set, two unseen. Queries are other images.

Reads pick the target among 150 candidates (mixed glyphs and sequence) or 1,500 (natural only). Abstentions are wrong. *Matched* reads set the threshold that answers 0.13 of unseen queries; this uses the unseen queries and only scores.

## Stage 1: can a network learn natural images at all?

The gate for the rest: if a frozen glyph encoder were as good, this test could not show a benefit. 40,000 updates on the stream; contrastive; 128 views per update; Adam at 1e-3.

| Encoder | Parameters | Mild view found | Strong view found | Class, 20-NN |
|---|---|---|---|---|
| Pixels | none | 0.92–0.96 | 0.18–0.22 | 0.30–0.36 |
| PCA, 32 | fit on 5,000 | 0.92–0.96 | 0.24–0.29 | 0.36–0.42 |
| Glyph encoder, frozen | 209,760 | 0.51–0.56 | 0.05–0.06 | 0.26 |
| Small, from glyph start, affine views | 209,760 | 0.95–0.96 | 0.15–0.16 | 0.38–0.39 |
| Small, from glyph start, strong views | 209,760 | 0.99 | 0.72 | 0.49–0.50 |
| Small, untrained start, strong views | 209,760 | 0.99 | 0.69–0.71 | 0.49–0.50 |
| Quarter-width ResNet-18, grey 28 | 720,528 | 1.00 | 0.93 | 0.58–0.59 |
| Quarter-width ResNet-18, colour 32 | 720,816 | 1.00 | 0.94–0.95 | 0.60–0.61 |

- **Mild views cannot reward learning:** pixels find them. Strong views and classes can.
- **The glyph encoder maps natural images almost onto one point:** the nearest other image lies at 0.97 similarity. Nothing it trained on required telling them apart.
- **Augmentation matters more than the start.** A glyph start gives no lasting advantage over an untrained one.
- **Colour adds little here;** the deep network is what helps, 0.60 against 0.59 in grey.
- **The encoder never stops moving:** 0.26–0.45 rad on gallery images in the last 8,000 updates.

For context, about 87% nearest-neighbour accuracy is reported for SimCLR on a full ResNet-18 at 200 epochs **[search summary]**. This run is a quarter of the width, about 50 epochs, Adam, so 0.60 is not the method's limit.

## Stage 2: natural images only

The encoder learns for P updates, then teaches one set for 2,000, the other for 2,000, then rests for up to 16,000, in both orders. Each arm reads and writes through an encoder of its own. Time constants of the averages are 1,000, 10,000 and 100,000 updates, all starting from the glyph encoder. Instance facts are seen eight times each, at a mild view.

Small network, load 450, **strong views**, at matched false answers, first set and second set. Ranges over two seeds and two orders. SE about 0.02.

| Taught at | Arm | End of teaching | +4,000 | +16,000 |
|---|---|---|---|---|
| P 500 | Working | 0.25–0.29, 0.53–0.55 | 0.12–0.17, 0.45–0.50 | **0.04–0.06**, 0.21–0.24 |
| | Frozen at teaching | 0.31–0.35, 0.31–0.33 | same | **0.31–0.35**, 0.31–0.33 |
| | β 0.999 | 0.11–0.20, 0.56–0.60 | 0.04–0.07, 0.44–0.47 | 0.01–0.02, 0.16–0.20 |
| | β 0.9999 | 0.00, 0.00 | 0.00, 0.00 | 0.00, 0.00 |
| | SDC on working | 0.34–0.38, 0.51–0.55 | 0.19–0.23, 0.45–0.49 | 0.07–0.12, 0.26–0.30 |
| | Linear map on working | 0.31–0.34, 0.54–0.56 | 0.17–0.22, 0.46–0.49 | 0.05–0.09, 0.18–0.20 |
| | Stored images, rebuilt | 0.56–0.59, 0.55–0.58 | 0.60–0.64, 0.60–0.62 | **0.62–0.68, 0.62–0.69** |
| | PCA | 0.24–0.28, 0.25–0.28 | same | same |
| P 4,000 | Working | 0.51–0.56, 0.58–0.60 | 0.48–0.50, 0.56–0.60 | 0.27–0.34, 0.42–0.50 |
| | Frozen at teaching | 0.52–0.57, 0.53–0.58 | same | 0.52–0.57, 0.53–0.58 |
| | β 0.999 | 0.55–0.60, 0.60–0.65 | 0.45–0.51, 0.60–0.64 | 0.24–0.28, 0.43–0.52 |
| | β 0.9999 | 0.00–0.01, 0.03–0.05 | 0.00, 0.02–0.03 | 0.00, 0.00 |
| | SDC on working | 0.50–0.55, 0.57–0.59 | 0.47–0.51, 0.56–0.59 | 0.33–0.40, 0.44–0.51 |
| | Stored images, rebuilt | 0.58–0.61, 0.59–0.61 | 0.61–0.66, 0.62–0.64 | 0.62–0.69, 0.64–0.69 |
| P 16,000 | Working | 0.62–0.68, 0.63–0.65 | 0.60–0.67, 0.62–0.66 | 0.56–0.64, 0.58–0.65 |
| | Frozen at teaching | 0.62–0.68, 0.62–0.67 | same | 0.62–0.68, 0.62–0.67 |
| | β 0.9999 | 0.41–0.51, 0.45–0.52 | 0.36–0.50, 0.46–0.54 | 0.20–0.30, 0.34–0.42 |
| | SDC on working | 0.60–0.66, 0.64–0.65 | 0.62–0.66, 0.62–0.65 | 0.55–0.62, 0.58–0.65 |
| | Stored images, rebuilt | 0.64–0.70, 0.63–0.67 | 0.64–0.69, 0.63–0.68 | 0.66–0.70, 0.66–0.71 |

β 0.99999 reads 0.00 throughout: it never learns natural images in these runs.

- **The benefit of later learning is real and depends on when a memory was made.** Rebuilt minus frozen on the first set at the end: about 0.31 at P 500, 0.10 at P 4,000, 0.03 at P 16,000.
- **No image-free arm collects it.** At P 500 the best without images is frozen, 0.31–0.35, about half of what stored images reach.
- **Slow averages lose both ways.** Too slow to have learned, they cannot tell the images apart. While they catch up they move more than the working encoder: over the same stretch at P 16,000, 0.69 rad for β 0.9999 against 0.47. β 0.999 behaves like the working encoder.
- **Compensation helps a little and stays below frozen** at every P: 0.04–0.06 to 0.07–0.12 at P 500, and 0.27–0.34 to 0.33–0.40 at P 4,000. SDC does better than the global map. This is its best case, since the stream shows the same kind as memory.
- **A trained encoder is nearly a stable one.** At P 16,000 the working encoder loses little over 20,000 updates. The harm comes from early learning.

On mild views, frozen at P 16,000 reads 0.99, 0.98 and PCA 0.93–0.96. **Class facts are weak everywhere:** with no threshold, address reads at P 16,000 are 0.36–0.49 for frozen and 0.38–0.53 for stored images, where chance is 0.125; at matched false answers 0.03–0.21. Telling an unseen class from a known one is hard on these codes, as expected, and one-winner reads lose against the 20-NN figures above.

**Deep network, colour:** see [Deep](#deep-network).

## Stage 3: glyphs and natural images in one memory

Small network. **GN:** glyph facts on a glyph stream, then the stream turns natural, 4,000 updates, natural facts, rest. **NG:** natural first and back to glyphs. One memory holds both. End of run, matched false answers, ranges over two seeds.

| Arm | GN: glyphs, natural mild, natural strong | NG: glyphs, natural mild, natural strong | Unseen answered at τ: glyph, natural |
|---|---|---|---|
| Working | 0.00, 0.98, 0.73–0.84 | 0.61–0.64, 0.00, 0.00 | 0.00, 0.00–0.02 |
| β 0.999 | 0.00, 0.96–1.00, 0.62–0.82 | 0.63–0.68, 0.00, 0.00 | 0.00, 0.00 |
| β 0.9999 | 0.01–0.02, 0.00–0.02, 0.00 | 0.53–0.57, 0.02–0.04, 0.00–0.02 | 0.00–0.06, 0.00–0.20 |
| Frozen glyph encoder | 0.61–0.70, 0.07–0.16, 0.00 | 0.58–0.72, 0.07–0.11, 0.00–0.02 | 0.04–0.10, 0.69–0.87 |
| Stored images, rebuilt | 0.01–0.04, 1.00, 0.76–0.84 | 0.72, 0.11–0.16, 0.00 | 0.01–1.00, 0.00–0.87 |
| PCA | 0.07–0.12, 1.00, 0.24–0.31 | 0.04–0.07, 0.93–0.98, 0.24–0.38 | 0.21–0.38, 0.00–0.02 |
| **One copy per taught block** | **0.61–0.70, 1.00, 0.62–0.73** | **0.71–0.74, 0.98–1.00, 0.64–0.84** | **0.04–0.10, 0.00** |

- **A single code loses one kind whichever way it goes.** This is the trade the study set out to show.
- **Stored images do not escape it.** Rebuilding through a current encoder that no longer tells the old kind apart reads as badly as anything.
- **Copies escape it,** as well as the best arm on each kind. Each copy holds its memory in the code it wrote with; a read tries each and takes the most similar answer. 100–125 KB of memory payload, as against 134–174 KB for working, plus one 0.84 MB copy per stretch.

### Telling it when to fork is not needed, if it forks late enough

Copies at a fixed interval, with no knowledge of the blocks. Copies with no writes are replaced, so they count only where something was written. `every-500` and `every-2000` are aligned with the blocks by construction; 700, 1,300 and 3,100 are not.

| Interval | Glyphs | Natural mild | Natural strong | Copies | Unseen at τ: glyph, natural |
|---|---|---|---|---|---|
| 700 | 0.58–0.74 | 0.98–1.00 | 0.58–0.82 | 7–8 | 0.04–0.10, 0.00–0.04 |
| 1,300 | 0.60–0.72 | 0.98–1.00 | 0.60–0.80 | 5 | 0.04–0.11, 0.00–0.04 |
| 3,100 | 0.61–0.74 | 1.00 | 0.64–0.80 | 2–3 | 0.04–0.10, 0.00–0.04 |
| 8,000 | 0.61–0.70 | 0.07–0.16 | 0.00–0.02 | 1–2 | 0.04–0.10, 0.69–0.87 |

The requirement: a copy must be made **after** the encoder has learned the kind it will write. At 8,000 the natural facts go through the glyph copy.

### With no prelude it fails

Teaching starts when the stream changes, so the copy at the block's start is blind to the new kind.

| Arm | Glyphs | Natural mild | Natural strong | Copies | Unseen at τ: glyph, natural |
|---|---|---|---|---|---|
| One copy per block | 0.00–0.70 | 0.04–0.11 | 0.00–0.02 | 2 | 0.04–0.98, 0.76–0.89 |
| Every 700 | 0.18–0.68 | 0.47–0.71 | 0.09–0.18 | 6 | 0.09–0.96, 0.82–0.93 |
| Every 100 | 0.35–0.63 | 0.69–0.87 | 0.13–0.24 | 40 | 0.08–0.96, 0.69–0.91 |
| Working | 0.00–0.59 | 0.00–0.58 | 0.00–0.42 | 1 | 0.00, 0.00 |

Faster forking helps a good deal, but needs 40 copies and answers unseen inputs far too often.

## Stage 4: six kinds in turn

Natural, MNIST, tree, Thai digits, banana, Chinese numerals, each for 2,000 updates of stream and 2,000 of teaching, with 45 facts per kind, one memory. A seeded order and its reverse. Read at the end. Means over the six kinds, ranges over runs.

**Pilot seeds 392, 393, from the glyph encoder:**

| Arm | Mild, matched | Mild, τ | Strong, matched | Strong, τ | Unseen answered at τ, mild | Copies |
|---|---|---|---|---|---|---|
| Working | 0.11–0.16 | 0.10–0.15 | 0.04–0.12 | 0.05–0.10 | 0.00–0.04 | 1 |
| Frozen glyph encoder | 0.46–0.50 | 0.55–0.59 | 0.06–0.08 | 0.07–0.09 | 0.41–0.43 | 1 |
| PCA | 0.46–0.54 | 0.54–0.56 | 0.12–0.13 | 0.10–0.13 | 0.33–0.36 | none |
| **One copy per block** | **0.63–0.76** | **0.72–0.81** | **0.35–0.40** | **0.43–0.46** | 0.27–0.39 | 6 |
| Every 700 | 0.66–0.80 | 0.77–0.86 | 0.39–0.43 | 0.49–0.51 | 0.37–0.46 | 23 |
| Every 1,300 | 0.67–0.73 | 0.76–0.82 | 0.36–0.41 | 0.48–0.51 | 0.34–0.45 | 18 |
| Every 3,100 | 0.41–0.74 | 0.54–0.82 | 0.18–0.35 | 0.34–0.45 | 0.38–0.77 | 8 |

Stored images, rebuilt, reads 0.19–0.35 on mild at matched false answers, with 0.61–0.83 of unseen inputs answered at τ. The working encoder reads the most recent kind at 0.53–0.98 and almost nothing else, at most 0.16.

**Fresh seeds 404, 405, from an untrained network:** one copy per block 0.52–0.73 on mild, 0.29–0.43 on strong; every 700, 0.64–0.75, 0.32–0.41; every 1,300, 0.60–0.72, 0.27–0.41; working 0.09–0.17, 0.05–0.13; PCA 0.51–0.59, 0.15–0.19. So copies hold on seeds that played no part in the design. On mild views they beat PCA by a smaller amount there. On strong views the gap is about 0.2.

### One blind copy poisons all reads

On the fresh seeds, **every 3,100 reads 0.00–0.04 and answers 1.00 of unseen inputs.** Its first copy is the untrained network, and it took writes. It gives every input a similarity near 1, so it wins every read from then on, even for kinds other copies know. On the glyph seeds the same happens more mildly (0.41 in one order).

### A gate on blindness

A copy that piles recent inputs onto one point cannot tell them apart. So measure the squared length of the batch's mean code, about the mean pairwise cosine, in O(n d), with no labels and nothing stored. Above a threshold: a write forks a fresh copy, or is refused if the working encoder is blind too; and a read skips that copy. Set at 0.5 before seeing it.

| Mean over kinds, matched | Every 3,100 | Gated | Every 1,300 | Gated | One per block | Gated |
|---|---|---|---|---|---|---|
| Fresh seeds, mild | 0.00–0.04 | **0.60–0.73** | 0.60–0.72 | 0.67–0.72 | 0.52–0.73 | 0.60–0.73 |
| Fresh seeds, strong | 0.00 | **0.34–0.41** | 0.27–0.41 | 0.40–0.44 | 0.29–0.43 | 0.37–0.45 |
| Pilot seeds, mild | 0.41–0.74 | 0.62–0.76 | 0.67–0.73 | 0.69–0.73 | 0.63–0.76 | 0.64–0.76 |
| Pilot seeds, strong | 0.18–0.35 | 0.37–0.39 | 0.36–0.41 | 0.40–0.42 | 0.35–0.40 | 0.39–0.41 |
| No prelude, mild | 0.31–0.37 | 0.42–0.53 | 0.35–0.43 | 0.53–0.59 | 0.14–0.26 | 0.36–0.49 |
| No prelude, strong | 0.06–0.17 | 0.09–0.25 | 0.11–0.16 | 0.20–0.27 | 0.03–0.10 | 0.08–0.17 |

- **It removes the poisoning.** Fresh seeds move from 0.00–0.04 to 0.60–0.73, and to the same level as other intervals.
- **It helps most without a prelude,** where copies are often blind.
- **At this threshold it barely touches false answers at τ when copies are good:** 0.27–0.39 to 0.27–0.37 on the pilot seeds. Unseen answers fall on fresh seeds at the bad interval, from 1.00 to 0.37–0.54.

### False answers remain

The figures above suggested a tighter gate: a kind's own copy collapses it to 0.03–0.34, and other copies have quartiles of 0.32, 0.46 and 0.68–0.70. So a threshold of 0.25 was **chosen after seeing this**, on the pilot seeds, and run on the fresh seeds as a check. Means over kinds, mild then strong:

| Arm | Seeds | Matched: no gate | Matched: gate 0.25 | Unseen at τ: no gate | Unseen at τ: gate 0.25 | Copies read, of all |
|---|---|---|---|---|---|---|
| One per block | Pilot | 0.63–0.76, 0.35–0.40 | 0.64–0.76, 0.41–0.42 | 0.27–0.39, 0.25–0.30 | 0.26–0.30, 0.18–0.24 | 1.5–2.0 of 6–7 |
| | Fresh | 0.52–0.73, 0.29–0.43 | 0.65–0.74, 0.39–0.50 | 0.32–0.69, 0.28–0.67 | 0.25–0.34, 0.12–0.29 | 0.8–1.7 of 6–10 |
| Every 700 | Pilot | 0.66–0.80, 0.39–0.43 | 0.67–0.80, 0.43–0.47 | 0.37–0.46, 0.31–0.43 | 0.31–0.36, 0.28–0.33 | 5.7–7.2 of 23 |
| | Fresh | 0.64–0.75, 0.32–0.41 | 0.68–0.77, 0.45–0.52 | 0.47–0.87, 0.45–0.85 | 0.36–0.42, 0.16–0.37 | 3.7–6.7 of 23–24 |
| Every 3,100 | Pilot | 0.41–0.74, 0.18–0.35 | 0.66–0.79, 0.40–0.43 | 0.38–0.77, 0.37–0.67 | 0.27–0.32, 0.24–0.29 | 2.0–4.0 of 10–11 |
| | Fresh | 0.00–0.04, 0.00 | 0.63–0.73, 0.36–0.44 | 1.00, 1.00 | 0.28–0.39, 0.14–0.27 | 1.7–2.8 of 10–14 |

The tighter gate holds on the fresh seeds, on reads and on false answers. **False answers at the plain τ are lower and are still too high**: 0.25–0.42 on mild against PCA's 0.29–0.38, so no better than a fixed code. Few copies answer a given batch, though each must still be run on a recent batch to decide that.

A margin read also failed: the distance from the best unit to the mean of the next k, in the manner of CSLS (**[familiar]**; the paper's abstract does not describe it). It separates known from unseen worse than raw similarity, on mild views AUC 0.68–0.83 against 0.83–0.95 across copy arms, and matched reads fall from 0.41–0.80 to 0.15–0.49. A copy holding few units gives anything a wide margin. Rejected as tested.

Raw similarity's AUC of 0.93–0.95 says that the information is there. The fixed threshold of 0.85 is what does not suit every copy.

## Deep network

Quarter-width ResNet-18 on colour images, P 4,000 and 16,000, two seeds, both orders; gates pass in all 8 tasks. Load 450, strong views, matched false answers, first set and second set.

| Taught at | Arm | End of teaching | +4,000 | +16,000 |
|---|---|---|---|---|
| P 4,000 | Working | 0.54–0.59, 0.73–0.75 | 0.42–0.50, 0.66–0.73 | **0.18–0.24**, 0.45–0.51 |
| | Frozen at teaching | 0.65–0.68, 0.63–0.71 | same | **0.65–0.68**, 0.63–0.71 |
| | β 0.999 | 0.66–0.69, 0.80–0.84 | 0.48–0.52, 0.76–0.81 | 0.21–0.26, 0.55–0.60 |
| | β 0.9999 | 0.00–0.01, 0.05–0.24 | 0.00, 0.02–0.15 | 0.00, 0.00–0.01 |
| | Stored images, rebuilt | 0.78–0.81, 0.78–0.82 | 0.83–0.84, 0.82–0.85 | **0.86–0.88, 0.86–0.89** |
| | PCA | 0.23–0.28, 0.23–0.27 | same | same |
| P 16,000 | Working | 0.82–0.87, 0.83–0.89 | 0.81–0.85, 0.84–0.85 | 0.78–0.86, 0.82–0.88 |
| | Frozen at teaching | 0.82–0.86, 0.82–0.86 | same | 0.82–0.86, 0.82–0.86 |
| | β 0.999 | 0.86–0.89, 0.88–0.89 | 0.85–0.89, 0.87–0.90 | 0.84–0.87, 0.86–0.87 |
| | β 0.9999 | 0.77–0.84, 0.79–0.85 | 0.73–0.83, 0.78–0.86 | 0.50–0.74, 0.71–0.82 |
| | Stored images, rebuilt | 0.85–0.88, 0.84–0.88 | 0.85–0.89, 0.85–0.89 | 0.89–0.91, 0.89–0.91 |

- **The same pattern, more clearly.** At P 4,000 the working encoder falls to 0.18–0.24, frozen stays at 0.65–0.68, and stored images rise to 0.86–0.88: a gap of about 0.2 that no image-free arm collects.
- **Once trained, it hardly matters.** At P 16,000 working, frozen and β 0.999 all read about 0.8–0.9. β 0.999 reads as well as anything there, within noise. The working encoder's cue movement over those 20,000 updates is 0.57 rad, and it keeps reads.
- **β 0.9999 remains worse than either,** moving 1.31 rad against 0.92 at P 4,000.
- **Against a mild view, PCA still reads 0.95–0.97,** so learning shows on strong views: 0.82–0.86 against 0.23–0.30.
- **Classes:** with no threshold, addresses read 0.42–0.59 for frozen and 0.47–0.63 for stored images at P 16,000.
- **Copies were not run on this network.** The glyph encoder has no deep version, so the mixed and sequence stages are small-network only.

## What this does not establish

- **That copies help a deep network.** They ran on the small network only.
- **That copies scale.** Six kinds and up to 23 copies. Cost grows with time spent writing, in copies and read passes, and nothing ever merges them. **A real system needs fewer.**
- **That old memories can gain from later learning without images.** They do not here. A memory stays as good as its copy was at writing. Doing better needs rehearsal, which is the separate mechanism kept for images, or something not tested here.
- **Rejection.** False answers stay at 0.25–0.42 on mild views, no better than PCA. See above.
- **The value of mild-view instance reads.** PCA and pixels do these. Strong views and fresh seeds are the informative figures.
- **Generality.** Small and quarter-width networks, 28 × 28 grey for most of it; 45–450 facts; CIFAR-10 at only about 50 epochs. Families of rendered digits contain near duplicates, which lowers instance reads there whatever the arm.
- **Why glyphs top out near 0.7.** That is the value read among 150; address reads are about 0.96–1.00.
- **Run design.** Teaching is announced in the sense that writes happen in blocks; the stream changes abruptly and once per kind. Two pilot seeds, then fresh seeds for sequence runs only. The stream never contains the facts.

## Scaling

- **Writes and reads** cost O(h d) per item, for h units, as before.
- **Copies:** storage O(V p) for V copies of p parameters, and a read needs V forward passes, O(V f) for a forward cost f, plus O(h d) over all units. At the small network, 0.84 MB per copy. V grows with the time spent writing, not with facts or the time between, since a copy with no writes is replaced. An expert-gate style selection would cut read passes to about one **[derived]**, and is untested.
- **The blindness gate** costs O(n d) on n inputs passing, per copy, at each write, and the same per read batch.
- **Slow averages** cost O(p) per update and do not scale with facts; they fail here regardless.

## Prior work

[Reading notes](natural-memory-prior-work.md) on MoCo, SDC, self-healing codes, Expert Gate, Progressive Networks and REMIND. Frozen copies are in the family of one network per task, which is known to be immune to forgetting; what differs here is that copies are per stretch of time on an unlabelled stream, with memory in each copy's own code, and no task told. That is not claimed as new. SDC's own fixed-after-first-task baseline beats finetuning with compensation on Flowers, 76.9 against 35.3 **[read]**, which agrees with this study. Not yet read: CSLS itself, SCALE, CaSSLe, CLIMB in full.

## Computation

GTX 1650, 12 cores, 15 GB. Several stages shared the GPU, so wall figures are not per-arm costs and timing is not comparable across stages.

| Stage | Wall s | Aggregate CPU s | Peak RSS per worker | GPU peak |
|---|---|---|---|---|
| Encoders: 10 fits of 40,000 updates, 8 fixed baselines | 9,415 | 22,999 | 2.1 GB | 270 MB |
| Memory, small: 12 tasks of 20,500–36,000 updates | not recorded as one figure | 14,534 | 1.9 GB | see runtime file |
| Transport: 12 tasks | 2,259 | 6,310 | 2.3 GB | 88 MB |
| Glyphs and natural: 4 tasks | 1,375 | 2,492 | 2.7 GB | 85 MB |
| Blind, misaligned, no prelude: 4 tasks each | 1,466; 1,431; 734 | 2,689; 2,596; 1,319 | 2.8 GB | 97–128 MB |
| Six kinds, plain and with margins: 4 tasks each | 1,359; 1,307 | 2,646; 2,530 | 2.7 GB | 130 MB |
| Six kinds, fresh: 4 tasks | 653 | 1,230 | 2.7 GB | 130 MB |
| Six kinds, gated: pilot, fresh, no prelude | 3,002; 3,072; 2,153 | 5,931; 6,061; 4,234 | 2.7 GB | 161–178 MB |
| Six kinds, gate 0.25: pilot, fresh | 2,854; 1,132 | 5,637; 2,201 | 2.7 GB | 185 MB |
| Memory, deep: 8 tasks | 12,600 | 34,801 | 3.5 GB | 284 MB |

Rebuild arms hold images and count as a reference only. Selection cost: these stages are the selection; nothing was confirmed separately except that the sequence ran on fresh seeds.

Fetching: 143 MB from Hugging Face in under a minute; the official server was about 50 times slower.

**Gates.** Working encoders in the memory stage match the encoder stage by state in 20 of 20 tasks across both networks; transport runs match memory runs on all 864 shared reads; the told arm matches `every-2000` exactly; the margin run reproduces 132 of 132 reads per task of the first sequence run. Runtime files record the runner's hash at a stage's end, and the runners were extended between stages, so a hash is not always the code a stage started with; those additions only added arms, and the checks above show shared arms unchanged.

Reports: `reports/natural-memory-v1*.json.gz`. Source files: `/data/external-downloads/cifar10-hf`, converted to `runs/external/cifar10/cifar10-full.npz`, neither in git.
