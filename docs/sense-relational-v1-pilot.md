# Sense-level relational machine: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast, and not a contract.** Two seeds; arms, augmentations, the font test and the memory grid's upper range were chosen during the study. Runners `experiments/sense_relational.py`, `experiments/sense_memory.py`; reports `reports/sense-relational-v1-*.json`, `reports/sense-memory-v1-*.json`.

**A limit on all of this (user, 2026-11-02).** These inputs are simple, one-font glyphs with small changes, so there may be few relations to find. Cosine beating a comparator may hold only for this reason. On richer inputs, geometric similarities would be expected to show up as relations. So nothing here says sense-level relations are unnecessary. Follows the [learning senses pilot](shared-core-sparse-memory-v1-learning-senses-pilot.md).

## Question

The user's proposal: the encoder itself could be a relational machine, as the core is, at every sense. The core's "same" means same count, so it keeps counts and collapses identity. A sense-level "same" would mean the same thing seen again. Step 1 asks whether identity remains in the host's encoder when read by something better than cosine. Step 2 trains a small encoder from scratch with such a relation, and asks whether it needs symbol labels.

Unseen symbols throughout: 90 from the test pool, prototypes from 64 renderings, 5 held-out renderings per symbol, chance 0.011.

**The frozen encoder's labels are weaker than this project has described.** Its supervised contrastive loss samples 64 different symbols and two renderings of each, so each item has exactly one positive. That is the same objective as contrastive learning on pairs. It knows only that two renderings go together and that the batch's symbols differ.

## Step 1: is identity still in the host's encoder?

Heads trained on stored features of the 1,987 background symbols, with two renderings as the pair, 4,000 steps, with features standardized. Raw cosine for comparison: 0.26, 0.27 on the encoder output, 0.15, 0.19 on the code. Identity at 90 symbols:

| Features | Linear head, cosine | Small MLP head, cosine | Comparator |
|---|---|---|---|
| Host encoder output, block 3 | 0.44, 0.55 | **0.60, 0.62** | 0.42, 0.41 |
| Host code, block 3 | 0.30, 0.34 | 0.40, 0.41 | 0.24, 0.23 |
| Untrained encoder, control | 0.19, 0.14 | 0.38, 0.42 | 0.18, 0.14 |

- **The external review was right.** Poor cosine identity did not mean identity was absent. A small learned head more than doubles it.
- **It is still short of the 0.80 bar, and of a purpose-trained encoder.** It also beats the untrained control by only about 0.2, so much of what the MLP finds may be generic to any convolutional features.
- **A comparator does worse here than a projection read by cosine.**

## Step 2: machines trained from scratch

The frozen encoder's architecture, 32-d, unit output, 8,000 steps on background symbols, without labels on the images. Read at 90 symbols, cosine on the code; with a comparator, also through it.

| Same-thing source | Matching | Cosine on code | Comparator | Top direction's share |
|---|---|---|---|---|
| Two renderings (the frozen encoder) | Cosine | **1.00, 1.00** | | 0.14, 0.15 |
| Two renderings | Dense comparator | 0.11, 0.95 | 0.01, 0.94 | 0.51, 0.54 |
| Two renderings | Sparse comparator | 0.66, 0.38 | 0.30, 0.25 | 0.88, 0.92 |
| Augmentations, no symbol knowledge | Cosine | **0.96, 0.94** | | 0.12, 0.13 |
| Augmentations, no symbol knowledge | Dense comparator | 0.72, 0.67 | 0.74, 0.72 | 0.40, 0.40 |
| Augmentations, no symbol knowledge | Sparse comparator | 0.16, 0.18 | 0.08, 0.06 | 0.90, 0.93 |

- **Identity without labels is reachable.** Contrastive learning on augmentations of single images, in batches of random images, reads 0.94–0.96. It clears the 0.80 bar, and also reads 0.89–0.90 at 180 symbols.
- **The relation at sense level is right, and learned matching is not what helps.** Plain cosine beats every comparator. The dense comparator is also unstable: on seed 392 with two renderings it failed outright, at chance loss, with every similarity at 1.00.
- **This sparse design collapses the code.** The comparator here is the core's reader shape, 48 sparse first-layer units and a dense second layer, but with only one reader and one relation, and one coordinate per operand. Live units read 1–4 of 32 coordinates, and the top direction holds 0.88–0.93 of the variance. [Reading more of the code removes this](#fan-in-and-two-stage-reading).

## How hard the task is

Every pool symbol is drawn in one font. Its renderings vary only in rotation, scale, position, brightness and noise, which is close to what the augmentations imitate. So these figures, and the frozen encoder's 1.00 throughout this project, may flatter. Retrained with saved weights, on new initial weights, with weaker augmentations and with baselines. "Other font": stored in the symbol's font, queried in another that draws it, for 90 of the 497 pool symbols that have one.

| Encoder | Identity, 90 | Identity, 180 | Same font, these 90 | **Other font** |
|---|---|---|---|---|
| Raw pixels | 0.14, 0.18 | 0.10, 0.13 | 0.16, 0.14 | 0.14, 0.11 |
| Untrained encoder | 0.12, 0.13 | 0.07, 0.08 | 0.13, 0.13 | 0.09, 0.08 |
| Two renderings (frozen encoder) | 1.00, 1.00 | 1.00, 0.99 | 1.00, 0.98 | **0.72, 0.74** |
| Augmentations, full | 0.94, 0.91 | 0.87, 0.88 | 0.90, 0.90 | **0.59, 0.66** |
| Augmentations, no rotation or scale | 0.89, 0.92 | 0.83, 0.89 | 0.86, 0.92 | **0.62, 0.70** |
| Augmentations, shift only | 0.84, 0.82 | 0.77, 0.74 | 0.83, 0.83 | **0.59, 0.58** |

- **The task is not trivial.** Pixels and untrained networks read 0.12–0.18, and shift alone, which does not copy rotation, scale or noise, still reaches 0.82–0.84.
- **A new font costs much, and all learned encoders still read far above chance.** The frozen encoder falls about 0.27, and unlabelled ones about 0.25–0.30. The stored address in the frozen encoder has about 1.00 within a font and 0.72–0.74 across fonts, which is a fairer statement of what "identifies symbols" has meant here.
- **Unlabelled augmentation is about 0.1 below rendering pairs across fonts.** Neither was shown a second font of the same symbol. The background symbols are also one font each.

## Fan-in and two-stage reading

The user's two proposals, after the above: read much more of the input than one coordinate, and use cosine as a cheap first stage. Same reader shape, augmentation source, 8,000 steps. Cosine shortlists K stored symbols, and the comparator picks among them, K passes per query. K 1 is cosine alone; K equal to all is the comparator alone.

The encoder trains through the comparator alone, jointly with cosine, or not at all (comparator on top of a finished unlabelled encoder).

| Arm | Cosine alone, 90 | Comparator alone, 90 | Cosine alone, other font | Comparator alone, other font | Top direction | Coordinates read |
|---|---|---|---|---|---|---|
| Through, one coordinate, RigL (earlier) | 0.16, 0.18 | 0.08, 0.06 | | | 0.90, 0.93 | 4, 1 |
| Through, k 4, RigL | 0.63, 0.68 | 0.68, 0.71 | 0.32, 0.43 | 0.31, 0.41 | 0.46, 0.45 | 32, 32 |
| Through, k 8, RigL | 0.68, 0.68 | 0.68, 0.69 | 0.33, 0.38 | 0.31, 0.39 | 0.42, 0.48 | 32, 32 |
| Through, k 16, RigL | 0.74, 0.67 | 0.73, 0.66 | 0.37, 0.38 | 0.31, 0.36 | 0.41, 0.49 | 32, 32 |
| Through, k 8, fixed masks | 0.67, 0.70 | 0.72, 0.72 | 0.33, 0.40 | 0.34, 0.41 | 0.40, 0.45 | 32, 32 |
| Joint, dense | 0.92, 0.90 | 0.91, 0.89 | 0.61, 0.66 | 0.51, 0.60 | 0.13, 0.16 | |
| Joint, k 8, RigL | 0.90, 0.92 | 0.88, 0.88 | 0.58, 0.62 | 0.52, 0.57 | 0.13, 0.17 | 32, 32 |
| On top, dense | 0.94, 0.91 | 0.92, 0.89 | 0.59, 0.66 | 0.56, 0.62 | 0.10, 0.13 | |
| On top, k 8, RigL | 0.94, 0.91 | 0.91, 0.88 | 0.59, 0.66 | 0.54, 0.61 | 0.10, 0.13 | 32, 32 |

- **Reading more removes the collapse.** At k 4 every live unit stays alive, all 32 coordinates are read, and identity rises from 0.16–0.18 to about 0.65. Beyond k 4, more fan-in changes little within noise, and RigL does not beat fixed masks here. A sparse reader then does as well as a dense one, 0.74, 0.72 trained the same way.
- **Training through a comparator is still much weaker than contrastive cosine:** 0.63–0.74 against 0.91–0.96, with the code on fewer directions.
- **A sparse reader can share an encoder without harming it.** Joint with k 8 keeps 0.90, 0.92 and a top direction of 0.13, 0.17. This is the shape the design would want, a contrastive code with sparse readers attached.
- **The second stage adds nothing to identity on a good code, and takes a little away.** On the finished encoder, other-font identity goes from 0.59, 0.66 to about 0.54–0.56, 0.61–0.62, and it helps by about 0.04 only on the weak codes at k 4. The comparator has seen no second font, and its negatives are random images rather than near neighbours, so this is not a fair test of what a reranker could do.
- **Cosine does work as a cheap first stage.** A shortlist of 5 gives the comparator's accuracy to within 0.01 at 90 symbols, for 5 passes per query instead of 90.

## Step 3: memory on the unlabelled encoder

Local memory addressed by the unlabelled encoder, k 1. Values and reading stay on block-3 host codes, so differences come from addresses. On frozen keys, the contract's selection rule, both orders; no setting was eligible on a first grid up to H 512 at τ 0.5, 0.6, 0.7, 0.85, so τ 0.75, 0.8 and H 1024, 2048 were added after seeing that. Memory level, then the rendering-pairs encoder for comparison:

| Keys, selected setting | Old after old | Old after both | New after both, by order | Worst old loss | Units | Bytes | Unknown answered |
|---|---|---|---|---|---|---|---|
| Unlabelled, H 1024, τ 0.85 | 0.92, 0.88 | 0.89, 0.84 | 0.85, 0.90 | 0.04 | 942 | 184,926 | 0.17 |
| Rendering pairs, H 128, τ 0.85 | 0.95, 0.91 | 0.93, 0.89 | 0.88, 0.94 | 0.02 | 118 | 23,324 | 0.13 |

Among candidates the unlabelled keys read 0.58–0.66, against 0.59–0.66.

- **Memory holds on keys trained with no symbol knowledge,** and A loses at most 0.04 across B.
- **It takes about eight times the units.** Renderings of one symbol spread more in this code, with same-symbol similarity about 0.83 against 0.96. So at τ 0.85 a symbol's renderings often allocate separate units, about 10 per symbol. Lower τ merges symbols, and A drops to 0.68–0.92 at τ 0.7–0.75. An explicit store on these keys reads 0.91–0.98, so the keys are good and this cost comes from the threshold rule.
- **Across fonts, memory reads 0.45, 0.48, abstaining on 0.40, 0.37 of queries;** in the same font, 0.85, 0.80. The rendering-pairs encoder reads 0.53, 0.49, abstaining on 0.33, 0.39. An explicit prototype reads these at 0.59–0.66 and 0.72–0.74, so the threshold costs a further 0.14–0.25 across fonts.

**With the encoder still learning,** from augmentations with no symbol knowledge, at τ 0.85, A then B, memory level at the end of block 6:

| Addresses | Rate 0.001, H 2048: A | B | Units | Rate 0.01, H 2048: A | B | Forced writes |
|---|---|---|---|---|---|---|
| Frozen, H 1024 | 0.91, 0.87 | 0.85, 0.85 | 948, 939 | 0.91, 0.87 | 0.85, 0.85 | 0 |
| No rebuild | 0.64, 0.76 | 0.84, 0.84 | 1,641, 1,495 | 0.29, 0.35 | 0.32, 0.33 | 18,019, 16,978 |
| Eight traces, every 2,000 | **0.91, 0.93** | **0.92, 0.91** | 1,636, 1,492 | 0.88, 0.91 | 0.46, 0.51 | 15,552, 14,045 |
| All traces, every 250 | 0.92, 0.92 | 0.93, 0.90 | 1,943, 1,741 | 0.94, 0.87 | 0.51, 0.58 | 19,110, 18,169 |

Cosine with the starting code ends at 0.83–0.85 at rate 0.001, and 0.66–0.71 at rate 0.01. Identity stays at 0.94–0.95.

- **At normal drift, this is the first run with no symbol knowledge and learning keys, and it holds.** With eight traces every 2,000 updates A falls 0.03, 0.02 across block 5, and block 6 changes nothing. Answers among candidates are A 0.59, 0.68 and B 0.60, 0.67.
- **At H 1024 the same run fills:** B reads 0.71, 0.73.
- **At ten times the rate, 2,048 units fill before B arrives,** and A holds at 0.87–0.94 while B reads about half. No rebuild loses both.
- **So unit growth is now the main limit.** About 1,500 units, with eight traces each, comes to about 9.4 MB of traces and 300 KB of addresses and contents, for 90 facts. Merging duplicates has not been tried, and this result makes it the next thing to test.

## What this does not establish

- **Learning from a stream.** The encoder trained offline on a fixed background set, with its symbol images curated beforehand. In step 3's learning arm it keeps learning, but its updates still sample that set, not what memory is being taught.
- **Everything moving at once, in both orders.** Step 3's values and reading stay on block-3 host codes, so addresses and values have not moved together. Its learning arm ran A then B only.
- **The relation to the core.** This encoder is separate from the host. Nothing here tests one trunk serving both, or taps, which have no such encoder yet.
- **The choice of augmentations.** They were designed knowing how renderings vary. In a real sense this has to come from experience, such as nearby moments in time.

## Prior work

The closest methods are Siamese networks, relation networks, and contrastive learning such as SimCLR, in which augmentation choice is known to matter. A learned-comparator advantage and instability both appear in that literature. All **[familiar; read before use]**; none read for this pilot, so this needs doing before any contract. This pilot reproduces a known method on this project's material and claims nothing new about it.

## Computation

GTX 1650, three workers, one thread each, float32. Step 1: 95 s wall, 266 s aggregate CPU, 18 fits. Step 2: 291 s wall, 810 s, 10 fits; about 100 s per unlabelled fit, 60 s otherwise, as run. Font test: 140 s wall, 388 s, 12 tasks, 6 of them fits. Fan-in: 540 s wall, 1,488 s, 16 fits, about 105 s each, or 78 s on a finished encoder. Memory: frozen grid on CPU, six workers, 88 s wall, 871 s aggregate CPU, 120 tasks in all; with learning keys on the GPU, 169 s wall, 484 s, 8 tasks. GPU peak 122 MB at most. Encoder 209,760 parameters, 839,040 bytes, comparator a further 8,385 or 6,321 parameters. A comparator read is a pass through it per stored symbol, against one dot product for cosine.
