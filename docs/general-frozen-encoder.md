# A general frozen encoder for images

Status: **direction and step 1 pilot, with DINOv2, 2026-11-03. Not a contract, and not evidence for any contrast.** Two seeds. Centrings were chosen after the risk below was seen. Steps 2 and 3 are scripted and **have not run**.

## The direction

User's proposal, 2026-11-03: a **frozen sensory encoder, broken down by features**. A person looking at an animal or bird they have never met understands something at once, because they see parts such as legs and a tail, and those parts are familiar. The user expects glyphs to be harder, since they are not part of the natural world.

So the question changes. It was: how can an encoder keep learning without breaking memory? It is now:

> **Which kinds of training images build a frozen encoder that is general, so that new kinds of input are already readable?**

If such an encoder exists, memory addresses through it, the core and host keep learning, and the copies of the [natural-image pilot](natural-memory-v1-pilot.md) may not be needed.

**Why this is a change of course.** The [unfreeze goal](experiment-index.md) dates from 2026-11-02, when addresses rested on an encoder trained with symbol labels. The [sense-relational pilot](sense-relational-v1-pilot.md) answered the labels objection the same day. The objection that remains is the one measured in the natural-image pilot: a frozen encoder fails on what it was not trained on (0.05–0.06 of natural images found from a strong view). That may come from what it trained on, not from being frozen.

## First measurement: what transfers

Exploratory, read only, seeds 392 and 393, from a scratch script that `experiments/general_encoder.py pilot transfer` is rewritten from; that stage itself has not run. Finding unseen glyphs among 90 (chance 0.011), and natural images from a strong view among 1,000. Reads centre on the batch's mean.

| Network | Glyphs, one example | Glyphs, 16 examples | Glyphs, other font | Natural, strong |
|---|---|---|---|---|
| Deep, natural images only, key | 0.77–0.82 | 0.94 | 0.71 | 0.92–0.94 |
| Same network, pooled stage 4 | 0.75–0.81 | 0.93–0.95 | 0.71–0.74 | 0.91–0.93 |
| Same network, pooled stage 3 | 0.79–0.80 | 0.94–0.95 | 0.72 | 0.61–0.73 |
| Same network, pooled stage 2 | 0.65–0.69 | 0.86–0.91 | 0.58–0.64 | 0.20–0.25 |
| Same network, stage 2 tied to position | 0.18–0.19 | 0.35–0.37 | 0.31 | 0.36–0.38 |
| Small, glyphs only, key | 0.64–0.67 | 0.85–0.92 | 0.60–0.71 | 0.06 |
| Small, natural images only, key | 0.42–0.46 | 0.68–0.71 | 0.51–0.55 | 0.70 |
| Deep, untrained, key | 0.04 | 0.07–0.09 | 0.05–0.06 | 0.05–0.06 |
| Deep, untrained, pooled stage 1 | 0.62–0.67 | 0.74–0.78 | 0.50–0.57 | 0.14–0.16 |
| Pixels | 0.13–0.15 | 0.19–0.21 | 0.15–0.17 | 0.19 |

- **Transfer runs one way.** Natural images to glyphs works, glyphs to natural images does not.
- **Position-free features beat position-tied ones on glyphs,** which fits "has this feature somewhere" over "looks like this whole image".
- **Middle layers did not beat the final code,** against what was predicted here. Later layers do at least as well as earlier ones.
- **This does not show parts.** Nothing yet checks what the features respond to, and these 28 × 28 grey images barely show legs or tails.
- **Depth and diet are confounded:** the deep network trained on natural images, the small one on glyphs. Step 3 separates them.
- **On glyphs being harder:** the deep network told them apart with no glyph training, so the hard part may be attaching arbitrary meaning to a shape, not seeing the shape. Glyphs are built of strokes and junctions, which natural scenes also contain (Changizi et al. 2006; Lake et al. 2015) **[familiar; read before use]**.

## Step 1: six kinds in one memory, through one frozen encoder

**Question.** Can one frozen encoder replace the copies? Same facts, draws, memory and reads as the [sequence stage](natural-memory-v1-pilot.md#stage-4-six-kinds-in-turn): natural, MNIST, tree, Thai digits, banana, Chinese numerals, 45 facts per kind, in an order and its reverse. No training, no image held. The encoder trained on CIFAR-10 only, 40,000 updates, so natural images are its own kind and the other five are new to it.

**Centring**, since the reads above centre and memory had not:

- **none:** the code as it comes;
- **natural:** subtract the mean over natural images;
- **kind:** subtract the mean over inputs of that kind, fixed from 5,120 stream images, one stored vector per kind; the read is told the kind;
- **recent:** writes as `kind`; a read subtracts the mean of its own batch of 45 queries, so it needs no label, only that recent inputs share a kind.

**Bar, set in advance.** The copies' reads, on mild views, with fewer false answers.

Means over the six kinds, ranges over two seeds and two orders, write threshold 0.85. Matched reads set 0.13 of unseen answered; SE about 0.03 per set, less on these means.

| Encoder, layer, centring | Mild: matched, τ | Strong: matched, τ | Unseen answered at τ: mild, strong | Units per run | Encoders |
|---|---|---|---|---|---|
| Natural deep, key, none | 0.17–0.19, 0.18–0.20 | 0.16–0.18, 0.13–0.15 | 0.81–0.83, 0.72–0.80 | 68–75 | 1 |
| Natural deep, key, natural | 0.17–0.19, 0.18–0.20 | 0.16–0.18, 0.12–0.14 | 0.81–0.83, 0.72–0.81 | 67–75 | 1 |
| Natural deep, key, kind | 0.48–0.52, 0.59–0.63 | 0.31–0.33, 0.28–0.33 | 0.33–0.38, 0.21–0.24 | 384–439 | 1 |
| Natural deep, key, recent | 0.41–0.49, 0.51–0.59 | 0.28–0.31, 0.20–0.23 | 0.27–0.29, 0.12–0.16 | 384–439 | 1 |
| **Natural deep, stage 4, kind** | **0.56–0.61, 0.67–0.69** | **0.33–0.36, 0.30–0.33** | **0.27–0.30, 0.15–0.20** | 471–490 | 1 |
| **Natural deep, stage 4, recent** | **0.52–0.55, 0.57–0.63** | **0.29–0.32, 0.18–0.21** | **0.19–0.23, 0.09–0.11** | 471–490 | 1 |
| Natural deep, stage 3, kind | 0.30–0.44, 0.40–0.53 | 0.18–0.23, 0.19–0.23 | 0.35–0.41, 0.26–0.41 | 523–559 | 1 |
| Natural small, key, kind | 0.41–0.47, 0.51–0.54 | 0.19–0.23, 0.15–0.18 | 0.26–0.30, 0.21–0.33 | 764–814 | 1 |
| Glyph small, key, kind | 0.54–0.60, 0.61–0.67 | 0.08–0.10, 0.08–0.10 | 0.29–0.35, 0.18–0.19 | 584–667 | 1 |
| Glyph small, key, none | 0.46–0.50, 0.55–0.59 | 0.06–0.08, 0.07–0.09 | 0.41–0.43, 0.23–0.26 | 470–531 | 1 |
| PCA, 32 | 0.46–0.54, 0.54–0.56 | 0.12–0.13, 0.10–0.13 | 0.33–0.36, 0.17–0.28 | 703–736 | 0 |
| Pixels | 0.24–0.27, 0.25–0.32 | 0.02–0.04, 0.02–0.06 | 0.68–0.70, 0.55–0.63 | 315–354 | 0 |
| Copies, one per block, no gate ([pilot](natural-memory-v1-pilot.md)) | 0.63–0.76, 0.72–0.81 | 0.35–0.40, 0.43–0.46 | 0.27–0.39, 0.25–0.30 | 324–371 | 6 |
| Copies, one per block, gate 0.25 | 0.64–0.76, 0.72–0.81 | 0.41–0.42, 0.42–0.44 | 0.26–0.30, 0.18–0.24 | | 6–7 |

### What this shows

- **Used as it comes, the natural-trained encoder fails on new kinds.** Natural images read at 1.00, and the other five at 0.00–0.11 with nearly every unseen input answered. The codes of a new kind pile up, as feared.
- **Centring by a mean over natural images does nothing.** The pile-up is specific to each kind.
- **Centring by kind recovers much of it,** from 0.17–0.19 to 0.56–0.61, in one encoder.
- **The bar is not cleared.** On mild views the copies stay ahead on these seeds, 0.63–0.76 against 0.56–0.61. On strong views stage 4 comes close to copies without a gate, 0.33–0.36 against 0.35–0.40, and falls short at the plain threshold, 0.30–0.33 against 0.43–0.46.
- **False answers are the one place it is ahead, at a price.** With `recent`, 0.19–0.23 on mild and 0.09–0.11 on strong, below PCA's 0.33–0.36 and 0.17–0.28 and the copies' 0.26–0.30 and 0.18–0.24. The small network with `recent` is slightly lower on mild, 0.16–0.21, with much weaker reads. The price is reads well below the copies' at the plain threshold: 0.57–0.63 against 0.72–0.81 on mild, and 0.18–0.21 against 0.42–0.44 on strong. So it answers less of everything, not only less of what it should not.
- **It needs one encoder and one forward pass per read,** against 6–23 copies. At 720,528 parameters, 2.9 MB, against 0.84 MB × 6–7 for small copies.
- **Stage 4 beats the key.** The head discards something other kinds need. Stage 3 is worse, so earlier layers again do not help.
- **On strong views, deep natural training gives about three times what a glyph diet or PCA gives** (0.33–0.36 against 0.08–0.13), though depth and diet are confounded. On mild views the glyph encoder is as good, and PCA is near, so mild views do not separate diets here.
- **Kinds differ widely.** With `kind` on the key, MNIST reads 0.67–0.82 and banana 0.11–0.16. These families hold many near-identical renderings, which limits instance reads in any arm.

### So what

A natural diet alone does not give a general encoder for memory, though it comes closer than any other single encoder tried. Reading these results as a first point on the diet question, the next thing to measure is whether a broader diet removes the pile-up without needing a centre per kind. The earlier collapse measure points the same way: the pile-up can be read without labels.

### What this does not establish

- **That features are parts,** or that a new category is understood. Step 2 asks.
- **Which diet is best.** One diet was tried, on one architecture. Step 3 asks.
- **Whether reads hold on fresh seeds.** Deep networks exist only for seeds 392 and 393.
- **Single-query reads.** `recent` assumes a batch of 45 of one kind, and `kind` is told it. A running mean over passing inputs would serve the same purpose when inputs are of one kind for a while, and neither tests a mix.
- **Colour.** All 28 × 28 grey here.

## Step 1 with DINOv2: does a far broader natural diet remove the pile-up?

User's call, 2026-11-03: at 21 million parameters, DINOv2 ViT-S/14 can stand in for a general encoder trained on far more natural images, 142 million, and test the idea much faster than training our own. Weights from the official backbone file, loaded as plain tensors; the network written in `experiments/dinov2_frozen.py`, so no downloaded code runs. Inputs: our 28 × 28 grey images, resized to 224 and copied to three channels.

**Implementation gate.** Nearest-neighbour class reads on CIFAR-10 test images: 0.93 in colour (class token), 0.84 in grey 28 × 28. Our natural-trained network reads 0.58–0.61 by the same measure. So the implementation works, and the grey, small inputs cost it about 0.09.

Same run as step 1, with a **new measure of resemblance**: when an unseen image gets an answer, does the stored fact it reaches share its kind, and its category (a class of natural image, a count for the rendered kinds)? Old arms read exactly as before, 2,160 of 2,160 reads.

| Encoder, layer, centring | Mild: matched, τ | Strong: matched, τ | Unseen answered at τ: mild, strong | Answered unseen: same kind, same category | Answering all unseen: same category |
|---|---|---|---|---|---|
| DINOv2, class token, none | 0.18–0.20, 0.19–0.21 | 0.11–0.15, 0.11–0.12 | 0.81–0.82, 0.69–0.71 | 0.79–0.97, 0.31–0.38 | 0.38–0.46 |
| DINOv2, class token, kind | 0.54–0.60, 0.56–0.66 | 0.32–0.35, 0.21–0.25 | 0.24–0.26, 0.13–0.14 | 0.83, 0.78–0.81 | 0.70–0.74 |
| DINOv2, class token, recent | 0.44–0.53, 0.44–0.48 | 0.31–0.37, 0.07–0.12 | 0.10–0.12, 0.01–0.02 | 0.67–0.83, 0.67–0.83 | 0.69–0.73 |
| **DINOv2, patch mean, kind** | **0.59–0.63, 0.59–0.66** | **0.33–0.36, 0.21–0.24** | 0.21–0.22, 0.09–0.11 | 0.83, 0.79–0.83 | 0.72–0.74 |
| Our natural deep, stage 4, kind | 0.56–0.61, 0.67–0.69 | 0.33–0.36, 0.30–0.33 | 0.27–0.30, 0.15–0.20 | 0.82–0.83, 0.68–0.78 | 0.57–0.59 |
| Our natural deep, key, none | 0.17–0.19, 0.18–0.20 | 0.16–0.18, 0.13–0.15 | 0.81–0.83, 0.72–0.80 | 0.53–0.69, 0.18–0.25 | 0.23–0.29 |
| Glyph small, key, kind | 0.54–0.60, 0.61–0.67 | 0.08–0.10, 0.08–0.10 | 0.29–0.35, 0.18–0.19 | 0.99–1.00, 0.73–0.80 | 0.52–0.56 |
| PCA | 0.46–0.54, 0.54–0.56 | 0.12–0.13, 0.10–0.13 | 0.33–0.36, 0.17–0.28 | 0.64–0.74, 0.42–0.47 | 0.29–0.32 |

Resemblance is on mild views, means over kinds; chance for the same category within a kind is 0.11.

- **DINOv2 piles up too.** As it comes, it reads new kinds at about 0.00–0.09 and answers almost every unseen input of them, like our network. So the pile-up is not from too narrow a diet. It is in reading a frozen code by plain cosine across kinds: images of one kind are all much nearer each other than to anything else, so the general direction of the kind swamps the differences within it.
- **Centred by kind, DINOv2 is no better here than our 0.72 M-parameter network:** 0.59–0.63 against 0.56–0.61 on mild, 0.33–0.36 on strong for both. On CIFAR-10 categories it is far better (0.93 against 0.60). So these instance reads are limited by something other than the encoder: rendered kinds contain many near-identical images, and the strong views are severe for small images.
- **Most answers to unseen images make sense.** Centred, about 0.8 of the unseen inputs answered reach a stored fact of the same kind and category, and 0.70–0.74 when DINOv2 answers all of them, against chance 0.11. So they are mostly resemblance, as the user expected, not arbitrary. Without centring they are mostly arbitrary within a kind, 0.31–0.38. Our network's are less often of the right category, 0.57–0.59 when answering all.
- **Centring on the batch with DINOv2 answers almost no unseen image at τ,** 0.01–0.02 on strong, and reads correspondingly less.

**So.** A broad natural diet makes an encoder whose answers to the unfamiliar mostly make sense, which supports the direction. The test that this step used, finding one stored image among near-copies, cannot show a better encoder. The pile-up needs a fix in how memory reads a frozen code, a centre per kind or some other normalization, whatever the diet. Step 3's bar should therefore be category-level reads and resemblance, not instance reads.

Compute: two workers, 1,575 s wall, 3,086 s aggregate CPU, peak RSS 2.5 GB, GPU peak 420 MB, most of it encoding about 40,000 images per task at 224 × 224. Gate: `NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.dinov2_frozen`, about 3 minutes. Run: `GE_DINO=1 GE_TAG=-dino NFI_DEVICE=cuda SC0_WORKERS=2 ... -m experiments.general_encoder pilot memory`; tables `GE_TAG=-dino ... -m experiments.general_encoder_summary`; report `reports/general-encoder-v1-dino-memory.json.gz`.

## Step 2: does anything carry over to categories never seen? (not run)

Tests the animal example directly. Quarter-width ResNet-18 in colour, 40,000 updates, on CIFAR-100 with categories held out of training, seeds 392 and 393:

| Arm | Trains on | Held out |
|---|---|---|
| `no-animals` | Flowers, food containers, fruit and vegetables, electrical devices, furniture, man-made and natural outdoor things, trees, vehicles | All animals |
| `some-animals` | The above, with aquatic mammals, fish, medium and small mammals, non-insect invertebrates | Insects, large carnivores, large omnivores and herbivores, reptiles |
| `everything` | All of those | None: the ceiling |

People are left out of both sides. Reads on the frozen encoder, on test images: which fine category and which group among the held-out animals, and animal or not, each from 1, 5 and 20 labelled examples.

**Reading:** `some-animals` beating `no-animals` on held-out animals says that animal features carry between animals. `no-animals` near `everything` says that natural images in general are enough. Near PCA says that nothing carried over.

Not yet in the script: a check of what features respond to, such as which image patches drive them. That would be needed before calling anything a part. It needs the saved weights only, so it can follow.

Run:

    mkdir -p /data/external-downloads/cifar100-hf runs/external/cifar100
    # commands in experiments/general_encoder_fetch.py fetch and convert CIFAR-100, about 143 MB
    NFI_DEVICE=cuda SC0_WORKERS=3 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.general_encoder pilot heldout

Estimate: 6 fits, about 2 to 3 hours, from the deep fits in the natural-image pilot. **The script has unit tests on its pure parts, and has not been run end to end.**

## Step 3: which diet? (not run)

The question the user set. One architecture, deep grey, 40,000 updates, on each diet, then the transfer reads on every kind, including those outside its diet:

| Diet | Why |
|---|---|
| Natural | The reference; no glyphs |
| Glyphs | Same architecture as natural, so it separates diet from depth |
| Digits: MNIST, tree, Thai, banana, Chinese | Simple rendered kinds |
| Dead leaves | Procedural; no real images at all |
| Coarse noise | Far from realistic; a floor |
| Natural + glyphs, natural + digits | Does a mix add to natural? |
| All | The broadest here |

**Bar, revised after the DINOv2 run:** on kinds outside the diet, category-level reads and resemblance near those within it. Pile-up without a centre is expected for any diet and is not the bar. Then the best diet goes through step 1.

Run: `NFI_DEVICE=cuda SC0_WORKERS=3 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.general_encoder pilot diet`. Estimate: 16 fits, about 6 to 8 hours. The same limits as step 2 apply: tested on pure parts only.

## Prior work

Search date 2026-11-03: arXiv pages named below.

- **Baradad et al. 2021, Learning to See by Looking at Noise ([arXiv 2106.05963](https://arxiv.org/abs/2106.05963)). [abstract]** Contrastive learners trained only on images from noise processes, including statistical image models. Good results "even with processes that are far from realistic", with structural properties and **diversity** as the key factors. *Differs:* no memory test. *Drawn from it:* the dead-leaves and noise diets, and the expectation that diversity may matter more than realism.
- **Asano et al. 2019, What we can learn from a single image ([arXiv 1904.13132](https://arxiv.org/abs/1904.13132)). [abstract]** With strong augmentation, the first few layers can be learned from one image nearly as well as from millions; for deeper layers the gap to supervision is not closed. *Drawn from it:* early features may come cheaply, and diet should matter more for deep layers. The pooled stage 1 of an untrained network reading glyphs at 0.74–0.78 fits this.
- **Yosinski et al. 2014, How transferable are features? ([arXiv 1411.1792](https://arxiv.org/abs/1411.1792)). [abstract]** Features move from general to specific with depth; transfer falls with task distance, yet distant features beat random. Our result that the last layers transfer as well as earlier ones differs, in a different setting (contrastive with no labels, small images).
- **Oquab et al. 2023, DINOv2 ([arXiv 2304.07193](https://arxiv.org/abs/2304.07193)). [abstract]** All-purpose features across distributions and tasks without finetuning, given "enough curated data from diverse sources". *Differs:* vastly larger network and data. A frozen general encoder is thus an established goal, and a cheaper alternative to this whole line is **using such a pretrained model as the frozen encoder**; it should be tested before training many of our own. Not done here: the design has kept to its own small networks.
- **Biederman 1987, recognition by components; GLOM (Hinton 2021). [familiar; read before use]** Parts as the unit of recognition.

Not read: Changizi et al. 2006, Lake et al. 2015, Biederman 1987.

## Scaling

Writes and reads cost O(h d) per item, for h units and d code width. One forward pass per read, whatever the number of kinds. Storage: the encoder once, plus one centre per kind, d floats, for `kind`; none for `recent`. So against copies, cost no longer grows with time, and grows only slightly with kinds. Units rise with what the memory holds. A fit costs O(p) per update for p parameters, once.

## Computation

Step 1: GTX 1650, two workers, 64 s wall, 73 s aggregate CPU, 4 tasks of 45 memories each; GPU peak 111 MB, peak RSS 2.2 GB. No training: its encoders come from the [natural-image pilot](natural-memory-v1-pilot.md#computation) and earlier.

**Gate.** PCA, pixels and the frozen glyph encoder read exactly as in the sequence stage: all 36 of 36 comparisons per arm, in all four runs.

Runner `experiments/general_encoder.py`, fetch `experiments/general_encoder_fetch.py`, tables `experiments/general_encoder_summary.py`, tests `tests/test_general_encoder.py`, report `reports/general-encoder-v1-memory.json.gz`.
