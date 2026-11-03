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

## Are addresses stable, with no labels?

The point of a frozen encoder is stable addresses. Through frozen DINOv2 they are stable by construction: an image always gets the same code. What could still move them is the centre each code is read against. So this test finds kind centres with no labels and reads every set after every later kind, in both orders, seeds 392 and 393 (`experiments/general_encoder.py pilot stability`, tables `experiments/general_encoder_stability.py`).

**Rule, set before running:** 1,024 passing inputs of the current kind arrive before its facts are taught, then are discarded. An input joins the nearest frozen centre if it reaches that centre's threshold; otherwise it joins the centre being formed, or starts one. A centre freezes after 512 inputs, keeping as its threshold the similarity that 99% of its own members reach. Writes and reads go to the nearest frozen centre. Labels only score. Compared with `told`, step 1's centre per kind, and `none`.

Mean over all kinds and runs: matched reads just after a kind is taught, then after all six; the change; and at the end, unseen inputs answered at τ and the share of answered unseen images reaching the same category.

| DINOv2 layer, centring | Mild: first, last | Change: mean, worst | Strong: first, last | Change: mean, worst | Unseen at τ: mild, strong | Resemblance: mild, strong |
|---|---|---|---|---|---|---|
| Patch mean, told | 0.60, 0.60 | **0.000, 0.00** | 0.35, 0.35 | **0.000, 0.00** | 0.22, 0.10 | 0.80, 0.82 |
| Patch mean, found | 0.43, 0.43 | −0.004, −0.07 | 0.27, 0.26 | −0.004, −0.04 | 0.41, 0.19 | 0.60, 0.60 |
| Patch mean, none | 0.18, 0.18 | −0.004, −0.02 | 0.13, 0.13 | −0.002, −0.04 | 0.82, 0.71 | 0.26, 0.23 |
| Class token, told | 0.56, 0.56 | 0.000, 0.00 | 0.34, 0.34 | 0.000, 0.00 | 0.25, 0.13 | 0.80, 0.79 |
| Class token, found | 0.37, 0.35 | −0.015, −0.18 | 0.24, 0.23 | −0.010, −0.09 | 0.34, 0.19 | 0.48, 0.51 |

Kind centres found, of six kinds: 4, 2, 2, 3 on the patch mean; 4, 2, 2, 4 on the class token.

- **With a fixed centre per kind, addresses do not move at all.** Reads after five more kinds equal reads just after teaching, in every kind, run and view. `told` also reproduces step 1 exactly.
- **Found centres keep addresses nearly as stable,** mean change −0.004, but find too few kinds, so reads are lower: 0.43 against 0.60 on mild.
- **Why too few: a broad kind swallows narrow ones.** Natural images spread widely, so the natural centre's threshold falls to about 0.3 and later rendered kinds pass it. When natural images come first, two centres serve all six kinds. Rendered kinds, white strokes on black, also look alike to DINOv2.
- **The small losses that remain have a known cause:** a centre formed later can be nearer to an older kind's queries than the centre they were written under, so a query and its stored fact are centred differently. Up to −0.07 on mild for the patch mean, −0.18 for the class token.

**So.** Stable addresses: yes, through frozen DINOv2, if each fact is read against the same centre it was written with. Finding those centres without labels is what fails, in a way that is understood.

**Two fixes, both untested:**

1. **Keep each unit's centre with it.** A read centres its query under each centre and compares it only with units written under that centre, taking the best. Then a new centre can never change an old read. It costs one vector per kind and K comparisons per read, not a network per kind.
2. **Split a centre when its inputs change.** Compare a batch of passing inputs with the centre's members, not each input with a fixed threshold. A new kind that falls inside a broad one still shifts the batch's mean.

Compute: two workers, 1,809 s wall, 3,554 s aggregate CPU, peak RSS 2.1 GB, GPU peak 340 MB. Gate: `told` matches step 1 in all 288 comparisons. The quantile and the 512 inputs were set before running and not tuned.

## Do glyphs need their own system? One centre per domain

User's question, 2026-11-03: pile-up appeared only on the rendered kinds (MNIST, tree, Thai digits, banana, Chinese numerals, all white strokes on black), so is glyph reading a separate system, taught apart from objects? Per kind, the glyph-trained small network reads glyphs with no centre (0.46–0.67 mild) and natural images poorly (0.19), the reverse of natural-trained encoders; DINOv2 centred by kind reads glyph categories better than it. This test separates two designs, with no training, on step 1's facts, draws and memory (`GE_DOMAIN=1`, seeds 392 and 393, both orders):

- **domain:** one centre per domain, natural and rendered, the rendered one weighting its five kinds equally; told the domain. What label-free centres would give if they merged exactly the rendered kinds.
- **routed:** two encoders, each domain in its own memory: natural images through our natural-trained network (stage 4), rendered ones through the glyph-trained small network. Told the domain.

Matched reads on mild views, per kind, means over four runs; then the mean over kinds, and over the five rendered kinds, as ranges over runs. Write threshold 0.85.

| Encoder, centring | Natural | MNIST | Tree | Thai | Banana | Chinese | All six | Rendered | Rendered, strong | Rendered unseen at τ, mild | Rendered same category, mild |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DINOv2 patch, none | 1.00 | 0.04 | 0.00 | 0.01 | 0.00 | 0.03 | 0.17–0.19 | 0.01–0.03 | 0.00–0.01 | 0.98–1.00 | 0.27–0.36 |
| DINOv2 patch, kind | 1.00 | 0.92 | 0.32 | 0.65 | 0.25 | 0.47 | **0.59–0.63** | **0.50–0.55** | **0.23–0.28** | 0.25–0.27 | **0.76–0.80** |
| DINOv2 patch, domain | 1.00 | 0.59 | 0.24 | 0.57 | 0.16 | 0.49 | 0.48–0.53 | 0.37–0.44 | 0.15–0.20 | 0.46–0.52 | 0.73–0.76 |
| Natural deep stage 4, kind | 1.00 | 0.80 | 0.57 | 0.61 | 0.27 | 0.25 | 0.56–0.61 | 0.48–0.53 | 0.20–0.23 | 0.32–0.36 | 0.60–0.66 |
| Natural deep stage 4, domain | 1.00 | 0.41 | 0.32 | 0.40 | 0.20 | 0.23 | 0.41–0.46 | 0.29–0.35 | 0.12–0.21 | 0.53–0.56 | 0.59–0.64 |
| Glyph small, none | 0.19 | 0.67 | 0.46 | 0.59 | 0.47 | 0.50 | 0.46–0.50 | 0.52–0.56 | 0.07–0.08 | 0.30–0.36 | 0.59–0.62 |
| Glyph small, kind | 0.39 | 0.79 | 0.53 | 0.59 | 0.52 | 0.59 | 0.54–0.60 | **0.56–0.64** | 0.08–0.11 | **0.23–0.28** | 0.59–0.64 |
| Glyph small, domain | 0.39 | 0.72 | 0.30 | 0.57 | 0.43 | 0.51 | 0.41–0.54 | 0.44–0.55 | 0.05–0.09 | 0.36–0.40 | 0.58–0.61 |
| Routed, none | 0.71 | 0.67 | 0.46 | 0.59 | 0.47 | 0.50 | 0.53–0.59 | 0.52–0.56 | 0.07–0.08 | 0.30–0.36 | 0.59–0.62 |
| Routed, domain | 1.00 | 0.72 | 0.30 | 0.57 | 0.43 | 0.51 | 0.54–0.63 | 0.44–0.55 | 0.05–0.09 | 0.36–0.40 | 0.58–0.61 |

DINOv2's class token reads as its patch mean, slightly lower. At the plain threshold, routed with a domain centre reads 0.66–0.69 on mild, level with stage 4 by kind (0.67–0.69) and above DINOv2 by kind (0.59–0.66).

- **One rendered centre is not enough for instances.** For every encoder it reads rendered kinds about 0.1 lower than a centre per kind, and answers about twice as many unseen inputs. MNIST loses most (0.92 to 0.59 through DINOv2): each rendered kind still has its own offset from the shared one. It recovers most of what no centre loses, though: 0.37–0.44 against 0.01–0.03.
- **For categories one rendered centre nearly suffices.** Unseen rendered images reach a stored fact of their category 0.73–0.76 through DINOv2, against 0.76–0.80 with a centre per kind.
- **A glyph-trained encoder needs no centre on glyphs.** 0.52–0.56 with none, 0.56–0.64 with one per kind; a domain centre slightly lowers it. So a separate glyph system removes the pile-up on its own domain, which supports the user's view.
- **But here it is the weaker reader.** On strong views it reads rendered kinds at 0.05–0.11 against DINOv2's 0.23–0.28, and categories at 0.58–0.64 against 0.76–0.80. Routing the two encoders matches the single best encoders on mild views, and falls well short on strong ones (0.21–0.24 over all six, against 0.33–0.36). The glyph network is small and was trained without labels; this compares these networks, not the two designs at scale.

**So.** Glyphs are a domain of their own to these encoders: a network trained on them reads them with no centre, and a general encoder needs at least a glyph centre. Neither design wins outright. A general encoder with a centre per kind reads best and most robustly; one glyph centre costs about 0.1 on instances but little on categories; a separate small glyph encoder needs no centre but is fragile to strong views. A larger glyph-trained encoder, or a glyph reader taught on top of the general encoder, is what would decide it.

For label-free centres this sets a target: merging the rendered kinds into one centre would cost about 0.1 against per-kind centres, far less than the natural centre swallowing them (0.43 in the stability test, against 0.48–0.53 here).

Compute: GTX 1650, two workers, 1,597 s wall, 3,126 s aggregate CPU, peak RSS 2.5 GB; most of it encoding through DINOv2 at 224 × 224. No training. Gate: the 768 reads shared with the DINOv2 run equal it exactly. Run: `GE_DOMAIN=1 GE_DINO=1 GE_TAG=-domain NFI_DEVICE=cuda SC0_WORKERS=2 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.general_encoder pilot memory`; tables `python3 -m experiments.general_encoder_domain reports/general-encoder-v1-domain-memory.json.gz`; report `reports/general-encoder-v1-domain-memory.json.gz`.

Prior work, familiar, read before use: reading in people uses the visual word form area, which forms in object-recognition cortex only with learning to read (Dehaene and Cohen 2007, neuronal recycling); modern OCR such as TrOCR (Li et al. 2021) starts from an image-pretrained transformer and is then trained on text. Both are shared early vision with a reader taught on top, the design not yet tested here.

## An off-the-shelf glyph expert beside DINOv2

User's direction, 2026-11-03: use something off the shelf for glyphs, small, since glyph recognition itself is not needed; then mix it with DINOv2, MoE-like. The glyph expert is the image encoder of **TrOCR-small, stage 1** (Li et al. 2021), a DeiT-small (21.8 M parameters) pretrained on printed text lines; its decoder is not used. Written in `experiments/trocr_frozen.py` from its shapes, weights loaded as plain tensors, no downloaded code run.

**Gate, seed 392, the transfer reads of the first measurement.** Its class token finds unseen glyphs at 0.88 from one example, 0.98 from 16 and 0.89 across fonts with inputs inverted to dark on light (as is: 0.75, 0.96, 0.82; patch mean 0.48 at best), against DINOv2's 0.84, 0.96, 0.86 (chance 0.011). From strong views it finds 0.11–0.28 of each rendered kind, DINOv2 0.14–0.57. So: class token, every input inverted, chosen on this gate. There is no reference output to match exactly, since the reference code is not run.

**Arms**, on step 1's facts, draws and memory (`pilot mix`), each image encoded once per expert; centres are told the kind or the domain unless said otherwise:

- **single experts:** DINOv2 patch mean by kind and by domain; TrOCR by none, kind and domain;
- **routed:** natural images to DINOv2, rendered ones to TrOCR, a memory each, domain centres; **told** the domain, or by a **label-free gate**: two clusters of DINOv2 codes over passing inputs, each served by the expert that piles up less on it, with centres from its members;
- **joined:** one memory over both codes, domain-centred, keys sqrt(2)[½ z_DINO, ½ z_TrOCR], so a query's weights (w, 1 − w) give the similarity w cos_DINO + (1 − w) cos_TrOCR. Stored keys never change; only the query is weighted. **equal** w = ½; **spread** w from how little each expert piles up in the domain; **learned**: a 770-parameter linear gate on both raw codes, trained with no labels for 300 steps so that a mild view matches the strong view of the same stream image (512 per kind) better than other images. No balancing term, since the domains are uneven;
- **read both by margin:** the two single-expert domain memories; each read trusts the expert whose best unit stands further above its next ten.

Means over the six kinds, and over the five rendered kinds, ranges over two seeds and two orders, write threshold 0.85.

| Arm | Mild: matched, τ | Rendered mild, matched | Strong: matched, τ | Rendered strong, matched | Unseen at τ: mild, strong | Rendered same category, mild | Units |
|---|---|---|---|---|---|---|---|
| DINOv2, kind | 0.59–0.63, 0.59–0.66 | 0.50–0.55 | 0.33–0.36, 0.21–0.24 | 0.23–0.28 | 0.21–0.22, 0.09–0.11 | 0.76–0.80 | 674–718 |
| DINOv2, domain | 0.48–0.53, 0.56–0.60 | 0.37–0.44 | 0.27–0.30, 0.20–0.24 | 0.15–0.20 | 0.38–0.43, 0.15–0.19 | 0.73–0.76 | 448–473 |
| TrOCR, none | 0.26–0.29, 0.34–0.38 | 0.23–0.26 | 0.04–0.10, 0.10–0.15 | 0.03–0.10 | 0.73–0.80, 0.46–0.52 | 0.70–0.79 | 236–250 |
| TrOCR, kind | 0.50–0.57, 0.48–0.54 | 0.45–0.53 | 0.12–0.20, 0.09–0.11 | 0.11–0.18 | 0.17–0.20, 0.04–0.07 | 0.80–0.82 | 928–988 |
| TrOCR, domain | 0.46–0.53, 0.49–0.55 | 0.41–0.49 | 0.11–0.19, 0.09–0.11 | 0.09–0.16 | 0.25–0.28, 0.07–0.10 | 0.80–0.83 | 834–855 |
| Routed, told or gate | 0.51–0.57, 0.57–0.64 | 0.41–0.49 | 0.21–0.26, 0.13–0.15 | 0.09–0.16 | 0.25–0.28, 0.07–0.10 | 0.80–0.83 | 691–718 |
| Joined, equal | 0.66–0.72, 0.67–0.73 | 0.60–0.67 | 0.25–0.31, 0.12–0.16 | 0.16–0.24 | 0.19–0.23, 0.06–0.08 | 0.84–0.86 | 768–782 |
| Joined, spread | 0.65–0.70, 0.67–0.71 | 0.59–0.64 | 0.26–0.31, 0.13–0.16 | 0.15–0.22 | 0.20–0.22, 0.05–0.08 | 0.83–0.86 | 768–782 |
| **Joined, learned gate** | **0.67–0.71, 0.74–0.80** | **0.61–0.65** | **0.32–0.38**, 0.21–0.23 | **0.23–0.31** | 0.24–0.29, 0.07–0.10 | **0.83–0.86** | 768–782 |
| Read both by margin | 0.51–0.54, 0.57–0.65 | 0.41–0.44 | 0.23–0.27, 0.20–0.23 | 0.15–0.20 | 0.33–0.36, 0.14–0.17 | 0.83–0.85 | 1,285–1,326 |
| Copies, one per block, no gate ([pilot](natural-memory-v1-pilot.md)) | 0.63–0.76, 0.72–0.81 | | 0.35–0.40, 0.43–0.46 | | 0.27–0.39, 0.25–0.30 | | 324–371 |

The label-free gate's clusters put every rendered stream image with TrOCR and all but 17–26 of 5,120 natural ones with DINOv2, so it reads exactly as told. Pile-up, the length of the mean unit code: DINOv2 0.56 on natural, 0.88 on rendered; TrOCR 0.84 and 0.74. The learned gate gives TrOCR 0.01 of the weight on natural images and 0.09–0.38 on rendered ones, most on tree and banana, least on Chinese numerals.

- **The off-the-shelf glyph encoder piles up on glyphs too.** Uncentred, TrOCR reads rendered kinds at 0.23–0.26 and answers 0.79–0.82 of unseen ones. Unlike our glyph-trained small network, it still needs a centre. Centred, it reads instances below DINOv2 and is weak on strong views, though its category reads are a little higher (0.80–0.83 against 0.76–0.80).
- **Switching glyphs to it loses.** Routing, told or label-free, reads 0.51–0.57 against DINOv2's 0.59–0.63 with a centre per kind, and far less on strong views. Reading both and trusting the larger margin does no better.
- **Mixing the two codes wins.** Joined with equal weights reads 0.66–0.72 on mild views, above either expert with a centre per kind, using only two domain centres. Rendered kinds gain most, 0.60–0.67 against 0.50–0.55, and categories rise to 0.84–0.86.
- **The learned gate keeps that and restores strong views.** Equal weights cost natural images on strong views (0.69 against DINOv2's 0.80); the gate sends natural images almost wholly to DINOv2 and keeps 0.32–0.38 on strong views, level with DINOv2 by kind.
- **Against step 1's bar.** The learned gate's 0.67–0.71 mild, 0.74–0.80 at τ, sits inside the copies' 0.63–0.76 and 0.72–0.81, with fewer unseen answered on mild (0.24–0.29 against 0.27–0.39); on strong views it stays short (0.32–0.38 against 0.35–0.40; 0.21–0.23 at τ against 0.43–0.46). It is the first arm in this study within the copies' range on mild views, with two frozen encoders and no copy per stretch of time.

**So.** A glyph expert helps as a second opinion mixed into every read, not as a separate system that glyphs are sent to. That fits shared vision with a specialised reader alongside, not instead. The mixture is soft and per input; the stored keys never move, since only the query is weighted.

**Not shown.** The centres and the gate saw stream images of every kind before any fact, as step 1's centres did; a gate trained on kinds that later arrive new is untested. Domain centres were told the domain; the label-free clusters find the same domains, but joined arms were not run on them. Pilot seeds only; the arms and layers were chosen after the TrOCR gate, not set in advance.

**Prior work, familiar, read before use.** Mixtures of experts with a learned gate (Jacobs, Jordan, Nowlan and Hinton 1991); sparse gating and its collapse (Shazeer et al. 2017; Fedus et al. 2021, Switch Transformer). Here the experts are frozen and only the gate learns, which is closer to late fusion of pretrained embeddings; the learned weighting resembles query-dependent fusion in multi-model retrieval. TrOCR, Li et al. 2021 ([arXiv 2109.10282](https://arxiv.org/abs/2109.10282)), not read beyond its model card.

**Scaling.** Each read runs both experts: TrOCR costs about 2.7 times DINOv2 per image on this GPU (384 × 384 against 224 × 224). The joined code doubles a unit's width, 768. With K experts, joined reads cost K passes; a top-k gate would cut that to k, with keys kept per expert so that the gate can change without moving them. The gate is 770 parameters, trained once in seconds.

Compute: GTX 1650, two workers, 6,592 s wall, 13,124 s aggregate CPU, peak RSS 2.6 GB; GPU-bound, about 43,000 images per task through both experts, most of them the 30,720 stream images that set the centres. The TrOCR gate took 150 s per polarity. Gate: the 96 DINOv2 reads by kind equal the earlier run exactly. Weights: `microsoft/trocr-small-stage1`, revision d94c4048, sha256 35ee4bee…49c5 (full value in `experiments/trocr_frozen.py`). Run: `GE_TAG=-mix NFI_DEVICE=cuda SC0_WORKERS=2 OPENBLAS_NUM_THREADS=1 /data/venvs/sri-cuda/bin/python -m experiments.general_encoder pilot mix`; TrOCR gate `NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.trocr_frozen`; tables `python3 -m experiments.general_encoder_domain reports/general-encoder-v1-mix-mix.json.gz`; report `reports/general-encoder-v1-mix-mix.json.gz`.

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
