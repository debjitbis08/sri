# Number systems v1: does experience with many number representations produce a shared number core?

Status: **draft, 2026-10-28. Not frozen.** Generators are implemented
(`src/number_systems.py`, tests in `tests/test_number_systems.py`). Still to do before
freezing: decisions marked **Open** below, a prior-work review, a pipeline pilot, and a
contract with implementation hashes.

## Motivation

The [number core pilots](number-core-network-v1-pilot-results.md) found that image and
Morse inputs of the same number get the correct label but occupy different places in
the core. The input kind is linearly decodable from the core at 0.999, and an MNIST-trained
number readout transfers to no unlabelled count. A tally-mark test with labels at 1, 2
and 5 put every other count into those three bins. With three input kinds, dense
within-kind relations and a roomy core, separate solutions cost no more than a shared
one, so there is no reason to expect sharing.

This study tests one way to make reuse pay: **many representations of number, each
grounded at only a few, different numbers.** With complementary grounding, labelling
every number in any representation needs knowledge pooled across representations. A
shared core is one solution to that, and possibly the cheapest. Nothing in the
objective declares any two images equal.

It also adopts the external reviewer's separation of **learning a relation system**
from **learning a new realization of it**
([abstraction notes](abstraction-learning-principles.md)): source experience first, then
bounded adaptation to held-out representations, measured at several source checkpoints.

## Inspiration: Lockhart, *Arithmetic* (2017)

Read: chapters "Language" through the opening of "India". Ideas used here:

- **A number is distinct from its representations.** Threeness is a property of
  collections; *3*, *III*, 三 and *three* are codes for it. The study asks whether the
  network separates the number from its codes.
- **Number permanence.** The size of a collection does not depend on its arrangement.
  Collection systems (dots, dice) vary their arrangements.
- **Representation families**, as the book develops them:
  - repetition (tally strokes, claps);
  - grouping with a group mark (five-barred gate);
  - stacking (Egyptian, never more than four strokes in a row, nine as three rows of
    three);
  - subgroup symbols (Roman, written additively, IIII and VIIII; Lockhart notes the
    Romans did not use subtractive IV for arithmetic);
  - a distinct symbol per number (Chinese, Hindu digits);
  - number names in languages.
- **The three tribes.** The Hand People (gestures, group of five), the Banana People
  (group of four) and the Tree People (writing, group of seven) count the same numbers
  with different group sizes and media. Seven is three marks for the Hand People, four
  for the Banana People and one for the Tree People. The number of marks therefore stops
  tracking the number, which removes the "count the marks" shortcut across grouping
  systems.
- **"The artful rearrangement of numerical information—in particular, the translation
  among different grouping sizes—is the soul and essence of arithmetic"**, and
  **"counting is really all about comparison."** Comparison ("which is larger?") is
  added to the within-representation relations.

The Banana and Tree written forms and the Hand gestures are this study's renderings of
Lockhart's fictional tribes; his own Tree signs for one to six are not reproduced.

## Representations (numbers 1–9)

| System | Lockhart family | Medium | Notes |
| --- | --- | --- | --- |
| dots | collection | image | random non-overlapping discs; half fixed radius, half fixed total area |
| dice | collection | image | v1 generator; 1–6 only |
| tally | grouping, size 5 (gate) | image | existing generator |
| banana | grouping, size 4 | image | square per four, curved stroke per leftover |
| tree | grouping, size 7 | image | tree sign for seven, rings (rows of three) for leftovers |
| hand | grouping, size 5, gestural | sequence | thump (wide, softer pulse) per five, clap (narrow) per leftover |
| egyptian | stacking | image | Lockhart's stacking patterns; four sometimes two over two |
| roman | subgroup symbols | image | additive I…VIIII, eight Latin fonts |
| mnist | distinct symbols | image | handwritten Arabic digits |
| chinese | distinct symbols | image | 一…九, four CJK fonts |
| devanagari_digits | distinct symbols | image | १…९ |
| arabic_digits | distinct symbols | image | ١…٩ |
| thai_digits | distinct symbols | image | ๑…๙ |
| english, french, german | names | image | words, Latin fonts, lower or capitalized |
| hindi | names | image | Devanagari words, complex-script shaping |
| morse | code | sequence | existing generator |

Shared rendering: 28 × 28, random font (text systems), rotation ±8°, size, position,
brightness 0.7–1 and Gaussian noise SD 0.02; sequences are 128 samples. Every pool is
built from named streams (`system / split / number / index`). Previews were checked by
eye. Known surface overlaps: one to three are stroke counts in tally, Egyptian, Roman
and Chinese, and small dot counts can be subitized. The grouping systems with different
group sizes break the shortcut above three.

**Held-out representations (never seen in source training):** tree (an unseen group
size), german (a held-out language in a known script), thai_digits (an unseen script of
distinct symbols) and hand (the gestural sequence; decided 2026-10-28). Fourteen source
representations remain.

## Architecture

The number core network from the pilots, at the wide size (64 → 32 → 16), extended to
many input kinds:

- **One satellite per medium** (decided 2026-10-28): an image satellite and a sequence
  satellite, each a front end (the pilots' image CNN or Morse 1-D CNN) plus one SRI input
  module, coupled to the shared core at corresponding layers through learned
  translations. Every image representation goes through the image satellite; Morse and
  the held-out Hand go through the sequence satellite. Sharing between representations of
  the same medium can therefore also happen in the satellite, as in the pilots; between
  media it can happen only in the core.
- Core, output module with fixed label codes, and core successor as in the pilots
  (uniform per-unit coupling, exact equilibria, weak prior).

## Training

**Source phase.**

- **Complementary sparse labels.** Each source representation is labelled at three
  numbers. Across representations every number 1–9 is labelled equally often, and no two
  representations share the same labelled set. Assignments are drawn once per seed from
  a named stream and recorded. MNIST is treated like every other system (no dense
  labels), so no representation is privileged.
- **Within-representation relations at all numbers:** *same*, *next* and *larger*, read
  off core codes by the structured judge (*larger* as a learned direction score in the
  core inner layer). Pairs never cross representations.
- **Label loss:** positive-only cosine pull to the label code (pilot 2 amendment) for
  every representation.
- Each step samples a fixed number of representations, so the cost per step does not
  grow with the number of sources.

**Adaptation phase (held-out representations, one at a time).** The held-out
representation goes through its medium's existing satellite, labelled at three numbers
(1, 2, 5, as in the tally test), with within-representation relations. Fixed budget
(300 steps in the pilot). Three modes:

- **zero-shot:** no adaptation; the held-out representation is evaluated as is;
- **interface:** the medium's satellite (front end, input module, translations) learns;
  core, output, successor and judge are frozen;
- **whole system:** everything learns.

Because the satellite is shared within a medium, retention of the source representations
is measured after both adaptive modes.

## Design

- **Main variable: number of source representations** N ∈ {1, 2, 4, 8, 14}, subsets
  drawn per seed, at an equal total source budget. The pilot runs N = 2 and 14.
- **Source checkpoints** for adaptation: 1,000, 3,000 and 10,000 steps.
- **Controls:**
  - *no source experience:* adaptation from random initialization;
  - *incompatible structure, equal budget:* each source representation's *next* and
    *larger* follow its own fixed random permutation of 1–9. The relations stay
    internally consistent and equally hard, but there is no single order shared across
    representations. Renaming one chain would keep it shared, so a per-representation
    permutation is used instead;
  - A compression arm (weight decay or description cost) is deferred (decided
    2026-10-28).

## Measures

Primary, all on held-out test images:

1. **Input-kind decodability from the core** (linear probe; chance 1/N). Descriptive
   only: a core can carry representation identity alongside a shared number code, so
   decodability above chance does not count against sharing (external review after the
   pilot).
2. **Cross-representation number readout:** a linear number probe trained on one
   representation's core codes, tested on every other representation.
3. **Complementary transfer:** label accuracy on numbers *never labelled* in that
   representation (six of nine per source representation).
4. **Held-out adaptation:** accuracy on the held-out representation's unlabelled
   numbers after the fixed budget, as a function of N, source checkpoint and adaptation
   mode, against the controls.

Secondary: cross-representation *same*/*next*/*larger* from core codes, core CKA,
composition of the successor, per-family breakdowns (to expose surface shortcuts),
retention of source representations after whole-system adaptation.

## Predictions (to be fixed in the contract)

1. With compatible structure, measures 1–3 improve with N; with incompatible structure
   they do not.
2. Held-out adaptation benefits from source experience only above some N, and more in
   interface-only mode if the core holds reusable structure.
3. Tree (unseen group size) gains least from surface overlap; if it still transfers, the
   transfer is unlikely to be mark counting.
4. If all measures stay flat up to N = 14, the remaining suspects are the slack in the
   shared operator and the lack of a cost on duplication (the compression arm).

## Prior work to review before freezing

Named from memory, to be verified: zero-shot translation and an emergent interlingua in
multilingual NMT (Johnson et al. 2017); emergence of shared concept spaces during
multilingual training ([arXiv 2601.22851](https://arxiv.org/html/2601.22851v1), title
only so far); multi-task and multi-domain learning; An et al. 2023 (read by the reviewer;
see the abstraction notes); grokking (Power et al. 2022; Nanda et al. 2023); Anthropic's
2026 workspace paper (cross-language routing through a shared representation);
Dehaene's triple-code model of number (verbal, visual-symbolic and magnitude codes);
Lockhart (inspiration, not evidence).

## Computation (to be completed in the contract)

Generators: rendering cost per image is dominated by 4× supersampled coverage
(112 × 112 per segment or glyph); pools of 2,000 / 300 / 500 per number per system
(about 25,000 images per system, 18 systems). Measured generation time and the training
cost model (encodings per step, factorizations per step = representations sampled,
adaptation runs = held-out × checkpoints × modes × conditions) go into the contract
after the pipeline pilot, following [computation reporting](computation-reporting.md).
