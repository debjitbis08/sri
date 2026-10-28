# Number systems v1: pilot results

Date: 2026-10-28. Specification: [number systems v1](number-systems-v1-spec.md) (draft;
not frozen). Command: `.venv/bin/python experiments/number_systems.py pilot`.
Artifacts: `reports/number-systems-v1-pilot-results.json`, `-runtime.json`; checkpoints
in gitignored `runs/`. One seed (100), validation pools, 1,000 training and 300
validation images per number per representation. **Pilot: descriptive, not headline
evidence.**

An earlier launch used one satellite per representation by mistake and was stopped
before producing results. Its partial checkpoints were deleted, and nothing from it is
reported.

## Setup

- **Satellites:** one per medium (image, sequence), 64 → 32 → 16 modules, a shared core
  and a label output.
- **Source training:** 14 source representations, each labelled at three different
  numbers (balanced coverage, no repeated sets). Within-representation *same*, *next*
  and *larger* relations. Four representations sampled per step, 3,000 steps,
  checkpoints at 1,000 and 3,000.
- **Source runs:**
  - C2: two compatible sources (Arabic digits, Chinese);
  - C14: all fourteen, compatible;
  - I14: all fourteen, each relation order a private permutation.
- **Adaptation:** held-out tree, german, thai_digits and hand, labelled at 1, 2 and 5,
  300 steps. Modes: zero-shot, interface (medium satellite only), whole system. Plus a
  no-source baseline.

Pipeline: a 20-step smoke run, then the pilot; no numerical failures.

## Source phase

Chance for a ten-way readout on true numbers 1–9 is about 0.11.

| Run, step | Labelled numbers | Unlabelled numbers (complementary transfer) | Input kind from core (chance 0.07; C2 0.50) | Cross-representation number probe, all pairs | … pairs not both mark/collection systems | Morse → MNIST / MNIST → Morse |
| --- | --- | --- | --- | --- | --- | --- |
| C2, 3,000 | 1.00 | 0.08 | 1.00 | 0.24 | — | — |
| C14, 1,000 | 0.58 | 0.12 | 0.65 | 0.20 | 0.18 | 0.22 / 0.24 |
| C14, 3,000 | 0.86 | 0.15 | 0.74 | 0.27 | **0.25** | **0.51 / 0.41** |
| I14, 1,000 | 0.62 | 0.13 | 0.66 | 0.17 | 0.14 | 0.11 / 0.01 |
| I14, 3,000 | 0.88 | 0.13 | 0.71 | 0.18 | 0.16 | 0.09 / 0.00 |

- **Complementary transfer through the label readout is at chance** (0.13–0.15) in
  both C14 and I14. Pooling labels across representations does not happen.
- **The core still encodes the input kind** (0.71–0.74 against 0.07 chance), and
  decodability rose with training in both runs. As the review notes, this alone is not
  evidence against a shared number code: a core can carry both number and
  representation, and the question is whether number-dependent behaviour reuses the
  shared part.
- **The cross-representation number probe shows a structure effect.** For pairs
  outside the mark-counting systems, C14 rises to 0.25 by step 3,000, against 0.16 in
  I14. The clearest case crosses media: a number probe trained on Morse core codes
  reads MNIST at 0.51 in C14 (0.41 the other way), against 0.09 / 0.00 in I14. Morse
  and MNIST share no front end and no surface, so this alignment comes through the core.
  It grows between steps 1,000 and 3,000. *Correction (external review):* an earlier
  version said they share no labels. They do: MNIST is labelled at 2, 8 and 9 and Morse
  at 2, 5 and 9, so two shared anchors could align them through the label pull alone.
  The [post-hoc analysis](#post-hoc-diagnostics) separates shared anchors from numbers
  labelled in neither.
- **The highest cross-probe scores are surface similarity.** Egyptian → dice 0.76,
  dots → dice 0.64, tally → dice 0.61; mark/collection pairs average 0.38 in C14. They
  remain high in I14 (0.27), so they come from the shared image front end seeing
  similar marks, not from number structure.

## Adaptation phase

Unlabelled held-out numbers after adaptation (3, 4 and 6–9): **0.00–0.02 in every
condition**. Labelled numbers (1, 2, 5) reach 0.9–1.0. The label readout again
snaps the rest to the trained labels, as in the [number core
pilots](number-core-network-v1-pilot-results.md).

Further observations:

- **Adaptation degrades the sources' label readout.** Mean source label accuracy (over
  all nine numbers) is 0.38–0.40 before adaptation and 0.09–0.38 after. The image
  satellite is shared, so interface-only adaptation interferes as well. The
  [diagnostics](#2-fixed-versus-refitted-readouts-across-adaptation) show this is mostly
  moved coordinates through the shared interface, not erased number information.
- **Source-probe readout of held-out core codes:** this probe is refitted on
  post-adaptation source codes, so it can compensate for moved coordinates (the
  [diagnostics](#3-where-adapted-held-out-codes-land) use a fixed one instead). It
  reads unlabelled German numbers at 0.25 zero-shot and 0.30 after adaptation from
  C14/3,000, against 0.09–0.18 from I14 and 0.05–0.12 from C2. German shares spellings
  with English and French (*neun*, *acht*, *sechs*), so a surface route is likely. Tree
  and Thai digits reach at most 0.19.
- **Zero-shot labels are near chance**, with isolated exceptions I cannot explain from
  one seed, such as tree 3 and 4 from C2 (0.56–0.63).
- Hand (sequence) adapts its labelled numbers perfectly from every source and transfers
  nothing.

## Reading

- One positive sign: with many representations sharing a compatible order, number
  information becomes partly shared in the core, even across media with no common front
  end (Morse ↔ MNIST 0.41–0.51 against 0.00–0.09 with incompatible order; they share
  anchors at 2 and 9). It
  strengthens with training, which fits the reviewer's point that transfer develops
  later than in-task performance.
- It is weak and does not reach the label readout. Complementary transfer and held-out
  labelling stay at chance, and adapting the shared image interface degrades the
  sources' readouts. The [post-hoc diagnostics](#post-hoc-diagnostics) refine both
  points.
- The source runs were short for 14 representations: each was sampled in about
  860 of 3,000 steps, and C14 labelled accuracy was still rising (0.58 → 0.86). Longer
  training is the cheapest next test of whether the structure effect keeps growing.
- The label readout is a known weak point under sparse labels. The probe measures are
  where the structure effect is visible.

## Post-hoc diagnostics

These three checks come from the external review (recorded in the [abstraction
notes](abstraction-learning-principles.md)) and use the saved checkpoints.

- Command: `.venv/bin/python -m experiments.number_systems_pilot_analysis pilot`.
- Artifacts: `reports/number-systems-v1-pilot-analysis-results.json.gz`, `-runtime.json`.
- The analysis reproduces the pilot's cross-probe means and its adaptation numbers
  (held-out labels, source retention) exactly.

Like the pilot, these are descriptive results from one seed.

### 1. Cross-representation probe by number

Each test number of a probe pair A → B falls into one of four groups:

- labelled in both A and B (a shared anchor);
- labelled in A only;
- labelled in B only;
- labelled in neither.

Numbers labelled in neither can only be aligned through the relations. Chance is about
0.11.

| Pairs | Number labelled in | C14 1,000 | C14 3,000 | I14 1,000 | I14 3,000 |
| --- | --- | --- | --- | --- | --- |
| not both mark systems (152 pairs) | both | 0.21 | 0.36 | 0.16 | 0.28 |
| | neither | 0.19 | **0.26** | 0.16 | 0.15 |
| | A only | 0.13 | 0.15 | 0.10 | 0.11 |
| | B only | 0.20 | 0.32 | 0.15 | 0.20 |
| Morse ↔ images (26 pairs) | both | 0.31 | **0.48** | 0.13 | 0.13 |
| | neither | 0.17 | **0.25** | 0.12 | 0.15 |
| | A only | 0.18 | 0.14 | 0.07 | 0.04 |
| | B only | 0.22 | 0.40 | 0.11 | 0.13 |

- **Transfer extends beyond the shared anchors.** With a compatible order, numbers
  labelled in neither representation reach 0.26 against 0.15 with private orders, and
  the gap grows with training.
- **Shared anchors alone do not align representations across media.** In I14, Morse and
  image representations sharing a labelled number read each other at 0.13 there, about
  chance. The label pull reaches the label output without making the core codes match.
  With a compatible order the same anchors reach 0.48.
- The four groups are not symmetric: "A only" is lowest everywhere. One possible cause:
  the probe learns A's labelled number at A's label-pulled position, and B's unlabelled
  codes for that number are not there. This is not tested.
- **Morse ↔ MNIST** (C14, step 3,000):
  - shared anchors 2 and 9: 0.87 / 0.95 (Morse → MNIST) and 0.79 / 1.00 (MNIST → Morse);
  - numbers labelled in neither (1, 3, 4, 6, 7): 0.97, 0.44, 0.50, 0.04, 0.86 and 0.32,
    0.00, 0.55, 0.00, 0.00;
  - in I14 every number except isolated cells is near zero.

  So the 0.51 is not just the anchors, but it is uneven across numbers and asymmetric.

### 2. Fixed versus refitted readouts across adaptation

Setup:
- A pooled source number probe is fitted on half of the pre-adaptation source core codes
  and **kept fixed**.
- A second probe is **refitted** on post-adaptation codes.
- Both are tested on the other half.
- The fixed label readout (the network's own output) is reported alongside.

Means over the three image held-out representations, source step 3,000:

| Run | Adaptation | Number probe before | Fixed probe after | Refitted probe after | Label readout before → after |
| --- | --- | --- | --- | --- | --- |
| C14 | interface (shared image satellite) | 0.56 | 0.24 | 0.45 | 0.40 → 0.18 |
| C14 | whole | 0.56 | 0.22 | 0.45 | 0.40 → 0.18 |
| C14 | whole, held-out on a copy of the satellite | 0.56 | 0.49 | 0.56 | 0.40 → 0.27 |
| I14 | interface | 0.51 | 0.25 | 0.43 | 0.39 → 0.20 |
| I14 | whole | 0.51 | 0.25 | 0.43 | 0.39 → 0.17 |
| I14 | whole, copy | 0.51 | 0.39 | 0.51 | 0.39 → 0.28 |

- **Most of the source loss is changed coordinates, not lost information.** After
  adapting the shared image satellite, a fixed probe falls from 0.56 to 0.22–0.24, while
  a refitted one still reads 0.45. Some information is lost too (0.56 → 0.45), and with
  the core frozen in interface mode that loss is in the satellite.
- **Most of the interference comes from the shared interface, not the core.** When the
  held-out representation adapts a copy of the satellite and the sources keep the
  original:
  - whole-system adaptation leaves the source number information intact (refitted 0.56);
  - the fixed probe stays near its starting value (0.49);
  - the network's own label readout still falls (0.40 → 0.27), so whole-system
    adaptation damages the path from core to labels more than the core's number
    information.
- Hand, whose satellite is shared only with Morse, interferes little in every mode
  (fixed probe 0.46–0.51 at step 3,000).
- Interface-only adaptation of a copy is not run: it trains the held-out representation
  exactly as interface mode does and leaves the sources untouched by construction.

### 3. Where adapted held-out codes land

The same fixed source probe applied to the held-out representation's **unlabelled**
numbers (3, 4, 6–9) asks whether adaptation places them where the source number
readout expects. The network's own label readout scores 0.00–0.02 here.

| Held out | C14 3,000: zero-shot / interface / whole | I14 3,000: zero-shot / interface / whole |
| --- | --- | --- |
| tree | 0.15 / **0.33 / 0.38** | 0.05 / 0.16 / 0.11 |
| thai_digits | 0.09 / **0.20 / 0.19** | 0.09 / 0.00 / 0.00 |
| german | 0.27 / 0.17 / 0.19 | 0.19 / 0.04 / 0.01 |
| hand | 0.15 / 0.24 / 0.18 | 0.15 / 0.21 / 0.08 |

- After adaptation from C14, tree and Thai unlabelled numbers partly land in the shared
  number geometry: 0.19–0.38 against 0.00–0.16 from I14.
- From C14 at step 1,000, tree reaches only 0.00–0.11, so this grows with source
  training.
- The label output still assigns these numbers to 1, 2 or 5.
- Tree (an unseen group size) is the case where surface overlap should help least.
- From step 1,000, German and Hand are mixed (I14 German 0.30–0.33), so per-cell values
  are noisy at one seed.

### What the diagnostics change

- The pilot's positive sign survives the anchor correction: compatible order aligns
  numbers labelled in neither representation, including across media.
- The failures separate as the review proposed:
  - **Shared information develops** (sections 1 and 3).
  - **The label output does not use it**: unlabelled numbers snap to trained labels
    even when they sit where a fixed number readout reads them correctly.
  - **Adapting a shared interface moves the sources' coordinates.** It mostly does not
    erase them.
- Input-kind decodability is not evidence against this: a core can carry both number and
  representation.
- Remaining to check: more seeds, longer source training, and whether the effects in
  sections 1 and 3 keep growing.

## Stage A: reuse cost

Following the external review, an abstraction's value is measured as the cost of
learning something new:

> reuse benefit = cost to learn without source experience − cost with it, to the same
> accuracy and retention requirements.

Fast learning that forgets the sources earns nothing.

### Design (fixed before running, 2026-10-28)

- **Command:** `.venv/bin/python -m experiments.number_systems_reuse_cost pilot`.
- **New task:** each held-out representation (tree, german, thai_digits, hand) with
  **all nine numbers labelled** but only **k ∈ {2, 8, 32} training examples per number**,
  plus *same*/*next*/*larger* relations among those examples. The pilot's adaptation
  (labels at 1, 2 and 5 only) could never reach any criterion on the other numbers, so it
  cannot measure cost.
- **Starting points:** C14 and I14 at steps 1,000 and 3,000, and no source experience
  (random initialization).
- **Adaptation:**
  - modes: interface (satellite only) and whole system;
  - learning rate 1e-3 as in the pilot, up to 1,000 updates;
  - scored every 25 updates on 100 test images per number.
- **Criterion:** held-out label accuracy ≥ 0.8 over all nine numbers.
- **Retention:** mean source accuracy on each source's own labelled numbers stays
  ≥ 0.9 × its value before adaptation (50 test images per number). It is read two ways:
  - **shared interface:** sources go through the adapted satellite;
  - **separate interface:** sources keep the original satellite, so only change in the
    core and output counts.
  
  The held-out representation trains a replica of its medium satellite. Its training is
  identical to training the shared one, so one run gives both readings.
- **Cost to criterion:** updates (and training encodings), examples (9k), and training
  CPU time excluding scoring. Runs that never reach the criterion are reported as
  censored at 1,000 updates.
- **Accounting:** source training cost (C14: 1,012 s wall for 3,000 steps, including
  evaluation) is reported next to the benefit, together with the number of new tasks
  needed to repay it.
- One seed. This is a measurement check on pilot checkpoints, not headline evidence.

### Results

Artifacts: `reports/number-systems-v1-pilot-reuse-cost-results.json.gz`, `-runtime.json`.

**The criterion was mostly out of reach, so cost to criterion is censored for the image
representations.** No image held-out representation reached 0.8 within 1,000 updates in
any condition, even at 32 examples per number. Accuracy levels off below the criterion.
The plateau could come from the examples, the representation, regularization or
optimization, and this run does not separate them. Only Hand reached the criterion, at
k = 32: 150–975 updates, against 250–275 from scratch, with no consistent ordering. As
specified, the measurement fails for the image representations. **Lower cost to the
criterion therefore remains unmeasured.**

What can still be read is a different result: the best accuracy reached within the
budget at a fixed number of examples. Means over tree, german and thai_digits:

| Mode | Examples per number | No source | C14 1,000 | C14 3,000 | I14 1,000 | I14 3,000 |
| --- | --- | --- | --- | --- | --- | --- |
| interface | 2 | 0.19 | 0.22 | 0.27 | 0.23 | 0.26 |
| | 8 | 0.29 | 0.36 | 0.44 | 0.36 | 0.43 |
| | 32 | 0.56 | 0.59 | 0.66 | 0.60 | 0.68 |
| whole | 2 | 0.18 | 0.22 | 0.28 | 0.24 | 0.27 |
| | 8 | 0.28 | 0.37 | 0.43 | 0.38 | 0.47 |
| | 32 | 0.55 | 0.62 | 0.66 | 0.64 | 0.70 |

- **Source experience raises the accuracy reached with limited examples**, more so after
  longer source training. At 8 examples per number, accuracy rises from 0.28–0.29 with no
  source to 0.43–0.47 from step 3,000.
- **Compatible order adds no detectable benefit in this protocol.** I14 matches or
  exceeds C14 in every row, whereas in the probes C14 led clearly. Both conditions train
  the front end, core and output, so this does not locate the benefit in any one part. It
  shows only that the shared order added nothing measurable here. The shared number
  information found by the probes has not been shown to make learning a new
  representation easier.
- **German** is the one held-out where C14 and I14 both clearly exceed no source at
  k = 32 (0.57–0.60 against 0.36–0.40 at step 3,000), but they do so equally.

Retention after 1,000 updates (final / before, mean over the image held-outs and k):

| Mode | Shared interface | Separate interface |
| --- | --- | --- |
| interface | 0.44–0.51 | 1.00 (by construction) |
| whole | 0.46–0.52 | 0.83–0.91 |

- **With a shared interface, learning a new image representation halves the sources'
  labelled accuracy in every condition**, so no run passes the retention requirement.
  Under the review's definition, a shared satellite earns no reuse benefit here.
- With separate interfaces, whole-system learning keeps 0.83–0.91. That is close to, but
  mostly below, the 0.9 requirement. A separate interface is also persistent added
  structure, and it would have to count toward cost.

**Reading.**
- Source experience raises the accuracy reached with limited examples. Compatible order
  adds no detectable benefit over private orders in this protocol.
- Learning through a shared interface costs the sources half their labelled accuracy.
- This measure does not yet show the shared number structure seen in the probes doing
  any work.
- Ordinary training rewards only the current label and relation tasks. Nothing in it
  favours parameter changes that make later learning easier, so measuring reuse after
  ordinary training cannot test whether such a signal would help (external review).
- For the design: 0.8 was too high for ≤ 32 examples per number. A later version should
  set criteria from the no-source learning curve (for example its accuracy at k = 32), or
  measure examples needed at a reachable level.
- Source training cost, about 1,000 s wall per 3,000-step run, cannot be credited
  against benefits that were not reached.

## Cost

Wall 1,481 s on 11 workers:
- data rendering 24 s (586.8 MB of pools);
- source runs 1,014 s (C2 479 s, C14 1,012 s, I14 1,014 s, run in parallel with the
  no-source adaptations);
- adaptation 443 s.

Worker CPU 7,372 s. Largest worker peak RSS 2.3 GB. A 60-second smoke run of the full
pipeline preceded it. The stopped per-representation launch ran for several minutes; its
time was not recorded. All of this counts as selection cost.

Post-hoc diagnostics (recorded run): wall 689 s on 11 workers, including data rendering. Work:
- 4 by-number probe jobs (182 probes each, about 120 s each);
- 64 adaptation jobs (300 steps plus three probe fits, about 105 s each).

Worker CPU 7,125 s; largest worker peak RSS 2.4 GB. It was run twice:
- an earlier run (695 s wall, 7,144 s CPU) before results were written gzipped, giving identical results;
- plus two launches that failed at import and at satellite naming before doing any work.

Stage A reuse cost: wall 4,004 s on 11 workers, 120 jobs of 1,000 updates with 41 scoring
points each. Worker CPU 43,107 s, of which scoring took 12,648 s; largest worker peak RSS
1.1 GB. My estimate beforehand was about 15 minutes. It missed by about 4× because jobs
slowed about 3× when 11 ran in parallel, and because I extrapolated from a 50-update
smoke run. The in-process smoke run (two 50-update jobs) counts as selection cost.
