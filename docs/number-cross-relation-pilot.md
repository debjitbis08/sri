# Number cross-relation pilot: does succession extend sparse cross-modal equivalents?

Date: 2026-10-28. **Exploratory pilot; no frozen contract.** It follows the
[v1 confirmation results](number-concept-network-v1-results.md) and
[alignment analysis](number-concept-network-v1-alignment-analysis.md), and changes the
question. v1 asked whether succession plus one anchor image transfers number to dice.
This pilot asks whether a shared number representation forms from **sparse
cross-modal equivalents**, and whether learning succession **extends** that alignment
to counts that were never paired. The design came out of discussion with the user and
an external reviewer. A positive result would justify a specification, a prior-work
review and a frozen contract; it is not itself confirmatory evidence.

Reproduce: `.venv/bin/python experiments/number_cross_relation_pilot.py` (11 workers;
`NCR_SMOKE=1` for a 20-step pipeline check). Output:
`reports/number-cross-relation-pilot-results.json.gz` (deterministic, gzip-compressed) and `-runtime.json`.

## Design

- **Data:** MNIST training partition and test set (v1 split), and standard dice faces
  1–6 from the v1 generator, train and test pools. The v1 per-pip jitter caveat
  applies.
- **One shared encoder** for both modalities: a small CNN, 16-dimensional code,
  207,696 parameters. Both modalities are trained jointly; nothing is frozen.
- **Judge:** answers two true/false questions about an ordered image pair (A, B).
  - *same:* do A and B show the same number?
  - *next:* is B's number A's number + 1?
- **Training pairs, per step** (64 per task, half positive, negatives drawn uniformly
  from all non-matching class pairs):
  - within digits and within dice: *same*; plus *next* in arms with succession on;
  - cross-modal *same*, in both orders, only among fixed **equivalents**: 10 images per
    modality for each paired count. The equivalents are keyed by seed, modality and
    count, so every arm of a seed uses identical images. Negatives are equivalents of
    different paired counts.
  - *next* is never trained across modalities.
  - The loss is the sum of per-task mean BCE, so each task has the same weight whether
    or not the succession tasks are present.
- **Judges:**
  - *free:* an MLP on the concatenated codes (20,994 parameters). The two answers
    share nothing beyond the encoder.
  - *structured:* `same` is a decreasing function of `||z_B − z_A||²`, and `next` of
    `||z_B − S(z_A)||²`, with one operator `S(z) = z + g(z)` (small MLP) shared by both
    modalities (2,132 parameters). The two answers are read off one representation, so
    "dice 3 = digit 3" and "digit 4 follows digit 3" geometrically imply "digit 4
    follows dice 3", provided S is accurate.
- **Arms** (3 seeds each; 3,000 Adam steps, learning rate 1e-3):

  | Arm | Judge | Succession | Paired counts | Unpaired dice counts |
  | --- | --- | --- | --- | --- |
  | free-succ-135 | free | on | 1, 3, 5 | 2, 4 (interpolation), 6 (extrapolation) |
  | free-nosucc-135 | free | off | 1, 3, 5 | same |
  | struct-succ-135 | structured | on | 1, 3, 5 | same |
  | struct-nosucc-135 | structured | off | 1, 3, 5 | same |
  | struct-succ-123 | structured | on | 1, 2, 3 | 4, 5, 6 (extrapolation) |
  | struct-nosucc-123 | structured | off | 1, 2, 3 | same |

## Measures

All on held-out test images. Accuracy on unpaired dice counts is reported by group:
interpolation and extrapolation.

- **Primary: cross-modal *same*, image level, both directions.** Each dice test image
  is compared with 20 fixed digit test references per class. The predicted digit is
  the class with the highest mean P(same), scored per dice count. In the other
  direction, digit test images (300 per class) are compared with dice references.
- **Cross-modal *next*** (succession arms only), scored the same way: dice n → digit
  n+1 and digit n → dice n+1. Never trained across modalities.
- **Representation readout:** each dice test code is read out by the nearest
  digit-centroid (cosine; centroids from 1,000 training digits per class). For the
  structured judge with succession, also `S(dice n)` against digit n+1.
- **Digit health:** digit nearest-centroid accuracy, within-modality judge accuracy,
  and digit composition `S^k` for k = 1–3 (structured with succession).
- **Class matrices and offsets:** mean P per (class a, class b) cell; the offset of
  the argmax from the correct class per dice count.

## Predictions (written while the full run was in progress, before any results)

1. The structured judge with succession beats the structured judge without
   succession on unpaired counts 2 and 4, consistently across seeds.
2. Count 6 (extrapolation) is weaker than 2 and 4. Pairs at 1–3 tested on 4–6 are
   weaker still.
3. The free judge shows little benefit from succession, because nothing ties *next*
   to *same*.
4. Without succession, unpaired counts are at or near chance for the cross-modal
   readouts: no ordering ties the unpaired dice classes to the digits.
5. Nothing near-perfect; digit accuracy remains reasonable under joint training.

## Results

Run completed 2026-10-28 (1,959 s wall). Runner SHA-256 `561998cd…` matches the
committed file. Mean over seeds 0–2, with per-seed values in parentheses. All values
are image-level accuracy on test images, averaged over the dice counts in each group.

### Primary: cross-modal *same*, dice → digit (ten-way)

| Arm | Paired | Unpaired, interpolation | Unpaired, extrapolation |
| --- | --- | --- | --- |
| free-nosucc-135 | 1.00 | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00, 0.01) |
| free-succ-135 | 1.00 | 0.27 (0.00, 0.30, 0.50) | 0.33 (0.00, 0.00, 1.00) |
| struct-nosucc-135 | 1.00 | 0.00 (0.00, 0.00, 0.00) | 0.00 (0.00, 0.00, 0.00) |
| **struct-succ-135** | 1.00 | **1.00 (1.00, 1.00, 1.00)** | **1.00 (1.00, 1.00, 1.00)** |
| struct-nosucc-123 | 1.00 | — | 0.00 (0.00, 0.00, 0.00) |
| **struct-succ-123** | 1.00 | — | **0.78 (0.67, 1.00, 0.69)** |

### Other cross-modal readouts on unpaired counts

| Arm | *same* digit → dice | *next* dice → digit | *next* digit → dice | Dice code → nearest digit centroid | `S(dice n)` → digit n+1 |
| --- | --- | --- | --- | --- | --- |
| free-succ-135 (2, 4 / 6) | 0.26 / 0.50 | 0.65 / 0.32 | 0.93 / 0.97 | 0.28 / 0.32 | — |
| struct-succ-135 (2, 4 / 6) | 0.99 / 0.99 | 1.00 / 1.00 | 0.98 / 0.99 | 1.00 / 1.00 | 1.00 / 1.00 |
| struct-succ-123 (4–6) | 0.77 | 0.56 | 0.82 | 0.78 | 0.55 |

Without succession, the nearest-centroid readout of unpaired dice is 0.00 in all
runs. The digit → dice *same* direction is not a reliable control there. It is scored
over only six dice classes, and the paired dice classes are already claimed by their
digits. In the no-succession arms, P(same) for every unpaired cross-modal cell is
small, typically below 0.3, so the argmax picks among near-zero values. For example,
digit 6 → dice 6 is 0.96 in struct-nosucc-135, where P(same) is only 0.11–0.28. Those
arms reject all unpaired cross-modal equivalences rather than aligning them.

### Where the 1–3 arm fails

Struct-succ-123 is exact in seed 1. In seeds 0 and 2, dice 4 and 5 are correct but
dice 6 is read as digit 4 (class offset −2), and *next* from dice 5 drifts by −2.
Errors appear at the far end of the extrapolated chain.

### Digit health under joint training

Digit nearest-centroid accuracy is 0.96–0.98 in every arm (v1 digit gates:
0.97). Within-modality judge accuracy is 0.94–0.98 for digits and 1.00 for dice.
Digit composition `S^k` for k = 1–3 is 0.98–0.99 in both structured succession
arms. Joint training did not degrade the digits.

### Against the predictions

1. **Confirmed, more strongly than predicted.** With succession, the structured judge
   aligns unpaired counts 2, 4 and 6 exactly in all three seeds. Without succession
   it gets 0.00.
2. **Partly.** Count 6 was *not* weaker than 2 and 4 with pairs {1, 3, 5}. Pairs
   {1, 2, 3} tested on 4–6 were weaker (0.78), with errors at count 6.
3. **Partly.** The free judge benefited from succession somewhat (0.00 → about 0.3),
   but unreliably across seeds. The structured judge's benefit was complete.
4. **Confirmed** for dice → digit and the representation readout. See above for the
   digit → dice caveat.
5. **Wrong in one place.** Digit health held, but struct-succ-135 was near-perfect.

## Interpretation

- **Learned succession extends sparse equivalents, if the relations share one
  representation.** The same paired images give 0.00 on unpaired counts without
  succession and 1.00 with it. The structured judge makes *same* and *next*
  properties of one code space with a shared operator S. With S accurate (digit
  composition ≥ 0.98), pairs at 1, 3 and 5 fix the dice chain's offset and frame, and
  every other count follows. This matches the earlier structural argument: a chain
  matched at a few points, under one shared successor, has a unique alignment.
- **Free relations are not enough.** With a free judge, *next* and *same* are
  separate skills over the same encoder. Succession helps a little, through the shared
  encoder, but nothing makes the answers consistent.
- **Extrapolation degrades with distance.** Pairs {1, 3, 5} bracket or sit beside
  every unpaired count. With pairs {1, 2, 3}, count 6 is three applications of S past
  the last pair, and two of three seeds slip by two counts there. That is the same
  kind of error accumulation that broke v1's one-anchor propagation, now much smaller.

## Limits

- Exploratory: three seeds, one budget, no tuning, no contract.
- **Not a comparison with v1.** Many factors changed together: a shared encoder in
  place of separate modules, joint rather than frozen training, binary relation
  judgments rather than contrastive ranking, and class-level equivalents (10 images
  per count) rather than one anchor image. The 2×2 here isolates succession and judge
  structure, not those factors.
- The dice are the v1 jittered standard faces; random layouts were not tested. The
  encoder may still read layout rather than count, so this says nothing about
  counting.
- "Image level" uses 20 fixed references per class, so a class-level judgment underlies
  each image score.

## Possible next steps

- A specification and frozen contract for the main contrast: structured judge,
  succession on vs off, sparse pairs. Seeds to 5 or more; add a pairs-at-{1} arm, the
  class-level version of v1's anchor; prior-work review (sparse-supervision alignment,
  relation networks, children's number-word learning).
- Factor analysis for the v1 gap: separate vs shared encoder, frozen vs joint.
- Random dice layouts and a blank (zero) face, to test counting rather than face
  recognition.

## Cost

Theoretical, per fit: training is `STEPS × tasks × 2 × 64` encodings. That is
768 encodings per step with succession (6 tasks) and 512 without (4 tasks). The
encoder forward pass is about 0.45 M multiply-adds per image, so training is
≈ 3 × 10¹² multiply-adds per succession fit counting forward and backward.
Evaluation is 13,000 test encodings plus about 1.8 M judge evaluations. Memory is
dominated by the in-memory datasets (≈ 0.9 GB per worker in the smoke run); no
checkpoints are saved.

Measured (11 workers, one thread each, CPU only): 1,959 s wall in total, 4.6 s data
preparation, 19,289 worker CPU s. Training wall per fit: 1,440–1,550 s for
1–3-5 succession arms, 850–1,020 s without succession. The 1–2–3 arms ran in the
second wave with less contention (400–1,010 s). Evaluation took 5–25 s per fit. Peak
RSS was 0.85–0.90 GB per worker. Selection cost: one 20-step smoke run of all
18 fits (40 s wall) for pipeline checks; no tuning.
