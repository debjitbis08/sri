# Number construction v1, Part D: senses, varied contexts, rehearsal and imagery

Protocol: [number construction v1 spec, "Part D
protocol"](number-construction-v1-spec.md#part-d-protocol-senses-varied-contexts-and-rehearsal-pilot-fixed-before-running-2026-10-29)
and its imagery probe section. Run autonomously on 2026-10-29 while the researchers were
away. Three seeds; **a pilot, not headline evidence.** Motivation:
[conceptual instinct notes](conceptual-instinct-notes.md); prior work:
[prior-work.md, "Senses, varied contexts, rehearsal and imagery"](prior-work.md).

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_part_d pilot` (25 minutes);
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_imagery pilot` (13 minutes).
- **Artifacts:** `reports/number-construction-v1-part-d-pilot-results.json.gz`,
  `reports/number-construction-v1-imagery-pilot-results.json.gz`, their `-runtime.json`
  files, and image grids `reports/number-construction-v1-imagery-pilot-<arm>-seed<seed>-{image,taps}.png`
  for D-base and D-varied+rehearsal, every seed. Checkpoints in `runs/number-construction-v1-part-d-pilot/`
  (not tracked).
- **Withheld counts** (as in Part A): seed 100 chinese 4, 8 and devanagari 2, 5; seed 101
  chinese 2, 5 and devanagari 3, 7; seed 102 chinese 4, 7 and devanagari 3, 5.
- **Replication:** D-base reproduces Part A pilot 3 A-order exactly on every isolated
  measure, for all three seeds.

All values below are means over seeds 100–102 (per-seed values are in the results file).
Chance is 0.11 for matching among 9 and 0.5 for comparison.

## 1. Rehearsal removes most of the forgetting

| End of training | D-base | D-rehearsal | D-varied | D-varied+rehearsal |
| --- | --- | --- | --- | --- |
| English | 0.09 | 0.28 | 0.08 | 0.35 |
| Chinese, paired forms | 0.20 | 0.53 | 0.22 | 0.60 |
| Chinese, withheld forms | 0.12 | 0.37 | 0.17 | 0.53 |

- Chinese at the end of its own block was 0.51 (base) and 0.48 (rehearsal), so with
  rehearsal it is essentially retained. English is retained at about its own-block level
  (0.21).
- As prior work predicts, and as expected of a standard component. Rehearsal takes 4 of
  16 language episodes per batch, and it did not reduce acquisition of the current
  language (Devanagari at its own block: 0.51 against 0.40 for base).

## 2. Varied contexts beat repeated ones at matched exposure

The two context arms see the same number of items per update; they differ only in
whether the second item is taps (with a comparison loss) or another collection.

| Measure | D-rehearsal | D-repeated+rehearsal | D-varied+rehearsal |
| --- | --- | --- | --- |
| English, own block | 0.21 | 0.23 | 0.31 |
| Devanagari paired, own block | 0.51 | 0.41 | 0.53 |
| Devanagari withheld, own block | 0.43 | 0.29 | 0.46 |
| Chinese paired, end | 0.53 | 0.42 | 0.60 |
| Chinese withheld, end | 0.37 | 0.37 | 0.53 |
| English, end | 0.28 | 0.22 | 0.35 |
| Two-sided revision, withheld, end | 0.52 | 0.56 | 0.58 |
| Collection succession, end | 0.92 | 0.91 | 0.93 |

- Varied contexts are better than repeated ones on every acquisition and retention
  measure, by 0.08–0.17.
- **Repeated contexts were worse than no extra contexts** on several measures (for example
  withheld Devanagari 0.29 against 0.43). Extra collection–collection matching competes
  with the language episodes instead of helping them.
- Without rehearsal, varied contexts also speed acquisition in each block (D-varied
  against D-base: English 0.31 against 0.21; withheld Chinese 0.47 against 0.41;
  withheld Devanagari 0.52 against 0.40). Forgetting remains without rehearsal.
- The varied-practice prediction (slower acquisition, better transfer) did not hold in
  its first half: varied contexts were not slower.
- **Not separated:** whether taps, the comparison loss or both drive the gain. The varied
  arms differ from the repeated arm in both.

## 3. Untaught compositions work

Taps never met forms, and comparisons never involved forms. D-varied+rehearsal at the end
of training (D-varied in brackets):

| Language | taps → form, paired | taps → form, withheld | form → taps, withheld | comparison, all pairs | comparison, pairs with a withheld form |
| --- | --- | --- | --- | --- | --- |
| English | 0.49 (0.07) | — | — | 0.77 (0.53) | — |
| Chinese | 0.91 (0.31) | 0.76 (0.18) | 0.82 (0.12) | 0.92 (0.70) | 0.91 (0.61) |
| Devanagari | 0.83 (0.82) | 0.76 (0.86) | 0.69 (0.66) | 0.77 (0.77) | 0.83 (0.83) |
| Thai (untaught) | 0.07 (0.09) | — | — | 0.36 (0.37) | — |

- Taps → collection matching (taught) is 0.66 (0.63). Comparison on held-out collections
  (0.97) and taps (1.00) is learned.
- **A concept learned through one sense is usable through the other, for symbols that
  never met that sense.** Withheld forms never met collections or taps, yet they match
  taps at 0.76.
- **A relation learned on collections transfers to symbols** ("which is more": 0.77–0.92).
- Where a language is forgotten (D-varied, Chinese and English), the compositions fall
  with it. They need the language to be retained.
- taps → form picks among one language's 9 forms, a different and easier task than
  matching a form against 9 fixed-area collections. Compare with chance, not with the
  isolated-matching numbers.
- **Unexplained:** untaught Thai forms are compared consistently *backwards* (0.36 in
  every seed of both varied arms), and taps → Thai is slightly below chance. Some Thai
  digits resemble Devanagari digits of other values; this is a hypothesis, not checked.
- Thai transfer (4 shots per count, 200 updates) stays near chance in every arm
  (0.12–0.15), as in Part A.

## 4. Coherence detection with no training for it

Five-step ascending runs, one interior step replaced by another count, scored by summed
successor residuals (AUROC):

| Kind | D-base | D-rehearsal | D-repeated+rehearsal | D-varied+rehearsal | D-varied |
| --- | --- | --- | --- | --- | --- |
| Collections | 0.83 | 0.86 | 0.85 | 0.89 | 0.89 |
| Devanagari | 0.82 | 0.93 | 0.86 | 0.97 | 0.98 |
| Chinese | 0.52 | 0.88 | 0.81 | 0.94 | 0.53 |
| English | 0.48 | 0.68 | 0.61 | 0.76 | 0.48 |
| Thai (untaught) | 0.42 | 0.47 | 0.44 | 0.52 | 0.45 |
| Taps | 0.50 | 0.51 | 0.49 | 1.00 | 1.00 |

- The sense that a sequence is wrong is present exactly where the concept is held: high
  for retained languages, at chance for forgotten or untaught ones.
- This is still an explicit residual calculation over given codes, not a trained fast
  detector. It is the baseline the instinct thread needs to beat on held-out combinations.

## 5. Imagery: what the system "sees" when a state starts in the core

A core state is clamped and the energy solved outward to a sense's input layer
(spec, imagery probe). Three readouts of the solved features:

- **Nearest real input** (cosine in feature space): which held-out input the solved
  features look most like. Scale-free, no decoder involved.
- **Decoded input** by a decoder trained after the fact on the frozen encoder; read by a
  count reader trained on reconstructions of real inputs.
- **The system's own reading** of the decoded input (encode again, match to collection
  prototypes).

Nearest real input has the clamped count (top 1; top-5 majority in brackets):

| Clamped state | Image sense, D-rehearsal | Image sense, D-varied+rehearsal | Sequence sense, D-varied+rehearsal |
| --- | --- | --- | --- |
| Collection prototype | 0.56 (0.59) | 0.70 (0.85) | 0.70 (0.63) |
| Taps prototype | 0.11 (0.11) | 0.81 (0.93) | 1.00 (1.00) |
| Chinese form prototype | 0.67 (0.74) | 0.81 (0.89) | 0.93 (0.93) |
| Devanagari form prototype | 0.74 (0.70) | 0.78 (0.93) | 0.89 (0.89) |
| English form prototype | 0.30 (0.33) | 0.41 (0.37) | 0.44 (0.44) |
| Imagined successor S(p₍ₙ₋₁₎) | 0.67 (0.79) | 0.71 (0.88) | 0.71 (0.79) |
| Gap between n − 1 and n + 1 | 0.52 (0.52) | 0.57 (0.71) | 0.71 (0.76) |
| Core middle layer, collections | 0.52 (0.59) | 0.67 (0.89) | 0.67 (0.70) |
| Random state (control) | 0.04 (0.07) | 0.04 (0.07) | 0.04 (0.04) |

- **Signals started in the core reach the senses with their count intact.** Thinking a
  Chinese "five" produces sequence-sense features nearest to five taps (0.93). Hearing a
  count produces image-sense features nearest to symbols of that count (0.81).
- **The gap is imagined as the missing count.** The state that fits "after n − 1 and
  before n + 1", with no item at all, reverses to features nearest to inputs of n (0.57
  image, 0.71 sequence). So the system can form a picture of a quantity it was not shown.
  This is the "something lies between 3 and 5" idea made visible.
- The imagined successor of n − 1 looks like n (0.71).
- Random core states of the same norm produce nothing count-like (0.04), so the result is
  not the readout's prior.
- In the image sense, what the core evokes is often a *symbol* rather than a collection:
  in the seed-100 D-varied+rehearsal grid, taps of 2 evoke 二, a Chinese form of 5 evokes
  五, and a Devanagari "4" evokes 四. The sense has no preference for the collection
  format.
- D-base gives chance in every row (0.00–0.22). Its core states for collections do not
  reverse to count-specific features, although forward matching works; the rehearsal
  arms do. Not explained.
- The sequence sense is untrained in arms without taps, so its rows are at chance there.

**Decoded inputs are weak.** As prior work predicts, the decoder outputs the average
input for a feature vector, and the features keep the count but not the dot positions:
- decoded collections are blobs whose spread grows with count, not countable dots;
  connected-component counting is 0.04–0.11 even on reconstructions of real collections;
- the count reader reads reconstructions of real collections at 0.53–0.60 and decoded
  imagery barely above chance;
- decoded taps are more legible: the reader reads reconstructions of real taps at 1.00 in
  the varied arms, and decoded imagery at 0.26–0.48 (gap 0.48, random 0.07);
- the system's own reading of decoded inputs is at chance: decoded inputs lie off the
  encoder's training distribution.

So the **feature-level** imagery is clear, and the **pixel-level** rendering needs a
generative decoder that can sample positions rather than average them.

The output module has no loss in Part D, so it is not reported.

## Computation

- **Part D:** 1,501 s wall on 11 workers; 13,221 worker CPU seconds. Per run 455–1,201 s
  wall (seed 102 ran in the second wave on a less loaded machine). Training CPU per run:
  D-base 839–864 s, rehearsal 433–874 s, repeated contexts 442–997 s, varied contexts
  491–1,144 s, so contexts add about 15–30% (taps through the sequence sense, a second
  factorization and the comparison loss). Evaluation 2–10 s per run. Peak RSS of the
  largest worker 1.16 GiB.
- **Imagery:** 772 s wall, 7,071 worker CPU seconds; per checkpoint 153–604 s, almost all
  of it training the two decoders (3,000 updates each) and the two count readers
  (1,500 each). The reverse solves are one linear solve of size 3 × 112 per target.
- **Selection costs:** none beyond what is reported. One smoke run per runner, one
  update-equivalence check (5 updates), the tap generator's test, and one decoder check
  on a Part A checkpoint (3,000 updates, which led to adding the count reader and the
  nearest-input readout before the full probe).
- A background watcher meant to wait for the Part D runs never exited (it matched its own
  command line). It used no CPU, but the probe started several hours later than planned.

## Status after Part D

- **Established in this pilot:**
  - rehearsal removes most forgetting;
  - varied contexts through two senses improve acquisition and retention over repeated
    contexts at matched exposure;
  - concepts compose where they were never taught: across senses, and for a relation
    learned on collections;
  - coherence violations are detectable without training wherever the concept is held;
  - core states, including the gap and the imagined successor, reverse to count-specific
    features in each sense.
- **Not yet:**
  - which part of the varied contexts helps (taps vs comparison);
  - transfer to a new language (Thai) remains at chance;
  - English stays slow (0.31–0.35);
  - pixel-level imagery.
- **Unexplained:** reversed comparison on untaught Thai forms; D-base's failure in the
  imagery probe.
- **Suggested next:**
  1. split the varied arm into taps-only and comparison-only;
  2. a trained single-pass coherence detector against the residual baseline, on held-out
     combinations (the instinct test);
  3. a generative decoder (sampling, not averaging) for pixel imagery;
  4. Part C questions, now with a concrete trigger: a coherence violation in a retained
     concept, or a gap state.
