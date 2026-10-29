# Cross-sense confirmation v1: is SRI's form → taps advantage real?

Status: **frozen 2026-10-29, before any run on seeds 200–209.** Follows
[architecture audit stage 1](architecture-audit-stage1-results.md). Researchers'
decision (2026-10-29): the standard network is the default for extending the number line.
SRI gets one focused confirmation experiment, this one, on its candidate cross-sense
advantage.

## Question

In stage 1, matching a withheld Chinese form to the taps of the same count (form → taps)
scored 0.82 on SRI and 0.50 on the standard network. Seeds: 0.90, 0.72, 0.85 against
0.37, 0.57, 0.57. The reverse direction (taps → form) was equal. The difference was found
post hoc, on three seeds already used to design the Part D pilot. Is it real, or a fluke?

## Design

- **Arm:** Part D's D-varied+rehearsal, the arm where the difference was seen. It runs
  unchanged, with the Part D pilot settings (`pilot` mode).
- **Networks:** SRI (`src/number_systems_net.py`) and the standard network
  (`src/standard_net.py`), exactly as in stage 1.
- **Seeds:** **200–209**, never used in the number line. Each seed gives both networks the
  same data streams, encoder initialization and withheld counts. The withheld counts
  follow Part A's rule for each seed.
- **Readout:** the runner's `evaluate_senses`, unchanged. Each withheld Chinese form (30
  evaluation items per withheld count) picks among one taps item per count by the *same*
  score. Chance is 1/9.
- **Commands** (`NSY_WORKERS=11`):
  - `.venv/bin/python -m experiments.cross_sense_confirmation pilot sri`
  - `.venv/bin/python -m experiments.cross_sense_confirmation pilot standard`
  - `.venv/bin/python -m experiments.cross_sense_confirmation_analysis`

## Primary measure and decision rule

**Primary:** `form_to_taps/chinese/withheld` at the end of training. The analysis uses
the paired difference d = SRI − standard per seed, 10 seeds.

Statistics: the mean of d, and its 95% bootstrap percentile interval (10,000 resamples of
the 10 seeds, generator seed 0). A two-sided sign test is reported alongside. Practical
margin: **0.10**.

| Outcome | Condition |
| --- | --- |
| **confirmed** | interval lower bound > 0 **and** mean d ≥ 0.10 |
| **not confirmed** | interval upper bound < 0.10 (any advantage is smaller than the margin) |
| **inconclusive** | otherwise |

What follows from each outcome:
- **Confirmed:** SRI has a real cross-sense advantage on this measure. The next question
  is whether it comes from the coupling (stage 2 or 3 rungs) and whether a standard
  method gets it too (see Prior work).
- **Not confirmed:** the stage 1 difference was a fluke of seeds 100–102. The standard
  network is the default without reservation.
- **Inconclusive:** reported as such. No further SRI seeds are run without a new
  decision.

## Secondary measures (reported, not decisive)

1. form → taps for withheld Devanagari forms, and for paired Chinese and Devanagari
   forms. Is the effect specific to withheld Chinese?
2. taps → form for withheld Chinese and Devanagari. Does the direction asymmetry
   replicate?
3. taps → collection (the taught pairing) and isolated matching of withheld Chinese at
   the end. Do differences in what was learned or retained explain the effect?

## Diagnostics (explanatory, not decisive)

The analysis recomputes the primary readout from the checkpoints and checks that it
matches the runner's value. It then reports:
- **Hubness of the taps gallery:**
  - the largest share of all form → taps picks taken by one count;
  - pick entropy;
  - the skewness of k-occurrence (k = 5) of taps codes among form codes' nearest
    neighbours (Radovanović et al. 2010).
- **A hubness-corrected readout (CSLS-style):** each taps code's score is reduced by its
  mean top-10 score over all form queries of the same language.

  If the standard network's deficit is hubness, the correction should close most of the
  gap.

## Prior work

Search date 2026-10-29. Queries: "hubness nearest neighbor retrieval Radovanović 2010",
"Conneau 2018 word translation without parallel data CSLS", "Dinu Lazaridou Baroni 2015
hubness zero-shot".
- **Hubness.** [Radovanović, Nanopoulos and Ivanović 2010, *Hubs in Space*, JMLR 11](https://jmlr.org/papers/v11/radovanovic10a.html)
  [abstract].
  - *Shown:* in high-dimensional data, the k-occurrence distribution becomes skewed. A
    few points become nearest neighbours of many others.
  - *Differs:* general data, not two learned modalities; the codes here are 16-dim and the
    galleries have 9 candidates.
  - *Implication:* a retrieval failure in only one direction can come from hubs in the
    gallery side, not from missing knowledge [analogy].
- **Hubness in cross-space mapping.** [Dinu, Lazaridou and Baroni 2015, ICLR workshop](https://arxiv.org/abs/1412.6568)
  [abstract].
  - *Shown:* after mapping one space into another, hubs crowd out correct neighbours in
    zero-shot retrieval. A correction for how often a candidate is near many mapped
    vectors improves cross-lingual and image retrieval.
  - *Implication:* the standard network's taps codes may act as hubs for form queries.
- **CSLS.** [Conneau et al. 2018, *Word Translation Without Parallel Data*, ICLR](https://arxiv.org/abs/1710.04087)
  [abstract and search summary].
  - *Shown:* subtracting each candidate's mean similarity to its nearest neighbours in
    the other space (CSLS) reduces hubness and improves word translation.
  - *Implication:* the cheap standard alternative. If CSLS closes the gap, the advantage
    is in the readout geometry, which a standard correction recovers, not in the
    knowledge.
- **Shared-anchor alignment** (ImageBind; [audit](architecture-audit.md#standard-explanations-for-each-result)).
  It predicts that untaught pairs align in both directions. It does not predict an
  asymmetry.

**What theory predicts:** nothing about which network is better. Hubness predicts that a
deficit confined to one retrieval direction should show a skewed pick distribution, and
should shrink under CSLS.

**What the study adds:** a fresh-seed test of the only cross-sense SRI advantage left
after stage 1. "Nothing new beyond that; a needed internal confirmation" applies.

## Known failure modes

- **Withheld counts vary by seed**, two per language, and some counts are harder than
  others. Pairing by seed controls this between networks. Ten seeds sample it.
- **The difference could come from Chinese being forgotten** rather than from cross-sense
  structure. Secondary 3 reports retention.
- **Three of the inputs were seen before.** The arm, measure and margin were chosen after
  seeing seeds 100–102. Only the seeds are fresh, which is why the rule above is fixed
  now.

## Computation

- **Dimensions:** as in stage 1. 6,200 updates per run: 3 blocks × 2,000, plus 200 Thai
  transfer updates. Batch of 16 language episodes, 4 context episodes and rehearsal. 16-dim
  codes; 64-dim features.
- **Runs:** 20 (2 networks × 10 seeds), on 11 workers.
- **Expected cost** from stage 1's measured runs, an estimate: about 900 s CPU per SRI run
  and about 750 s per standard run, so about 5.5 CPU hours in total. Wall time is about
  2 × 25 minutes, since each network runs as one job of two waves.
- **Order of costs:**
  - per update, SRI adds one Cholesky factorization of a 272-state system,
    O(272³), per solver;
  - per encoded item, an O(272²) solve against O(64 × 16) for the standard projection;
  - the CNN front ends dominate both.
- **Analysis:**
  - data preparation as in the runners;
  - per checkpoint, codes for 270 forms and 270 taps, plus a 270 × 270 score matrix;
  - a few minutes.
- **Storage:** 20 checkpoints of about 1.5 MB each (not tracked); compressed results JSON.
- **Selection and tuning:** none. All settings are inherited from the Part D pilot and
  stage 1.
- **Timing** goes to the runners' runtime files and the analysis runtime file. Results
  and timing are kept separate.
