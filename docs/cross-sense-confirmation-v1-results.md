# Cross-sense confirmation v1: results

Contract: [cross-sense confirmation v1](cross-sense-confirmation-v1-contract.md), frozen
before the run. Run 2026-10-29, seeds 200–209.

- **Commands** (`NSY_WORKERS=11`, `.venv/bin/python -m`):
  - `experiments.cross_sense_confirmation pilot sri` (17 minutes);
  - `experiments.cross_sense_confirmation pilot standard` (14 minutes);
  - `experiments.cross_sense_confirmation_analysis` (under a minute).
- **Artifacts:**
  - `reports/cross-sense-confirmation-v1-{sri,standard}-pilot-{results.json.gz,runtime.json}`;
  - `reports/cross-sense-confirmation-v1-results.json.gz`;
  - `reports/cross-sense-confirmation-v1-analysis-runtime.json`;
  - follow-up: `reports/cross-sense-confirmation-v1-calibration-{results.json.gz,runtime.json}`
    from `.venv/bin/python -m experiments.cross_sense_calibration`;
  - checkpoints in `runs/cross-sense-confirmation-v1-*-pilot/` (not tracked).
- **Recomputation:** the analysis recomputes the primary readout from the checkpoints. It
  matches the runner's values exactly (maximum error 0).

## Summary

- **By the frozen rule, the primary result is confirmed.** SRI matches withheld Chinese
  forms to taps far better than the standard network on fresh seeds.
- **The diagnostic explains it as hubness, and a standard correction closes the gap.**
  - The standard network's deficit is one-directional and not specific to withheld or
    Chinese forms: form → taps fails for paired forms and for Devanagari too.
  - The reverse direction is equal.
  - Its taps codes form hubs: a few taps codes are the nearest match for many form
    queries.
  - A CSLS-style correction, applied at readout with no training, closes the gap:
    0.81 against 0.80, a difference of −0.02 [−0.06, 0.03].
  - The same holds when the correction is estimated from a separate calibration set,
    frozen, and test inputs are scored one at a time
    ([follow-up](#follow-up-frozen-calibration)).

**SRI learns representations that work substantially better with the original matching
rule, with consistently less hubness. A hubness correction recovers comparable
performance from the standard representations.** This does not show that the two
representations hold identical knowledge or support identical future learning. SRI's
better direct matching is a real benefit. Whether it justifies SRI's training cost is a
separate question, and which architectural difference produces it is not known.

## Primary (decision rule as frozen)

| | Mean | Seeds 200–209 |
| --- | --- | --- |
| SRI | 0.77 | 0.80, 0.63, 0.50, 0.78, 0.95, 0.77, 0.78, 0.82, 0.75, 0.93 |
| Standard | 0.37 | 0.18, 0.03, 0.68, 0.43, 0.63, 0.13, 0.58, 0.52, 0.23, 0.28 |
| SRI − standard | **+0.40** | 95% bootstrap interval [0.23, 0.54]; 9 of 10 seeds positive; sign test p = 0.021 |

The interval's lower bound is above 0 and the mean exceeds the 0.10 margin: **confirmed**.
Chance is 0.11. The stage 1 difference (0.82 against 0.50) replicates on fresh seeds,
and is larger.

## Secondary

| Measure | SRI | Standard | SRI − standard |
| --- | --- | --- | --- |
| form → taps, withheld Devanagari | 0.65 | 0.25 | +0.39 |
| form → taps, paired Chinese | 0.85 | 0.36 | +0.49 |
| form → taps, paired Devanagari | 0.83 | 0.26 | +0.57 |
| taps → form, withheld Chinese | 0.85 | 0.83 | +0.02 |
| taps → form, withheld Devanagari | 0.70 | 0.70 | −0.01 |
| taps → collection (taught) | 0.60 | 0.61 | −0.01 |
| Withheld Chinese, matched alone to collections, end | 0.52 | 0.48 | +0.04 |

- **The deficit is in one direction only, for all forms.** It is not about withheld
  forms or Chinese.
- **On these measures the networks perform alike.** The taught pairing, the reverse
  direction and the forms' own matching are equal, so forgetting does not explain the
  difference. This is equal performance on these tasks, not evidence of identical
  representations.

## Diagnostics: hubness

Means over the 10 seeds.

| | SRI | Standard | Paired difference |
| --- | --- | --- | --- |
| Largest share of form → taps picks taken by one count (uniform: 0.11) | 0.15 | 0.24 | SRI lower in 10 of 10 seeds |
| Pick entropy (bits; uniform: 3.17) | 3.14 | 2.73 | SRI higher in 10 of 10 |
| k-occurrence skewness of taps codes among form codes' 5 nearest neighbours | 1.46 | 2.71 | SRI lower in 10 of 10 |
| form → taps, withheld Chinese, CSLS-corrected | 0.80 | 0.81 | −0.02 [−0.06, 0.03]; 5 of 10 positive |
| form → taps, all forms, CSLS-corrected | 0.85 | 0.82 | +0.02 |
| Largest pick share after CSLS | 0.14 | 0.14 | — |

- **The standard network's taps codes are hubs** for Chinese form queries. In every seed,
  the picks concentrate on a few counts, and the k-occurrence distribution is skewed.
- **The CSLS-style correction removes the hubs, and the large gap disappears.** The
  correction: subtract each taps code's mean top-10 score over all form queries.
  Afterwards both networks' picks are equally spread. The accuracy interval
  [−0.06, 0.03] supports "the gap disappears", not exact equality.
- **This diagnostic uses the whole evaluation population of queries.** It needs no answer
  labels, but each corrected answer draws on information beyond the current query. The
  follow-up below removes that access.
- In the reverse direction, taps queries select among form codes, which are not hubs.
  This explains the asymmetry: Radovanović et al. 2010; Dinu et al. 2015; Conneau et al.
  2018 (see the contract's Prior work).
- **Why SRI's codes are less hub-prone is not tested.** Candidates: the coupled
  parameterization or the learned penalty from the output module
  ([audit](architecture-audit.md#what-the-current-network-computes), facts 1–2). That
  would be stage 3's T1/T2 rungs. It matters only if hub resistance is wanted as a
  property of training, not of the readout.

## Follow-up: frozen calibration

Added after an external review of this result (2026-10-29). The intended system meets
inputs one at a time, so the correction must not use the test queries.

**Design** (`experiments/cross_sense_calibration.py`; criterion fixed before running):
- each taps candidate's penalty r(g) is estimated from a fixed calibration set: 270
  training-split forms, never evaluation items, sampled without labels by a fixed
  generator;
- r(g) is the mean of g's top-10 scores over the calibration set, frozen before any test
  query;
- each test query is then scored alone, against its 9 candidates.

Two calibration sets:
- **same language:** Chinese training forms;
- **other language:** Devanagari training forms.

**Criterion:** the correction works under these conditions if SRI (plain) − standard
(frozen, same-language calibration) has a 95% bootstrap upper bound below 0.10.

| form → taps, withheld Chinese (seeds 200–209) | SRI | Standard |
| --- | --- | --- |
| Plain (as in the primary) | 0.77 | 0.37 |
| Frozen correction, same-language calibration | 0.78 | **0.80** (0.68–0.90) |
| Frozen correction, other-language calibration | 0.72 | 0.71 (0.55–0.87) |

- SRI (plain) − standard (frozen, same language): **−0.03 [−0.10, 0.04]**, 5 of 10 seeds
  positive. **The criterion is met.** The cheap remedy works with a frozen calibration
  set and one query at a time.
- With both networks corrected the same way: −0.02 [−0.09, 0.05].
- Calibrating on another language still recovers most of the gap (0.37 → 0.71), with less
  than same-language calibration.
- **Cost of the remedy:**
  - storage: 270 calibration codes (270 × 16 float32, 17 KB) or one penalty per
    candidate;
  - work: one score per calibration code and candidate, O(N_cal · d) per candidate;
  - no training.

  Storing past input codes is a form of retained history. Under the program's rule
  against a replay store ([research program](research-program.md) §2.4), it would have
  to be justified as a fixed calibration statistic.
- **Run cost:** under a minute, mostly data preparation (`-calibration-runtime.json`).

## Computation

| | SRI | Standard |
| --- | --- | --- |
| Worker CPU, 10 runs × 6,200 updates (plus 200 transfer updates) | 9,711 s | 7,954 s |
| Job wall time, 10 workers busy | 1,037 s | 858 s |
| Run wall time per seed | 928–1,008 s | 743–830 s |

- **Analysis:** 28 s wall, 8 s CPU. Data preparation is 27 s of that; diagnostics on all
  20 checkpoints take 0.6 s.
- **CSLS cost:** for a gallery of G candidates against Q queries, one Q × G score matrix
  and a top-k per column, O(QG(d + log k)). Here 270 × 270.
- **Selection and tuning:** none. The CSLS k (10) and the hubness k (5) were fixed in the
  contract.

## Bearing on the audit

- **SRI has demonstrated better direct cross-sense retrieval and consistently less
  hubness** in this setting. "SRI offers no benefit" would be incorrect.
- **Standard training plus a hubness correction is a strong, cheaper alternative.** It
  also works with a frozen, independent calibration set. The standard network stays the
  practical default. Where retrieval between senses matters, it should use a
  hubness-corrected readout (CSLS or similar) or a training-time remedy.
- **Open:** which SRI difference produces the hub-resistant geometry (stage 3's T1/T2
  rungs), and whether that geometry matters beyond retrieval, for example for later
  learning.
- **For mismatch-driven learning (asking questions):** a system should distinguish a
  retrieval distortion from missing knowledge before deciding what to ask. Here a
  hubness distortion made the standard network look ignorant of form → taps
  correspondences that its codes support. Its apparent uncertainty would have
  misdirected its questions.
- **Caveat:** the arm, the measure and the margin were chosen after seeing seeds 100–102.
  The seeds and the diagnostics were fixed before this run.
