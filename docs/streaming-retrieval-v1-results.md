# Streaming retrieval v1: results

Status: **confirmation complete, 2026-11-01.** Seeds 370–379, under the [frozen contract](streaming-retrieval-v1-contract.md). `reports/streaming-retrieval-v1-analysis.json`.

**Outcome by the frozen rule: `does-not-beat`.** On real sensory keys, input-driven retrieval makes about 11 times as many wrong switches as `hysteresis` at the comparable-cost setting (252 against 23 per 1,000 frames) and about 4 times as many at 133× the multiplications (90), and it is slower to recognize real changes. On the independent random patterns of its published setting it works, but it does not beat smoothing there either. On this evidence it gets no place in the sensory memory; `hysteresis`, one held decision on a smoothed query, does the job at about a quarter of the multiplications.

## Primary comparison

`degraded`, P = 120, ten seeds, settings frozen on seeds 299 and 298.

| method | wrong switches per 1,000 frames | outside burst windows | delay, mean (median) | missed changes | failure to reject | dropout |
|---|---|---|---|---|---|---|
| `nearest` | 134.3 | 119.2 | 14.9 (6.0) | 0.24 | 0.46 | 0.26 |
| `ema` | 50.6 | 45.8 | 16.3 (8.0) | 0.24 | 0.40 | 0.29 |
| `hysteresis` | **22.9** | 19.7 | 14.8 (6.2) | 0.22 | 0.45 | 0.25 |
| `hmm` (reference) | 45.3 | 37.6 | 14.2 (4.4) | 0.24 | 0.40 | 0.29 |
| `idp-key`, comparable cost | 251.9 | 228.6 | 38.2 (41.1) | 0.70 | 0.64 | 0.34 |
| `idp-wide` | 90.1 | 79.7 | 33.1 (37.1) | 0.63 | 0.52 | 0.44 |

Against the better smoother, `hysteresis`, paired by seed, bootstrap 95% intervals:

| candidate | wrong switches | delay (margin +1) | failure to reject (margin +0.02) | dropout (margin +0.02) | beats |
|---|---|---|---|---|---|
| `idp-key`, comparable cost | +229.0 [205.5, 247.6] | +23.4 [20.9, 25.9] | +0.19 [0.14, 0.23] | +0.09 [0.03, 0.15] | no |
| `idp-wide` | +67.2 [57.2, 79.4] | +18.3 [15.4, 21.2] | +0.07 [0.00, 0.15] | +0.19 [0.13, 0.25] | no |
| `hmm` | +22.4 [18.5, 26.3] | −0.6 [−1.1, −0.2] | −0.05 [−0.06, −0.03] | +0.04 [0.03, 0.04] | no |

`idp-key` selected without a cost limit chose the same setting as at comparable cost (K = 1): extra steps did not help. It also had no setting within budget on real keys, so these are its closest settings. Selection among those favours meeting budgets over few wrong switches, so the check below ignores rejection altogether.

**The loss does not depend on the rejection budget.** On the pilot seeds and across the whole grid, no input-driven setting reaches delay 24 on `degraded` even with rejection ignored (the lowest are 33.1 for `idp-key` and 29.5 for `idp-wide`). At delay ≤ 32, `idp-wide` has 121 wrong switches at best and `idp-key` has no setting. That is a tuned best case, and it is still behind the frozen `hysteresis`.

**Thresholds drifted.** All methods moved from pilot to confirmation seeds: for `hysteresis`, delay 21.0 → 14.8 and failure to reject 0.36 → 0.45. So the frozen settings no longer sit within budget on fresh encoders, including the smoothers' (`hysteresis` 0.45 against 0.40). The contrasts are paired and within seed, so they stand; a fixed similarity threshold across encoders is a separate weakness.

## Other conditions (descriptive)

Each condition uses its own selected settings and budgets.

| condition | `nearest` | `ema` | `hysteresis` | `hmm` | `idp-key` comparable | `idp-key` any | `idp-wide` |
|---|---|---|---|---|---|---|---|
| `clean`: wrong switches | 28.3 | 3.1 | **0.5** | 5.1 | 240.1 | 240.1 | 19.7 |
| `clean`: delay | 3.5 | 12.9 | 12.3 | 4.0 | 33.4 | 33.4 | 22.0 |
| `held`: wrong switches | 101.7 | 27.9 | **9.7** | 13.0 | 126.8 | 126.8 | not run |
| `held`: delay | 12.6 | 17.3 | 17.7 | 17.5 | 38.1 | 38.1 | not run |
| `oracle`: wrong switches | 171.4 | 0.08 | **0.03** | 1.20 | 0.27 | 0.16 | not run |
| `oracle`: delay | 5.7 | 7.3 | 7.2 | 4.5 | 7.7 | 7.6 | not run |

- **`held`** was expected to expose smoothing's weakness, since errors persist for 8 frames. Smoothing does not collapse, and input-driven dynamics do not gain from it.
- **`oracle`** is the mechanism's setting, and it works: every method but `nearest` is near zero, at 4× the multiplications of `ema` for the comparable `idp-key`. There wrong switches sit at a floor (`hysteresis` 0.03), so a 20% reduction could not be shown; `idp-key` is ahead of `nearest` and level with smoothing. The `oracle` frontier was read on pilot seeds only: at delay ≤ 4, `hysteresis` 0.10, `ema` 0.49, `idp-key` 0.69, `hmm` 1.35. Nothing there favours the mechanism.

**Burst capture, primary condition.** Share of bursts followed by a wrong switch:

| method | intrude | flash | corrupt | 1 frame | 2 frames | 4 frames |
|---|---|---|---|---|---|---|
| `nearest` | 0.90 | 0.70 | 0.74 | 0.73 | 0.77 | 0.83 |
| `ema` | 0.36 | 0.29 | 0.39 | 0.27 | 0.25 | 0.51 |
| `hysteresis` | 0.33 | 0.25 | 0.34 | 0.20 | 0.21 | 0.49 |
| `hmm` | 0.72 | 0.49 | 0.50 | 0.52 | 0.55 | 0.65 |
| `idp-key`, comparable cost | 0.77 | 0.80 | 0.76 | 0.74 | 0.77 | 0.81 |
| `idp-wide` | 0.65 | 0.61 | 0.62 | 0.50 | 0.62 | 0.75 |

These are inflated by each method's base rate of wrong switches in the same window, which is largest for `idp-key`. Even so, the mechanism's published strength, ignoring a brief glitch, shows no advantage over `ema` here. For the frozen settings most wrong switches come from ordinary frames outside bursts, in every method.

**Frontier.** Four seeds, each method tuned on the test streams, fewest wrong switches at the primary condition's rejection limits:

| delay budget | `ema` | `hysteresis` | `hmm` | `idp-key` | `nearest` |
|---|---|---|---|---|---|
| 16 | none | none | 27.5 | none | none |
| 24 | 22.6 | none | **8.5** | none | 101.3 |
| 32 | 10.3 | **3.0** | 7.4 | none | 101.3 |

Tuned to the test streams, `hmm` is the strongest method at short delay and `hysteresis` at longer delay. The rule's frozen `hmm` setting underperformed its tuned one (45.3 against 8.5). `idp-key` has no setting at any budget. `hmm` costs little more than the smoothers, and it is the established reference.

## Why it fails on real keys (post hoc)

`experiments/streaming_retrieval_diagnosis.py`; `reports/streaming-retrieval-v1-diagnosis.json`. Not part of the rule.

**Settling.** Constant *exact* cue, no stream, published time step (0.01), 40 words on each of four seeds. Readout after 20 time units, ranges over gains 4–64, σ ∈ {0, 2}:

| patterns | correct | top overlap | runner-up | mean \|cosine\| | 99th percentile \|cosine\| | effective rank |
|---|---|---|---|---|---|---|
| real keys, width 32 | **0.56–0.59** | 0.76–0.77 | 0.69–0.70 | 0.21 | 0.76 | 12.7 |
| sign patterns of real keys, width 1,024 | **0.79** | 0.87–0.92 | 0.60–0.61 | 0.14 | 0.56 | 23.9 |
| independent patterns, width 1,024 | **1.00** | 0.99–1.00 | 0.09 | 0.03 | 0.08 | 107.6 |

Nearest-key retrieval is correct on all of these exact cues by construction, so **the dynamics discard identity the similarities still contain**. Between the last two rows width is the same and correlation differs, and that accounts for 0.79 → 1.00. Between the first two, the same keys at greater width go from about 0.6 to 0.79. Stored patterns' correlations are what the published theory rules out by assumption (orthogonal, or independent at large N). This is a property of retrieval on these keys, independent of streaming, and streams can only make it worse. Our keys were trained to separate nearest neighbours, not to be near-orthogonal, and that is sufficient for the methods that read similarities directly.

## Rejection is limited by the keys, not by time (post hoc)

AUROC of the top similarity, familiar against unfamiliar settled frames, ten seeds:

| condition | single frame | smoothed (a = 1/8) | keys and queries centred, single frame | centred, smoothed | norm of the stream's mean query |
|---|---|---|---|---|---|
| `clean` | 0.92 | 0.92 | 0.93 | 0.93 | 0.29 |
| `degraded` | 0.70 | 0.71 | 0.78 | 0.80 | 0.44 |
| `held` | 0.70 | 0.70 | 0.78 | 0.80 | 0.44 |

- **Smoothing does not help rejection.** A familiar symbol's top similarity and an unfamiliar symbol's top similarity differ persistently, not by per-frame noise, so more frames add little. This accounts for the limit near 0.36 that every method met.
- **Corruption adds a common component, partly.** The mean query's norm rises with corruption, and centring raises `degraded` to 0.78–0.80. It does not close the gap to `clean`, so the common component is only part of the loss. It uses no labels, but it was chosen after seeing this result and estimates the mean from the stream's first 2,000 frames, so it is a lead, not a measured remedy.

The contract expected `idp` not to beat `hysteresis`, with possible departures on `held` or `oracle`. No departure occurred.

## Scaling probe

Settings frozen at P = 120; primary condition; wrong switches, then failure to reject and dropout.

| P | `ema` | `hysteresis` | `hmm` | `idp-key`, comparable | `idp-wide` | `nearest` |
|---|---|---|---|---|---|---|
| 30 | 10.7 (0.11, 0.34) | **5.6** (0.13, 0.30) | 12.5 (0.11, 0.34) | 30.9 (0.04, 0.87) | 31.7 (0.87, 0.05) | 35.4 (0.14, 0.33) |
| 120 | 50.6 (0.40, 0.29) | **22.9** (0.45, 0.25) | 45.3 (0.40, 0.29) | 251.9 (0.64, 0.34) | 90.1 (0.52, 0.44) | 134.3 (0.46, 0.26) |
| 360 | 100.9 (0.49, 0.25) | **40.1** (0.54, 0.21) | 78.3 (0.48, 0.25) | 281.2 (0.94, 0.05) | 118.0 (0.34, 0.64) | 256.6 (0.59, 0.20) |

Smoothers degrade gradually and their threshold shift is modest. Input-driven settings do not transfer: `idp-key` goes from dropping 87% of familiar frames at P = 30 to accepting 94% of unfamiliar ones at P = 360, because the sum of saliencies grows with P; `idp-wide` swings in the reverse direction. The gain would have to be retuned with load, or normalized, which the mechanism does not specify. Work per frame is linear in P for every method, as the contract derives, and no advantage in state size exists at any P.

## Compute

Multiplications are derived; µs per frame are measured: one thread, one warm-up pass then 15 passes over 2,000 frames (median, interquartile range), including allocation; `nearest` and the similarity reads are done as one batch per pass, which is why `nearest` looks so much faster than its multiplications suggest. It is not a per-frame online figure. Encoder: batch of 256, 20 passes.

| method | multiplications per frame | × `ema` | µs per frame | recurrent state, floats | stored patterns, bytes |
|---|---|---|---|---|---|
| `nearest` | 3,840 | 1.0 | 0.13 (0.004) | 0 | 15,360 |
| `ema` | 3,960 | 1 | 3.13 (0.01) | 32 | 15,360 |
| `hysteresis` | 3,960 | 1 | 3.57 (0.01) | 32 + one index | 15,360 |
| `hmm` | 4,200 | 1.1 | 10.76 (0.35) | 121 | 15,360 |
| `idp-key`, K = 1 | 15,664 | 4.0 | 24.2 (0.5) | 32 | 15,360 |
| `idp-wide`, K = 1 | 526,576 | 133 | 628 (3.5) | 1,024 | 491,520 patterns as float32 + 131,072 projection |
| encoder, shared | 458,816 | 116 | 19.2 (4.6) | none | 839,040 |

Measured, `idp-key` costs about 6.8× `hysteresis`, more than its 4× in multiplications. Against the encoder, retrieval is cheap for every method but `idp-wide`, so per-frame compute would not decide between the others; quality does.

**Selection and tuning, included in the comparison.** Pilot: 168,800 setting-threshold pairs across 106 tasks, 13,491 worker CPU seconds (`idp-wide` about a quarter), 11 workers, peak worker RSS 484 MB. Stream building, two seeds: 13 s wall. Severity stage: 2 s wall.

**Confirmation.** Ten encoders: 66 s each on the GTX 1650 (render and fit; 27 s fit alone). Confirmation: 718 s wall: streams 28 s (256 s worker CPU); 400 selected runs 122 s; frontier, 48 tasks and 102,932 pairs on 4 seeds, 568 s (6,297 worker CPU seconds); children's peak RSS 801 MB. Diagnosis: 420 s wall, of which settling 1,628 worker CPU seconds. Timing is in `reports/streaming-retrieval-v1-*-runtime.json`, separately from the deterministic files.

## Reading and limits

**What this establishes.** On this stream, with one stored rendering per word and keys from a nearest-neighbour encoder, input-driven retrieval as implemented does not beat a held decision on a smoothed query at any budget we could set, and costs more. With independent patterns it does not beat smoothing either, though the floor there limits how much that can show.

**Decision.** Temporal stage for sensory memory: `hysteresis`, with `hmm` as the alternative to weigh if short delay matters. Recurrent attractor state has **no established role** from this test, and the reason on real keys is specific and measured.

**What it does not establish.**
- It is not a test of the published model's claims in its own setting, which unit tests and `oracle` here are consistent with. The paper studies ambiguity and noise on constructed mixtures, with long-term learning left open; none of that is contradicted here.
- Keys engineered for near-orthogonality might raise settling to `oracle`'s level, but that would tie smoothing, not beat it, on the evidence of `oracle`.
- Frames here are independent or held for 8; real sequences have richer correlation. Conflicting cues across senses, the note's suggested extension, were not run.
- `idp-key` uses K far below the published step count: only K ≤ 16 was tried, at τ ≥ 0.25. Settling at the published step size under an exact cue is still only 0.56–0.59 correct on real keys, by the diagnosis.
- `idp-wide` was not extended at its τ and σ edges, per the contract.
- `hmm`'s vMF evidence and constant unfamiliar evidence are not fitted, and the rule's frozen setting is much worse than its tuned one; it is a reference, not a finding about filters.

## Reproduction

Commands in the [contract](streaming-retrieval-v1-contract.md#reproduction). Post hoc: `NSY_WORKERS=11 .venv/bin/python -m experiments.streaming_retrieval_diagnosis diagnose`. Confirmation ran from the frozen revision 2db201f with one uncommitted change, GPU setup in the `encoders` stage, recorded as `dirty` in its runtime file; the change is committed with these results.
