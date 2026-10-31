# Streaming retrieval v1: contract

Status: **draft, 2026-10-31**. The pilot on seeds 299 and 298 fixes the budgets below; the contract is frozen before confirmation seeds 370–379 are built.

## Question

Does input-driven retrieval ([Betteti et al., 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12017325/)) beat simple temporal smoothing at comparable cost, on identical sensory sequences? If it does, recurrent state has a concrete role in the sensory memory system. If it does not, we learn that cheaply. This is a small retrieval test, not an architecture replacement: no learning, no host, no writes during a stream.

Readings: wrong switches, delay in recognizing real changes, failure to reject unfamiliar input, compute.

## Material

- **Sensor.** The frozen key encoder of the [stable key revision](translation-memory-v1-pilot.md#stable-keys-perception-gate-passes) (base network, 32-dimensional unit-norm key, 839,040 bytes), one per seed, trained on background symbols only.
- **Symbols.** The task's 512-symbol pool, which no encoder trained on. Per seed, a fixed permutation: the first 360 can be stored (nested across loads), the last 100 are always unfamiliar. Pool membership is shared across seeds, as it is in the translation-memory gates.
- **Memory.** One key per word, from one training rendering. Primary load P = 120. Fixed for the whole stream.
- **Frames.** A frame is one image through the encoder. Every method reads the same frames.

## Streams

14,400 frames per stream. Dwells last 32–64 frames, uniformly. After a familiar dwell, the next is unfamiliar with probability 0.2, showing one unfamiliar symbol. Consecutive dwells never repeat a word. An unfamiliar dwell is always followed by a familiar one.

Half of familiar dwells contain one **transient burst** of 1, 2 or 4 frames, starting at least 16 frames after onset and ending at least 8 before the dwell's end:

- `intrude`: another stored word;
- `flash`: an unfamiliar symbol;
- `corrupt`: the same word at burst-level corruption.

A burst never changes the truth, which is the dwell's word.

**Conditions.** Within a seed and load all conditions share one schedule.

| condition | frames |
|---|---|
| `clean` | independent renderings per frame |
| `degraded` (**primary**) | independent renderings plus fresh noise and occluder per frame |
| `held` | as `degraded`, but the rendering is held for 8 frames, so errors from the rendering persist. Smoothing's expected failure mode |
| `oracle` | independent random ±1 patterns of width 1,024 and random query noise, in the mechanism's published setting. Separates "the mechanism has no advantage" from "our keys defeat it" |

**Corruption levels, by rule on validation symbols of seed 299, before any stream exists:** the first level of a fixed ladder of noise and occluder sizes whose single-frame 120-way nearest-key accuracy is at most 0.70 (`degraded`) and at most 0.20 (burst). Oracle signal strengths are bisected to accuracy 0.96, 0.70 and 0.20; `oracle` uses 0.70 and 0.20.

## Methods

All read the same similarities (cosines of the frame's key against P stored keys), online, with no look-ahead. Each outputs per frame a stored word or REJECT.

| name | method | state |
|---|---|---|
| `nearest` | Largest similarity; REJECT below a threshold | none |
| `ema` | Exponential smoothing of similarities at rate a, then as `nearest` | D floats, **[derived]** since linearity makes it nearest-key on the smoothed query |
| `hysteresis` | As `ema`, plus a held decision: yields only to a word ahead by δ, or when falling below τ − δ. A word is accepted at τ | D floats, one index |
| `hmm` | Forward filter over P words and an unfamiliar state, vMF evidence κ·similarity, constant evidence κτ for unfamiliar, uniform switch hazard h. **Reference, not one of the compared mechanisms** | P + 1 floats |
| `idp-key` | Input-driven dynamics on the keys themselves, D = 32 | D floats |
| `idp-wide` | Input-driven dynamics at N = 1,024, with stored patterns the signs of a fixed Gaussian projection of the keys, and input its projection. Primary condition only | N floats |

**Input-driven dynamics** (Eqs. 7, 8 and 19 of the published paper, Euler–Maruyama):

x ← x + (τ/K)(−x + W(u)Ψ(x)) + σ√(τ/K)·η, W(u) = (1/N) Σ_μ α_μ ξ^μ ξ^μᵀ, α_μ = g·(similarity_μ)², Ψ = tanh(5x), x(0) ~ N(0, I).

K steps run per frame over a frame duration τ. The gain g places similarities in the α range the paper uses, since our cosines are not its constructed mixtures. W is never formed; each step costs two P×N matrix-vector products. Readout: the pattern with the largest |overlap| m_μ = ξ^μᵀΨ(x)/N, REJECT below a threshold. The paper's "confusion state", where small α lets x fall to the origin, is what makes rejection natural here.

The implementation is called *inspired by*, not a reproduction. Differences: real cosine saliencies, K smaller than the published 0.01 time step (from the [preprint](https://arxiv.org/abs/2411.05849); the published main text refers to its supplement, which was not read), correlated keys, a 32-dimensional key against published N = 1,024 at `idp-key`. The preprint's weight is linear in overlap and its ψ is tanh(z); the contract follows the published version. The unit tests check the published behaviour on random ±1 patterns at N = 1,024: retrieval and tracking of the dominant memory, a brief glitch left without effect, a vague input left at the origin, noise shortening a switch.

**Not included:** the classic additive-input Hopfield (Eq. 4). The paper already reports it failing under noise, and our earlier [unclamped completion](translation-memory-v1-pilot.md) added nothing. `hmm` covers the statistical optimum, so an added leaky competing accumulator is not needed for a first answer.

## Settings

| method | grid |
|---|---|
| `nearest` | threshold 0 to 2× the median nearest similarity on the first pilot stream of that condition, steps of 0.05× |
| `ema` | a ∈ {1/2, 1/3, 1/4, 1/6, 1/8, 1/12, 1/16, 1/24}, threshold as above |
| `hysteresis` | a as above, τ ∈ {0.5, 0.65, 0.8, 1, 1.2}× median, δ ∈ {0.05, 0.1, 0.2, 0.3, 0.5}× median |
| `hmm` | κ ∈ {5, 10, 20, 40, 80}/median, h ∈ {0.001, 0.003, 0.01, 0.03, 0.1}, τ as `hysteresis` |
| `idp-key` | g ∈ {1, 2, 4, 8, 16, 32}, τ ∈ {0.1, 0.25, 0.5, 1, 2}, K ∈ {1, 2, 4, 16} (16 only for τ ≥ 0.25), σ ∈ {0, 1, 2, 4}, overlap threshold 0–1 in steps of 0.025 |
| `idp-wide` | g ∈ {4, 8, 16, 32}, τ ∈ {0.25, 0.5, 1}, K ∈ {1, 2}, σ ∈ {0, 2}, threshold as above |
| `idp` at `oracle` | g ∈ {64, 128, 256, 512, 1024}, τ, K, σ as `idp-wide`, threshold as above |

If a selected setting lies on a grid edge in the pilot, the grid may be extended before freezing; the result of doing so is recorded.

## Readings

Per stream, on the decision sequence y:

- **Wrong switches per 1,000 frames (primary).** Frames where y changes to a stored word other than the truth. Staying on an old word is not a switch; it counts as delay. Also split: outside burst windows, and capture rate by burst kind and length (bursts with at least one wrong switch before 8 frames after the burst).
- **Delay.** For each change into a familiar word, frames from onset until y holds the word for 3 frames, censored at the dwell's length; mean, median and miss rate. Unfamiliar-to-familiar changes count. The stream's first dwell does not.
- **Failure to reject.** Share of frames more than 16 after an unfamiliar onset with y not REJECT. Also frames until REJECT holds for 3 frames.
- **Dropout.** Share of familiar frames, more than 16 after onset and outside burst windows, with y = REJECT.
- **Compute.** Multiplications per frame (derived), measured µs per frame on one thread, recurrent state and stored floats.

## Selection and decision rule

**Selection, on pilot seeds 299 and 298, primary condition, P = 120, one setting per method.** Among settings whose means meet the budgets, the fewest wrong switches, then the fewest multiplications. The provisional budgets are **delay mean ≤ 8 frames** (twice the longest burst), **failure to reject ≤ 0.10**, **dropout ≤ 0.10**. If no method can meet them, the pilot changes them before freezing, and the same numbers apply to every method. A method with no setting within budget reports its closest and is marked.

`idp-key` is selected twice: among settings with **at most 6× the retrieval multiplications of `ema`** (**comparable cost**; this admits K ≤ 2), and without limit. The factor is generous because a single dynamics step is already 2 extra matrix-vector products, and the smoothing methods have nothing further to spend. Comparison uses retrieval work only, even though the encoder dominates each frame.

**Better smoother:** whichever of `ema` and `hysteresis` is selected with fewer wrong switches on the pilot (within budget first).

**Confirmation, seeds 370–379, fresh encoders, streams and word tables, settings frozen.** Paired by seed, bootstrap over seeds, 10,000 resamples:

A candidate **beats** the better smoother if all hold:

1. the mean reduction in wrong switches is at least 20%, and the 95% interval of the difference excludes 0;
2. the upper 95% bound of the difference is at most +1 frame on delay, +0.02 on failure to reject and +0.02 on dropout.

**Outcomes:**
- `earns-role`: `idp-key` at comparable cost beats it.
- `beats-only-at-higher-cost`: only `idp-key` without limit, or `idp-wide`, beats it.
- `does-not-beat`: neither. Then this mechanism has no claim on a place in the sensory memory, on this evidence.

Descriptive only: every other condition, loads, capture by kind, `hmm` against everything, and the frontier.

**Frontier.** On four confirmation seeds, the full grid (`idp-wide` excepted) at the primary condition gives the fewest wrong switches at delay budgets 4, 8 and 16. This tunes on test streams for every method alike. It checks whether frozen settings or thresholds hide a win, and does not count toward the rule.

## Prior work

Shared survey: [prior-work.md](prior-work.md#input-driven-retrieval-betteti-et-al-2025).

**Established names.** Input-driven dynamics: Betteti et al. `nearest` with a threshold: nearest-neighbour open-set rejection. `ema`: EWMA, a standard change-detection statistic. `hysteresis`: a Schmitt trigger on it, or switching with a margin. `hmm`: HMM forward filtering, in the two-state view a Shiryaev statistic **[read: Xie et al. survey, section II]**. The dynamics in overlap coordinates resemble competing accumulators with a leak ([Usher and McClelland, 2001](https://pubmed.ncbi.nlm.nih.gov/11488378/) **[abstract]**), though coupling here is multiplicative and is **not equated** with that model.

**What theory predicts [derived].**

- Delay against false alarms is a fundamental trade-off. Asymptotically the optimal delay is |log α|/D, with D the post-change divergence ([Xie et al., 2021](https://arxiv.org/abs/2104.04186) **[read: sections II–III]**), for known distributions. So no method wins on both axes by much on evidence alone. A real win has to come from better use of the same evidence, and on stationary independent frames the filter is the benchmark.
- A burst of b frames resembles a real change for its first b frames. No online method can tell them apart sooner, so wrong switches at short delay budgets are not fully avoidable.
- In the near-orthogonal limit, each overlap evolves as c ← c + dt(−c + α(t)·ψ(c)), an input-*multiplied* leaky state. Expected advantage over `ema`: an attractor holds through a brief drop in α. `hysteresis` is a much cheaper way to get the same hold, so **the contract's expectation is that `idp` does not beat `hysteresis`** at comparable cost on independent frames. The interesting departures would be on `held`, or `oracle`.
- Rejection through the origin needs all α small. With saturating Ψ, a state at the old memory persists while α_old ≥ about 0.2 at tanh(5x), which could make rejection lag. Our tests show switches of about 2 time units on random patterns at σ = 0–4, so delay ≤ 8 frames needs τ ≥ about 0.25 per frame. The published time step (0.01) would then cost 25 or more steps per frame.
- Correlated keys in 32 dimensions at P = 120 > D exceed the paper's orthogonal setting and any classical capacity reasoning, so `idp-key` may fail where `oracle` does not. `idp-wide` tests whether width repairs it; it cannot make keys less correlated than they are.

**Known failure modes and handling.** Smoothing lags and fails on correlated errors (`held` measures this). Attractor models can lock in or fail to switch (delay and miss rate measure this). Chosen thresholds may drift across encoders (settings are frozen on other seeds, and the frontier checks this). A multiple-comparison of many settings on small streams (one selection per method; confirmation on fresh seeds).

**What this adds.** A cost-matched test of a recent mechanism against cheap baselines, on real sensory keys it was not designed for, including unfamiliar input the paper does not treat. Otherwise nothing new.

## Scaling

Per frame, P keys of width D, dynamics width N, K steps; multiplications as counted by the code.

| method | work per frame | recurrent state | stored | what limits scale |
|---|---|---|---|---|
| `nearest` | PD | none | PD | linear in P |
| `ema`, `hysteresis` | PD + P | D (or P) | PD | linear in P; state does not grow with P |
| `hmm` | PD + 3P | P + 1 | PD | state grows with P |
| `idp-key` | (2 + 2K)PD + K(P + 2D) | D | PD | linear in P, multiplied by the steps |
| `idp-wide` | ND + (2 + 2K)PN + K(P + 2N) | N | PN + ND | the same, multiplied by N/D |

- **Facts.** All are linear in P. A sublinear index could replace the first read for any of them, and truncating W(u) to the k largest α would make a dynamics step O(kN). **[derived]**, not run. None of the state advantages favours `idp` over `ema`, since smoothing the query has the same D-wide state, which is the one scaling argument that might otherwise have favoured it.
- **Probe.** The selected settings, frozen at P = 120, run at P = 30 and 360 on the primary condition. Gains and thresholds tuned for one load may not survive another, and this measures that.
- **What would not survive scale-up.** One stored rendering per word, 28×28 glyphs, and a 460-symbol table are this experiment's scale. Frames here are independent or held; real sequences have richer correlation, which `held` only begins to test.

## Computation plan

- **Construction.** Encoders: pilot seeds exist; confirmation about 70 s each on the GTX 1650. Streams: rendering and encoding, CPU, under a minute per seed.
- **Selection.** Pilot grid: on the order of 500–600 settings per method-condition for `idp-key`, about 30 CPU-hours bound at smoke rates and full stream length, scored on 11 workers; measured and reported, rejected settings included.
- **Confirmation.** Selected settings on ten seeds; the frontier on four.
- **Deterministic and timed output kept apart.** `reports/streaming-retrieval-v1-{pilot,confirm,analysis}` and separate `-runtime` files with revision, versions, wall and worker CPU by phase, peak RSS, settings scored. `timing` measures one thread, a warm-up pass, 15 passes over 2,000 frames, median and spread, batch similarity included, encoder separately.
- **Persistence.** Tasks are durable (`experiments/durable.py`); rerunning the same command resumes. Streams and fits are written atomically.
- **Dtype.** float32. Keys and streams are not retained after the run other than in gitignored `runs/`.

## Reproduction

```
.venv/bin/python -m experiments.streaming_retrieval severity
NSY_WORKERS=11 .venv/bin/python -m experiments.streaming_retrieval pilot
NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.streaming_retrieval encoders
NSY_WORKERS=11 .venv/bin/python -m experiments.streaming_retrieval confirm
.venv/bin/python -m experiments.streaming_retrieval timing
.venv/bin/python -m experiments.streaming_retrieval analyse
```
Tests: `.venv/bin/python -m unittest tests.test_streaming_retrieval`.
