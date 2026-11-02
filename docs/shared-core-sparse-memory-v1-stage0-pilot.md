# Shared-core sparse memory v1: stage 0 pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, used to lock the route. Pairs below are seed 392, seed 393.
[Contract](shared-core-sparse-memory-v1-contract.md); runner `experiments/shared_core_stage0.py`; reports `reports/shared-core-sparse-memory-v1-stage0-*.json`.

## Result

**No arm addresses, so the lock table gives the `conjunctive` route: host `sparse48`, sensory source the frozen encoder, stage 0 candidate `split-recon-bg` at λ 0.1.** The best
configuration read 0.25 on 90-way identity against a bar of 0.80 and the frozen encoder's 1.00. `sparse48`'s encoder output before projection also fails the bar, at 0.26, 0.27.

The gate passed: `sparse48` through this runner has the same block-3 state hash as the long-term memory host's cache, on both seeds.

| Configuration | Identity, 90 | Store, 90 | Before projection, identity | `transfer8` mean against `sparse48`, pilot mean | MSE, new symbols |
|---|---|---|---|---|---|
| `sparse48` | 0.15, 0.19 | 0.25, 0.29 | 0.26, 0.27 | reference: 0.46, 0.40 | |
| `wide32` | 0.23, 0.24 | 0.31, 0.30 | 0.34, 0.34 | −0.03 | |
| `wide32-recon`, λ 0.1 | 0.12, 0.19 | 0.20, 0.27 | 0.20, 0.33 | −0.02 | 0.038, 0.042 |
| `wide32-recon`, λ 1 | 0.18, 0.22 | 0.26, 0.31 | 0.30, 0.36 | −0.04 | 0.042, 0.040 |
| `wide32-recon-bg`, λ 0.1 | 0.14, 0.20 | 0.23, 0.28 | 0.21, 0.28 | −0.04 | 0.039, 0.040 |
| `wide32-recon-bg`, λ 1 | 0.18, 0.20 | 0.27, 0.29 | 0.24, 0.32 | +0.03 | 0.039, 0.035 |
| `split-recon-bg`, λ 0.1 | 0.25, 0.25 | 0.34, 0.33 | 0.34, 0.30 | −0.02 | 0.033, 0.034 |
| `split-recon-bg`, λ 1 | 0.24, 0.24 | 0.32, 0.34 | 0.30, 0.31 | −0.00 | 0.034, 0.035 |
| Frozen encoder, reference | 1.00, 1.00 | 1.00, 1.00 | | | |

Chance is 0.011 for identity and 0.11 for the store. Predicting the mean image gives an MSE of 0.043. Every configuration passes the transfer screen, but a difference on this mean has a standard error of about
0.04–0.06, so these differences are not read as effects. Mean-centring changes identity by at most 0.04.

## What limits identity

**This is not a test of strong reconstruction.** At λ ≤ 1 the term barely trained, reaching 0.033–0.042 against the mean image's 0.043. It is a per-pixel mean, small beside the relational losses.

**Exploratory, chosen after seeing the above, not in the lock table.** The background arms at larger λ:

| Configuration | Identity, 90 | Store, 90 | Before projection | `transfer8` against `sparse48` | MSE, new symbols | Value ceiling, head |
|---|---|---|---|---|---|---|
| `wide32-recon-bg`, λ 10 | 0.18, 0.21 | 0.29, 0.30 | 0.24, 0.37 | +0.01, −0.12 | 0.031, 0.030 | 0.47, 0.77 |
| `wide32-recon-bg`, λ 100 | 0.26, 0.32 | 0.32, 0.40 | 0.38, 0.35 | −0.06, −0.08 | 0.018, 0.019 | 0.49, 0.45 |
| `split-recon-bg`, λ 10 | 0.19, 0.27 | 0.29, 0.37 | 0.25, 0.34 | −0.01, −0.00 | 0.031, 0.027 | 0.43, 0.61 |
| `split-recon-bg`, λ 100 | 0.28, 0.30 | 0.36, 0.37 | 0.36, 0.38 | +0.08, −0.15 | 0.018, 0.019 | 0.50, 0.27 |

Reconstruction now trains, and at λ 100 identity is 0.26–0.32, far below the bar. At λ 100 the value ceiling on cosine falls to 0.26–0.32. So in this range more reconstruction buys little identity and costs value readability.

Other readings:

- **Still a line.** On new symbols the top direction holds 0.95–0.99 of the variance in `sparse48` and the `wide32` arms at λ ≤ 1, against 0.14–0.15 on the frozen encoder. Its cosine with the collections' top direction is 0.95–1.00 at λ ≤ 1, so it is the count direction. The widest of these is `split-recon-bg` at λ 0.1, seed 392, at 0.77; at λ 100 the share is still 0.69–0.95.
- **The encoder output is weak as well as the projection.** Cosine identity before projection never exceeds 0.38 in any configuration. A learned projection can change the metric and suppress nuisance directions, so this does not show identity is absent there. It shows that the tested representations and readers fail.
- **The added part of `split-recon-bg`** reads 0.34, 0.29 at λ 0.1, above its read part at 0.20, 0.14, but holds 3–4% of the code's variance, so the whole reads 0.25.
- **Readers do not spread.** At λ ≤ 1, live units of both readers together use 4–9 of 32 coordinates in the `wide32` arms, and 3–6 of 16 in `sparse48` and `split-recon-bg`.
- **Pixels are a poor identity cue.** In the key encoder's selection, raw pixel cosine read 0.09 on 120 validation symbols, from one stored rendering. This is the contract's stated failure mode for pixel reconstruction.

Expectations recorded before the run: `wide32` alone did not reach 0.80, as expected. That background reconstruction helps identity most is not supported, since `wide32` without any reconstruction read 0.23, 0.24. That `split-recon-bg` holds relations best cannot be read at this noise.

## Consequences for stage 1

- **Value ceiling, `sparse48`.** Matching head 0.52, 0.61, mean 0.56, below the decoder floor of 0.60. Cosine 0.60, 0.62, mean 0.61. Count prototypes 0.61, 0.66. By the contract's rule the head is `decoder-limited`, so the draft now decodes by cosine. This clears the floor by 0.01 on two seeds, so it is not a comfortable margin, and `acquires`' floor of 0.50 leaves little room below it.
- **Frozen encoder has no taps.** With a zero sensory part, a tap query's similarity is at most half the core's, so below every τ in the grid, and `conjunctive` could not answer a tap at all. The draft now reads the core part alone when a query has no sensory code.
- **Two addresses coincide.** On this route `sense-only` and `frozen-encoder` are the same address, so one run serves both.
- **`perception` is not needed on this route.**

## Computation

Measured on the GTX 1650, three workers, one thread each, float32. Stage 0 training is construction cost, on the GPU, and says nothing about CPU viability.

| Phase | Wall | Aggregate worker CPU | Scope |
|---|---|---|---|
| Host fits, 16 | 1,020 s | 2,704 s | all configurations, all selection cost; 113 s per fit for `sparse48`, up to 201 s with background |
| Readings, 18 | 72 s | 52 s | wall includes 51 s of data preparation and rendering in the parent, whose CPU is not in the aggregate |
| Exploratory, 8 fits and readings | 608 s | 1,553 s | not selection for the lock, charged separately |

Each fit is 6,000 updates and reads about 1.2 million curriculum images. Background arms add 384,000 background images, from 1,987 symbols at 32 renderings. Host parameters: 371,114 for `sparse48`, 377,306 for `wide32`, 373,194 for `split-recon-bg`. Decoder: 211,265 parameters, training only. GPU peak allocated: 84 MB without a decoder, 126 MB with. Peak RSS per worker about 2.4–2.6 GB, counting pages shared after the fork. Timing is in the `-runtime.json` files.
