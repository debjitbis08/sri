# What a learning encoder's movement is made of: measurement (seeds 392, 393, 404, 405)

Status: **exploratory measurement, 2026-11-02. Not a contract.** Images here only measure; nothing below is a memory mechanism. The families, networks and thresholds of interest were chosen for this study, with settings left over from the [image-free pilot](sense-image-free-v1-pilot.md).

## Question

Before building a calm space for memory, how is a code's movement divided between jitter, a predictable change of coordinates and the rest? And is that an effect of this network and stream, or of the images used to measure it?

## Design

A code's path is sampled every 250 updates over three blocks of 2,000. Four axes vary:

- **Input family, 15:** background glyphs, cue symbols, unseen symbols, target symbols, other fonts, dots, tally, tree, banana, Thai digits, MNIST, dice, natural images, white noise, coarse noise. Up to 512 images each.
- **Network, 3:** the unlabelled glyph key encoder; the host, from saved weights, with a different architecture and objective; the same key architecture trained from scratch on natural images for 8,000 updates.
- **Stream:** stationary, shift to dots and formats, or shift to natural images from block 5. Host streams are its own pair teaching.
- **Seed and rate:** normal and 10×. 392 and 393, plus fresh seeds 404 and 405 for the from-scratch encoder, which reuse 392 and 393's measuring images.

Natural images are the first 5,123 of CIFAR-10, the part fetched before the slow download was stopped, grey and centre-cropped to 28 × 28, with the last 1,000 kept out of training.

**Measures,** per block:

- **Movement structure:** S(τ) = E‖z(t+τ) − z(t)‖² fitted non-negatively as a + bτ + cτ². a is jitter, which does not accumulate. Over only eight lags, b and c trade against each other, so "accumulating" is the reliable reading, not diffusion versus trend.
- **Map:** the share of a block's movement that a linear map predicts, on inputs it did not fit, within and across families. Negative means it adds movement.
- **Slow directions:** C_move w = λ C w, fitted on one family and read on others, and whether they keep identity. This fit sees the whole block, so it is retrospective, not a mechanism.
- **Parameter averages:** β 0.99 and 0.999 of the key encoders, measured the same way.

## Results

**Jitter and a linear map overlap.** They are not parts of a whole. Within a family, jitter and map shares together often exceed 1, so much of the wobble is shared across inputs and is linear. So the planned three-way split is not a partition, and no "remainder" is reported.

Medians across families, block 5 unless stated:

| Network and stream | Rate | Movement (symbols, far families) | Jitter | Net over path | Map, same family | Map, symbols to other symbols | Map, far family to symbols |
|---|---|---|---|---|---|---|---|
| Glyph, stationary | normal | 0.36–0.40 | 0.68 | 0.16 | 0.60 | 0.29 | −0.90 |
| Glyph, stationary | 10× | 0.56–0.57 | 0.67 | 0.17 | 0.69 | 0.40 | −0.62 |
| Glyph, shift | normal | 0.88–0.90 | 0.16 | 0.34 | 0.78 | 0.61 | −0.14 |
| Glyph, shift | 10× | 1.01–1.02 | 0.21 | 0.29 | 0.79 | 0.66 | −0.29 |
| Glyph, natural | normal | 1.14–1.18 | 0.14 | 0.39 | 0.93 | 0.82 | 0.46 |
| Glyph, natural | 10× | 1.20–1.25 | 0.16 | 0.35 | 0.94 | 0.87 | 0.58 |
| From scratch, natural | normal | 0.38–0.39 | 0.42 | 0.19 | 0.87 | 0.86 | 0.69 |
| From scratch, natural | 10× | 0.52–0.53 | 0.61 | 0.16 | 0.90 | 0.88 | 0.72 |
| Host | own teaching | 0.57–0.68 | 0.22 | 0.38 | 0.55 | 0.40 | 0.04 |

Jitter at normal rate, block 5, by family and by run. Family range is the mean per family; run values are medians across families:

| Condition | Across families | Across runs |
|---|---|---|
| Glyph, stationary | 0.39–0.73 | 0.53, 0.82 |
| Glyph, shift | 0.11–0.24 | 0.16, 0.17 |
| Glyph, natural | 0.10–0.16 | 0.14, 0.14 |
| From scratch | 0.41–0.71 | 0.59, 0.27, 0.43, 0.39 |
| Host | 0.03–0.37 | 0.18, 0.21, 0.19, 0.23 |

**Maps do not carry to far families.** On the glyph encoder, stationary, normal rate, a map fitted on natural images reads cue symbols at −1.08 to −1.70, and one fitted on white noise at −3.48 to −7.28, so both add movement. Fitted on background glyphs, it explains 0.21–0.33 on cues. After a real change it does better: 0.52 on a shift, 0.72–0.81 on a shift to natural images. From scratch, background reads cues at 0.79–0.86, while natural images, its own training kind, read cues at only 0.09–0.23.

**Averaged endpoints do not rescue maps.** With jitter reduced by averaging neighbouring snapshots, same-family reads are 0.48–0.60 and symbols to symbols 0.17–0.32, stationary, normal rate. So the accumulating part is not more linear than the whole.

**Slow directions exist and transfer, and keep less than they seem to.**

| Network and stream (normal rate, block 5) | Their λ: slowest 8, own family | Other symbols | Far families | Stale read | Stale, slow 8 | Stale, slow 16 | Fresh read |
|---|---|---|---|---|---|---|---|
| Glyph, stationary | 0.05 | 0.05–0.06 | 0.11 | 0.87 | 0.59–0.66 | 0.80–0.84 | 0.88 |
| Glyph, shift | 0.20–0.21 | 0.25–0.29 | 0.59–0.60 | 0.37 | 0.33–0.42 | 0.54–0.61 | 0.76 |
| Glyph, natural | 0.31–0.38 | 0.38–0.46 | 0.71–0.76 | 0.10 | 0.23–0.28 | 0.36–0.43 | 0.60 |
| From scratch | 0.05 | 0.05 | 0.08 | 0.26 | 0.27 | 0.34 | 0.39 |
| Host | 0.24 | 0.25–0.27 | 0.20–0.22 | 0.03 | 0.11 | (0.03, no slow subspace left) | 0.23 |

Reads are nearest neighbour among 90 symbols, cue at the block's start against query at its end. The range covers fitting on background glyphs and on cues. In a stationary stream the subspace loses identity and gains nothing. After a change, it recovers a good part. Its fit saw the change, so whether directions chosen in advance would hold is not known. On the host, fresh MNIST reads 0.29 and slow-8 from stale 0.25–0.26, against 0.20 stale.

**Parameter averages are the most general result.** β 0.999, cue symbols, nearest neighbour, one block, fresh and stale:

| Network and stream | Rate | Block | Working: movement, fresh, stale | Average: movement, fresh, stale |
|---|---|---|---|---|
| Glyph, stationary | normal | 5 | 0.40, 0.88, 0.87 | 0.20, 0.89, 0.89 |
| Glyph, stationary | 10× | 5 | 0.57, 0.86, 0.81 | 0.31, 0.89, 0.88 |
| Glyph, shift | normal | 5 | 0.90, 0.76, 0.37 | 0.84, 0.80, 0.51 |
| Glyph, shift | 10× | 5 | 1.00, 0.71, 0.22 | 0.89, 0.74, 0.37 |
| Glyph, natural | normal | 5 | 1.18, 0.60, 0.10 | 1.16, 0.69, 0.14 |
| From scratch | normal | 5 | 0.39, 0.39, 0.26 | 0.25, 0.38, 0.33 |
| From scratch | 10× | 5 | 0.53, 0.37, 0.12 | 0.40, 0.39, 0.24 |

Per seed, glyph, block 5, stale at normal and 10×: 392, working 0.88, 0.79 and average 0.92, 0.88; 393, working 0.86, 0.82 and average 0.85, 0.88. With a shift: 392, 0.33, 0.19 and 0.49, 0.31; 393, 0.41, 0.26 and 0.54, 0.43.

An average's jitter is about 0 and its net over path rises from about 0.16 to about 0.6. It halves movement on a stationary stream and reads as well as or better than the working encoder, so it costs nothing visible here. **It removes the wobble, not a real change.** Under a shift, movement barely falls, though stale reads gain 0.14–0.16 on formats and little on natural images. It was not measured on the host, where jitter is small. β 0.99 removes about half of the jitter and less movement.

## How general is this?

| Finding | Across input families | Across networks | Across streams | Across seeds |
|---|---|---|---|---|
| Key movement is mostly non-accumulating on a stationary stream | Yes, broadly (0.39–0.73) | From scratch somewhat lower; **not the host** (0.22) | **No:** 0.10–0.24 under a change | Wide (0.27–0.82) |
| A real change is accumulating movement | Yes (0.10–0.24) | Two streams on one network | By definition, once | Tight |
| Linear map, same kind of input | Works, medians 0.52–0.95 | Varies: higher from scratch | Higher after a change | Not separated |
| Linear map, far inputs | **Fails, often negative:** specific to the inputs fitted | From scratch carries further, still not from natural images or noise | Better after a change | Not separated |
| Slow directions read on other families | Yes, about 2× on far families | Yes | Weaker after a change | Not separated |
| Slow directions keep identity | Measured on symbols and MNIST only | Differs | Only useful after a change | Not separated |
| Parameter average lowers movement and stale error | Measured on cues and MNIST; structure measured on all | Both key encoders; **not the host** | Stationary yes; real change little | Consistent in two seeds, and fresh seeds in structure |

So a measure is specific to the images when it relies on them for its fit. Maps are the clear case. Structure measured on a family reflects the network and stream more than the family, but the family range is as wide as the seed range, so single figures are not precise.

**Still specific:** every image is 28 × 28 and grey. Natural images are a small part of one dataset. Two of three networks share a small architecture, and none are deep. A shift is one change, at one time. Reads depend on the task: from scratch reads symbols at only 0.38.

## What this suggests, not establishes

1. **For a learning key encoder on a stable input kind, averaging is the cheap calm copy.** It needs one extra network and removes the wobble that likely confounded address maps.
2. **It does not answer real change.** Neither do linear maps from far inputs. A retrospective slow subspace recovers some, so protection or rehearsal remains the question there.
3. **The host needs its own answer.** Its movement is mostly accumulating, and value maps from recurring inputs fit that.

## Prior work

Slow feature analysis (Wiskott and Sejnowski 2002), parameter averaging (Polyak and Juditsky 1992), momentum encoders (He et al. 2020), neural tangent kernels (Jacot et al. 2018), self-healing codes (Rule and O'Leary 2022) **[familiar; read before use]**. None was read for this measurement. Reading them remains owed before any contract.

## Computation

GTX 1650, three workers, 707 s wall, 3,484 s aggregate CPU, 24 tasks, GPU peak 88 MB, peak RSS 2.6 GB per process. A task's measures cost O(F² d³) per block for F families. Fetching the natural images took about 15 minutes of a slow connection.
