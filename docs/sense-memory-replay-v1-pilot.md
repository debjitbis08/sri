# Step 10, stage 2: latent replay from memory, read with memory removed

Status: **pilot, 2026-11-04, seeds 300–303, one order (A then B). Not a contract.** Design: [sense-memory-replay-v1-design.md](sense-memory-replay-v1-design.md), revised after external review before this run. Code: `experiments/sense_memory_replay.py`, with tables from `experiments/sense_memory_replay_summary.py`. Report: `reports/sense-memory-replay-v1-pilot.json.gz`.

Replaying stored frozen codes is rehearsal: **latent replay from memory**. `no-replay` never learns the associations, so acquisition here shows what replay teaches; it is not protection after equal teaching.

## Setup in brief

- **The core:** the step 8 calibrated core, trained exactly as limited-memory main. `no-replay` reproduces step 8's `joined-mlp-calibrated` in 1,540 of 1,540 measures.
- **Facts:** 45 new symbols per set. Set A is written during block 2 and set B during block 3: 32 writes per symbol, each a training rendering seen with a dot collection of its count.
- **Memory:** stage 1's kept-centre memory, with raw values and raw cue sums. It is built from the same writes at the same updates in every arm.
- **Replay:** 16 stored (symbol, collection) pairs per update, from the first write on, through the delay and relearning. They enter as a separate matching term with λ = 16 / (data matching items in the batch). There were 72–75 items per batch, so λ was 0.33–0.35.
- **Arms:**
  - `memory-replay`: units sampled uniformly.
  - `buffer-replay`: a reservoir of single feature pairs, uniform. Its fixed capacity was declared before the core ran: memory's final total bytes divided by 6,144 bytes per pair.
  - `memory-replay-true-count`: as `memory-replay`, with the slot set by the unit's majority count.
  - In both other replay arms, the core chooses the slot, with gradients stopped.
- **Reads with memory removed:**
  - **acquisition:** fresh renderings of each set's symbols through the core alone, against nine candidates per read (chance 1/9) or as comparison (chance 1/2);
  - **retention:** the step 8 snapshot measures.

Every cell is a mean over 4 seeds, with the range in brackets.

## Storage

| Seed | Writes | Memory units | Units per write | Memory, total | Buffer capacity |
|---|---|---|---|---|---|
| 300 | 2,880 | 852 | 0.30 | 10.49 MB | 1,706 pairs |
| 301 | 2,880 | 711 | 0.25 | 8.75 MB | 1,424 pairs |
| 302 | 2,880 | 853 | 0.30 | 10.50 MB | 1,708 pairs |
| 303 | 2,880 | 796 | 0.28 | 9.80 MB | 1,594 pairs |

- **Storage binds at 32 writes.** Memory needs 0.25–0.30 units per write, under the break-even of about 0.5, so the buffer at the same bytes keeps 49–59% of the writes.
- **At 8 writes it would not have bound.** The smoke run had 0.54 units per write, and stage 1 needed 4.1–4.5 units per fact.
- **Memory's total** is addresses, raw values, raw cues (25%), centres and gates.

## Acquisition, memory removed

Fresh renderings of each set's symbols through the core alone:

| Set, read | Arm | Block 2 | Block 3 | Delay | Relearn |
|---|---|---|---|---|---|
| A, taps | `no-replay` | 0.11 | 0.11 | 0.11 | 0.12 |
| | `memory-replay` | 0.32 (0.11–0.54) | 0.73 (0.66–0.80) | 0.76 (0.64–0.82) | 0.58 (0.51–0.67) |
| | `buffer-replay` | 0.44 (0.19–0.64) | 0.74 (0.62–0.84) | 0.79 (0.68–0.87) | 0.62 (0.45–0.72) |
| | `true-count` | 0.61 (0.55–0.67) | 0.72 (0.64–0.79) | 0.77 (0.59–0.90) | 0.60 (0.49–0.68) |
| B, taps | `no-replay` | | 0.12 | 0.11 | 0.10 |
| | `memory-replay` | | 0.72 (0.68–0.77) | 0.78 (0.72–0.82) | 0.51 (0.47–0.53) |
| | `buffer-replay` | | 0.64 (0.40–0.84) | 0.77 (0.59–0.88) | 0.51 (0.32–0.69) |
| | `true-count` | | 0.70 (0.66–0.80) | 0.78 (0.72–0.86) | 0.52 (0.38–0.68) |
| A, comparison | `no-replay` | 0.54 | 0.50 | 0.51 | 0.49 |
| | `memory-replay` | 0.82 (0.80–0.85) | 0.82 (0.78–0.85) | 0.75 (0.71–0.80) | 0.75 (0.68–0.86) |
| | `buffer-replay` | 0.81 (0.71–0.93) | 0.89 (0.84–0.94) | 0.85 (0.80–0.91) | 0.85 (0.83–0.90) |
| B, comparison | `no-replay` | | 0.47 | 0.49 | 0.47 |
| | `memory-replay` | | 0.64 (0.44–0.78) | 0.59 (0.38–0.72) | 0.59 (0.39–0.82) |
| | `buffer-replay` | | 0.77 (0.70–0.83) | 0.77 (0.69–0.83) | 0.78 (0.73–0.85) |

Dots follow taps to within 0.05 in every arm.

`memory-replay` minus `buffer-replay`, paired by seed:

| | Block 3 | Delay | Relearn |
|---|---|---|---|
| A, taps | −0.004 (−0.040 to +0.040) | −0.032 (−0.089 to +0.004) | −0.044 (−0.209 to +0.093) |
| B, taps | +0.081 (−0.067 to +0.271) | +0.013 (−0.098 to +0.138) | −0.001 (−0.222 to +0.213) |
| A, comparison | −0.067 (−0.151 to +0.013) | −0.102 (−0.191 to −0.004) | −0.100 (−0.218 to +0.027) |
| B, comparison | −0.127 (−0.253 to −0.036) | −0.184 (−0.360 to −0.107) | −0.188 (−0.342 to +0.004) |

## Retention, memory removed

Paired differences from `no-replay` on the primary measures. Only measures where any arm's mean moves by more than 0.05 are shown. Comparison on collections and taps stays at 1.00 in every arm, and taps → collections at 0.95–0.97 (all changes within ±0.03 on average).

| Measure | `no-replay` | `memory-replay` | `buffer-replay` | `true-count` |
|---|---|---|---|---|
| Block 3, Devanagari matching, withheld | 1.00 | −0.125 (−0.233 to −0.017) | −0.050 (−0.150 to 0.000) | −0.025 (−0.067 to 0.000) |
| Block 3, taps → Devanagari, withheld | 0.99 | +0.013 | −0.117 (−0.450 to 0.000) | −0.100 (−0.417 to +0.033) |
| Block 3, comparison, Chinese | 0.83 | +0.068 | +0.082 | +0.082 |
| Delay, Devanagari matching, withheld | 0.58 | −0.192 (−0.300 to −0.067) | −0.096 (−0.333 to +0.400) | −0.083 (−0.433 to +0.200) |
| Delay, taps → Devanagari, withheld | 0.62 | −0.192 (−0.383 to +0.100) | −0.108 (−0.467 to +0.067) | +0.033 (−0.117 to +0.100) |
| Relearn, Chinese matching, withheld | 0.71 | −0.283 (−0.567 to −0.067) | −0.154 (−0.500 to +0.083) | −0.188 (−0.483 to +0.150) |
| Relearn, Chinese → taps, withheld | 0.67 | −0.212 (−0.517 to +0.167) | −0.150 (−0.333 to +0.033) | −0.250 (−0.550 to +0.133) |
| Relearn, Devanagari matching, withheld | 0.89 | −0.129 (−0.450 to +0.033) | −0.087 (−0.250 to +0.117) | −0.083 (−0.150 to 0.000) |

## The teacher, mixed units and exposures

- **Accuracy of the slot the core chose, against the true count:**
  - 0.88–0.93 in blocks 2 and 3 and the delay, in both arms where the core chooses;
  - 0.69–0.71 at relearning, the moving teacher the review anticipated.
- **Replays from units holding more than one fact:** 290–838 of 99,184 per seed (0.3–0.8%). On those, the core's slot matches the majority count only 0.33–0.44 of the time.
- **Exposures:**
  - Uniform over units replays some counts twice as often as others (shares 0.07–0.14 per count), and some symbols far more than others (per-fact coefficient of variation 0.86).
  - The buffer is near uniform per count (0.11–0.12), and its per-fact variation is 0.47, from the order of the writes.
- **With memory present** (stage 1's end-to-end read), every arm is within about 0.05 of `no-replay` at block 3, where taps on set A read 0.70–0.79. Replay does not change how the core interprets memory beyond that.

## What this shows

- **Replay teaches the core associations it never learned otherwise.** With memory removed, the core alone answers a new symbol's count as taps at 0.72–0.79 after replay, against 0.11 without. Comparison between new symbols rises from 0.50 to 0.6–0.9. This is acquisition through rehearsal, as the design stated.
- **What the core learns is partly lost at relearning, even with replay running.** Taps fall from 0.76–0.79 at the delay to 0.51–0.62 at relearning. The teacher also degrades there (slot accuracy 0.70), but the true-count arm falls just as far (0.77 to 0.60). So the loss comes from relearning itself, not from wrong slots alone.
- **Memory replay is not better than a plain buffer at the same bytes, and is worse on comparison.**
  - On taps and dots the two are within about 0.05 on average, with wide seed ranges.
  - On comparison between new symbols, memory replay is 0.07–0.19 lower: worse for set B at every point, and for set A at the delay.
  - Memory stores the experience in a quarter of the units, but at the bytes the buffer is allowed, the buffer still keeps half of all writes, enough to match it.
  - The uneven exposures of uniform-over-units sampling are a candidate cause of the comparison deficit. That is not established.
- **Self-labelling costs little where the core reads well.** `memory-replay` and `true-count` differ by at most 0.05 on taps and dots on average, and by 0.04–0.08 on comparison.
- **Replay costs retention of withheld script transfers.**
  - Every replay arm loses ground on withheld Devanagari and Chinese matching and transfer at the delay and at relearning, by 0.08–0.28 on average, with wide ranges.
  - Memory replay loses most: Devanagari withheld −0.125 already at block 3, and Chinese withheld −0.283 at relearning.
  - The 0.05 retention bar is not met by any replay arm on these measures.
  - The core's reading of collections, taps and their comparison is unaffected. Taps → collections stays at 0.95–0.97.
- **Taps retention was a target, not a guarantee.** In this run, `no-replay` itself keeps taps → collections at 0.95 at relearning, so there was no taps loss for replay to repair on this measure.

## What this does not establish

- **Whether memory adds anything beyond retained features.** It did not here. Settings where storage binds harder (more writes, or a smaller budget) or sampling that evens out exposures might change that, and were not run.
- **Why replay costs withheld transfers.** A replay loss weight of λ ≈ 0.33 is one candidate, and capacity competition between new symbols and scripts is another.
- **Both orders, and significance:** one order, four pilot seeds.
- **Similarity-selected replay** was left out by design.

## Prior work

- **Latent replay with a fixed extractor** (Pellegrini et al. 2020) and **tiny episodic memories** (Chaudhry et al. 2019) predict a plain feature buffer to be a strong baseline. It was here, at equal bytes.
- **Pseudo-labels** (Lee 2013): errors are small while the labeller is good, and grow as it degrades. Seen here at relearning.
- **Complementary learning systems** predict that interleaved replay protects what is replayed. It did, for the replayed associations through the delay. They do not predict protection of untouched skills, and none was seen.

## Scaling

- **Per update:** 16 replay pairs, 32 extra item passes, and one 16 × 9 slot evaluation. This is independent of the number of facts.
- **Memory:** 711–853 units for 2,880 writes, 8.8–10.5 MB including raw cues. Units grow with distinct experience.
- **Buffer:** a fixed capacity, the same bytes.
- **No input is stored or re-encoded.**

## Computation

- **Prepare** (GTX 1650, frozen encoding of 2,880 training and 675 query renderings per seed, passing runs and gate views, plus the full memory once): 228–231 s per seed, 15 min 53 s wall for four seeds, peak RSS 2.8 GB.
- **Core runs** (three workers, GPU):
  - 16 runs, 209–298 s wall each: `no-replay` 268–278 s, replay arms 209–298 s;
  - 27 min 5 s wall, 9,991 s aggregate worker CPU, peak RSS 3.7 GB, GPU peak 353 MB per worker;
  - 99,184 replayed pairs per replay run, 198,368 extra item passes;
  - zero image cache misses.

Seed 303's `true-count` run was the shortest, at 209 s.
