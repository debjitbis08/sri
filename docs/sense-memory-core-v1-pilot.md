# Step 10, stage 1: memory on frozen sight supplies facts to a changing core

Status: **pilot, 2026-11-04, seeds 300–303 (the step 8 seeds), both orders. Not a contract.** Design: [sense-memory-core-v1-design.md](sense-memory-core-v1-design.md), revised before this run after the user's decisions and an external review. Code: `experiments/sense_memory_core.py`, with tables from `experiments/sense_memory_core_summary.py`. Report: `reports/sense-memory-core-v1-pilot.json.gz`.

**The claim is narrow:** memory on frozen sight preserves arbitrary symbol-to-collection associations that the current core can interpret, across senses and through a changing core. It says nothing about general retention of arbitrary sensory values, and nothing about training through memory: memory never changes the core here.

## Setup in brief

- **Facts:** 45 new symbols per set (A and B, disjoint, balanced over counts 1–9). Each symbol is written 8 times, as a training rendering seen with a different dot collection of its count.
- **Memory:** kept-centre memory with a frozen gate per centre, addressed by the symbol's frozen DINOv2-S + TrOCR-small code. It stores the collection's **raw frozen code** as its value.
- **Centres:** opened by the windowed shift rule on runs of other pool symbols. One centre opened in every run, and the second run opened none.
- **Queries:** 5 unwritten evaluation renderings per symbol.
- **Hosts:** the step 8 checkpoints. A recalled value goes through the current calibrated projection and core.
- **Schedule:** the first set is written after block 1, the second after block 3, in both orders.
- **Thresholds:** τ is 0.85, as in the stability pilot. A secondary read abstains only below 0.70; it was added after the seed-300 smoke run.

Every cell is a mean over 4 seeds × 2 orders, with the range in brackets. Chance is 1/9 for dots, taps and taps → symbol, and 1/2 for comparison. Abstentions count as wrong.

## Results

**At block 3, both sets written, first set read** (the second set reads the same to within 0.01, except for copies):

| Arm | Dots | Taps (across senses) | Comparison | Taps → symbol |
|---|---|---|---|---|
| `trace`, end to end, τ 0.85 | 0.65 (0.59–0.72) | 0.66 (0.56–0.71) | 0.50 (0.39–0.61) | 0.66 (0.53–0.76) |
| `trace`, end to end, abstaining below 0.70 | 0.90 (0.81–0.96) | 0.92 (0.88–0.96) | 0.96 (0.92–1.00) | 0.94 (0.89–1.00) |
| `trace`, queries retrieved correctly at τ 0.85 | 0.93 (0.85–0.97) | 0.94 (0.92–0.97) | 0.70 (0.63–0.79) | |
| `trace`, on the fact's own stored value (`bypass`) | 0.94 (0.87–0.98) | 0.95 (0.90–1.00) | 1.00 | 0.98 (0.93–1.00) |
| `trace-single`, `bypass` (one actual collection per unit) | 0.94 (0.90–0.96) | 0.95 (0.93–0.98) | 1.00 (0.99–1.00) | 0.97 (0.91–1.00) |
| `copy` (the core's code at write time, block 1) | 0.14 (0.11–0.20) | 0.12 (0.06–0.18) | 0.48 (0.39–0.60) | 0.24 (0.13–0.33) |
| `core-only` (the symbol through the core, no memory) | 0.12 (0.08–0.16) | 0.11 (0.07–0.14) | 0.49 (0.36–0.59) | 0.09 (0.02–0.18) |
| `fresh-dots` (a new collection of the right count) | 0.90 (0.86–0.93) | 0.91 (0.88–0.95) | 1.00 | 0.95 (0.89–1.00) |
| `copy-existing` (the core's own encoder) | 0.09 (0.07–0.12) | 0.08 (0.02–0.12) | 0.46 (0.36–0.53) | 0.21 (0.11–0.31) |
| `fresh-dots`, `existing` host | 0.39 (0.33–0.50) | 0.39 (0.32–0.46) | 0.96 (0.94–0.97) | 0.66 (0.58–0.78) |

**Memory alone:**
- **Retrieval:** right fact retrieved and answered, 0.69 (0.60–0.76) at τ 0.85 and 0.96 (0.94–0.99) at 0.70. Answered at all, 0.70 and 0.99.
- **Unseen symbols answered:** 0.02–0.03 at τ 0.85, and 0.13 (`first`) or 0.20 (`both`) at 0.70.
- **Sight alone:** matching the recalled value to the dot candidates by frozen cosine gives 0.31 (0.27–0.36). Sight alone cannot answer taps at all.

**Host change.** The first set on `first`, read through each checkpoint (dots and taps; comparison stays at 1.00 on `bypass`):

| Arm | Block 1 | Block 3 | Delay | Relearn |
|---|---|---|---|---|
| `trace`, `bypass`, dots | 0.71 (0.49–0.88) | 0.94 (0.88–0.98) | 0.96 (0.92–0.97) | 0.84 (0.76–0.97) |
| `trace`, `bypass`, taps | 0.67 (0.43–0.88) | 0.96 (0.90–1.00) | 0.93 (0.88–1.00) | 0.72 (0.51–0.91) |
| `fresh-dots`, dots | 0.68 (0.56–0.80) | 0.90 (0.86–0.93) | 0.91 (0.89–0.94) | 0.78 (0.70–0.87) |
| `fresh-dots`, taps | 0.65 (0.48–0.84) | 0.91 (0.88–0.95) | 0.89 (0.85–0.94) | 0.66 (0.50–0.80) |
| `copy`, dots | 0.49 (0.36–0.59) | 0.14 (0.11–0.20) | 0.11 (0.07–0.16) | 0.33 (0.14–0.67) |
| `copy`, taps | 0.46 (0.32–0.62) | 0.12 (0.06–0.18) | 0.11 (0.06–0.15) | 0.28 (0.12–0.60) |

**Interference.** The first set on `first` against `both`, through the same checkpoint, paired. At every checkpoint the mean change is at most −0.004 on retrieval, dots, taps and comparison, and the worst single run is −0.027 (retrieval). Both orders agree.

**Answer dependence on the interpreter.** For the same stored value and the same candidates, the share of answers that change:

| Change of checkpoint | Dots | Taps |
|---|---|---|
| Block 1 → block 3 | 0.28 (0.11–0.49) | 0.34 (0.15–0.57) |
| Block 3 → delay | 0.06 (0.02–0.11) | 0.06 (0.00–0.10) |
| Block 3 → relearn | 0.16 (0.04–0.27) | 0.25 (0.07–0.44) |

**Side effects.** Every host's parameters and buffers hash identically before and after all memory reads, at every checkpoint and seed.

## What this shows

- **Memory and core together answer what neither answers alone.** Through the core, a recalled dot value answers taps, a different sense, at 0.95 on the correct stored value and 0.94 on correctly retrieved queries. Without memory the core is at chance (0.11). Without the core, sight cannot answer taps at all, and it reads dots at only 0.31.
- **Integration costs nothing beyond retrieval.**
  - Reads on correctly retrieved queries (0.93–0.94) are within 0.02 of reads on the correct stored value (0.94–0.95).
  - Those reads are at or above the core's reading of a fresh collection (0.90–0.91).
  - The paired bars in the design are met.
- **Averaging does not hurt.** Averaged values read like one actual collection (0.94–0.95 against 0.94–0.95 on `bypass`), within the 0.05 bar.
- **Retrieval at τ 0.85 is the limit end to end.**
  - 30% of queries abstain, which caps dots and taps at about 0.65 and comparison, which needs two answers, at about 0.50.
  - Abstaining only below 0.70 lifts end-to-end reads to 0.90–0.96, at the cost of answering 0.13–0.20 of unseen symbols, against 0.02–0.03.
  - This is the familiar threshold trade-off of earlier pilots. The fresh renderings differ from the written ones, and τ 0.85 is strict for them.
- **Stored sensory traces follow the interpreter; copies go stale.**
  - Copies coded by the block-1 core read 0.46–0.49 through it, but 0.11–0.14 through the block-3 core.
  - The traces rise from 0.67–0.71 at block 1 to 0.94–0.96 as the core learns to read dots better. This is the [traces pilot](shared-core-sparse-memory-v1-traces-pilot.md)'s finding, now with no image stored.
  - The core's own encoder gives no usable copy (0.08–0.09).
- **The moving-host bar fails, and the core is the cause.** From block 3 to relearning, taps on the stored value fall from 0.96 to 0.72, against the bar of 0.05. A fresh collection falls just as far through the same core (0.91 to 0.66), so the relearning core reads dots and taps worse, whatever their source. Fixed values do not give fixed answers: a quarter of taps answers change between block 3 and relearning on identical inputs. Memory keeps the experience, and what it means depends on the core that reads it.
- **Interference is negligible here** (at most −0.004 on average), in both orders.
- **R6:** reads have no side effects. As designed, this says nothing about whether memory spares the core when it trains it.

## What this does not establish

- **Training through memory** (stage 2, latent replay from memory, which is rehearsal), or whether memory spares or improves the core.
- **Values beyond dots,** which the current core keeps reading. Forms of a taught script would be forgotten by the core, and taps are not stored.
- **A threshold rule.** The secondary threshold was added after the smoke run, and neither threshold was selected on these seeds.
- **Significance:** four pilot seeds and two orders.
- **One kind of symbol, one centre.** The second passing run opened no centre, so growth during writes was not exercised.

## Prior work

The design's prior-work section applies:
- **Hippocampal indexing and complementary learning systems.** An index recalls an experience, and the cortex interprets it. The answer dependence measured here is the expected consequence of a changing cortex.
- **Retrieval-augmented networks with frozen keys and values and a learning reader** (kNN-LM, Memorizing Transformers, RETRO). Stale copies are their known failure when stored codes come from a moving encoder. Here the stored codes come from frozen models, and only the reader moves.

## Scaling

- **Write:** one pass per frozen model per experience, plus O(k d) to code the cue on k centres.
- **Read:** O(k d + h d) to retrieve among h units, then one pass through the projection and core per recalled value.
- **State:**
  - 9,224 bytes per unit (address sum, address, raw value sum, count), with 4.1–4.5 units per fact at 8 writes and τ 0.85;
  - 3.4–3.7 MB occupied for 90 facts, against 18.9 MB allocated for one centre of 2,048 units;
  - `trace-single`, a diagnostic, adds 1.1–1.2 MB.
- **Host changes cost memory nothing,** because no stored input is ever re-encoded.

## Computation

- **Memory stage** (GTX 1650): 152–156 s wall per seed, 147–151 s of it frozen encoding of symbol renderings, passing runs and gate views. Four seeds ran in sequence, 10 min 52 s wall in total, peak RSS 2.8 GB.
- **Read stage** (CPU, four workers): 50 s wall, 84 s aggregate worker CPU, peak RSS 2.4 GB.
- **Latency:**
  - batch-one retrieval, 0.56–0.60 ms;
  - projection and core per recalled value, in batches of 225, 4–5 µs.
- **Gate refits:** one per seed, at 300 steps.

No model was trained.
