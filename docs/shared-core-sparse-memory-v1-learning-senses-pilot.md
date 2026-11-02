# Shared-core sparse memory v1: learning senses pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, A then B only; H 512 and 2048 were added after H 128 filled, so they are post hoc. Runner `experiments/shared_core_learning_senses.py`; reports `reports/shared-core-sparse-memory-v1-learning-senses*.json`.

## Question and setup

Addresses have so far rested on a frozen encoder. If the sensory encoder keeps learning, do addresses go stale the way host values did, and can stored cue traces rebuild them?

The encoder continues its own labelled training on background symbols for 6,000 updates, 2,000 per block, with no host involved: A is taught in the first, B in the second, nothing in the third. Each write uses the encoder as it is at that update. Values and reading stay on block-3 host codes, so **every difference here comes from addresses**. This separates drift from the harder question of identity without labels, which is not tested.

Rebuilding: every so often, each unit's address becomes the mean direction of its kept cue traces under the current encoder. Arms keep 1 trace, 8 traces or all, every 250 or 2,000 updates. τ stays at 0.85, the setting locked on stable keys.

| Learning rate | Same image's cosine with its start, end of block 6 | Identity, 90 symbols | Same-symbol, other-symbol similarity |
|---|---|---|---|
| 0.001, as in the encoder's own training | 0.87, 0.89 | 1.00, 0.99 | 0.96, 0.07 |
| 0.01, for stronger drift | 0.69, 0.73 | 1.00, 1.00 | 0.96, 0.05–0.08 |

So the code really moves, and identity is unharmed. For scale, the host's count prototypes end at about 0.88–0.91 with block 3.

## Result

**With room to allocate, rebuilding keeps A and B as well as a frozen encoder, and A does not fall across B. Without rebuilding, A decays even with no writes. The new cost is unit growth, and at H 128, the locked setting, memory fills and B fails.**

Memory level, nearest training-collection prototype, end of block 6, abstentions wrong:

| Arm | Rate 0.001, H 512: A | B | Units | Rate 0.01, H 2048: A | B | Units |
|---|---|---|---|---|---|---|
| Frozen encoder | 0.96, 0.90 | 0.88, 0.91 | 108, 149 | 0.96, 0.90 | 0.88, 0.91 | 108, 149 |
| No rebuild | 0.88, 0.82 | 0.93, 0.94 | 202, 301 | **0.46, 0.56** | 0.85, 0.91 | 915, 1,160 |
| One trace, every 250 | 0.96, 0.92 | 0.97, 0.94 | 384, 512 | 0.96, 0.91 | 0.91, 0.93 | 1,064, 1,386 |
| Eight traces, every 2,000 | **0.96, 0.93** | **0.97, 0.93** | 202, 303 | **0.97, 0.94** | **0.95, 0.97** | 899, 1,147 |
| Eight traces, every 250 | 0.96, 0.92 | 0.98, 0.91 | 283, 437 | 0.99, 0.93 | 0.93, 0.98 | 1,039, 1,341 |
| All traces, every 250 | 0.98, 0.94 | 0.97, 0.95 | 270, 412 | 0.99, 0.96 | 0.96, 0.96 | 1,037, 1,340 |

Among candidates, Euclidean, eight traces every 2,000 reads A 0.63, 0.70 and B 0.60, 0.67 at rate 0.001, and A 0.64, 0.70 and B 0.60, 0.74 at rate 0.01. The frozen encoder reads A 0.64, 0.65 and B 0.56, 0.63 at H 512.

- **Addresses do go stale.** Without rebuilding at the higher rate, A falls 0.99, 0.94 to 0.46, 0.56 by block 6. Its coverage falls to 0.46, 0.66, so old cues mostly abstain rather than answer wrongly, as the contract predicted.
- **Rebuilding fixes it, and cheaply in time.** A's memory-level change across block 5 is +0.03, +0.01 with eight traces every 2,000 at the higher rate, and block 6 changes little. Rebuilding every 2,000 updates is about as good as every 250.
- **The memory fills.** A cue that has drifted below τ allocates a new unit instead of updating its old one, and rebuilding realigns addresses without merging the duplicates. Units reach about 7–8 times the frozen encoder's at the higher rate.

At H 128 this is decisive. Whichever way the units fill, B has nowhere to go:

| H 128, rate 0.001, end of block 6 | A | B | Forced writes |
|---|---|---|---|
| Frozen encoder | 0.96, 0.91 | 0.88, 0.88 | 0, 219 |
| No rebuild | 0.81, 0.60 | 0.56, 0.28 | 2,626, 13,604 |
| Eight traces, every 2,000 | 0.90, 0.79 | 0.54, 0.04 | 3,806, 12,117 |
| Eight traces, every 250 | 0.60, 0.52 | 0.03, 0.02 | 24,565, 26,065 |

Rebuilding often is worst here, since memory has already filled before B arrives. H 512 at rate 0.01 fails similarly for B, at 0.56–0.67. This is the sensitivity to vigilance and presentation, and the growth of categories, that is known for adaptive resonance methods **[familiar; read before use]**.

## What this does not establish

- **Identity without labels.** The encoder keeps its labelled training on background symbols. Its similarity structure barely changes while it moves, which probably makes rebuilding easier than it will be on an encoder learning a different job.
- **Everything moving at once.** Values and reading were held fixed here. Learning addresses with re-encoded values has not been run together.
- **Merging.** Whether merging duplicates after a rebuild restores unit counts is the obvious next test, and it has not run.
- **Order and settings.** A then B only, and H was picked after seeing H 128.

## Computation

GTX 1650, three workers, one thread each: 249 s wall, 667 s aggregate CPU, for 12 tasks of 6,000 encoder updates each, including memory writes for all six arms. GPU peak 86 MB. These are study costs, not a benchmark.

Storage is the main new cost. Eight cue traces take 6,272 bytes per unit at one byte per pixel, so about 1.9 MB at 300 units and about 7 MB at 1,147 units, beside the 23 KB memory on a frozen encoder. Addresses and contents are 196 bytes per unit. Rebuilding costs 8 forward passes per unit, and every 2,000 updates this is small next to training. A separate trace store for values, as in the sensory trace pilot, would add to this. The encoder's own training is also a continuing cost, which a frozen encoder does not have.
