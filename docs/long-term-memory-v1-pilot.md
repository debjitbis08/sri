# Long-term memory v1: pilot (seeds 392 and 393)

Status: **pilot, 2026-11-01. Not evidence for any contrast**: settings were chosen on these seeds, and the diagnosis came after seeing them. Kept as found: `reports/long-term-memory-v1-selection.json`, `reports/long-term-memory-v1-diagnosis.json`; `experiments/long_term_memory.py pilot`, `experiments/long_term_memory_diagnosis.py`. Outcome: **the contract cannot be frozen as written.** The selected candidate does nothing measurable. The diagnosis finds why, and a stronger version retains something, but not enough, and with costs.

## The sweep, by the contract

Retained at the end of block 6; means over both seeds.

| | retained A | retained B | answer transfer, block 3 |
|---|---|---|---|
| `sparse48`, readers alone | 0.09 | 0.19 | 0.29 |
| `mixed` | 0.12 | 0.22 | 0.21 |
| `flat-store`, the explicit diagnostic | 0.99 | 0.96 | |
| candidate: `k-sparse`, `live` keys, p 0.02, η 0.01, β 0.9, C 1 | 0.09 | 0.19 | 0.28 |
| best of the sweep: `k-sparse`, `frozen` keys, `slow` values | 0.12 | 0.22 | 0.29 |

By the rule's shape, this reads **`no-retention-shown`**. Scale `s` ended at −0.02 and 0.12. `live` was chosen by the key-source preference, as within 0.03 of a best that was itself barely above the readers.

## Diagnosis (post hoc)

**1. Values were the limiter, and gate 2 removed the best source.** Memory alone, Adam, `k-sparse`, trained on A then B or on both, read on the block 6 cache:

| keys, values | width | A after A | A after B | joint A | joint B |
|---|---|---|---|---|---|
| `live`, `live` | 52 | 0.11–0.14 | 0.07–0.17 | 0.08–0.16 | 0.16–0.17 |
| `live`, `live` | 1024 | 0.08–0.16 | 0.08–0.15 | 0.08–0.13 | 0.21 |
| `slow`, `slow` | 52 | 0.12–0.14 | 0.07–0.13 | 0.10–0.14 | 0.16 |
| `slow`, `slow` | 1024 | 0.11 | 0.07–0.17 | 0.08–0.14 | 0.18–0.22 |
| `frozen`, `slow` | 52 | 0.27 | 0.12–0.13 | 0.13–0.19 | 0.29 |
| `frozen`, `slow` | 1024 | 0.22–0.28 | 0.12–0.14 | 0.14–0.21 | 0.27–0.37 |
| **`frozen`, `frozen`** | 52 | 0.52 | 0.16 | 0.44–0.51 | 0.48–0.51 |
| **`frozen`, `frozen`** | 1024 | 0.55–0.59 | 0.21–0.23 | 0.56–0.57 | 0.55–0.62 |

Ranges are over the two seeds.

Codes for values have moved by the time they are read; frozen ones have not. Gate 2 sent `frozen` keys to `slow` values because the frozen encoder reads counts at only 0.72. That threshold was wrong in direction: a value source that identifies counts at 0.72 and stays put beats one that drifts. So 0.72 is roughly its ceiling, and joint training's 0.55–0.62 is near that.

**2. It is not mainly capacity or the optimizer, at these sizes.** With frozen values, joint training reads about 0.5 at width 52 and about 0.57 at width 1,024, at 14× the floats. Adam and momentum SGD are close.

**3. Interference is the main loss once values hold.** Trained in sequence, A falls from 0.52–0.59 after A to 0.16–0.23 after B, at either width. Nothing here protects it.

**4. With the contract's update rule and frozen values, retention rises, with new costs.** Combined answers, mean of both seeds. Each column is the best of its settings on these seeds, 18 settings for each form at the budget's width, so these are optimistic.

| reading | readers alone | in budget: `mlp`, η 0.03, C 1 | about 800 KB: `k-sparse`, width 1,024, η 0.003, C 1 |
|---|---|---|---|
| A at the end of block 4 | 0.30 | 0.59 | 0.60 |
| B at the end of block 5 | 0.40 | 0.62 | 0.62 |
| A at the end of block 5, after B | 0.15 | 0.20 | 0.32 |
| retained A, block 6 | 0.09 | 0.14 | 0.22 |
| retained B, block 6 | 0.19 | 0.40 | 0.38 |
| memory alone, A then B, block 6 | | 0.16, 0.50 | 0.32, 0.53 |
| answer transfer, block 3 | 0.29 | 0.29 | 0.32 |
| of readers' correct answers, lost at block 6 | | 0.50 | 0.33 |
| of unknown symbols' answers, changed at block 6 | | 0.39 | 0.34 |

So at block 6, against the rule's +0.15: A gains +0.05 to +0.13 and misses it; B gains +0.19 to +0.21. Acquisition is much better and real. Holding A through B is not solved, even far over budget. Sparse activation did not beat `mlp` within budget; it helped A only with width.

**5. The combined answer decays without any writes.** B falls from 0.62 to 0.38–0.40 during block 6, when memory has no items, while memory alone holds 0.50–0.53. The readers' logits keep changing, so a residual fitted to them at writing no longer fits. Adding to moving logits is fragile, and memory alone beats the combined answer by block 6.

**6. It does harm that block 3's reading cannot see.** A third to a half of readers' correct answers are lost at block 6, and over a third of unknown symbols' answers change. Memory answers confidently where it knows nothing. Its base at block 6 is small, as readers are near chance on most matching.

**7. No relational rescue.** I had hoped memory might hold form-to-collection matching as the host forgets it. Paired matching reads 0.08–0.10 combined against 0.08 alone at block 6.

**8. Where pairs are held (`experiments/long_term_memory_where.py`).** Frozen keys and values, query by query, means of both seeds, the budget's `mlp` then the width-1,024 `k-sparse`. "Memory right" is memory's term alone. Chance is 0.11 per source.

| reading | readers right | memory right | right by memory only | right by readers only | right by both, against chance overlap | combined keeps memory-only answers |
|---|---|---|---|---|---|---|
| A, block 4 | 0.30, 0.30 | 0.49, 0.48 | 0.33, 0.32 | 0.13, 0.14 | 0.16, 0.16 against 0.14 | 0.80, 0.81 |
| B, block 5 | 0.40, 0.40 | 0.51, 0.53 | 0.30, 0.29 | 0.19, 0.16 | 0.21, 0.24 against 0.21 | 0.74, 0.74 |
| A, block 6 | 0.09, 0.09 | 0.16, 0.32 | 0.16, 0.30 | 0.08, 0.07 | 0.01, 0.02 against 0.01–0.03 | 0.37, 0.42 |
| B, block 6 | 0.19, 0.19 | 0.50, 0.53 | 0.42, 0.44 | 0.12, 0.10 | 0.08, 0.09 against 0.10 | 0.50, 0.43 |

Pairs do reach memory, and by block 6 what remains is mostly there. The readers hold them only while being taught. Overlap is what independence would give, so memory is **not** specializing in what readers get wrong, despite the residual loss. By block 6 the combined answer drops about half of memory's correct answers, since memory's term and the readers' logits are about the same scale and the readers are wrong with confidence. Memory's answers agree across a symbol's five renderings only 0.60–0.68 of the time, so a pair is not held crisply, though the frozen encoder separates symbols perfectly. This does not show where inside the network a pair is held. The selected candidate, with `live` keys, held nothing measurable.

## What the pilot leaves open

- **Bytes.** A width-1,024 memory, with its charged copies, is about 800 KB, against `flat-store`'s 23 KB and 0.99. A network pays heavily here for what it does.
- **Protection.** No test here of frozen experts, replay, gating by key or slower schedules beyond C and η.
- **Two seeds, and the best of many settings:** the figures in finding 4 are optimistic.

## Computation (measured costs only)

Encoders 67 s each. Host: 318 s wall per full run, 220–256 s for `mixed`, 201 s plain, three concurrent on the GTX 1650; 340 MB of cache at about 46 items per update, 146 MB peak GPU per process. Gates under a minute. Sweep: 28 replays executed and 12 reused, 136 s of worker CPU, 38 s wall, far below the 1–2 CPU-hours estimated. Diagnosis about 150 s wall on 11 workers.
