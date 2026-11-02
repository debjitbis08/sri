# Shared-core sparse memory v1: refresh pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02. Not evidence for any contrast**: two seeds, A then B only, refresh at block ends only, and `snap-train` and the map diagnostic were added after a first look. Runner `experiments/shared_core_refresh.py`; reports `reports/shared-core-sparse-memory-v1-refresh*.json`.

## Question and setup

Can stored values keep up with a learning host, without a frozen copy of it and without replaying pairs? The [moving host pilot](shared-core-sparse-memory-v1-moving-host-pilot.md)'s replay, with the same memory and writes. At the end of blocks 4, 5 and 6, stored values are changed by one rule. Probes are evaluation collections with even item index; reading uses those with odd index, so these figures are not the same query set as the moving host pilot's.

| Arm | Rule | Extra state |
|---|---|---|
| `none` | Nothing | |
| `map-135`, `map-27` | Affine map from probes' previous codes to current codes, applied to values written before this block | Probe images, 423 KB or 85 KB as float32, plus their codes |
| `map-135-half` | As `map-135`, and moves values written in this block half way, since they lag by about half a block | The same |
| `snap` | Nearest count prototype of the probes, replaced by that count's current prototype | Probe images, count labels on probes |
| `snap-train` | The same, with training-collection prototypes from the block's last 4,000 writes | Nine prototypes, count labels on collections |
| `oracle-count`, `oracle-train` | Each unit's majority true count, replaced by that count's current prototype, of probes or of training collections | A ceiling, using labels |

The planned oracle, re-encoding the true images, is not possible on this cache. These two stand in for it. Block 6 has no training collections in the cache, so `snap-train` and `oracle-train` have no block-6 read.

Gates: candidate order reproduces from its stream, the probe table is complete, and `none` equals the unrefreshed memory.

## Result

**Refresh can work in principle, but neither candidate rule works yet.** The ceilings recover the answers. The drift map is unstable, and that comes from the mismatch between probe and value renderings rather than from drift being too complex. Nearest-prototype snapping helps moderately, and its limit comes from values already being stale when written.

Answers among candidates, Euclidean, abstentions wrong:

| Arm | A, after 4 | A, after 5 | A, after 6 | B, after 5 | B, after 6 |
|---|---|---|---|---|---|
| `none` | 0.52, 0.39 | 0.44, 0.44 | 0.35, 0.27 | 0.50, 0.69 | 0.30, 0.33 |
| `map-135` | 0.52, 0.39 | **0.09**, 0.67 | 0.24, 0.41 | 0.50, 0.69 | 0.53, **0.10** |
| `map-27` | 0.52, 0.39 | 0.11, 0.50 | 0.27, 0.48 | 0.50, 0.69 | 0.24, 0.12 |
| `map-135-half` | 0.29, 0.31 | 0.37, 0.31 | 0.24, 0.29 | 0.36, 0.48 | 0.43, 0.15 |
| `snap` | 0.39, 0.57 | 0.46, 0.54 | 0.37, 0.51 | 0.48, 0.57 | 0.50, 0.56 |
| `snap-train` | 0.50, 0.44 | 0.55, 0.51 | no read | 0.50, 0.73 | no read |
| `oracle-count` | 0.55, 0.81 | 0.58, 0.83 | 0.58, 0.76 | 0.58, 0.77 | 0.63, 0.72 |
| `oracle-train` | 0.65, 0.62 | 0.63, 0.85 | no read | 0.54, 0.80 | no read |
| Block-3 values and codes, frozen | 0.65, 0.74 | 0.66, 0.70 | 0.66, 0.70 | 0.59, 0.63 | 0.59, 0.63 |

Memory level, nearest training-collection prototype:

| Arm | A, after 4 | A, after 5 | B, after 5 |
|---|---|---|---|
| `none` | 0.66, 0.56 | 0.68, 0.30 | 0.68, 0.87 |
| `snap-train` | 0.66, 0.56 | 0.66, 0.54 | 0.68, 0.85 |
| `oracle-train` | 0.97, 0.96 | 0.97, 0.95 | 0.92, 0.93 |
| Block-3 values and codes, frozen | 0.95, 0.95 | 0.95, 0.92 | 0.89, 0.85 |

**Why the map fails.** Share of squared drift it removes, 1 is perfect and below 0 is worse than leaving codes alone:

| Step | Held-out evaluation items | Training-collection prototypes |
|---|---|---|
| Block 3 to 4 | 0.95, 0.94 | 0.99, 0.97 |
| Block 4 to 5 | 0.74, 0.82 | **−5.37**, 0.50 |
| Block 5 to 6 | 0.90, 0.89 | no read |

So the map does capture drift on the rendering it was fitted to, and it can fail badly on the other. Fitting it on training-rendering probes is the obvious repair, and it is untested, because the cache holds those probes at no later checkpoint.

**Why snapping is limited.** `snap-train` meets the forgetting bar at the memory level: across block 5, A goes 0.66 to 0.66 and 0.56 to 0.54, where `none` loses 0.26 on seed 393. But it starts from only 0.66, 0.56, against 0.97, 0.96 with the true count. Host drift within a block, about 7 RMS in block 4 against neighbouring gaps of about 3, means a value averaged over the block is already nearest the wrong prototype by its end. So refreshing at block ends comes too late, and the value needs placing while it is being written.

**This points to a design question.** `oracle-train` amounts to storing a pointer to a concept with a live prototype, rather than a copy of a code. That would make drift irrelevant to stored facts, and it is why this arm does as well as frozen values. **Withdrawn after discussion: see the [sensory trace pilot](shared-core-sparse-memory-v1-traces-pilot.md).** It needs a way to know the concept at write time. The contract says count IDs are not stored, so using one would be a departure, and it would need a stored prototype per concept, which is the same scale limit as the memory-level read.

## Computation

CPU, one process, float32: 9.6 s wall, 9.6 s CPU, for both seeds, 128,000 writes per seed and refreshes per arm, loading cached codes included. No new host fit. This does not include the host forward passes a real refresh would pay for its probes: K images per refresh. An affine fit is O(Kd² + d³), and applying it to h units is O(hd). At 423 KB, the larger probe set is larger than the memory it serves, 23 KB. Nine prototypes at 16 dimensions take 576 bytes.
