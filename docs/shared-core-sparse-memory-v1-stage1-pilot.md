# Shared-core sparse memory v1: value diagnostic and small stage 1 pilot (seeds 392 and 393)

Status: **pilot, 2026-11-02, following [external review](shared-core-sparse-memory-v1-contract.md#changes-from-external-review) of the [stage 0 pilot](shared-core-sparse-memory-v1-stage0-pilot.md). Not evidence for any contrast**: settings were chosen on these seeds, and the fixture's script was changed after a first look. No trainable comparator, probe or stress condition ran. Pairs are seed 392, seed 393, unless marked as means.
Runners `experiments/shared_core_value.py`, `experiments/shared_core_stage1_pilot.py`, `src/sparse_local_memory.py`; reports `reports/shared-core-sparse-memory-v1-value*.json`, `-stage1-pilot*.json`.

## Result

1. **Stored values are not the limit on answers. Reading rendered candidates is.**
2. **Sense-only local memory holds 90 arbitrary pairs in about 25 KB with little loss.** Adding the core makes it worse.
3. **No address reaches across senses.** Taps read at or below chance on every setting, and this follows from the code's geometry.

So the core has not earned a place in the address, on these seeds and this code. Stage 1 confirmation should not run on this design: it would spend its seeds on an answer the geometry already gives. Two directions remain, neither tested: reading memory through the reader's head, since that is where the cross-sense link is, and a separately contracted rendering-invariant objective for the core.

## Value diagnostic

Hosts frozen at block 3, the real 64,000-write stream, set A and set B, four reads in all, range over them:

| Content, then address | Nearest count prototype, no candidates | Euclidean among 9 candidates | Cosine among 9 | Matching head |
|---|---|---|---|---|
| Right count's mean code | 1.00 | 0.62–0.76 | 0.57–0.61 | 0.44–0.64 |
| Accumulated value, oracle address | 1.00 | 0.62–0.75 | 0.57–0.61 | 0.45–0.64 |
| Accumulated value, explicit store on frozen keys | 1.00 | 0.62–0.75 | 0.57–0.61 | 0.45–0.64 |
| Accumulated value, explicit store on the core's code | 0.23–0.33 | 0.17–0.27 | 0.16–0.27 | 0.12–0.22 |

Accumulating values costs nothing against perfect contents, and frozen keys lose nothing against an oracle. The whole gap to 1.00 comes from candidates: a single evaluation collection reads its own count at 0.61, 0.66 (within one count: 0.90, 0.96). Answers among these candidates therefore cap at about 0.70 whatever the memory does. **Corrected in the [moving host pilot](shared-core-sparse-memory-v1-moving-host-pilot.md#correction-to-the-value-diagnostic):** most of this gap is a systematic shift between training and evaluation renderings rather than noise, and 1.00 means the right stored value returned.

This corrects the stage 0 report. Over these reads cosine averages 0.59, below the floor of 0.60, which its 450-query reading had cleared at 0.61. Euclidean averages 0.69. The contract now reads memory by nearest count prototype, with Euclidean among candidates as the answer level. Prototype decoding has its own scale limit: it needs a stored prototype per answer.

## Pair memory

Selected by the contract's rule on pilot means and worst order, reading by count prototype, abstentions wrong. Old, then new, each order A then B and B then A.

| Address, selected setting | Old after old | Old after both | New after both | Worst old loss | Units of limit | Bytes used | Unknown answered |
|---|---|---|---|---|---|---|---|
| `sense-only`, H 128, k 1, τ 0.85 | 0.95, 0.91 | 0.93, 0.89 | 0.88, 0.94 | 0.02 | 118 of 128 | 23,324 | 0.13 |
| `conjunctive`, H 512, k 1, τ 0.85, marked | 0.89, 0.80 | 0.86, 0.75 | 0.76, 0.88 | 0.05 | 440 of 512 | 114,530 | 0.27 |
| `sensory`, none eligible; best near it, H 256, k 1, τ 0.70 | 0.90, 0.75 | 0.86, 0.71 | 0.71, 0.89 | 0.05 | 208 of 256 | | 0.36 |

Here the pairs are the two orders, as means over seeds. The answer level, Euclidean among candidates, is 0.59–0.66 for `sense-only` and 0.51–0.64 for `conjunctive`. `conjunctive`'s worst loss is just above the bar, so it is marked. Both its τ and H sit at the ends of the grid.

- **The core costs addressing.** The explicit store on `conjunctive` reads 0.76–0.94, against 1.00 on `sense-only`. Half of each cue is a code that places symbols on a number line, which is consistent with unrelated symbols being pulled together.
- **Whitening did not help.** `sensory` has no eligible setting, with the τ grid coarse around its best point at 0.70. Its explicit store reads 0.98–1.00, so the keys are good and the fixed τ fits them badly. `W_K` as whitening does not earn its place on these reads.
- **τ again matters most.** At τ 0.5, `sense-only` reads 0.31–0.43 for old after old. Forced writes appear at H 128, 133 on average for the selected setting and thousands elsewhere.
- **Local rule.** The learned-address rule approaches the store from below, as derived, and the known-to-unknown coverage gap is large at τ 0.85.

## Cross-sense fixture

A first run on English read at chance everywhere, and an oracle store showed why: English was taught first with no rehearsal, and at block 3 even an oracle store of its form prototypes reads 0.21, 0.22 by cosine, in the table below. That was my error in fixing the script, so the fixture now uses Devanagari, the most recently taught. Pair-memory codes were byte-identical across the change.

Oracle ceiling on the core's code, nine form prototypes with each write's count given, query's own count:

| Script | Forms, cosine | Forms, head | Taps, cosine | Taps, within one | Taps, head | Right count's similarity, taps |
|---|---|---|---|---|---|---|
| English | 0.21, 0.22 | 0.13, 0.14 | 0.00, 0.11 | 0.00, 0.33 | 0.00, 0.00 | 0.01, 0.03 |
| Chinese | 0.49, 0.53 | 0.35, 0.38 | 0.11, 0.11 | 0.22, 0.33 | 0.11, 0.14 | 0.19, 0.20 |
| Devanagari | 0.67, 0.80 | 0.66, 0.69 | 0.30, 0.22 | 0.64, 0.56 | 0.44, 0.67 | 0.57, 0.40 |

The memory, Devanagari, per address at its setting. Target derangement, chance 0.11:

| Address, setting | Same sense | Taps | Taps answered | Units |
|---|---|---|---|---|
| `core`, k 4, τ 0.5 | 0.11, 0.12 | 0.11, 0.11 | 0.89, 0.69 | 6, 9 |
| `conjunctive`, k 1, τ 0.85 | 0.80, 0.66 | 0.11, 0.00 | 0.23, 0.01 | 119, 69 |
| `sense-only`, k 1, τ 0.85 | 0.78, 0.64 | abstains | 0 | 37, 38 |
| `sensory`, k 1, τ 0.70 | 0.73, 0.61 | abstains | 0 | 82, 56 |

Across all settings, `core` taps never exceed 0.11 as a mean and `conjunctive` taps never exceed 0.11, so reach is not shown on any of them.

**Why, and why more τ cannot fix it [derived from these readings].** On `core`, forms of different counts have a best-other similarity of 0.96–0.97 against the right count's 0.94–0.96, so separating them needs τ above about 0.97, and units merge at every τ in the grid: 3–14 units for nine counts. Right-count taps sit at 0.40–0.57. So no threshold both separates forms and admits taps.

**The cross-sense link is in the readers, not in code proximity.** Taps find their form through the head at 0.44, 0.67 and by cosine at 0.30, 0.22. The contract's prediction that forms and taps of a count land together is not supported on this code, in cosine. Reading the head across stored cues would keep that link, at a cost linear in units, and is not tested here.

`conjunctive`'s same-sense read comes from its sensory part: it matches `sense-only` within 0.02, with about 2–3 times the units.

## Computation

CPU for memory, six workers, one thread each, float32. Code cache on the GTX 1650: 45 s per run, for all codes of both seeds.

| Phase | Wall | Aggregate worker CPU | Scope |
|---|---|---|---|
| Value diagnostic | 38 s | 14 s, parent | GPU reads, in one process |
| Grid, 18 settings × 3 addresses × 2 orders × 2 seeds | 126 s, first run | 810 s | 13.8 million writes; 44 µs per write, with Python overhead, at up to 512 units and 48 dimensions. The runtime file's wall is the later resume |

No backward pass is needed for memory, and the frozen encoder's forward cost is not included. This is one Python implementation, not a CPU inference benchmark. Memory bytes above exclude the 839,040-byte frozen encoder, which is charged to every address using it, and the `sparse48` host.
