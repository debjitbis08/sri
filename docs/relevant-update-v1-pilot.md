# Relevant update v1: stage 1 pilot (seeds 380 and 381)

Status: **pilot, 2026-11-01. Not evidence for any contrast**: settings were chosen on these seeds, and some rules and grids were added after seeing the first results (recorded in the [contract](relevant-update-v1-contract.md#changes-from-the-pilot)). `reports/relevant-update-v1-selection.json`; `experiments/relevant_update.py pilot`.

**Reading if this held on fresh seeds: `improved-below-bar`.** The chosen rule, `expert`, beats `rank-one` by a wide margin but sits just under the bar at 4 experiences, 0.68 against 0.70. Comparison B finds no benefit from joint processing as such.

## Primary cell

`clean`, 16 new rules, k = 4, no contamination, means over both seeds. Each rule's setting is the highest rule generalization with old knowledge at least 0.85.

| rule | setting | rule generalization | old knowledge | breadth | immediate | operations per update |
|---|---|---|---|---|---|---|
| `rank-one` | none | 0.42 | **0.66** | 0.43 | 0.84 | 15,872 |
| `ridge` | λ 10 | 0.05 | 0.99 | 0.03 | 0.06 | 29,248 |
| `ridge-rehearse` | λ 1, r 8, w 0.25 | 0.29 | 0.92 | 0.20 | 0.62 | 92,352 |
| `ridge-interfered` | λ 0.1, r 8, w 0.25 | 0.28 | 0.93 | 0.23 | 0.80 | 892,672 |
| `ridge-orthogonal` | λ 0.01, q 4 | 0.29 | 0.90 | 0.23 | 0.55 | 53,824 |
| `ridge-keep` | λ 10, r 32, w 0.25 | 0.44 | 0.85 | 0.29 | 0.06 | 1,214,208 |
| **`expert`** | τ_x 0.425 | **0.68** | 0.88 | 0.19 | 0.70 | 12,800 |
| `omega` | 2 layers, α 0.001, η 0.003 | 0.07 | 0.92 | 0.19 | 0.12 | 428,140 |
| `titans-m0` | 2 layers, α 0.001, η 0.003 | 0.08 | 0.91 | 0.20 | 0.17 | 179,008 |
| `titans` | 2 layers, α 0.001, β 0.5, η 0.001 | 0.07 | 0.91 | 0.20 | 0.08 | 89,464 |

Immediate is rule generalization right after a rule's own update. `rank-one` has no settings and misses the old-knowledge limit.

## What this shows

- **Shared memories learn a rule, then lose it.** `rank-one` reads 0.84 immediately and 0.42 after 16 rules; `ridge` at small λ 0.87 and 0.43. Neural memories at their highest rule generalization read 0.9 and 0.18, with old knowledge at most 0.65. Interference across a shared map is the largest loss here, and old knowledge pays too.
- **Local memory avoids that, at a cost in bytes.** `expert` changes no shared weights, and reads the same immediately and at the end. It holds up with load (0.68, 0.68, 0.63 at L = 4, 16, 32), where `rank-one` falls (0.76, 0.42, 0.26). Its storage is linear in rules: **it is not an update of shared knowledge.** It is close to a nearest-prototype classifier with a rejection threshold, so its win is evidence for a prototype store rather than a better shared update.
- **`expert` depends on a sensitive threshold.** τ_x 0.4, 0.425, 0.45 and 0.5 read 0.75/0.85, 0.68/0.88, 0.60/0.91 and 0.43/0.95 on rules and old knowledge. Similarity thresholds drifted across encoders in streaming retrieval, so the same setting may land elsewhere on fresh seeds.
- **More evidence helps a lot.** `expert` reads 0.25, 0.44, 0.68, 0.84 at k = 1, 2, 4, 8, old knowledge staying 0.87–0.91. At k = 8 it is well over the bar.
- **Protective rules protect old knowledge and do not fix acquisition.** They trade rule generalization for it. They protect old rules only, as declared.
- **Keeping all experiences does not restore a shared map.** `ridge-keep` refits every kept set each time, the global least-squares case, and its best reads 0.63 at old knowledge 0.60, or 0.44 within the limit. The joint ceiling of 0.89–0.93 used 384 events per rule. So what is short here seems to be examples per rule, not memory of them. A few cues leave item noise in a shared map.
- **Neural memories, with these constants, are the weakest.** This is not a test of OmegaNet, Titans or ATLAS: no kernels, projections, learned gates or optimizer changes.

## Together

**Comparison B, matched, on joint processing as such: no benefit shown.**

- `omega` against `omega-seq`: 0.07 and 0.07 at k = 4; 0.11 and 0.11 at k = 8; old knowledge within 0.01. As derived, they coincide at the small steps selected.
- `ridge` against `ridge-cycled`: at λ ≤ 0.1, 0.43 against 0.43 at k = 4 and 0.50 against 0.49 at k = 8. At larger λ they diverge, since repeating steps regularizes less overall: λ 1 at k = 4, `ridge` 0.38/0.82 and `ridge-cycled` 0.44/0.66; λ 10, 0.05/0.99 and 0.31/0.90. These trace a similar tradeoff rather than one being better, so the contract's single-setting comparison is not a fair match, and comparison across λ replaces it.

**Comparison A, natural: `expert` gains from taking a set at once.** 0.68/0.88 at k = 4 against 0.56/0.79 in turn, and 0.84/0.87 against 0.70/0.71 at k = 8, with lower breadth, and needing 1 expert per rule against k. The likely reason, not tested: averaging cues cancels item noise and leaves the family, so a mean cue is a better key. In a shared map, the same averaging gave no gain (`rank-one` 0.42 at both). So on this pilot, what helps is **forming a prototype from several experiences**, rather than taking gradients together.

## Contamination: what selection must deliver

`expert` at the primary cell, k = 4 and k = 8, rule generalization:

| kind | p = 0 | p = 0.25 | p = 0.5 |
|---|---|---|---|
| `unrelated`, k = 4 | 0.68 | 0.47 | 0.16 |
| `same-outcome`, k = 4 | 0.68 | 0.46 | 0.23 |
| `same-family`, k = 4 | 0.68 | 0.69 | 0.55 |
| `unrelated`, k = 8 | 0.84 | 0.68 | 0.28 |
| `same-outcome`, k = 8 | 0.84 | 0.67 | 0.38 |
| `same-family`, k = 8 | 0.84 | 0.83 | 0.78 |

- **Wrong family hurts, wrong outcome barely does.** A contaminant from another family shifts the mean cue off the family, so the gate misses. A contaminant in the right family only shifts the mean residual. `rank-one` differs: it loses to all three kinds.
- **Tolerance.** By the contract's 0.8 rule: k = 4 tolerates p = 0 for other-family contaminants, and k = 8 tolerates 0.25. So selection must find **the right family** above all. That favours cue similarity, whereas v1's error grouping found the outcome, which matters least here.

## Other cells

- **`degraded`, descriptive:** `expert` 0.33/0.58, `rank-one` 0.24/0.41. Nothing comes near the bar, as the gates predicted.
- **Load L = 32 is descriptive:** ceilings are below the bar on both seeds.

## Computation (measured costs only)

Encoders 67 s each on the GTX 1650. Pools 24 s wall for both seeds. Gates 55 s wall. The pilot's first run took under 2 minutes wall, and in total 2,968 cells used 519 s of worker CPU on 11 workers (`reports/relevant-update-v1-pilot-runtime.json` records the last session's wall time only). Settings scored: `titans` 60, `ridge-rehearse` 36, `omega` 30, `titans-m0` 30, `ridge-interfered` 24, `ridge-keep` 24, `ridge-orthogonal` 20, `expert` 10, `ridge` 4, `rank-one` 1.
