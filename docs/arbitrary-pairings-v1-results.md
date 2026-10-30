# Arbitrary pairings v1: results

Frozen contract: [arbitrary-pairings-v1-contract.md](arbitrary-pairings-v1-contract.md) (2026-10-31). Seeds 320–329, N = 45
symbol–count pairs per set, L = 2,000 updates per pairing block, 64 training renderings, GPU. All six arms under Option A
(primary architecture comparison) and Option B (separate disruption test). Deterministic results:
`reports/arbitrary-pairings-v1-analysis.json` (from `reports/arbitrary-pairings-v1-pairs-{A,B}-N45-L2000-results.json.gz`).
Chance for recall is 0.111. No arm, seed, endpoint or reading was changed after the results were seen.

Sanity checks: `dense` is identical under A and B (B − A = 0 exactly on every disruption measure, all seeds); recall rescored
with shuffled labels is 0.111 (chance) under both options; the learning check (A against the second count assignment A') is
inconclusive for both arms checked, intervals centred near 0 (A: mixed +0.008, sparse48 −0.026), so no assignment-specific
difficulty is detected.

## Option A: means over ten seeds

Recall is on independent renderings (training renderings in brackets); "b4" and "b5" are the ends of blocks 4 and 5 in `P-AB`.

| arm | early acquisition (above chance) | recall A b4 | recall A b5 | interference | (a) at b5 | (a) damage |
|---|---|---|---|---|---|---|
| sparse48 | +0.042 | 0.34 (0.41) | 0.12 | +0.01 | 0.695 | +0.010 |
| dense | +0.106 | 0.53 (0.60) | 0.15 | +0.01 | 0.613 | −0.028 |
| mixed | +0.078 | 0.48 (0.57) | 0.14 | +0.02 | 0.625 | −0.026 |
| mixed-h125 | +0.083 | 0.46 (0.51) | 0.13 | +0.00 | 0.638 | −0.037 |
| mixed-h500 | +0.086 | 0.47 (0.54) | 0.13 | −0.02 | 0.665 | −0.011 |
| mixed-reset | +0.090 | 0.52 (0.58) | 0.14 | +0.02 | 0.650 | −0.033 |

## Prespecified readings (Option A; paired difference, 95% interval)

- **Memory adds specifics (`mixed` − `sparse48`): recall on A at the end of block 4 better** (+0.136 [0.082, 0.190];
  training renderings +0.153); whole-block acquisition area better (+0.101). The primary early-window acquisition measure is
  **inconclusive** (+0.035 [0.015, 0.055]: the interval crosses the margin).
- **Retained arbitrary recall:** recall on A at the end of block 5 is equivalent (+0.016 [0.002, 0.030]) but this is a floor
  result, not retention: every arm is at 0.12–0.15 against chance 0.111 after block 5, so the pairs were mostly gone for all
  arms (including `dense`) when the second set was taught. Interference (`P-A` − `P-AB` at b5) is inconclusive (+0.016
  [−0.022, 0.053]); it has almost nothing left to measure.
- **Structure preserved: no.** `mixed` is **worse** than `sparse48` on absolute (a) at the end of block 5 (−0.070
  [−0.114, −0.025]); the damage of pairing (`P-AB` − `C`) is inconclusive (−0.036 [−0.071, −0.002]). Memory bought
  specifics at the price of transfer structure, as the contract's reading anticipated, though through the arms' absolute
  level rather than through measured pairing damage.
- **Decay preserves the benefit (`mixed-h125` − `mixed`):** recall at b5 equivalent (−0.012 [−0.035, 0.011]) and early
  acquisition equivalent (+0.005 [−0.013, 0.024]); absolute (a) at b5 inconclusive but noninferior (+0.013 [−0.029, 0.055]).
  Recall at b4 is inconclusive (−0.023 [−0.099, 0.053]). **Decay sacrifices it:** not observed (nothing is worse).
- **Active decay against active reset (`mixed-h125` − `mixed-reset`):** recall at b4 **worse** (−0.068 [−0.134, −0.001]; training
  renderings inconclusive); early acquisition, b5 recall, interference, (a) and damage all equivalent.
- Where the memory lives (silencing, block 4, means): `sparse48` 0.34 → 0.12 with structure silenced (the sparse readers hold it);
  `mixed` 0.48 → 0.12 when memory rows are silenced (the memory holds it, the structure alone does not); `mixed-h125`
  0.46 → 0.36 when memory is silenced and 0.17 when structure is silenced (shared). In `dense` the silencing is a no-op by design.
- Reference: nearest-neighbour lookup on count 0.58 and on symbol identity 0.43; `sparse48` recall (0.34) sits below both and
  below `dense` (0.53).

## Option B (rewiring and resets continue through pairing): B − A per arm, ten seeds

| arm | recall A b4 | recall A b5 | early acquisition | (a) at b5 |
|---|---|---|---|---|
| dense | 0 (exact) | 0 | 0 | 0 |
| mixed | +0.004 [−0.030, 0.037] equivalent | +0.002 equivalent | −0.004 equivalent | −0.003 equivalent |
| mixed-h500 | −0.064 [−0.132, 0.003] inconclusive | −0.003 equivalent | −0.020 equivalent | +0.008 equivalent |
| mixed-h125 | −0.152 [−0.227, −0.076] **worse** | −0.009 equivalent | −0.031 equivalent | +0.048 inconclusive |
| sparse48 | −0.200 [−0.264, −0.135] **worse** | −0.026 inconclusive | −0.034 equivalent | **+0.066 better** |
| mixed-reset | −0.292 [−0.357, −0.227] **worse** | −0.029 inconclusive | −0.050 **worse** | **+0.055 better** |

Ability to acquire survives disruption only where nothing is rewired or reset (`mixed`, `dense`) or the memory is slow
(`mixed-h500`, inconclusive). Rewiring arms lose a third to a half of their block-4 recall (`sparse48` 0.34 → 0.14,
`mixed-reset` 0.52 → 0.23, `mixed-h125` 0.46 → 0.31) while their absolute (a) at block 5 rises. Caveats: (1) B also changes the
RigL anneal and parents in blocks 1–3, so B − A is not purely the effect of rewiring during pairing; (2) `sparse48` under B has a
nearly flat acquisition curve and ends near the floor (0.14), so its "worse" is partly floor-bound; (3) the (a) rise under B
is read on the absolute level only and differs in direction from Option A's contrast across arms; this is descriptive. Differences
in survival between arms are descriptive per the contract.

## What this does and does not show

- A decaying/mixed memory acquires arbitrary pairings faster and stores more of them than sparse structure alone, and this
  extra storage sits in the memory rows; structure alone holds less (and is what the sparse arm uses).
- Within this design the pairs do not survive the next block for any arm; the retained-recall and interference readings
  are therefore uninformative rather than favourable. A follow-up that measures retention at smaller delay (or with
  rehearsal) is needed before claiming anything about interference or consolidation of arbitrary pairs.
- Memory comes with a transfer-structure cost in the absolute level (`mixed` vs `sparse48`: −0.070) for all dense-containing
  arms; `mixed-h500` and `mixed-reset` are closer to `sparse48` than `mixed` is, descriptively.
- Decay with half-life 125 does not cost recall or acquisition against `mixed` at the ends of blocks 4–5 but is below
  `mixed-reset` at b4.
- Limits: one fixed cohort (no second cohort or stopping rule); symbol–count pairs only; GPU (composition-measure caveat from
  earlier studies); the Option B contrast changes blocks 1–3 as well; sparse arms are RigL-wired — the random-rewiring
  follow-up on fresh seeds (agreed) is still to run.

## Computation (measured costs)

Theoretical: per update the cost is that of the limited-memory study's step on 64 reader rows (48 sparse + 16 memory) with
batch 16; time O(updates × batch × active weights), storage O(parameters) plus the stored rendering pool (64 renderings × 90
symbols × 784 floats ≈ 18 MB) per seed. No checkpoints are written.

Measured (GPU GTX 1650, 6 worker processes, 60 arm-seeds per option): Option A runs 5,340 s (data 28 s, rendering 3 s), Option B
5,357 s; peak RSS 1.58 GB (parent) and 2.58 GB (largest worker); parent CPU 7.4 s each; analysis 0.17 s wall. Selection and
tuning: the seed-299 pilot grid took 1.0 h, plus the diagnostic pilots (8 and 64 renderings, fixed font; see the pilot doc).
Total for the frozen study ≈ 3.0 h, of which ≈ 2.97 h is the two runs. Timings are in the `-runtime.json` files.
