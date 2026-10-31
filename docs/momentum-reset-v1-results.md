# Momentum reset v1: results

Frozen contract: [momentum-reset-v1-contract.md](momentum-reset-v1-contract.md) (2026-10-31). Fresh seeds 340–349, four arms (`sparse48`, `dense`, `mixed`,
`mixed-h125`), N = 45 pairs, L = 2,000 updates, Option A, GPU, from the end of block 4. Deterministic results: `reports/momentum-reset-v1-analysis.json`, from
`reports/momentum-reset-v1-results.json.gz`. No arm, seed, branch, endpoint or reading was changed after the results were seen.

D(t) = recall(t) − recall(0), paired over ten seeds; margin 0.05; intervals are 95% paired t-intervals; positive differences mean less loss. Recall at update 0:
sparse48 0.36, dense 0.48, mixed 0.51, mixed-h125 0.43 (chance 0.111).

## Reading 2: checks
All pass in all four arms: the first-moment reset zeroed every initialized state and left `exp_avg_sq` and the step count unchanged (no uninitialized states), and
`zero-grad-m0` moved no weight and left recall constant.

## Reading 1: replication
Zero-gradient stepping with inherited Adam state is worse than update 0 in every arm, and the loss is already present at update 50 and does not change afterwards:
D = −0.112 (sparse48), −0.105 (dense), −0.194 (mixed), −0.138 (mixed-h125); every interval excludes 0. The earlier forgetting study found 0.09–0.22.

## Reading 3: `ctx-m0` − `ctx`
| Arm | D(50) `ctx` | D(50) `ctx-m0` | difference D(50) [95% CI] | outcome | D(250) difference | D(2,000) difference |
|---|---|---|---|---|---|---|
| sparse48 | −0.053 | −0.039 | +0.014 [−0.062, +0.091] | inconclusive | +0.007, equivalent | −0.010, equivalent |
| dense | −0.043 | −0.011 | +0.032 [−0.036, +0.099] | inconclusive | −0.008, inconclusive | +0.012, equivalent |
| mixed | −0.081 | −0.024 | +0.058 [+0.003, +0.112] | better | +0.011, inconclusive | +0.008, equivalent |
| mixed-h125 | −0.088 | −0.006 | +0.082 [+0.033, +0.131] | better | +0.034, inconclusive | +0.002, equivalent |

Zeroing the first moments reduced the drop at 50 updates in the two arms with a memory component, by about three-quarters or more of `ctx`'s drop in `mixed-h125`
(−0.088 to −0.006) and about two-thirds in `mixed`. The benefit is early and transient: by 250 updates it is no longer established in any arm, and by 2,000 updates the
two branches are equivalent in every arm (D(2,000) about −0.2 to −0.3 in both, no prediction was made). The `mixed` lower bound (+0.003) is barely above zero; the
`mixed-h125` bound is clearer. In `sparse48` and `dense` the data neither show a benefit nor exclude one up to the interval width (about ±0.07). "Equivalent" at 250
and 2,000 means no practically meaningful difference at that checkpoint, not that momentum plays no role.

## Reading 4: cost gates (noninferior = lower bound > −0.05)
| Arm | (i) context performance | (ii) transfer (a) | (iii) `hold-m0` − `hold` D(250) | (iii) end recall | all noninferior |
|---|---|---|---|---|---|
| sparse48 | −0.002 [−0.029, +0.024] yes | +0.002 yes | +0.031 [+0.003, +0.059] yes | +0.011 yes | yes |
| dense | +0.016 [−0.022, +0.053] yes | −0.007 yes | −0.008 [−0.085, +0.070] **no** | +0.025 yes | no |
| mixed | +0.009 [−0.023, +0.042] yes | +0.002 yes | −0.028 [−0.091, +0.034] **no** | −0.004 [−0.054, +0.047] **no** | no |
| mixed-h125 | −0.005 [−0.039, +0.029] yes | +0.008 yes | −0.002 [−0.048, +0.044] yes | +0.002 yes | yes |

Gate (iii) failing for `dense` and `mixed` means the data cannot rule out a loss under rehearsal of more than 0.05 (wide intervals), not that a loss was found.
The four training losses are descriptive and are in the analysis JSON (`loss_diff_m0_minus_ctx`); for example the `mixed-h125` match-loss difference over updates
1–50 is −0.022 [−0.058, +0.013]. They are not summarized further here.

## Reading 5: remedy candidate (D(50) better and every gate noninferior)
- `mixed-h125`: **yes**. `ctx-m0` reduces the early drop (+0.082), with no demonstrated cost in continued context performance, transfer (a) or rehearsal.
- `mixed`: no. D(50) is better, but gate (iii) is not established.
- `sparse48`, `dense`: no, since D(50) is inconclusive.

Scope of that verdict: "candidate" is the contract's term for something worth carrying into the remedy comparison. It is one arm, ten seeds, one checkpoint (50
updates), and the benefit is gone from the evidence by update 250 and absent by 2,000, so it delays rather than prevents the loss. The contract has no multiplicity
adjustment across the four arms. A reset is not a retention solution on this evidence; it is a cheap way to avoid an early drop that the later decline overtakes.

## Reading 6: descriptive
- `ctx-fresh` (state fully cleared) minus `ctx`: not better than the first-moment reset anywhere; at update 250 it is worse than `ctx` in all arms (sparse48 −0.059, dense
  −0.150, mixed −0.075, mixed-h125 −0.077; intervals exclude 0), so also clearing the second moments appears to hurt. Not a prespecified comparison.
- Recall at 2,000 updates is 0.12–0.20 in `ctx` and `ctx-m0` in all arms (chance 0.111), again close to the floor.

## Interpretation (bounded)
Inherited first-moment state is a supported contributor to the early recall drop in real context training in the memory-bearing arms: removing it reduces the drop
at 50 updates. It cannot account for the long-term loss, which is the same with or without the reset. The reset costs nothing measurable on the gates in
`mixed-h125` (and `sparse48`). It is a candidate component for the remedy comparison, not a remedy.

## Computation (measured costs only)
Theoretical: the reset is one pass over the optimizer state, O(P) time once per branch and no extra storage (it overwrites `exp_avg` in place); the study's cost is
7 branches × 4 arms × 10 seeds × up to 2,000 updates plus evaluation. Measured on the GTX 1650: 4,102 s wall (runs phase 4,066 s; data 28 s; rendering 3 s), 23,434 s
total worker CPU, peak worker RSS 2.61 GB (parent 1.85 GB), 40 arm-seeds, about 100 s of wall time per arm-seed on average with six workers in parallel. Analysis: 0.6 s. No tuning or selection was performed. Timing is in
`reports/momentum-reset-v1-runtime.json` and `-analysis-runtime.json`; the results JSON holds only deterministic content.
