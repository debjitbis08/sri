# Surprise lifecycle v1: pilot (seeds 299 and 298)

Status: **pilot, 2026-11-01. Not evidence for any contrast**: it selected settings on the seeds it reports, and it decides whether the [contract](surprise-lifecycle-v1-contract.md) can be frozen. Outcome: **no, and this is kept as v1's failed result.** Gate 6 passes, but `life` learns no rules, for reasons in both the mechanism and the contract's choice of gaps (see *Diagnosis*). `reports/surprise-lifecycle-v1-selection.json`; `experiments/surprise_lifecycle.py pilot`.

## Readings at the primary cell

q = 0.2, 64 KiB, `degraded` stream, audited on clean renderings with `degraded` prototypes. Selected settings, means over both seeds, gaps 128 / 512 / 2,048 / 8,192 / 32,768.

| arm | rule generalization | binding recall | old knowledge, all gaps | noise capture, all gaps | operations per event |
|---|---|---|---|---|---|
| `life` | 0.00 at every gap | 0.75, 0.30, 0.25, 0.08, 0.03 | 0.87 | 0.02 | 18,463 |
| `life-sketch` | 0.00 at every gap | 0.59, 0.26, 0.11, 0.06, 0.07 | 0.89 | 0.01 | 13,030 |
| `life-cuegroup` | at most 0.01 | 0.75, 0.35, 0.02, 0.03, 0.05 | 0.86 | 0.00 | 19,158 |
| `count` | at most 0.03 | 0.77, 0.17, 0.09, 0.05, 0.04 | **0.72** | 0.01 | 19,826 |
| `count-sketch` | at most 0.07 | 0.47, 0.19, 0.06, 0.10, 0.03 | **0.77** | 0.01 | 13,950 |
| `life-selfcount` | 0.00 at every gap | 0.75, 0.46, 0.17, 0.05, 0.07 | 0.87 | 0.03 | 17,958 |
| `life-oracle` | 0.46, 0.40, 0.48, 0.33, 0.44 | 0.84, 0.86, 0.78, 0.83, 0.77 | **0.80** | 0.01 | not counted |
| `reservoir-replay` | 0.00 at every gap | 0.55, 0.28, 0.03, 0.06, 0.06 | **0.85** (0.848) | 0.02 | 28,451 |
| `sleep-periodic` | at most 0.02 | 0.12, 0.04, 0.04, 0.02, 0.03 | **0.81** | 0.00 | 29,199 |
| `slow` | at most 0.01 | 0.05, 0.05, 0.03, 0.03, 0.02 | 0.87 | 0.00 | 10,906 |
| `surprise-fifo` | 0.00 at every gap | 0.09, 0.05, 0.04, 0.03, 0.02 | 0.85 | 0.01 | 19,432 |
| `fifo-replay` | at most 0.01 | 0.05, 0.03, 0.03, 0.02, 0.02 | 0.85 | 0.00 | 45,040 |
| `titans-m0` | at most 0.01 | 0.06, 0.01, 0.03, 0.02, 0.01 | 0.88 | 0.01 | 62,938 |
| `titans` | at most 0.03 | 0.02, 0.05, 0.03, 0.01, 0.00 | 0.86 | 0.02 | 31,312 |
| `omega` | at most 0.02 | 0.05, 0.06, 0.05, 0.04, 0.01 | 0.86 | 0.01 | 1,650,006 |
| `frozen` | 0.00 at every gap | 0.05 at every gap | 0.90 | 0.01 | 5,376 |
| `sleep-grow` | at most 0.01 | 0.03, 0.02, 0.03, 0.03, 0.03 | **0.81** | 0.01 | 30,597 |

Bold old knowledge is below the provisional limit of 0.85. Chance is 0.03; with so few cues and seeds, values up to about 0.05 are not distinguished from it here.

## What this shows

- **Gate 6 passes, which shows only partial feasibility.** `life-oracle` exceeds the best comparator's objective by 0.61, 0.55 and 0.58 at the long gaps, with rule generalization of 0.33–0.48 from its first 4 events per rule. So rules can be learned with sorting supplied. But its old knowledge falls to 0.80, below the limit: it has not shown acquisition and preservation together.
- **Nothing without the oracle learns rules.** The largest value in any non-oracle arm at any gap is 0.07.
- **`life` is ahead on bindings at gap 2,048 only.** It reads 0.25 there against at most 0.05 for the comparators, on seeds differing widely (0.12, 0.39). At 8,192 it reads 0.08 and the best comparator 0.06; at 32,768, 0.03 and 0.06. By the frozen rule's shape it would read as `no-advantage-shown`, which needs two long gaps.
- **Comparators selected near doing nothing.** Most chose the smallest η on the grid, and several have a lower objective than `frozen` (0.024). Updates large enough to learn from few events cost old knowledge first. So the comparators' low scores are not all they could reach without the limit.
- **Validation goes with higher old knowledge, but that is not yet a learning result.** `life` holds 0.87 against `count`'s 0.72, on two seeds and with selected settings. Since `life` learns no rules, this may be rejection of harmful updates, or simply near-total suppression of learning. Useful and harmful promotions must be counted separately to tell.
- **The sketch does not help.** `life-sketch` trails `life` on bindings at most gaps, with half its traces given up. Its fingerprints did not offset that.
- **Self-support did not do clear harm here.** `life-selfcount` reads about the same as `life`, with noise capture 0.12 at gap 128 against 0.02 and otherwise close. The expected harm is visible only at the shortest gap, on two seeds.
- **`degraded` can stay primary.** `frozen` reads 0.90 on old knowledge across seeds (0.82, 0.97), and eleven arms meet both limits, so the concern recorded after the gates did not hold up on these two seeds.

## Diagnosis (post hoc)

`experiments/surprise_lifecycle_diagnosis.py`; `reports/surprise-lifecycle-v1-diagnosis.json`. Seeds 299 and 298, gaps 128, 512 and 2,048, q = 0.2, 64 KiB, `degraded`. The evaluator's labels read the mechanism and never reach it. Twelve configurations were traced: `life` and `life-cuegroup` at selected and starting settings, a retention variant, the sketch, and m = 2. Descriptive, and on tuning seeds. The first explanation here, from one rule at starting settings, was incomplete and is replaced by this section.

A rule is lost at different stages depending on the gap:

| stage | gap 128 | gaps 512 and 2,048 |
|---|---|---|
| rule events kept as traces | yes, at θ ≤ 0.3. At the selected θ = 0.5 seed 298 keeps only 19–21 of 32 | the same |
| a rule's traces together in the store | yes: up to 8 | **almost never**: at most 1–3 in all but one run, which reached 7 |
| a fit only on rule traces | 47 in all | **none** |
| that shadow predicts unseen items | median 0.2–0.97; better with larger m | not reached |
| validation accepts it | **1 of 47** | not reached |
| the update survives | that one ended at rule generalization 0.03 | not reached |

**1. Retention sets a limit no grouping can pass.** Fitting m traces needs them alive together, so (m − 1)·g must fit inside a trace's lifetime. With group support removed, the median lifetime of a rule trace is 260–400 events, close to the store's size over the surprise rate, so even m = 2 fails beyond a gap of about 400. **All three long gaps in the decision rule lie beyond this.** So `earns-role` is effectively unreachable at the primary cell for any arm that needs traces to meet, regardless of grouping. This is a design fault in the contract: the reach calculation asked only how long a singleton survives, not whether m traces coexist. It explains most of `life-cuegroup`'s failure.

**2. Group membership is treated as evidence, and crowds out new traces.** Eviction uses group episode counts, so joining a group buys protection before anything has been validated. At starting settings and gaps of 512 and 2,048, one residual group held nearly the whole store (up to 110 of 110 traces, 94–96% of them one-off noise on seed 299), and 98 of 126 rule traces were evicted on arrival, with median lifetime 0.

A minimal variant, `support='validated'`, counts only predictions that came true, at the same bytes and update budget. It removes this: evictions on arrival fall from 98 to 0, and median rule-trace lifetime rises from 0 to about 300. **But it does not rescue rules, because of finding 1,** and at the selected settings it lowers binding recall at gaps 512 and 2,048 (0.26, 0.34, 0.12, 0.39 to 0.08, 0.17, 0.01, 0.09). Group protection was keeping some binding traces alive by accident.

**3. Validation mostly fails to run, rather than rejecting.** Of 47 shadows fitted only on rule traces, 34 were displaced, 7 retired on damage, 5 retired on failures, and 1 was promoted. Their median life was 26–299 events, and only 17–61% of rule events were eligible for them (cue similarity to the shadow, τ_s), so few shadows ever saw a rule event they could score. Eligibility is selective: about 1–2% of old and other events qualify. Displacement is by shadows from chance groups, as the contract warned.

**4. Promotions are few, and mostly not useful.** Across all traced runs there were 46 promotions: 1 useful (rule gain of at least 0.10 on some rule), 9 harmful (known-rule accuracy down at least 0.02), the rest neither. So `life`'s higher old knowledge against `count` looks closer to suppression of learning than to selective rejection. This does not show validation can't be selective, since it had almost nothing good to judge.

**5. Slots are mostly false beyond short gaps.** Selected `life`: 52 of 70 slot writes at gap 128 came from bindings, exceptions or rare cues; 16 of 65 at 512; 11 of 104 at 2,048. The rest came from noise.

**6. The damage is partly real, and the oracle shows it.** Damage retired 7 pure shadows, at 2–20% of R-hit events. `life-oracle`'s old knowledge falls from `frozen`'s 0.82 to 0.67–0.70 on seed 299, with known rules 0.88 to 0.67–0.77 and exception families 0.98 to 0.66–0.73, and from 0.97 to 0.91–0.95 on seed 298. Its rule generalization is 0.30–0.51 against an offline ceiling of about 0.87. So the rank-one update itself leaks on these keys even with perfect sorting. Oracle feasibility is partial on both sides.

**7. The sketch matches almost everything.** At its selected radius of 4, 73–93% of kept surprises matched some fingerprint. The familiar bit then barely distinguishes traces. `life-sketch` also has half the traces.

**8. `life-cuegroup` has a further problem at gap 128.** At its selected threshold its traces rarely gather (at most 2 of a rule on one seed) and no fit is pure; at the starting threshold of 0.4 one group holds most of the store. A threshold on whole-cue similarity seems unable to isolate a family, which carries one third of the cue. That reason is inferred, not tested.

## What the pilot leaves open

- **The verdict's gaps.** Any rule win needs a gap traces can span. A revised contract must derive its gaps from coexistence, or hold evidence some other way across long gaps.
- **Selected settings at grid edges:** `life` chose the edge on `tau`, `tau_e`, `tau_s`, `kappa`, `m`, `h_b`, `h_r`, `h_f`, `S`, `delta` and `E`. The comparators chose the smallest `eta`.
- **Update allowance for `sleep-periodic`.** The contract asks the pilot to record reader-update operations. The runner records total operations only, so the matching rule cannot yet be set.
- **Gate 4.** Arms receive keys only through `step(c, o)`, and only `life-oracle` is told kinds and causes, but the promised test is not written.
- **`life-oracle` is not comparable on old knowledge and operations.** It reads 0.80 on old knowledge, below the limit, and has no operation count.
- **Cost estimate.** The contract's 1–4 CPU-hours was too low.

Any revision searched on these seeds is judged only on fresh confirmation seeds; these two seeds no longer count as independent evidence for it.

## Computation (measured costs only)

2,610 runs, 5,533 s wall, 34,672 s worker CPU (about 9.6 CPU-hours) on 11 workers, children's peak RSS 462 MB. Configurations scored per arm: `life` 86, `sleep-periodic` 30, `fifo-replay` 26, `surprise-fifo` 25, `reservoir-replay` 22, `slow` 21, `titans` 15, `titans-m0` 10, `omega` 9, `count` 5, `life-sketch` 3, `life-cuegroup` 3, `sleep-grow` 2, the rest 1. The cap of 1,500 was never reached. Timing is in `reports/surprise-lifecycle-v1-pilot-runtime.json`.
