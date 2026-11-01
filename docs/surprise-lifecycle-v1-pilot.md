# Surprise lifecycle v1: pilot (seeds 299 and 298)

Status: **pilot, 2026-11-01. Not evidence for any contrast**: it selected settings on the seeds it reports, and it decides whether the [contract](surprise-lifecycle-v1-contract.md) can be frozen. Outcome: **no.** Gate 6 passes, but `life` learns no rules, for reasons in the mechanism's design. `reports/surprise-lifecycle-v1-selection.json`; `experiments/surprise_lifecycle.py pilot`.

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

Bold old knowledge is below the provisional limit of 0.85. Chance is 0.03. `frozen`'s binding recall is above chance because a known family's rule sometimes gives the right outcome.

## What this shows

- **Gate 6 passes.** `life-oracle` exceeds the best comparator's objective by 0.61, 0.55 and 0.58 at the long gaps. Rule generalization of 0.33–0.48 from its first 4 events per rule shows that rules can be learned in this world, with sorting supplied.
- **Nothing without the oracle learns rules.** The largest value in any non-oracle arm at any gap is 0.07.
- **`life` is ahead on bindings at gap 2,048 only.** It reads 0.25 there against at most 0.05 for the comparators, on seeds differing widely (0.12, 0.39). At 8,192 it reads 0.08 and the best comparator 0.06; at 32,768, 0.03 and 0.06. By the frozen rule's shape it would read as `no-advantage-shown`, which needs two long gaps.
- **Comparators selected near doing nothing.** Most chose the smallest η on the grid, and several have a lower objective than `frozen` (0.024). Updates large enough to learn from few events cost old knowledge first. So the comparators' low scores are not all they could reach without the limit.
- **Later-experience validation protects old knowledge.** `life` holds 0.87 against `count`'s 0.72, on two seeds and with selected settings: the one contrast here that points the way the proposal predicted.
- **The sketch does not help.** `life-sketch` trails `life` on bindings at most gaps, with half its traces given up. Its fingerprints did not offset that.
- **Self-support did not do clear harm here.** `life-selfcount` reads about the same as `life`, with noise capture 0.12 at gap 128 against 0.02 and otherwise close. The expected harm is visible only at the shortest gap, on two seeds.
- **`degraded` can stay primary.** `frozen` reads 0.90 on old knowledge across seeds (0.82, 0.97), and eleven arms meet both limits, so the concern recorded after the gates did not hold up on these two seeds.

## Why `life` learns no rules

Post hoc, tracing one rule through three streams at the search's starting settings. Findings 1 and 2 are read from the trace; the cause in 2 is inferred from the eviction rule.

1. **Residual groups form by outcome, not by rule.** Rule events are surprising, and their residuals agree (cosine 0.85–0.97, noise about 0.03). But a residual is dominated by its outcome, so any event with that outcome joins, and one group reached 16 traces at gap 128 for a rule with 10 events in the stream. A shadow fits the group's oldest m traces, so its cue direction rarely matches the family (cosine to rule cues mostly below 0.3), and it retires. `kappa` does not stop this. Retired fit traces are consumed, so the few real traces are used up.
2. **Those groups may hold the store.** Their traces count as supported, so they outlast new singletons, and several rule events were evicted on arrival. That departs from the FIFO assumption behind the reach figures.

`life-cuegroup` also learns no rules, though cue-similar groups should find a family. Its failure has not been traced.

## What the pilot leaves open

- **Selected settings at grid edges:** `life` chose the edge on `tau`, `tau_e`, `tau_s`, `kappa`, `m`, `h_b`, `h_r`, `h_f`, `S`, `delta` and `E`. The comparators chose the smallest `eta`.
- **Update allowance for `sleep-periodic`.** The contract asks the pilot to record reader-update operations. The runner records total operations only, so the matching rule cannot yet be set.
- **Gate 4.** Arms receive keys only through `step(c, o)`, and only `life-oracle` is told kinds and causes, but the promised test is not written.
- **`life-oracle` is not comparable on old knowledge and operations.** It reads 0.80 on old knowledge, below the limit, and has no operation count.
- **Cost estimate.** The contract's 1–4 CPU-hours was too low.

Any revision searched on these seeds is judged only on fresh confirmation seeds; these two seeds no longer count as independent evidence for it.

## Computation (measured costs only)

2,610 runs, 5,533 s wall, 34,672 s worker CPU (about 9.6 CPU-hours) on 11 workers, children's peak RSS 462 MB. Configurations scored per arm: `life` 86, `sleep-periodic` 30, `fifo-replay` 26, `surprise-fifo` 25, `reservoir-replay` 22, `slow` 21, `titans` 15, `titans-m0` 10, `omega` 9, `count` 5, `life-sketch` 3, `life-cuegroup` 3, `sleep-grow` 2, the rest 1. The cap of 1,500 was never reached. Timing is in `reports/surprise-lifecycle-v1-pilot-runtime.json`.
