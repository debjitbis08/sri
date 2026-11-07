# Rule critic binding v1: results

Run 2026-11-07 under the [frozen contract](rule-critic-binding-v1.md). Four seeds (450–453), three arms,
3,000 updates per fit, identical training examples/exposures within each seed. Direct proof supervision only.
Means below; seed ranges are descriptive, not confidence intervals. Lengths share the same local transitions
and are not independent evidence for general reasoning. No checkpoint or hyperparameter selection.

Contract SHA-256: `a1a42bacb366da09e9d7ecd9c07b2150c41e20040120cb94a3368d3bbd07c415`. Revision `33731d90f6f28cd6c4d54ac3a3b20d353c1b7b9e` plus the uncommitted sources
identified by hashes in the runtime report. Historical experiment files were not changed.

## Interpretation

**Matching repairs valid-proof acceptance on this fragment, but does not produce a reliable proof checker.**
All four matching seeds accept every valid program at every tested length, including held-out relations. However,
38.9% of single-relation substitutions and 57.3% of nearest-code substitutions pass. The predeclared success gate
therefore fails. The increased acceptance is not a usable overall improvement by the contract's criterion.

- Canonicalization removes variable-renaming errors by construction, but held-out-relation valid acceptance
  falls from 54.0% to 44.9%. Variable naming was not the only difficulty.
- Relative matching makes the right relation usable across all held-out relations. This does not establish exact
  identity discrimination: wrong relations can still pass, especially in the interior of a path.
- Of 4,444 accepted negatives in the matching arm's standard program reads, 4,387 (98.7%) first violate the ideal
  check at an internal transition. These are counts over sampled programs/lengths, not independent examples.
- Seed 453 is particularly weak: 64.9% relation substitutions, 34.5% single-direction reversals and 97.2%
  nearest-code substitutions pass. The other seeds' nearest-code acceptance is still 40.1–49.1%. No seed passes
  the full criterion. All seeds, including 453, are retained.
- Matching has fewer parameters and somewhat lower training cost here, but much more expensive full-table
  evaluation. It is not an established efficiency win or a deployment recommendation.

The next diagnostic should make the **internal checks** identifiable, while retaining the matching representation
as a candidate. A direct local-transition supervision control can establish whether it can reject the wrong
relation at every position. A sequence-label alternative needs distinguishing examples with correct endpoints and
wrong interiors, beginning with three-hop negatives; that is a separate experiment, not a change made after this run.
Answer-only training and walker expansion remain premature.

## Proof acceptance and generalization

| Arm | Held-out relations, length 1 | Lengths 3–12 | Seed range, length 3 | Held-out instances, length 3 | Mean success gate |
|---|---|---|---|---|---|
| canonical | 47.8% | 44.9% | 22.1%–70.6% | 88.8% | Fail |
| matching | 100.0% | 100.0% | 100.0%–100.0% | 100.0% | Fail |
| original | 54.8% | 54.0% | 27.9%–88.2% | 88.0% | Fail |

## Invalid programs accepted

Each category is averaged over lengths 2–12; lower is better. Nearest-code substitutions test every position.

| Arm | Reverse one step | Replace one step | Splice | Wrong orientation | Random | Nearest-code replacement |
|---|---|---|---|---|---|---|
| canonical | 11.9% | 15.0% | 8.5% | 0.0% | 0.4% | 28.8% |
| matching | 8.6% | 38.9% | 1.8% | 0.0% | 0.0% | 57.3% |
| original | 10.6% | 17.4% | 8.8% | 0.0% | 0.3% | 37.7% |

## Invariance

Share of verdicts changed on held-out relations; lower is better.

| Arm | Fresh derived symbol | Variable renaming | Rule order |
|---|---|---|---|
| canonical | 0.3% | 0.0% | 0.0% |
| matching | 0.0% | 0.0% | 0.0% |
| original | 0.5% | 15.4% | 0.0% |

Canonicalization enforces variable-renaming invariance by construction. The matching arm also removes raw
derived-symbol identity from this single-derived-symbol fragment. These are designed invariances, not learned discoveries.

## Checkpoints

Held-out-relation valid acceptance at length 3 (the longer repeated chains use the same checks).

| Arm | 500 | 1,000 | 2,000 | 3,000 |
|---|---|---|---|---|
| canonical | 69.9% | 61.0% | 40.1% | 44.9% |
| matching | 100.0% | 100.0% | 100.0% | 100.0% |
| original | 70.6% | 58.5% | 54.0% | 54.0% |

## Post hoc analytical control: short sequence labels leave the interior ambiguous

This control was conceived after inspecting the first matching seed; it is **not a preregistered result or an extra
training arm**. It establishes an ambiguity in the data, not the exact internal algorithm learned by the networks.

Consider a pairwise critic that permits START→R and R→END only, but permits every internal transition. Its language
is all paths that start and end with R. On lengths 1 and 2, this accepts exactly R and RR, so it agrees with every
valid/invalid training label. At length 3 it incorrectly accepts R S R. There is no sequence-label loss at lengths
1–2 that distinguishes this critic from the intended R-only checker.

An exhaustive check over the three-symbol alphabet {R, S, reversed R} confirms agreement on all 12 programs of
length 1–2 and disagreement on R S R. The argument holds for any alphabet. Artifacts:
`reports/rule-critic-binding-v1-identifiability.json` and its separate runtime file (no fits).

Thus even direct **whole-program** labels do not necessarily supervise each local decision. This qualifies the
earlier grammatical-inference intuition: the local checker can represent R*, but the chosen short training strings
do not uniquely identify it. The observed internal-substitution failures are consistent with this ambiguity.
They should not be attributed solely to similar relation codes or insufficient training time.

## Computation

Per-fit means, CPU seconds; evaluation includes checkpoint, final and symbolic-reference reads.

| Arm | Parameters | Parameter bytes | Training CPU s | Evaluation CPU s | Whole fit CPU s | Peak process RSS range, MiB |
|---|---|---|---|---|---|---|
| canonical | 33,849 | 135,396 | 70.7 | 36.4 | 109.8 | 501.6–504.0 |
| matching | 13,697 | 54,788 | 58.3 | 183.8 | 245.1 | 502.4–504.6 |
| original | 33,849 | 135,396 | 66.4 | 37.3 | 106.7 | 501.6–502.9 |

- Main: 1846.5 worker CPU s (0.51 CPU h); 645.3 wall s (10.8 min), three workers.
- Smoke: 168.9 worker CPU s, 92.6 wall s, three 100-update fits. No tuning followed.
- Mechanical verification: 4.19 CPU s; postprocessing cost is recorded separately.
- The post hoc analytical control has its own runtime file and used no training or model selection.
- Each main fit used 192,000 program exposures; distinct (rule-set, program) examples numbered
  62,234 / 62,219 / 62,486 / 62,375 for seeds 450 / 451 / 452 / 453, identical across arms.
- Aggregates exclude unmeasured parent orchestration CPU and research/editing time. Peak RSS is per process, not a whole-job sum.
- Exact optimizer/buffer/reader/checkpoint bytes, distinct examples, exposures and local-check counts are in the runtime JSON.
- Checkpoints are measured, not selected. This compares equal data and updates, not equal compute. Full-table evaluation is not a deployment-latency benchmark.

## Validation and artifacts

- All four original-arm final program reads exactly reproduce the previous diagnostic’s direct arm.
- All paired arms have identical distinct training-example counts and exposures for each seed.
- Symbolic reference: all valid programs accepted, all random/hard/nearest-code invalid programs rejected, all invariance checks unchanged.
- Mechanical verification checked canonicalization, rule-order/fresh-symbol invariance, independence from answer-bearing metadata, score agreement and finite gradients.
- All 12 checkpoint hashes, frozen contract hash and recorded source hashes verified after execution.

- Runner: `experiments/rule_critic_binding.py`.
- Scientific results: `reports/rule-critic-binding-v1.json`.
- Seed summaries: `reports/rule-critic-binding-v1-summary.json`.
- Main/smoke/verification/postprocessing computation: matching `-runtime.json` files.
- Checkpoints: `runs/rule-critic-binding-v1/`; durable per-fit records in `done/`.
