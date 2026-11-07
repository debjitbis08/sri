# Everyday evidence transfer v1: results

Run 2026-11-07 under the [frozen contract](everyday-evidence-transfer-v1.md). All 12 fits completed, four seeds
(450–453), three neural arms; the ridge baseline has no meta-training. Final 2,000-update checkpoints were used.
No rules, proofs, task definitions, relevant-attribute flags or concept identities were inputs to the learner.

## What worked and what did not

**Examples changed predictions appropriately, but the neural adapter did not meet the acquisition or exception criteria.**
Using the existing reader’s facts, eight examples gave 70.1% balanced accuracy on new combinations, against 50.0%
without examples and 50.2% with shuffled labels. Reversing all example labels preserved performance at 69.9% on the
reversed target. This supports evidence-dependent prediction, not merely a changed acceptance threshold.

The inexpensive ridge baseline matched ordinary transfer (70.9%) and accepted explicitly taught exceptions much
more reliably (99.3% versus 44.0%). The neural adapter was better under noisy or missing evidence. Neither met the
75% primary acquisition threshold; no method dominates the entire test suite. This run does not justify promoting
the neural adapter into the main architecture or growing a proof engine.

## Data and scope

- 34 animals and 31 existing attributes; 26 animals used in adapter practice, eight excluded from every practice context and target.
- 396 eligible concepts after removing equivalent extensions and complements: 176 training, 76 novel-combination tests, 144 held-out-structure concepts, of which 100 were tested.
- The novel-combination quota of 100 has a shortfall of 24; all 76 available concepts were used. No performance-dependent resampling.
- Four contexts per tested concept; 304 novel-combination and 400 held-out-structure episodes per seed. Test questions concern the eight excluded animals.
- The old reader already knows these entities. This tests new use of old knowledge, not acquisition of an entirely unseen entity.
- Entity codes themselves already contain learned knowledge. The codes arm is an explicit-profile ablation, not a knowledge-free control.
- These are arbitrary new groupings/preferences over a small curated factual fixture. This is not field-data validation, natural-language understanding, or proof of general reasoning.

## Acquisition from examples

Balanced accuracy, mean across seeds and context-level class recalls. Seed ranges are descriptive, not confidence intervals.
Examples and concepts overlap within the fixed fixture; they are not independent population samples.

| Input / method | New combinations, 0 examples | 4 examples | 8 examples | Seed range at 8 | Held-out structure at 8 |
|---|---|---|---|---|---|
| codes | 50.0% | 63.0% | 65.2% | 59.8%–70.1% | 58.4% |
| learned | 50.0% | 66.4% | 70.1% | 67.0%–75.1% | 64.0% |
| oracle | 50.0% | 69.7% | 74.2% | 73.9%–74.8% | 65.9% |
| ridge | 50.0% | 66.2% | 70.9% | 67.7%–75.6% | 65.3% |

The explicit factual profile adds **4.93 percentage points** over entity codes. This falls just below the predeclared
five-point knowledge-use criterion; it must not be rounded into a pass. Oracle facts raise performance to 74.2%,
but the same learner still misses the 75% acquisition criterion. The oracle arm is not a Bayes-optimal ceiling:
limited examples can leave several plausible groupings, and the trained predictor can also make errors.

## Following evidence and handling missing information

New combinations; eight examples. Noisy means two labels flipped. Missing means approximately half the source
profile entries removed with an explicit mask; entity codes remain available.

| Method | Clean | Shuffled labels | Reversed labels / target | Noisy labels | Missing source entries | Clean Brier score |
|---|---|---|---|---|---|---|
| codes | 65.2% | 50.2% | 65.4% | 60.8% | 65.2% | 0.217 |
| learned | 70.1% | 50.2% | 69.9% | 63.8% | 64.3% | 0.202 |
| oracle | 74.2% | 49.9% | 74.1% | 66.3% | 68.2% | 0.184 |
| ridge | 70.9% | 50.0% | 70.9% | 58.4% | 53.3% | 0.193 |

The learned-profile neural arm exceeds ridge by 5.4 points with noisy labels and 11.0 with missing profiles, while
ridge has slightly higher clean accuracy and lower clean Brier score. These are suite-level predictive comparisons,
not measured deployment-latency or calibrated-uncertainty claims. The contract’s reversed-label accuracy target of
75% is missed even though the learner clearly responds to reversing its evidence.

## Explicit exceptions and conflicting reports

One held-out animal is explicitly assigned the opposite of its general grouping. Its query intentionally overlaps
with that newly supplied observation; this is evidence uptake, not unseen-query generalization. Other animals are
still tested against their original labels.

| Method | Exception accuracy before | After teaching | Other-query BA drop | Conflict confidence |
|---|---|---|---|---|
| codes | 37.3% | 44.6% | -0.0% | 60.8% |
| learned | 30.0% | 44.0% | -0.1% | 62.6% |
| oracle | 26.1% | 39.2% | 0.6% | 63.8% |
| ridge | 28.9% | 99.3% | 0.9% | 55.1% |

Ridge passes the exception criterion in both test groups (99.3% on new combinations, 99.5% on held-out structure),
with under one percentage point ordinary-query BA loss on average. All neural arms fail. Low collateral change
alone is not success: the neural learner often preserves its previous grouping instead of accepting the exception.

Conflicting observations supply both labels for the same entity. There is no scored truth label for that entity.
The learned arm’s mean confidence falls from 68.4% before the conflict to 62.6%; ridge falls from 67.6% to 55.1%.
These are descriptive sensitivity measures, not proof of a principled unknown/both belief state.

### Post hoc practice-distribution audit

After observing the exception failure, a read-only diagnostic tested every final checkpoint on 1,024 fresh
practice-distribution episodes, using a new RNG. No parameters or selected checkpoints changed. Context-size
stratification was added in a second audit pass to avoid mixing zero-example and eight-example cases; both costs
are retained. Practice has 10% context-label noise and familiar concepts/entities, unlike the clean held-out test,
so the comparison is diagnostic rather than a fully isolated causal experiment.

| Neural arm | Practice ordinary accuracy, k8 | Practice exception accuracy, k8 | Held-out new-combination exception accuracy |
|---|---|---|---|
| codes | 71.3% | 53.0% | 44.6% |
| learned | 71.8% | 57.3% | 44.0% |
| oracle | 71.6% | 54.2% | 39.2% |

Exception uptake is weak even in the practiced setting, so unfamiliar animals alone cannot explain the failure.
Possible causes include underfitting specific observations, their small share of training targets, and imperfect
attention/retrieval. This run does not separate those mechanisms. It does show that the same representations support
much stronger explicit-evidence uptake with a simple local fit.

## Checkpoints and retention

New-combination balanced accuracy with eight examples; no best-checkpoint selection.

| Neural arm | 500 | 1,000 | 2,000 |
|---|---|---|---|
| codes | 62.4% | 64.1% | 65.2% |
| learned | 68.8% | 67.1% | 70.1% |
| oracle | 72.4% | 71.6% | 74.2% |

All frozen-reader state hashes remained unchanged; all 1,054 probed old factual scores per fit were exactly
unchanged. Previously stored context predictions were identical after reading another context. This follows from
frozen weights and separate example memories; it does not establish retention under shared-weight consolidation.

## Computation

Per-fit means, process CPU seconds. Sampling is separated from parameter optimization. Final evaluation includes
all nine conditions; checkpoint evaluation is additional and counted in whole-fit totals.
The learned arm's whole-fit total also includes running the ridge reference once; that cost is itemized below.

| Neural arm | Construction | Episode sampling | Optimization | Final evaluation | Whole fit | Peak RSS range, MiB |
|---|---|---|---|---|---|---|
| codes | 2.84 | 5.21 | 7.29 | 0.43 | 15.95 | 501.1–503.4 |
| learned | 2.69 | 5.18 | 7.31 | 0.43 | 16.24 | 505.0–507.6 |
| oracle | 2.67 | 5.19 | 7.31 | 0.43 | 15.78 | 507.7–510.5 |

- Main: **191.86 worker CPU s**, **64.72 wall s**, three workers. This was substantially below the conservative 5–15-minute estimate.
- Smoke: 9.84 worker CPU s, 3.43 wall s; three 50-update fits. No tuning followed.
- Both mechanical-verification passes and both post hoc audit passes: 3.08 CPU s total, separately recorded. Report-generation/integrity checks have their own runtime file.
- Neural parameters: **60,097** shared across all new properties (240,388 float32 parameter bytes); Adam moments 480,776 bytes; gradients another 240,388 bytes.
- Shared frozen profile: 12,784 bytes per arm. An eight-example new-property memory needs 96 bytes of int64 entity IDs and float32 labels, before container overhead. Reader weights are counted separately in runtime JSON.
- Ridge: no meta-training or retained learned shared parameters; 5,632 small context fits per seed across all conditions, 22,528 total. Fit plus prediction evaluation averaged **0.444 CPU s per seed**, comparable here to neural final evaluation (0.429 s), before neural training cost.
- Each neural fit used 64,000 sampled episodes from 176 fixed concepts. Distinct episode digests: 63,980 / 63,982 / 63,985 / 63,981 across seeds; contexts/targets overlap, so these are not independent concepts or facts.
- Sampling, examples, targets, query counts, model/buffer/reader/checkpoint bytes and hashes are in runtime JSON. All paired arms used identical episode digests and exposures within a seed.
- Peak RSS is per process, not whole-job memory. CPU totals exclude unmeasured parent orchestration and research/editing time. No standalone inference-latency or scaling claim.

## Decision

Keep this as a controlled result about an example-based learner. The encouraging result is evidence-dependent
generalization using existing knowledge. The missing capability is reliable application of a specific new
observation, especially an exception, alongside broader statistical patterns.

The next design should preserve the factual reader and compare generic local adaptation/example memory with the
neural conditional predictor, retaining noise/missing-evidence tests. Ridge is a strong cheap reference, not a
problem-specific rule dispatcher. More machinery should earn its place against that reference; a formal proof
checker is not the next target. This experiment did not change the main architecture or deploy the new component.

## Artifacts and validation

- Contract SHA-256: `4df325b8ecd6696ddad85b7bedd57b213fa81e26230849259d410c8aa677acf2`.
- Revision: `33731d90f6f28cd6c4d54ac3a3b20d353c1b7b9e` plus the recorded uncommitted source hashes.
- Runner: `experiments/everyday_evidence_transfer.py`; post hoc audit: `experiments/everyday_evidence_transfer_audit.py`.
- Scientific results and summary: `reports/everyday-evidence-transfer-v1{,-summary}.json`.
- Main/smoke/verification/audit computation is in separate `-runtime.json` files; checkpoints and durable records are under `runs/everyday-evidence-transfer-v1/`.
- Mechanical checks: extension/complement deduplication, entity/task separation, permutation invariance of contexts, empty/masked context behavior, finite gradients, label-reversal and conflict targets.
- All 12 saved checkpoint hashes, frozen contract hash and main source hashes verified. All 12 paired-stream and retention checks passed.
- The frozen contract and earlier experiment artifacts were not rewritten.
