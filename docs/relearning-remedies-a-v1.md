# Remedies, stage A: four sequence-loss arms through relearning

Status: **screen, 2026-11-04, seeds 300–303. Not a contract.** Plan: [relearning-remedies-plan.md](relearning-remedies-plan.md), revised after review before this run. Code: `experiments/relearning_remedies.py`, with tables from `experiments/relearning_remedies_summary.py`. Report: `reports/relearning-remedies-a-v1.json.gz`.

## Setup

- **Starting point:** the captured pre-relearning state per seed. The captured weights equal step 8's delay checkpoint exactly in all seeds.
- **Training:** 200 relearning updates per arm, with only the sequence-prediction loss changed.
- **Arms:**
  - `unchanged`;
  - `seq-cand-stop`: gradient into sequence prediction's candidates stopped, on all sequence rows;
  - `seq-relative`: batch-relative distance logits, with s₀ = 361–656 per seed and ε = 10⁻⁶·s₀;
  - `seq-weight-0.1`.
- **Reads every 10 updates,** on identical pairs and candidates:
  - the interpreter;
  - memory-present reads;
  - English;
  - sequence prediction and composition (`sequence_pilot.composition`), by distance among nine collections, k = 1–3 steps.
- **Outcomes:** each is the worst transient (the minimum over updates 0–200) and the level at update 200, both as changes from update 0. Cells are the mean, with the range over seeds.

**Limitation:** the core was trained with the original loss. The screen shows only how each loss behaves at this transition.

## Results

**The interpreter and memory:**

| Measure (at 0) | Arm | Worst transient | At 200 |
|---|---|---|---|
| Collections → taps, fresh (0.89) | `unchanged` | −0.67 (−0.79 to −0.54) | −0.21 (−0.41 to −0.05) |
| | `seq-cand-stop` | **−0.29** (−0.41 to −0.19) | **−0.13** (−0.34 to +0.03) |
| | `seq-relative` | −0.36 (−0.67 to −0.15) | −0.25 (−0.62 to 0.00) |
| | `seq-weight-0.1` | −0.53 (−0.77 to −0.28) | −0.22 (−0.38 to −0.02) |
| Collections → taps, stored values (0.94) | `unchanged` | −0.71 | −0.21 |
| | `seq-cand-stop` | **−0.31** | **−0.17** |
| | `seq-relative` | −0.40 | −0.27 |
| | `seq-weight-0.1` | −0.56 | −0.23 |
| Taps → collections (0.99) | `unchanged` | −0.54 | −0.02 |
| | `seq-cand-stop` | **−0.23** | −0.04 |
| | `seq-relative` | −0.31 | −0.07 |
| | `seq-weight-0.1` | −0.32 | −0.01 |
| Memory present, set A, symbol → taps (0.76) | `unchanged` | −0.59 | −0.16 |
| | `seq-cand-stop` | **−0.25** | **−0.10** |
| | `seq-relative` | −0.31 | −0.19 |
| | `seq-weight-0.1` | −0.46 | −0.17 |
| Memory present, set A, taps → symbol (0.81) | `unchanged` | −0.41 | −0.01 |
| | `seq-cand-stop` | −0.16 | −0.03 |
| | `seq-relative` | −0.23 | −0.03 |
| | `seq-weight-0.1` | −0.21 | 0.00 |

Set B follows set A.

**Geometry,** at update 200, against 26.7 spacing and 0.28 alignment at update 0:
- `unchanged`: spacing 8.2, same-count collection–taps distance over spacing 0.73.
- `seq-cand-stop`: 24.0 and 0.29.
- `seq-relative`: 37.0 and 0.42. The codes **expand**: the training rows' candidate spread rises from 478 to 1,455.
- `seq-weight-0.1`: 9.9 and 0.79.

**Acquisition and the abilities the loss teaches:**

| Measure (at 0) | `unchanged` | `seq-cand-stop` | `seq-relative` | `seq-weight-0.1` |
|---|---|---|---|---|
| English matching, at 200 (0.15) | +0.39 | +0.36 | +0.27 | +0.42 |
| Sequence, English forms, 1 step up, at 200 (0.16) | +0.40 | +0.41 | +0.26 | +0.32 |
| Sequence, collections, 1 step up: worst / at 200 (0.92) | −0.45 / −0.06 | **−0.18** / −0.08 | −0.28 / −0.11 | −0.26 / −0.07 |
| Sequence, collections, 1 step down: worst / at 200 (0.91) | −0.54 / −0.16 | −0.34 / **−0.08** | −0.38 / −0.15 | −0.32 / −0.18 |
| Sequence, taps, 1 step up: worst / at 200 (0.94) | −0.43 / −0.02 | **−0.15** / −0.03 | −0.33 / −0.12 | −0.27 / −0.08 |
| Sequence, Devanagari forms, 1 step up, at 200 (0.59) | +0.18 | +0.09 | +0.05 | +0.14 |
| Composition, collections, 2 steps up: worst / at 200 (0.71) | −0.38 / +0.06 | −0.35 / **−0.11** | −0.30 / −0.06 | −0.30 / +0.08 |
| Composition, collections, 3 steps up: worst / at 200 (0.62) | −0.44 / −0.01 | −0.40 / **−0.09** | −0.38 / −0.11 | −0.31 / +0.13 |
| Composition, taps, 2 steps up: worst / at 200 (0.76) | −0.35 / +0.13 | −0.27 / **−0.05** | −0.45 / −0.11 | −0.30 / +0.06 |
| Sequence loss as trained, update 199 | 0.86 | 3.30 | 3.02 | 0.10 (×0.1 weight) |

## What this shows

- **No arm passes every stage A criterion.** None protects the interpreter while keeping sequence prediction and composition at least as good as `unchanged`.
- **`seq-cand-stop` protects the interpreter most, and keeps the geometry.**
  - It halves the worst transient on collections → taps (−0.29 against −0.67), on stored values and on memory-present reads.
  - It is better at the deadline (−0.13 against −0.21).
  - It keeps the spacing and the senses' alignment almost unchanged, and English relearns about as fast (+0.36 against +0.39).
  - One-step sequence prediction holds up, transiently better.
  - But multi-step composition at the deadline is worse than `unchanged`: −0.11 against +0.06 for collections two steps up, −0.09 against −0.01 three steps, and −0.05 against +0.13 for taps. The trained sequence loss stays higher (3.3 against 0.9), and Devanagari form prediction gains less.
  - Under the plan's rule it does not pass. It protects interpretation partly at the expense of what the sequence loss teaches.
- **`seq-relative` fails, in the way the review anticipated.**
  - It removes the contraction but creates the opposite incentive: the codes expand, with candidate spread growing threefold, to raise confidence through the denominator.
  - Its early reads are best (collections → taps 0.85 and 0.80 at updates 10 and 20), but it degrades steadily to 0.64 at 200, below `unchanged`.
  - It slows English (+0.27) and weakens sequence prediction and composition.
- **Lowering the weight does not prevent the collapse.** At weight 0.1, the code space still contracts (spacing 9.9) and the senses still separate (0.79). The interpreter's worst transient is smaller (−0.53) only because the collapse is slower. At the deadline it matches `unchanged`.
  - Its sequence and composition reads are the best of the four (composition three steps +0.13), and English the fastest (+0.42).
  - **So magnitude alone is not the cause.** The direction of the candidate-side pull persists at a tenth of the weight.

## Interpretation

The candidate-side gradient of sequence prediction does two things at once:
- it causes the early collapse;
- it is also part of how collection codes come to support multi-step prediction.

Stopping it entirely protects the interpreter and costs composition. Rescaling the distances moves the incentive from contraction to expansion. Reducing the weight slows the collapse without changing its direction.

## Stage B

**Not started:** no arm meets its condition.

`seq-cand-stop` is the closest. Its failure is specific: candidates no longer learn from sequence structure. That suggests, for a next screen, arms that keep a weaker or slower candidate path rather than none. Examples:
- the candidate-side gradient scaled down;
- candidates from a slowly moving copy of the core, the mean-teacher form of the same idea.

These are proposals for review, not run.

A constrained update from the [loss analysis](relearning-loss-analysis-v1.md) was tested afterwards and failed ([audit and continuation](relearning-constrained-audit-v1.md)).

## Computation

- **GPU** (GTX 1650), three workers.
- **Capture:** about 170–230 s per seed.
- **Arms:** four of 200 updates with 21 read points each, including composition.
- **Totals:** 9 min 26 s wall, peak RSS 3.7 GB.
