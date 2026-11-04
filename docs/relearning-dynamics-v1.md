# Why ordinary learning temporarily destroys the interpreter memory depends on: the first relearning updates

Status: **diagnostics, 2026-11-04, seeds 300–303. Not a contract.** Follows the [relearning branches](relearning-branches-v1.md), from the same captured pre-relearning state per seed. Code: `experiments/relearning_dynamics.py`, with tables from `experiments/relearning_dynamics_summary.py`. Report: `reports/relearning-dynamics-v1.json.gz`.

**Approved scope (user, 2026-11-04):** measure first. That means logging gradients and actual updates over the first 20 relearning updates, running the two form-loss ablations, and extending the unchanged branch to 1,000 updates. The optimizer is not changed.

## Method

From the captured state, four continuations:

| Branch | What changes | Length |
|---|---|---|
| `as-run-1000` | nothing: relearning as run | 1,000 updates |
| `no-forms-20` | the delay's material, context episodes only: a reference for step sizes from the same optimizer state | 20 updates |
| `no-form-seq` | relearning as run, with the sequence-prediction rows of form episodes removed | 200 updates |
| `no-form-match` | relearning as run, with the matching rows whose query is a form removed | 200 updates |

- **The form material** in relearning is English episodes plus rehearsal of earlier scripts. Both ablations remove rows of both.
- **The ablations keep every batch draw and the original denominators.** Removed rows add zero to a mean over the same count, so the context rows keep their weight.
- **Each ablation tests that loss's contribution, not its isolated effect.** Removing a loss also changes the combined gradient and the clipping.

**Logged for each of the first 20 updates:**
- each loss's gradient norm (matching, succession, comparison, sequence), per parameter group, before clipping;
- cosines between the losses' gradients;
- the total gradient norm and the clipping factor (the norm is clipped to 5);
- the actual parameter update relative to each group's norm.

**Reads every 10 updates** are those of the branches: the relation's symmetry, discrimination, ranking and geometry; collections ↔ taps; English matching; memory-present reads.

**Reproduction.** The unchanged branch matches the earlier exact `as-run` branch in every one of its 200 updates' losses (to four decimals) and in every read at every point, in all seeds. Its state-dict comparison at update 200 was invalid: `state_dict()` returns live tensors, so the stored "update 200" weights were those of update 1,000. The code now clones them. The earlier branch report's exact state comparison stands.

Cells are the mean over seeds, with the range in brackets.

## The first 20 updates

| | `as-run` | `no-forms` (reference) | `no-form-seq` | `no-form-match` |
|---|---|---|---|---|
| Total gradient norm, update 0 | 971 (229–2,027) | 7.9 (1.7–25) | 103 (70–138) | 959 (221–2,030) |
| Total gradient norm, update 19 | 96 (45–244) | 5.7 (0.1–14) | 78 (39–126) | 62 (17–155) |
| Clipping factor, update 0 | 0.01 (0.00–0.02) | 0.80 (0.20–1.00) | 0.05 (0.04–0.07) | 0.01 (0.00–0.02) |
| Summed update / parameter norm, image projection | 0.21 (0.16–0.25) | 0.05 (0.03–0.06) | 0.13 (0.10–0.15) | 0.23 (0.16–0.26) |
| Same, core projection | 0.22 (0.18–0.26) | 0.03 (0.03–0.04) | 0.09 (0.07–0.11) | 0.25 (0.18–0.28) |
| Same, scale (inverse temperature of the sequence loss) | 0.10 | 0.00 | 0.00 | 0.11 |
| Same, comparator `g` | 0.05 (0.04–0.07) | 0.01 | 0.07 (0.06–0.08) | 0.03 (0.02–0.04) |
| Same, taps encoder | 0.02 | 0.03 | 0.02 | 0.02 |

**Per-loss gradient norms before clipping,** `as-run`, mean over updates 0–19:

| Group | Matching | Succession | Comparison | Sequence prediction |
|---|---|---|---|---|
| Image projection | 21 (16–24) | 4.0 | 0.7 | 107 (48–225) |
| Core projection | 23 (20–26) | 5.4 | 0.7 | 125 (58–264) |
| Comparator `g` | 57 (42–73) | 12 | 1.4 | 0 |
| Predictor `P` | 0 | 0 | 0 | 240 (69–606) |
| Taps encoder | 6.8 | 0.1 | 0.7 | 1.6 |

At update 0, the reference's matching and sequence gradients in the image projection are 0.5 and 1.9, against 29 and 295 in `as-run`.

**Cosines between the losses' gradients over all parameters** are near zero: −0.14 to +0.19 on average at every logged update. Within the image projection, matching against sequence prediction swings from −0.46 (update 2) to +0.47 (update 10). The losses are mostly orthogonal, not consistently opposed.

## Loss ablations: what each contributes

| Read | Branch | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|
| Collections → taps, fresh | `as-run` | 0.53 | 0.26 (0.11–0.39) | 0.42 | 0.45 | 0.67 (0.49–0.81) |
| | `no-form-seq` | 0.67 | 0.81 (0.77–0.86) | 0.67 | 0.79 | 0.86 (0.80–0.91) |
| | `no-form-match` | 0.44 | 0.29 (0.22–0.39) | 0.38 | 0.52 | 0.68 (0.56–0.81) |
| Collections → taps, stored values | `as-run` | 0.54 | 0.26 | 0.44 | 0.49 | 0.73 |
| | `no-form-seq` | 0.77 | 0.86 | 0.70 | 0.83 | 0.90 (0.81–0.97) |
| | `no-form-match` | 0.46 | 0.29 | 0.41 | 0.55 | 0.74 |
| Taps → collections | `as-run` | 0.59 | 0.50 | 0.94 | 0.96 | 0.97 |
| | `no-form-seq` | 0.79 | 0.89 | 0.97 | 0.98 | 0.99 |
| | `no-form-match` | 0.65 | 0.55 | 0.93 | 0.97 | 0.97 |
| Collection class spacing (26.7 at 0) | `as-run` | 14.6 | 8.9 | 6.5 | 6.7 | 8.2 |
| | `no-form-seq` | 24.2 | 23.5 | 23.7 | 24.1 | 24.5 |
| | `no-form-match` | 14.5 | 8.4 | 5.9 | 6.2 | 7.6 |
| Same-count collection–taps distance, over spacing (0.28 at 0) | `as-run` | 0.86 | 1.46 | 1.06 | 0.95 | 0.73 |
| | `no-form-seq` | 0.34 | 0.33 | 0.33 | 0.31 | 0.31 |
| | `no-form-match` | 0.88 | 1.50 | 1.01 | 0.88 | 0.65 |
| Symmetry: correlation of g(c, t) with g(t, c) (0.60 at 0) | `as-run` | 0.26 | 0.11 | 0.36 | 0.34 | 0.41 |
| | `no-form-seq` | −0.02 | 0.16 | 0.23 | 0.28 | 0.33 |
| | `no-form-match` | 0.66 | 0.69 | 0.63 | 0.76 | 0.80 |
| English matching (0.15 at 0) | `as-run` | 0.18 | 0.23 | 0.29 | 0.37 | 0.54 (0.12–0.74) |
| | `no-form-seq` | 0.17 | 0.22 | 0.30 | 0.44 | 0.54 (0.14–0.77) |
| | `no-form-match` | 0.16 | 0.18 | 0.19 | 0.24 | 0.33 (0.11–0.48) |
| Memory present, set A, symbol → taps (0.76 at 0) | `as-run` | 0.43 | 0.19 | 0.36 | 0.38 | 0.60 |
| | `no-form-seq` | 0.60 | 0.70 | 0.55 | 0.69 | 0.73 |
| | `no-form-match` | 0.35 | 0.23 | 0.33 | 0.48 | 0.60 |

## The long continuation

`as-run`, 1,000 updates:

| Read | 0 | 20 | 200 | 300 | 500 | 1,000 |
|---|---|---|---|---|---|---|
| Collections → taps, fresh | 0.89 | 0.26 | 0.67 | **0.78 (0.72–0.87)** | 0.70 (0.33–0.86) | 0.69 (0.39–0.82) |
| Collections → taps, stored values | 0.94 | 0.26 | 0.73 | 0.86 | 0.74 | 0.72 |
| Collections → taps, trained order | 0.94 | 0.39 | 0.85 | 0.92 | 0.91 | 0.89 |
| Taps → collections | 0.99 | 0.50 | 0.97 | 0.97 | 0.96 | 0.97 |
| Pair AUROC g(c, t) | 0.98 | 0.73 | 0.93 | 0.96 | 0.94 | 0.93 |
| Collection class spacing | 26.7 | 8.9 | 8.2 | 9.3 | 10.9 | 12.9 |
| Same-count collection–taps distance, over spacing | 0.28 | 1.46 | 0.73 | 0.61 | 0.53 | 0.51 |
| English matching | 0.15 | 0.23 | 0.54 | 0.62 | 0.69 | 0.75 (0.44–0.86) |
| Memory present, set A, symbol → taps | 0.76 | 0.19 | 0.60 | 0.69 | 0.62 | 0.59 |

Per seed, (English matching, collections → taps) at updates 200, 300, 500 and 1,000:

| Seed | 200 | 300 | 500 | 1,000 |
|---|---|---|---|---|
| 300 | (0.74, 0.81) | (0.80, 0.80) | (0.81, 0.83) | (0.86, 0.81) |
| 301 | (0.63, 0.49) | (0.70, 0.72) | (0.80, 0.33) | (0.84, 0.39) |
| 302 | (0.67, 0.67) | (0.75, 0.74) | (0.81, 0.80) | (0.84, 0.82) |
| 303 | (0.12, 0.71) | (0.21, 0.87) | (0.34, 0.86) | (0.44, 0.76) |

## What this shows

- **The first relearning updates are far outside the delay's regime.**
  - The total gradient norm is 229–2,027 at update 0, against 1.7–25 for the same optimizer state on delay material. Clipping scales it by about 0.01.
  - Over 20 updates the image and core projections move 0.21–0.22 of their norm, against 0.03–0.05 in the reference: 4–7 times more, from the same optimizer state.
  - These are measured step sizes. Whether Adam's history makes them larger than a fresh optimizer would is not measured here, so it remains a hypothesis.
- **Before clipping, the form sequence-prediction loss supplies most of the gradient to the shared projections.**
  - In the image and core projections it is about five times the matching gradient.
  - It is the only loss reaching the predictor and the scale.
  - Clipping rescales the whole gradient, so the step's direction in the projections is mostly the sequence loss's.
  - The losses' gradients are mostly orthogonal. This is not two objectives pulling against each other, but one dominating.
- **Removing the form sequence-prediction rows removes most of the disruption.**
  - The code space keeps its spacing (24.5 at 200, against 26.7) and the senses stay aligned (0.31, against 0.28).
  - Collections → taps dips only to 0.67 at 10 updates and ends at 0.86, with stored values at 0.90.
  - Memory-present symbol → taps ends at 0.73, against 0.60.
  - English reacquisition is unchanged at 200 updates (0.54 against 0.54).
  - The removal also lowers the gradient tenfold and the clipping from 0.01 to 0.05, so this shows the loss's contribution under these conditions, not its effect alone.
- **Not everything is preserved without the sequence rows.** Symmetry across all pairs still falls (correlation 0.33 at 200, against 0.60). Same-count pairs keep theirs (violation 0.15, against 0.10). The matching updates on forms affect the relation too, more mildly.
- **Removing the form matching rows does not prevent the disruption, and slows reacquisition.**
  - The disruption is the same as in `as-run`: spacing 8.4 and alignment 1.50 at 20 updates; collections → taps 0.29.
  - English reaches only 0.33 by 200.
  - Symmetry is best kept here (0.80).
  - Form matching drives reacquisition of English and carries part of the symmetry loss. The sequence-prediction rows drive the collapse of the shared code space.
- **Recovery is incomplete and not monotone.**
  - Collections → taps peaks around 300 updates (0.78; stored values 0.86) and then declines to 0.69–0.72 by 1,000, below its pre-relearning 0.89–0.94.
  - The trained order (0.89–0.93) and taps → collections (0.97) stay high.
  - Alignment improves steadily to 0.51 of the spacing, but not back to 0.28.
  - Recovery does not follow English reacquisition. Seed 301 relearns English to 0.84 while collections → taps falls to 0.39. Seed 303 relearns English slowly (0.44) while collections → taps reaches 0.76–0.87.
  - So the limitation persists within 1,000 updates. It is not simply recovery that has not finished, and it is not tied to how well the new material is learned.

**Why ordinary online learning temporarily destroys the interpreter memory depends on (so far):**
- The reintroduced form material arrives confidently wrong.
- Its sequence-prediction loss produces gradients two orders of magnitude above the delay's.
- Clipped and passed through Adam, these produce several times the usual movement of the projections that every image shares, mostly along the sequence loss's direction.
- The shared code space collapses and the two senses separate.
- Continued training re-establishes what it trains directly: taps → collections and the trained order. It does not fully restore the alignment that made the reverse read work. That read peaks and then sags while new learning continues.

## What this does not establish

- **Whether Adam's history enlarges the steps.** The reference shows the steps are larger, not why. A fresh-moment or lower-step continuation would test it; it is not run, as approved.
- **Why sequence prediction on forms moves the projections so strongly.** Its targets are collections, which the forgotten forms' predictions miss confidently. Its inverse temperature (`scale`) also changes.
- **English separately from rehearsed scripts.** Both ablations remove rows of both.
- **Why the reverse read sags after 300 updates** while alignment keeps improving.

## Computation

- **GPU** (GTX 1650), three workers:
  - capture 155–205 s per seed, as in the branches;
  - four continuations per seed: 1,000, 20, 200 and 200 updates, with reads every 10 and 20 logged updates;
  - 10 min 4 s wall, 1326 s aggregate worker CPU, peak RSS 3.6 GB.
