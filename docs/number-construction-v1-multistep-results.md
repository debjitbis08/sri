# Number construction v1: multi-step teaching

Protocol: [number construction v1 spec, "Multi-step teaching"](number-construction-v1-spec.md).
Follows the [composition test and its comparison-direction diagnostic](number-construction-v1-composition-results.md),
which found that forward composition falls short because a learned "add one" step covers
only part of a real step. Three seeds; **a pilot, not headline evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_multistep pilot` (24 minutes);
  - `NCC_RUNS=number-construction-v1-multistep-pilot NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_composition pilot`;
  - `NCC_RUNS=number-construction-v1-multistep-pilot NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_projection pilot`
    (under a minute each).
- **Artifacts:** `reports/number-construction-v1-multistep-pilot-results.json.gz`,
  `reports/number-construction-v1-composition-pilot-multistep-pilot-results.json.gz`,
  `reports/number-construction-v1-projection-pilot-multistep-pilot-results.json.gz`, each with a
  `-runtime.json`. Checkpoints in `runs/number-construction-v1-multistep-pilot/` (not tracked).
- **Replication:** M-control reproduces Part D's D-varied+rehearsal exactly on every isolated
  measure, for all three seeds.

Arms: **M-control** (D-varied+rehearsal); **M-multistep** (the successor applied k = 1–3 times
to a real item must match the real collection at the end count); **M-equality** (same-count
pairs, "not more" in both orders); **M-multistep+equality**. Means over seeds 100–102, per-seed
values in brackets where they matter.

## 1. The multi-step objective fixes forward composition

**Step length.** Advance ratio along the comparison direction (1 = a learned step moves as far
as real items of the next count lie; collections):

| | k = 1 | k = 2 | k = 3 |
| --- | --- | --- | --- |
| M-control, add | 0.38 (0.17, 0.18, 0.80) | 0.44 | 0.49 |
| M-multistep, add | **0.95** (1.14, 0.94, 0.78) | 0.78 | 0.67 |
| M-control, remove (exact inverse) | 0.51 | 0.77 | 1.04 (up to 1.70) |
| M-multistep, remove (exact inverse) | 0.92 | 0.77 | 0.70 |

Taps: forward 0.33–0.45 → 0.54–0.64.

**Forward strict crossings** (reference strictly between start and result, the cleanest
composition diagnostic):

| k | M-control composed | M-multistep composed | M-multistep shown result |
| --- | --- | --- | --- |
| 2 | 0.53 (0.41, 0.44, 0.72) | **0.68** (0.64, 0.74, 0.65) | 0.68 |
| 3 | 0.54 (0.35, 0.43, 0.84) | **0.69** (0.64, 0.76, 0.67) | 0.74 |

- One learned step now covers about a full real step, and imagined results are compared as
  well as real items of that count are. All three seeds pass, against one in the control.
- **The diagnosis is confirmed by intervention:** the forward failure was short steps, and
  teaching repeated application to land on real endpoints removes it.
- **Removal no longer overshoots** (0.70–0.92 instead of up to 1.70), and inverse crossings
  stay about the same (0.74 at k = 3, against 0.77). So the earlier "removal works" was
  overshoot, as diagnosed.
- **Actions are respected better.** Composed against a wrong step count: 0.61 against 0.46
  (control 0.69 against 0.59, a smaller gap). Against the reversed action: 0.79 against 0.23.
- Imagined states keep their count somewhat better: collections at k = 3, 0.44 → 0.54.
- **Full-set verdicts do not rise** (composed 0.80 → 0.79, shortcut 0.75 → 0.74), because
  non-critical accuracy falls (0.87 → 0.82) together with the shown result (0.82 → 0.78):
  the comparison head itself became slightly worse on real items.

## 2. Equality through the comparison head interferes with ordering

Ties (the result equals the reference; the correct answer is "neither more nor fewer"):

| | M-control | M-equality | M-multistep+equality |
| --- | --- | --- | --- |
| Shown result | 0.48 | 0.68 | 0.72 |
| Composed | 0.34 | 0.45 | 0.64 |

- Equality pairs teach the tolerance region (shown result 0.48 → 0.68).
- **But alone they break forward composition:** advance ratios fall to about zero or below
  (collections −0.04, −0.16, −0.29 for k = 1–3), forward crossings fall to 0.40 and 0.35, and
  comparison of forms falls (Chinese 0.92 → 0.71, English 0.77 → 0.68). The comparison head's
  tolerance for equal pairs is bought by flattening the successor's movement along the
  comparison direction.
- **With the multi-step objective** both survive in part: ties 0.64–0.72, crossings 0.61, step
  length 0.62 at k = 1. One seed (101) does well on everything (crossings 0.70–0.78, ties 0.76).

## 3. Part D measures

| End of training | M-control | M-multistep | M-equality | M-multistep+equality |
| --- | --- | --- | --- | --- |
| Devanagari withheld, own block | 0.46 | **0.60** | 0.55 | 0.53 |
| Chinese withheld, own block | 0.38 | 0.46 | 0.39 | 0.42 |
| taps → withheld Devanagari form | 0.76 | 0.88 | 0.81 | 0.90 |
| taps → withheld Chinese form | 0.76 | 0.81 | 0.88 | 0.78 |
| Devanagari paired, own block | 0.53 | 0.46 | 0.51 | 0.48 |
| English, own block | 0.31 | 0.27 | 0.33 | 0.27 |
| Chinese paired, end | 0.60 | 0.54 | 0.59 | 0.55 |
| Comparison, Chinese forms | 0.92 | 0.86 | 0.71 | 0.80 |
| Coherence (successor residual), Chinese | 0.94 | 0.93 | 0.95 | 0.92 |

- The multi-step objective helps withheld forms, which reach meaning only through succession
  (Devanagari withheld 0.46 → 0.60; untaught taps → withheld form 0.76 → 0.81–0.88).
- It costs a little on paired forms, English and form comparisons (0.04–0.07). Several of
  these declines are about the size of the seed-to-seed spread, so the protocol's "no loss
  beyond seed variation" condition is met only marginally; they are recorded as a cost.

## Reading

- **Established in this pilot:** forward composition fails because one-step training allows
  short steps, and an explicit multi-step objective fixes it: steps reach full length, forward
  crossings rise from chance to the level of the shown result in every seed, and inverse
  overshoot disappears.
- **Equality** is learnable through the comparison head but, taught alone, it flattens the
  successor along the comparison direction and undoes composition. The *same* judgment is the
  alternative design.
- **Costs:** a small loss on paired forms, English and form comparisons with the multi-step
  objective; the overall hidden-result verdict does not improve because the comparison head
  weakens on real items.
- **Not decided:** whether to keep the multi-step objective as the default, and how to teach
  equality.

## Computation

- **Training:** 1,439 s wall on 11 workers; 12,441 worker CPU seconds; per run 1,084–1,123 s
  wall (the last run, alone on the machine after the others finished: 327 s). Training CPU 1,062–1,102 s per run in every arm:
  the multi-step and equality losses add no measurable cost next to encoding. Evaluation
  6–9 s per run. Peak RSS of the largest worker 1.17 GiB.
- **Analyses:** composition 35 s wall (71 worker CPU s), projection 28 s wall (2 worker CPU s),
  mostly data preparation.
- **Selection costs:** a smoke-mode equivalence check of 4 updates per arm type (control
  against Part D, and the pair layout).
