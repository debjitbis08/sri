# Thinking-loop pilot v1: results

Spec: [thinking-loop pilot v1](think-pilot-v1-spec.md). Run 2026-10-29, seeds 100–102,
paired with the [sequence pilot](sequence-pilot-v1-results.md) (same batches; the shared
parameters start equal), the heads network and the generic network. **A pilot, not
headline evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.think_pilot pilot` (25 minutes);
  - `.venv/bin/python -m experiments.transition_pilot_analysis pilot think` (40 s);
  - `.venv/bin/python -m experiments.think_pilot_effort pilot` (47 s).
- **Artifacts:**
  - `reports/think-pilot-v1-pilot-{results.json.gz,runtime.json}`;
  - `reports/think-pilot-v1-comparison.json.gz` and its `-runtime.json`;
  - `reports/think-pilot-v1-effort.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/think-pilot-v1-pilot/` (not tracked).
- **Recomputation:** matches exactly.
- In the analysis printout, the "Generic" column is the sequence pilot and the
  "Generic + transition" column is this pilot (the analysis script's column names are
  shared across variants).

## Summary

**The thinking loop makes the system better at what it is taught and worse at placing
what it was not.**
- Taught matching and retention rise:
  - Chinese paired at its own block: 0.74;
  - retained with rehearsal: 0.67;
  - Thai: 0.20.
- **Placement of never-paired forms falls back to chance:** Chinese 0.10, Devanagari
  0.12, against 0.31 and 0.23 for the sequence pilot with the same prediction loss.
- **More thinking helps little.** A few measures rise from 1 to 2–4 steps. None improves
  beyond the trained 4 steps, and some degrade at 8.

The loop compares all candidates at once. It can answer taught questions by relative
comparison within the candidate set, without needing consistent positions in the code.
That weakens the pressure that arranged the code. It is the same pattern as removing the
heads: **a more flexible answerer arranges the representation less.**

## Results

Chance: 0.11 for matching, 0.5 for comparison and AUROC. Outcome: the thinking pilot
against the sequence pilot (paired by seed; proposed margins).

| Measure | Heads | Generic | Sequence | Thinking loop (seeds) | vs sequence |
| --- | --- | --- | --- | --- | --- |
| **Never directly paired forms** | | | | | |
| Chinese withheld, own block | 0.42 | 0.13 | 0.31 | 0.10 (0.12, 0.05, 0.13) | worse |
| Devanagari withheld, own block | 0.47 | 0.10 | 0.23 | 0.12 (0.08, 0.22, 0.05) | worse |
| **Untaught compositions** | | | | | |
| taps → Chinese form, withheld | 0.79 | 0.04 | 0.00 | 0.14 (0.37, 0.03, 0.02) | better |
| taps → Devanagari form, withheld | 0.78 | 0.14 | 0.44 | 0.40 (0.58, 0.42, 0.20) | inconclusive |
| form → taps, Chinese withheld | 0.50 | 0.28 | 0.31 | 0.22 | inconclusive |
| Comparison, Chinese forms | 0.88 | 0.51 | 0.58 | 0.69 (0.74, 0.59, 0.74) | better |
| Comparison, Devanagari forms | 0.79 | 0.55 | 0.65 | 0.68 | equivalent |
| Comparison, English forms | 0.76 | 0.47 | 0.49 | 0.51 | inconclusive |
| **Taught** | | | | | |
| Chinese paired, own block | 0.57 | 0.72 | 0.70 | 0.74 | inconclusive |
| taps → collection | 0.57 | 0.77 | 0.69 | 0.71 | inconclusive |
| Comparison, collections | 0.96 | 0.98 | 0.98 | 0.98 | equivalent |
| Succession pairwise, collections | 0.93 | 0.92 | 0.91 | 0.90 | equivalent |
| **Retention** | | | | | |
| Chinese paired, end (D-rehearsal) | 0.50 | 0.57 | 0.61 | 0.67 | inconclusive |
| English, end | 0.31 | 0.44 | 0.42 | 0.38 | inconclusive |
| **Judgment revision**, withheld | | | | | |
| Isolated | 0.44 | 0.12 | 0.25 | 0.14 | worse |
| Revised | 0.62 | 0.72 | 0.65 | 0.71 | inconclusive |
| Swapped item: answer follows the context | 0.29 | 0.66 | 0.54 | 0.69 | inconclusive |
| **Next-judgment coherence AUROC** (Chinese / English) | 0.95 / 0.76 | 0.90 / 0.82 | 0.91 / 0.78 | 0.96 / 0.81 | better / inconclusive |
| Thai, 4 shots, after 200 updates | 0.16 | 0.18 | 0.15 | 0.20 | better |

- Composition through the loop's *same* answer is still low (one step from collections:
  0.22). By distance it is 0.57, as in the sequence pilot. The loop does not make
  imagined states readable.

## Thinking effort

D-varied+rehearsal checkpoints, trained with K = 4, re-evaluated at each K (means over
seeds):

| Measure | K = 1 | K = 2 | K = 4 | K = 8 |
| --- | --- | --- | --- | --- |
| Chinese paired, end | 0.57 | 0.67 | 0.66 | 0.66 |
| taps → Devanagari form, withheld | 0.24 | 0.36 | **0.40** | 0.34 |
| Judgment revision, withheld: revised | 0.68 | 0.72 | 0.71 | 0.64 |
| taps → collection | 0.70 | 0.72 | 0.71 | 0.71 |
| Comparison, Chinese / Devanagari forms | 0.74 / 0.68 | 0.70 / 0.68 | 0.69 / 0.68 | 0.70 / 0.67 |
| Comparison, collections | 0.98 | 0.98 | 0.98 | 0.98 |
| Succession pairwise, collections | 0.91 | 0.91 | 0.90 | 0.84 |
| Chinese withheld, end | 0.23 | 0.18 | 0.16 | 0.15 |
| Devanagari withheld, end | 0.15 | 0.13 | 0.12 | 0.11 |

- **Extra steps help on a few matching questions:** paired forms, and one cross-sense
  composition (taps → withheld Devanagari, 0.24 → 0.40, in two of three seeds). Most of
  the gain comes by K = 2.
- **Judgments do not benefit.** Comparison and succession are flat, and succession drops
  at K = 8, beyond the trained steps.
- **Withheld forms get worse with more thinking** (Chinese 0.23 → 0.15). The loop pulls
  their answers toward what fits the taught candidates, not toward their meaning.
- The prediction in the spec was right that the curve rises to K = 4 and not beyond, but
  only for some measures. It was wrong about placement: the loop made it worse, not
  unchanged.

## Interpretation

- **Thinking before answering is not what is missing.** The questions here (match one of
  9, judge a pair) need little multi-step inference. The loop's extra capacity goes into
  answering taught questions better, and the shared code becomes less arranged.
- **This is the third instance of one pattern:**
  - generic comparator instead of heads;
  - action tokens removed (weaker prediction signal);
  - loop instead of one-pass comparator.

  Every change that gives the answerer more freedom, or the prediction less signal,
  reduces placement and untaught transfer. **Transfer comes from pressure on the
  representation, not from the power of the reasoner.**
- **Where a loop could still matter:** questions that need evidence combined over steps.
  Examples: context items as nodes (revision inside the loop), or relational conflicts
  that appear only when several judgments are combined. Neither is tested here.

## Computation

| | Sequence pilot | Thinking loop |
| --- | --- | --- |
| Worker CPU, 9 runs | 5,512 s | 11,713 s |
| Per run | 612 s | 1,301 s |
| Job wall time (11 workers) | 688 s | 1,491 s |
| Answerer parameters | 6,873 (comparator) | 13,665 (loop; the comparator is kept but unused) |

- Effort curve: 47 s for 3 checkpoints × 4 values of K. Analysis: 40 s.
- Selection and tuning: none.

## Caveats

- Three seeds, reused. The withheld-placement drop is consistent across seeds; several
  effort-curve gains come from two of three seeds.
- One loop design: fully connected messages, hidden size 32, K = 4 in training.
