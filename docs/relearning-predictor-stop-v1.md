# Stopping form rows' new gradients into the predictor from update 10

Status: **diagnostic, 2026-11-04, seeds 300–303. Not a contract and not a remedy.** It runs the intervention proposed in [margin attribution on the stop trajectory](relearning-margins-stop-v1.md), as revised before the run.

Code: `experiments/relearning_predictor_stop.py`, with tables from `experiments/relearning_predictor_stop_summary.py`. Report: `reports/relearning-predictor-stop-v1.json.gz`.

## Arms

All arms start from the captured pre-relearning state and run 20 updates on identical batches and draws.

| Arm | Change |
|---|---|
| `unchanged` | The original relearning. |
| `form-cand-stop` | Ordinary stop-gradient of form rows' sequence gradient through the candidate collections, for all 20 updates. |
| `form-cand-stop+P` | Identical to `form-cand-stop` up to update 10 (largest difference 0 over all reads at 0 and 10). From update 10, form rows' predictions are also computed with P's parameters detached, keeping their gradient into the codes they are made from. |

- **Optimizer state:** Adam's state at update 10 is shared and kept. This stops **new** form-row gradients into P only; earlier form-driven contributions in the first moment keep decaying into later steps.
- **Unchanged in all arms:** context rows' sequence gradients and every other loss.

**Reads** at 0, 10, 15 and 20:
- the interpreter;
- memory present, with actual retrieval;
- English matching;
- sequence prediction and composition.

**Swapped reads.** For the two stop arms at 15 and 20, sequence prediction and composition are also read with the update-10 state of one part substituted:
- current codes with **P as at update 10**: a change here is due to the representations;
- **codes as at update 10** with the current P: a change here is due to the predictor.

The codes are the image head, the taps encoder and the core projections.

## Results

Means over seeds at updates 15 / 20; the update-10 value, shared by both stop arms, is in parentheses.

| Read (at 10) | `unchanged` | `form-cand-stop` | `form-cand-stop+P` | Expected (+P) |
|---|---|---|---|---|
| **Memory A, symbol → taps** (0.68) | 0.26 / 0.19 | 0.58 / **0.61** | 0.56 / **0.58** | about 0.62 |
| **Memory B, symbol → taps** (0.72) | 0.27 / 0.20 | 0.62 / **0.65** | 0.58 / **0.62** | about 0.65 |
| Memory A, taps → symbol (0.72) | 0.39 / 0.41 | 0.67 / 0.76 | 0.66 / 0.73 | — |
| Collections → taps, fresh (0.75) | 0.33 / 0.26 | 0.64 / 0.69 | 0.62 / 0.65 | about 0.67 |
| Collections → taps, stored (0.84) | 0.34 / 0.26 | 0.75 / 0.77 | 0.73 / 0.74 | — |
| Taps → collections (0.81) | 0.47 / 0.50 | 0.75 / 0.87 | 0.75 / 0.86 | — |
| Composition, collections up 2 (0.69) | 0.33 / 0.44 | 0.60 / 0.53 | 0.64 / **0.59** | about 0.71 |
| Composition, collections up 3 (0.62) | 0.29 / 0.32 | 0.48 / 0.45 | 0.52 / **0.50** | about 0.66 |
| Composition, taps up 2 (0.82) | 0.50 / 0.53 | 0.73 / 0.64 | 0.79 / **0.77** | about 0.87 |
| Sequence, collections down 1 (0.84) | 0.50 / 0.43 | 0.71 / 0.64 | 0.73 / **0.68** | about 0.82 |
| Sequence, collections up 1 (0.89) | 0.47 / 0.52 | 0.86 / 0.80 | 0.86 / **0.83** | about 0.86 |
| Sequence, taps up 1 (0.91) | 0.67 / 0.64 | 0.90 / 0.86 | 0.91 / **0.90** | — |
| English matching (0.17) | 0.17 / 0.23 | 0.17 / 0.21 | 0.18 / 0.19 | little change |
| Sequence, English forms up 1 (0.18) | 0.18 / 0.22 | 0.22 / 0.21 | 0.20 / 0.22 | — |
| Sequence, English forms down 1 (0.17) | 0.25 / 0.20 | 0.18 / 0.20 | 0.19 / 0.24 | — |
| Sequence, Chinese forms up 1 (0.14) | 0.18 / 0.26 | 0.20 / 0.33 | 0.21 / 0.38 | — |
| Sequence, Devanagari forms up 1 (0.56) | 0.49 / 0.53 | 0.60 / 0.57 | 0.59 / 0.62 | — |
| Sequence, Devanagari forms down 1 (0.60) | — | 0.43 / 0.34 | 0.47 / 0.43 | — |
| Collection spacing (23.5) | 11.0 / 8.9 | 23.0 / 22.9 | 23.1 / 23.2 | — |

**Swapped reads at update 20,** as read → with the update-10 predictor → with the update-10 codes:

| Read | `form-cand-stop` | `form-cand-stop+P` |
|---|---|---|
| Composition, collections up 2 | 0.53 → **0.68** → 0.53 | 0.59 → **0.68** → 0.62 |
| Composition, collections up 3 | 0.45 → **0.61** → 0.46 | 0.50 → **0.61** → 0.48 |
| Composition, taps up 2 | 0.64 → **0.83** → 0.51 | 0.77 → **0.83** → 0.65 |
| Sequence, collections down 1 | 0.64 → **0.81** → 0.67 | 0.68 → **0.81** → 0.69 |
| Sequence, English forms up 1 (0.18 at 10) | 0.21 → 0.22 → 0.21 | 0.22 → 0.19 → 0.18 |
| Sequence, English forms down 1 (0.17) | 0.20 → 0.20 → 0.18 | 0.24 → **0.23** → 0.17 |
| Sequence, Chinese forms up 1 (0.14) | 0.33 → **0.28** → 0.15 | 0.38 → **0.34** → 0.14 |
| Sequence, Devanagari forms down 1 (0.60) | 0.34 → 0.54 → 0.39 | 0.43 → 0.55 → 0.45 |

## What this shows

1. **Composition is partly protected; the direction is right, but the size is well below the expectation.**
   - At update 20, taps composition is 0.77 against 0.64 (expected about 0.87). Collection composition is 0.59 and 0.50 against 0.53 and 0.45 (expected about 0.71 and 0.66).
   - Descending prediction is 0.68 against 0.64 (expected about 0.82).
2. **Composition still falls because P still changes after update 10.**
   - With the update-10 predictor in place, composition reads the same in both arms (0.68, 0.61, 0.83, 0.81). With the update-10 codes, it does not recover.
   - So the decline is in the predictor, not in the codes, even when new form-row gradients into P are stopped.
   - Two sources remain and are not separated here: Adam's first-moment history from form rows before update 10 (β₁ = 0.9, so about 35% of the moment at update 10 is left after ten steps), and context rows' sequence gradients into P.
3. **Form-sequence learning continues, through the representations.**
   - With new form gradients into P stopped, English forms' descending prediction rises 0.17 → 0.24, Chinese forms' ascending 0.14 → 0.38, and Devanagari forms' ascending 0.56 → 0.62.
   - The swaps place these gains in the codes. With the update-10 predictor the gains remain (0.23, 0.34). With the update-10 codes they vanish (0.17, 0.14).
   - English forms' ascending prediction moves little in any arm (0.18 → 0.20–0.22). It is too noisy to attribute.
   - The same holds in the plain stop arm. There, the predictor's form-driven change after update 10 adds no measurable form-sequence gain (codes at 10 with the current P: 0.15–0.21), while it costs composition.
4. **Memory and the interpreter are slightly lower, not unchanged.**
   - Memory-present symbol → taps is 0.58 and 0.62 at 20, against 0.61 and 0.65 (seed ranges overlap); collections → taps is 0.65 against 0.69.
   - The expectation was no change. The difference is small and unexplained. It may come from representations carrying more of the form-sequence adaptation, which was not measured.
5. **English matching is 0.19 against 0.21 at 20** (seed ranges 0.09–0.27 and 0.09–0.26). The prediction of little change roughly held, with a small decrease.

## Answer, against the outcomes stated before the run

**Composition partly survives, and form-sequence learning continues.** Its gains come from the form representations adapting to the existing predictor. In this setting and window, that supports the route of adapting representations to the existing predictor: new forms reaching shared relational knowledge without rewriting it.

The form-driven updates to P were not needed for the form-sequence gains measured here, and they cost composition.

It is not a remedy:
- composition still falls, through P changes that this intervention does not stop;
- memory-present reads are slightly lower;
- the window is 10 updates.

## What remains open

- **Which remaining change to P lowers composition:** Adam's carried-over form-driven history, or context rows' gradients. Restarting P's first moment at the branch would separate them. So would the margin decomposition run on this trajectory, which splits the history as its own path.
- **The small memory-present decrease.**
- **Longer windows.**

## Computation

- **GPU** (GTX 1650), three workers, float32, deterministic algorithms.
- **Per seed:** capture 158–204 s; each arm's 20 updates with 4 read points, 3–6 s. The swapped reads add two composition reads per point in the stop arms.
- **Totals:** 7 min 9 s wall, 970 s aggregate worker CPU (mostly the capture), peak RSS 3.7 GB.
- **Cost of the intervention:** one functional evaluation of P with detached parameters on form rows per update.
- **Selection and tuning:** none.
