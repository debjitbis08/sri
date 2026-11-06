# Multi-step reader v1: results

Contract: [multi-step reader v1](multistep-reader-v1.md), frozen at commit 9762718. Run 2026-11-06, seeds 450–453, all 16
runs complete.

- **Command:** `EV_WORKERS=8 .venv/bin/python -m experiments.multistep_reader main` (4.0 h wall).
- **Artifacts:** `reports/multistep-reader-v1{,-summary,-runtime}.json`; checkpoints in `runs/multistep-reader-v1/`
  (not tracked).

## Summary

1. **The loops do not use their steps.**
   - Both loops halt almost at once: expected steps 0.0–1.0 on every test; the halting step does not follow chain length
     (Spearman −0.24 and −0.22).
   - Forcing a stop at step 0, 1, 2, 3 or 4 changes accuracy by at most a few points.
   - The first produced code is rarely the true middle thing:
     - the reptile for an anonymous reptile: 0.05 (`pred-loop`), 0.00 (`unit-loop`);
     - the named animal for a word: 0.32 and 0.11.
   - Each loop is at or below its own one-pass reader. `unit-loop` is far below `unit-one` (size 0.63 against 0.96,
     grandparent 0.52 against 0.97).
2. **Why, most likely [derived from the training setup, untested]:** nothing in training requires a step. Every trained
   statement, including the 1–2-step closures and every higher-order relation, can be fitted in one pass (one-pass fit
   1.00). So the halting loss and the geometric prior favour stopping at once. The loop is free to think but has no
   pressure to.
3. **The one-pass reader with the built-in truth unit learns these chains surprisingly well:**
   - size order at distances 3–11 from training at 1–2: 0.89–1.00;
   - grandparent 0.97;
   - held-out words on the higher-order relations 0.90;
   - ancestors at depth 3: 0.86 (AUROC 1.00 over all depths; depths 4–5 are rejected at threshold).

   In v1, order trained on adjacent pairs only failed. Here, with closures trained at distances 1–2, it generalizes to
   every distance.
4. **Taught truth's cost is large and persists.** The one-pass taught-truth reader trails the built-in unit on every
   chain:

   | Chain | taught truth | built-in unit | Gap |
   |---|---|---|---|
   | Size | 0.73 | 0.96 | 0.23 |
   | Grandparent | 0.53 | 0.97 | 0.44 |
   | Named higher-order relations | 0.78 | 0.90 | 0.12 |
   | Order | 0.61 | 0.78 | 0.17 |

   Its ranking is closer (AUROC 0.82–0.91 against 0.96–1.00), but it answers F for most held-out true facts (true
   accepted 0.06–0.61). The contract's criterion was a gap above 0.10 in the loop comparison; both loops failed alike
   (`pred-loop` 0.67/0.74 against `unit-loop` 0.63/0.72). In the one-pass comparison the gap is well above 0.10.
5. **L0 is not solved:** "sometimes → unknown" for new individuals is 0.00 (loop) and 0.07 (one-pass), as in taught
   truth v1.

## Results

Balanced accuracy at the arm's own decision rule (taught truth: T must be predicted for a positive and F for a negative;
unit: logit > 0). Mean of 4 seeds.

| Test | `pred-one` | `pred-loop` | `unit-one` | `unit-loop` |
|---|---|---|---|---|
| P1 inheritance (`default`) | 0.84 | 0.78 | 0.88 | 0.82 |
| P2 symmetry | 0.68 | 0.65 | 0.71 | 0.65 |
| P3 inverse | 0.68 | 0.62 | 0.74 | 0.61 |
| P4 transitivity (other scales) | 0.61 | 0.54 | 0.78 | 0.52 |
| P6 grandparent | 0.53 | 0.57 | **0.97** | 0.52 |
| P6 same colour | 0.53 | 0.53 | 0.58 | 0.48 |
| G2 category from meaning | 0.70 | 0.70 | 0.62 | 0.61 |
| G4 silly vs sensible | 0.57 | 0.56 | 0.58 | 0.54 |
| **L1 kind-of** (depth 3 / 4) | 0.63 (0.60 / 0.66) | 0.56 (0.56 / 0.56) | 0.69 (0.85 / 0.53) | 0.65 (0.73 / 0.57) |
| **L1 ancestor-of** (depth 3 / 4 / 5) | 0.55 (0.57 / 0.52 / 0.54) | 0.58 (0.58 / 0.56 / 0.58) | 0.71 (0.86 / 0.56 / 0.54) | 0.57 (0.57 / 0.54 / 0.62) |
| **L1 size** (distance 3–11) | 0.73 (0.50–0.81) | 0.67 (0.56–0.74) | **0.96 (0.89–1.00)** | 0.63 (0.50–0.68) |
| **L2 named-\*** (held-out words) | 0.78 | 0.74 | **0.90** | 0.72 |
| can / has / kind | 0.63 / 0.73 / 0.86 | 0.55 / 0.67 / 0.85 | 0.81 / 0.84 / 0.98 | 0.61 / 0.67 / 0.80 |
| L2-x (rows kept out) | 0.77 | 0.71 | 0.79 | 0.72 |
| P6 ancestor rows of tree B's colin | 0.72 | 0.74 | 0.98 | 0.66 |

**AUROC and true accepted on the chains:**

| Chain | `pred-one` AUROC / true acc. | `pred-loop` | `unit-one` | `unit-loop` |
|---|---|---|---|---|
| L1 size | 0.88 / 0.49 | 0.82 / 0.37 | 1.00 / 0.92 | 0.69 / 0.38 |
| L2 named | 0.86 / 0.61 | 0.83 / 0.52 | 0.96 / 0.84 | 0.90 / 0.46 |
| Grandparent | 0.91 / 0.06 | 0.82 / 0.19 | 1.00 / 0.94 | 0.61 / 0.12 |
| Ancestor | 0.91 / 0.10 | 0.89 / 0.15 | 1.00 / 0.42 | 0.84 / 0.15 |

**Reading at T = 6** changes nothing: the loops halt before step 1.

**Taught-truth reads:**

| Read | `pred-one` | `pred-loop` |
|---|---|---|
| S2 kind quantities | 0.69 | 0.74 |
| S3b "sometimes → U" | 0.07 | 0.00 |
| S4a unknown | 0.42 | 0.47 |
| P7 unknown | 0.50 | 0.68 |
| control U | 0.00 | 0.00 |

**Inspecting the loops:**

| Read | `pred-loop` | `unit-loop` |
|---|---|---|
| Expected steps (S3b / L2 / L1 by length) | 0.00 / 0.26 / 0.46–1.01 | 0.06 / 0.02 / 0.27–0.47 |
| Halting vs chain length (Spearman) | −0.24 | −0.22 |
| First produced code = true middle (S3b / L2) | 0.05 / 0.32 | 0.00 / 0.11 |
| Forced stop t = 0…4, L1 size | 0.57 0.60 0.55 0.52 0.53 | 0.55 0.56 0.53 0.52 0.51 |
| Forced stop t = 0…4, L2 named | 0.73 0.69 0.64 0.61 0.59 | 0.72 0.58 0.65 0.56 0.61 |

The produced codes are spread over unrelated entities (for an anonymous reptile: `fly-w`, `rabbit`, `meat`, `dark`, …).

**Fast mapping:**
- dax "never" 0.89–1.00; other classes rejected 1.00 in every arm.
- dax "usually" 0.12–0.62; "sometimes → U" 0.03–0.06.
- blick-w (names an owl): `unit-one` 0.93, `pred-one` 0.86, `unit-loop` 0.79, `pred-loop` 0.72.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | The loop solves L0 (sometimes → U ≥ 0.5); the produced code is the right kind ≥ 0.6 | **Fails:** 0.00; 0.05 |
| 2 | L2 held-out words: `pred-loop` ≥ 0.75 against one-pass ≤ 0.65 | **Fails:** `pred-loop` 0.74, below `pred-one` 0.78, and far below `unit-one` 0.90 |
| 3 | Length partially generalizes, length 3 above one-pass by ≥ 0.15 | **Fails** for the loops. The one-pass `unit-one` generalizes size to every distance |
| 4 | `pred-loop` within 0.05 of `unit-loop`; a gap above 0.10 means rethinking taught truth | **Holds** for the loops (both fail alike). The one-pass comparison shows a gap of 0.12–0.44 |
| 5 | Risk: the loops lose arrangement by ≥ 0.05 | **Holds:** P1 `default` −0.06 in both loops; P3 −0.06 and −0.13 |
| 6 | Halting follows chain length | **Fails:** −0.24, −0.22 |

## Interpretation

- **[derived] No pressure, no steps.** The program's own principle applies: constraints are pressure. A loop that can fit
  every training statement at step 0 will. For the loop to learn to step, some trained answers must be unreachable in
  one pass. For example: questions about entities whose relevant facts the reader could only reach through another
  fact, or episodes with fresh entities each time, so that the answer cannot be memorized per pair.
- **[read: results] One-pass chaining was underestimated.** With closures trained at 1–2 steps and their base relations,
  the built-in reader generalizes size order to distance 11 and grandparent fully. The v1–v3 failures were failures
  where the chain was never trained in any form (inferred properties, two different relations). Here every tested chain
  has its own trained closure or higher-order relation. So these tests measure generalization of a learned closure,
  which one pass can do, not chaining as such.
- **[read: results] Taught truth costs most where answers must be accepted.** The taught-truth reader ranks held-out true
  facts well (AUROC 0.82–0.91) but rarely answers T for them. The built-in unit does both. The cause is not established.
  Candidates: the 7-way softmax favours F, which is the majority individual value in closed-world training. And
  answering T for a statement never seen is harder through a nearest-code readout than through a sign. A diagnostic on
  the checkpoints (the decision between T and F logits alone, and per-relation calibration) would separate these.

## Computation (measured)

- Wall 4.0 h; aggregate worker CPU 30.2 h, against the declared 8–10 CPU h and 1–1.5 h wall.
- Training CPU per run: `pred-loop` 241 min, `unit-loop` 183 min, `pred-one` 24 min, `unit-one` 13 min. The smoke
  projection (68, 53, 9, 7) was measured with 4 workers; 8 workers on 6 cores ran about 3.5× slower per run.
- Inference (1 thread, forward only, per 1,000 statements): `pred-loop` 13.7 ms, `unit-loop` 20.5 ms, `pred-one` 2.6 ms,
  `unit-one` 0.8 ms.
- The runner had no progress logging during training; added to the checklist for future runners.
- No tuning or selection runs.

## Next (for the user)

1. **Give the loop pressure.** Train some answers that one pass cannot reach. Then the loop can be tested for whether it
   learns to step.
2. **Understand taught truth's cost** before deciding its fate: a cheap diagnostic on the existing checkpoints, with no
   training.
3. **Stronger chain tests:** chains whose composition is never trained as its own relation (as in v1–v3), now that
   trained closures turn out to be learnable in one pass.
