# Everyday relations v2: results

Contract: [everyday relations v2](everyday-relations-v2.md), frozen at commit 0f2f8b8. Run 2026-11-05, seeds 410–413, all
40 runs complete. There is an addendum after the freeze; it is described below and kept separate.

- **Commands:**
  - `.venv/bin/python -m experiments.everyday_relations_v2 part0` (Part 0);
  - `EV_WORKERS=8 .venv/bin/python -m experiments.everyday_relations_v2 main` (main, 24 min wall);
  - `.venv/bin/python -m experiments.everyday_relations_exclusivity` (post-hoc, v1 checkpoints);
  - `EV_WORKERS=8 .venv/bin/python -m experiments.everyday_relations_v2_addendum` (addendum, 12.5 min wall).
- **Artifacts:**
  - `reports/everyday-relations-v2{,-summary,-runtime}.json`;
  - `reports/everyday-relations-v2-part0{,-runtime}.json`;
  - `reports/everyday-relations-exclusivity{,-runtime}.json`;
  - `reports/everyday-relations-v2-addendum{,-runtime}.json`;
  - checkpoints in `runs/everyday-relations-v2/` (not tracked).

## Summary

1. **The open-world arms do not test their question.** In a world this small, 4 random negatives per fact cover nearly
   every pair. About 400 of the 490 held-out true facts were drawn as negatives at every step, so open-world training
   taught the test answers as false. Every open-world test is exactly 0.50, with every held-out fact rejected. The
   design was wrong for a small world. Below, only closed-world arms are read for the P-tests.
2. **With several relations per property, properties partly carry over.** Closed world, against the new shuffled control:

   | Property | v1 `two` | v2 `two` / `tworel` / `flat` | Shuffled |
   |---|---|---|---|
   | Inverse (P3) | 0.50 | 0.78 / 0.83 / 0.85 | 0.66–0.70 |
   | Order between items never compared (P4) | 0.51 | 0.72 / 0.84 / 0.82, with a distance effect | 0.59–0.69 |
   | Symmetry (P2) | 0.00 | 0.67 / 0.71 / 0.73 | 0.59–0.62 |

   It is uneven by relation. Comparatives and `has-baby` reach 0.8–1.0, while `child-of`, `prey-of` and `sibling-of`
   stay at chance.
3. **"Opposite" is not inferred from meaning (P5).** For speed and weight, whose poles mean something through `is` and the
   comparatives, held-out opposites are at chance in every arm. They rank below false pairs (AUROC 0.13–0.38), as in the
   shuffled world.
   A new word `toasty`, taught as hot through 4 facts, takes the meaning (which temperature things are toasty: 0.94–1.00).
   Its opposite is found only weakly (0.59–0.69; ComplEx 0.76).
4. **Fast mapping works when the new code stays among the known codes.**
   - Told only "a dax is a bird", the dax rejects every other class (1.00) and takes the bird properties (0.92–1.00) in
     all MLP arms. No exclusions are given; the exclusivity is learned from examples.
   - The same holds on v1 checkpoints (other classes rejected 0.95–1.00), where the free v1 fit made the dax a bird, a
     fish and a mammal at once (0.00–0.60).
   - ComplEx is weak here (0.55–0.95, and it does not reject other classes reliably).
5. **Telling G the relation (`tworel`) closes most of the gap to `flat`:**
   - grandparent 0.92 against `two` 0.56 and `flat` 0.98;
   - order 0.84 against 0.72 and 0.82.

   `flat` is still as good or better everywhere.

## Results (closed world unless stated)

Balanced accuracy at threshold 0 (AUROC). Mean of 4 seeds. † = the shuffled labels leave one class; accuracy shown, not
comparable. The closed-world shuffled controls are from the addendum.

| Test | `two` | `tworel` | `flat` | `complex` (open) | `two` shuffled | `tworel` shuffled | `flat` shuffled | `prior` |
|---|---|---|---|---|---|---|---|---|
| P1 inheritance (`default`) | 0.82 (0.94) | 0.87 (0.97) | 0.88 (0.97) | 0.60 (0.76) | 0.54 (0.54) | 0.57 (0.58) | 0.55 (0.52) | 0.65 |
| P1 hierarchy | 0.67 (0.88) | 0.81 (0.97) | 0.77 (0.94) | 0.50 (0.95) | 0.57 (0.71) | 0.47 (0.56) | 0.68 (0.70) | 0.50 |
| P1 lives-with | 0.71 (0.79) | 0.78 (0.87) | 0.78 (0.87) | 0.50 (0.40) | 0.52 (0.53) | 0.59 (0.60) | 0.54 (0.56) | 0.59 |
| P2 symmetry | 0.67 (0.81) | 0.71 (0.82) | 0.73 (0.84) | 0.57 (0.80) | 0.59 (0.66) | 0.62 (0.70) | 0.62 (0.70) | 0.50 |
| P3 inverse | 0.78 (0.89) | 0.83 (0.93) | 0.85 (0.93) | 0.55 (0.76) | 0.66 (0.75) | 0.69 (0.80) | 0.70 (0.80) | 0.51 |
| P4 transitivity | 0.72 (0.81) | 0.84 (0.95) | 0.82 (0.94) | 0.48 (0.50) | 0.59 (0.66) | 0.69 (0.75) | 0.65 (0.74) | 0.65 |
| P5 opposite from meaning | 0.49 (0.38) | 0.48 (0.21) | 0.48 (0.13) | 0.50 (0.49) | 0.49 (0.35) | 0.46 (0.29) | 0.49 (0.32) | 0.50 |
| P6 grandparent | 0.56 (0.94) | 0.92 (0.98) | 0.98 (1.00) | 0.59 (1.00) | 0.88† | 0.88† | 0.88† | 0.50 |
| P6 same colour (composition) | 0.53 (0.57) | 0.56 (0.63) | 0.57 (0.68) | 0.55 (0.68) | 0.50 (0.50) | 0.52 (0.50) | 0.50 (0.48) | 0.49 |
| P7 not inferable | 0.47 (0.49) | 0.47 (0.49) | 0.51 (0.54) | 0.50 (0.53) | 0.46 (0.41) | 0.47 (0.49) | 0.47 (0.42) | 0.43 |
| P8 word selection | 0.52 (0.57) | 0.54 (0.61) | 0.53 (0.65) | 0.50 (0.61) | 0.46 (0.50) | 0.41 (0.36) | 0.46 (0.34) | 0.54 |
| Train fit | 1.00 | 1.00 | 1.00 | 0.99 | — | — | — | — |

All open-world arms of `two`, `tworel` and `flat` read 0.50 on every test (AUROC 0.59–0.89), with false-positive rate 0.00.
See the summary point 1.

**Pass criterion (the real world beats its shuffled control by ≥ 0.10 balanced accuracy):**

| Arm | P1 `default` | P2 | P3 | P4 | P5 | P6 colour | P8 |
|---|---|---|---|---|---|---|---|
| `two` | +0.28 | +0.08 (no) | +0.12 | +0.13 | 0 | +0.03 | +0.06 |
| `tworel` | +0.30 | +0.09 (no) | +0.14 | +0.15 | +0.02 | +0.04 | +0.13 |
| `flat` | +0.33 | +0.11 | +0.15 | +0.17 | −0.01 | +0.07 | +0.07 |

**By relation** (closed `two` / `tworel` / `flat`, balanced accuracy):
- **P2 symmetry:**
  - `lives-with` 0.68 / 0.78 / 0.79;
  - `spouse-of` 0.54 / 0.70 / 0.72;
  - `opposite` 0.57 / 0.55 / 0.56;
  - `same-class-as` 0.59 / 0.55 / 0.57;
  - `same-colour-as` 0.53 / 0.50 / 0.54;
  - `sibling-of` 0.49 / 0.50 / 0.50.

  ComplEx (open): `opposite` 1.00, `sibling-of` 0.87, `spouse-of` 0.81.
- **P3 inverse:**
  - `lighter-than` 0.96 / 1.00 / 1.00;
  - `slower-than` 0.80 / 0.89 / 0.98;
  - `smaller-than` 0.81 / 0.90 / 0.95;
  - `darker-than` 0.71 / 0.88 / 0.84;
  - `has-baby` 0.83 / 0.92 / 0.88;
  - `child-of` 0.54 / 0.49 / 0.48;
  - `prey-of` 0.48 / 0.49 / 0.49.
- **P4 by rank distance 2 → 7:**
  - `tworel` 0.79, 0.82, 0.83, 0.91, 0.92, 1.00;
  - `flat` 0.77 → 1.00;
  - `two` flat at 0.68–0.76.

**Fast mapping.**

| Entity | `two` | `tworel` | `flat` | `complex` (open) |
|---|---|---|---|---|
| **No exclusions given (addendum, primary)** | | | | |
| `dax` k = 1 / 2 / 4 | 0.95 / 0.95 / 0.92 | 0.94 / 0.95 / 0.97 | 1.00 / 1.00 / 0.96 | 0.69 / 0.95 / 0.63 |
| `fep` k = 1 / 2 / 4 | 0.95 / 0.95 / 0.95 | 0.98 / 0.97 / 0.97 | 0.98 / 0.98 / 0.98 | 0.55 / 0.65 / 0.65 |
| Other classes rejected | 1.00 | 1.00 | 1.00 | 0.40–1.00 |
| **As frozen (exclusions given; comparison only)** | | | | |
| `dax` k = 1 | 0.97 | 0.96 | 0.98 | 0.90 |
| `zorp` k = 1 / 4 | 0.94 / 0.95 | 0.98 / 0.99 | 0.98 / 0.98 | 0.66 / 0.79 |
| `kim` k = 1 / 4 | 0.67 / 0.79 | 0.71 / 0.79 | 0.83 / 0.78 | 0.62 / 0.50 |
| `toasty` k = 4: meaning (`is`) | 0.94 | 1.00 | 1.00 | 0.56 |
| `toasty` k = 4: `opposite` | 0.63 | 0.59 | 0.66 | 0.76 |

`zorp`, `kim` and `toasty` were given no exclusions in the frozen design either, so their rows are valid as they stand.

**New relations** (only the code fitted, k = 4 / 16):
- `smaller-than-v` (a copy of a trained relation): `two` 0.88 / 0.92, `tworel` 0.86 / 0.94, `flat` 0.93 / 0.95. That is
  finding a known relation by its examples. v1's equivalent was at chance; v1 had no `smaller-than` to find.
- `both-fly` (genuinely new): 0.55–0.65.

**Relation space:**
- **Probe from q:** `two` 0.35. `tworel` closed 0.30, open 0.67 (q contains r by construction, so this is not evidence of
  abstraction).
- **Probe from pair codes:** 0.29–0.33 (majority chance 0.18).
- **Property analogy** (is the nearest q from another relation one with the same property?): `two` 0.51, `tworel` 0.32,
  against chance 0.29 and shuffled (open) 0.44.

**Learning order** (v1 prediction 2 retried, logs every 25 steps): every relation reaches 0.9 training balanced accuracy
by step 25–94. The slowest are the relations with broad class structure (`has` 75, `same-class-as` 88, `lives-with` 94).
That is the opposite of broad-first, but the measure is per relation, not per property level, and the relations differ
in size, so it does not test the Saxe prediction cleanly.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Open world raises P2 and P3 by ≥ 0.15; false-positive rate ≤ 0.10 | **Not tested.** The negatives drew the held-out facts (about 400 of 490 per step), so open world trained the answers as false |
| 2 | `tworel` reaches `flat` on composition and fast mapping (within 0.05); `two` stays below | **Mostly holds:** grandparent 0.92 against 0.98 (0.06), fast mapping within 0.06; `two` 0.56 on grandparent |
| 3 | P2 and P3 above shuffled by ≥ 0.15 (open arms) and better than v1; P4 ≤ 0.65 | Open arms not readable. Closed: above v1 (P3 0.78–0.85 against 0.50; P2 0.67–0.73 against 0.00). Above shuffled by +0.12–0.15 (P3) and +0.08–0.11 (P2), short of +0.15. **P4 fails the "weak" prediction:** 0.72–0.84 |
| 4 | P3 ≥ P2 ≥ P4 | P3 > P2 **holds**; P4 > P2, so the order **fails** |
| 5 | P5 above shuffled, below P2; `toasty`'s opposite found at k = 4 in at most half the seeds | P5 **fails:** at chance, same as shuffled. `toasty` 0.59–0.69 |
| 6 | Part 0: the prior fit ≥ 0.9; the false-fact fit alone in between | Prior fit **holds** (0.92–0.99); the false-fact fit alone is as good (0.91–1.00), not in between. Both supply the exclusions; the prior-only fit without them is the fair test (summary point 4) |
| 7 | P7 at chance; shuffled at `prior` level | P7 **holds.** Shuffled does **not** stay at `prior` on P2–P4 (0.59–0.70): see interpretation |

## Interpretation

- **[derived from the design] Why the shuffled control is high on P2–P4.** Shuffling relabels a relation's heads and tails
  by independent permutations. That breaks symmetry, inverses and links between relations. It does not break a relation's
  own threshold structure: a shuffled comparative is still "head rank below tail rank" under two relabelings, which is
  learnable from columns. So the shuffled world is a weaker control than intended for P2–P4. The real-world margin over it
  (+0.08 to +0.17) is the part that uses the property.
- **[derived; untested] P4 may be partly attribute sharing.** Dog, cat and mouse appear on several scales in the same
  relative order (size, speed, weight, loudness), and the sun and snow are on both temperature and brightness. Order on the
  adjacent-only scales could be read off shared items rather than from transitivity. The distance effect in
  `tworel`/`flat` (0.79 → 1.00) is consistent with either.
- **[read: results] Inverse carries over where the relation has many examples per entity.** Comparatives (each thing
  related to 6–11 others) and `has-baby` work; `child-of` (2 parents per person) and `prey-of` (1–3 predators) do not.
- **[derived] P5 fails for the reason v1's opposite failed, in a weaker form.** The held-out poles' rows are held out, but
  their columns are trained false, since no other adjective is their opposite. So "fast is nobody's opposite" is still the
  simplest fit. Inferring an opposite from meaning needs "every pole has exactly one opposite, on its own scale". That is a
  cardinality generalization, which the reader did not make. The below-chance AUROC (0.13–0.38) shows those pairs are
  actively pushed down.
- **[read: results] Exclusivity is learned, and fitting must respect it.** In every MLP arm, a new code kept among the
  existing codes inherits the class partition learned from examples. A code fitted freely leaves that region, and there
  every class says yes. This answers the user's point: "a bird cannot be a fish" is in the trained network. The v1 fit
  failed to use it.
- **[analogy] `tworel` ≈ `flat`.** Once G sees the relation, the pair space is a relation-specific view (TransH/TransR in
  spirit), and it behaves like `flat`. The relation-agnostic pair space (`two`) is the weakest MLP arm on composition and
  order. No arm shows q organized by property beyond the shuffled level.

## Computation (measured)

| Phase | Wall | CPU |
|---|---|---|
| Part 0 (v1 checkpoints; 360 fits, 8 workers) | 12 s | 94 s aggregate worker CPU (estimate 2 CPU min) |
| Main (40 runs, 8 workers) | 24.0 min | 11,075 s aggregate worker CPU (4.6 CPU min per run; estimate 5–8) |
| Exclusivity check (v1) | 34 s | 34 s |
| Addendum (12 shuffled-closed runs, 28 prior-only fit sets) | 12.5 min | 4,943 s |

- Peak RSS per worker is under 400 MiB.
- Checkpoints are about 120 kB each.
- No tuning or selection runs.
- Smoke runs (contract): 18 s wall.

## What the next step could test (for the user to decide)

1. **A working open world.** Draw negatives only for (head, relation) pairs that have a known true tail (Dong et al.'s
   local closed world). Or draw from a much larger pool of entities, so that a held-out fact is rarely drawn. Either needs
   a world where untaught pairs are not all near to taught ones.
2. **A cleaner transitivity test:** a scale whose items appear on no other scale.
3. **Cardinality ("each pole has one opposite"; "each person has two parents"):** this is what P5, `child-of` and
   `sibling-of` need. Is it learnable as a property shared across relations?
