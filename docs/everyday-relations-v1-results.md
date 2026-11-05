# Everyday relations v1: results

Contract: [everyday relations v1](everyday-relations-v1.md), frozen at commit 6726ffa. Run 2026-11-05, seeds 400–403, all
arms complete (28 runs).

- **Commands:**
  - `EV_WORKERS=8 .venv/bin/python -m experiments.everyday_relations main` (13 min wall);
  - `.venv/bin/python -m experiments.everyday_relations_posthoc` (post-hoc diagnostics, not in the contract; under 1 s).
- **Artifacts:**
  - `reports/everyday-relations-v1.json`, `-summary.json` and `-runtime.json`;
  - `reports/everyday-relations-v1-posthoc.json` and its `-runtime.json`;
  - checkpoints in `runs/everyday-relations-v1/` (not tracked).

## Summary

**The two-space core learns what items share, not what relations do.**

**What works.** Every arm learns the taught facts (train balanced accuracy 1.00). Every arm also generalizes facts that
follow from **shared attributes**:
- an untaught animal gets its class's properties (T1 `default`: `two` 0.93, shuffled control 0.59);
- it is an animal (T2: `two` 0.88, shuffled 0.51);
- when penguin's exception is not trained, it gets the bird default and flies. That happened by accident: seed 401's
  validation split held the fact out.

This is the Rogers–McClelland and Hinton result, and it holds in all arms.

**What fails.** Every test where a relation's own property must be applied to an untaught pair:
- **order:** T4 transitivity is at chance in every arm, ComplEx included, with no distance effect;
- **symmetry:** T5, every reversed opposite or sibling is judged false;
- **inverse:** T6, `child-of` from `parent-of`, at chance in `two`;
- **new relations:** a new relation fitted from 4–16 examples is at chance.

The MLP readers judge about 90% of untaught size pairs false. Held-out true facts rank above false ones (AUROC 0.6–0.8),
but they score far below zero. **Untaught means false** to these readers.

**The pair space does not help.** `two` is no better than `flat` anywhere, and worse on:
- composition (T7: 0.75 against 1.00);
- inverses (T6 AUROC 0.67 against 0.79);
- fast mapping from one fact (0.56 against 0.76–0.80).

**The relation space does not hold relations.** A linear probe reads the relation from q at 0.40, the same as on the
shuffled world (0.38). From the raw pair codes it reads 0.65.

**Fast mapping is weak.** A new animal given "a dax is a bird" gets bird properties at 0.56 in `two` and 0.76 in `flat`.
More facts do not help. In ComplEx, the dax code lies nearest the bird centroid in every seed, yet the properties do not
follow (0.55).

## Results

Balanced accuracy at threshold 0; AUROC in brackets. Mean of 4 seeds. For the positive-only tests (T3b, T5) the cells
give accuracy, with AUROC against the relation's trained false queries in brackets. Shuffled-world labels differ at the
held-out positions; where they leave one class, the cell is accuracy (marked †) and is not comparable.

| Test | `two` | `two-16` | `two-64` | `flat` | `complex` | `two` shuffled | `flat` shuffled | `prior` |
|---|---|---|---|---|---|---|---|---|
| T1 inheritance, all | 0.81 (0.88) | 0.78 (0.85) | 0.79 (0.86) | 0.81 (0.87) | 0.77 (0.84) | 0.60 (0.71) | 0.61 (0.68) | 0.58 |
| T1 `default` (inferable) | **0.93 (0.98)** | 0.90 (0.97) | 0.92 (0.97) | 0.93 (0.97) | 0.90 (0.96) | 0.59 (0.71) | 0.59 (0.68) | 0.65 |
| T1 `specific` (not inferable) | 0.11 (0.10) | 0.15 (0.11) | 0.12 (0.12) | 0.18 (0.14) | 0.47 (0.16) | 0.61 | 0.68 | 0.15 |
| T2 hierarchy | **0.88 (0.96)** | 0.90 (0.92) | 0.77 (0.90) | 0.88 (0.98) | 0.94 (0.99) | 0.51 (0.70) | 0.54 (0.72) | 0.50 |
| T3b bat flies (acc) | 0.75 (1.00) | 0.75 | 0.75 | 0.50 | 0.75 | 0.75† | 0.50† | 1.00 |
| T4 order | 0.51 (0.45) | 0.51 (0.54) | 0.49 (0.47) | 0.54 (0.55) | 0.47 (0.45) | 0.56 (0.67) | 0.63 (0.69) | 0.59 |
| T5 symmetry (acc) | 0.00 (0.62) | 0.00 (0.70) | 0.00 (0.56) | 0.00 (0.68) | 0.80 (0.90) | 0.95† | 0.95† | 0.40 |
| T6 inverse | 0.50 (0.67) | 0.52 (0.69) | 0.50 (0.58) | 0.65 (0.79) | 0.52 (0.87) | 0.48 (0.47) | 0.49 (0.50) | 0.36 |
| T7 composition | 0.75 (0.82) | 0.66 (0.96) | 0.66 (0.85) | **1.00 (1.00)** | **1.00 (1.00)** | 0.99† | 1.00† | 0.50 |
| T8 word selection | 0.60 (0.59) | 0.60 (0.61) | 0.54 (0.52) | 0.64 (0.69) | 0.55 (0.59) | 0.47 (0.57) | 0.51 (0.56) | 0.54 |
| T9 not inferable | 0.47 (0.43) | 0.47 (0.44) | 0.46 (0.39) | 0.51 (0.57) | 0.42 (0.51) | 0.53 (0.50) | 0.48 (0.31) | 0.43 |
| Train fit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | — |

Per seed, the tests that vary:
- **T7 `two`:** 1.00, 0.63, 0.88, 0.50;
- **T8 `two`:** 0.65, 0.68, 0.61, 0.45;
- **T6 `flat`:** 0.58, 0.66, 0.71, 0.67.

**Fast mapping** (frozen network; only the new code is fitted; balanced accuracy on the read set; k = number of given facts):

| Entity | k | `two` | `two-16` | `two-64` | `flat` | `complex` |
|---|---|---|---|---|---|---|
| `dax` (bird) | 1 / 2 / 4 | 0.56 / 0.60 / 0.50 | 0.69 / 0.76 / 0.67 | 0.78 / 0.62 / 0.59 | 0.76 / 0.78 / 0.73 | 0.55 / 0.66 / 0.56 |
| `fep` (fish) | 1 / 2 / 4 | 0.54 / 0.55 / 0.50 | 0.66 / 0.71 / 0.58 | 0.64 / 0.59 / 0.54 | 0.80 / 0.78 / 0.72 | 0.41 / 0.47 / 0.56 |
| `wug-w` (names owl) | 1 / 2 | 0.57 / 0.53 | 0.62 / 0.50 | 0.50 / 0.53 | 0.62 / 0.55 | 0.53 / 0.55 |
| `zorp` (size) | 1 / 2 / 4 | 0.60 / 0.45 / 0.53 | 0.56 / 0.40 / 0.50 | 0.58 / 0.49 / 0.46 | 0.56 / 0.44 / 0.49 | 0.49 / 0.49 / 0.49 |
| `kim` (family) | 1 / 2 / 4 | 0.44 / 0.66 / 0.48 | 0.47 / 0.62 / 0.70 | 0.40 / 0.56 / 0.59 | 0.52 / 0.70 / 0.60 | 0.53 / 0.54 / 0.54 |
| `dax` class rank (1 = birds nearest) | 1 / 2 / 4 | 2.0 / 1.25 / 1.75 | 2.0 / 2.75 / 1.75 | 1.0 / 1.0 / 1.5 | 1.0 / 1.75 / 1.25 | 1.0 / 1.0 / 1.0 |

**New relations** (only the relation code fitted; balanced accuracy on the rest): all arms 0.43–0.66. `smaller-than`
k = 16: `two` 0.51, `flat` 0.64, `complex` 0.62. `same-class` k = 16: `two` 0.53, `flat` 0.55, `complex` 0.66.

**Relation space** (true trained pairs; relation probe grouped by head entity; majority chance 0.24; analogy chance
Σp² = 0.11):

| | `two` | `two-16` | `two-64` | `flat` | `complex` | `two` shuffled | `flat` shuffled |
|---|---|---|---|---|---|---|---|
| probe from q | 0.40 | 0.38 | 0.41 | — | — | 0.38 | — |
| probe from [c_a, c_b] | 0.65 | 0.55 | 0.68 | 0.53 | 0.69 | 0.48 | 0.27 |
| analogy (nearest q, same relation) | 0.47 | 0.35 | 0.35 | — | — | 0.41 | — |

## Post-hoc diagnostics (not in the contract)

From the checkpoints (`experiments/everyday_relations_posthoc.py`):

- **T4 order: no distance effect.** By size-rank distance 2…11, `two` ranges 0.44–0.60 and `flat` 0.45–0.62, with no
  trend. Trained adjacent pairs are 0.97–0.99 correct. The MLP arms judge only 10–12% of the held-out pairs true, so the
  reader learned "bigger only for the taught pairs". `complex` judges 46% true, and its accuracy swings with distance
  (0.26 at distance 2, 0.78 at 4, 0.29 at 9): that is a periodic pattern, not an order.
- **T5–T7: the held-out true facts are ranked, but not accepted.** Mean logits for `two`:

  | Test | Held-out true | Held-out false | Trained true | Trained false |
  |---|---|---|---|---|
  | T5 | −13.7 | — | +5.4 | −18.2 |
  | T6 | −12.5 | −16.6 | +6.5 | −20.2 |
  | T7 | −7.1 | −16.6 | +6.6 | −28.0 |

  `flat` is closer on all three (T7 held-out true +4.6, hence its 1.00). `complex` puts T5 held-out at +2.0, its only
  symmetric success.
- **T3a exceptions.** With penguin's exception trained (seeds 400, 402, 403), `can(penguin, fly)` is −3 to −6 in every arm
  and other birds still fly (0.97–1.00): the exception is kept without leaking. In seed 401 the fact fell in the 5%
  validation split. Every arm then said penguin flies (+1.9 to +11.8). That is the class default, the overgeneralization
  children show.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | T1 and T2 ≥ 0.85, well above shuffled, in `two` and `flat` | **Holds** (T1 `default` 0.93, T2 0.88; shuffled 0.51–0.59). `two-64` T2 0.77 |
| 2 | Broad properties are learned before specific ones | **Not readable:** every relation reached 0.9 by the first log at step 250. The logging was too coarse |
| 3 | T3a kept; T3b gets the class default | T3a **holds** (and the class default appears when the exception is untrained). T3b **fails** in `two`: 3 of 4 seeds say the bat flies, so the wings cue beats the class default |
| 4 | T4 > 0.8 with a distance effect; `complex` learns the order | **Fails** in every arm. No distance effect |
| 5 | T5, T6 above chance but below trained facts in `flat`; `two` uncertain | AUROC above 0.5 (0.56–0.79), but at threshold every reversed fact is rejected (T5 0.00) and T6 is 0.50–0.65. **Fails** as a judgment, holds only as a ranking |
| 6 | T7 the weakest structural test: above shuffled, below 0.8 | **Fails for `flat` and `complex`** (1.00: grandparent from the isomorphic tree is easy). `two` 0.75, seeds 0.50–1.00 |
| 7 | T8 above shuffled, below T1 | **Holds weakly:** `two` 0.60 against shuffled 0.47 (+0.13, past the 0.10 margin); below T1 |
| 8 | Fast mapping at k = 1 weak in `two` and `flat` (0.6–0.75), rising with k, tracking the centroid; `complex` higher | `two` weak (0.56) **holds**. `flat` higher than predicted (0.76–0.80). No rise with k. `complex` is not higher (0.55), and its codes sit among the birds without the properties following. So the centroid does not track accuracy. **Mostly fails** |
| 9 | `two-16` loses fit; `two-64` matches `flat` | **Fails:** `two-16` fits fully. `two-64` generalizes worse than `two` (T2 0.77, T7 0.66) |
| 10 | Relation probe from q > 0.9, lower on shuffled | **Fails:** 0.40 against 0.38 shuffled. q is not organized by relation |
| 11 | T9 and shuffled at chance | **Holds** for T9 (0.42–0.51). Shuffled T1 is 0.60, which matches `prior` (0.58): base rates, not structure |

## Interpretation

- **[derived from the results] Attribute sharing is free; relation properties are not.** Inheritance needs only that items
  with shared facts get similar codes. Any embedding with any reader gives that. Order, symmetry and inverses need the
  reader to apply a property of the relation to a pair it never saw. A generic MLP reader (`two`, `flat`) has no reason to
  do so. Trained on a closed world in which most pairs are false, it defaults to false.
- **[derived from the design] T5 is partly built against the property.** For a one-order opposite, `opposite(cold, ·)` is
  trained false for all 15 other adjectives. So "cold has no opposite" is what the head's own training row says. Symmetry
  has to override that, and it does not. T6 has no such conflict (the whole rows are held out), and it still fails at
  threshold.
- **[derived] The relation-agnostic pair space is a bottleneck without a benefit here.** q has to carry everything about a
  pair for every relation, so it does not organize by relation (probe 0.40). It also loses to `flat` on composition and
  fast mapping. Equality v2's two-space advantage came from collections aligning other inputs; one-hot items have no
  such alignment.
- **[analogy; Socher et al. 2013]** The KB literature's bilinear forms help on symmetry (ComplEx T5 0.80, T6 AUROC 0.87). But
  ComplEx also fails on order and fast mapping, so a built-in form alone does not give the abstractions asked for here.
- **For the user's goal** (a new name for an understood concept): fitting one code against a frozen generic reader does
  not carry the concept's properties to the new name. That held even in the arm where the code lands among the right class.

## Computation (measured)

- **Main run:** 13.3 min wall on 8 workers (Ryzen 5 3600). Aggregate worker CPU 6,123 s (1.7 h), about 3.6 CPU min per
  run. The estimate was under 1 CPU h. The difference is the full-batch reads every 250 steps, the fast-mapping fits
  (14 per run) and the LBFGS probes, which the estimate did not count separately.
- Largest child process peak RSS: 294 MiB. Checkpoints about 100 kB per run.
- `prior`: under 1 s. Post-hoc diagnostics: 0.3 s.
- No selection or tuning runs. The 100-step smoke run (10.6 s wall, 36 s CPU) checked only that the runner works.
- The runtime JSON records `dirty: true`: the working tree had untracked documents from other work. The runner, world and
  contract were committed at 6726ffa.

## What the next step could test (for the user to decide)

Following the program's rule to understand a failure before remedying it, the measured failure is:
- the reader rejects every untaught pair;
- the pair space carries no relation structure.

Candidates, none run:
1. **Open-world training.** Train on sampled negatives instead of the full closed world, to see whether the false
   default is the training statistics or the reader.
2. **Relation-conditioned pair space.** Let G see r, so that q is a relation's view of the pair.
3. **Reader-level relation properties learned from examples.** Train symmetry and inverse on some relations, and test
   whether they transfer to a relation where only one direction was taught. That is the program's universal-relation
   question, made explicit.
