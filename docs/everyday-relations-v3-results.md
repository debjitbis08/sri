# Everyday relations v3: results

Contract: [everyday relations v3](everyday-relations-v3.md), frozen at commit 5381aec. Run 2026-11-05/06, seeds 420–423,
all 52 runs complete.

- **Command:** `EV_WORKERS=8 .venv/bin/python -m experiments.everyday_relations_v3 main` (32.6 min wall).
- **Artifacts:** `reports/everyday-relations-v3{,-summary,-runtime}.json`; checkpoints in `runs/everyday-relations-v3/`
  (not tracked).

## Summary

1. **Few "no"s moves the threshold; it does not add understanding.** Held-out true facts are accepted more often:
   - symmetry (P2): 0.42–0.50 → 0.63–0.69;
   - inverse (P3): 0.56–0.69 → 0.73–0.84.

   But a third of the never-mentioned false pairs are accepted too (0.35–0.37). Balanced accuracy does not rise, and
   ranking (AUROC) falls: P3 from 0.86–0.92 to 0.79–0.83. Order's distance effect disappears. A child hearing few "no"s
   would be told much more than this world's facts. Here, accepting held-out facts costs as much as it gains.
2. **Word order between categories is learned (G1 0.99–1.00), but that is not evidence of grammar.** The shuffled world
   also scores 0.93–0.97, because word order is a block pattern (every noun row is the same) that survives relabelling.
   Learning it does not require knowing which words are nouns.
3. **A word's category from its meaning (G2) barely beats the controls:**
   - `tworel`/`flat` 0.71–0.75 against shuffled 0.67–0.70 and `prior` 0.63; `two` closed 0.58;
   - a new word naming an owl (`blick-w`): order 0.70–0.96, `is-a(·, noun)` 0.44–0.88.

   Going from "names an owl" to "is a noun" is the same two-step weakness as before.
4. **A word's category from its use (G3) works:**
   - 0.88–0.95, with `is-a(·, noun)` following (0.78–0.94);
   - a new word `zib-w` seen only in "zib runs", "zib swims" and "runs zib" ✗ gets the rest at 0.91–1.00;
   - the margin over shuffled is small in closed world (+0.02–0.06) and larger with few "no"s (+0.13–0.17).
5. **Grammatical but silly (G4) is not separated from sensible.** In closed world, 85–90% of held-out silly pairs
   ("dog flies") are judged good order and not sensible. That passes the contracted read. But held-out *sensible* pairs
   ("eagle flies") are also judged not sensible (accepted 0.25–0.31): `makes-sense` rejects everything untaught. With
   few "no"s it accepts everything (silly rejected only 0.18–0.21). G4 balanced accuracy is 0.51–0.58 everywhere. Order is
   always right (1.00); meaning, which needs `names` plus `can`, is not inferred.
6. **Carried over from v2:** exclusivity in fast mapping (a new bird rejects other classes 1.00 in closed world; 0.88–1.00
   with few "no"s). `tworel` ≈ `flat` > `two`. Opposite from meaning (P5) at chance.

## Results

Balanced accuracy (true-positive rate / true-negative rate where it matters). Mean of 4 seeds. cl = closed world;
fn = few "no"s; sh = shuffled world.

| Test | `two` cl | `tworel` cl | `flat` cl | `two` fn | `tworel` fn | `flat` fn | `complex` fn | shuffled cl | shuffled fn | `prior` |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 untaught word pairs | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 0.97 | 0.97–0.98 | 0.93–0.96 | 0.68 |
| G2 category from meaning | 0.58 | 0.75 | 0.71 | 0.71 | 0.74 | 0.71 | 0.56 | 0.68 | 0.65–0.70 | 0.63 |
| G3 category from use | 0.88 | 0.90 | 0.89 | 0.90 | 0.92 | 0.95 | 0.94 | 0.82–0.88 | 0.74–0.79 | 0.43 |
| G4 silly vs sensible | 0.58 | 0.58 | 0.58 | 0.51 | 0.52 | 0.51 | 0.51 | 0.50 | 0.46–0.61 | 0.50 |
| G4 sensible accepted | 0.25 | 0.25 | 0.31 | 0.84 | 0.84 | 0.84 | 0.97 | | | |
| G4 silly rejected | 0.90 | 0.90 | 0.85 | 0.19 | 0.21 | 0.18 | 0.06 | | | |
| P1 inheritance (`default`) | 0.79 | 0.87 | 0.86 | 0.73 | 0.78 | 0.80 | 0.67 | 0.52–0.58 | 0.49–0.54 | 0.65 |
| P2 symmetry | 0.68 | 0.68 | 0.71 | 0.65 | 0.67 | 0.67 | 0.75 | 0.58–0.61 | 0.52–0.54 | 0.50 |
| P2 true accepted | 0.44 | 0.42 | 0.50 | 0.63 | 0.69 | 0.69 | 0.89 | | | |
| P3 inverse | 0.76 | 0.81 | 0.84 | 0.71 | 0.72 | 0.74 | 0.79 | 0.65–0.68 | 0.55–0.59 | 0.51 |
| P3 true accepted | 0.56 | 0.64 | 0.69 | 0.73 | 0.81 | 0.84 | 0.99 | | | |
| P4 transitivity | 0.62 | 0.75 | 0.78 | 0.64 | 0.63 | 0.63 | 0.49 | 0.62–0.67 | 0.60–0.66 | 0.65 |
| P5 opposite from meaning | 0.49 | 0.48 | 0.49 | 0.47 | 0.40 | 0.49 | 0.60 | 0.48–0.51 | 0.45–0.51 | 0.50 |
| P6 grandparent | 0.50 | 0.98 | 0.97 | 0.59 | 0.62 | 0.61 | 0.66 | 1.00† | 0.66–0.74 | 0.50 |
| P7 not inferable | 0.53 | 0.53 | 0.51 | 0.58 | 0.48 | 0.59 | 0.55 | 0.46–0.51 | 0.53–0.55 | 0.43 |
| False pairs accepted* | 0.00 | 0.00 | 0.00 | 0.37 | 0.35 | 0.35 | 0.44 | 0.00 | 0.42–0.45 | — |

\* Closed world: trained false pairs. Few "no"s: never-mentioned false pairs (mentioned ones: 0.00). † One class
after shuffling; accuracy, not comparable.

**AUROC on P3:**
- closed: 0.86 / 0.91 / 0.92;
- few "no"s: 0.79 / 0.81 / 0.83.

**P4 by rank distance 2 → 7:**
- closed `flat`: 0.72 → 1.00;
- few "no"s `flat`: 0.60–0.69, no trend.

**G2 and G3 by relation** (balanced accuracy):

| | `two` cl | `tworel` cl | `flat` cl | `two` fn | `tworel` fn | `flat` fn |
|---|---|---|---|---|---|---|
| G2 `good-phrase` | 0.54 | 0.75 | 0.72 | 0.65 | 0.75 | 0.70 |
| G2 `is-a` (noun) | 0.46 | 0.58 | 0.47 | 0.58 | 0.66 | 0.67 |
| G2 `makes-sense` | 0.50 | 0.50 | 0.50 | 0.57 | 0.63 | 0.62 |
| G3 `good-phrase` | 0.84 | 0.85 | 0.84 | 0.85 | 0.94 | 1.00 |
| G3 `is-a` (noun) | 0.94 | 0.94 | 0.82 | 0.88 | 0.89 | 0.78 |
| G3 `makes-sense` | 0.50 | 0.56 | 0.62 | 0.84 | 0.89 | 0.88 |

G3 `makes-sense` is mostly word order. The G3 words were taught no meaning (`names` held out). Among their sense queries
the few true ones are noun–verb pairs, so ranking by order alone gives AUROC 0.95–0.99. Its balanced accuracy only shows
where each regime puts the threshold.

**Fast mapping** (code kept among the known codes; no exclusions given; balanced accuracy):

| New | `two` cl | `tworel` cl | `flat` cl | `two` fn | `tworel` fn | `flat` fn | `complex` fn |
|---|---|---|---|---|---|---|---|
| `dax` k = 1 | 0.95 | 0.93 | 0.97 | 0.85 | 0.81 | 0.89 | 0.86 |
| `fep` k = 1 | 0.88 | 0.98 | 1.00 | 0.84 | 0.81 | 0.88 | 0.78 |
| `blick-w` (names owl) | 0.68 | 0.84 | 0.72 | 0.77 | 0.87 | 0.75 | 0.62 |
| `zib-w` k = 3 (from use) | 0.95 | 1.00 | 0.97 | 0.94 | 0.93 | 0.93 | 0.90 |
| `zorp` k = 1 | 0.89 | 0.97 | 0.97 | 0.88 | 0.95 | 0.95 | 0.64 |
| `toasty` k = 4 | 0.52 | 0.65 | 0.60 | 0.83 | 0.48 | 0.45 | 0.76 |
| `kim` k = 4 | 0.67 | 0.80 | 0.77 | 0.48 | 0.57 | 0.46 | 0.58 |

- `blick-w` by relation (`tworel` cl): order 0.84, `is-a` noun 0.88, `makes-sense` 0.75. Other arms: `is-a` 0.44–0.69.
- `zib-w` `is-a` noun: 0.63–1.00.

**New relations:** `smaller-than-v` 0.75–0.96; `both-fly` 0.50–0.66 (as v2).

**Mentioned "no"s:** 1,301 per seed (13 true facts found no allowed corruption). Never mentioned: about 10,400 false
pairs.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Few "no"s: true-positive rate on P2, P3, G1 up ≥ 0.2; balanced accuracy up; unmentioned false acceptance < 0.15; P5 at chance | True-positive rate up: P2 +0.19 to +0.27, P3 +0.15 to +0.17 (**partly**). Balanced accuracy **does not rise** (P3 falls 0.05–0.10). False acceptance 0.35–0.37 (**fails**). P5 at chance (**holds**) |
| 2 | G1 ≥ 0.9, above shuffled by ≥ 0.3 | ≥ 0.9 **holds**. Margin over shuffled 0.02–0.06 (**fails**): word order between categories survives relabelling |
| 3 | G2 weakest, 0.6–0.8 in training, lower for `blick-w` | **Holds** as weakest (0.58–0.75), barely above shuffled and `prior`. `blick-w` is not lower (0.68–0.87) |
| 4 | G3 ≥ 0.85, with `is-a(·, noun)` following | **Holds** (0.88–0.95; `is-a` 0.78–0.94). Adjective positions were not read separately |
| 5 | G4 dissociation ≥ 0.7 in `tworel` and `flat`; `two` lower | Nominally **holds** in closed world (0.85–0.90), but `two` is equal (0.90), and the read is an artefact: held-out sensible pairs are rejected too. G4 balanced accuracy 0.51–0.58. **Fails in substance** |
| 6 | P1, exclusivity and P7 as v2 in both regimes | Closed **holds**. Few "no"s lowers P1 (0.73–0.80 against 0.79–0.87) and exclusivity (0.88–1.00) |

## Interpretation

- **[read: results] Few "no"s trades misses for false alarms.** The held-out facts were already ranked above false pairs
  in closed world. Few "no"s lowers the bar for everything, held-out true facts and never-mentioned false pairs alike.
  In this world a held-out fact looks like a never-mentioned pair, so no regime of labels alone separates them. What
  separates them is structure the network must use, and AUROC shows it uses less of it with fewer "no"s.
- **[derived from the design] The shuffled control cannot test category-level patterns inside one relation.** Word order
  (G1) and the use-based part of G3 are block patterns within `good-phrase`. Relabelling heads and tails preserves
  blocks. A control that breaks them would have to shuffle within a relation's rows separately. These tests show the
  pattern is learned, not that it is tied to word meaning.
- **[read: results] The two-step weakness is now the common thread:**
  - category from meaning (G2: `names` then `is-a` of the named thing);
  - sense of a phrase (G4: `names` then `can`);
  - same colour (P6: `colour-of` both ways);
  - opposite from meaning (P5: `is` then scale end).

  Single-relation patterns and properties shared across many relations are learned. Inferences that chain two different
  relations through a middle entity are not, except grandparent, where the isomorphic family tree supplies a copy.
- **[analogy; child language] Use-based categories are easier here than meaning-based ones.** That matches the
  distributional (Elman) and syntactic-bootstrapping direction (G3), not semantic bootstrapping (G2). In this system the
  reason is the two-step weakness, not a claim about children.

## Computation (measured)

- 52 runs on 8 workers: 32.6 min wall; 15,568 s aggregate worker CPU (5.0 CPU min per run; estimate 5–7).
- Peak RSS per worker is under 400 MiB.
- Smoke: 24 s wall.
- No tuning or selection runs.
- The runtime file's `dirty` flag is set by uncommitted tracked files from other work (`docs/computation-reporting.md`,
  `docs/prior-work.md` and others), not by this study's code.

## What the next step could test (for the user to decide)

1. **Two-step inference as its own question.** Every remaining failure chains two relations through a middle entity
   (word → animal → ability). Options:
   - teach many two-step chains, so that chaining is a pattern learned across relations (the meta-learning direction);
   - let the reader take more than one step (read an intermediate code and read again). That is an architecture change.
2. **A control that breaks category blocks within a relation,** for G1 and G3.
3. **Cardinality** ("each pole has one opposite"), still open from v2.
