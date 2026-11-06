# Conflict search v1: results

Contract: [conflict-search-v1.md](conflict-search-v1.md) (frozen 72511cf, with the smoke-stage amendments recorded
there: R_L fallback, action space = the frozen reader's edges, target-exclusion fix, relation and direction positives
on found paths). Runner `experiments/conflict_search.py`, reports `reports/conflict-search-v1{,-summary,-runtime}.json`.
Seeds 450–453, 4 phase-1 and 16 phase-2 runs, 1,000 phase-2 steps each. Numbers are means over the 4 seeds.

## Summary

Logic after the search beats a trivial baseline only on ancestor. Balanced accuracy (`tf_bal`) counts U as wrong and
weights T and F rows equally, so answering F to everything scores 0.50 and answering U to everything scores 0.00.
Logic's one pass answers U on every withheld relation. After the search: ancestor 0.71 (`bt-forward`) beats 0.50;
grandparent 0.38, kind 0.31–0.43, size 0.13–0.21, named 0.18–0.26 and colour 0.34–0.51 do not. Those numbers are
better than all-U but no better than always-F.

- **On those tests, the score comes from false rows.** An exhausted search gives a learned "against" vote. On a true
  row, logic can say T only if a path was found, so its accuracy on true rows is at most the found rate. Grandparent
  `bt-forward`: found on true rows 0.00 and balanced accuracy 0.38, so about 0.76 of false rows are answered F and no
  true row is answered T. T recall, F recall and U rate were not reported separately; they should be.
- **"True-path rate" is weaker than proof validity.** The read (`search_reads`, `experiments/conflict_search.py`)
  counts a found path as true when every hop is a true individual fact under any relation, withheld relations
  included. It does not check that the hops are permitted base relations, or that their order and directions prove
  the asked relation (two true `parent-of` hops in the wrong directions do not make a grandparent). Even by this loose
  read, found paths are rarely true: `bt-forward` 0.22, `bt-forward-notrace` 0.28, `bt-bidir` 0.01 (prediction ≥ 0.8).
  Proof validity is lower or equal.
- **The action space leaks.** The walk uses the withheld relations themselves and grammar relations (`good-phrase`,
  `and`, `or`) as hops. `bt-bidir` answers `bigger-than` with `>bigger-than <bigger-than|b` (187 found paths in seed 452
  on withheld rows) and named relations with `<named-can >named-can|b`. Every hop is checked by both routes, but
  intuition was trained on the withheld relations, so it confirms those hops: the search brings back intuition's own
  answer through the hop check. The per-hop read (1.00 true, 0.00 false) covered only the base relations, so it did not
  catch this.
- **Ancestor is promising in some seeds, on few rows.** The program is `parent-of` repeated 3–5 times in seeds 451 and
  453 (`bt-forward`; also 450 with 4 paths for `notrace`), loose true-path rate 0.67–0.75. Depth 4–5 has 18 rows per
  seed (9 pairs, each with its reverse). Per seed, logic on those rows and found true pairs at depth 4 / 5 (of 6 / 3):

  | Seed | bt-forward | notrace | bt-bidir |
  |---|---|---|---|
  | 450 | 0.50, 0/6, 0/3 | 0.50, 0/6, 0/3 | 0.50, 3/6, 1/3 |
  | 451 | 0.83, 4/6, 2/3 | 0.78, 4/6, 1/3 | 0.50, 0/6, 0/3 |
  | 452 | 0.61, 2/6, 0/3 | 0.50, 0/6, 0/3 | 0.44, 1/6, 0/3 |
  | 453 | 0.72, 3/6, 1/3 | 0.89, 5/6, 2/3 | 0.61, 3/6, 0/3 |

  "Found" is not "valid proof" (see above). This does not support a general claim about depth.
- **The trace earned nothing at this budget.** `bt-forward-notrace` matches or beats `bt-forward` on every read. The
  ablation removes both the trace and the exclusion of tried candidates, so it does not say which, if either, matters
  in general.
- **The failure is not in the per-hop check of base relations,** which is 1.00 / 0.00. Proposer recall on true rows is
  0.26 (`bt-forward`), 0.16 (`bt-bidir`) and 0.33 (`notrace`), but this read is loose: `World.paths` treats base edges
  as undirected and accepts them in any order, so it does not separate relation choice, entity ranking and budget.
  Which of the three fails is not established.

## Answers (balanced accuracy, logic after the search / arbiter)

Intuition and logic step 0 are the same in every arm (the readers are frozen). Logic step 0 is 0.00 on every withheld
relation (it answers U, the R_L fallback).

| Test | intuition | one-pass | soft-loop | bt-forward | bt-bidir | notrace |
|---|---|---|---|---|---|---|
| P6-grandparent | 0.97 | 0.00 / 0.50 | 0.10 / 0.71 | 0.38 / 0.74 | 0.38 / 0.78 | 0.41 / 0.81 |
| P6-colour | 0.57 | 0.00 / 0.06 | 0.04 / 0.26 | 0.40 / 0.34 | 0.51 / 0.50 | 0.34 / 0.40 |
| L1-anc | 0.72 | 0.00 / 0.16 | 0.10 / 0.45 | 0.71 / 0.68 | 0.49 / 0.67 | 0.74 / 0.76 |
| L1-anc depth 4–5 (18 rows) | 0.54 | 0.00 / 0.03 | 0.11 / 0.33 | 0.67 / 0.53 | 0.51 / 0.53 | 0.67 / 0.64 |
| L1-kind | 0.66 | 0.00 / 0.36 | 0.06 / 0.43 | 0.31 / 0.55 | 0.43 / 0.53 | 0.25 / 0.39 |
| L1-size | 0.96 | 0.01 / 0.45 | 0.03 / 0.52 | 0.17 / 0.65 | 0.21 / 0.61 | 0.13 / 0.56 |
| L2-named | 0.88 | 0.00 / 0.37 | 0.06 / 0.49 | 0.18 / 0.55 | 0.26 / 0.59 | 0.18 / 0.50 |
| P1-default | 0.92 | 0.81 / 0.90 | 0.83 / 0.92 | 0.76 / 0.87 | 0.80 / 0.88 | 0.76 / 0.86 |
| P2 | 0.70 | 0.60 / 0.66 | 0.59 / 0.69 | 0.58 / 0.68 | 0.62 / 0.70 | 0.58 / 0.70 |
| G2 | 0.66 | 0.69 / 0.80 | 0.65 / 0.78 | 0.52 / 0.70 | 0.58 / 0.68 | 0.52 / 0.69 |

The search costs logic up to 0.17 on tests where it was not needed (P1, P4, G2: triggered on 0.18–0.28 of rows). With
whole relations withheld, the arbiter no longer keeps intuition's chains: size 0.96 → 0.45–0.65, named 0.88 → 0.37–0.59,
already in `one-pass`. It follows logic's U (`one-pass` arbiter U-rate 0.55 on size, 0.73 on named).

## Search reads (withheld training rows, `withheld-fit`)

| | bt-forward | bt-bidir | notrace |
|---|---|---|---|
| found, true rows | 0.39 | 0.41 | 0.43 |
| found, false rows | 0.03 | 0.22 | 0.02 |
| exhausted, true rows | 0.61 | 0.59 | 0.58 |
| true-path rate | 0.22 | 0.01 | 0.28 |
| proposer recall | 0.26 | 0.16 | 0.33 |
| first candidate succeeded | 0.70 | 0.77 | 1.00 |
| checks, true / false rows | 7.0 / 8.0 | 5.6 / 4.8 | 5.7 / 9.0 |
| backtracks, true rows | 4.0 | 3.1 | 3.2 |
| path length found | 2.6 | 3.6 | 2.0 |
| backward waste | — | 0.03 | — |
| training found on true rows, step 1,000 | 0.36 | 0.34 | 0.49 |

`notrace` cannot exclude, so a retried candidate counts as the same one: its "first candidate" read is 1.00 by
construction.

Programs (most frequent relation sequences on found paths, per seed) for `bt-forward`: ancestor `parent-of` ×3–5
(seeds 451, 453), otherwise `lives-with`, `has`; size `heavier-than`, `prey-of`, `or`, `bigger-than >lives-with`;
kind `is-a` (seeds 451, 453) or `good-phrase` repeated; named `good-phrase`, `lives-in`, `lives-with`, `heavier-than`;
grandparent almost never found on test rows. `bt-bidir` builds paths from the withheld relations and `lives-with`,
`is-a` in every seed.

## Predictions

| # | Prediction | Result |
|---|---|---|
| 1 | `one-pass` logic near chance on withheld; step 0 unchanged; trigger ≥ 0.8 | **Held.** Step 0 0.00 (U) in every arm, trigger 0.99–1.00 |
| 2 | bt arms beat `one-pass` by ≥ 0.15 on grandparent, size, named, with base relations on ≥ 0.8 of found paths | **Failed.** The margins over `one-pass` are met (grandparent +0.38, size +0.17/+0.21, named +0.18/+0.26), but none beats always-F (0.50), the programs are not base relations, and the score comes from false rows |
| 3 | Ancestor depth 4–5 ≥ 0.7; bidir ≥ forward; proposer recall ≥ 0.9 at depth 1–3 | **Failed.** 0.67 forward, 0.51 bidir; recall 0.16–0.33 overall |
| 4 | True-path rate ≥ 0.8 | **Failed.** 0.22 / 0.01 / 0.28; the hop check admits withheld and grammar relations |
| 5 | `notrace` exhausts ≥ 1.5× and is below `bt-forward` at depth ≥ 4 | **Failed.** Exhausts the same (0.58 vs 0.61), equal or better at depth 4–5 |
| 6 | `soft-loop` gains ≤ 0.08 over `one-pass` and does not pass depth 3 | **Mostly held.** +0.03 to +0.11 logic (ancestor 0.10, depth 4–5 0.11), far below the bt arms |
| 7 | Arbiter keeps intuition's chains within 0.05; U ≥ 0.5 on S4a with ≤ 0.05 on inferable; U after exhaustion ≥ 0.5 | **Mostly failed.** Chains not kept (size, named) in any arm including `one-pass`. S4a U 0.57 / 0.44 / 0.56 / 0.47 / 0.55 (one-pass, soft, fwd, bidir, notrace); inferable U 0.06–0.11; U after exhaustion on P7 0.66 / 0.59 / 0.80 (held) |
| 8 | "Sometimes → unknown" ≤ 0.3 for logic | **Held** (0.07–0.29), as expected for a known limit |
| 9 | Compounding: depth curve no steeper than the per-hop product | **Steeper.** Per-hop 1.00 on base relations, so the reader is not the limit. The prediction's branch names the proposer, but the recall read is too loose to separate relation choice, entity ranking and budget |
| 10 | Stopping rule: found on true training rows < 0.3 at 250 steps | **Would have fired.** The smoke probe was 0.22 at 250 steps, and in the main run 9 of 12 search runs were below 0.3 at both step 200 and step 300 (`bt-forward` 0.15–0.41, `bt-bidir` 0.17–0.36, `notrace` 0.19–0.45). The run continued on the user's explicit direction, recorded in the contract's smoke-stage record. Step-1,000 values (0.34–0.49) do not bear on the rule |

## What this means

1. **Define a successful proof before spending more compute on the search.** Three reads are needed: every hop a true
   fact; every hop a permitted base relation; and a relation-valid proof, with the base relations in the right order
   and directions for the asked relation. The last is an evaluation diagnostic only; the controller is not given it.
2. **Hops should be restricted.** A withheld relation used as a hop checks the question with the route that already
   knows the answer. Grammar statements are not facts between entities. Both should be removed from the action space.
3. **Not finding evidence is not evidence of falsity.** At this F:T balance, a search that never finds anything still
   scores on false rows, and the learned give-up vote turns exhaustion into F. Reads should include T recall, F recall
   and U rate, an exhaustion → U control, and the always-F baseline.
4. **Failures should be isolated with cheap controls before another trained run.** On the same frozen reader graph:
   given (oracle) relation choices, exhaustive enumeration of candidates, and plain bounded exhaustive search, with
   graph construction and inference costs counted. These separate relation selection, entity ranking and budget. The
   action space is already the frozen reader's edges ordered by Q (smoke amendment 3), so that is not a new option.
5. **The arbiter should be compared with intuition alone.** With whole relations withheld it loses much of
   intuition's accuracy on chains (size 0.96 → 0.45–0.65), already in `one-pass`, where logic answers U on every
   withheld row and the arbiter follows it.

## Computation

Theoretical costs as declared in the contract. Measured (`reports/conflict-search-v1-runtime.json`, CPU, float32,
6 workers, 1 thread each):

| Stage | Runs | CPU min total | CPU min per run |
|---|---|---|---|
| phase 1 | 4 | 38.8 | 9.7 |
| one-pass | 4 | 0.1 | 0.0 |
| soft-loop | 4 | 200.0 | 50.0 |
| bt-forward | 4 | 72.4 | 18.1 |
| bt-bidir | 4 | 56.2 | 14.1 |
| bt-forward-notrace | 4 | 50.2 | 12.6 |
| **total** | 24 | 417.7 (7.0 CPU h) | |

Phase-2 wall 66.6 min on 6 workers. Peak RSS per worker 0.37–0.43 GB. Parameters 151,486 in the model, 77,685 in the
controller. Checkpoints about 0.63 MB per run. Reads cost 5–10 CPU s per run. The declared inference cost (reader calls
and cosine scorings per question, wall per 1,000 questions) was not measured by the runner; the checks per question
in the search reads are the nearest figure. No selection or tuning runs: τ, B, k, D_max and the arbiter were fixed by
the contract. The smoke stage cost is recorded in the contract.
