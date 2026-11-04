# Shared output v1, K arms: pilot results

Status: **pilot of the [K-arm extension](shared-output-v1-k-arms.md), run 2026-11-04, seeds 300–303, two new arms.** `gp` and
`common-raw` come from the [shared output pilot](shared-output-v1-pilot.md). Not headline evidence. The pre-declared
criteria fail for `common-k-shared`, the arm built to transfer.

- **Commands:** see the extension's *Commands*. Tables:
  `.venv/bin/python -m experiments.shared_output_summary reports/shared-output-v1-pilot.json reports/shared-output-v1-pilot-k.json`.
- **Reports:** `reports/shared-output-v1-pilot-k.json` and `-runtime.json`; the reproduction in
  `reports/shared-output-v1-pilot-repro.json` and `reports/shared-output-v1-repro-check.json`.
- **Reproduction check:** `common-raw` at seed 300, rerun after the readout refactor, equals the pilot in **3,281 of 3,281
  measures** (6 added substitutability measures excluded). Reuse of the pilot's `gp` and `common-raw` stands.
- **Reader parameters:** 14,018 in both K arms, against 13,003 (`gp`) and 13,025 (`common-raw`).

## Summary

- **Neither K arm meets the transfer criterion.** At delay, `common-k-shared` − `common-raw` on N1 is **−0.05** (seeds −0.15,
  +0.04, +0.15, −0.24). `common-k-cand` − `common-raw` is +0.03 (−0.09 to +0.17).
- **The shared K does what was derived, but only while naming lasts.** At block 3 names land nearest their own count more often
  in K space than in core space (0.60 against 0.44). By the delay this has faded to 0.28, along with identification (0.63 → 0.30).
  N1 given correct identification is unchanged (0.77 against `common-raw`'s 0.83). The limit is retention of the name, not
  reading it.
- **The shared K is more exposed to relearning, as predicted (4).** After relearning, `common-k-shared`'s familiar continuation
  falls to 0.55 (others 0.81–0.94), composition up 1 to 0.72, and taps → collection to 0.76.
- **`common-k-cand`, the proposal as written, is the best raw arm on rejection and retention.** It is not better on transfer.
  These comparisons are reported attribution, not pre-declared advance criteria:
  - **Rejection.** With the correct item removed, names abstain at **0.44** at delay, against 0.13 for `gp` and 0.11 for
    `common-raw` (paired +0.31 over `gp`, every seed positive). Reverse completions are accepted at 0.03.
  - **Name identification** at delay is 0.48, against 0.37 (`gp`) and 0.24 (`common-raw`).
  - **Earlier abilities hold or improve.** Devanagari matching at delay is 0.77 (`gp` 0.50, `common-raw` 0.27). Composition up 1
    after relearning is 0.95 (`gp` 0.83). English after relearning is 0.66 against `gp`'s 0.68.
  - **N1 over `gp`:** +0.11 at block 3 (every seed positive) and +0.07 at delay (3 of 4 seeds). This is below the +0.10 bar at
    delay, and within noise of `common-raw`.
- **Never-taught symbols are still accepted:** 0.94 in `common-k-cand` and 1.00 in `common-k-shared`, at delay.

## Results

Mean over four seeds, range in brackets. Chance is 1/9 for recall. Abstention below the frozen threshold counts as wrong.

**Naming and transfer**

| Read | Point | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|---|
| Name identification, counts 2–8 | block 3 | **0.82** | 0.59 | 0.76 (0.71–0.87) | 0.63 (0.54–0.70) |
| | delay | 0.37 | 0.24 | **0.48** (0.31–0.65) | 0.30 |
| | relearn | 0.21 | 0.20 | **0.32** | 0.28 |
| N1, mean of directions | block 3 | 0.47 | 0.53 | **0.58** (0.53–0.63) | 0.49 |
| | delay | 0.32 | 0.36 | **0.39** (0.32–0.53) | 0.31 (0.16–0.40) |
| | relearn | 0.22 | 0.29 | **0.30** | 0.18 |
| N1 given correct identification | block 3 | 0.55 | 0.70 | 0.70 | 0.66 |
| | delay | 0.78 | **0.83** | 0.67 | 0.77 |
| Substitutability, names nearest own count (2–8), core / K | block 3 | — | — | 0.52 / 0.61 | 0.44 / 0.60 |
| | delay | — | — | 0.34 / 0.40 | 0.26 / 0.28 |
| Familiar (collection, collection) | delay | 0.93 | 0.94 | **0.96** | 0.90 |
| | relearn | 0.81 | **0.94** | 0.91 | 0.55 |
| Never-taught symbols accepted | delay | 1.00 | 1.00 | 0.94 (0.80–1.00) | 1.00 |

**Rejection, at delay**

| Read | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|
| Names: correct item removed, abstains | 0.13 | 0.11 | **0.44** (0.27–0.77) | **0.44** (0.24–0.57) |
| Names: reverse completion accepted | 0.08 | 0.14 | 0.03 | 0.05 |
| Names: repeated count accepted | 0.16 | 0.21 | 0.13 | 0.11 |
| Names: correct completion accepted | 0.37 | 0.44 | 0.46 | 0.34 |
| Names: AUROC against reverse and repeat | 0.68 | 0.60 | **0.74** | 0.64 |
| Familiar: correct item removed, abstains | 0.49 | 0.44 | 0.51 | 0.47 |

At relearn, `common-k-shared`'s name abstention with the correct item removed falls to 0.04. `common-k-cand` keeps 0.36.

**Earlier abilities**

| Read | Point | `gp` | `common-raw` | `common-k-cand` | `common-k-shared` |
|---|---|---|---|---|---|
| Devanagari matching, paired | delay | 0.50 | 0.27 | **0.77** | 0.49 |
| Devanagari matching, withheld | delay | 0.72 | 0.41 | **0.77** | 0.55 |
| Taps → Devanagari, withheld | delay | 0.67 | 0.38 | **0.83** | 0.77 |
| Comparison, Devanagari | delay | 0.84 | 0.79 | 0.95 | **0.97** |
| Composition, collections up 1 | relearn | 0.83 | 0.88 | **0.95** | 0.72 |
| Composition, collections up 2 | relearn | 0.69 | 0.77 | **0.94** | 0.55 |
| English matching, paired | relearn | **0.68** | 0.46 | 0.66 | 0.49 |
| Collection code spread, between / within | delay | 20.5 | 19.7 | 15.0 | 17.7 |

**Pilot criteria, K arms** (pre-declared, at delay)
- **Gates.** `common-k-cand` fails identification at block 3 (0.76 against 0.80). `common-k-shared` fails it too (0.63). Both
  pass familiar continuation (0.96, 0.90). By the rule, N1 given identification is read alongside.
- **Transfer.** `common-k-shared` − `common-raw` is −0.05, so it fails.
- **Rejection.** `common-k-shared`'s removed-set false acceptance is 0.31 *below* `gp`'s, so it passes.
- **Prior abilities.** These fail: withheld Devanagari matching is −0.17 against `gp`, and English after relearning −0.19.
- **Attribution, N1 at delay.**
  - `common-k-cand` − `common-raw` (a key tower on candidates): +0.03.
  - `common-k-shared` − `common-k-cand` (K in the slots): −0.08 (3 of 4 seeds negative).
  - `common-k-shared` − `gp`: −0.01.

## What this shows

1. **Prediction 1 holds.** A key tower on candidates leaves naming transfer where `common-raw` had it.
2. **Prediction 2 holds only at block 3.** The shared K raises K-space substitutability while names are being taught. That does
   not survive the delay, because nothing rehearses names. The pilot's finding stands: **identification fades without
   rehearsal**, and transfer follows identification. Changing the readout space does not change that.
3. **Prediction 3 (bi-encoder costs familiar accuracy) fails for `common-k-cand`.** It matches or beats the cross-encoder on
   familiar reads. Only `common-k-shared` loses some (0.83 against 0.88 at block 3).
4. **Prediction 4 holds strongly.** In `common-k-shared` every slot passes through K, so relearning's movement reaches every
   relation through one place.
5. **The gain in `common-k-cand` is in rejection, identification and protection of earlier abilities, not in transfer.** One
   reading is that the dot product separates the two operands. Candidates are fixed points in K space and queries move to them,
   so relearning that moves Q leaves the candidates' layout intact. This is not tested here. A direct read would be the drift of
   K(M) across relearning against the drift of Q.
6. **The proposal's own critical experiment was not run.** Recall and acceptance agree by construction because they read one
   logit. Testing whether learning to predict improves verification needs an arm that trains only one of them.

## Open

- **Whether `common-k-cand`'s rejection and retention gains survive confirmation** (seeds 424–433). That would need a criterion
  declared for that arm now, since none was declared for it before this pilot.
- **Why names fade.** Every arm loses identification after block 3 (0.82 → 0.37 in `gp`). Transfer is bounded by it.
- **Training one read and testing the other,** the proposal's own test.
- **Rejection of never-taught symbols,** still untrained.

## Computation

- **Runs.** 8 K-arm runs on 3 GPU workers (GTX 1650): 888 s wall, 2,275 s aggregate worker CPU. `common-k-cand` took
  275–277 s per run and `common-k-shared` 273–309 s. `common-raw` took 230–261 s in the pilot. GPU peak allocated memory was
  358 MB per worker.
- **Selection and checks.** The reproduction run took 232 s on 1 worker, and the smoke run 56 s on 2 workers. There was no
  tuning.
- **Storage.** 14,018 reader parameters (56 KB float32) per arm, plus Adam's two moments. Cached keys for M are 4.6 KB.
