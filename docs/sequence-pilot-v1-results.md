# Sequence prediction pilot v1: results

Spec: [sequence pilot v1](sequence-pilot-v1-spec.md). Run 2026-10-29, seeds 100–102,
paired with the heads network, the generic network and the
[transition pilot](transition-pilot-v1-results.md). **A pilot, not headline evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.sequence_pilot pilot` (11 minutes);
  - `.venv/bin/python -m experiments.transition_pilot_analysis pilot sequence` (30 s).
- **Artifacts:**
  - `reports/sequence-pilot-v1-pilot-{results.json.gz,runtime.json}`;
  - `reports/sequence-pilot-v1-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/sequence-pilot-v1-pilot/` (not tracked).
- **Recomputation:** matches exactly.

## Summary

**Next-item prediction without action tokens recovers part of what the heads gave: more
than the generic network, less than the action-token transition network.** This is as
predicted in the spec.
- **Placement of never-paired forms:**
  - Chinese 0.31 (0.47, 0.28, 0.17) and Devanagari 0.23 (0.22, 0.27, 0.20);
  - against 0.13 and 0.10 in the generic network, and 0.37 and 0.26 in the transition
    pilot;
  - Devanagari is steadier across seeds than in the transition pilot.
- **Untaught comparison** recovers for Devanagari forms (0.65) and partly for Chinese
  (0.58), not for English (0.49).
- **Untaught cross-sense matching:**
  - taps → withheld Devanagari recovers on average (0.44) but varies widely by seed
    (0.80, 0.43, 0.08);
  - taps → withheld Chinese does not recover (0.00 in every seed).
- **Direction is inferred from history.** One predicted step lands near the right
  collection about half the time, going up and going down. Rolled-out predictions
  degrade fast, especially downward.
- **Imagined states are again not readable by `g`.**

## Results

Chance: 0.11 for matching, 0.5 for comparison and AUROC. The outcome column compares the
sequence pilot with the generic network, using the proposed margins, paired by seed.

| Measure | Heads | Generic | Transition (tokens) | Sequence (no tokens; seeds) | vs generic |
| --- | --- | --- | --- | --- | --- |
| **Never directly paired forms** | | | | | |
| Chinese withheld, own block | 0.42 | 0.13 | 0.37 | 0.31 (0.47, 0.28, 0.17) | better |
| Devanagari withheld, own block | 0.47 | 0.10 | 0.26 | 0.23 (0.22, 0.27, 0.20) | better |
| **Untaught compositions** | | | | | |
| taps → Chinese form, withheld | 0.79 | 0.04 | 0.14 | 0.00 (0.00, 0.00, 0.00) | inconclusive |
| taps → Devanagari form, withheld | 0.78 | 0.14 | 0.51 | 0.44 (0.80, 0.43, 0.08) | better |
| form → taps, Chinese withheld | 0.50 | 0.28 | 0.31 | 0.31 (0.03, 0.60, 0.28) | inconclusive |
| Comparison, Chinese forms | 0.88 | 0.51 | 0.73 | 0.58 (0.65, 0.57, 0.51) | inconclusive |
| Comparison, Devanagari forms | 0.79 | 0.55 | 0.66 | 0.65 (0.69, 0.63, 0.63) | better |
| Comparison, English forms | 0.76 | 0.47 | 0.61 | 0.49 (0.48, 0.47, 0.51) | equivalent |
| **Taught** | | | | | |
| taps → collection | 0.57 | 0.77 | 0.66 | 0.69 | worse |
| Chinese paired, own block | 0.57 | 0.72 | 0.61 | 0.70 | inconclusive |
| Comparison, collections | 0.96 | 0.98 | 0.98 | 0.98 | equivalent |
| Succession pairwise, collections | 0.93 | 0.92 | 0.92 | 0.91 | equivalent |
| **Retention** | | | | | |
| Chinese paired, end (D-rehearsal) | 0.50 | 0.57 | 0.63 | 0.61 | inconclusive |
| English, end | 0.31 | 0.44 | 0.43 | 0.42 | inconclusive |
| **Judgment revision**, withheld | | | | | |
| Isolated | 0.44 | 0.12 | 0.35 | 0.25 | better |
| Revised | 0.62 | 0.72 | 0.71 | 0.65 | inconclusive |
| **Next-judgment coherence AUROC** (Chinese / English) | 0.95 / 0.76 | 0.90 / 0.82 | 0.93 / 0.82 | 0.91 / 0.78 | — |
| Thai, 4 shots, after 200 updates | 0.16 | 0.18 | 0.19 | 0.15 | inconclusive |

## Composition: predicted steps

From two real items, `P` is rolled forward k steps and the result matched among 9
collections, by distance (D-varied+rehearsal; chance 0.11).

| Start | up k=1 | up k=3 | down k=1 | down k=3 |
| --- | --- | --- | --- | --- |
| Collections | 0.59 | 0.31 | 0.47 | 0.26 |
| Taps | 0.63 | 0.33 | 0.70 | 0.06 |
| Chinese forms | 0.51 | 0.23 | 0.45 | 0.07 |
| Devanagari forms | 0.48 | 0.37 | 0.50 | 0.11 |
| English forms | 0.48 | 0.25 | 0.30 | 0.12 |

- **One step works in both directions,** so the history carries the direction. Rolled-out
  predictions degrade quickly, more so downward (30% of training episodes descend).
  Predicted states feed back as history, so errors compound.
- **Read by `g(same)`**, every cell is 0.00–0.23. `g` does not read predicted states,
  as in the transition pilot.

## Interpretation

- **Removing the action tokens costs some recovery.**
  - Withheld placement: Chinese 0.37 → 0.31, Devanagari 0.26 → 0.23.
  - Untaught comparison: English 0.61 → 0.49, Chinese 0.73 → 0.58.
  - taps → withheld Chinese drops to 0.00.

  Two candidate reasons, not separated: about half as many prediction rows per batch;
  and withheld forms appear only as history, not as sources of predictions in both
  directions.
- **A tokenless predictive objective still supplies some arranging pressure.** Placement
  and some untaught transfer rise clearly above the generic network.
- **The pattern of the last three pilots:**
  - transfer needs the shared code arranged;
  - a predictive objective in code space arranges it partly;
  - the more training rows a withheld item takes part in, the more it is placed.

## Computation

| | Generic | Transition (tokens) | Sequence (no tokens) |
| --- | --- | --- | --- |
| Worker CPU, 9 runs | 5,323 s | 5,438 s | 5,512 s |
| Per run | 591 s | 604 s | 612 s |
| Job wall time (11 workers) | 686 s | 694 s | 688 s |
| Added parameters | — | 6,817 | 7,313 |

- Analysis: 30 s.
- Selection and tuning: none.

## Caveats and next questions

- Three seeds, reused. Several measures vary widely by seed (taps → withheld Devanagari:
  0.08–0.80).
- **Open questions:**
  1. **More prediction rows without tokens:**
     - predict every item, not only collections, as the next target;
     - longer histories.

     Would that close the gap to the transition pilot?
  2. **Readable imagined states:** train `g` on predicted states too, or read composition
     by the shared metric.
  3. **Multi-step rollouts in training,** to reduce compounding.
