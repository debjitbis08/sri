# Transition pilot v1: results

Spec: [transition pilot v1](transition-pilot-v1-spec.md). Run 2026-10-29, seeds 100–102,
paired with the standard network with heads (stage 1) and the generic network (generic
relations v1). **A pilot, not headline evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.transition_pilot pilot` (12 minutes);
  - `.venv/bin/python -m experiments.transition_pilot_analysis pilot` (30 s).
- **Artifacts:**
  - `reports/transition-pilot-v1-pilot-{results.json.gz,runtime.json}`;
  - `reports/transition-pilot-v1-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/transition-pilot-v1-pilot/` (not tracked).
- **Recomputation:** matches exactly.

## Summary

**A generic predictive transition network partly restores what the heads gave.**
- **Placement of never-paired Chinese forms recovers almost fully,** in every seed:
  0.37 against 0.13 generic and 0.42 with heads.
- **Untaught comparison on forms recovers partly:** Chinese 0.73, Devanagari 0.66,
  English 0.61, against about 0.5 generic and 0.76–0.88 with heads.
- **One untaught cross-sense composition recovers:** taps → withheld Devanagari, 0.51
  against 0.14 generic.
- **Taught relations and retention stay at the generic network's level,** which is above
  the heads' on retention.
- **Recovery is uneven across seeds and languages.** Devanagari placement is 0.18, 0.53
  and 0.07 across seeds. taps → withheld Chinese recovers in one seed only.
  form → taps does not recover.
- **Composition works in code space but not through `g`.** Imagined states land near the
  right collection by distance: 0.58–0.62 for one step on collections and taps, falling
  with more steps. `g`'s *same* judgment does not recognize them (about 0.2).

The prediction written into the spec was right about placement and partial cross-sense
recovery. It was **wrong about comparison:** that recovered partly too. The code that `T`
arranges is regular enough for `g`'s *more* judgment to generalize to forms in some seeds.

## Results

Chance: 0.11 for matching, 0.5 for comparison and AUROC. Outcome labels compare the
transition network with each other network, using the proposed margins (paired by seed).

| Measure | Heads | Generic | Generic + transition (seeds) | vs heads | vs generic |
| --- | --- | --- | --- | --- | --- |
| **Never directly paired forms** | | | | | |
| Chinese withheld, own block | 0.42 | 0.13 | **0.37** (0.38, 0.38, 0.35) | inconclusive | better |
| Devanagari withheld, own block | 0.47 | 0.10 | 0.26 (0.18, 0.53, 0.07) | inconclusive | inconclusive |
| **Untaught compositions** | | | | | |
| taps → Chinese form, withheld | 0.79 | 0.04 | 0.14 (0.00, 0.00, 0.42) | worse | inconclusive |
| taps → Devanagari form, withheld | 0.78 | 0.14 | **0.51** (0.47, 0.50, 0.57) | worse | better |
| form → taps, Chinese withheld | 0.50 | 0.28 | 0.31 (0.37, 0.00, 0.57) | inconclusive | inconclusive |
| Comparison, Chinese forms | 0.88 | 0.51 | **0.73** (0.74, 0.50, 0.94) | inconclusive | inconclusive |
| Comparison, Devanagari forms | 0.79 | 0.55 | **0.66** (0.71, 0.57, 0.70) | worse | better |
| Comparison, English forms | 0.76 | 0.47 | **0.61** (0.73, 0.48, 0.61) | worse | inconclusive |
| **Taught** | | | | | |
| taps → collection | 0.57 | 0.77 | 0.66 | better | worse |
| Chinese paired, own block | 0.57 | 0.72 | 0.61 | inconclusive | worse |
| Comparison, collections | 0.96 | 0.98 | 0.98 | equivalent | equivalent |
| Succession pairwise, collections | 0.93 | 0.92 | 0.92 | equivalent | equivalent |
| **Retention** | | | | | |
| Chinese paired, end (D-rehearsal) | 0.50 | 0.57 | 0.63 | better | inconclusive |
| English, end | 0.31 | 0.44 | 0.43 | better | inconclusive |
| **Judgment revision**, withheld | | | | | |
| Isolated | 0.44 | 0.12 | 0.35 (0.31, 0.55, 0.18) | inconclusive | better |
| Revised | 0.62 | 0.72 | 0.71 | better | equivalent |
| Swapped item: answer follows the context | 0.29 | 0.66 | 0.63 | better | inconclusive |
| **Next-judgment coherence AUROC** (Chinese / English) | 0.95 / 0.76 | 0.90 / 0.82 | 0.93 / 0.82 | — | — |
| Thai, 4 shots, after 200 updates | 0.16 | 0.18 | 0.19 | equivalent | equivalent |

## Composition: imagined steps

`T` is applied k times to an item's code, and the result is matched among 9 collections.
D-varied+rehearsal; chance 0.11.

| Start | Read by | add k=1 | add k=2 | add k=3 | remove k=1 | remove k=2 | remove k=3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Collections | distance | 0.62 | 0.46 | 0.34 | 0.58 | 0.45 | 0.28 |
| Taps | distance | 0.59 | 0.46 | 0.22 | 0.60 | 0.22 | 0.10 |
| Chinese forms | distance | 0.54 | 0.40 | 0.32 | 0.56 | 0.40 | 0.24 |
| Devanagari forms | distance | 0.50 | 0.39 | 0.34 | 0.50 | 0.33 | 0.22 |
| English forms | distance | 0.41 | 0.37 | 0.36 | 0.41 | 0.29 | 0.27 |
| Collections | `g(same)` | 0.19 | 0.19 | 0.19 | 0.22 | 0.20 | 0.18 |
| Chinese forms | `g(same)` | 0.22 | 0.16 | 0.17 | 0.24 | 0.28 | 0.28 |

- **One imagined step from a form lands near the right collection** about half the time,
  for every language. This includes withheld forms, and Chinese and English after their
  blocks. Errors compound over steps.
- **`g` cannot read imagined states.** It was trained only on codes of real items, and
  `T`'s outputs are near collections in distance but not where `g` expects real items.
  Composition through the system's own answer mechanism therefore fails. Reading it
  needs either `g` trained on imagined states, or a shared metric readout.
- Not directly comparable with the heads network's composition measures (crossings
  through `S` and the *more* head; [multi-step results](number-construction-v1-multistep-results.md)).

## Interpretation

- **A generic world-model objective supplies some of the arranging pressure.** It does
  this without any hand-shaped relation form, which supports the generic-relations
  diagnosis: transfer needs the code arranged, and predicting the next item's code
  arranges it.
- **It is weaker and less reliable than the heads.** The heads impose their structure
  on every pair. `T` only constrains items through the transitions they take part in,
  and the arrangement it induces varies between seeds.
- **The remaining structure is in the prediction loss's distance.** That is one metric
  for the whole space, not one per relation. The readout gap for imagined states shows
  that this metric and `g` are not yet aligned.

## Computation

| | Generic (generic relations v1) | Generic + transition |
| --- | --- | --- |
| Worker CPU, 9 runs × 6,200 updates (plus 200 transfer updates) | 5,323 s | 5,438 s |
| Per run | 591 s | 604 s |
| Job wall time (11 workers) | 686 s | 694 s |
| Added parameters | — | 6,817 |

- Analysis: 30 s.
- Selection and tuning: none.

## Caveats and next questions

- Three seeds, reused. Devanagari placement and several compositions vary widely by seed.
- **Open questions:**
  1. **Can `g` read imagined states?** For example, train `g(same)` also on `T`'s
     predictions, or read composition by the shared metric.
  2. **Are multi-step targets needed?** Training `T^k` on k-step endpoints, as the
     multi-step objective did for `S`, might help.
  3. **Does seed variability fall with more transitions,** for example withheld forms in
     more varied episodes?
