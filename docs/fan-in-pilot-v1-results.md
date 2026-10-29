# Fan-in pressure pilot v1, stage 1: results

Proposal: [fan-in pressure](fan-in-pressure-proposal.md). Run 2026-10-29, seeds 100–102,
D-varied+rehearsal, paired by seed with the [sequence pilot](sequence-pilot-v1-results.md).
**A pilot, not headline evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.fan_in_pilot pilot` (32 minutes);
  - `.venv/bin/python -m experiments.fan_in_pilot_analysis pilot` (36 s).
- **Artifacts:**
  - `reports/fan-in-pilot-v1-pilot-{results.json.gz,runtime.json}`;
  - `reports/fan-in-pilot-v1-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/fan-in-pilot-v1-pilot/` (not tracked).
- **Implementation check:** the dense configuration reproduces the sequence pilot's
  D-varied+rehearsal results exactly in all three seeds.

## Summary

**By the frozen rule, no configuration advances to stage 2.** One configuration shows a
consistent partial signal, in the predicted place.

- **Limiting the readers to one code dimension per hidden unit (readers, φ = 1/16)
  improves untaught comparison on every script:**
  - Chinese forms 0.58 → 0.81 (0.79, 0.81, 0.81 against 0.65, 0.57, 0.51);
  - Devanagari 0.65 → 0.70, English 0.49 → 0.54.

  Chinese comparison exceeds the transition pilot's 0.73 and approaches the heads
  network's 0.88, without heads or action tokens. Placement of withheld forms rises on
  average (Chinese 0.31 → 0.43, Devanagari 0.23 → 0.29) but not consistently across seeds.
  It fails the advance rule on one primary measure: taps → withheld Devanagari forms falls
  0.44 → 0.27 (the dense values vary widely, 0.80, 0.43, 0.08).
- **Limiting the encoder hurts.** At φ = 1/16, placement of withheld Devanagari forms and
  comparison on Chinese and Devanagari forms are worse, while taught measures hold or
  improve. Transfer falls without underfitting.
- **Combining both limits is worse than readers alone:** the encoder limit's harm
  dominates.
- **The arrangement probe does not improve with reader limits.** The gain is in how `g`
  reads the code, not in a code whose count is more linearly placed for forms.

## Results

Means over seeds 100–102; outcome against the dense rerun (paired, margins 0.05 for
accuracies and 0.03 for AUROCs): **bet**ter, **wor**se, **equ**ivalent, **inc**onclusive.
Chance: 0.11 for matching, 0.5 for comparison, order and AUROC.

| Measure | Dense | Enc 1/4 | Enc 1/16 | Read 1/4 | Read 1/16 | Both 1/4 | Both 1/16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Primary: never directly paired forms** | | | | | | | |
| Chinese withheld, own block | 0.31 | 0.19 inc | 0.23 inc | 0.35 inc | 0.43 inc | 0.18 wor | 0.15 wor |
| Devanagari withheld, own block | 0.23 | 0.22 inc | 0.06 wor | 0.33 bet | 0.29 inc | 0.21 inc | 0.14 wor |
| **Primary: untaught compositions** | | | | | | | |
| taps → Chinese form, withheld | 0.00 | 0.01 equ | 0.11 inc | 0.04 inc | 0.26 inc | 0.00 equ | 0.12 inc |
| taps → Devanagari form, withheld | 0.44 | 0.20 wor | 0.18 inc | 0.36 inc | 0.27 wor | 0.43 inc | 0.21 inc |
| form → taps, Chinese withheld | 0.31 | 0.17 inc | 0.24 inc | 0.27 inc | 0.29 inc | 0.35 inc | 0.24 inc |
| Comparison, Chinese forms | 0.58 | 0.55 inc | 0.47 wor | 0.64 inc | **0.81 bet** | 0.50 inc | 0.49 wor |
| Comparison, Devanagari forms | 0.65 | 0.63 inc | 0.56 wor | 0.68 inc | **0.70 bet** | 0.66 inc | 0.60 inc |
| Comparison, English forms | 0.49 | 0.48 equ | 0.46 inc | 0.50 equ | **0.54 bet** | 0.49 equ | 0.48 equ |
| **Fit check: taught** | | | | | | | |
| Chinese paired, own block | 0.70 | 0.68 inc | 0.76 bet | 0.67 inc | 0.66 inc | 0.69 equ | 0.64 wor |
| taps → collection | 0.69 | 0.64 inc | 0.74 bet | 0.75 bet | 0.67 equ | 0.69 inc | 0.74 inc |
| Comparison, collections | 0.98 | 0.98 equ | 0.98 equ | 0.98 equ | 0.97 equ | 0.98 equ | 0.98 equ |
| Succession pairwise, collections | 0.91 | 0.93 equ | 0.94 equ | 0.90 equ | 0.92 equ | 0.93 equ | 0.94 equ |
| **Secondary** | | | | | | | |
| Chinese paired, end | 0.60 | 0.62 inc | 0.68 bet | 0.63 inc | 0.58 inc | 0.63 inc | 0.64 inc |
| English, end | 0.42 | 0.45 inc | 0.50 inc | 0.43 inc | 0.35 wor | 0.47 inc | 0.40 inc |
| Judgment revision: isolated, withheld | 0.25 | 0.20 inc | 0.12 wor | 0.34 bet | 0.33 inc | 0.20 inc | 0.14 wor |
| Judgment revision: revised, withheld | 0.65 | 0.67 inc | 0.72 inc | 0.68 inc | 0.69 inc | 0.70 inc | 0.71 bet |
| Next-judgment coherence AUROC, Chinese | 0.91 | 0.93 inc | 0.91 inc | 0.94 inc | 0.94 inc | 0.91 inc | 0.90 inc |
| Next-judgment coherence AUROC, English | 0.78 | 0.79 inc | 0.78 inc | 0.80 inc | 0.83 bet | 0.81 inc | 0.72 wor |
| Thai, 4 shots, after 200 updates | 0.15 | 0.19 inc | 0.21 bet | 0.16 equ | 0.13 inc | 0.20 inc | 0.20 inc |

Per seed, for the two reader configurations:

| Measure | Dense | Read 1/4 | Read 1/16 |
| --- | --- | --- | --- |
| Chinese withheld, own block | 0.47, 0.28, 0.17 | 0.37, 0.62, 0.07 | 0.40, 0.35, 0.55 |
| Devanagari withheld, own block | 0.22, 0.27, 0.20 | 0.23, 0.32, 0.45 | 0.32, 0.48, 0.07 |
| taps → Chinese form, withheld | 0.00, 0.00, 0.00 | 0.00, 0.00, 0.13 | 0.05, 0.00, 0.73 |
| taps → Devanagari form, withheld | 0.80, 0.43, 0.08 | 0.68, 0.08, 0.30 | 0.55, 0.22, 0.05 |
| Comparison, Chinese forms | 0.65, 0.57, 0.51 | 0.62, 0.71, 0.60 | 0.79, 0.81, 0.81 |

### Decision

| Configuration | Underfit on | Better on (primary) | Worse on (primary) | Advance |
| --- | --- | --- | --- | --- |
| Enc 1/4 | — | — | taps → Devanagari | no |
| Enc 1/16 | — | — | Devanagari placement; comparison, Chinese and Devanagari | no |
| Read 1/4 | — | Devanagari placement | — | no (one measure) |
| Read 1/16 | — | comparison, all three scripts | taps → Devanagari | no |
| Both 1/4 | — | — | Chinese placement | no |
| Both 1/16 | Chinese paired | — | Chinese and Devanagari placement; comparison, Chinese | no |

### Arrangement probe and diagnostics

Ridge regression of count from collection codes, fitted on collections and applied to
other codes unchanged. Order: pairwise order accuracy of predicted counts.

| Measure | Dense | Enc 1/4 | Enc 1/16 | Read 1/4 | Read 1/16 | Both 1/4 | Both 1/16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Probe order, Chinese forms | 0.65 | 0.57 wor | 0.52 wor | 0.61 inc | 0.62 inc | 0.51 wor | 0.39 wor |
| Probe order, Devanagari forms | 0.66 | 0.60 wor | 0.59 wor | 0.65 equ | 0.62 inc | 0.63 inc | 0.59 wor |
| Probe order, English forms | 0.51 | 0.47 inc | 0.48 inc | 0.50 equ | 0.56 inc | 0.53 inc | 0.51 equ |
| Probe order, taps | 0.99 | 1.00 equ | 0.66 inc | 0.58 inc | 1.00 equ | 1.00 equ | 0.66 inc |
| Probe count error, Chinese withheld | 3.71 | 4.28 | 4.85 | 3.38 | 2.97 | 4.46 | 5.21 |
| Probe count error, Devanagari withheld | 1.66 | 1.74 | 1.73 | 2.04 | 1.68 | 1.70 | 1.67 |
| Code participation ratio (collections) | 1.36 | 1.31 | 1.36 | 1.11 | 1.15 | 1.16 | 1.26 |
| `g` first layer, units never active | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `P` first layer, units never active | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |

- Every limited layer has exactly its declared edge count (for example `proj` 1,024 →
  256 → 64; `g` first layer 2,560 → 1,024 → 640 including the token columns).
- **Deviation from the proposal:** dead units are measured as first-layer units never
  active on 4,000 random pairs of evaluation codes (and, for `g`, every relation token),
  not as units without gradient in the last 10% of training.
- The participation ratio is near 1 in every configuration: the collection codes' variance
  lies mostly along one direction. Reader limits concentrate it further.

## Interpretation

- **The prediction about sites held.** Readers helped more than the encoder; the encoder
  at 1/4 was close to dense (one primary measure worse). The prediction that the encoder
  at 1/16 would underfit was wrong: taught measures held while transfer fell. Restricting
  how the code is *formed* harms its arrangement for forms (probe order falls with every
  encoder limit).
- **Reader limits help comparison, not by arranging the code more.** At one dimension per
  hidden unit, `g` must build its answer from per-dimension comparisons of the two items.
  Untaught comparison on forms improves in every seed for Chinese, yet the probe does not
  improve. The likely reading: a per-dimension comparator applies the same comparison to
  form codes as to collection codes, where a freely mixing comparator learned
  collection-specific directions. This is the
  ["more flexible answerer arranges less"](think-pilot-v1-results.md#interpretation)
  pattern seen from the other side, but the gain here is in the reader's systematicity
  rather than in the code.
- **Cross-sense matching and placement remain noisy.** Seed variance is larger than most
  differences (dense taps → Devanagari ranges 0.08 to 0.80). Three seeds cannot resolve
  them.
- **Not established:** that reader limits improve placement; that any effect survives
  fresh seeds; whether the gain comes from the limit or from the same-subset tying across
  operands (a per-dimension comparison design), since the two were not separated.

## Computation

| | Value |
| --- | --- |
| Runs | 21 (7 configurations × 3 seeds), two waves on 11 workers |
| Worker CPU, all runs | 19,490 s |
| Worker CPU per run, mean by configuration | dense 979 s; enc 1/4 951; enc 1/16 952; read 1/4 956; read 1/16 902; both 1/4 879; both 1/16 878 |
| Job wall time | 1,899 s (data 25 s, rendering 3 s, runs 1,872 s) |
| Peak RSS, largest worker | 1.19 GB |
| Analysis | 36 s wall, 16 s parent CPU (checkpoint reloads, generic measures, probe, diagnostics) |
| Selection and tuning | none |

- Masked layers use dense weights times a fixed mask, so per-update arithmetic is the same
  as the dense network's. Per-run CPU differences between configurations are within the
  spread expected from running 11 workers at once; the dense rerun's mean (979 s) is
  higher than the sequence pilot's (583–661 s for the same runs at 9 concurrent tasks).
- Effective parameters in the limited layers are given in the proposal's computation plan.

## Caveats

- Three seeds, reused from the earlier pilots. The rule was applied as frozen; a
  confirmation on fresh seeds is needed before any claim.
- One curriculum arm (D-varied+rehearsal).
- Fixed random masks, one draw per seed and layer.
