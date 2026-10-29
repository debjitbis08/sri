# Architecture audit, stage 1: the number line on a standard network

Protocol: [architecture audit, stage 1](architecture-audit.md#stage-1-decided-2026-10-29-running).
Run 2026-10-29. Three seeds (100–102), the same seeds as the SRI pilots. **A pilot, not
headline evidence.**

- **Commands** (`NSY_WORKERS=11`, then `.venv/bin/python -m`):
  - `experiments.standard_baseline pilot <step>`, for each step in `part-d` (20 minutes),
    `multistep` (18 minutes), `composition-d`, `projection-d`, `composition-m`,
    `projection-m` (under a minute each) and `imagery` (13 minutes);
  - then `experiments.standard_baseline_compare` and `experiments.standard_baseline_costs`.
- **Artifacts:**
  - `reports/number-construction-v1-standard-{part-d,multistep,imagery}-pilot-*`;
  - `reports/number-construction-v1-{composition,projection}-pilot-standard-{part-d,multistep}-pilot-*`;
  - `reports/architecture-audit-stage1-comparison.json`;
  - `reports/architecture-audit-stage1-costs.json`, and its `-runtime.json` for timings;
  - checkpoints in `runs/number-construction-v1-standard-*-pilot/` (not tracked).
- **Model** (`src/standard_net.py`): the same CNN front ends, one `nn.Linear(64, 16)` per
  sense and the same relation heads. No modules, no energy, no output module.
  - Everything else runs unchanged: data streams, losses, curriculum, rehearsal,
    evaluations and encoder initialization. Each run therefore pairs with its SRI pilot
    run.
  - Revision is code-space least squares with scalar evidence weight 1.
  - Imagery uses a ridge decoder from codes to features.
  - The energy confidence feature is unavailable. The runner records it as zeros; read
    it as null.
- **Outcome labels:** paired differences (standard − SRI) against the proposed margins:
  0.05 for accuracies, 0.03 for AUROCs, not yet confirmed by the researchers.
  - *Equivalent:* every seed within the margin.
  - *Better* or *worse:* a higher or lower standard value, with the mean beyond the
    margin and all seeds in the same direction. For the random-state control, higher is
    worse.
  - *Inconclusive:* otherwise.

## Summary

| Checkpoint result | On the standard network |
| --- | --- |
| 1. Withheld forms get meaning through relations | **reproduced** (0.42 / 0.47 against 0.38 / 0.46) |
| 2. Revision through relations recovers answers | **reproduced, larger gain** (+0.18 against +0.08); responds less to a swapped context |
| 3. Rehearsal against forgetting | **reproduced** (gain +0.40 against +0.34) |
| 4. Varied contexts beat repeated ones | **not reproduced**: repeated is better (−0.09 in every seed) |
| 5. Untaught compositions | **mostly reproduced**: taps → form and comparison on forms equal; form → taps on withheld Chinese **worse** (0.50 against 0.82) |
| 6. Successor residuals flag violations | **reproduced** (AUROCs within 0.01) |
| 7. Imagery toward the right count | within a sense, reproduced or better; across senses, worse. **A probe only, not evidence for or against an architecture** (researchers, 2026-10-29) |
| 8. Composition needs multi-step teaching | **the premise does not hold:** the standard control already composes (0.71–0.72) |

Most of the line's results, the strongest ones included, do not need the SRI
architecture. Imagery is excluded from the assessment (researchers, 2026-10-29: it was a
probe). Without it, three differences favour SRI, all post hoc on three reused seeds:
- form → taps matching for withheld Chinese forms (0.82 against 0.50);
- less forgetting without rehearsal;
- a larger multi-step gain on withheld forms. In two places, SRI produced the problem a later study solved: short successor
steps, and a weak repeated-context arm.

## 1. Results reproduced without SRI

| Measure | SRI | Standard | Outcome |
| --- | --- | --- | --- |
| Chinese withheld, own block | 0.38 (0.40, 0.40, 0.35) | 0.42 (0.47, 0.45, 0.35) | inconclusive |
| Devanagari withheld, own block | 0.46 (0.50, 0.42, 0.47) | 0.47 (0.42, 0.43, 0.55) | inconclusive |
| Rehearsal: English, end | 0.28 | 0.27 | equivalent |
| Rehearsal: Chinese paired, end | 0.53 | 0.50 | inconclusive |
| taps → Chinese form, withheld | 0.76 (0.65, 0.70, 0.92) | 0.79 (0.68, 0.82, 0.88) | inconclusive |
| taps → Devanagari form, withheld | 0.76 | 0.78 | inconclusive |
| Comparison, Chinese / Devanagari / English forms | 0.92 / 0.77 / 0.77 | 0.88 / 0.79 / 0.76 | inconclusive / equivalent / inconclusive |
| Coherence AUROC, collections / Chinese / Devanagari / English | 0.89 / 0.94 / 0.97 / 0.76 | 0.89 / 0.95 / 0.96 / 0.76 | equivalent or inconclusive, all within 0.01 |

- "Inconclusive" here means seed spread beyond the margin. It does not mean a
  consistent difference: mean differences are 0.04 or less.
- The untaught compositions are what alignment through a shared anchor sense predicts
  (the ImageBind pattern; [audit](architecture-audit.md#standard-explanations-for-each-result)).
  They appear without the coupled modules.

## 2. Revision

| Measure (withheld forms, end) | SRI | Standard |
| --- | --- | --- |
| Isolated | 0.50 (0.52, 0.52, 0.46) | 0.44 (0.41, 0.46, 0.45) |
| Two-sided revision | 0.58 (0.55, 0.67, 0.52) | 0.62 (0.61, 0.62, 0.62) |
| Gain | +0.08 (0.03, 0.15, 0.06) | **+0.18** (0.20, 0.16, 0.17) |
| Swapped item: answer follows the context | 0.58 | 0.34 |

- Revision helps more in the standard network, from a lower isolated level.
- With a swapped item, the standard revision follows the context less. This mixes two
  differences: the revision estimator (scalar evidence weight against SRI's
  direction-dependent precision `P`; [audit](architecture-audit.md#what-the-current-network-computes),
  fact 3) and the trained codes. Stage 2 (`P`-weighted against scalar revision on the
  same frozen SRI codes) separates them.

## 3. Varied against repeated contexts: not reproduced

| Chinese withheld, end | SRI | Standard |
| --- | --- | --- |
| D-varied+rehearsal | 0.53 (0.53, 0.62, 0.45) | 0.41 (0.40, 0.48, 0.35) |
| D-repeated+rehearsal | 0.37 (0.57, 0.22, 0.33) | 0.51 (0.52, 0.53, 0.47) |
| Varied − repeated | +0.16 (−0.03, 0.40, 0.12) | **−0.09** (−0.12, −0.05, −0.12) |

- In the standard network, repeated contexts are better than varied ones in every seed.
  Repeated contexts are also not worse than no contexts: Chinese withheld at the end is
  0.51, against 0.47 for D-rehearsal.
- SRI's repeated arm was the weak one, with a wide seed spread (0.57, 0.22, 0.33). Its
  advantage for varied contexts was not consistent across seeds (one of three negative).
- The Part D finding "varied beats repeated; repeated is worse than none" is therefore
  not a property of the training design. It appeared only in SRI, and there not reliably.

## 4. Cross-sense differences

Imagery rows are recorded for completeness. They do not count toward the assessment:
imagery was a probe of what core states look like, not a capability claim.

| Measure | SRI | Standard | Outcome |
| --- | --- | --- | --- |
| form → taps, Chinese withheld | 0.82 (0.90, 0.72, 0.85) | 0.50 (0.37, 0.57, 0.57) | worse |
| taps → Chinese form, paired | 0.91 | 0.83 | inconclusive |
| Imagery: taps prototype → image sense | 0.81 (0.89, 0.78, 0.78) | 0.30 (0.33, 0.11, 0.44) | worse |
| Imagery: Chinese prototype → sequence sense | 0.93 (1.00, 0.89, 0.89) | 0.63 (0.56, 0.67, 0.67) | worse |
| Imagery: collection prototype → sequence sense | 0.70 | 0.30 | — |
| Imagery: gap → sequence sense | 0.71 | 0.14 | — |
| Imagery: imagined successor → sequence sense | 0.71 | 0.38 | — |

Imagery rows show the share of counts whose nearest real input has the clamped count
(D-varied+rehearsal).

- **Within a sense, standard imagery is as good or better:**
  - collection prototype → image: 0.70 against 0.70;
  - Chinese prototype → image: 1.00 against 0.81;
  - Devanagari prototype → image: 0.93 against 0.78;
  - taps prototype → sequence: 1.00 in both.
- **Across senses, it is worse throughout.** States from one sense, or inferred states,
  decoded into the *other* sense's features land on the right count far less often.
- The imagery difference mixes the network and the decoder. SRI uses its own reverse
  conditional, the joint Gaussian across senses through the core. The standard network
  uses a separate ridge regression per sense, fitted on that sense's inputs. Stage 2 can
  separate these by fitting the same ridge decoder to frozen SRI codes.
- form → taps uses forward codes only, with no decoder. So that difference comes from
  training. Taps → form on the same items is equal (0.79 against 0.76), so the gap is in
  one direction only. Not explained.
- The random-state control is higher in the standard network (0.15 against 0.04): its
  random code states are more often near some count.

## 5. Composition: the short-step problem was SRI's

| Forward strict crossings | SRI | Standard |
| --- | --- | --- |
| M-control, k = 2 | 0.53 (0.41, 0.44, 0.72) | **0.71** (0.73, 0.67, 0.74) |
| M-control, k = 3 | 0.54 (0.35, 0.43, 0.84) | **0.72** (0.67, 0.70, 0.79) |
| M-multistep, k = 2 | 0.68 | 0.65 |
| M-multistep, k = 3 | 0.69 (0.64, 0.76, 0.67) | 0.67 (0.67, 0.75, 0.59) |
| M-multistep, shown result k = 3 | 0.74 | 0.75 |
| Advance ratio, k = 1, M-control (collections) | 0.38 (0.17, 0.18, 0.80) | 1.81 (3.98, 0.59, 0.85) |
| Advance ratio, k = 1, M-multistep | 0.95 | 0.71 |

- **Every standard control seed composes** at the level SRI reached only with the
  multi-step objective. Seed 100's successor overshoots (advance ratio 3.98) and still
  crosses correctly (0.73).
- The multi-step objective gives the standard network nothing on crossings (−0.05 at
  k = 3). In SRI, it repaired short steps that SRI's own training produced.
- The multi-step gain on withheld forms is smaller in the standard network: Devanagari
  withheld, own block, 0.47 → 0.50 against 0.46 → 0.60 in SRI. At the multi-step arm, SRI
  is ahead in every seed (0.60 against 0.50).
- **Equality through the comparison head flattens the successor in both networks**
  (advance ratio at k = 1: −0.04 SRI, 0.10 standard; crossings 0.35–0.46). That is a
  property of the head design.

## 6. Forgetting without rehearsal

| D-base, end | SRI | Standard |
| --- | --- | --- |
| Chinese paired | 0.20 (0.19, 0.25, 0.15) | 0.10 (0.10, 0.12, 0.09) |
| Chinese withheld | 0.12 | 0.07 |
| Devanagari paired (last block) | 0.40 | 0.61 |

Without rehearsal, the standard network learns the current language better and forgets
the earlier one more. This is a stability–plasticity difference, not a capability
difference. With rehearsal, the two networks retain equally.

## Computation

From `reports/architecture-audit-stage1-costs.json` and `-costs-runtime.json`. SRI
figures are from the frozen pilots' runtime files, run on the same machine with the
same worker count.

| | SRI | Standard |
| --- | --- | --- |
| Parameters: encoders / perception / relation heads | 356,032 / 26,368 / 294 | 356,032 / 2,080 / 294 |
| Parameters, total (float32 bytes; Adam state bytes) | 382,694 (1.53 MB; 3.06 MB) | 358,406 (1.43 MB; 2.87 MB) |
| Part D: training CPU, 15 runs × 6,200 updates | 12,622 s | 10,524 s |
| Part D: job wall time, 11 workers | 1,501 s | 1,200 s |
| Multi-step: training CPU, 12 runs × 6,000 updates | 12,215 s | 10,093 s |
| Multi-step: job wall time | 1,439 s | 1,101 s |
| Peak RSS, largest worker | 1.19 GB | 1.18 GB |
| Perception, batch of 270 (features → code), median | 1,635 µs (image), 1,545 µs (sequence) | 15.6 µs |
| SRI solver construction (Cholesky of the 272-state system), median | 1,168 µs (image), 1,154 µs (sequence) | — |
| Imagery analysis, worker CPU | 7,071 s | 7,196 s |

- Training costs about 17% less CPU on the standard network. The CNN front ends dominate
  both.
- Perception alone is about 100 times cheaper in the standard network. The microbenchmark
  is one thread, float32, 20 warm-up and 200 timed repeats, quartiles within 3% of the
  median.
- Selection and tuning: none. The evidence weight (1) and the ridge strength (1e-3) were
  fixed before running, and no candidates were scored.
- Rehearsal keeps a buffer of 500 episode records per language in both networks.

## Caveats

1. **Seeds reused.** Seeds 100–102 are the SRI pilots' seeds, chosen and amended after
   their results were seen. Three seeds cannot establish equivalence, and the margins
   are still proposals.
2. **Revision and imagery differ in mechanism as well as in training** (sections 2 and 4).
   Stage 2's comparisons on frozen SRI codes separate the two.
3. **The SRI-favouring differences are post hoc.** They were not predicted, and they need
   fresh seeds before being treated as findings.
4. **Learning effort is compared only at matched training.** Both networks see the same
   6,200 updates on the same data, and the standard network uses about 17% less CPU for
   them. Evaluations exist only at the end of each 2,000-update block, so the updates
   needed to reach a given level are not measured. Two partial indications:
   - training matching losses track closely in every block (D-varied+rehearsal, Chinese
     block, update 100 → 2,000: SRI 1.17 → 0.35, standard 1.18 → 0.30);
   - 4-shot Thai transfer stays near chance in both (0.13 against 0.16 after 200 updates).
5. **The relation heads and the core-code readout are kept** to audit existing results
   (researchers' decisions 1 and 2 in the [audit](architecture-audit.md)).

## Bearing on the audit

- The checkpoint's results are largely **sufficient on a standard shared-embedding
  network**: results 1, 2, 3, 6 and most of 5. Result 4 does not hold there. Result 8's
  premise was an SRI training effect.
- With imagery excluded, the remaining candidate SRI-specific signals are:
  - form → taps matching for withheld forms;
  - retention without rehearsal;
  - the multi-step gain on withheld forms.

  None is established. Each rests on three reused seeds and was not predicted.
- **Next steps proposed:**
  1. Stage 2 on frozen checkpoints, reduced to revision: exact `P`-weighted revision
     against scalar revision on the same SRI codes. This decides whether the
     swap-following difference is mechanism or training. The imagery comparison is
     dropped.
  2. Fresh seeds for SRI and standard on the three candidate measures, to decide whether
     any SRI advantage is real.
