# Generic relations v1: results

Spec: [generic relations v1](generic-relations-v1-spec.md). Run 2026-10-29, seeds 100–102,
paired with the stage 1 standard network (with relation heads). **A pilot, not headline
evidence.**

- **Commands:**
  - `NSY_WORKERS=11 .venv/bin/python -m experiments.generic_relations pilot` (11 minutes);
  - `.venv/bin/python -m experiments.generic_relations_analysis` (30 s).
- **Artifacts:**
  - `reports/generic-relations-v1-pilot-{results.json.gz,runtime.json}`;
  - `reports/generic-relations-v1-comparison.json.gz` and its `-runtime.json`;
  - checkpoints in `runs/generic-relations-v1-pilot/` (not tracked).
- **Recomputation:** the analysis recomputes the generic revision from the checkpoints.
  It matches the runner's value exactly.

## Summary

**Removing the relation-specific heads keeps what is taught and loses what was never
taught.**
- Taught pairings and judgments survive or improve:
  - taps → collection: 0.57 → 0.77;
  - comparison of collections: 0.96 → 0.98;
  - succession judgments: 0.92–0.95;
  - retention.
- What the line counted as assimilation **collapses to chance**:
  - forms never paired with collections are no longer matched (0.42 → 0.13);
  - untaught taps → form drops from 0.79 to 0.04;
  - untaught comparison of forms drops from 0.88 to 0.51.

The prediction written into the spec held, and the collapse is wider than predicted.

**Interpretation [derived from the design, untested].** The heads' fixed forms were the
mechanism of transfer, not decoration:
- a distance judge forces "same" items to be close in the code;
- an affine step forces counts to lie along a path;
- a linear order vector gives every point of the code an order.

Those constraints arrange the shared code geometrically. Anything mapped into it then
inherits the relations. A generic comparator can satisfy every taught judgment without
arranging the code at all, so nothing carries over to items it was not trained on.

## Results

Chance: 0.11 for matching, 0.5 for comparison and AUROC. Outcomes: generic − standard
against the proposed margins.

| Measure | Standard (heads) | Generic | Δ | Outcome |
| --- | --- | --- | --- | --- |
| **Never directly paired forms** | | | | |
| Chinese withheld, own block | 0.42 (0.47, 0.45, 0.35) | 0.13 (0.23, 0.12, 0.05) | −0.29 | worse |
| Devanagari withheld, own block | 0.47 (0.42, 0.43, 0.55) | 0.10 (0.05, 0.18, 0.07) | −0.37 | worse |
| **Untaught compositions** | | | | |
| taps → Chinese form, withheld | 0.79 | 0.04 (0.00, 0.00, 0.13) | −0.75 | worse |
| taps → Devanagari form, withheld | 0.78 | 0.14 | −0.63 | worse |
| form → taps, Chinese withheld | 0.50 | 0.28 | −0.22 | worse |
| Comparison, Chinese forms | 0.88 | 0.51 | −0.37 | worse |
| Comparison, Devanagari forms | 0.79 | 0.55 | −0.24 | worse |
| Comparison, English forms | 0.76 | 0.47 | −0.30 | worse |
| **Taught** | | | | |
| taps → collection | 0.57 (0.61, 0.59, 0.52) | 0.77 (0.77, 0.82, 0.73) | +0.20 | better |
| Comparison, collections | 0.96 | 0.98 | +0.02 | equivalent |
| Succession pairwise, collections | 0.93 | 0.92 | −0.00 | equivalent |
| Succession pairwise, Chinese forms | 0.98 | 0.95 | −0.03 | equivalent |
| **Retention** | | | | |
| Chinese paired, end (D-rehearsal) | 0.50 | 0.57 | +0.07 | inconclusive |
| Chinese paired, end (D-base, no rehearsal) | 0.10 | 0.21 | +0.11 | better |
| English, end | 0.31 | 0.44 | +0.14 | better |
| **Judgment-combined revision** (withheld) | | | | |
| Isolated | 0.44 | 0.12 | −0.31 | worse |
| Revised | 0.62 | 0.72 (0.75, 0.77, 0.63) | +0.10 | inconclusive |
| Swapped item: answer follows the context | 0.29 | 0.66 | +0.36 | better |
| **Next-judgment coherence AUROC** | | | | |
| collections / Chinese / Devanagari / English / taps | 0.89 / 0.95 / 0.96 / 0.76 / 1.00 | 0.93 / 0.90 / 0.98 / 0.82 / 1.00 | +0.04 / −0.05 / +0.01 / +0.07 / 0.00 | mixed, all high |
| **New script** | | | | |
| Thai, 4 shots, after 200 updates | 0.16 | 0.18 | +0.02 | equivalent |

- **Paired forms still match** (Chinese paired at the end: 0.57). The failure is
  specific to items whose meaning had to come through relations.
- **The comparator knows the order of forms but does not place them.** Succession
  judgments on Chinese forms are 0.95, and next-judgment coherence on forms is high. Yet
  withheld forms are not matched to their collections. The *next* knowledge is in `g`,
  not in the code geometry, so it does not reach the *same* judgment.
- **Revision now works almost entirely from the context.** It recovers 0.72 from an
  isolated 0.12. With a swapped item, the answer follows the context in 0.66 of cases.
  The relational knowledge is usable when the context is supplied, but it has not
  become part of the items' own meaning.
- The generic network is better at what it is directly trained on: taught pairing,
  retention and English. This fits a more flexible comparator that is not constrained
  to a geometric form.

## SRI variant: the coupling does not replace the heads

The same replacement on the SRI network ([spec](generic-relations-v1-spec.md#sri-variant-added-2026-10-29-researchers-request)),
paired by seed with the SRI Part D pilot (with heads).
- **Commands:** `NSY_WORKERS=11 .venv/bin/python -m experiments.generic_relations pilot sri`
  (14 minutes), then `experiments.generic_relations_analysis pilot sri`.
- **Artifacts:** `reports/generic-relations-v1-sri-pilot-*` and
  `reports/generic-relations-v1-comparison-sri*`.
- **Recomputation:** matches exactly.

| Measure | SRI with heads | SRI generic | Standard generic (above) |
| --- | --- | --- | --- |
| Chinese withheld, own block | 0.38 | 0.22 (0.37, 0.28, 0.02) | 0.13 |
| Devanagari withheld, own block | 0.46 | 0.05 (0.02, 0.13, 0.00) | 0.10 |
| taps → Chinese form, withheld | 0.76 | 0.02 | 0.04 |
| taps → Devanagari form, withheld | 0.76 | 0.32 (0.33, 0.37, 0.25) | 0.14 |
| form → taps, Chinese withheld | 0.82 | 0.14 | 0.28 |
| Comparison, Chinese / Devanagari / English forms | 0.92 / 0.77 / 0.77 | 0.47 / 0.58 / 0.47 | 0.51 / 0.55 / 0.47 |
| taps → collection (taught) | 0.66 | 0.74 | 0.77 |
| Comparison, collections (taught) | 0.97 | 0.98 | 0.98 |
| Succession pairwise, collections / Chinese forms | 0.93 / 0.97 | 0.92 / 0.95 | 0.92 / 0.95 |
| Judgment revision: isolated / revised, withheld | 0.50 / 0.61 | 0.12 / 0.60 | 0.12 / 0.72 |
| Swapped item: answer follows the context | 0.32 | 0.60 | 0.66 |
| Next-judgment coherence AUROC, collections / Chinese / Devanagari / English | 0.89 / 0.94 / 0.97 / 0.76 | 0.93 / 0.87 / 0.97 / 0.78 | 0.93 / 0.90 / 0.98 / 0.82 |
| Chinese paired, end (D-rehearsal / D-base) | 0.53 / 0.20 | 0.57 / 0.23 | 0.57 / 0.21 |

- **The pattern is the same as in the standard network.**
  - Taught relations survive or improve.
  - Never-paired forms and untaught compositions fall to near chance.
  - Revision works from the context alone.
- **SRI's coupled energy does not supply the arranging pressure the heads supplied.**
  A slight residue may remain:
  - Chinese withheld 0.22 against 0.13, from one seed at 0.37;
  - taps → withheld Devanagari 0.32 against 0.14, in every seed.

  Both are small against the collapse from 0.76–0.92.
- **SRI's cross-sense advantage is gone without the heads.** form → taps is 0.14 against
  0.82 with heads. The hub-resistant geometry seen in the
  [cross-sense confirmation](cross-sense-confirmation-v1-results.md) depended on the
  heads too, or at least on the distance-based matching readout they provided.
- **Cost:** worker CPU 6,089 s for 9 runs (677 s per run), against 7,888 s (876 s per
  run) for SRI with heads on the same arms. Job wall time is 823 s on 11 workers.

## Bearing on the program

1. **Result 1 and result 5 of the [checkpoint](number-line-checkpoint.md)** (meaning
   through relations, untaught compositions) **depend on hand-structured relation forms,
   in both architectures.** With the structure removed, the shared code no longer carries
   the relations, and those results vanish. SRI's coupling does not substitute for the
   structure.
2. **This sharpens H.** The audit found the line had trained knowledge into a shared
   space through the heads. This pilot shows the heads' *structure* was what made that
   knowledge reusable. A generic learner needs some pressure that arranges the shared
   representation so relations become properties of where items sit. That pressure has
   to come from somewhere other than a hand-chosen form for each relation.
3. **Candidate generic sources of that pressure** (not tested; for the researchers'
   decision):
   - **A generic action-conditioned transition** `T(c, action)` trained to *predict the
     next item's code* (a world model). A prediction target in code space forces counts
     onto a path, as `S` did, but the transition is one network for all actions. This is
     the deferred decision in the [spec](generic-relations-v1-spec.md#what-is-lost-and-what-replaces-it),
     and it would also restore composition.
   - **A fixed similarity for "same"** (a distance in code space, as in prototypical
     networks) with a generic comparator only for other relations. This is minimal
     structure, but it is still a choice of form.
   - **Agreement between senses** as the only cross-sense signal, as in the H-test designs
     in the [audit](architecture-audit.md#bearing-on-h).

## Computation

| | Standard (heads), same 3 arms | Generic |
| --- | --- | --- |
| Worker CPU, 9 runs × 6,200 updates (plus 200 transfer updates) | 6,440 s | 5,323 s |
| Per run | 716 s | 591 s |
| Relation parameters | 294 | 6,873 (comparator 6,849 + tokens 24) |

- The generic network costs **less** CPU per run despite its larger relation module. The
  likely cause is that it removes the per-pair distance and affine computations on 9 ×
  batch candidates. This is not profiled.
- Job wall time: 686 s on 11 workers (9 busy).
- Analysis: 30 s wall.
- Selection and tuning: none.

## Caveats

- Three seeds, reused. The size of the collapse (to chance, in every seed) makes the main
  finding unlikely to be seed noise. The improvements on taught measures are smaller and
  less certain.
- One comparator design (concatenation MLP). A comparator with built-in symmetry, or
  difference features, would add back some structure; that is the same question as
  item 3 above.
- Composition, multi-step teaching and imagery are not testable without a transition
  network.
