# Where the count lives in the code (erasure-stage checkpoints)

Date: 2026-10-30. **Post hoc analysis of saved final checkpoints, seeds 230–239; no training, no inference tests.**
Motivated by the [erasure results](fan-in-pilot-v1-erasure-results.md) ("count becomes readable from single
coordinates of form codes") and the draft [limited-memory proposal](limited-memory-proposal.md).

- **Command:** `NFI_THREADS=2 NFI_CHANNELS_LAST=1 .venv/bin/python -m experiments.fan_in_count_coordinate pilot erasure`.
- **Artifacts:** `reports/fan-in-count-coordinate-erasure.json.gz` and `-runtime.json` (34 s wall, 16 s CPU, 80 checkpoints).
- **Terms:** a coordinate is one of the 16 entries of the shared code. Each seed trains its own code, so coordinate
  indices are not comparable across seeds; every quantity below is within a seed, then averaged.

## Findings

Means over 10 seeds. Orders are pairwise order accuracy of an affine count map fitted on collection codes only.

| Arm | Linear-direction alignment | Order, mean over all 16 coordinates (taps / English / Chinese / Devanagari) | Order at the coordinate chosen on collections |
| --- | --- | --- | --- |
| dense | 0.52 | 0.71 / 0.52 / 0.59 / 0.60 | 0.91 / 0.50 / 0.59 / 0.65 |
| random untied readers | 0.53 | 0.74 / 0.54 / 0.61 / 0.61 | 1.00 / 0.54 / 0.58 / 0.65 |
| reset in place | 0.52 | 0.81 / 0.65 / 0.76 / 0.69 | 0.83 / 0.68 / 0.76 / 0.72 |
| random rewire | 0.52 | 0.95 / 0.71 / 0.90 / 0.77 | 0.90 / 0.72 / 0.91 / 0.77 |
| rewire, keep weight | 0.53 | 0.99 / 0.64 / 0.70 / 0.73 | 1.00 / 0.65 / 0.75 / 0.74 |
| rewire, double rate | 0.55 | 0.98 / 0.75 / 0.92 / 0.79 | 1.00 / 0.75 / 0.93 / 0.78 |

(The English / Chinese / Devanagari columns in the first printed run of the script were mislabelled; the script now
prints the names. Dense-in-place and half-rate arms are in the artifact.)

1. **The count is distributed, not carried by one coordinate.** The unit direction of the collection → count map has
   its largest component at about 0.52 (0.25 would be spread evenly over 16, 1.0 one coordinate) in every arm, dense
   included, and on average 13–16 of 16 coordinates have collection R² ≥ 0.5 in every arm. Any single coordinate orders
   collections almost perfectly (best possible 1.00). "Readable from single coordinates" is a property of most
   coordinates, not of a special one.
2. **The arms differ in whether each coordinate's count map transfers to taps and forms.** Dense and random-reader
   coordinates order collections but are near chance on Chinese and English forms (0.52–0.61 averaged over
   coordinates). Random rewire lifts the average coordinate to 0.90 on Chinese forms and 0.95 on taps. The gain is in
   the average coordinate; the best-on-collections coordinate is only slightly better than the average in the
   rewire arms. Most coordinates have become count-monotone in the same direction across senses.
3. **The readers do not locate the count.** The share of reads (both operands, first layer of `g`) falling on the
   coordinate with the best collection R² is 0.06–0.09, against 1/16 = 0.0625 by chance; the rank correlation of reads
   with collection R² across coordinates is about 0 (±0.1) in the rewire arms. The coordinate best on a given set
   equals the coordinate best on collections in 0–30% of runs. The effect is not "readers find the count axis".
4. **Retention cannot be examined.** Only final checkpoints are saved.

## Reading (not tested)

Consistent with the data: a reader whose connections are repeatedly disrupted cannot depend on any particular
coordinate, so the encoder is pushed to make every coordinate carry the count in a form that transfers: redundancy
across coordinates, not a localized feature. Dense readers see all coordinates at every step and have no such
pressure; dense reset at the same rate gains nothing because there is no sampling of coordinates to disrupt. This
reading predicts that (a) the transfer gain should fall if readers could not be disrupted per coordinate, and (b)
average-over-coordinates transfer, not best-coordinate transfer, is the right measure. Only (b) is available now.

## Consequences for the memory proposal

- Measure count readability averaged over all coordinates, and under random reader ablation, not at a chosen
  coordinate; the draft's per-unit story (which dense unit keeps what) has no support in this analysis.
- Save checkpoints before and after the delay and after relearning; this analysis shows retention cannot be studied
  from final checkpoints alone.
- Decay should be compared with reset in place and rewire, because the erasure and rewire arms are the established
  version of "disrupt sparse connections"; decay of dense units is not.

## Limits

Post hoc; descriptive; ten seeds, no intervals. R² ≥ 0.5 is an arbitrary descriptive threshold. The alignment
statistic uses raw codes and depends on coordinate scales. The mechanism above is an interpretation, not a result.
