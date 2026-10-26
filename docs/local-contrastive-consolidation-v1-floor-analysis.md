# Local contrastive consolidation v1: post-hoc noise-floor analysis

Post-hoc analysis, 2026-10-26. It is not part of the frozen contract or its
declared criteria, and it changes none of the pass/fail outcomes reported in
the [results](local-contrastive-consolidation-v1-results.md). It came out of
the [prior-work review](prior-work.md). In this fixture the best attainable
error can be derived, so raw RMSE ratios can hide large differences in
avoidable error.

[Script](../experiments/local_contrastive_floor_analysis.py) ·
[results](../reports/local-contrastive-consolidation-v1-floor-analysis-results.json) ·
[runtime](../reports/local-contrastive-consolidation-v1-floor-analysis-runtime.json).
Reproduce with `.venv/bin/python -m experiments.local_contrastive_floor_analysis`.
The script reads the committed v1 results JSON and recomputes every final
parameter set's 32-step predictions on the same test sets. It asserts that
observed RMSE matches the recorded values to 1e-12 for every arm, frozen
baseline and ridge fit.

## The floor and the noiseless mean

In v1's `fixture(seed)`, each module observes an amplitude a ~ U[0.8,1.2]
times one column of its private orthonormal 8×2 map. Nuisance is added only in
the 6-dimensional complement of both columns, independently per module. The
noiseless value μ of a hidden module is therefore its projection onto its own
pattern span, and an oracle that knows the maps can recover label and amplitude
exactly from the observed modules.

Writing the hidden target as y = μ + ε, with prediction ŷ:

    ‖ŷ − y‖² − ‖ε‖² = ‖ŷ − μ‖² − 2(ŷ − μ)ᵀε

So three quantities are reported per seed:

- **observed MSE**: ‖ŷ − y‖²;
- **noise-subtracted MSE**: observed minus the sample floor ‖ε‖². The cross
  term vanishes only in expectation, so on a finite test sample this can be
  near zero or negative.
- **avoidable MSE**: ‖ŷ − μ‖², computed with evaluator-only maps. It is
  non-negative and is the robust measure.

The sample floor has mean RMSE 0.0861 over seeds 3-12, range 0.0829 to
0.0889. Its expected value is √(0.06/8) = 0.0866.

## Results

Means over seeds 3-12, MSE per coordinate:

| Arm | Observed RMSE | Noise-subtracted | Avoidable | Avoidable range | Avoidable vs 2a |
| --- | ---: | ---: | ---: | --- | ---: |
| 2a Clamped energy, C+C | 0.0919 | 0.00102 | 0.00112 | 0.00101-0.00122 | 1.00× |
| 2b Clamped energy, 9C+9C | 0.0921 | 0.00105 | 0.00114 | 0.00102-0.00125 | 1.01× |
| 10 Ridge | 0.0979 | 0.00217 | 0.00231 | 0.00191-0.00267 | 2.05× |
| 3 Two-phase, uniform | 0.0989 | 0.00237 | 0.00241 | 0.00234-0.00255 | 2.14× |
| 6 Exact-gradient reference | 0.0990 | 0.00238 | 0.00241 | 0.00235-0.00254 | 2.15× |
| 5 Two-phase, no outer | 0.1048 | 0.00357 | 0.00363 | 0.00331-0.00384 | 3.23× |
| 5b Two-phase, outer-only chain | 0.1213 | 0.00731 | 0.00734 | 0.00632-0.00841 | 6.53× |
| 9 B-only, uniform slowing | 0.1237 | 0.00807 | 0.00820 | 0.00507-0.01517 | 7.29× |
| 4 Two-phase, inner only | 0.1276 | 0.00887 | 0.00892 | 0.00819-0.00939 | 7.93× |
| 8 B-only, importance inertia | 0.1409 | 0.01258 | 0.01258 | 0.00807-0.02059 | 11.2× |
| 7 B-only, no protection | 0.1791 | 0.02512 | 0.02522 | 0.01554-0.04077 | 22.4× |
| 1 Frozen | 0.3662 | 0.12688 | 0.12673 | 0.10015-0.13714 | 113× |

Here the cross term is small: mean absolute value at most 1.4e-4, and no
noise-subtracted value is negative. The two measures agree to within about
10% for the best arms. That is a property of these samples, not a guarantee.

## Reading

- The headline conclusions stand. Clamped energy is best, and the two-phase
  rule's final error matches the exact-gradient reference.
- Measured as avoidable error, differences are much larger than raw RMSE
  ratios suggest:
  - The two-phase rule has 2.1× clamped energy's avoidable error; the raw
    RMSE ratio is 1.08×.
  - The topology arms, left unresolved by the frozen S1 thresholds on raw RMSE
    (1.06-1.29×), have 1.5×, 3.0× and 3.7× arm 3's avoidable error (arms 5,
    5b and 4).
  - Ridge is not a ceiling; its shrinkage leaves twice clamped energy's
    avoidable error.
  - Every arm that replays both structures removes more than 92% of frozen
    avoidable error.
- Near the floor, raw-RMSE thresholds are weak tests: 1.25× raw RMSE from arm
  2a corresponds to about 5× its avoidable error. These observations are post
  hoc and do not revise the declared criteria. Future contracts on this
  fixture should declare criteria on avoidable error.

## Computation

Per parameter set: three whole-module settles, each one O(S³)
factorization (S ≤ 42) plus 32 O(nS²) steps on n = 24 test examples. The run
performed 330 settles and scored 120 parameter sets including ridge fits:
0.19 s wall, 0.19 s CPU, peak RSS 64 MB on one thread. Exact counts, hashes of
inputs and outputs, and versions are in the runtime JSON.
