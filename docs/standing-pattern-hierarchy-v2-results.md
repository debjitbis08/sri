# Graded stored-pattern hierarchy: results and interpretation

The requested two-module hierarchy was implemented with coupling at every layer,
weak outside and strong inside. It transmits input to a separate outer output,
uses stored local relationships to reconstruct detail, and produces usable partial
responses before settling. However, **identity coupling between corresponding
layers biases the output toward the input pattern**. Faster response did not mean
better final interpretation.

## What was actually tested

Each module has 8→4→2→1 continuous activities. Neighbor averages pass inward;
neighbor differences are omitted from that inward coordinate. This exactly
removes the pairwise antisymmetric Haar mode, not every high Fourier frequency.
Fourteen local quadratic relationships were fitted from nine paired examples.
Each predicts the preferred child difference from its parent value:

```
f(c) = w0 + w1*c + w2*c²
preferred children = (c + f(c), c - f(c))
```

The energy penalizes disagreement with these relationships and disagreement
between corresponding units of the two modules. Per-coordinate coupling strengths
are 0.05, 0.2, 1, 4 from outermost to innermost. Activity updates are reciprocal and
simultaneous. Output is simply the eight outer activities of the output module.
No correct inner code, class label or target output is supplied during inference.

Eight held-out midpoint signals test interpolation in a deliberately constructed
family with one shared underlying scalar. The hierarchy, pairing, polynomial
family and inter-module alignment are supplied. This does not test discovery of
modes, arbitrary independent outer details or general sensory processing. Learning
uses offline least squares; an additional arm allows local parameter changes
during inference, starting from those fitted weights.

## Outer output errors

RMSE is across eight test signals and eight output coordinates; lower is better.

| Configuration | Step 64 | Step 256 | Step 1024 | Step 2048 |
| --- | ---: | ---: | ---: | ---: |
| Graded coupling, learned quadratic | 0.37189 | 0.23283 | 0.21875 | 0.21874 |
| Graded coupling, quadratic adapting | 0.37194 | 0.23355 | 0.22273 | 0.22703 |
| Graded coupling, learned affine | 0.34812 | 0.23897 | 0.23405 | 0.23405 |
| Graded coupling, no preferred detail | 0.39715 | 0.29995 | 0.29747 | 0.29747 |
| Innermost-only coupling, quadratic | 0.48900 | 0.35010 | 0.10037 | 0.02745 |
| No coupling, quadratic | 0.50574 | 0.50574 | 0.50574 | 0.50574 |

The graded quadratic model starts at 0.50574. All its queries are still unsettled
at steps 64 and 256 (free gradient norm >1e-5), despite substantial improvement.
By step 1024 all eight fall below that threshold. At step 2048 their gradient
norms are approximately 1.3e-11–1.3e-10: residual output error is not merely a
failure to wait long enough. The innermost-only model remains above the residual
threshold at step 2048 but has much lower task error. Residual is a diagnostic,
never a gate on readout.

The adapting arm gets worse between steps 1024 and 2048; mean parameter drift is
0.02974. Its energy decreases at every update while task error can increase.
This again shows why lowering local consistency energy is insufficient to
establish useful continuous learning.

## Why graded identity coupling compromises the answer

In this fixture, input and output represent different detailed signals sharing
the same coarse value. At the exact desired pair of signals, local relationship
residuals are zero. But each non-core output unit still receives coupling gradient
`gamma * (output - input)`, which is nonzero whenever the two signals differ.
Thus the correct detailed output is generally not a stationary point of this
energy. The system is being asked to reconstruct its own pattern and also copy a
different pattern. It settles to a compromise.

This analytically explains the conflict in this implementation; it is not evidence
that graded coupling itself is impossible. A compatibility relation or learned
alignment between modules is a candidate next change. Its choice and learning
remain untested. The fixed strength profile is not optimized or ruled out in
general. Weak outer coupling also supplies a direct shortcut: early response alone
does not prove an inward-then-outward causal route. The innermost-only control
isolates a route through the cores.

With no external evidence, independent defaults are stationary. Under graded
coupling they interact and move: after 2048 unclamped steps the cores are about
0.03193 and 0.03206, and the free gradient norm is 0.000575. This joint default is
not yet settled; no uniqueness or universal convergence claim is made. An absent
input is different from an externally clamped zero signal.

## Computation and verification

Gradient checks passed with maximum error 3.45e-10. Quadratic coefficients were
recovered to 3.33e-16 in the realizable training family. Every activity update
preserved input clamps and decreased energy; no backtracking halvings were needed.
The explicit innermost-only control reproduces the v1 output curves.

Direct reconstruction from the input mean and the stored output hierarchy gives
quadratic RMSE 1.33e-16, affine 0.13620 and zero-detail 0.25748. It is a task
comparator, not a solution to the graded energy. It establishes both representational
capacity and the absence of an efficiency advantage for iterative inference here.

| Graded quadratic inference method | Output RMSE | Median measured time/query |
| --- | ---: | ---: |
| Direct task reconstruction | 1.33e-16 | 0.0684 ms |
| 64 activity steps | 0.37189 | 8.241 ms |
| 2048 activity steps | 0.21874 | 264.073 ms |

Timings include allocations and inference; iterative timings include gradient and
line-search energy evaluation, but exclude scoring. One warmup and three measured
batches of eight queries were used. Full spreads are in the runtime artifact.
This is a small CPU mechanism test, not a scaling benchmark.

The v2 run took 22.11 seconds including construction, 28 local fits, checks,
98,304 main updates, 2,048 no-evidence updates, 67,584 benchmark updates, direct
comparisons and scientific serialization. There were 167,936 updates total,
384 main outer readouts, and 252 factor training rows derived from just nine
independent training pairs. No search, tuning, candidate rejection or best-seed
selection occurred. v1's separate 18.04-second control study is additional research
cost, not hidden deployment cost. Combined execution was approximately 40.15 seconds;
manual development and documentation time were not measured.

Each deployed quadratic model has 336 bytes of stored coefficients and 240 bytes
of activity, plus 120 bytes of coupling constants and 168 bytes of fixed index
arrays: 864 retained array bytes, excluding Python object overhead and temporary
workspace. There is no replay, optimizer state or retained training covariance.
Peak process RSS was 35,356 KiB. Fit cost is O(N*(M*p²+p³)), update O(Np), and
T-step inference O(TJNp) for J line-search trials. Direct reconstruction is O(Np).

Artifacts: [contract](standing-pattern-hierarchy-v2-contract.md),
[runner](../experiments/standing_pattern_hierarchy_graded.py),
[scientific results](../reports/standing-pattern-hierarchy-v2-results.json),
[runtime and provenance](../reports/standing-pattern-hierarchy-v2-runtime.json),
[innermost-only study](standing-pattern-hierarchy-v1-results.md).
