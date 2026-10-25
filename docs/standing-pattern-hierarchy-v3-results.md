# Intermediate module: graded hierarchy v3 results

Adding an intermediate module reduces final output error in this synthetic test,
but slows the response and does not eliminate distortion from identity coupling.
The final graded quadratic RMSE is 0.17599 versus 0.21874 for two modules, a 19.55%
reduction. This is a result for one explicitly chosen neutral intermediate module,
not a general result for arbitrary intermediate modules or learned mode discovery.

## Architecture and controlled changes

The chain is input ↔ intermediate ↔ output, without a direct input–output edge.
All three modules have 8→4→2→1 layers. Both edges couple corresponding layers
with strengths 0.05,0.2,1,4, weak outside and strong inside. Only the input module's
outer values are clamped; only the output module's outer values are read as output.
Inner activity is never a target supplied during inference.

Endpoint training and evaluation signals reproduce v2 exactly. The middle has the
same local quadratic operator but starts with all 21 coefficients zero, preferring
equal children. No middle labels or desired intermediate signals were fitted.
It has evolving activities even in the frozen-weight arm. In the plastic arm its
coefficients adapt along with the endpoint coefficients; this starts from pretrained
endpoints for each query and is not an uninterrupted learning stream.

## Results on the same eight held-out signals

Errors are continuous outer-output RMSE. All rows use learned quadratic endpoint
preferences unless stated otherwise.

| Configuration | Step 64 | Step 256 | Step 1024 | Step 2048 |
| --- | ---: | ---: | ---: | ---: |
| Two modules, graded, frozen (v2) | 0.37189 | 0.23283 | 0.21875 | 0.21874 |
| Three modules, graded, frozen | 0.45792 | 0.26446 | 0.17683 | 0.17599 |
| Three modules, graded, adapting | 0.45793 | 0.26500 | 0.18089 | 0.18434 |
| Three modules, graded, affine endpoints | 0.44118 | 0.25463 | 0.18010 | 0.17941 |
| Three modules, graded, zero detail everywhere | 0.48258 | 0.32096 | 0.27111 | 0.27100 |
| Three modules, innermost-only chain | 0.49675 | 0.40778 | 0.17809 | 0.06544 |
| Three modules, uncoupled | 0.50574 | 0.50574 | 0.50574 | 0.50574 |

The frozen graded chain starts at RMSE 0.50574. All eight queries still have free
activity gradient norm >1e-5 at step 1024; none do at 2048. Thus the improved
output is available before settling. Residuals never gate readout. The innermost-
only chain remains unsettled at 2048 but has lower task error, again separating
stability from correctness. The no-evidence graded run remains unsettled after
2048 steps (gradient norm 0.000545); convergence and uniqueness are not proved.

The adapting chain gets worse between steps 1024 and 2048 despite monotone energy
descent. Mean coefficient drift is 0.03171. Useful continuous learning remains
unestablished. Also, the affine endpoint control nearly matches the quadratic
model in this coupled test, so the improvement does not establish that nonlinear
local preferences are the main cause.

An intermediate module removes the direct pressure for output to equal input.
It instead introduces pressure for each endpoint to agree with the middle, whose
zero-detail preference itself suppresses differences. The result is consistent
with reduced transfer of incompatible input detail, but this experiment does not
isolate that mechanism from changed effective coupling strength, the extra prior
and the added dynamical distance. A weaker two-module coupling control would be
needed to distinguish those explanations. No such strengths were tuned here.

Early output movement alone is not proof of transmitted information: independently
initialized module defaults also interact. The main task errors measure actual
input-conditioned output quality, but do not isolate every route of transmission.

## Computation, checks and reproducibility

| Method | RMSE | Median measured time/query |
| --- | ---: | ---: |
| Direct reconstruction, quadratic | 1.33e-16 | 0.0695 ms |
| Three-module graded, 64 steps | 0.45792 | 9.311 ms |
| Three-module graded, 2048 steps | 0.17599 | 303.499 ms |
| Two-module graded, 64 steps (v2) | 0.37189 | 8.241 ms |
| Two-module graded, 2048 steps (v2) | 0.21874 | 264.073 ms |

These are measured quality–cost points, not equal-compute or randomized hardware
comparisons. The three-module 2048-step run costs about 15% more time/query in
these measurements. At equal update counts it evaluates 21 local factors rather
than 14 and two coupling edges rather than one. Direct reconstruction bypasses
the coupled energy and remains both more accurate and cheaper for this task.
Benchmark allocation and gradient/line-search work are included, scoring excluded;
one warmup and three measured batches of eight queries were used. Full spread
and CPU measurements are retained in runtime JSON.

Gradient checks passed at 4.76e-10 maximum error. Endpoint coefficient recovery
error is 3.33e-16. Checks verify exact v2 endpoint fixtures, no direct endpoint
coupling, the expected adjacent-module derivative, clamping and descent at every
step. No backtracking halvings were needed.

Whole execution took 25.56 seconds. It includes 28 endpoint fits, 252 factor rows
from nine independent training pairs, 98,304 main updates, 2,048 no-evidence
updates, 67,584 benchmark updates, 384 main readouts, direct comparisons and
serialization. There was no selection, tuning or rejected candidate. Earlier v1
and v2 research execution adds approximately 40.15 seconds; it is not charged as
deployment cost. Manual development time is unavailable.

The model retains 504 coefficient bytes, 360 activity bytes and 288 fixed coupling/
index bytes: 1,152 array bytes excluding Python overhead and temporary workspace.
No replay or optimizer state is stored. Peak process RSS was 35,804 KiB. For K
modules, update time is O(KNp), activity O(KN), parameters O(KNp), and T updates
with J line-search trials O(TJKNp). Endpoint fit cost remains unchanged. This
runner's direct comparator expands all modules and costs O(KNp); expanding only
the output would cost O(Np). Historical reports have not been rewritten.

Artifacts: [contract](standing-pattern-hierarchy-v3-contract.md),
[runner](../experiments/standing_pattern_hierarchy_chain.py),
[scientific results](../reports/standing-pattern-hierarchy-v3-results.json),
[runtime and provenance](../reports/standing-pattern-hierarchy-v3-runtime.json),
[previous two-module results](standing-pattern-hierarchy-v2-results.md).
