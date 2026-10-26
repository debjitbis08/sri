# Distributed evidence: feedback helps these dynamics, direct inference wins

2026-10-26. Reciprocal inference reconstructs missing outputs in both modules
from complementary cues. It beats both directed edge-deletion controls, but
the direct joint-input baseline solves the task exactly and much faster.
This establishes a scoped feedback contribution, not recurrence necessity or
an advantage over a suitable feedforward calculation.

## Task and architectural scope

Two modules each have four outer activities and two inner activities. Module A
observes a; module B observes b. Their supplied, distinct local maps expand the
shared inner coordinates (a,b) into outer patterns. Each isolated cue leaves
one degree of freedom undetermined. The full 16-case Cartesian fixture includes
cue changes that alter missing outputs in both modules. No learning, STM,
hidden-code clamps, test-driven tuning, or random sampling is used.

This is a new linear 4→2 mechanism test, not an implementation of the previous
8→4→2→1 hierarchy with the same parameters. Increasing the inner dimension and
supplying a correct alignment are explicit architectural assumptions. Only inner
coordinates with shared meanings are coupled. No pressure equates the different
outer patterns. Thus this experiment does not isolate the effect of fixing
alignment in the previous studies, and it does not test learned alignment.

## Results

Missing-coordinate RMSE averages six free outer coordinates per case:

| Method | Step 64 | Step 256 | Step 1024 | Median time/query at 1024 |
| --- | ---: | ---: | ---: | ---: |
| Isolated | 0.64701 | 0.64550 | 0.64550 | 10.203 ms |
| A→B | 0.51363 | 0.48412 | 0.48412 | 11.109 ms |
| B→A | 0.51363 | 0.48412 | 0.48412 | 11.179 ms |
| Reciprocal | 0.36157 | 0.03065 | 0.00000158 | 11.959 ms |
| Direct joint-input | 0 | 0 | 0 | 0.00286 ms (one calculation) |

All 16 cases in every iterative arm have update residual <=1e-5 at 1024.
Settling alone is therefore insufficient for correctness here too. The one-way
source has RMSE 0.64550; its receiving module has RMSE 0.22822. One-way coupling
also pulls the receiving code toward the source's unsupported default. Its
limitation is not only the missing reverse information path. A different
feedforward fusion rule need not suffer that bias; the direct comparator does
not. These controls retain the declared local rules, not all possible acyclic
implementations.

The reciprocal system has a zero-energy solution: each module reconstructs its
outer pattern and both inner codes equal the observed pair (a,b). It is unique
under these clamps: zero local residual requires x_m=D_m c_m, zero coupling
requires the codes to agree, and the two clamped coordinates fix both entries.
Since the objective is a sum of squares, its constrained Hessian is positive
definite (the same argument eliminates every homogeneous null direction).
This explains why feedback can propagate both observations without a competing
prior. The direct baseline simply computes that solution from the same cues and
maps. No learned generalization is demonstrated by this exhaustive fixture.

## Verification and cost

Executed checks cover every activity derivative (maximum finite-difference
error 1.15e-10), cue preservation, reciprocal descent, byte-identical fixed maps,
exact direct reconstruction, isolated ambiguity and sensitivity to either cue.
The runner fails immediately if a check fails and refuses to overwrite artifacts.
Directed dynamics are not assumed to descend the reciprocal energy.

Total execution took 7.0435 seconds before runtime serialization, including
verification, all scientific trajectories and benchmarks. There were 589,760
updates and 2,500 inference calls including checks and warmups. Before benchmarks
there were 65,600 updates and 100 inference calls. There are 16 independent
fixture pairs; repeated exposures are not new data. Fits, candidates, optimizer
steps and tuning were zero. Manual design time is unmeasured.

Benchmarks use one warmup and five measured batches of 16 cases, with allocation
and updating included, scoring excluded. Full wall and CPU ranges and all horizon
costs are in the runtime artifact. The reciprocal 1024-step wall range is
11.928–11.993 ms/query; direct is 0.00281–0.00318 ms/query. These tiny dimensions
are not a hardware scaling study. In this fixture the direct method dominates
every measured recurrent quality–latency point. Construction plus Q times
inference determines deployment cost; there is no training/selection cost here.

Stored maps occupy 128 bytes; per-query activities occupy 96 bytes, for 224
array bytes excluding interpreter overhead and temporary workspace. The fixture
occupies another 256 bytes. No replay, optimizer or learned statistics are kept.
Single-process lifetime peak RSS was 35,880 KiB, including reporting arrays.
For K modules of outer width N and inner width d with E edges, each update costs
O(KNd+Ed), T updates O(T(KNd+Ed)), maps O(KNd), activity/workspace O(K(N+d)).
Direct reconstruction costs O(KNd). Actual backend thread counts were not
measured; environment settings and BLAS configuration are recorded. Historical
scalar tasks are not equivalent-task comparators and their timings are not used
for a relative efficiency claim.

## Next learning question

Inference activity may use compatibility energy without making energy reduction
the criterion for retaining relationships. The separate
[cross-experience consolidation proposal](cross-experience-consolidation-proposal.md)
distinguishes a temporal stability gate from independent predictive support,
and includes matched inertia, gate-removal, repeated-trace, forgetting and
collapse controls. It remains a proposal: no sleep/dream learning was implemented
or validated in this run.

Artifacts: [prospective contract](ambiguous-evidence-v1-contract.md),
[runner](../experiments/ambiguous_evidence.py),
[scientific results](../reports/ambiguous-evidence-v1-results.json),
[runtime/provenance](../reports/ambiguous-evidence-v1-runtime.json).
