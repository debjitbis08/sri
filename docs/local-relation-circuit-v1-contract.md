# Local relation circuit v1: prospective contract

Status: frozen before execution, 2026-10-25. Tests a compact circuit, exact
contextual return, and finite supervised identification of local detectors.
This is not an unsupervised grouping-discovery claim, a graded recurrent neural
implementation, or a witness on the frozen 72-scene task. Preserve earlier
sources/results. Follow [computation reporting](computation-reporting.md).

## 1. Questions and supplied structure

Continue the [four-class construction](inner-relation-quotient-analysis.md).
There are four outer binary units on a cycle and an inner state in {uniform,
one different, adjacent pairs, alternating}, numbered 0,1,2,3. The exact reference
assigns these classes by rotation/complement orbits. Reference labels are used
for declared training labels and evaluation, not supplied as a full deployment
pattern table.

Each edge applies two shared binary detectors g1(a,b),g2(a,b). Aggregate

```
t1 = sum_edges g1(b_i,b_(i+1))
t2 = sum_edges g2(b_i,b_(i+1))
c  = decoder(t1,t2).
```

Sharing and summing detectors impose cyclic rotation invariance. Complement
invariance is NOT imposed on their tables or decoder. A positive/negative
orientation convention for ordered edges is supplied. Counts are integers,
not oscillating waves. This is a fixed two-channel scaffold with a learned local
truth table and a learned count decoder, not discovery of arbitrary architecture.

The supplied control uses XOR and AND. Transition count distinguishes uniform
(0) and alternating (4). At transition count 2, AND count 1 identifies adjacent
pairs; AND counts 0 or 2 identify one different. The control decoder is defined
only on these reachable pairs. Check all 16 patterns before learning.

## 2. Finite learning protocol

Training patterns, labels, and split are declared now:

```
0000 -> 0, 1111 -> 0
0001 -> 1, 1110 -> 1
0011 -> 2, 1100 -> 2
0101 -> 3, 1010 -> 3.
```

The remaining 8 patterns are held out until after candidate selection. The
training set contains one representative and its complement for each rotation/
complement class. Rotation invariance of the architecture supplies generalization
to the remaining members: this split cannot establish discovery of rotation
equivalence. Labels provide the desired grouping explicitly. Complement examples
provide its complementary-polarity behavior. There are no validation examples,
random draws, optimizer, epochs, or hyperparameter tuning.

A binary local detector has 16 possible truth tables. Its code encodes entries
at indices 2*a+b as bits (least significant entry is input 00). Compare:

- One-detector control: all 16 tables, one count and a learned count-to-class map.
- Two-detector candidate: all 16^2 ordered table pairs, two counts and a learned
  count-pair-to-class map.

For each candidate, process every training pattern without early rejection.
Assign the observed feature tuple its label; reject a candidate if any tuple
receives conflicting labels. The retained decoder contains only training-derived
entries. No guessed entries are added. Select the lexicographically smallest
consistent detector-code tuple using training consistency alone. If a family has
no candidate, report no fit. The one-detector case is a comparison, not a fallback
that alters the two-detector experiment. Do not expand the family after failure.

After selection, report held-out and all-pattern accuracy, unknown feature tuples,
and agreement across every consistent candidate. Scoring the full version space
is a diagnostic, not test-based selection. Assert only mechanical correctness;
unknowns or misclassifications are scientific negatives and must be reported.
Do not assert desired learning success in a way that suppresses its receipt.

Training cost: 272 candidates, 8 labeled examples each, 2,176 training exposures;
one-detector gate evaluations 16*8*4=512, two-detector evaluations
256*8*4*2=16,384. Rejected candidates count in full. Exact finite identification
is not a biologically local plasticity rule even though the fitted gates act
locally during deployment.

## 3. Contextual return without full-pattern enumeration

Use the earlier energy: outer Hamming preference, inner preference penalty 5,
and class inconsistency penalty 6, with hard observed outer bits. For any complete
decoder, every optimum has c=decoder(t1,t2), because changing c to that value
costs at most 5 in prior and saves 6 in compatibility. Eliminate c accordingly.

For each possible initial outer bit, propagate a conditional cost profile along
the chain. A profile key contains (current bit, accumulated t1, accumulated t2).
Appending one bit incurs its local Hamming cost (or is disallowed by evidence)
and adds the two local detector outputs to the counters. Keep the lowest cost
per key, with predecessor pointers. Add the closing edge's detector outputs
before decoding the class and charging its preference. Select the lowest total
and traceback to outer bits. Ties use declared sorted traversal and keep the
first equal-cost predecessor; do not claim global lexicographic equivalence.

This circuit does NOT enumerate all complete outer patterns as inference states.
It merges partial paths with the same sufficient counters and boundary bit.
Its profile entries do enumerate those counter possibilities. A count register
is not a single fixed-precision binary neuron: it needs O(log n) bits. The return
uses vector cost messages and a scheduled dynamic program, not scalar autonomous
relaxation. Branch conditioning and closing-edge handling are explicit.

Unknown terminal feature tuples cause an explicit incomplete-decoder result;
they are not filled from test labels or silently excluded. Deployment completeness
is checked after selection over the finite 16-pattern domain. If incomplete,
report it and do not score that model as an exact return solver.

Induction proves each profile entry gives the exact minimum prefix cost at its
key. Conditioning the root and restoring the closing edge give the exact cycle
minimum. Traceback recovers an optimal completion. All preferences/evidence are
counted once. No claim extends to unscheduled recurrent or loopy message updates.

## 4. Exhaustive validation and fixed timing menu

The direct comparator enumerates all 16 outer patterns, obtains each reference
class, and minimizes Hamming+5*class-change under evidence. Compatibility was
eliminated analytically; verify this comparator agrees with the existing model
semantics by the proof above. Tests use all 16 stored outer defaults, all 4 inner
defaults, and all 81 evidence patterns: 5,184 cases.

First validate the supplied circuit on all cases. After learning and held-out
evaluation, if the selected learned model is total, validate it on the same
cases. Separate (a) agreement with the exhaustive solver for its own learned
class function and (b) agreement with the target/reference grouping. The first
must hold even if the learned grouping is scientifically wrong. Check evidence,
energy, and membership in the optimum set rather than identical tie choices.
Run the four hand-worked queries from the quotient analysis for both models.

Record actual gate calls, prefix candidates, merged profile entries, terminal
candidates, traceback nodes and peak live/profile-storage entries, including
failed or discarded alternatives. Reference visits are counted separately.
Learning and exhaustive checking are not deployment cost.

Timing comparisons: use exactly the 81 evidence patterns with stored outer
default 0011 and inner default adjacent pairs. Benchmark the supplied circuit,
the total selected learned circuit if available, and direct 16-pattern enumeration.
Use one warmup batch and 7 measured batches for each. Report batch wall/CPU
median and min/max, with batch size 81, and derived amortized per-query time
explicitly labeled. Include allocation, counters, gate lookup, profile building,
traceback and Python call overhead. No compilation, accelerator transfer, or
synchronization. The exact reference computes features/classes on demand; no
hidden cache of all output patterns is used by the local solvers. Benchmark
the same implemented call paths used in validation; do not extrapolate speed.

## 5. Complexity and storage before execution

For k shared binary detectors, feature construction costs O(kn) per complete
pattern. Selecting tables from all 16^k possibilities with T examples costs
O(16^k*T*k*n), with O(Tk) decoder entries per candidate as a safe upper bound
on coordinate storage. For fixed k the candidate menu is finite; growing k makes
this learning method exponential. Count selection separately from inference.

The chain has at most 2*(n+1)^k profile keys per position/root. Transition work
is O(k*n^(k+1)) with fixed binary branching and two root conditions. Current
profiles need O(n^k) entries; stored traceback layers need O(n^(k+1)) entries.
For k=2 these are O(n^3) work and O(n^3) traceback storage, versus O(n*2^n)
direct enumeration. Constants and the tiny n=4 crossover are measured, not
assumed favorable. Arbitrary additional features can destroy this bound.

Deployment stores 4k detector bits and a sparse count decoder (up to T entries,
each k counter coordinates plus a class label), shared topology and declared
preference coefficients. Learned detector/decoder Python object and JSON bytes
are recorded, as are transient profile-entry counts and process RSS. Outer and
inner preferences are query inputs; raw training patterns are not needed by
the deployed circuit. The research artifact retains the training split and
candidate audit, which are not a runtime replay mechanism. No optimizer state,
covariance, or replay is used for inference. Precise allocator workspace is
unavailable and will be marked as such, not zero.

The decoder trained on n=4 is NOT valid automatically at larger n: new count
values and relation shapes can appear. The polynomial inference bound concerns
the specified feature/decoder family if a total decoder is supplied, not proven
learned scaling. Correct grouping beyond n=4 is not tested in this contract.

## 6. Execution and provenance

Implement `experiments/local_relation_circuit.py` and execute once after source
and this contract are fixed. Exact integer costs/boolean tables; float timing
only in a separate runtime JSON. No seeds, numerical solver tolerances, BLAS,
accelerator, workers, or external dataset. Record source/contract hashes,
revision/dirty status, command, Python/platform/CPU/thread information, named
phase wall/CPU, aggregate counts, peak RSS with units, and all selection costs.
Scientific results are deterministic; repeated timing summaries are separate.
Whole-run timing excludes provenance/imports and runtime receipt writing; phase
timing scope is explicit. Manual development is unmeasured. All earlier frozen
reports remain untouched; no historical costs are invented in comparisons.

Success would demonstrate locally factored representation and exact scheduled
completion, plus supervised identification within a supplied scaffold. It would
not establish unsupervised grouping discovery, a local learning law, efficient
general constraint solving, or a graded reciprocal unit architecture.
