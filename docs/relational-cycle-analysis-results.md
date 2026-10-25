# Small cycles: compatibility, defaults, and exact interpretation

Completed 2026-10-25. Supplied binary same/opposite relations on triangles and
squares; no learning or new witness on the existing capacity task.

[Analysis and prospective check](relational-cycle-analysis.md) ·
[checker](../experiments/relational_cycle_analysis.py) ·
[deterministic results](../reports/relational-cycle-analysis.json) ·
[runtime receipt](../reports/relational-cycle-analysis-runtime.json)

## 1. What the cycle changes

A closed loop supports a nonzero perfectly aligned scalar pattern exactly when
it contains an even number of opposite relations. An odd number forces at least
one disagreement. This follows by propagating a value around the loop: an odd
number of sign reversals returns its negative. All 24 triangle/square relation
assignments passed exact characteristic-polynomial and binary compatibility checks.

This obstruction is not instability. A frustrated signed Laplacian has positive
eigenvalues and no zero mode. Its unforced smoothing attenuates every component,
but evidence and stored preferences may maintain a driven pattern. The finite
interpreter can also settle with a residual mismatch.

The two ways of resolving incompatible preferences are distinct:

- **Compatibility stronger than stored-relation preference:** revise the relation
  assignments. Every optimum is balanced and satisfies all selected relations.
- **Stored-relation preference stronger than compatibility:** retain the preferred
  relations. Odd-parity preferences remain frustrated; a mismatch is accepted.
- **Equal strengths:** revising a relation and accepting its mismatch may be
  equally good explanations. The ambiguity must be retained in the report.

These are exact consequences of the declared costs, not discovered optimal
hyperparameters. Local elimination reduces the model to ordinary pairwise binary
cycle inference; no advantage over that simpler description is established.

## 2. Evidence changes the selected completion

For a triangle, observe only the first outer bit as 1. With same/same/same stored
relations and compatibility stronger than preference, the unique completion is
111. With opposite/same/opposite stored relations it is 100. The observation is
preserved in both cases; stored relations determine the missing components.

With no evidence and all three stored relations opposite, compatibility strength
4 and preference strength 2 select one of the three one-hot outer patterns,
changing one stored relation (energy 3). Preference strength 6 selects the same
three possible outer patterns but keeps every stored relation, leaving one edge
mismatch (energy 5). At preference strength 4, the two relation explanations tie,
giving six optima at energy 5.

Thus no-evidence behavior need not realize all individual stored defaults when
those defaults conflict. The neutral-message identity of an isolated unit does
not imply that messages from coupled neighbors are neutral.

## 3. Exact inference and ambiguity

Condition on one outer node's possible values, solve the remaining chain, and
include the closing-edge cost before selection. Traceback returns an optimum.
Every one of the 4,536 evidence/preference/regime cases agreed with exhaustive
global selection at every possible cut: 17,496 checked inference calls.

| Cycle | Relation preference beta | Cases | Tied optima | All optima balanced | Mixed balanced/frustrated optima | All optima frustrated |
| --- | --- | --- | --- | --- | --- | --- |
| Triangle | 2 | 216 | 7 | 216 | 0 | 0 |
| Triangle | 4 | 216 | 156 | 60 | 156 | 0 |
| Triangle | 6 | 216 | 7 | 108 | 0 | 108 |
| Square | 2 | 1,296 | 55 | 1,296 | 0 | 0 |
| Square | 4 | 1,296 | 1,051 | 248 | 1,048 | 0 |
| Square | 6 | 1,296 | 55 | 648 | 0 | 648 |

Compatibility strength is fixed at 4. Counts exhaust the specified menu; they
are not empirical success-rate estimates. Different cuts select different
equally optimal representatives in 7 triangle cases and 55 square cases per
regime. Optimum energy is cut-independent. Tie-breaking is deterministic given
the cut and ordering, but is not a symmetry-preserving learned preference.

Exact cycle inference is O(n*q^3) after O(n*q^2*p) edge-relation elimination,
where q is the outer alphabet and p the relation alphabet. It is linear in
cycle length for fixed alphabets. This avoids a generic exponential search on
one cycle, but relies on independent edge-relation priors. Additional couplings
between relation states can invalidate the elimination. No convergence claim
is made for unconditioned loopy message passing or scalar recurrent dynamics.

## 4. Verification and computation

Executed `python3 -m experiments.relational_cycle_analysis`; all assertions
passed on the first run. Exact integer checking covered 24 signed geometries,
320 binary geometry cases, 2,112 local relation-construction candidates,
1,036,800 exhaustive state visits (208,896 feasible scores), and 474,336
conditioned-chain transition/closing candidates. All decoded outputs satisfied
the evidence and belonged to their exhaustive optimum sets. The hand-worked
examples above were asserted explicitly. No fits, tuning, empirical examples,
optimizer steps, or convergence failures occurred.

Whole-check time including deterministic serialization was 1.3114 seconds wall
and 1.3102 seconds CPU. Peak process RSS was 35,540 KiB, including the exhaustive
reference, interpreter, and scientific output. Supplied configuration storage
was 1,201 recursively counted Python object bytes or 222 JSON bytes. There is
no learned state or replay. Timings exclude startup, provenance collection, and
runtime receipt writing; manual development and separate allocator workspace
are unmeasured. This is a computation receipt, not an inference benchmark or
scaling measurement. The source/contract/results hashes are recorded separately.

## 5. Architectural implication and remaining gap

Circular structure is not itself a barrier to interpretation or finite settling.
It imposes a global consistency condition that must be included in inference.
The relation geometry determines which modes can survive; the preference/
compatibility tradeoff determines whether that geometry is revised.

This model has n outer bits and n relation bits: it has not yet achieved the
desired inward reduction in possible modes. The relations are supplied and the
solver is exact finite optimization, not a learned graded unit dynamics. The
next structural problem is a smaller inner representation that controls several
edge relations while retaining contextual interpretation and tractable inference.
Merely storing cycle parity is insufficient: every balanced relation pattern
would have the same parity despite encoding different completions.

Uniform closure should not be imposed on the existing task without checking
its boundary semantics; the earlier
[composition analysis](relational-capacity-and-composition-analysis.md) already
shows that physical circular layout need not imply periodic task progression.
No Stage B search extension follows from this illustration.
