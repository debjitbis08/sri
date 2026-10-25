# Relational modes and tree interpretation: checked findings

Completed 2026-10-25. Analytical illustration with supplied geometry/preferences;
no learning, tuning, or new witness on the frozen relational task.

[Derivations and check specification](relational-mode-tree-analysis.md) ·
[checker](../experiments/relational_mode_tree_analysis.py) ·
[scientific results](../reports/relational-mode-tree-analysis.json) ·
[computation receipt](../reports/relational-mode-tree-analysis-runtime.json)

## What the continuation establishes

**Relational agreement can define frequency.** For a same relation the smooth
mode is (1,1); for an opposite relation it is (1,-1). Their respective Laplacians
preserve these modes exactly while attenuating the orthogonal mode. The previous
binary interpretation penalty is exactly the squared relational mismatch in
this geometry. The inner relation label specifies which geometry applies; it
does not encode the amplitude along the preserved mode.

**There is a concrete candidate combination algebra.** Represent a unit's
preference as a cost for each possible state. Combining an all-zero incoming
profile with its stored profile leaves the stored profile unchanged. Inward
aggregation minimizes over compatible child interpretations; outward traversal
selects the child states attaining those conditional minima. Nonlinearity enters
the interpretation, even though profile combination uses addition. This is a
change in representation, not a scalar arithmetic trick.

**On a tree, this interpretation has an exact finite realization.** Conditional
cost profiles preserve alternatives until enough context is available. One
inward traversal and one traceback produce a global energy minimum, avoiding
the first example's scalar coordinate-descent trap. All 90 declared evidence
patterns matched exhaustive enumeration in optimum energy and decoded-state
membership. Actual observed values, including zero, were preserved. No-evidence
queries recovered the supplied whole-state defaults.

**Defaults do not necessarily imply uniqueness.** The two-leaf tree has unique
optima for all 9 patterns. The four-leaf tree has ties in 20 of its 81 patterns.
The declared smallest-state/child-pair tie rule selects deterministically; it
does not establish a unique energy minimum, symmetry preservation, or learned
interpretive preference. Reporting the ties is essential.

## What remains unresolved

- This is exact tree dynamic programming, not a convergence result for arbitrary
  recurrent unit updates. Profile messages cost more than one scalar per unit.
- Fixed-relation smoothing and discrete relation selection are individually
  specified; an alternating continuous hybrid has not been defined or tested.
- On a signed cycle, incompatible relations can eliminate all nonzero perfectly
  aligned modes. The positive triangle has eigenvalues 0,3,3; changing one edge
  sign gives 1,1,4. Circular topology therefore needs its own analysis.
- The relation family and priors are supplied. Learned geometry, graded states,
  cross-coordinate interactions, and a benefit over ordinary factor-graph
  inference remain open. The tree construction is not a new neural architecture
  merely because its messages are described as inward/outward activity.

## Verification and computation

Executed `python3 -m experiments.relational_mode_tree_analysis`; all assertions
passed on the first run. Checks use exact integers and rational fractions:
6 filter cases, 8 compatibility identities, and 2 triangle spectra/determinants.
The interpretation checks visited 10,440 exhaustive states (2,080 feasible),
evaluated 2,016 dynamic-programming factor candidates, constructed 1,188 profile
entries, and decoded 594 node states over 90 queries. Rejected alternatives are
included; there was no parameter search.

The whole successful verification and scientific serialization took 0.01278 s
wall and 0.01277 s CPU. Peak process RSS was 17,276 KiB, including interpreter
and exhaustive reference. The two supplied tree configurations occupy 1,605
recursively counted Python object bytes or 168 JSON bytes; these figures exclude
the shared cost coefficients held in the implementation. There is no learned
state, optimizer, or replay. Detailed scope, hashes, environment, and unavailable
workspace measurements are in the separate receipt. This is not an inference
latency or scaling benchmark; manual development cost is unmeasured.

General q-state tree inference costs O(Nq^3) time and O(Nq) transient profile/
argmin entries, excluding generic compatibility-table storage. Independent
feature coordinates factorize; arbitrary coupling across d binary coordinates
instead gives q=2^d. No cheap multidimensional scaling is implied without that
independence or additional structure.

## Next target

The next analysis should address a small cycle with multiple admissible relation
assignments: characterize consistent low modes, frustration, contextual selection,
and exact inference cost. That will test whether this candidate can extend toward
the module's circular geometry before proposing learning or a graded relaxation.
