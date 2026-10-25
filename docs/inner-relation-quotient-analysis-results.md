# Fewer inner distinctions with contextual outer reconstruction

Completed 2026-10-25. Supplied four-unit binary construction; no learning or
new witness on the existing capacity task.

[Derivation and declared check](inner-relation-quotient-analysis.md) ·
[checker](../experiments/inner_relation_quotient_analysis.py) ·
[scientific results](../reports/inner-relation-quotient-analysis.json) ·
[runtime receipt](../reports/inner-relation-quotient-analysis-runtime.json)

## 1. The inner reduction works in this example

The 16 outer patterns reduce to four inner states:

| Inner state | What remains shared | Outer possibilities |
| --- | --- | --- |
| Uniform | All values agree | 2 |
| One different | One value differs from the other three | 8 |
| Adjacent pairs | Two matching adjacent pairs, opposite to each other | 4 |
| Alternating | Every neighbor is opposite | 2 |

This is a quotient by cyclic rotation and global complement. It keeps relation
shape while dropping its absolute position and polarity. Every outer pattern
remains available. The inner state has four possible values, or two logical
bits; calling it one variable does not make it one binary unit. Its transient
cost profile has four entries.

One inner state organizes several edges together: adjacent pairs permits either
edge-relation pattern 0101 or 1010. It does not prescribe the position of those
relations without context. The outer configuration still carries the full detail.

## 2. Return interpretation uses evidence and stored preference

With stored outer preference 0011 and stored inner preference adjacent pairs:

| Observations | Selected outer pattern | Selected inner state |
| --- | --- | --- |
| None | 0011 | Adjacent pairs |
| First two bits 00 | 0011 | Adjacent pairs |
| First two bits 11 | 1100 | Adjacent pairs |
| All four bits 0101 | 0101 | Alternating |

All four selections are unique. Thus the same inner state supports different
contextual outer interpretations, and complete evidence can change that inner
state. No stored residual is needed to recover the old pattern when context
instead supports a different member of the same class.

The exhaustive menu varied all outer defaults, all inner defaults, and all
observation masks/values: 5,184 cases. Conditional fiber selection matched direct
joint enumeration in every case. The preferred class was evidence-feasible in
2,944 cases and was selected in all of them; it changed in the remaining 2,240.
All complete observations were preserved. All 16 coherent no-evidence defaults
were recovered uniquely.

There were 384 cases with tied outer completions. No case in this finite menu
had multiple optimal inner classes. A deterministic lexicographic convention
selects among tied outer completions; it is not a learned preference or a proof
of symmetry-equivariant selection.

## 3. A precise mode connection, with an important limitation

On functions over the 16 possible outer configurations, rotation/complement
define a Laplacian with a four-dimensional zero-mode space. Its zero modes are
exactly the functions constant within each of the four inner classes. Exact
projection and rank checks passed. Fixed smoothing preserves that space and
contracts its orthogonal complement by at least a factor of three per step for
the declared unit smoothing strength.

These are modes of a configuration-function space, not ordinary spatial Fourier
modes across four physical unit values. The two positions of adjacent-pair
structure have different signed-cycle low-mode lines whose intersection is only
zero. A single fixed physical mode cannot represent both positions without an
additional selection mechanism.

Actual interpretation minimizes costs within each class, rather than averaging
them. Both operations remove within-class distinctions, but they are different
algebras. In particular, separately minimizing two evidence/preference profiles
and adding their minima can combine incompatible explanations. The checker
confirms a strict counterexample: separate minima sum to 0 while the best shared
interpretation costs 2. Contributions must agree on an outer interpretation
before inward reduction, or preserve sufficient conditional information.

## 4. Cost and scope

Executed `python3 -m experiments.inner_relation_quotient_analysis`; all assertions
passed on the first run. Exact integer/rational verification included 128 group
actions, the 16x16 projection/Laplacian identities, and all inference cases.
The direct reference visited 331,776 joint assignments, scoring 65,536 feasible
ones. Fiber inference visited 82,944 outer candidates (16,384 feasible) and
20,736 inner profile entries. Rejected candidates are included. No tuning,
learning, empirical examples, optimizer steps, or solver failures occurred.

Whole-check and scientific serialization time was 0.48364 s wall and 0.48348 s
CPU. Peak process RSS was 37,164 KiB, including interpreter, exact matrix checks,
reference enumeration and output. Supplied configuration occupies 1,069 Python
object bytes or 241 JSON bytes; these exclude generated membership/matrix tables,
whose separate allocation was not measured. There is no learned state or replay.
The receipt records scope, source/contract hashes, environment, and unavailable
workspace data. These are one-off computation measurements, not deployment latency
or a scaling benchmark; manual development time is unmeasured.

For general n, a rotation/complement class contains at most 2n patterns. There
are therefore at least 2^n/(2n) classes. The reference canonicalization and query
algorithms remain exponential in n; the exact construction is not a scalable
compression algorithm for arbitrary patterns. The global class prior also breaks
the previous independent-edge elimination. Complexity/storage derivations and
the direct equivalent appear in the analytical specification.

## 5. Conclusion and next obligation

This is a constructive example of the requested inward reduction: fewer inner
distinctions, all outer patterns still represented, and contextual outward
interpretation. The relation classes are deliberately supplied. The remaining
implementation problem is to realize such classes and conditional information
compactly through local operations, rather than enumerating all configurations.
Learning the relevant equivalences and maintaining a graded recurrent equilibrium
remain separate, unproved obligations.
