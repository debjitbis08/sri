# Relation-dependent modes and exact interpretation on trees

Status: analytical continuation, 2026-10-25. A candidate algebra with supplied
relations, not a selected SRI architecture or a learning/capacity study.
This develops the [first analysis](combination-interpretation-analysis.md).
Historical artifacts and the previous checker are unchanged.

## 1. A geometry where opposite activations can be smooth

Ordinary smoothing assumes neighboring units should agree. Instead, let a
relation r in {+1,-1} specify whether two scalar activities should agree or be
opposites. Define

```
L_r = [[1,-r],[-r,1]]
x^T L_r x = (x1-r*x2)^2.
```

The normalized low and high modes are respectively

```
v_low  = (1,r)/sqrt(2),     eigenvalue 0
v_high = (1,-r)/sqrt(2),    eigenvalue 2.
```

For fixed r, S_r=(I+tau*L_r)^-1 preserves the low mode and multiplies the
high mode by 1/(1+2*tau), for tau>=0. In explicit form,

```
S_r = [[1+tau, tau*r],[tau*r,1+tau]]/(1+2*tau).
```

Thus an alternating pair is high-frequency under the same relation but
low-frequency under the opposite relation. "Smooth" can mean agreement after
relational alignment, not visual similarity in raw coordinates. This is an
exact supplied geometry; choosing or learning r is a separate operation.

For the binary example, encode outer bits as xj=(-1)^bj and set r=(-1)^h.
Then

```
[h != (b1 XOR b2)] = (x1-r*x2)^2/4.
```

The previous compatibility penalty is exactly the high-mode energy of this
relation-dependent Laplacian. When h agrees with the pair's relation, the pair
lies in its low mode. This connects relational compatibility to frequency
without asserting that ordinary smoothing preserves every relation.

**Representation caveat.** h labels the geometry; it is not the low-mode
amplitude. Pairs 01 and 10 have the same h but opposite low-mode amplitudes.
The amplitude/pose can remain outside and be inferred from outer evidence or
defaults. A hierarchy transmitting only h is not the same as repeatedly
projecting onto a fixed low-frequency basis. A richer layer could transmit both
relation and amplitude, but must count both states and justify what still shrinks.

## 2. Generalization and the obstruction on cycles

For vector activities xi in R^d and orthogonal transports T_ij, define

```
E_rel(x) = 1/2 sum_edges w_ij ||xi - T_ij xj||^2.
```

This is a positive-semidefinite block-Laplacian energy. On a tree, choose a root
frame and orthogonal matrices Gi recursively so that T_ij=Gi Gj^T. Writing
xi=Gi yi turns every edge residual into Gi(yi-yj). Consequently the spectrum
equals that of the ordinary weighted tree Laplacian repeated d times. The
connected tree has d exact zero modes: relationally aligned constant fields.

The fixed-transport filter preserves these zero modes and attenuates each
positive eigenmode by 1/(1+tau*lambda). Applying an arbitrary stored default
as a force or prior changes the equilibrium; this spectral statement concerns
the filter alone. Observation masks can also change the effective operator.

On a cycle, the product of transports around the cycle must fix the propagated
root vector. A nonidentity product can reduce the zero-mode dimension; it
eliminates it when the product has no fixed nonzero vector. For signed scalar
edges, a cycle with product -1 has no nonzero perfectly aligned field. This is
frustration, not an inference implementation bug. On a three-node unit-weight
cycle with one negative edge the eigenvalues are 1,1,4, instead of 0,3,3 for
all positive edges. Thus the exact tree statements do not transfer to the
repository's circular topology without additional work.

No eigendecomposition is required by the finite verifier below. The two-site
identities and signed-cycle determinants are checked with exact arithmetic.
For a dense Nd-by-Nd block matrix, full eigendecomposition or dense direct
factorization costs O((Nd)^3) time and O((Nd)^2) storage. Building dense edge
blocks costs O(E*d^2), and a sparse matrix-vector multiply costs the same.
An iterative filter costs O(K*E*d^2) for K iterations plus vector storage, but
K and a residual tolerance must be specified before any benchmark; none is run
here. The tree frame transform costs O(N*d^3) with general dense matrix products
to construct all frames, then O(N*d^2) per application. More structured transports
may be cheaper. No high-dimensional speed claim follows from two-site algebra.

## 3. A candidate combination operator on preference profiles

Rather than immediately committing each unit to one value, let its transient
message be a cost profile over its possible values. A stored preference is a
fixed profile p(z), not a scalar activation. A neutral message has every entry
zero. Define

```
C(m,p)(z) = m(z) + p(z).
```

Now C(0,p)=p exactly. This is a concrete candidate for the previously unspecified
combination symbol: arithmetic on *profiles*, not assumed arithmetic on unit
activations. Finite stored preference yields a selected default if it has a
unique minimizing state. If the default state is s, a final argmin maps the
zero-message profile back to s. Thus the default law holds at the profile level
and, given uniqueness, at the realized-state level.

Evidence has its own profile: 0 for allowed states, +infinity for excluded ones.
An observed numerical zero excludes other states; no evidence is the all-zero
profile. Conflicting hard evidence can yield an all-infinite profile and must
be reported as infeasible. Do not subtract infinity when normalizing messages.

Combining profiles is additive, but interpreting them by minimization over
relational configurations can produce nonlinear responses in the declared
unit states. This is not proof that any additive scalar architecture is enough.
It changes the transient representation and charges for all profile entries.

## 4. The inward and outward operators on a binary tree

Each internal node v has children l,r and a q-valued state h. Use a stored unary
preference p_v and a supplied compatibility cost k_v(a,b,h). Let e_v be its
evidence profile (only leaves are observed in the check). The inward profile is

```
F_v(h) = p_v(h) + e_v(h)
         + min_{a,b} [k_v(a,b,h) + F_l(a) + F_r(b)].
```

At a leaf, F_v=p_v+e_v. The min over a,b is the candidate aggregation operator.
Each F_v describes the best achievable cost in that subtree conditional on h;
it does not discard competing h values prematurely.

Choose a minimizing root state. On the return pass, choose the child pair
that attained the corresponding conditional minimum and recurse. Ties use the
smallest state, or child pair, in declared enumeration order. This is a
deterministic convention, not a learned preference or a symmetry guarantee.
Record ties rather than claiming the optimum is always unique.

**Exactness proof.** At a leaf the profile is its exact constrained cost. Given
h, two child subtrees share no variable and no factor. By induction their
profiles give their exact conditional minima, so independently minimizing their
sum with the node compatibility enumerates all compatible subtree assignments.
Root minimization and traceback therefore recover a global optimum. Every
unary preference and compatibility factor is charged exactly once.

This is finite dynamic programming, not oscillatory dynamics or repeated scalar
coordinate descent. It terminates after one inward and one return traversal,
regardless of a scalar initialization. It avoids the previous local trap by
retaining the conditional cost of a joint reinterpretation until it is selected.
It establishes termination and global optimality on trees, not convergence of
an arbitrary asynchronous recurrent implementation.

In a more general tree factor graph, directed messages exclude the receiver's
contribution; repeatedly circulating full profiles would double-count evidence
and priors. On graphs with loops, these recurrences have no general exactness
or convergence guarantee. Exact elimination costs grow with factor width /
treewidth, even if each original factor is local.

## 5. Declared finite check, before execution

Use exact Python integers, infinity for infeasibility, and rational fractions
for spectral identities. No random draws, fitting, solver tolerances, parameter
selection, or numerical eigensolver. No empirical examples or training updates.

Check the two relation signs at tau in {0,1,2}: low-mode preservation,
high-mode gain, and the inverse identity (I+tau*L)S=I. Check all eight binary
triples against the squared-residual compatibility identity. Check determinants
of positive and one-negative-edge triangles, respectively 0 and 4.

The tree check contains exactly two supplied structures:

- Two leaves (nodes 0,1), root 2. Defaults (0,1,1).
- Four leaves (nodes 0..3), internal nodes 4=(0,1), 5=(2,3), root 6=(4,5).
  Defaults (0,1,1,0,1,1,0).

Both use compatibility 4*[h != (a XOR b)], leaf default cost 1 and internal
default cost 2. Defaults are consistent at every factor. Hard evidence ranges
over {missing,0,1} at every leaf: 9 and 81 patterns, respectively. For each,
compare dynamic programming and traceback to exhaustive enumeration of all
2^3 or 2^7 whole-state assignments, including minimum energy and membership of
the decoded state in the exhaustive optimum set. Check evidence preservation
and no-evidence default recovery. Preserve all optimum multiplicities.

There are exactly 72+10,368=10,440 exhaustive candidate visits. Dynamic
programming evaluates all 2^3 combinations per factor, with no pruning, for
9*8+81*3*8=2,016 conditional factor candidates. These counts include all
rejected alternatives. The direct exhaustive solver is a validation reference;
no speed conclusion is drawn from this small run. The coefficient values are
inherited from the first example; no tuning or rejected fitted candidates exist.

## 6. Complexity, storage, and cost recording

For N nodes with q states and bounded ternary factors, building supplied unary
profiles costs O(Nq). Inference costs O(Nq^3) for inward enumeration and O(N)
for traceback using stored argmins. Profiles and argmins use O(Nq) entries;
generic explicit compatibility tables use O(Nq^3) persistent entries, whereas
the supplied XOR predicate and defaults take O(N) storage. The checker instead
uses O(N*2^N) time for its exhaustive reference and retains all optimum tuples,
which can take O(N*2^N) memory. This reference cost is not deployment cost.

For vectors of d independent binary coordinates, solve each coordinate tree
separately in O(Nd) time/storage. If d bits constitute one jointly interacting
state, q=2^d and generic factor enumeration is O(N*2^(3d)); writing a compact
predicate does not eliminate candidate enumeration. Fewer nodes inward does
not automatically mean fewer state possibilities or less total memory.

No learned parameters, optimizer state, covariance, replay, retained raw samples,
or workers are present. Report supplied configuration Python-object bytes and
serialized bytes separately from logical counts. Profile/argmin peak allocation
is not independently instrumented; process RSS includes the interpreter and
exhaustive reference. BLAS and accelerators are unused.

The checker is `experiments/relational_mode_tree_analysis.py`; outputs use the
same stem under reports, with timing in a separate `-runtime.json`. Its receipt
records source/contract hashes, revision/dirty status, command, environment,
work counts, and whole-check wall/CPU time after provenance collection through
scientific serialization. Startup and runtime receipt writing are excluded.
This one-off analytical measurement is not a latency benchmark. No historical
artifact is rewritten and no comparison imputes missing historical costs.

## 7. What this would and would not establish

Together, relation-dependent Laplacians and conditional-cost interpretation
provide a concrete connection between modes, defaults, and exact settling on
a tree. They do not yet constitute one continuous graded architecture: the
spectral filter is a fixed-relation calculation, whereas the finite interpreter
selects discrete relations. In particular, the check does not compose repeated
filtering with message passing or prove convergence of that hybrid.

The substantial remaining questions are learned relation geometry, a suitable
graded state representation, cost of profiles, loops/circular topology, and
whether a local implementation preserves the exact solver's guarantees. The
next prospective study must choose which of these to address, and must compare
against the simpler direct factor-graph inference already represented here.
Renaming that inference as a module would not establish a distinct benefit.
