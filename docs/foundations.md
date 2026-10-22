# Mathematical foundations — version 0.1

## Research objective
Determine whether finite, nested, graded-coupling structures can support efficient collective inference, intrinsic structural memory, and eventually cross-module abstraction. No claim of intelligence follows from the baseline model.

## Commitments from the research discussion
- Modules are core plus specialized satellites, including a structurally plastic STM satellite.
- Layers are response modes, **not** levels of abstraction.
- Inter-module coupling is graded, typically stronger toward the interior but not limited to it.
- Input interpretation participates in feedback; external evidence remains distinct from its interpretation.
- Exact mathematics may admit weak global excitations; practical inference may truncate with an explicit approximation budget.
- Abstraction is a property of the *whole* coupled module response, not an individual shell.
- Dreaming is endogenous randomized excitation of the learned structure; no historical experience/replay store.
- Neither spheres, linearity, oscillation, nor any one algebra is presumed necessary.
- Ordinary consumer CPUs and bounded inference cost are the eventual design target.

## Baseline model (deliberately narrower)
Finite undirected weighted graph G=(V,E,w), 0<=w<=255, with each node labeled by module and layer. For fixed threshold h, define N_h(A) as all endpoints adjacent to A by edges of weight >=h. Given seed set X define F_h(A)=X union A union N_h(A).

Iterate A_0=X, A_(t+1)=F_h(A_t). A star is the least fixed point above X.

**This is merely an executable reachability reference.** It does NOT yet model cancellation, nonadditive interference, mode equivalence, internal reconstruction, multi-hole topology, inference correctness, or learned abstraction. In particular its thresholded reachability is not the same as the originally proposed nonzero weak global excitation. Nothing in the baseline establishes those richer properties.

## Proof 1: finite convergence
Every iteration is inflationary: A_t subset A_(t+1). If not stationary, at least one previously inactive node becomes active. Since V is finite, at most |V|-|X| strict expansions occur. Hence the process reaches its least fixed point within at most |V|-|X| strict expansions (plus at most one terminal no-change check).

## Proof 2: threshold inclusion
For h_1<=h_2, the permitted edge set at h_2 is contained in that at h_1. Therefore reachable_h2(X) subset reachable_h1(X). This is an exact inclusion guarantee, **not** a numerical approximation error bound for an interference model.

## Proof 3: adding an edge
If G' is obtained by adding an edge to G with no deletion or weakening, then reachable_G(X) subset reachable_G'(X) at the same threshold. This does not protect responses for nonmonotone algebras.

## Limitations and next proof obligations
1. Define at least two competing **nonadditive** local algebras and observable response signatures.
2. Produce examples where coupling changes a response without reducing inference to reachability.
3. Establish a stability or bounded-error theorem for pruning weak excitation, including feedback amplification and path multiplicity.
4. Separate temporary state from persistent structural learning; test whether STM topology/coupling can change quickly while long-term structure remains stable.
5. Define a structural, no-history dream generator and verify that its changes improve out-of-sample behavior under controls.
6. Formalize whole-module abstraction as preservation of families of responses; do not hard-code correct abstractions.
7. Compare graphs, cyclic cell complexes, and multi-hole complexes under matched budgets and rules.
8. Measure active-set growth, memory traffic, operation counts, latency, and storage against baselines.

## Experimental standards
- Every claim must identify assumptions, proof, tests, and known counterexamples.
- Use fixed random seeds and commit frozen, machine-readable reports.
- Distinguish exhaustive verification from samples and heuristics.
- Keep counterexamples and negative results.
- Do not infer intelligence, abstraction, or convergence merely from activation.
