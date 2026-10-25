# Small-cycle relational modes and exact interpretation

Status: analytical preflight, 2026-10-25. Continuation of the
[tree analysis](relational-mode-tree-analysis.md). Supplied binary relations and
preferences; no learning, new capacity witness, or selected SRI architecture.
Earlier artifacts remain unchanged.

## 1. States, observations, and stored preferences

Use an n-cycle for n=3 or 4. Outer bits b_i have stored preference 0. Each edge
(i,i+1 mod n) has a variable relation bit h_i, with 0 meaning same and 1 opposite,
and a supplied stored preference t_i. Relation states are inferred, not observed.
Outer evidence is hard and uses an explicit missing/0/1 mask. The actual outer
bits are outputs; there is no extra decoder. The energy is

```
E(b,h | t) = sum_i [b_i != 0]
           + beta * sum_i [h_i != t_i]
           + 4 * sum_i [h_i != (b_i XOR b_(i+1))].
```

This is a local factor-graph model with n outer and n relation bits. It does not
yet compress n outer units into fewer inner units. It isolates the consistency
and interpretation problem caused by a cycle. Stored preferences are fixed
during inference; no query writes a learned state. Coefficients beta=2,4,6
represent the three analytical regimes below, not a tuned candidate menu.

## 2. Exact cycle closure and frequency

Write x_i=(-1)^b_i and r_i=(-1)^h_i. Fixed relations define the signed Laplacian

```
x^T L_h x = sum_i (x_i-r_i*x_(i+1))^2.
```

Each zero-energy edge requires x_(i+1)=r_i*x_i. Propagation around the cycle
therefore requires x_0=(product_i r_i)*x_0.

- If XOR_i h_i=0, the product is +1. There is one scalar zero mode, generated
  by choosing x_0 and propagating it. Two binary outer configurations (global
  complements) satisfy every relation.
- If XOR_i h_i=1, the product is -1. The only real perfectly aligned field is
  zero, and no binary outer configuration satisfies every relation. At least
  one edge must disagree. Without outer evidence, exactly one mismatch can be
  achieved by propagating along a cut chain.

Let e_i=h_i XOR b_i XOR b_(i+1) indicate a mismatch. Cancellation of each b_i
twice gives XOR_i e_i=XOR_i h_i. This parity identity holds with any evidence;
evidence may require more than the minimum number of mismatches.

By a diagonal change of sign, every signed cycle reduces to positive edges
except possibly the closing edge. Its spectrum depends only on that edge's
sign (the cycle product):

```
lambda_k = 2-2*cos((2*pi*k + phi)/n),
phi = 0 for balanced signs, pi for frustrated signs.
```

| Cycle | Balanced eigenvalues | Frustrated eigenvalues |
| --- | --- | --- |
| n=3 | 0,3,3 | 1,1,4 |
| n=4 | 0,2,2,4 | 2-sqrt(2),2-sqrt(2),2+sqrt(2),2+sqrt(2) |

Thus fixed-relation smoothing (I+tau L_h)^-1 preserves the relational zero mode
on a balanced cycle. A frustrated cycle has no exactly preserved mode for
tau>0. It can still have low-frequency modes: for n=4 its smallest eigenvalue
is 2-sqrt(2). Repeated filtering of the whole unforced field tends to zero in
the frustrated case, but this is NOT a prediction that a driven system or the
binary constrained interpreter below becomes zero. Priors and evidence are
separate forces/constraints. A positive-definite frustrated Laplacian is not
intrinsically unstable; its issue is lack of perfect relational compatibility.

For d-dimensional orthogonal transports, closure similarly requires the product
around the cycle to fix the root vector. The zero-mode dimension equals the
dimension of that product's fixed subspace. A nonidentity product may still
preserve some directions; the scalar sign case is only the simplest instance.

## 3. The relation/default tradeoff is analytically solvable

For fixed neighboring bits a,b, eliminate their edge relation:

```
phi_i(a,b) = min_h {4*[h != (a XOR b)] + beta*[h != t_i]}
           = min(4,beta) * [(a XOR b) != t_i].
```

If the pair agrees with the stored relation, h=t_i is uniquely optimal. If it
does not, the cases are:

| beta | Selected h | Interpretation |
| --- | --- | --- |
| 2 < 4 | a XOR b | Change the preferred relation; keep exact compatibility |
| 4 | Either a XOR b or t_i | Equal-cost interpretations; record the tie |
| 6 > 4 | t_i | Retain the preferred relation; accept an edge mismatch |

For beta=2 every global optimum has h_i=b_i XOR b_(i+1), hence balanced relation
signs and zero compatibility cost. This does not make every stored preference
achievable. An odd-parity preference necessarily changes on at least one edge.
For beta=6 every optimum keeps h=t. An odd-parity stored cycle therefore remains
frustrated with at least one mismatch. For beta=4 the objective may not decide
whether a disagreement is explained by a revised relation or a residual error.

This identifies where the inferred geometry comes from: it depends on the
declared relative strength of relation preference and compatibility. It is not
a free consequence of reciprocal communication or smoothing.

After elimination the outer objective is

```
E_eff(b) = sum_i [b_i != 0] + sum_i phi_i(b_i,b_(i+1)),
```

with hard observed bits retained. It is a binary pairwise cycle model. The
reference must therefore be compared to ordinary exact cycle inference; this
construction by itself is not a new algorithmic advantage.

## 4. Exact finite realization without ignoring the closing edge

Choose a cut node c and condition on b_c=0, then on b_c=1. For each root value,
perform chain dynamic programming around the cycle with profiles

```
F_0(a) = unary_c(a) if a equals the conditioned root, else infinity
F_j(b) = unary_j(b) + min_a [F_(j-1)(a) + phi_(j-1)(a,b)].
```

Add the last-to-root edge cost BEFORE selecting the terminal/root states.
Trace back the selected chain states, then select a minimizing relation on
each edge. Preferences are counted once; infeasible root conditions remain
infinite. Ties select the smallest state at each declared minimization. This
can depend on the cut when optima are not unique. All cuts must agree on optimum
energy, not necessarily on the decoded representative.

Proof: every complete cycle assignment belongs to exactly one conditioned root
case; the chain recurrence is exact within that case, and the closing cost
restores the omitted factor. Minimizing across root cases is global minimization.
This is a bounded realization procedure, not a convergence theorem for loopy
belief propagation or asynchronous scalar activations. It includes the closing
edge rather than treating the graph as a tree.

On a q-state outer cycle with p possible relation states per edge, local relation
elimination costs O(n*q^2*p) time and O(n*q^2) pair-cost/argmin storage. Conditioned
chain inference costs O(n*q^3) time. Processing root cases sequentially needs
O(n*q) chain traceback workspace plus pair tables; costs alone need O(q) chain
workspace. Traceback and relation selection cost O(n). This is linear in cycle
length for fixed alphabets, unlike general densely coupled graphs. Testing all
n cuts multiplies inference cost by n; a deployment needs only one cut.

The simplification relies on each h_i occurring in its own edge factor and unary
preference only. Coupling relation variables through a deeper core invalidates
this independent elimination and requires a new complexity/consistency analysis.
For d independent feature coordinates, costs multiply by d; arbitrary joint
coordinates can give q=2^d and exponential alphabet costs. No multidimensional
scaling claim is made without that qualification.

## 5. Context, defaults, and expected hand-checkable cases

On the triangle with t=000 and no evidence, b=000,h=000 has unique energy 0
for every declared beta. With t=000, beta=2, and b0 observed as 1, the optimum
is b=111,h=000 with energy 3. Under the same evidence but t=101, the optimum
is b=100,h=101 with energy 1. These are different contextual completions using
different stored relationships, not a test of learned relation discovery.

For triangle t=111 with no evidence:

- beta=2: any one-hot b minimizes energy at 3; h follows its even-parity relation
  pattern, revising exactly one stored edge. There are three optima.
- beta=6: the same three one-hot b minimize energy at 5, but h remains 111;
  exactly one edge mismatch remains. There are three optima.
- beta=4: energy 5, with both relation explanations per one-hot b; six optima.

These predictions are fixed before enumeration. The all-opposite square has
balanced preferences, but its two alternating outer patterns can tie. Defaults
can be mutually incompatible or insufficiently asymmetric: C(0,p)=p for each
isolated profile does not imply simultaneous recovery of all local defaults
after nonneutral messages arrive from its neighbors. Symmetric ties are reported,
not presented as instability. Tie-breaking is a realization convention.

## 6. Declared verification menu and cost scope

Before execution, enumerate n in {3,4}, all 2^n relation-preference patterns,
beta in {2,4,6}, and all 3^n outer evidence patterns in {missing,0,1}^n.
There are 648 triangle and 3,888 square cases (4,536 total). Each case is checked
against exhaustive enumeration of all 2^(2n) outer/relation assignments. Count
all candidate visits, feasible evaluations, and rejected alternatives. Expected
exhaustive visits: 1,036,800; feasible scores: 208,896.

Precompute eliminated pair costs/selected relations once per n,t,beta and reuse
across evidence/cuts. All eight (a,b,h) alternatives per edge are counted:
2,112 construction candidates. Check the analytical elimination identity for
every pair. Run exact conditioned-chain inference at every possible cut: 17,496
calls with 474,336 transition/closing candidates, counting infeasible alternatives.
Check optimum energy, decoded state membership in the exhaustive optimum set,
observations, relation/mismatch parity, and the beta-specific conclusions.
Record optimum multiplicities and possible balanced/frustrated explanations.
Assert the hand-checkable cases in section 5. No numerical optimizer is used.

For each of the 8 triangle and 16 square sign assignments, construct L and check
its characteristic polynomial by exact integer determinant expansion of zI-L.
Expected ascending coefficients are (0,9,-6,1) or (-4,9,-6,1) for triangles, and
(0,-16,20,-8,1) or (4,-16,20,-8,1) for squares. Check all 2^n binary outer states
for the mismatch parity identity and the squared-residual identity. These are
finite exact checks, not numeric eigensolver tests or a spectral scaling sweep.

The checker is `experiments/relational_cycle_analysis.py`, with deterministic
results and a separate runtime JSON under reports. Integer arithmetic and an
infinity sentinel are sufficient. No random seeds, solver tolerances, training,
tuning, empirical examples, replay, optimizer, covariance, BLAS or accelerator.
Two topology sizes and three analytically selected regimes are verification
coverage, not models selected by observed scores.

Exhaustive reference inference costs O(n*2^(2n)) per case; it retains all optimum
tuples, up to O(n*2^(2n)) storage, and is not deployment cost. Polynomial
determinant expansion costs O(n!*n^2) with naive small polynomial multiplication,
used only for n<=4. Persistent supplied configuration includes topology sizes,
coefficients, and the recipe for all preferences; explicit preference arrays
are generated, not learned. Report configuration object/serialized bytes and
process peak RSS separately. Fine-grained native/allocator workspace is
unavailable; whole-process RSS includes the exhaustive reference and interpreter.

Measure the whole analytical check through scientific serialization after
imports/provenance. Record source/contract hashes, revision and dirty state,
command, environment, wall/CPU time, RSS and work counts; runtime receipt
serialization and interpreter startup are excluded. This is not an inference
latency benchmark. Manual research/editing time is unmeasured. Do not retrofit
old receipts or compare these small checks as evidence of architecture speed.

## 7. Decision boundary

Passing these checks would establish compatible low modes and exact contextual
interpretation for this small supplied cycle family. It would not prove learned
geometry, dimensional reduction, convergence of a graded recurrent system, or a
capacity witness on the existing task. The next structural question is how a
smaller inner representation constrains multiple edge relations without losing
the mode and inference guarantees obtained here.
