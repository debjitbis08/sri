# Properties of the current sparse-reader hypothesis class

2026-11-01. Symbolic analysis of the implemented architecture; no training,
checkpoint computation or new frozen contract. All project-specific propositions
below are **[derived]**. They describe possible functions, not their probability
under the optimizer.

## Exact object and scope

`src/generic_net.py` defines g and P; `experiments/fan_in_pilot.py::reader_mask`
masks only their first code-reading layer. Each row reads one of d=16 coordinates
from each operand. g also reads all eight relation-token entries. The next layer
is dense, width m=64, with ReLU, followed by a linear output (one for g, 16 for P).
`experiments/limited_memory.py::SplitLinear` gives sparse48 h=48 first-layer rows
and 16 padded zeros; earlier sparse64 has h=64. The later dense weights on padded
zeros have no effect. This note covers both and excludes extra memory branches.

For a fixed relation r and codes a,b, the comparator is exactly

    z_j = ReLU(alpha_j a[i_j] + beta_j b[k_j] + t_j(r)), j=1,...,h
    g_r(a,b) = w^T ReLU(V z + v) + beta,

where V is 64 by h and t_j(r) is bias plus the token contribution. P has the same
form without a token and with a 16-row final matrix. Fixing r is legitimate for
properties in code space. On the three discrete relations, replacing the token
path by arbitrary relation-specific first-layer biases gives a superset suitable
for capacity upper bounds; no continuous-token assumption is necessary.

Distinguish H_M (weights vary at fixed mask), union_M H_M (masks may vary),
the fitted function (weights also fixed), and the end-to-end class in which
encoders vary. Training on both g and P restricts fit to the evidence, not the
architectural function class by itself. No exact characterization of which
parameters a finite RigL/Adam schedule reaches is claimed.

## 1. Dense-class containment; no global pairwise-additivity restriction

Every masked reader is a dense reader with the missing weights set to zero.
Sparse48 additionally zeros 16 first-layer outputs. Thus, for the same encoders,
the allowed sparse-reader class is contained in the width-64 dense class. This
does not assert strict containment for every restricted input set, or equal
learning behavior.

First-layer features each involve two code entries, but the next ReLU mixes all
of them. Therefore the final function need not be a sum of independent functions
of two coordinates. For instance, on a bounded nonnegative domain and a mask
with suitable rows, pass a_1,a_2,b_1 through three rows (unused allowed weights
are zero), then compute ReLU(a_1+a_2+b_1-t) in the second layer. The threshold
involves three coordinates. Sparse first-layer connectivity is not a theorem
that the complete reader has low interaction order.

## 2. Exact blind directions and a mask-level test

Write x=(a,b) in R^32 and the first code-reading matrix as A. For fixed weights,

    A u=0  =>  g_r(x+u)=g_r(x),  P(x+u)=P(x).

Proof: the first preactivations agree, hence all later computations agree. For
a particular reader, every displacement in ker(A) is globally invisible.
If a coordinate is absent from its mask, all functions in that fixed-mask
class are independent of it. A coordinate absent from g need not be absent
from P. Extra ReLU clipping or later weights can hide still more information.

Sparsity does not force rank(A)<32 when h=48 or 64. There is an exact structural
rank test: make a bipartite graph between rows and the 32 input coordinates,
with an edge for each permitted weight. The maximum possible rank of A equals
the maximum matching size. A matching of size r permits an r-by-r permutation
submatrix with chosen weights 1 and others 0; conversely a nonzero r-minor has
a determinant term giving such a matching. Full column matching therefore
permits rank 32. Coverage alone is weaker than this condition. Actual learned
rank and ReLU information loss remain separate questions.

These are code-space statements. A null direction matters for real observations
only if distinct realizable codes differ along it. For trainable encoders,
information can be moved into coordinates the reader observes.

## 3. Piecewise-affine geometry and finite oscillation

At fixed r, g and P are continuous piecewise-affine functions of (a,b). A scalar
decision boundary g=0 is piecewise planar (possibly degenerate), not necessarily
a single plane, an order or a metric. On any affine input line x(t)=x0+t u:

    number of affine pieces <= (h+1)(64+1).

Proof: the first h ReLUs introduce at most h breakpoints, yielding h+1 intervals.
Within each interval each second-layer preactivation is affine and has at most
one isolated zero, so 64 units split it into at most 65 pieces. Identically zero
preactivations do not add breakpoints. The final linear layer adds none.

This gives at most 3,185 pieces for sparse48 and 4,225 for sparse64. These are
loose upper bounds, not attained capacities. Along any unbounded affine ray,
the reader is eventually affine, so a thresholded output cannot alternate
forever. It also cannot exactly equal a genuinely quadratic function on a
nontrivial line interval, though it can approximate it. These facts concern
logits/raw predictor outputs, not softmax probabilities or candidate-distance
scores. An encoder can map a sensory sequence to a nonlinear path, and repeated
P application increases computational depth; neither is covered by the one-pass
line bound. The bound does not imply correct arithmetic extrapolation.

## 4. A constructive non-counting relation in this class

Let three abstract items A,B,C have scalar labels 0,1,2, replicated across all
16 code coordinates. This is a code-level construction, not a claim that our
perceptual encoders will learn it. Any permitted row can compute

    t = ReLU(3 a[i] + b[k]) = 3a_label + b_label.

All nine ordered pairs have distinct integer t in 0,...,8. Define triangular
bumps using three second-layer ReLUs per bump:

    T_s(t)=ReLU(t-s+1)-2 ReLU(t-s)+ReLU(t-s-1).
    g(a,b)=T_1(t)+T_5(t)+T_6(t)-1/2.

At these integer inputs the score is positive exactly for (A,B), (B,C), (C,A)
and negative for the other six pairs. This uses one first-layer row and nine
second-layer rows, within sparse48. Set token weights to zero and unused outputs
to zero. Thus every ordinary mask with at least one row reading both operands
admits this finite cyclic classifier on these codes. The class does not impose
transitivity, symmetry, reflexivity, metric behavior or a scalar ordering.
The same bump construction can realize any binary relation on these three items
using at most 27 second-layer rows. No empirical execution is needed for the
identity T_s(integer)=1 at s and 0 at other integers.

## 5. Basis dependence and encoder compensation

For fixed codes, general orthogonal rotation can turn each one-coordinate
first-layer row into a dense row when compensating the reader weights. The
sparse architecture is therefore not guaranteed to be closed under the direct
inverse-rotation reparameterization. This argument alone is not a proof that no
other nonlinear reparameterization realizes the same function on a finite code
set. Coordinate permutations and invertible coordinatewise scalings preserve
the union over masks (weights and indices compensate); permutations need not
preserve an individual mask. Common shifts can be absorbed into biases.

If a fixed invertible transform Q is inserted after an unconstrained learned
projection L, choose L'=Q^-1 L, including its bias, so Q L'=L. Consequently that
insertion alone leaves the end-to-end representable class unchanged. It can
still change initialization, regularization and optimization. Frozen-code
rotations and joint-training rotations ask different questions.

## 6. Parameter and wiring complexity

Functionally relevant scalar parameter slots, including biases and the 24 token
embedding entries for g, but excluding encoders/projections and P's loss scale:

| Reader | sparse48 | sparse64 | dense64 |
| --- | ---: | ---: | ---: |
| g | 3,753 | 4,953 | 6,873 |
| P | 4,320 | 5,392 | 7,312 |

Derivation: g has 11h first-layer slots, 64h+64 second-layer slots, 65 output
slots and 24 token slots: 75h+153. P has 3h+(64h+64)+1040 = 67h+1104. Dense g
has 64*41+64*65+65+24 slots; dense P has 64*33+64*65+16*65. These are parameter
slots, not intrinsic dimensions (token factorizations and ReLU rescalings are
redundant), allocated tensor sizes, or FLOPs. Sparse48 also changes width; its
comparison to dense64 is not solely a change in connections.

For untied masks there are at most d^(2h) choices per reader, with description
length 2h log2(d): 384 bits for h=48 and 512 bits for h=64. Double these for
independent g and P masks. Balanced initialization or finite schedules may
restrict the reachable subset; these are upper bounds. A learner-selected
mask cannot be treated as a mask fixed independently of the observations.
At fixed weight precision b, an explicit encoding uses at most bW plus mask
bits (plus a declared format); a finite-class bound then applies under iid
bounded-loss assumptions. Encoding the encoders and any predictive memory is
necessary for an end-to-end guarantee. Training trajectories need not be encoded
merely because they were explored if the bound already covers all final models.

For fixed relation and mask, standard ReLU-network results give VC/pseudodimension
upper bounds of order O(W L log W), with parameters and depth for the compiled
reader. For all three tokens, use arbitrary relation-specific biases as a
superset. The cited general lower bounds are existence results over architectures,
not lower bounds for our particular sparse graph. A union bound over mask-specific
growth functions accounts for mask choice; alternatively dense containment gives
a safe upper bound. None of these asymptotic statements is a useful numerical
generalization certificate without further work.

## 7. Sparsity alone supplies no robustness bound

With token fixed, ReLU is 1-Lipschitz, so

    Lip(g_r) <= ||w||_2 ||V||_2 ||A||_2.

If each row has at most two code weights of magnitude at most B, and any input
coordinate has at most Delta incident rows, then

    ||A||_2 <= sqrt(||A||_1 ||A||_infinity) <= B sqrt(2 Delta).

Here matrix 1-norm is maximum absolute column sum and infinity-norm maximum
absolute row sum. No uniform B or later-layer norm bound is enforced by the
mask. Scaling output weights can make sensitivity arbitrarily large. A
conditional norm/margin guarantee is possible, but sparse wiring alone does
not prove noise robustness or cross-sense invariance.

## Prior work, limits and computation

Search 2026-11-01: "nearly tight VC dimension bounds piecewise linear neural
networks Bartlett Harvey Liaw Mehrabian 2019" and "expressivity deep neural
networks linear regions Montufar 2014". **[read: abstract, definitions and
Theorem 7 statement, not full proof]** [Bartlett et al. (2019)](https://jmlr.org/papers/volume20/17-612/17-612.pdf)
give general piecewise-linear-network capacity upper bounds. Their lower bounds
do not characterize this graph. **[abstract]** [Montufar et al. (2014)](https://papers.neurips.cc/paper_files/paper/2014/hash/fa6f2a469cc4d61a92d96e74617c3d2a-Abstract.html)
study linear regions and expressive effects of depth; the elementary one-line
upper bound above is independently derived for our two hidden layers. Neither
paper establishes transfer benefits for our training procedure. No novelty claim.

This note performs symbolic derivations only; parameter counts and the cyclic
construction are algebra, not new measured experiments. Reading source files
does not inspect learned masks, ranks or norms. A follow-up checkpoint audit
should report hashes, rank tolerances, norm algorithms and costs. With d=16,
dense SVD of A costs O(h(2d)^2) when h>=2d; computing coverage costs O(hd) for
the stored dense mask or O(h) for an edge list. Per-example sparse reader work
is O(h+64h+64o), with o=1 or 16 and fixed token dimension; the actual masked
dense implementation still allocates and multiplies dense first-layer tensors.
Include encoder extraction and any fitting/selection separately under
[computation reporting](computation-reporting.md). No new runs were launched.
