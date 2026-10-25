# A masked-capacity obstruction from radial nullspaces and weak lateral coupling

**Status:** analytical derivation, 2026-10-25. Derived after seeing progress
from the frozen [bounded capacity search](single-module-masked-capacity-v1-contract.md).
This is an additional proof, not an inference that failed optimization proves
impossibility. The main search protocol and its output classifications remain
unchanged. The accompanying receipt checks this proof against saved states;
it introduces no fitting, task selection or new observations.

## Claim and scope

For the exact masked module in the contract, with fixed parameters, there is
at least one scene in **each** (relation, amplitude-ratio) group whose
normalized hidden squared error is at least

```
0.5 * (1 - 8 sqrt(15)/65)^2 > 0.136.
```

This excludes the <=0.10-on-every-scene capacity criterion for every allowed
parameter setting, including matched input-dependent routes. The statement
is about this energy, gate floor, coefficients, orthonormal maps and rank-one
terminal nullspace. It is not an impossibility theorem for circular modules,
reciprocal inference, general masked energies or arbitrary nonlinear networks.

## 1. Structural decomposition

For a particular input, fix the routes selected by the pilot. Let alpha=0.1,
lambda=0.001, and let L be the block-diagonal unweighted cycle Laplacian of
the four rings. Its spectral norm is at most 4. Write the second-pass energy

```
E(x) = alpha/2 ||M_obs(x_0-h)||^2
       + 1/2 sum_j x_j^T K_j x_j + lambda/2 x^T L x.
```

x_j concatenates the four rings' coordinates in sector j. K_j is the radial
and terminal Hessian for that sector, excluding all lateral and observation
terms. Every scalar route a_l,j is >= a_min=0.325 because g>=0.1 and
a=0.25+0.75g. The actual gate formula can give a stronger bound; this one is
sufficient and conservative.

With orthonormal-column D_0,D_1,D_2 and unit v, define the unit outer vector

```
w_j = D_0,j D_1,j D_2,j v_j,
P_j = I - w_j w_j^T.
```

K_j has one null direction, proportional to
`(D_0 D_1 D_2 v, D_1 D_2 v, D_2 v, v)`. The outer component of that null
direction is w_j, independent of the positive route values. Changing gates
cannot rotate this null direction while parameters remain fixed.

## 2. Bound on radial response outside the passthrough direction

Choose an orthonormal basis (v,v_perp) in ring 3, lift it through D_2 and
complete a basis of ring 2, then repeat through D_1 and D_0. In these bases
the radial energy separates into scalar chains:

- two length-one anchored chains;
- two length-two anchored chains;
- two length-three anchored chains;
- one length-four anchored chain, ending at v_perp;
- one free length-four chain, ending at v.

Projection by P_j removes the entire outer direction of the free chain.
For an anchored chain of length k<=4, its energy is

```
1/2 sum_(i=0)^(k-2) a_i (z_i-z_(i+1))^2 + a_(k-1)/2 z_(k-1)^2.
```

The first row of the inverse Hessian has entries
`b_j = sum_(i=j)^(k-1) 1/a_i`, j=0,...,k-1. Hence its Euclidean norm is
at most

```
sqrt(k^2+(k-1)^2+...+1)/a_min <= sqrt(30)/a_min.
```

The chains occupy orthogonal coordinates. The operator B_j taking a radial
force to the outer response perpendicular to w_j therefore satisfies
`||B_j||_2 <= sqrt(30)/a_min`. Equivalently, B_j is P_j times the outer rows
of K_j's pseudoinverse. This inverse is only an analytical object; inference
still uses the complete positive-definite masked Hessian.

## 3. Bound on the force supplied through lateral edges

For each hidden sector, no observation anchor or right-hand-side input occurs
at any of its rings. Stationarity gives

```
K_j x_j = -lambda (Lx)_j.
```

Combining the two hidden sectors and using the bound above,

```
||(P_2 x_0,2, P_3 x_0,3)||
 <= sqrt(30)/a_min * lambda ||Lx||.
```

The second-pass state minimizes its energy with routes fixed. Its energy
cannot exceed that of x=0. Since ||h_visible||^2=2, E(0)=alpha. Nonnegativity
of every term gives `lambda x^T L x <= 2 alpha`. The Laplacian bound implies

```
lambda^2 ||Lx||^2 <= 4 lambda^2 x^T L x <= 8 alpha lambda.
```

Consequently the combined hidden output perpendicular to its two passthrough
lines has norm at most

```
C = sqrt(240 alpha lambda)/a_min = 8 sqrt(15)/65 < 0.477.
```

The comparison E(x)<=E(0) uses the **same** realized routes on both sides.
Input dependence of those routes does not invalidate the argument.

## 4. The task requires a larger perpendicular component

Hold r and q fixed and average over the eight equally spaced orientations t.
For either hidden part, the target rotates in a fixed two-dimensional plane.
For any fixed unit w_j, its average squared component perpendicular to w_j
is at least half of that part's squared amplitude. Orthogonal renderings
preserve this statement.

The two hidden squared amplitudes sum to 2, so

```
mean_t ||(P_2 s_2(t), P_3 s_3(t))||^2 >= 1.
```

There is therefore at least one of those eight scenes with target
perpendicular norm >=1. Projecting its prediction error and using the reverse
triangle inequality gives

```
||prediction_hidden-target_hidden||^2/2 >= (1-C)^2/2 > 0.136.
```

Every relation-ratio group includes all eight t. Thus the conclusion holds
within each group, including each held-out group. The projection directions
depend on learned parameters but not on t; the predicted routes may depend
on t arbitrarily within their positive bounds. This proves the claim for
the full routed predictor.

## Interpretation

The mask corrects the observation semantics but cannot supply a second freely
expressed direction in each hidden sector. In this configuration, that
direction must be driven through lateral interactions against positive radial
penalties, and the available force is uniformly too small for the task target.

Any next capacity design must explicitly reconsider at least one assumption
used here: the one-dimensional terminal nullspace, gate floor, relative
coupling strengths, or connection structure. Making this bound inconclusive
would not establish capacity. Such a change needs a new prospective contract
and a fresh witness test; it must not be introduced silently into this study.
