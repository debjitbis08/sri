# Analytical preflight: relational capacity and module composition

Status: analytical study, 2026-10-25. This develops the
[proposal](relational-capacity-and-composition-study-proposal.md), before any new
capacity fitting. The statements below are derived for declared quadratic or
explicitly nonlinear candidates, not asserted as foundations of SRI. Numerical
checks are finite proof illustrations, not a parameter search or learning study.

## 1. Conclusions that change the study design

1. Two retained terminal dimensions remove the established rank-one obstruction,
   and the existing positive circular lateral energy still guarantees a unique
   masked equilibrium. Capacity on the relational task remains unproved.
2. Multiple modules can supply complementary coordinates. However, multiplicity
   and reciprocal interaction must be distinguished: summed independent outputs
   already enlarge the output span. Under fixed quadratic coefficients, any
   finite composition remains linear and fails the known three-relation task.
3. The repository assigns evidence and observation to different modules. Use a
   designated output module for the primary composition candidate; keep summed
   outputs as a separately labelled capacity control. A disconnected designated
   output is only a connectivity control, not a sufficient abstraction baseline.
4. Adding an unobserved passive module cannot enlarge the zero-energy subspace
   of unchanged existing states. It can alter their effective stiffness and
   correlations. New output states and input-dependent interactions require
   separate analysis; the result does not rule them out.
5. Fixed relational transports can preserve one relationship. Several different
   relations cannot be represented by simultaneously enforcing all their
   constraints, or by freely admitting their linear span. Relation-dependent
   inference is the unresolved mechanism, not simply extra width.
6. Uniform rotation transport around a four-part cycle is incompatible with two
   of our three task relations. Physical circular layout does not imply periodic
   closure of the task's progression. A correct boundary correspondence matters.
7. The current scalar-coordinate cycle penalty is not automatically equivariant
   to rotations of paired feature coordinates. A geometric arm changes more
   than its terminal rank; its lateral energy and masks must also be compatible.

## 2. Rank-two retention: unique inference, unproved task capacity

For sector j write radial maps D_l with orthonormal columns and terminal
projector Q=I-VV^T, where V has k orthonormal columns. All radial coefficients
are strictly positive. The sector radial nullspace is

```
x3 = V c
x2 = D2 V c
x1 = D1 D2 V c
x0 = D0 D1 D2 V c,          c in R^k.
```

The maps are injective, so the nullity is k. With terminal width two, k=2
means Q=0: there is no terminal penalty or effective terminal route. A learned
orthogonal 2-by-2 V changes no projector and should not be counted as a useful
learned parameter. The old stored terminal vectors would be redundant.

For the homogeneous full masked energy, zero energy makes every ring constant
because its scalar cycle is connected and has positive weight. One observed
outer coordinate makes the outer constant zero. The radial equalities then
force each inner sector to zero, because D0,D1,D2 are injective. Consequently
the Hessian is positive definite, even with Q=0. This holds for any realized
positive routes and at least one observed outer coordinate. It does not need
random maps, a ridge, a terminal anchor, or full outer observation.

Without the lateral cycles, a completely hidden sector has k unanchored modes.
Thus removing smoothing while widening terminal retention is a different design
and generally destroys uniqueness in the isolated masked module. Positive
definiteness gives no useful condition-number bound by itself.

The old perpendicular-target bound no longer forces error: a rank-two lifted
plane can contain a target's entire orientation plane. This is only removal of
an obstruction. The radial maps must learn the plane, and the system must still
infer a relation and propagate missing amplitudes and directions.

## 3. General coupling, ports and evidence

Let x concatenate all module states and write the fixed-route energy as

```
E(x;h) = 1/2 ||W^(1/2)(A x-h)||^2
       + 1/2 ||R x||^2
       + 1/2 sum_e mu_e ||P_e x_i - T_e Q_e x_j||^2.
```

R includes local radial, terminal and lateral residuals. W contains observation
weights, A only actual observations, and P,Q select exposed ports. Every
coupling contributes a symmetric PSD block; the derivative on the other side
contains the transpose transport. Arbitrary one-way replacement updates do not
implement this energy. Evidence is external; inferred input states can change.

The Hessian is positive definite exactly when the intersection of the kernels
of all weighted observation, local and coupling residual operators is {0}.
This is a directly checkable identifiability criterion, not a blanket guarantee
for every port choice. If it fails, the output Cx is nevertheless unique iff
`ker(H) subset ker(C)`. Otherwise a minimum-norm solver supplies an extra
selection convention and cannot be presented as unique model inference.

A sufficient two-module construction is: input-only masked anchors, the current
positive ring cycles in both modules, full-rank radial maps, and an injective
correspondence between their complete terminal states. Zero energy first forces
all input states to zero as in section 2. Coupling forces the output terminal
state to zero, and radial equalities force its other rings to zero. This works
with either terminal rank. It establishes solvability, not representability.

This follows the evidence/output separation in
[interaction semantics](interaction-semantics.md). That document does not select
quadratic energy as SRI's foundation. The older
[reciprocal participation contract](learned-reciprocal-participation-v1-contract.md)
also cautions against external expert selection or dropping modules. A new
relation selector cannot silently bypass that architectural question.

For a fixed block Hessian and no direct output evidence,

```
[A  -B][x] = [F h]
[-B' D][z]   [ 0 ],

z = D^-1 B' (A-B D^-1 B')^-1 F h.
```

With SPD blocks, the fixed-route input-to-output rank is at most rank(B).
Scalar communication can carry only a rank-one linear image in this setting;
two independent channels can carry two. Multiple edges can add rank. Adaptive
routes may change the transfer operator between observations, so this is not
a universal rank bound on all nonlinear compositions.

## 4. What a passive added module can and cannot do

Suppose an existing state x has stiffness H0 and an unobserved added state z
has local PSD stiffness G. Add `mu/2 ||P x-Q z||^2`. If
`D=G+mu Q'Q` is invertible, elimination gives

```
H_eff = H0 + Delta
Delta = mu P'P - mu^2 P'Q D^-1 Q'P >= 0.
```

The inequality follows because `x'Delta x/2` is the minimum of the added
nonnegative energy over z. Therefore `ker(H_eff) subset ker(H0)` for PSD H0.
A passive helper cannot create a new zero-penalty direction at x while all
existing penalties remain. It can create useful cross-coordinate terms and
change predictions; PSD ordering does not order individual reconstruction
errors. If G=0 and Q is square invertible, Delta=0: the helper can simply follow
P x, imposing no additional constraint on x.

If the output is newly introduced z, its coordinates can express information
absent from the previous output interface. If the added module has observations,
it also contributes a right-hand side. If gates change with input, a family of
effective operators is obtained. None of these cases is excluded by the passive
zero-mode statement.

For independent scalar states z1,z2 and output `w1 z1+w2 z2`, the accessible
span is span(w1,w2), possibly dimension two without any interaction. For output
in module 1 only, adding a disconnected module changes nothing. These must not
be conflated. A disconnected designated output has no information about h;
beating it establishes communication, not relational abstraction.

## 5. Fixed linear composition still cannot solve the task

With fixed masks and fixed coefficients, arbitrary reciprocal depth, ports,
elimination, and fixed linear readouts still yield a linear predictor. Biases
only enlarge this to affine prediction. The q=1 subset of the existing task has
best affine mean error `(8-2 sqrt(2))/21`, approximately 0.2463. It follows that
no such system can achieve error <=0.10 on every scene. This excludes fixed-route
multi-module constructions as successful capacity arms, while retaining them as
mechanism controls. It does not exclude adaptive routing, nonlinear interactions,
or a changed task.

Moreover, positive coefficients on fixed PSD penalties preserve the intersection
of their nullspaces. Adaptive strengths can alter positive-curvature responses,
but cannot rotate the common zero-energy subspace. That is why the existing
gate nonlinearity may be insufficient even after a width change; whether it is
insufficient requires a new bound or a witness, not this observation alone.

## 6. Relations, mixtures and circular closure

Consider normalized planar parts u_j and one relation delta. Let T=R(delta).
The open-chain constraints `u_(j+1)=T u_j` for j=0,1,2 admit a two-dimensional
joint subspace. One observed complete vector determines the rest if T is known.
Actual task amplitudes alternate, so pure rotation constraints on raw parts are
wrong for q != 1. They need amplitude factors (q,1/q,q), separate amplitude
states, or a normalization mechanism; none should be supplied secretly.

For different deltas, these joint subspaces are distinct. Two such subspaces
intersect only at zero: equal first vectors and different planar rotations
force that vector to zero. Enforcing all relation penalties with positive
weights therefore destroys the free signal. Conversely, their sum/span admits
mixtures that are not any valid relation. In the q=1 fixture, three relation
subspaces have combined real dimension six, rather than three discrete choices
of a two-dimensional family. Neither conjunction nor unrestricted superposition
implements the union of valid relationships.

Now add an identical fourth edge `u0=T u3` to close the physical cycle.
Consistency requires `u0=T^4 u0`. For delta=r*pi/4, T^4=(-1)^r I. Relations
r=1 and r=3 force u0=0; r=2 permits a two-dimensional family. This is a task/
boundary mismatch, not a failure of circular architectures. Options include an
open relational chain or a closing transport T^-3, whose product around the
cycle is identity. The latter preserves circular incidence but specifies a
different boundary role. With amplitudes, the closing factor is also 1/q.
Its dependence on the inferred relation and amplitude must be declared.

More generally, on an invertible-transport cycle the permissible root states
are fixed vectors of the product of edge transports. The cycle product tests
global consistency. It does not explain how transports are acquired or how
conflicting observations should be resolved.

## 7. Geometric compatibility without hiding a task oracle

For a common planar rotation g, assign an orthogonal action U(g) on states.
Sufficient conditions for a fixed quadratic solve to be equivariant are
`H U = U H`, `F V = U F` for input action V, and a compatible output map.
For adaptive H(h), require `H(Vh)=U H(h) U'`. Pilot invariance and invariant
residual norms can give invariant scalar gates, but only if the maps, lateral
terms and observation model already transform compatibly.

The ordinary cycle on scalar coordinates does not generally commute with
block-diagonal rotations of adjacent coordinate pairs. A vector-channel cycle
with Hessian `L_sites tensor I2` does. Therefore adopting vector channels entails
an explicit change in lateral connectivity/weights, not simply renaming scalars.
Radial maps must intertwine the actions as well. Whole-vector observation masks
are compatible with rotation; a fixed single-coordinate mask generally is not.
For partial measurements, transform the measurement operator together with the
scene when evaluating equivariance. Do not demand a false fixed-mask symmetry.

For local orthogonal frame changes G_i, a transport must change as
`T_ij -> G_j T_ij G_i'` in residual `x_j-T_ij x_i`. This preserves residual norms.
Independent learned sector renderings do not provide the G_i or the signal plane
for free. A known-frame geometric control and a learned-frame arm make different
claims. For SO(2), linear vector-to-vector maps commuting with every rotation
have form aI+bJ (J is a quarter turn); imposing reflection equivariance removes
the bJ term for ordinary vectors. Reflection could erase signed distinctions
required by this task and must not be assumed to be a nuisance.

The normalized dot product and determinant of two nonzero vectors identify their
relative angle while preserving chirality. A supplied formula for these features
would already extract the task relation. Use it as a privileged feasibility
reference, not as evidence of learned abstraction.

## 8. Decision table and next contract

| Candidate | Established here | Still required |
| --- | --- | --- |
| Isolated rank two, existing lateral energy | Unique masked equilibrium; old rank-one bound removed | Nonlinear capacity witness |
| Two modules, terminal ports, designated output | Sufficient SPD condition; fixed-route port-rank limit | Route mechanism, port correspondence and capacity |
| Independent summed outputs | Complementary dimensions possible without interaction | Explicit non-primary readout control and matched budget |
| Fixed-route composition | Cannot pass current all-scene criterion | Control only; no capacity search needed |
| Fixed transport for one relation | Joint 2D family and completion when relation supplied | Multiple relations, amplitude handling, learning |
| Several positive fixed relation penalties | Common zero-mode intersection can collapse | Context-dependent relation mechanism |
| Geometric vector channels | Algebraic compatibility conditions available | Changed lateral energy, frame acquisition, learning |

Recommend retaining S2 as the minimal bottleneck control, and making the primary
composition candidate two modules with input-only evidence, complete terminal
ports, and output at the other module. Use fixed-route composition as an excluded
capacity/control arm. Do not yet launch R/G arms: a relation-dependent reciprocal
mechanism and its learning/inference semantics remain unspecified. Joint latent
relation states are a candidate, but making transport depend on them makes energy
nonquadratic and requires a new existence/selection/convergence analysis. An
external oracle router would answer a different question.

For a useful M0 comparator, disconnected output alone is inadequate. Include a
direct fixed linear transfer reference and, if testing distributed coordinates,
the explicitly summed independent-module control. Match observation strength,
output semantics within each causal comparison, retained bytes and actual costs.
Do not present all these as a single exact parameter-matched factorial.

## 9. Bounded numerical receipt and computation

`experiments/relational_composition_analysis.py` checks these algebraic statements
on fixed fixtures: two map seeds, terminal ranks one/two, routes all one, terminal
identity coupling mu=0.1, actual module dimensions 80/160; tiny passive-helper,
transport-cycle and geometric commutator examples. No training data are fit,
no candidate is selected, and no reconstruction score guides changes. The proof
scope comes from the derivations, not two sampled map configurations.

Dense assembly costs O(D^2) per residual block, eigenvalue checks and direct
solves O(D^3) time and O(D^2) storage. The two-module 160-by-160 float64 Hessian
alone occupies 204800 bytes; process workspace is larger. Persistent learned
state, optimizer state and replay are zero: parameters here are constructed
fixtures. Retained output reports are verification artifacts, not a learner.
An eventual coupled solver has dense O((Km)^3) time versus K independent
O(m^3) solves; equal state/parameter counts do not imply equal compute.

The receipt records actual decomposition/solve counts, environment, source
hashes, revision/dirty status, wall and process CPU time, and peak RSS in a
separate runtime JSON. Native LAPACK workspace and FLOPs are not measured.
Run a second execution to check deterministic scientific output and retain its
runtime separately. No inference-speed, learning, or statistical claim is made.
