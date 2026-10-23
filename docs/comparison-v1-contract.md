# Comparative modeling — bounded contract v1

**Status at declaration:** specified before running the new comparison. This is
an in-session contract, not an external preregistration. It fixes four small
representatives; it does not exhaust their mathematical families or choose SRI.

## Shared questions and limits

Each representative has a dedicated evidence interface and a distinct output
coordinate. Two evidence labels are encoded explicitly below. Test distinguishable
driven responses, change under context at unchanged evidence, internal disturbance
recovery, withdrawal, and native approximation/cost properties. These are common
questions, not a common capability task: real-valued and categorical outputs,
precision, storage, and native perturbations are not equated. No overall score.

Report unresolved architecture requirements rather than silently supplying
layers, learning, autonomous core mediation, or a topology advantage. The models
are deliberately restricted controls. Static reconstruction and asymptotic
dynamical recovery receive different labels. No all-fair schedule claim is made.
Uncoupled or core-relation-disabled controls use the same coordinates and readout.

## S — grounded spectral/operator configuration

Four real coordinates `(I,K,O,S)` are the input interpretation, core, output,
and context satellite. The whole configuration minimizes

`E(z) = (sum_i z_i^2 + (z_I-z_K)^2 + (z_K-z_O)^2
          + g*(z_K-z_S)^2)/2 - e*z_I`.

Evidence `e ∈ {-1,+1}` enters only I. Context coupling
`g ∈ {0,1/4,1/2,1,2}`; g=0 retains but decouples S. Core-disabled control
removes all three pair terms. Exact equilibrium solves `Qz=e*unit_I` over
rationals. Modes are eigenspaces of symmetric Q; repeated eigenvalues identify
subspaces, not distinguished basis vectors. Jacobi eigenvectors are diagnostic
floating approximations, checked by residuals, never used to certify the response.

Continuation is simultaneous `z <- z-(Qz-b)/16`. Withdrawal sets b=0 and leaves
Q unchanged; evidence removal is not a persistent memory store. Disturb each of
the four internal coordinates by ±1 at equilibrium. Run 128 continuation steps;
report exact errors and a contraction certificate. Nesting by spectral cutoff is
an available representation choice but is not asserted to produce SRI layers.
For approximation, test a residual error bound and omission of the context term.
Also test numerical spectral cutoffs 1,2,4,8, converting their constructed
approximations to rational coordinates before certifying the residual bound.

## A — affine finite algebra over F3

The four coordinates `(I,K,O,S)` take values in F3. Evidence labels map to I=1,2.
Collective compatibility is

`-I+K+O=0,  K+2O+S=0,  S=t`, with context `t ∈ {0,1}`.

Driven response additionally requires I=e. Use modular elimination; if free
coordinates exist, keep their previous values and solve pivots in fixed column
order. This is the declared joint selector, not an independently justified
physical dynamic. Withdrawal removes I=e; all other relations remain. Test all
81 starts for each driven case and all single-coordinate replacements at driven
and released responses. Report whether persistence comes from a retained free
coordinate. Core-disabled control removes the first two equations.

Nesting is the affine constraint filtration with direction-space dimensions
2,1,0 as context and evidence constraints are added. Field coefficients are not
real-valued strengths: no graded robustness or weak-coupling bound is claimed.

## R — graded higher-arity relation

Four bits `(I,K,O,S)`, evidence I=0,1, context S=t=0,1. Hard compatibility is
`I xor K xor O xor S = 0`. Preference cost is `K + g*(O xor I)` with
`g ∈ {1/4,1/2,1,2}`. Keep the current tuple if it is a global minimum; otherwise
choose the lexicographically first minimum. Enumerate 16 tuples explicitly:
this is a tiny exact solver control, not an efficient global representation.

Report the entire minimum set and output set before selection; a tie is not a
unique response. Withdrawal removes I=e while retaining the structural cost and
S=t. Test all 16 starts and all single-bit disturbances. Disable the core parity
relation as a matched control. Check whether reconstructing parity from its
three-coordinate projections loses information. Native approximation drops the
O/I cost term; additionally scale both costs by 1/1024 to test a weak-but-decisive
term. Layer nesting and structural learning remain unspecified.

## C — cellular cochain/Hodge configuration

Use three vertices, three oriented triangle edges, and one 2-cell. Coordinates
on edges are `(I,K,O)`; the context satellite supplies the 2-cell attachment.
Compare two cellular chain complexes with identical cell counts: a constant
attachment (`d2=0`, triangle with a 2-sphere attached at a vertex), and a disk
attachment (`d2=(1,1,1)^T`). Both satisfy d1*d2=0. Their graph skeletons agree.

The joint response operator is `L=d1^T*d1+g*d2*d2^T`, with
`g ∈ {0,1/4,1/2,1,2}`. Modes are harmonic cochains and positive eigenspaces.
Evidence clamps edge I to -1 or +1; K,O minimize `z^T L z/2` under that constraint.
Continuation updates free coordinates by gradient step 1/8, reclamping I.
Withdrawal releases all edges and uses the same step on L. Test ±1 disturbances
on each edge, including reconstruction of a disturbed clamped input interpretation.

Compute exact ranks, kernel dimension, equilibria, and limiting withdrawal
projection; compare with 128 explicit continuation steps and certificates.
The cell-skeleton filtration is topological nesting, not an abstraction hierarchy.
This is also a spectral model: the families intentionally overlap. The context
intervention here changes attachment, not merely activation of a fixed satellite.
That difference must remain visible in the comparison.

This candidate is motivated by cellular Hodge operators, not by a claim that
signal processing is SRI. See the primary study [Representing Edge Flows on Graphs
via Sparse Cell Complexes](https://arxiv.org/abs/2309.01632); all tiny matrices and
claims used here are separately derived and tested.

## Evidence, cost, and stopping

Store a complete per-case certificate, failures/counterexamples, arithmetic
operation counters or enumerated candidates, precision metadata, and the profile
of pass/fail/unresolved/not-claimed outcomes. Record measured elapsed time and
Python peak allocation in a separate run receipt, never use unlike operation
units as a cross-family speed ranking. Exact fractions carry bit-growth costs.

Use independent closed forms or brute-force tiny oracles to verify each solver.
Retain the existing repository regression suite. Stop at these declared bounds.
Add another structure only in a separately documented follow-up motivated by a
specific failure; do not retune this grid or replace failures with new witnesses.
