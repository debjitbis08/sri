# Graded cellular experiment v1 — bounded contract

**Status at declaration:** specified before running this experiment. This is an
in-session contract, not an external preregistration. It is a mechanism test,
not a replacement SRI architecture. Preserve all earlier controls and reports.

## Model and scope discount

Use the same triangle cochain z=(z0,z1,z2) as comparison v1, plus a scalar context
configuration b. Input perturbs only z0; output observes z2. The context has a
persistent preference c=-1 or +1, stored as a parameter of its local energy.
Changing c is a structural context intervention, not another external evidence
channel. This is one structured cochain and a scalar context surrogate: input,
core, and output are not yet separately structured full modules.

Write a=mean(z), kappa=2, and define

`E(z,b) = z^T L z/2 + beta*(a^2-1)^2
          + gamma*(a-b)^2/2 + kappa*(b-c)^2/2`.

The cellular operator is `L=3*Identity-J+f*J` when enabled, where J is the
all-ones matrix. A constant-attached 2-cell has f=0; disk attachment supplies
f>0. Both retain the same three vertices, three edges, and one 2-cell. When the
cellular term is disabled L=0 but all coordinates and readouts remain.

This energy combines cellular agreement and graded compatibility of whole-mode
descriptors a,b. The two preferred amplitudes are hand-designed. The gamma
parameter grades this single coupling; it is **not** graded layer connectivity.

| Component | Treatment and permitted credit |
| --- | --- |
| Cellular cochain, attachment, polynomial preference, context coupling | Implemented; only scoped mechanism evidence. |
| Multiple layers per module; graded inter/intra-layer connections | Not implemented, zero evidence; their omission is not a falsification. |
| Structured input/output/core/satellite modules and selective core mediation | Not implemented; distinct observation/perturbation coordinates are proxies. |
| Learning, acquired modes, STM, dreaming, abstraction and transfer | Not implemented; the preferred modes and context preferences are supplied. |
| Higher-arity inter-module relations and autonomous satellite activation | Not implemented; mean-based coupling is one chosen descriptor. |
| All-fair discrete schedules, continuing noise, global CPU scaling | Not tested. |

Do not award an architectural pass from a successful proxy, or reject an omitted
mechanism from a proxy failure. Report mechanisms separately from this ledger.

## Matched controls and bounds

Six variants, all with four transient coordinates and the same output:

1. Cellular only: enabled L with f=0, beta=0.
2. Graded relation only: L=0, beta=1.
3. Combined: enabled L with f=0, beta=1.
4–6. Changed attachment: enabled L with f=1/4,1/2,1 respectively, beta=1.

For each use gamma in {0,1/4,1,2,4} and c in {-1,+1}: 60 configurations.
Removing the nonlinear preference or cellular term changes the law intentionally;
holding coordinates and readout fixed isolates its contribution. No parameter
search or extra layers are added to rescue a failure.

## Response, basins, and recovery

Use continuous gradient descent of E. Computation uses RK4 with step 1/16 and
explicit residual checks, not a claimed exact integrator. After withdrawal,
the contrast q=z-a*ones obeys q'=-3q when cellular and q'=0 otherwise. Integrate
the remaining two-dimensional (a,b) system and evolve q analytically. Validate
this reduction against independent full-coordinate integration.

Pulse evidence e=-1,+1 by clamping z0=e for 16 time units, starting z1=z2=0 and
b=c. Then release z0 and continue for 128 time units. Test the two pulses under
the same context/structure; report whether their output responses remain distinct.
Do not credit an optimizer that installs a preferred mode instead of this pulse.

Find stationary means from the cubic obtained by eliminating b. Classify stable,
unstable, marginal, and flat directions, retaining degeneracies. Analytic local
stability does not establish a complete basin. Measure basins on all 81 starts
z0,z1,z2,b in {-1,0,1}; cache only identical reduced (a,b) trajectories, accounting
for contrast separately. These are sample counts, not continuous basin volumes.

Disturb every isolated stable mode by ±1/4, ±1, ±4 on each coordinate; also test
representative means ±1 for the flat cellular control and zero-contrast minima
for the relation-only control. Continue 128 units and compare the **output**,
full configuration, and stationary mode separately. A contrast left behind in
the relation-only control must not count as restored output.

Numerical convergence requires full gradient infinity-norm <=1e-7. Recovery
requires convergence plus output error <=1e-5; full-state recovery is separate.
Unconverged runs are unresolved, not successes or impossibility proofs. Repeat
both pulse experiments with step 1/32 to check discretization sensitivity.
Also check whether nonzero cochain contrast can be stationary: this is stability
selectivity, not a hard admissibility exclusion imposed on all configurations.

## Approximation, limits, and reporting

Test a coupled-versus-uncoupled approximation on the pulse outcomes. Additionally
inspect stationary branches at gamma=2±1/64 and 2±1/1024 for the combined variant
and c=+1. Compare face thresholds f=4/9±1/1024 at gamma=0. These are declared
local sensitivity probes, not grid retuning; stationary-root changes do not by
themselves certify finite-time switching.

Record numeric tolerances, root residuals, convergence status, energy changes,
sample counts, evaluations, runtime and memory. Separate analytic identities,
local-stability conclusions, and numerical trajectories. Check a gradient by an
independent finite difference, the reduced flow against full RK4, and stationary
roots against known special cases. Run historical regressions unchanged.

The result must answer whether the combination repairs the two earlier controls'
failures **in this proxy**, whether strong coupling or attachment destroys modes,
and whether any benefit actually needs topology. Stop at the declared bounds.
