# Graded cellular v1 — recoverable modes in a limited proxy

**Result:** combining cellular agreement with a nonlinear graded preference
repairs both controls' recovery failures in part of the declared parameter range.
The conclusion is about this four-coordinate mechanism, not a multilayer SRI
architecture. Preferred modes are hand-designed, and a selective core is absent.

The [contract](graded-cellular-v1-contract.md) fixed six variants, five couplings,
two context preferences, input pulses, disturbance sizes, and numerical budgets.
The [certificate](../reports/graded-cellular-v1.json) contains all 60 configurations,
4,860 basin samples, 1,872 disturbance trials, and 120 pulse experiments with
step-halving checks. [Runtime data](../reports/graded-cellular-v1-runtime.json)
are recorded separately. All earlier reports remain unchanged.

## What the combination achieved

For the combined constant-attachment model, gamma=0,1/4,1 each supports two
isolated stable modes for either context preference. Opposite input pulses reach
different modes after the input is removed. Every tested single-coordinate
disturbance of magnitude 1/4 or 1 returns to its original output and full state.
That is 96 successful trials at each magnitude across these six configurations.
It is not a proof for every disturbance in a ball or every update schedule.

For context preference c=+1, the post-input observations are:

| Coupling gamma | Negative pulse | Positive pulse | Interpretation |
| --- | --- | --- | --- |
| 0 | -1 | +1 | Two persistent, locally restoring modes |
| 1/4 | approximately -0.940959 | +1 | Context shifts one mode while retaining distinction |
| 1 | approximately -0.788675 | +1 | Stronger contextual shift; both modes still recover |
| 2 | approximately -0.504135 at the time cap | +1 | Negative trajectory unresolved near a marginal stationary point |
| 4 | +1 | +1 | Input distinction collapses |

Changing c to -1 gives the sign-reflected results. At gamma=1/4 and 1, this
context intervention changes amplitude rather than switching the sign selected
by a pulse. At gamma=4, context determines the sign and overwhelms input history.
This is sensitivity to a prescribed context preference, not learned contextual
interpretation or autonomous activation of a structured satellite.

Large disturbances can switch modes. In the two-mode combined cases, magnitude
4 changes the recovered output in 36 of 96 trials. Stable modes do not imply
unlimited error correction, and switching is not silently scored as recovery.

## What the controls establish

| Variant | Isolated two-mode configurations / 10 | Pulse results: distinct / collapsed / unresolved | Magnitude-1/4 trials: restored / changed / unresolved |
| --- | --- | --- | --- |
| Cellular only | 0 | 2 / 6 / 2 | 60 / 24 / 12 |
| Graded relation only | 0 | 10 / 0 / 0 | 32 / 96 / 0 |
| Combined, constant attachment | 6 | 6 / 2 / 2 | 128 / 0 / 0 |
| Combined, disk weight 1/4 | 4 | 4 / 6 / 0 | 112 / 0 / 0 |
| Combined, disk weight 1/2 | 0 | 0 / 10 / 0 | 80 / 0 / 0 |
| Combined, disk weight 1 | 0 | 0 / 10 / 0 | 80 / 0 / 0 |

Trial counts differ because each existing stable mode is tested. A single mode
can recover perfectly while conveying no input distinction; therefore recovery
percentages alone are not a ranking. Flat controls use the declared representative
means, not an enumeration of their continuum of stationary configurations.

The two controls fail for different reasons. Cellular-only dynamics can restore
agreement but, when uncoupled from context, preserve a disturbed mean. The
nonlinear relation can restore a preferred mean but leaves cochain contrasts
unchanged. Thus relation-only output distinctions can persist in unprotected
contrasts even after both pulses select the same mean. The combination restores
both mean and contrast near its isolated minima.

## Mathematical explanation

Let a=mean(z), q=z-a*ones, and b be the dynamic context variable. The released
energy separates as

`E = 3*||q||^2/2 + 9*f*a^2/2 + beta*(a^2-1)^2
     + gamma*(a-b)^2/2 + 2*(b-c)^2/2`

when cellular coupling is enabled. Without that term, q is neutral. Gradient
flow gives q'=-3q in the cellular case and q'=0 in the relation-only control.
The remaining two-dimensional flow is integrated explicitly, not replaced by a
global minimum selector. Input pulses act only at the input coordinate.

At a stationary point,

`b=(gamma*a+2*c)/(gamma+2)`,

and a satisfies

`4*beta*a^3 + (9*f+lambda-4*beta)*a - lambda*c = 0`,
where `lambda=2*gamma/(gamma+2)`.

For the combined f=0, beta=1, c=+1 case, the cubic factors as

`(a-1)*(4*a^2+4*a+lambda)=0`.

The second minimum exists for lambda<1, equivalently gamma<2. At gamma=2 it
meets a saddle at a=-1/2; above that only the context-aligned minimum remains.
The Hessian is positive definite at the isolated minima. Continuity gives a
neighborhood with a positive lower curvature bound; within a sufficiently small
ball the gradient flow contracts distance to the minimum. This proves local
restoration, not the finite sizes of the sampled basins.

With gamma=0 and disk attachment, nonzero minima require `f<4/9`. Hence a filled
complex can still support two nonlinear modes at f=1/4, while f=1/2 and 1 erase
them. These modes need not be harmonic cochains: the nonlinear term balances the
cellular penalty. The fixed attachment alone does not determine their number.

## Topology and approximation: limited credit

Changing attachment and its weight affects the modes under matched cell counts.
But the successful f=0 cellular term also equals ordinary pairwise agreement:

`z^T(3*Identity-J)z/2 = ((z0-z1)^2+(z1-z2)^2+(z2-z0)^2)/2`.

Consequently this result does not establish that holes, cell complexes, or
topological nesting are necessary for recoverability. A flat agreement penalty
with the same nonlinear preference reproduces that energy. Topological structure
is a representation here, with an observable attachment effect, not an established
advantage over every alternative.

The declared probes gamma=2±1/64 and 2±1/1024 change the number of isolated minima
from two to one. The disappearing branch approaches -1/2, while the surviving
one is +1. Small coupling changes near this boundary can therefore remove a
response with a substantially different value. These are stationary-branch
results, not a finite-time switching guarantee. The face probes similarly locate
a change around f=4/9. No uniform weak-coupling approximation bound follows.

## Numerical evidence and unresolved cases

RK4 uses step 1/16; the contrast component is evolved analytically. Independent
full-coordinate RK4 checks the reduction, and finite differences check the
gradient. Step halving for all pulse experiments changes final coordinates by
at most approximately 3.21e-12 in this run. The largest recorded increase in
the reduced released energy is approximately 6.66e-16, consistent with roundoff.
These diagnostics support the calculations; they are not interval-certified
trajectory or basin proofs.

There are 232 unresolved basin samples and 44 unresolved disturbance trials at
the fixed time budget, concentrated in slow or marginal regimes. They retain
their unknown status. Exact saddle starts are also recorded and do not count as
stable attractors. The 81-point grid measures sample frequencies, not continuous
basin volumes or robustness to arbitrary initial states.

## Scope discount and research meaning

| Requirement | Grade from this experiment |
| --- | --- |
| Separated post-input modes with local restoration | Supported analytically and by scoped numerical trials |
| Recovery under the declared small disturbances | Supported for specified configurations; larger disturbances expose basin switches |
| Contextual influence without immediate distinction loss | Supported at two positive coupling values, for a scalar preference surrogate |
| Graded layer connections and multiple layers per module | **Untested: not implemented** |
| Structured input/output/core/satellite modules | **Untested: only coordinate interfaces and one scalar context** |
| Autonomous selective core mediation | **Untested: no specialized core mechanism** |
| Learned modes, STM, dreaming, abstraction, transfer | **Untested: preferred modes and context preferences are hand-designed** |
| Topological necessity or computational advantage | Not established; an equivalent flat agreement energy exists |
| General approximation, all-fair schedules, continuing noise | Not established |

Omissions receive neither success credit nor failure penalties. This is a positive
mechanism result within a narrow proxy; the architecture grade remains
`not_evaluated`. It motivates testing whether restoration and contextual influence
survive richer, explicitly specified module and layer structures. It does not
choose those structures or demonstrate the eventual learning substrate.

Reproduce with `python -m experiments.graded_cellular` and
`python -m unittest discover -s tests -v`. Numerical certificates carry source
hashes and tolerances; runtime and a representative memory sample are separate.
The sample memory measurement excludes full-report storage and is not a CPU
scalability claim.
