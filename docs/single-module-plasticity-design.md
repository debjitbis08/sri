# Current design: one circular module with routed plasticity

**Design decision, 2026-10-24.** The next study remains a **single module**.
Multiple modules, inter-module routing, satellites and a shared core are
deferred unless single-module evidence later motivates them. The broader
research program is not replaced, but those features are not prerequisites
or claims of this study.

This note follows the [layered analytical study](layered-reuse-analysis.md)
and specifies the next candidate in the
[routed-plasticity contract](single-module-plasticity-v1-contract.md).
The analytical study and its receipts remain unchanged: its 81-unit centre
comparisons are historical analytical witnesses, not the final experiment
architecture.

## Module and interface

Use four distinct rings of 32,24,16,8 scalar units: 80 activity coordinates.
The innermost ring is the interior; **omit the scalar centre**. The analysis
found no equilibrium role for its summary version, while its original spring
attenuated useful directions. This omission is an explicit revision, not a
claim that the original disk already had the new properties.

Immutable external input drives the outer ring. The output is its settled
32-dimensional state. Neighboring rings interact reciprocally through the
same learned local reconstruction maps. Small fixed lateral ring couplings
join neighboring sectors, so the four sectors are not independent modules.
There are no separate global Input/Output modules in this experiment.

```mermaid
flowchart LR
    H["External observation"] --> O["Outer ring: 32 units\nsettled output"]
    O <-->|"local reciprocal maps"| R1["24 units"]
    R1 <-->|"local reciprocal maps"| R2["16 units"]
    R2 <-->|"local reciprocal maps"| R3["Inner ring: 8 units"]
```

## What routing and learning mean here

Routing is bounded soft modulation of **within-module sectors and radial
couplings**, not selecting among modules. A provisional equilibrium produces
local compatibility scores. A second, fixed-route energy solve yields the
output. Well-matched sectors receive stronger reciprocal coupling and a
larger share of their ring's plasticity budget. All sectors have a positive
floor; poorly matched evidence is never completely blocked.

The same gates influence inference and learning, but they are not themselves
learned parameters. Structural changes minimize paired-view prediction loss
through equilibrium inference; lowering the inference energy by weakening
connections is not the learning objective. Routes are held fixed within each
loss/gradient calculation and recomputed for the next update. This is an
alternating procedure, not a proof of descent of one global joint energy.

Outer connections have a higher baseline learning rate than inner ones in
the candidate. Activity relaxation is not slowed inward. Familiarity grants
participation; prediction error supplies the correction. A well-explained
example can therefore produce little learning despite a high route weight.
Observation noise can also create error, so usefulness of these gates remains
a hypothesis rather than an assumed confidence estimate.

Uniform and reversed depth profiles, spatially averaged and deliberately
misassigned gates, and a frozen layered model are controls. A Schur-complement
implementation checks direct equivalence of each fixed-route inference solve.
No advantage of layers over all flat models, biological locality of the
learning gradient, emergent abstraction, or necessity of iterative settling
can be inferred from this experiment.

## Evidence and open questions

The reuse experiment supports retaining a learned grouping while adapting
weights under its generator. The analytical study establishes compatibility
and capacity conditions for shrinking rings. Neither establishes that this
new learning rule discovers useful inner representations or preserves old
ones. The contract measures adaptation, source retention and return to a
previous distribution, with input-dependent risk evaluated on held-out draws.

This first study keeps the acquired support pattern fixed, learns maps and
the inner retained direction, and tests ring-specific plasticity. It does not
learn topology or detect episode boundaries. A negative outcome narrows this
specific design; it does not justify silently returning to a single projector
or moving directly to multiple modules.
