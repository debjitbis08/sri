# Stage B decisions for the implementation agent

Status: prospective design decisions, 2026-10-25. These resolve the open choices
in the [proposal](relational-capacity-and-composition-study-proposal.md) and
[handoff audit](relational-capacity-handoff-audits.md). Integrate them into one
execution contract and freeze it before implementing/running new fits. This is
not a claim that capacity has been established. Preserve historical artifacts.

## 1. Scope and arm ledger

Use the existing 72-scene task, both existing renderings and unchanged witness
criterion: all-scene normalized hidden error <=0.10 and every joint orientation
correct, followed by independent serialized-parameter verification. All-scene
fitting is privileged capacity search, not transfer learning.

Run only these configurations unless a valid prospective impossibility bound
excludes them:

| Name | Modules | Retention | Evidence | Output | Coupling |
| --- | --- | --- | --- | --- | --- |
| S2 | 1 | rank two | alpha=0.1 | own outer hidden state | none |
| P0 | 2 | rank one each | alpha/2=0.05 each | arithmetic mean of outer hidden states | none |
| D1(mu) | 2 | rank one each | input module only, alpha=0.1 | output module's outer hidden state | terminal identity, mu in {0.001,0.01,0.1} |

P0 is the previously discussed summed-output control, with a precise fixed
normalization of 1/2. No unnormalized sum, learned gain, concatenation, projection,
or tuned readout. It controls for distributed coordinates and parallel capacity;
it is not a causal no-communication counterpart to D1 because the interfaces
and evidence placement differ. Do not credit a D1/P0 difference solely to
communication. A communicating mean-output comparison is deferred rather than
silently added. Fixed-route D1 evaluations and disconnected-output analysis
provide limited mechanism controls, not a full communication-benefit experiment.

S1 is already excluded. Fixed-route compositions are analytically excluded as
successful capacity arms. R and G are deferred: there is no specified relational
selection mechanism. Neither R/G nor coupled rank-two modules enter this budget.
Do not select their scaffold based on these capacity results.

## 2. Routing: one collective pilot, local residual gates

For D1, construct one 160-state coupled pilot with all local routes equal to
one. Use the declared mu in both passes, input-only masked anchoring, and zero
direct evidence at the output module. Solve the collective pilot once for all
scene right-hand sides. Each module then computes its own local residual gates
from its own 80-state slice of that same collective pilot, using its own maps
and terminal vector. Output gates are neither copied from input nor computed
from an isolated zero-drive pilot.

For each radial residual use the existing normalized squared residual and
`g=0.1+0.9 exp(-r/0.25)`, `a=0.25+0.75g`, epsilon=1e-12. Use the existing
terminal residual gate for rank one. Assemble the final collective Hessian with
both modules' local routes and unchanged mu. Solve once; no iterative rerouting,
coupling gate, additional router, or extra pass. Full differentiation through
pilot, gates and final solve is permitted for this capacity optimizer; it is
not a proposed local plasticity law.

For S2 omit terminal residual blocks, terminal vectors, and terminal gates:
Q=0 and retaining redundant parameters would add no capacity. Its 12 radial
gates remain adaptive. For P0 both independent modules use their own anchored
pilot and gates under alpha/2. Block-diagonal solving is allowed and must have
the equivalent dense reference; do not deliberately waste compute for parity.

All arms retain the original scalar-coordinate lateral cycles, lambda=0.001,
radial support, Stiefel constraints and actual state outputs. No geometric
vector-cycle replacement occurs in this stage.

## 3. Port convention and coupling coefficient

For D1, P selects ring 3 in existing sector-major, coordinate-major order.
Use `mu/2 ||P x_input - P x_output||^2`: identical sector numbers, both terminal
coordinates, no permutation, no cross-sector edges, no learned transport.
T=I is the fixed numerical coordinate convention. Do not align trained maps to
the renderer or rotate them after fitting to obtain a better correspondence.

This is not an exact gauge symmetry of the full current energy: sector-local
basis changes alter the terminal scalar-cycle penalty unless that operator also
transforms. The audit's near-gauge observation is useful, but it does not justify
silently canonicalizing parameters with the lateral operator held fixed. Do not
create separate trained 'aligned' and 'misaligned' arms by a fixed sector-local
rotation. Such rotations may be frozen-parameter interventions in later work.

Declare mu={0.001,0.01,0.1} as a search grid, not an analytically optimal choice.
It spans the lateral coefficient through the data-anchor coefficient and lies
above the current weak-coupling exclusion threshold. Crossing that threshold
does not establish capacity. All three values and their rejected candidates
count toward selection cost. No grid extension or learned mu after failure.

The anchor energy at zero is 0.1 for S2, P0 and D1 on every task scene:
visible norm squared is 2, and the total coefficient is 0.1. P0 observes the
same example twice; report two local observation uses but only one independent
example and one scene exposure through the whole system. Matching E(0) does
not equate the statistical or causal observation interfaces.

## 4. Initialization and seeds

Root seed 2026102506. Use NumPy SeedSequence([root, world, start, module]),
world in {0,1}, start in {0,1}, module in {0,1}. Renderer seeds remain exactly
those in masked-capacity v1; these are known development fixtures.

For each stream generate Gaussian radial blocks in ring-major, sector-major
order, shapes (8,6), (6,4), (4,2), four blocks each. Apply reduced QR with
positive diagonal as in the existing sampler. Then draw four Gaussian
two-coordinate terminal vectors and normalize. Retain those vectors for rank-one
arms; discard them for S2. Use module 0's maps for S2 and both modules for P0/D1.
Thus overlapping initial parameters are paired across configurations. Do not
seed with mu, step size or arm name. No identity/signal-plane initialization,
symmetry breaking hand tuning, or reseeding. Save exact initial arrays and hashes.

All coupling/readout parameters are fixed constants, so there are no additional
learned parameters to initialize: S2 stores 320 floats; P0/D1 store 656. Count
constraint degrees of freedom separately if desired; these are storage counts.

## 5. Objective, optimization and candidate selection

Use the exact maximum of the 72 normalized hidden squared errors as the training
objective. At exact ties use the uniform average of active-scene gradients
(PyTorch amax semantics, independently checked). Do not use a changing softmax
temperature, mean-loss warmup, or objective schedule. Record mean loss as a
secondary metric. Nonsmooth optimization can still fail; failure is not a proof.

Use projected/retracted Adam as in the previous capacity search: float64 CPU,
one thread, betas=(0.9,0.999), epsilon=1e-8, no weight decay, global projected
gradient clip norm 10, QR positive-diagonal and unit-sphere retractions, ambient
moments retained without transport. For S2 omit terminal-vector operations.
Use constant step sizes {0.003,0.01}, two starts, at most 1,000 full-batch updates.
This is a prospective heuristic menu, not an optimum derived from a proof.

Checkpoints: {0,100,250,500,1000}. Select the smallest lexicographic tuple
(worst-scene error, mean error, update index), using actual float64 metrics,
without a numerical tie band that could hide crossing the pass threshold.
Across candidates use the same first two metrics followed by enumeration order.
Parameter verification uses independent NumPy and Schur inference after saving
12-significant-digit parameters, with constraints and solver tolerances from
masked-capacity v1. A verification mismatch is a correctness failure.

Enumeration: S2, P0, D1; world ascending; within D1, mu ascending; start ascending;
step size ascending. At a scheduled checkpoint satisfying the complete witness
criterion, verify immediately. On success stop the remaining candidates for
that arm/world (including remaining mu values for D1), mark them unexecuted,
and continue other arm/world groups. This searches for existence, not the best
mu. Do not claim an optimal coupling from first-witness stopping. If no witness
appears, finish the declared menu. No stagnation or low-sensitivity early stop.

At most (1+1+3)*2 worlds*2 starts*2 rates = 40 candidates, 40,000 updates,
2,880,000 scene exposures, 200 checkpoint evaluations, plus controls and
verification. Candidate zero-update checks count too. This keeps the original
update ceiling by deferring R/G, not by treating mu tuning as free. Wall/CPU,
memory, per-dimension solver counts and all audit/selection costs remain required.
This is an update budget, not an equal-compute comparison.

## 6. Pre-search exclusions versus reported screens

An arm or coefficient is excluded before fitting only by a completed theorem
or independently certified bound that covers every allowed parameter setting,
the actual observation/output interface, and all allowed routes. A universal
lower bound strictly greater than 0.10 suffices. Retain the proof and charge
its verification cost. An open conjecture or sampled observation does not
qualify. Do not spend an indefinite budget trying to close every open bound.

Apply the existing S1 and fixed-route exclusions. Apply the weak-coupling bound
to the declared D1 grid; it excludes none of the three values. The open
sector-aligned and rank-two bounds are reported as unresolved unless actually
completed before freezing. If a new proof becomes available mid-run, preserve
incurred work and amend the ledger transparently; do not rewrite the initial
protocol or silently reallocate skipped budget.

For the route-sensitivity screen, at every scheduled checkpoint compute a
fixed-route reference with all local routes one, same parameters, mu, evidence
and output interface. On the 24 q=1 scenes report

```
d = sqrt(mean ||f_routed - f_fixed||^2)
F = (8-2 sqrt(2))/21
tau = sqrt(2F)-sqrt(0.2).
```

Passing the capacity criterion requires d>=tau. Also report the parameter-
specific necessary threshold `max(0, sqrt(mean ||f_fixed-target||^2)-sqrt(0.2))`.
Use absolute output units, not changes relative to the model's own output norm.
Failing this screen at an initialization or checkpoint excludes that parameter
setting, not the arm or its future trajectory. Record it and continue. Passing
it is not sufficient. Only a uniform upper bound d<tau over all allowed
parameters can make it an arm-exclusion gate. A verified apparent witness that
violates the necessary inequality beyond tolerance indicates a correctness bug.

Numerical correctness gates (finite values, constraints, SPD solve, residuals,
independent-reference agreement, target leakage) are mandatory. They stop the
affected execution as an error, not as a scientific negative result. Freeze
tests, tolerances, objective-gradient checks away from and at exact ties, and
provenance in the integrated contract before running.
