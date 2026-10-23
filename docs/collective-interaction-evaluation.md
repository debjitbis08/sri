# Collective interaction: representation and mediation obligations

**Status:** semantic evaluation following the [comparison note](interaction-semantics.md).
No configuration algebra, transition rule, or experiment is selected here.
The signed-threshold OP-04 proposal remains withdrawn. This note gives a proposed
representation direction and conditions under which it would be justified.

**Worked follow-up:** the [role-binding example](mediation-example.md) implements
one illustrative mechanism under explicit user authorization. It does not select
the SRI architecture or replace the OP-04 contract.

**Direction update:** the [comparative modeling program](model-comparison.md)
now governs next steps. The representation direction discussed below is one
candidate to evaluate alongside other mode, nesting, and substrate choices.

## What a collective interaction must describe

Let `Theta` be persistent structure and `c_i` a realized structured configuration
of module `i`. A configuration may be described by a finite representation rather
than an index into an enumerated state set. Layers describe mode aspects of this
configuration. Evidence enters through the input interface; output is observed
at a different module. Neither interface determines an execution path.

For participating modules `H`, write an interaction schematically as

\[
J_{\Theta,H}(c_H,c'_H;b,x).
\]

It relates a joint before-configuration to a joint after-configuration, given
declared boundary/overlap context `b` and evidence `x`. Evidence may enter this
relation only through the designated input interface and its constraints; this
notation does not grant every module direct access to input. Other modules keep
their current configurations for this interaction. The choice of participants,
and whether their number is bounded, remains a modeling decision.

Hard validity, graded assessment, transformation, and overlap agreement are
properties of this same interaction. They need not be separate mechanisms.
Specifying `J` is still insufficient: discovering applicable interactions,
constructing outcomes, choosing among them, and continuing after a perturbation
are additional obligations. A joint transformation can be realized by sequential
CPU instructions without treating modules as independent signal processors.

Collective dependence must be established by interventions on participating
configurations. Merely writing a tuple-valued operation proves nothing: even a
deterministic joint function has coordinate projections. The relevant question
is whether changing one participant changes the other's allowed or realized
response under the declared controls, not whether the code returns a tuple.

## Three descriptions of that same mechanism

| Description | What must be represented | What it resolves | What it leaves unresolved |
| --- | --- | --- | --- |
| Implicit global relation | A predicate or symbolic description over complete before/after tuples. | Defines joint semantics without storing every tuple; retains arbitrary higher-arity dependence. | A cheap membership test need not supply a cheap way to find an outcome. Global context inspection and search may still dominate. |
| Supported joint transformations | Intensional rules describing participants, configuration-dependent applicability, joint changes, and affected overlaps. | Can construct candidate outcomes from current structure and charge work to inspected participants. | Bounded support can exclude essential interactions; overlapping transformations may conflict or depend on order. Candidate generation and completeness need proofs. |
| Overlap-based assembly | Local descriptions plus restriction maps and an extension/gluing condition. | Makes consistency across shared mode structure explicit; may avoid duplicating whole configurations. | Agreeing local descriptions may fail to extend. Descriptors may discard distinctions needed by later interaction. It does not itself define transformation. |

**Proposed direction for further specification:** use supported joint
transformations as an operational description, retain an implicit relation as
their semantic specification, and use overlap checks only where their adequacy
can be demonstrated. This is conditional, not a choice of final SRI architecture.
Do not force pairwise support, assume the core is a scalar bottleneck, or assume
local checks are globally complete. If these restrictions lose the required
collective behavior, reject them rather than enlarge a lookup table.

## When can a compact interface be exact?

Suppose a module exposes `p(c)` to an interacting context. For a fixed admissible
external context `b`, let `B(c,b)` denote the externally relevant interaction
behavior: allowed observable outcomes, and graded assessments if used. An exact
implementation using only `p(c)` and `b` is possible only if

\[
p(c)=p(d)\quad\Longrightarrow\quad B(c,b)=B(d,b)
\quad\text{for every declared }b.
\]

**Necessity:** identical interface data gives the implementation identical
arguments, so it cannot return different behavior. **Sufficiency at the semantic
level:** if this condition holds, define behavior on each descriptor by choosing
any representative; the result is well-defined. This gives no efficient algorithm
for constructing that behavior, nor any bound on the number of descriptors.

For repeated interaction, matching one-step outputs is insufficient. Resulting
configurations must also support matching continuations under future contexts,
or the interface must expose more information then. Whole-module semantics is
therefore compatible with compact interfaces only when the abstraction preserves
the behavior being claimed. Layer descriptors cannot be assumed sufficient just
because they are labeled “interior.”

This identifies a concrete rejection criterion before implementation: exhibit
two configurations with identical proposed descriptors but different required
collective behavior. Such a pair refutes that interface's exactness. Refinement
or adaptive inspection may repair it; either has a resource cost.

## What counts as selective core mediation?

Fix satellite structures, evidence, allowed initial conditions, and a common
observation/equivalence convention. Specify a **disabled-mediation** system by
disabling the core's mediating interactions while retaining the modules' own
validity requirements and explicitly defined unmediated interactions. A system
with all compatibility conditions removed is not this control.

Let `A_on` and `A_off` denote the sets of jointly admissible satellite observations
with mediation enabled and disabled. Let `R_on(x)` and `R_off(x)` instead denote
responses reachable under the respective realization semantics. Keep these
claims separate:

| Claim | Required witness |
| --- | --- |
| Compatibility mediation | A declared combination in `A_on` but not `A_off`, with an explanation of the changed compatibility semantics. |
| Realization mediation | A declared response in `R_on(x)` but not `R_off(x)`; it may already be admissible in both systems. |
| Selectivity | A separately declared, representable incompatible combination remains excluded with mediation enabled. Exclusion must follow from structure, not removal from the observation vocabulary. |
| Collective transformation | With evidence and pre-existing participants fixed, changing another participant changes an existing participant's response through the joint interaction. |

Membership witnesses are existential and do not establish reliable selection.
A later experiment must additionally specify whether the desired response is
required under every allowed continuation, under one controlled procedure, or
with a declared probability. Recovery likewise requires a continuation law and
a response-equivalence class, not just membership in `A_on`.

If participation transforms satellite configurations, compare their meanings
under the same observation convention; renaming a state cannot demonstrate newly
enabled coexistence. If enabling the core only conjoins restrictions to an
unchanged satellite relation, compatibility mediation is impossible by the
inclusion argument in the earlier note. A candidate must identify how it escapes
that assumption. Learned mediation further requires acquiring reusable structure,
preserving prior responses, and improving held-out combinations. None follows
from a hand-specified mediated response.

## Grading, topology, and execution

For this representation direction, layer-dependent grading would assess the
participation or outcomes of joint interactions. A subsequent specification must
choose the order/comparison semantics and how grades affect realization; weights
cannot silently become additive signal amplitudes. Stronger permitted interior
coupling does not imply stronger influence in every realized context.

Overlaps must refer to actual configuration structure. If topology is to matter,
its incidence or extension conditions must affect validity or joint interaction.
An interaction's participant hypergraph is not automatically the topology of a
module, and bounded-support transformations do not themselves detect holes.

Avoiding a global table is possible at the level of representation: store the
realized configurations, intensional interaction descriptions, and necessary
overlap data. To claim efficient execution, additionally account for

\[
W = W_{\mathrm{discover}} + \sum_{t=1}^{N}
\bigl(W_{\mathrm{inspect},t}+W_{\mathrm{generate},t}
+W_{\mathrm{validate},t}+W_{\mathrm{select},t}+W_{\mathrm{commit},t}\bigr).
\]

This is an accounting identity, not a complexity theorem. Include rejected
candidates and consistency checks in these terms. Bound candidate count `N`,
matching cost, intermediate storage, and continuation length before claiming a
useful budget. Bounded support alone bounds none of those globally. A capped
procedure may return unresolved rather than claim an admissible settled response.
Concurrent or overlapping changes require explicit consistency semantics; CPU
sequentialization does not license an arbitrary module update order.

## Decision and next deliverable

The semantic comparison supports investigating **implicitly represented joint
transformations with explicit compatibility and overlap obligations**. It does
not yet establish that this description realizes SRI, preserves topology, or
executes efficiently. Constraint satisfaction remains one descriptive aspect
and a control, not the selected foundation.

Before a replacement OP-04 experiment, produce one worked mathematical mediation
example with (i) structured module configurations and their descriptors,
(ii) an explicit joint interaction and matched disabled-mediation meaning,
(iii) an enabled combination and a retained incompatibility, and (iv) a resource
account that includes outcome construction. First assess interface sufficiency
and distinguish compatibility from reachability; only then select an update
algebra and experiment bounds. The [worked role-binding example](mediation-example.md)
now supplies this illustration, including its limitation: a generic matching
solver can reproduce the responses, so an essential core remains unproved.

OP-02 through OP-03B and their historical certificates remain unchanged.
