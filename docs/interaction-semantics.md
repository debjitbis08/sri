# Interaction semantics before OP-04

**Status:** mathematical comparison; no architecture or update algebra selected.
This note supersedes the signed-threshold OP-04 proposal. It authorizes neither
implementation nor a new search. The examples below are logical illustrations,
not proposed SRI state alphabets or experiments.

**Follow-up:** [Representation and mediation evaluation](collective-interaction-evaluation.md)
compares descriptions of a joint mechanism, gives an interface-sufficiency
criterion, and specifies what would establish selective core mediation.

## What is established, and what is open

The [research program, §§2–3](research-program.md) describes mutually coupled
structured modules, whole-module responses, and layers as families of response
modes. Perturbation and observation interfaces do not impose a processing path.
External evidence is distinct from its revisable interpretation. Structural
resonance need not oscillate; neither a fixed point nor any particular algebra
is prescribed. Stronger permitted interior coupling is a hypothesis about
coupling profiles, not a theorem about actual influence.

The [foundations](foundations.md) prove convergence and monotonicity only for
the reachability control. They establish no compatibility, selection, recovery,
topological, or learning law for SRI. OP-02 through OP-03B remain valid restricted
controls, including the bounded impossibility result and scoped `D+R=5` theorem.
Their schedules and state coordinates need not become architectural commitments.

The current architectural clarification assigns external evidence to a specialized
input module and observation to a different output module. Both participate in
the collective configuration. Restricting initial inter-module coupling to
interior modes is a possible experimental restriction; the foundations permit
broader graded layer coupling. Neither choice supplies interaction semantics.

## Three distinct mathematical questions

Let `Theta` denote persistent structure and `C_i(Theta)` the possible whole-module
configurations, specified implicitly if necessary. A tuple of currently realized
configurations does not require constructing the space of all possible tuples.
When useful, write that semantic space as `C(Theta) = product_i C_i(Theta)`;
neither an explicit product table nor independent module behavior is assumed.
A layer descriptor
`p_i,l : C_i -> M_i,l` exposes a mode aspect of a whole configuration. Such
descriptors need not be independent coordinates or determine that configuration.

1. **Admissibility:** which `c ∈ C(Theta)` can coexist? A candidate defines a set
   `A_Theta`, or a graded assessment of configurations. Empty, singleton, and
   multiple-response cases must be distinguished.
2. **Realization:** how does a perturbation `x`, starting activity, and possibly
   history select or modify a realized response? Evidence may restrict input
   configurations or bias their assessment; that is itself a modeling choice.
   A selection relation, trajectory law, or distribution must be specified
   separately. Output is a declared observation of the realized configuration
   or behavior at the output module, not necessarily an input-to-output map.
3. **Learning:** how does experience change `Theta`, the spaces `C_i`, and their
   couplings? If configuration spaces change, preservation requires a declared
   correspondence or observable equivalence across old and new spaces. Editing
   persistent structure is distinct from revising transient interpretation.

Existence of an admissible configuration does not establish that a realizable
procedure reaches it. Nor does successful inference supply a learning rule.

## Aspects of a possible collective interaction

| Aspect | Meaning of collective compatibility | Consequences and unresolved choices |
| --- | --- | --- |
| Hard relations | For each module subset `H`, require `c_H ∈ R_H ⊆ product_{i∈H} C_i`; admissibility is the intersection of these requirements. | Directly expresses forbidden coexistence and higher-arity dependence. May admit none or many configurations; supplies no preference, response trajectory, or recovery mechanism. |
| Graded compatibility | Each subset has an assessment `phi_H(c_H)`; a separately chosen combination and comparison rule evaluates collective configurations. | Can express partial compatibility and layer-dependent importance. Costs, likelihoods, and ordinal preferences have different meanings; a number alone specifies none of them. Optimization or sampling adds further assumptions. |
| Compositional interaction | Whole-module configurations participate in a declared partial composition, possibly producing a joint configuration or behavior not determined by their unchanged coexistence. | Can express transformation as well as exclusion. Must define composability, order dependence, associativity or bracketing, and what is retained. A binary operation must not silently exclude irreducible collective interactions. |
| Local-to-global agreement | Module configurations restrict to overlap data; compatible restrictions must extend to a collective configuration. | Gives a precise interpretation of shared structure. Requires actual overlap and restriction maps, consistency on multiple overlaps, and an extension condition. Ordinary equality of labels does not establish a structural overlap. |

These can be aspects of one mechanism, not four competing mechanisms. Hard
compatibility can restrict outcomes of a joint transformation, graded compatibility
can distinguish those outcomes, and overlap agreement can check their collective
consistency. Neither constraint satisfaction nor optimization is selected as the
foundation. The comparison identifies obligations, not separate required engines.
Soft-constraint frameworks illustrate why combination and comparison operations
must be specified independently of numeric weights ([Bistarelli, Montanari and
Rossi, 1995](https://www.ijcai.org/Proceedings/95-1/Papers/082.pdf)).

**Pairwise relations can lose information.** Consider the illustrative relation
`R = {000,011,101,110}` on three binary descriptors. Every projection onto two
descriptors contains all four pairs. Consequently, intersecting those pairwise
projections admits all eight triples, including `111`, which `R` excludes.
No conjunction of binary relations on these same three variables can recover
`R`: each binary relation would have to admit its full pairwise projection.
Auxiliary variables can encode the missing constraint, but introduce additional
representation and computational costs. This is a reason to allow higher arity,
not a recommendation to adopt parity as an SRI algebra.

Local-to-global methods provide another language for extension obstructions
([Abramsky and Brandenburger, 2011](https://arxiv.org/abs/1102.0264)). Their physical
contextuality results are not SRI results, and their use of “contextuality” is not
the repository's satellite-influence test. No physics is imported here.

## Joint transformation and the core-mediation hypothesis

The schematic operation

\[
(c_K,c_A,c_B)\overset{\mathcal C}{\longmapsto}(c'_K,c'_A,c'_B)
\]

acts on participating configurations together. Each resulting component may
depend on the entire participating tuple; the notation does not mean independent
maps `c_i -> c'_i`. It also does not require a deterministic or unique outcome.
An implicit relation `J_H(c_H,c'_H; b)` could describe such an interaction over
participants `H`, with declared overlap/context data `b`. This is an interface
for evaluating semantics, not a new update law. Hard admissibility, graded
assessment, and overlap consistency may all constrain the same transition.

Activating another satellite can change this joint interaction and the responses
available to its existing participants, rather than merely conjoin another
static restriction. Distinguish **currently available responses**, conditional
on the realized collective configuration, from **structurally representable
configurations** under `Theta`. The former can change during inference without
learning. Persistent changes to spaces, generators, or coupling laws belong to
structural learning. If transient structure itself changes, its lifetime and
its separation from persistent `Theta` must be declared.

**Hypothesis, not an established consequence of the foundations:** the core is
a dynamically evolving structural mediator enabling selective collective
compatibility among specialized satellite responses, including response
combinations that cannot coexist without its participation. Abstraction would
then be acquisition of reusable compatibility-mediating structure, rather than
similarity extraction alone. Evolution of the core's transient configuration
and learning of its persistent structure remain distinct questions.

There is an important limit to a static interpretation. If adding the core only
adds constraints to an unchanged satellite relation `R_AB`, then

\[
\{(a,b):\exists k,\ R_{AB}(a,b)\land R_{KAB}(k,a,b)\}
\subseteq R_{AB}.
\]

Such a core cannot enable a previously forbidden pair. Mediation therefore needs
an explicit account of what changes: the applicable compatibility relation,
the participants' realized configurations, or the jointly reachable behaviors.
Adding a latent core variable alone is not an explanation. Conversely, a joint
transformation into a previously admissible but unreachable response demonstrates
changed reachability, not expansion of the admissible set.

To evaluate the hypothesis, hold evidence, satellite structure, and response
equivalence fixed, and define what disabling core mediation means. Compare
collective responses with and without the mediating interaction; do not simply
remove all constraints and call the unconstrained product the core-free system.
Require selectivity: enabling designated combinations while retaining specified
incompatibilities. For learned mediation, additionally require preservation and
held-out reuse of the acquired structure against fixed, memorizing, and matched
nonmediating controls. Neither all-purpose permissiveness nor a table of observed
combinations establishes abstraction.

## Graded layers and topology

For relational semantics, coupling could constrain tuples of layer descriptors
`(p_i,l(c_i))`; nested relations could represent increasingly restrictive
compatibility. Stronger restriction shrinks admissibility, but need not mean
stronger causal influence or better recovery. Under graded semantics, a layer
profile could bound an assessment's contribution, provided its scale and
combination rule are fixed. Under composition, strength could govern permitted
transformations or participation. These meanings are alternatives, not equivalent
implementations of a single scalar edge weight.

Interior modes may have larger permitted influence while actual effects depend
on the configurations present. Omitting weak contributions needs an error bound
for the selected observable; weak terms can decide between otherwise tied
responses. Neither depth nor coupling strength defines an abstraction level.

Separate topology **inside a module** from the incidence pattern of module
relationships. A hyperedge records a higher-arity relation; it is not by itself
a filled topological cell. To give holes a semantic role, configurations and
compatibility must depend on incidence, boundary, overlap, or extension data
that distinguish the relevant structures. Reusing the same pairwise equations
on an unchanged graph cannot detect whether an unrepresented face was filled.

For a simple consistency illustration, binary descriptors constrained by
`a=b`, `b=c`, and `c≠a` have individually satisfiable pair relations but no global
assignment. This establishes a collective consistency obstruction, not that
holes cause useful responses. A topology claim still needs matched structural
budgets and declared rules explaining how incidence affects admissibility or
realization. A drawing or a Betti number is insufficient.

## Perturbation, other satellites, and recovery

For the static hard-relation aspect alone, with evidence restriction `E_x`, possible observations
would be `Y_Theta(x) = {O(c_out) : c ∈ A_Theta ∩ E_x}`. This is an illustrative
set-valued semantics, not yet a response algorithm. Compare the same evidence
with a satellite coupled versus its compatibility conditions disabled, retaining
the other spaces and conditions. A changed `Y_Theta(x)` demonstrates a change
in possible responses. An unchanged set with different realized behavior would
instead locate the effect in selection. Under hard conjunction, removing
conditions can only enlarge admissibility; a narrowed response set can therefore
be a meaningful collective effect without creating a new admissible output.
This comparison is insufficient for the stronger mediation hypothesis above:
there the participation of a satellite may alter joint transformations themselves.

After evidence withdrawal, the admissible set alone cannot specify which
previous response persists. That requires an explicit continuation law and
transient state. Likewise, restoring compatibility after a disturbance is weaker
than recovering the original response: a repair procedure might settle on a
different legitimate configuration. Define response equivalence, permitted
disturbances, and selection/continuation assumptions before claiming recovery.
Collective resonance may be compatible organization or sustained behavior;
the term must not silently turn either into an oscillation requirement.

## CPU implications and next decision

Defining a joint interaction need not construct a global configuration table.
Possible representations include symbolic configuration generators, predicates
on structured tuples, and bounded-support joint transformations checked against
overlap data. Their semantics must say which outcomes they represent and how
applicable interactions are found. They may be combined; none is selected here.
Overlapping transformations also need consistency and ordering semantics: local
validity alone does not prove global extendability, convergence, or recovery.

An explicit joint table has `product_i |C_i|` entries. A relation of arity `r`
over `q` configurations per module can itself need `q^r` entries. Structured
predicates, sparse allowed tuples, bitsets, and indexed restrictions can reduce
particular costs; they do not automatically avoid global search. In particular,
pairwise filtering cannot detect the parity restriction after it has been lost
by projection. Graded optimization adds comparison and aggregation work;
composition adds compatibility and representation costs. Global extension may
remain unresolved even after inexpensive local checks.

Compact specification is not an efficient-execution proof. Bounded interaction
arity does not bound the number of applications or the cost of finding a globally
consistent response. Any CPU proposal must bound representation size, candidate
generation, relation-evaluation cost, intermediate configurations, and selection
work, including irregular memory
access. Algorithmic messages used to solve constraints need not imply signal
propagation as the architecture's semantics. Conversely, describing a system
as collective does not make computation free or guarantee efficient locality.

The next decision is semantic: compare implicit descriptions of whole-module
configurations and collective transformations; specify how compatibility,
grading, and overlap agreement constrain the same interaction; define a fair
core-mediation comparison; and identify a plausible bounded realization procedure.
Static compatibility remains a useful aspect and control, not the chosen
foundation. Only then write
a replacement OP-04 contract with perturbation, observation, recovery, and
satellite controls appropriate to that choice. Structural learning remains a
separate obligation. No candidate in this note has been selected as SRI.
