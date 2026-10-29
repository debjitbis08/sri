# Continuous learning and dedicated consolidation: reading notes

2026-10-27. Living notes, not a contract or an architecture selection.
Evidence tags follow [prior work](prior-work.md). Published findings,
implementation costs and SRI hypotheses are separate below.

## Direction from the research discussion

Study the strongest relevant solutions to our problem, understand why they
work, then identify improvements that advance the [SRI program](research-program.md).
Neither defending an existing design nor finding an unoccupied literature
topic is the objective. Known components can discharge program obligations
without becoming claims of original invention.

Learning must support continuous acquisition and dedicated phases on ordinary
consumer CPUs. Dedicated phases also need explicit resource budgets; they
must not hide an eventual dependence on corporate hardware. This is a design
requirement, not a demonstrated capability of any candidate below.

## Representation reuse can cause both transfer and interference

Source: Holton et al., [Humans and neural networks show similar patterns of
transfer and interference during continual learning](https://www.nature.com/articles/s41562-025-02318-y),
published online 2025, issue 2026. [Open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12846915/).
Read: main results, discussion and methods; supplementary analyses not audited.

**Shown [read].** In sequential A–B–A rule tasks, related but different rules
encouraged transfer alongside interference. Human “lumpers” generalized more
but interfered more; “splitters” retained distinctions with weaker transfer.
Networks exhibited corresponding rich/lazy representation regimes controlled
by initialization. The main model was a 12–50–4 layerwise-linear network
trained with online SGD, repeating participant schedules 100 times per phase.

**Limits [read].** Human representation geometry was not measured directly.
The findings do not establish a universal impossibility of transfer without
forgetting. The study does not demonstrate a dedicated sleep solution or a
CPU efficiency benchmark.

**SRI interpretation [analogy; untested].** We need to learn what to share,
what to retain separately, and when to revise shared structure. For two
experiences involving rotations of 120° and 150°, sharing a rotation mechanism
while retaining context-specific angles is a possible decomposition. Merely
replacing one shared angle loses the old relation; storing everything
independently misses reuse. The example supplies the decomposition by hand;
discovering it remains the obligation.

**Evaluation implication [derived design choice].** Measure acquisition,
held-out generalization, transfer and retention together, including similar
but conflicting experiences. Low forgetting alone can reward isolation;
good transfer alone can hide overwritten distinctions. The core/satellite
layout does not automatically resolve this trade-off.

## Hybrid predictive coding: learn to make inference cheaper

Source: Tschantz et al., [Hybrid predictive coding: Inferring, fast and slow](https://pmc.ncbi.nlm.nih.gov/articles/PMC10395865/),
2023. Evidence: abstract and selected results passages, not a full methods audit.

**Shown [abstract/selected passages].** A learned recognition pathway supplies
an initial estimate, iterative inference refines it, and corrected estimates
train recognition. The authors report fewer required inference iterations as
recognition improves.

**Differs.** This does not establish the same behavior for SRI's private module
coordinates and graded lateral connections, or provide our CPU cost target.

**Reuse hypothesis [analogy].** Predict a module's jointly inferred response,
then spend correction work where needed. Dedicated inference could improve the
fast predictor used during continuous operation. Faster recognition does not
itself solve forgetting, relation discovery or delayed credit assignment.

## Two different uses of internally generated experience

**Wake–sleep [read, abstract and opening description].**
[Hinton et al. 1995](https://www.cs.toronto.edu/~hinton/absps/ws.pdf) train
generative connections from observation-driven recognition during wake and
recognition connections from model-generated activity during sleep. This gives
a concrete way to train inference using internal generation. It does not by
itself establish retention or discovery of transferable abstractions.

**Generative replay [read, abstract and introduction].**
[Shin et al. 2017](https://papers.nips.cc/paper/6892-continual-learning-with-deep-generative-replay.pdf)
combine a generator and solver, mixing generated old-task examples with new
experience. Their continual-learning results provide a preservation reference.
SRI requires generation within its learned system, rather than an uncounted
external generator or historical replay database. That adaptation is untested;
generation cannot recover distinctions the system has already lost.

**Scheduling hypothesis [analogy].** Continuous learning could acquire new
distinctions with bounded local changes. Dedicated phases could reconsider
sharing, rehearse retained behavior, and train recognition from more accurate
inference. These are separate purposes to evaluate; calling extra updates
“sleep” establishes none of them.

## Temporal credit and inexpensive building blocks

- [E-prop, Bellec et al. 2020](https://www.nature.com/articles/s41467-020-17236-y)
  uses eligibility traces with learning signals for recurrent learning
  [abstract and selected passages]. It avoids storing a full BPTT trajectory,
  but trace storage/update work remains. Transferring its derivation to SRI
  requires specifying our dynamics and feedback objective.
- [Online dictionary learning, Mairal et al. 2010](https://www.jmlr.org/papers/volume11/mairal10a/mairal10a.pdf)
  alternates code inference and dictionary updates using accumulated statistics
  [abstract and algorithm excerpt]. Useful as a component or comparator;
  sparse code inference and statistics storage must be counted.
- [Streaming PCA/Oja](https://proceedings.mlr.press/v49/jain16.html)
  supplies a compact online subspace-learning reference [abstract]. Variance
  extraction alone does not provide behavioral preservation or relational
  discovery.

These are candidates to understand, not a claim that their combination is
already a working learning system.

## Brain rhythms as constraints on mechanisms

2026-10-27 follow-up: can observed brain waves help infer useful activity?

**Inverse problem [read, selected passages].**
[Buzsáki, Anastassiou and Koch 2012](https://buzsakilab.com/content/PDFs/BuzsakiKoch2012.pdf)
explain that extracellular fields aggregate multiple current sources; different
cellular events can produce similar waveforms. Their inverse-problem section
motivates combining forward models, field recordings, cellular activity and
interventions. A frequency band alone does not identify a learning algorithm.

**Coordination evidence [read, abstract and discussion].**
[Staresina et al. 2023](https://www.nature.com/articles/s41593-023-01381-w)
combine human intracranial recordings and multiunit activity. Slow oscillations,
spindles and ripples occur in a nested sequence associated with increased
firing and cross-region interaction. The study lacks a behavioral consolidation
assay, so those observations alone do not establish improved memory.

**Causal evidence [abstract].**
[Maingret et al. 2016](https://pubmed.ncbi.nlm.nih.gov/27182818/)
used timed stimulation in rats to reinforce hippocampal–cortical coordination
during sleep, improving subsequent recall in their spatial task. This supports
the importance of timing in that setting, not a universal optimal frequency.

**SRI hypothesis [analogy; untested].** Reuse the computational principle of
coordinating communication and plasticity: activate selected learned patterns,
let participating modules exchange information, then apply bounded learning
updates under an explicit objective. Selection, temporal credit, preservation
and generation still need mechanisms; an oscillation supplies none by itself.
An event-driven scheduler may implement such coordination without simulating
biophysical spikes or matching human frequencies. Its cost advantage is unmeasured.

**Discriminating comparison [proposed, not a contract].** Hold the patterns,
updates and computation fixed while comparing coordinated, shuffled-timing
and ordinary sequential schedules. Evaluate retention and transfer, not
similarity to an EEG spectrum. A scheduling benefit would not by itself prove
that literal oscillations are necessary. This remains consistent with the
program's definition of structural resonance, which need not oscillate.

## Neuron diversity: separate representation, routing and plasticity

2026-10-27. Targeted reading/search on dendritic computation, cortical PV/SST/VIP
interneurons, dopamine prediction errors and inhibitory plasticity. These
are selected examples, not a taxonomy or a one-to-one mapping of cell types
to software functions. Roles depend on subtype, circuit and state.

- **Separate input compartments.** [Guerguiev et al. 2017](https://elifesciences.org/articles/22901)
  model learning using segregated dendritic compartments [abstract and selected
  passages]. This is a computational proposal, not proof that brains implement
  its algorithm. SRI hypothesis: keep evidence-bearing activity and contextual
  or teaching feedback distinguishable, rather than summing everything into
  one undifferentiated state. Small compartment variables may suffice for an
  engineering test; detailed dendritic simulation is not required.
- **Different targets of inhibition.** [Pi et al. 2013](https://www.nature.com/articles/nature12676)
  identify VIP-mediated disinhibition involving primarily SST and some PV
  interneurons [abstract]. [Yang et al. 2016](https://www.nature.com/articles/ncomms12815)
  model how dendritic disinhibition can selectively gate pathways [abstract and
  introductory passages]. PV basket-cell perisomatic inhibition and SST
  dendritic inhibition motivate distinguishing control of a unit's output
  from control of particular incoming pathways. These labels encompass
  heterogeneous cells and do not define universal functions.
- **Outcome-modulated learning.** [Schultz, Dayan and Montague 1997](https://pubmed.ncbi.nlm.nih.gov/pubmed/9054347/)
  link dopamine activity to reward-prediction errors [abstract/search record].
  SRI analogy: distinguish a representation signal from a signal authorizing
  or modulating a weight change. Eligibility traces provide a candidate way
  to associate delayed feedback with earlier local activity. Dopamine is not
  a universal scalar error channel, and reward is not the sole learning signal.
- **Learned inhibitory regulation.** [Vogels et al. 2011](https://pubmed.ncbi.nlm.nih.gov/22075724/)
  study a model in which inhibitory plasticity balances excitation and
  inhibition [abstract]. SRI hypothesis: adapt local thresholds or suppression
  to regulate participation. Stable activity levels do not guarantee stable
  memories, convergence or useful representations.

**Program implication [analogy; untested].** Graded connection strength says
how much interaction is permitted. Context-dependent gating could additionally
determine which pathway participates, while plasticity control determines
which participating connections may change. Keep those three decisions
separate when defining a mechanism. Waking and dedicated phases could apply
different routing/plasticity policies without changing the entire substrate.

**CPU implication [conditional design].** Gates, activity averages and bounded
traces can be small state variables, but their update and routing cost must
be counted. Suppressing an already-computed value saves no computation. A
benefit needs a way to skip expensive work without scanning every candidate
pathway. None of the cited biology establishes CPU efficiency for SRI.

## Candidate edge algebra for transmission, control and learning

Follow-up reading: [compositional interactions and efficient edge
operators](edge-algebra-notes.md) connects this sketch to tensor-network
diagrams, structured arithmetic circuits, and executable categorical models.

2026-10-27. Derived illustrative construction, not a selected architecture,
biological equivalence, frozen contract or implemented experiment.

An edge can be a stateful, typed operator rather than a scalar weight. For a
directed edge e use state (w_e, g_e, p_e, z_e): a persistent weight, activity
gate in [0,1], plasticity gate in [0,1], and bounded transient eligibility
trace. Keep control messages distinct from evidence-bearing messages.

A scalar transmission example is m_e = g_e w_e x_source. A control operator
can set a specified edge gate to clip(g_e+c,0,1), or update a separate
plasticity gate. Routing a control message to a gate is an explicit operation;
it is not automatically implemented by multiplying negative weights.

For an illustrative local squared prediction loss with fixed activities and
gates, let y_hat_j = sum_e m_e and r_j = y_j-y_hat_j. Then
q_e = g_e x_source r_j is the instantaneous negative weight gradient. A
candidate trace is z_e' = clip(rho z_e+q_e,-Z,Z), 0<=rho<1, and a candidate
weight change is w_e' = project(w_e+eta p_e z_e'). For rho=0 without clipping,
this is a gated local delta update. For rho>0 it is a heuristic trace rule;
it is not an exact recurrent gradient or an established solution to delayed
credit. The source of y_j/r_j and gate learning remain unspecified obligations.

**Algebra.** Each elementary operation is a transformation of the declared
global state, acting on a small subset. These transformations compose
associatively and have an identity, but generally do not commute. Thus
“open gate; transmit” differs from “transmit; open gate.” Distinguish this
execution composition from semantic composition of learned relations. Ordinary
matrix products describe fixed linear paths only; stateful gating requires
carrying state and execution order.

**Node and scheduling obligations.** Edges alone do not define the network.
Specify how a node aggregates messages in each compartment, changes state,
and exposes readouts. A synchronous schedule can read old states and commit
updates together; a sequential one needs an explicit ordering. Neither
associativity nor bounded weights proves convergence of feedback loops.
Pairwise operators may require auxiliary state or higher-arity interactions
for collective relations, as noted in [interaction semantics](interaction-semantics.md).

**Cost [derived, conditional].** Scalar operations take O(1) per visited edge,
with O(E) edge storage; a pass costs O(E_active+V_active) only if active-edge
selection, control propagation and trace decay also fit that bound. Dense
k-dimensional translations instead cost O(k^2) per edge. No performance or
learning-quality measurements exist. The next mathematical task is to specify
the node law, feedback objective and update schedule before testing utility.

## CPU obligations

Local-looking updates are insufficient: activity inference, module selection,
trace maintenance and memory access may dominate. Under constant work per
visited edge, K passes over E visited edges cost O(KE) [derived, conditional],
but selecting those edges must be included. Sparse activity does not imply
sparse computation if all connections are scanned.

Compare sustained acquisition and retention at fixed wall/CPU time and memory,
with separate continuous latency and dedicated-phase budgets. Initialization,
base training, tuning and internal generation also count. No CPU benchmark was
run for these notes. Follow [computation reporting](computation-reporting.md)
before implementing a substantive comparison.
