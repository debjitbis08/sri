# Continuous sensory memory: research direction

Discussion record, 2026-10-31. **Not a contract or an implemented architecture.**
This records the user's requirements and the architecture discussion following
the [translation-memory pilot](translation-memory-v1-pilot.md). It does not
change historical gates or results. Architectural mechanisms below are
hypotheses unless explicitly identified as requirements or existing evidence.

## User requirements

- Target a **multimodal, continuously learning memory system**. Inputs enter
  through the senses. Teaching can be paired images, sounds or other sensory
  experience; hidden evaluator identities must not become privileged memory
  inputs. Operational feedback must also arrive through an allowed channel.
- Perfect memory is unnecessary. Useful retention, new learning and selective
  forgetting under bounded resources are the objective.
- Inadequate perception should be upgraded. A permanently frozen symbol
  encoder is a diagnostic or scoped baseline, not the final answer to
  continually adapting perception.
- **Retrieval should influence retention.** Repeated use should help memories
  persist, while less useful material can lose priority. This is a desired
  behavior, not a guarantee that every retrieval is correct or beneficial.
- Include a level for **recent, ongoing work**, between immediate context and
  durable knowledge. The user's example is retaining many ML-memory details
  tomorrow and over several days while absorbed in this research, then losing
  some accessibility after prolonged absence.

## Proposed functional organization

| Level | Role | Candidate implementation, not selected |
|---|---|---|
| Immediate context | Current sensory observations and recent interactions in detail | Bounded transformer attention |
| Active-work memory | Ongoing problems, relevant associations, recent discoveries and unfinished questions across sessions | Persistent gated recurrent/neural memory, possibly with bounded episodic storage |
| Durable memory | Knowledge and experiences useful beyond the current work | Adaptable neural memory and selective consolidation |

These are functional roles, not an asserted three-store account of the brain.
Three physical modules are not required in advance: compare with simpler
systems and demonstrate distinct contributions through interventions. The
relational/generalization function in [the earlier memory note](memory-kinds-notes.md)
remains relevant; it is not replaced by this timescale organization.

**[hypothesis]** Sensory experience enters immediate context, relevant content
updates active work, retrieval reactivates useful context, and selected content
is integrated into durable memory. Returning to a project should restore
continuity without replaying its entire history. Elapsed time, intervening
experience and reuse frequency must be separated experimentally. Calendar days
are a target use case, not a prespecified decay constant or measured timescale.

## Mechanisms discussed and their limits

- LSTM-style gates offer learned retention, writing and exposure of state.
  State adaptation and online parameter learning remain separate questions.
- [Titans, MIRAS and ATLAS](test-time-memory-prior-work.md) provide established
  approaches to test-time neural-memory updates, retention objectives and
  bounded recent-window optimization. Do not assume a novel mechanism is needed
  before identifying a gap in these approaches.
- Input-driven Hopfield dynamics motivate persistent sensory guidance during
  retrieval; they do not establish long-term learning or encoder compatibility.
- Explicit stores remain useful baselines/components. Success at 120 words
  does not establish scalable continuous memory. Their selection, indexing,
  eviction and maintenance during encoder changes require evaluation.
- The frozen key pathway isolates retrieval from changing perception. It does
  not solve the final interface problem. One frozen encoder has constant model
  storage; accumulating a snapshot per batch does not.

Source readings and evidence qualifications are in [prior work](prior-work.md),
including LSTM gates, retrieval practice, input-driven dynamics and test-time
memory. No novelty claim is made here.

## Retrieval as an update, not only a read

**[hypothesis]** Sensory cue → retrieval → use → memory update should support
use-dependent retention. Distinguish retrieval without feedback from sensory
confirmation or correction. Do not let evaluator correctness choose updates.
Self-reinforcing errors and loss of related unused memories are failure modes.
LSTM output gating and surprise-driven neural-memory writes do not by
themselves guarantee retrieval-induced strengthening.

Natural interaction queries may update state. Audit evaluations should use
discarded copies so repeated measurements do not silently rehearse every item.
Compare query practice, matched sensory re-exposure and no practice, including
a control that disables retrieval-related updates.

## Next questions, not a frozen experiment plan

1. Can paired sensory experiences establish associations without hidden labels?
2. Can the system resume task A after interruption by B, recovering unfinished
   context under a fixed total budget? Compare repeated resumption with
   abandonment and measure interference between projects.
3. Does retrieval improve delayed retention beyond re-exposure, and can errors
   still be corrected?
4. Can perception improve while old memories remain usefully accessible?
5. Which behaviors require the intermediate level rather than a larger short
   window or a single recurrent memory at matched resources?
6. What selects durable content and schedules consolidation without externally
   supplied batch boundaries or privileged project identifiers?

Before a contract: review relevant prior work, specify allowed sensory channels,
learning signals, persistence/reset boundaries and resource budgets. Count
encoders, parameters, persistent states, optimizer state, buffers, replay and
selection costs. Derive costs and collect runtime data under
[computation reporting](computation-reporting.md). No new run or commit is
implied by this discussion record.

## Assimilation, exceptions and reorganization

Discussion follow-up, 2026-10-31. The user re-centred the objective on the
original sparse-network retention problem: distinguish experiences that fit
existing organization from those requiring that organization to change.

**[derived from project results]** Do not equate sparse connectivity with a
universal retention penalty. The limited-memory and arbitrary-pairing results
separate transfer, acquisition and delayed retention. Sparse readers supported
transfer but acquired fewer arbitrary bindings; later recall was poor across
arms. The forgetting-mechanism study implicated both encoder and reader updates
and optimizer-state movement. Protecting stored values alone is insufficient.

**[hypothesis]** Treat the decision as sequential evidence gathering, not a
classifier that can reliably label one surprising event. Distinguish:

- a new binding expressible with existing features;
- a specific exception worth remembering without changing a general rule;
- repeated, coherent mismatch suggesting a missing feature or relation;
- perceptual ambiguity or noise, which may warrant none of those updates.

Active-work memory can temporarily retain sensory evidence and alternative
interpretations. Compare limited candidate updates: binding/readout adaptation,
an exception memory, and controlled representation/topology changes. Prefer
reorganization only when it improves predictions on independent subsequent
sensory experience at an acceptable cost to older experiences and resources.
Errors alone are insufficient; repeated arbitrary labels need not reveal any
general rule. Gradient conflict diagnoses potential interference, not whether
the old or new interpretation is correct. Any trial copies, replay, update
selection and optimizer state count toward cost.

[Sun et al., 2023](https://www.nature.com/articles/s41593-023-01382-9)
**[read: indexed abstract/model/results excerpts, extending the existing survey]**
model selective consolidation for generalization: indiscriminate transfer can
harm generalization under limited predictability. Difference: linear
teacher–student plus Hopfield notebook, not our sparse reader or a ready-made
online detector. **[hypothesis]** Consolidate repeatable structure selectively
while retaining useful unpredictable bindings in memory.

For the sparse architecture, test whether a binding route preserves necessary
sensory distinctions without undermining relational transfer. Do not assume
that adding dense units solves the tradeoff; earlier mixed arms did not meet
that combined objective. A bounded-budget diagnostic should interleave novel
instances of known structure, arbitrary bindings, genuinely new structured
relations and noisy observations. Category labels belong to the evaluator,
not the model. Assess transfer, old bindings, new acquisition and reorganization
cost together. Retrieval-practice frequency and independent evidence must be
distinguished so repeated recall does not masquerade as confirmation of a rule.

## Accumulating surprises before changing shared knowledge

Discussion follow-up, 2026-10-31. **[user proposal]** Rather than adding a gate
or classifier and asking it to learn when reorganization is needed, retain
memories of surprises. Those that recur accumulate support and drive association
updates; others fade. This is a research direction, not a frozen architecture,
an established result, or a novelty claim.

**[hypothesis: possible mechanism]**

1. Unexpected sensory experiences leave temporary traces containing the evidence
   that caused the mismatch, not merely a scalar surprise score.
2. Related subsequent experiences reactivate and strengthen those traces.
   Isolated traces decay under a bounded memory budget.
3. Reactivated groups contribute jointly to learning. Existing features may
   suffice for an association update; persistent shared errors may instead
   supply pressure to change the representation.
4. If learning explains the experiences, subsequent surprise falls. Temporary
   traces can then fade while the learned association or structure persists.

This gives the intermediate active-work memory a possible computational role:
hold unresolved experience until enough evidence accumulates to explain it.
Different rates of association and representation learning might allow the
distinction to emerge through learning dynamics. That behavior must be
demonstrated; recurrence alone does not establish a need for reorganization.

Important unresolved details:

- Retrieval may strengthen accessibility, but self-retrieval must not count as
  a new independent observation. Otherwise erroneous memories can manufacture
  their own apparent support.
- Related surprises can share a cause even when their sensory inputs differ.
  How to group them when current features miss that cause is an open problem.
  Repeated noise can also remain surprising. Frequency alone is insufficient.
- Evidence for a shared update is improved prediction on independent experiences
  with acceptable loss on older knowledge. This is an evaluation criterion,
  not permission to feed evaluator labels into the operational system.
- Rare useful experiences may fade under a frequency-only rule. Imperfect
  retention is allowed, but that tradeoff must be measured explicitly.
- A memory receiving only a representation that has discarded identity cannot
  recover the missing distinction. Temporary traces need sufficiently rich
  sensory features; their storage and maintenance costs must be counted.

**[connection to the original problem]** Test whether temporary storage of
unresolved experiences lets a sparse shared representation preserve its transfer
advantage while improving acquisition and retention of bindings. Improvement
must be assessed at matched total storage and computation, including temporary
traces and replay. Adding a larger memory alone would not resolve the tradeoff.

**[connection to prior work]** The [Titans/MIRAS/ATLAS reading](test-time-memory-prior-work.md)
identifies relevant ingredients: surprise-sensitive neural-memory updates,
retention objectives, and updates using multiple recent experiences. These do
not by themselves establish the proposed lifecycle from unresolved surprise to
durable association or representation change. Compare accumulated-surprise
updates with ordinary bounded replay and immediate surprise-weighted updates;
test whether any benefit requires grouping, repeated independent evidence, or
retrieval-induced strengthening. Literature review and computation accounting
remain prerequisites to a new experiment contract. No experiment is authorized
or launched by recording this proposal.
