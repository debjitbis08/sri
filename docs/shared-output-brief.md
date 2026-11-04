# SRI · A relational theory of knowledge and its first neural test

**Research question.** Can the same learned structure represent a relation, evoke a missing participant, and establish whether a proposed participant fits? Over longer timescales, can it learn where that relation applies, retain the grounds for accepting it, and revise the right conclusions when those grounds change?

## The proposed theory

Consider a claim p = R(x,y). Its content can be shared by several episodes. A particular observation or inference provides a ground J for accepting or opposing p under conditions C. Write this as *J supports p under C*. This is itself a relation. J may refer to observations, assumptions, or earlier claims, and its applicability can in turn be challenged. This gives the web a common representational form while preserving the different consequences of seeing, inferring, and imagining the same thing.

**One neural read.** With R, x, and the relevant context active, the web evokes support across possible fillers y. A supplied filler probes that pattern. An open slot lets the pattern recall a filler whose state can participate in another relation. In the full theory, support can depend on several grounds, including grounds against a binding. A scalar activation may suffice to choose an answer today while losing information needed to revise it tomorrow. The dependencies must survive somewhere in the learned state. The notation describes required behavior. All inference and revision in the proposed architecture must be neural.

### What context, scope, and grounds each contribute

| Question | Example | Required distinction |
|---|---|---|
| Under which interpretation does this hold? | Ordinary counting gives 4 after 3. A board arrow takes 3 to 2. | A learned default can apply until cues activate an exception specific to this relation. The board may change succession while preserving numerical equality. |
| What does the learner have grounds to say here? | A bird standing still has not flown during the observation. Its capacity to fly remains open. | Separate positive and negative support permit supported, opposed, unknown, and disputed judgments. A failed observation establishes absence only under suitable detection conditions. |
| Across which cases does it hold? | One flying sparrow and one flightless penguin establish that some birds can fly and some cannot. | Always and never quantify over admissible cases. Sometimes requires a qualifying case, and “sometimes but not always” requires a counterexample too. Most needs evidence about a population. A default can survive a counterexample to a universal. |
| How was it established? | A proof of 1 + 1 = 2 depends on arithmetic definitions and rules. Observing many pairs gives a different warrant. | Repetition can strengthen an empirical generalization. Deductive necessity depends on a derivation within stated assumptions. |

Mīmāṃsā motivates the distinction among means of knowing (*pramāṇas*). For SRI, perception raises the question of what an observation licenses, testimony raises the question of source dependence, and inference raises the question of which premises a conclusion inherits. Comparison can suggest a relation, and a postulated explanation can reconcile observations while still needing independent support. These are proposed computational roles inspired by the tradition. How to learn them remains open.

### Continual learning as selective revision

Suppose two learners accept “this bird can fly.” One saw it fly and also inferred flight from its species. The other relied entirely on the species identification. Correct that identification. The first learner should still accept the claim. The second should return to unknown. If both histories produced identical internal states, the same update would produce identical answers. Thus the learner must preserve some distinction between independent grounds even when their current answers agree.

| Operation | Observable requirement |
|---|---|
| Establish | An episode can introduce a claim with its source and conditions of observation. |
| Derive | A reusable relation can combine premises or fill a missing role while retaining the dependencies that license the result. |
| Challenge | New evidence can target a premise, a particular application, or the scope of a general rule. |
| Revise | Withdrawing one ground changes conclusions that relied on it. Independent grounds continue to support their conclusions. |

**Connection to Algebraically Constrained Relational Webs.** For a specified family of queries and updates, two histories can safely share a representation exactly when every admissible future update sequence leaves them giving the same answers to every such query. This is a criterion for abstraction. It allows compression while identifying distinctions that an exact learner must preserve. A possible role for slow consolidation is to discover reusable groupings that satisfy this criterion approximately. Which updates matter, and how the network learns such groupings, are open research questions.

## How the pilot fits the theory

| Stage | New capability | Discriminating test |
|---|---|---|
| 1 · Shared output | Probe and recall through one neural binding pattern in a single default setting. | Teach a name only through equality. Test whether it inherits ascending and descending continuation, including rejection of a wrong candidate. |
| 2 · Context and exceptions | Let cues select the applicable relation. Attention is the planned extension. ATLAS-style test-time memory is one candidate when relevant observations are distant. | Change the earlier cue, hold the immediate query fixed, and recover the original answer when its cue returns without reteaching the answer. |
| 3 · Provenance and revision | Preserve which observations, rules, and assumptions support each claim. | Give two items equal initial support through different grounds. Retract one shared assumption and verify that only the independently supported item remains accepted. |
| 4 · Abstraction | Discover relations that can be reused across episodes and revised after consolidation. | Compare answers before and after grouping, including after a correction that distinguishes two otherwise similar histories. |

Only stage 1 is an implementation proposal for the present pilot. Later stages specify capabilities and tests. Their neural mechanisms remain research questions. [ATLAS, Behrouz et al.](https://proceedings.mlr.press/v306/behrouz26b.html)

## Stage 1 implementation brief

**Decision.** Build a small pilot in a single default setting. Defer provenance, learned context selection, ATLAS, and attention. Keep the two observed items already used for sequence prediction. They are the evidence that distinguishes ascending from descending sequences.

**Hypothesis.** An activation that lets SRI recall an item can also let it judge whether that item fits. The useful outcome is transfer: a symbol taught only as another name for three should inherit what SRI knows about three.

## Why change the output?

The current comparator g scores pairs. The predictor P generates a vector and ranks collections by distance to it. In the sequence pilot, one ascending step from collections reached the right collection by distance 59% of the time (11% chance). Reading predicted states through g's sameness judgment gave 0–23% across the reported conditions. These criteria can disagree even when each learns its own task. [Sequence pilot](https://github.com/debjitbis08/sri/blob/main/docs/sequence-pilot-v1-results.md) · [Inventory of trained relations and reads](https://github.com/debjitbis08/sri/blob/main/docs/relational-expression-inventory-v1.md)

## Proposed forward pass

Encode a query once, before seeing any proposed answer. Use the existing relation cue and source for a pair query, or a continuation cue with the ordered previous and current item for a sequence query. A missing previous item can use a fixed empty slot. Pass the query and each addressable item code through the same learned scorer to obtain one logit per item. A small MLP suffices for the pilot. Keep the scorer parameters shared across relations and query modes.

| Query | Neural read |
|---|---|
| Is 3 equal to 2? | Form the sameness query from 3 and probe the logit at 2. |
| What comes after 3 in the sequence 2, 3? | Score the internal items, select the strongest accepted item, and return its existing core code. |
| Does 4 continue 2, 3? | Probe the exact logit at 4 from the preceding row. |

For a first finite implementation, retain familiar collection exemplars from the training split as the internal inventory. Encode them through the current core. Document how exemplars are selected, storage cost, and inventory coverage. Keep evaluation renderings distinct. This pilot measures recall of familiar items. Construction of unseen item states remains an open question.

Use the same logit for ranking and binary supervision. Choose an acceptance threshold on validation examples and freeze it before testing. For a single-answer query, abstain if every available item falls below threshold. For a relation with several valid fillers, judge each filler independently. All relation scores come from the network. Arithmetic labels and sequence metadata belong in the training and evaluation harness only.

## Training and comparison

Reuse the existing encoder, curriculum, relation examples, ascending and descending histories, and relearning schedule. Preserve argument order. A descending continuation differs from the fixed, upward-oriented relation called “next” in the existing pair task. For continuation judgments, use the observed next item as a positive and sample known wrong completions in the finite counting task. Keep the direction implicit in the two observations.

Compare against the current g/P model on paired seeds and identical data splits. Give the baseline access to the same positive and negative continuation examples through its distance logits, and retain the original baseline result for reference. Match or report model size, training budget, and inventory access. Otherwise an advantage could come from extra supervision or stored exemplars.

## Tests that decide whether to continue

1. **Naming-only transfer.** Teach a held-out symbol X solely through X = 3. Exclude X from all sequence training, including negatives. Test (2, X) and (4, X) for recall of 4 and 2, respectively, and for acceptance and rejection of individual completions. Report both directions separately alongside the familiar-3 control.
2. **Rejection.** Evaluate single candidates and sets with the correct item removed. Score the latter using the same stored logits restricted to the available set. Report false acceptance and correct acceptance at the frozen threshold. Here rejection means “wrong completion in this fixture.” General epistemic truth remains a separate question.
3. **Continual retention.** Repeat the same probes before and after script relearning. Report paired changes in transfer, rejection, acquisition of the relearned script, and the existing cross-sense and composition reads. [Relearning diagnostics](https://github.com/debjitbis08/sri/blob/main/docs/relearning-margins-v1.md)

**Interpretation.** Agreement between recall and verification is a wiring check. Advance the design if held-out naming transfer improves across paired seeds while rejection, relearning, and prior abilities remain within tolerances declared before the run. If only familiar-item recall improves, investigate memorized pairings. If transfer works before relearning and disappears afterward, the remaining problem is update interference. Either outcome can be diagnosed before introducing attention or provenance.
