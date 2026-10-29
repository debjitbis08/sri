# Conceptual instinct and asking to learn

Discussion proposal, 2026-10-29. Not a frozen contract or implemented experiment.

The target is an immediate sensitivity to violated learned relations, distinct
from identifying the correction or estimating whether an answer is reliable.
Part A's agreement predictor should not be judged solely as a revision gate.

## Motivating intuition

The user's words, 2026-10-29:

> You know what great scientists have, instinct when something is not right.
> They do not calculate or think, rather they have assimilated the concepts so
> well, that that understanding has now become instinct. They can tell
> something is wrong, even without knowing why. That is the capability we
> need. Confidence is needed for declare "I don't know" so let me ask and
> learn.

How this reads against the design below:

- **Not calculating.** The signal comes before any revision solve or search,
  as one cheap pass over the learned relations. It is not a slow-path result
  reported afterwards.
- **Assimilated concepts.** It comes from the same learned relations that do
  the reasoning, not from a separate detector trained on how inputs look. The
  energy-ranking objective below is one way to make that happen; amortizing
  slow judgments into a fast one (Gershman and Goodman) is another.
- **Wrong without knowing why.** Detection is its own capability, measured
  apart from finding the faulty item or correcting it.
- **"I don't know", so ask and learn.** The sense that something is wrong has
  to become a declared lack of knowledge, which then triggers a question
  (Part C), not a guess or a silent revision.

What Part A already shows:

- **Supports the intuition.** On the pilot 3 checkpoints, a relational
  consistency check before revision predicted instinct–inference agreement
  (AUROC 0.66–0.96). Recognition-only signals barely did (0.45–0.86), and the
  SRI energy ran the wrong way. So the sense that something is off has to come
  from the relations, as the intuition says.
- **What is still missing.** That check evaluates the successor map
  explicitly against given neighbours, so it is still a calculation. It also
  tells us *that* instinct and inference disagree, not *which* one is right.
  The open test is whether the sense can be absorbed into the fast path:
  present without neighbours being supplied, still working on held-out
  combinations, and producing an "I don't know" that leads to useful
  questions.

## Reading record

Searches on 2026-10-29 covered predictive coding and novelty/mismatch detection,
energy-based learning, amortized inference, and information-gain active learning.

- **[read: abstract, methods 2.5, results 3.4]** Li, Tang and Bogacz (2025),
  [Predictive Coding Model Detects Novelty on Different Levels of Representation
  Hierarchy](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618029/).
  Layer-specific prediction errors distinguish sensory and semantic novelty;
  the MNIST experiment separates new exemplars of a trained digit from other
  digits. Its semantic experiment performs inference until convergence. This
  demonstrates neither immediate relational contradiction detection nor
  scientific intuition. **[analogy]** Use errors at the quantity/relation level,
  rather than undifferentiated activity energy, to detect conceptual mismatch.
- **[abstract]** Rao and Ballard (1999),
  [Predictive coding in the visual cortex](https://www.nature.com/articles/nn0199_79).
  Predictions and residual errors provide the established architectural idea;
  this visual-cortex model is not a number-learning result.
- **[abstract/search excerpt]** LeCun et al. (2006),
  [A Tutorial on Energy-Based Learning](https://yann.lecun.org/exdb/publis/pdf/lecun-06.pdf).
  Compatibility energies rank observed configurations below alternatives.
  Full PDF retrieval failed. **[derived]** A margin separating coherent and
  deliberately inconsistent episodes is an energy-ranking objective, not a
  novel error family or the same as minimum quadratic SRI activity energy.
- **[read: introduction, model, experimental results]** Gershman and Goodman
  (2014), [Amortized Inference in Probabilistic Reasoning](https://web.stanford.edu/~ngoodman/papers/amortized_inference.pdf).
  Human Bayesian-network query experiments support reuse of sub-inferences.
  **[analogy]** Learning to approximate costly resolved judgments with a fast
  predictor is a route to practiced intuition. This paper does not establish
  our proposed neural distillation method or its generalization.
- **[read: section 2, equations 1–2]** Houlsby et al. (2011),
  [Bayesian Active Learning for Classification and Preference Learning](https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf).
  BALD chooses queries by mutual information between answers and model
  parameters; its predictive-entropy difference favors disagreement among
  individually confident posterior models. The paper uses Gaussian processes.
  **[derived/proposal]** A small neural committee could approximate this
  criterion, but is not an exact posterior and can share blind spots.

## Concrete proposed mechanism

**[derived/proposal]** Keep the existing independently inferred quantity codes
z_i and shared relation maps T_r. For an asserted relation edge (i,r,j), expose
the residual z_j - T_r(z_i) before revising the states. Normalize residual
coordinates using development-set variability and a variance floor. Aggregate
the residuals as an episode compatibility energy; keep perceptual novelty
separate from this semantic score.

Train with grounded matching plus relation prediction, and a ranking loss
max(0, margin + E(coherent) - E(inconsistent)). Construct inconsistent
episodes by changing relations among familiar items, balancing appearance,
language, length and individual-item frequencies. A permutation is negative
only when it violates the stated operation/task. Shared core parameters receive
this loss; there is no separate appearance-only anomaly classifier. Grounding
and negative examples prevent the trivial all-codes-equal solution. These are
established predictive/contrastive ingredients. Existing successor BCE already
contains part of this mechanism; the new test concerns episode consistency and
generalization, not renaming that loss.

Mismatch does not imply ignorance or identify the faulty observation. Train
and evaluate answer reliability separately on verified feedback. For asking,
compare random and uncertainty sampling with committee information gain over
permitted teacher questions. Unknown symbols can warrant asking without
producing a contradiction. Learn from verified answers with retained examples;
do not train away genuine contradictions merely to make their energy low.

## Discriminating evidence and cost

**[proposal]** Hold out language/count/relation combinations. Measure violation
detection before revision separately from correction accuracy, false alarms on
novel coherent episodes, questions to target accuracy, and source retention.
Compare current residuals, the trained compatibility objective, and a matched
capacity appearance-only detector. Perturb relations with perception fixed.

**[derived]** For M edges and d-dimensional dense affine relation maps, residual
evaluation costs O(M d^2), with O(R d^2) relation storage for R relation types
and O(M d) optional residual storage. Input encoding and ordinary SRI solves
are additional, not free. Negative episodes multiply training evaluations.
K complete committee models multiply model storage and inference roughly K
times; Q candidate questions require O(Q K C) entropy work for C answer
categories after their model predictions. Any distillation adds teacher-data
and student-training costs. A future contract must report these and tuning,
wall/CPU and peak-memory measurements separately from scientific results.

## Extension: varied contexts and source reliability (2026-10-29)

The researchers propose assimilating a concept through different contexts,
then extending belief to account for source count and previously learned
source reliability. These should remain distinct: context diversity helps
identify an invariant; independent reliable evidence supports belief.

- **[abstract/search excerpt]** Yu and Smith (2007),
  [Rapid word learning under uncertainty via cross-situational statistics](https://pubmed.ncbi.nlm.nih.gov/17576281/):
  adults acquired word–referent mappings from ambiguous trials through
  cross-trial information. This supports the possibility of accumulating
  grounding across situations, not retention or relational composition in SRI.
- **[abstract]** Dawid and Skene (1979),
  [Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm](https://academic.oup.com/jrsssc/article/28/1/20/6953573):
  latent true responses and observer error rates are jointly estimated.
  **[proposal]** Start more conservatively with source/domain error estimates
  from independently verified outcomes; treat unknown reliability as uncertain.
- **[abstract/search excerpts, including results and conclusion]**
  [The effect of speaker reliability on adult cross-situational word learning](https://escholarship.org/uc/item/1p34f58m):
  its two experiments did not establish the predicted impairment from an
  unreliable speaker. Contextual learning and source trust are not empirically
  interchangeable mechanisms. Full-page retrieval was blocked.

**[proposal]** Teach a quantity through object collections, adding/removing an
object, comparisons and different forms, then ask the shared representation
to predict an omitted view or relation. Supply equivalence through observed
events or explicit teaching, not hidden evaluator labels. Include retained
old-language examples; diversity alone does not prevent forgetting. Compare
varied versus repeated contexts with matched exposures, replay and compute.
Evaluate novel compositions of taught relations, not only their practiced pairs.

**[derived]** Varied contexts also address a weakness of the instinct proposal
above: with successor as the only learned relation, a fast detector can learn
to re-evaluate successor, which is a memorized check rather than an
assimilated concept. With several contexts feeding a shared core, a violation
can surface through a context that was not checked directly. Assimilation is
then testable: learning a concept in some contexts should predict its behavior
in a held-out context it was never paired with.

**[researchers' guidance, 2026-10-29]** Representations should be limited in
number, but one is not required: they act as **senses** for different kinds of
input (seeing, hearing), each feeding the shared core. One per context would
let each context be learned separately with nothing assimilated; one for
everything is not needed either. **Rehearsal** is also important, alongside
varied contexts.

**[researchers' proposal, 2026-10-29] Imagery.** Trigger the core or middle
layers directly and read what the input and output modules show: the signal
flows in the opposite direction to perception. The question is what the system
"sees" when a signal starts inside it.

**[implemented, 2026-10-29]** Number construction Part D tests rehearsal and
varied contexts through two senses (images; taps as a heard count), with
taps–form matching and form comparisons left untaught. An imagery probe clamps
core states (prototypes, the imagined successor, the gap between two
neighbours, middle-layer states, random controls) and solves the energy
outward to each sense's input layer. See the Part D protocol in
`number-construction-v1-spec.md`.

**[recalled, not retrieved this session]** The motor-learning
contextual-interference effect (Shea and Morgan, 1979) and later
interleaving studies report that varied practice slows acquisition but
improves retention and transfer. **[prediction]** Varied contexts may look
worse than repeated ones on training curves. Score the comparison on transfer,
held-out compositions and retention, not acquisition speed.

**[derived]** Evidence from repeated renderings, copied statements, or multiple
predictions made by one model is dependent. Counting these as independent
confirmations creates unjustified confidence. Keep provenance and dependence
groups; calibrate source reliability on resolved claims rather than agreement
with current beliefs. Source competence can differ by domain. A later study
must distinguish one-source repetition from independent corroboration at equal
claim count. No claim of a novel method or of measured improvement is made.

**[derived/proposal]** Concept strength and source reliability meet when a
claim contradicts what the learner already holds: is the source wrong, or is
the learner? A claim that contradicts a well-assimilated, well-corroborated
concept should lower trust in its source. A claim about a weakly held concept
should defer to a source with a good resolved record. Belief therefore
combines two quantities: how firmly the concept is held, from independent
contexts and verified outcomes, and how reliable the source is in that domain.
The same dependence rule applies inside the model: agreement across distinct
contexts is evidence, but repeated renderings of one context are not.
Updating reliability from agreement with current beliefs would teach the
learner to trust whoever agrees with it, so this combination must still be
calibrated on independently resolved claims. **[recalled, not retrieved this
session]** Truth-discovery methods that jointly estimate claim truth and source
trustworthiness, such as TruthFinder (Yin, Han and Yu, KDD 2007), are the
standard starting point alongside Dawid and Skene.

**[proposal]** Ordering: varied contexts belong after the forgetting fix and
before a full test of the single-pass detector. They can be generated in the
current setting (dots, add/remove events, comparisons, scripts). Source belief
needs a different setting with sources, claims and a reading history, so it
stays at design stage until the concepts themselves are stable.
