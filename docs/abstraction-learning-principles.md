# Learning reusable structure rather than coincident codes

2026-10-28. Research discussion following number-core pilot 2. Not a contract,
selected algorithm, or experimental result. Historical artifacts are unchanged.

## Diagnosis from the current implementation

The runner learns discriminative losses through a quadratic equilibrium. The
equilibrium defines a differentiable map; training does not fit a generative
likelihood for jointly observed modules. Each solve includes one input kind,
core and output. There is no learned consolidation operation comparing a
shared explanation with separate explanations [derived from
`experiments/number_core_network.py` and `src/number_core.py`].

This does not prove that backpropagation cannot learn abstractions. Nor does
it prove that a different optimizer or local rule would discover them.
Architecture, evidence, objective and optimization are separate obligations.

A shared parameter container need not implement shared computation. As an
illustrative linear construction, encode two contexts in separate blocks,
`c_a=(u,0)` and `c_b=(0,v)`, use `S=diag(S_a,S_b)` and a decoder
`D=[D_a D_b]`. Each context can be processed separately despite one named
successor and decoder. This is a representational counterexample to parameter
sharing alone, not a fitted solution or proof about every pilot loss, in
particular its cross-modal equivalence terms [derived].

Conversely, differing coordinates need not imply absence of abstraction:
learned translations can expose the same reusable operation. Coordinate
agreement is useful instrumentation, but the program's criterion is preserved
behavior and held-out reuse across whole-module responses.

## General learning principle to investigate

Prefer a compact predictive mechanism that works across contexts, while
retaining context-specific distinctions needed for other responses [hypothesis].
Two pressures are needed: preserve predictive sufficiency, and make duplicated
explanations more expensive than genuine reuse. Compression alone can discard
useful distinctions; prediction alone can retain parallel solutions.

A schematic criterion is predictive error plus a declared description cost
for shared structure, private structure, interfaces and any retained state.
This is not an executable MDL objective until a coding scheme and precision
are specified. A fixed parameter count is constant across dense models, and
weight decay or low activation magnitude is not automatically a description
length. Moving duplicated information into translators must count too.

For this project, a candidate consolidation phase would propose shared
factors of already learned response operators, retain private residuals,
refit interfaces, and accept changes only when declared predictive behavior
is preserved and total description cost decreases. Whether those changes
improve unseen transfer must be tested separately. No merge/split search,
representation family or resource budget is selected here.

Prediction should cover families of queries and transformations, not only the
current output label. Experiences can be grouped when they support the same
relevant predictions under those queries; equivalence is relative to this
query family, not a claim that all sensory distinctions are irrelevant.
Missing-context prediction and held-out combinations can test reuse. Targets
must come from actual available evidence; a mask does not create information
about an unobserved cross-modal correspondence [derived design constraint].

Joint latent inference is a possible acquisition mechanism: infer a common
explanation for available evidence and train modules to support it. However,
joint reconstruction, alternating inference/learning, and sleep scheduling
alone do not supply the preference for reuse or resolve unidentifiability.
Self-generated consistency is not independent evidence of correctness.

## Relevant prior work and limits

Search date: 2026-10-28. Queries/sources: “Locatello impossibility unsupervised
disentanglement inductive biases 2019”; “Federici Learning Robust
Representations via Multi-View Information Bottleneck”; “Littman Sutton Singh
predictive representations of state”; “DreamCoder wake sleep library
compression”. Sources below are primary. This is a targeted mechanism search,
not a novelty assessment.

- **Multi-view information bottleneck**, [Federici et al., ICLR 2020](https://arxiv.org/html/2002.07017)
  [read: abstract and introduction]. The method retains task-relevant shared
  information while discarding excess view information, assuming views supply
  the same task-relevant information. Unlike this pilot, its correspondence
  structure and redundancy assumptions are provided. SRI implication [analogy]:
  predictive sufficiency needs an explicit economy pressure; common information
  alone is insufficient when modules provide complementary evidence.
- **Predictive state representations**, [Littman, Sutton and Singh, 2001](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)
  [read: abstract]. Represents dynamical state through action-conditional
  predictions, with a linear representational result relative to minimal
  POMDP state count. This is not a multimodal abstraction-learning result.
  SRI implication [analogy]: define retained distinctions by their consequences
  under tests, rather than by coordinate similarity alone.
- **DreamCoder**, [Ellis et al., PLDI 2021](https://www.neurosymbolic.org/papers/EllisWNSMHCST21.pdf)
  [read: abstract and introduction; compression/sleep discussion in the
  [2023 account](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0050)].
  Learns reusable program libraries and a search policy; refactoring extracts
  common program components. Differs: symbolic program search, explicit task
  solutions and replay, not SRI's continuous modules or intrinsic-memory
  requirement. SRI implication [analogy]: abstraction can be a distinct
  refactoring operation on acquired solutions. Neither its implementation nor
  its compute requirements are adopted here.
- **Identifiability**, [Locatello et al., ICML 2019](https://proceedings.mlr.press/v97/locatello19a.html)
  [abstract]. Establishes limitations of unsupervised disentanglement without
  appropriate inductive biases. This theorem is not directly about the
  supervised relation pilot. SRI implication [analogy]: specify the evidence
  and assumptions distinguishing shared explanations; no learning rule can
  be credited with resolving an ambiguity left intact by its observations.

## Next research obligation

### Follow-up: abstraction probing changes the priority

2026-10-28. User-supplied source:
[An et al., Does Deep Learning Learn to Abstract?, v1](https://arxiv.org/html/2302.11978v1).
Search record: direct reading of the supplied version, sections 3–6 and
appendices A–E, G and K.

Source findings [read]: controlled transfer appears under gradient training;
transfer develops later than in-task performance. Generic pretraining matters
strongly in the tested models. Their probe includes downstream adaptation.

Revision [derived]: the consolidation proposal above is a candidate, not a
necessary condition for abstraction. The pilot does not establish that its
optimizer or lack of explicit description-length minimization causes failure.
An empirical preference for reuse can arise from the combined architecture,
training history, regularization and optimization dynamics. Which of these
matters here remains unresolved.

SRI hypothesis [analogy; untested]: distinguish acquisition of relations,
development of reusable mechanisms, and adaptation of a new interface. Measure
transfer at successive source-training checkpoints, using the same target
adaptation data and budget. Compare structurally compatible source experience,
matched incompatible experience, and no source experience. A control obtained
only by renaming the same chain is still structurally compatible. Target
examples must identify the new interface without revealing every tested
composition; otherwise success can come from relearning the whole task.

Differences and limits [derived]: SRI uses randomly initialized small encoders
and quadratic modules, whereas the paper's positive results concern pretrained
Transformers. Its success does not establish SRI representational sufficiency,
that more updates will repair this pilot, or that a particular biological/local
learning rule is needed. Sparse-label zero-shot transfer, adaptation-assisted
transfer and representational alignment are different outcomes.

Priority [proposed]: audit transfer dynamics and experience before treating
explicit compression/refactoring as the missing mechanism. Preserve the
compression proposal as a comparison. Charge source training, all checkpoint
adaptation runs, validation and selection to computation cost; use validation
tasks for design and keep final transfer tasks untouched.

For the explicit consolidation candidate, first specify a small
shared-versus-duplicated mechanism comparison with an explicit complexity
measure and preserved responses. Show
that the intended reuse is representable, that evidence distinguishes required
correspondences, and that the proposed learning criterion actually prefers
reuse. Include genuinely different mechanisms so excessive merging fails.
Then test acquisition of a new domain through an interface while retaining the
core, alongside retention and held-out compositions. Frozen-core transfer is
one diagnostic; it is not a demand that a developing system never revise its
core.

### Number-systems pilot: interpretation audit

2026-10-28. Read `docs/number-systems-v1-pilot-results.md`, its saved results
JSON, and `experiments/number_systems.py` [read: project artifacts].

- The saved C14 label assignments are MNIST {2,8,9} and Morse {2,5,9}.
  They share labels 2 and 9. The report's assertion of no common labels is
  incorrect; the cross-medium transfer remains interesting but is transfer
  with common grounding, not without it.
- Representation identity and number can coexist in a core. Decoding the
  former does not refute reuse of the latter [derived]. Requiring identity
  to reach chance tests a stronger invariance property than the program's
  behavioral abstraction criterion.
- The probe is fitted on complete source-number labels in validation pools.
  It measures access to representation information, separately from the
  trained sparse-label output's transfer. The adaptation probe is refitted
  on post-adaptation source codes; it does not measure preservation of an
  unchanged readout [derived from runner].
- Cross-medium transfer rules out reuse of the same front end at inference,
  but does not localize learned abstraction to core parameters. Shared losses
  can train the separate front ends into compatible coordinates. A fixed-core
  comparison or selective parameter intervention is needed for localization
  [derived].
- The I14 permutations preserve chain structure while changing its agreement
  with grounded numeric labels. This is a control for shared grounded order,
  not for the existence of an abstract chain [derived].

Hypothesis [untested]: the current evidence reflects partial reusable number
information coexisting with representation-specific information and fragile
access through shared interfaces. Validate transfer separately on labels
shared by a probe pair and on the remaining numbers, measure source-only
learning curves, and separate stable readout transfer from post-adaptation
probe refitting before attributing failure to missing abstraction.

### User proposal: connection-sensitive learning error

2026-10-28. User proposes ordinary input/output error plus an error weighted
by direct connection strength versus connection through the core.

Interpretation [hypothesis, not an agreed implementation]: distinguish
outcome error, route-specific prediction error, and disagreement between
direct and core-mediated predictions. Multiplying the existing outcome error
by a connection-dependent factor alone still gives zero corrective signal
when that outcome error vanishes. An additional disagreement or route-only
prediction target can train a weak route after the combined response succeeds.

Illustrative consolidation update [derived candidate]: let p_d and p_c be
predictions at the same destination under explicitly defined direct and
core-mediated inference conditions. Add
`lambda * stopgrad(r_d) * D(stopgrad(p_d), p_c)` to supervised outcome loss.
Here r_d is independently assessed reliability, not raw weight magnitude;
only the receiving core route learns from this term. Its supervision is
distillation, not a new ground-truth observation. Use true targets where
available, and do not allow self-confirmation to count as correctness.
This is a directional learning rule, not a claim that both routes should
always be made equal. Shared parameters and gradient destinations require
explicit specification before implementation.

Connection strength can measure effective influence, but influence differs
from reliability. Raw edge norms are sensitive to coordinate scales, and
favoring the already stronger route can starve the other. In recurrent
equilibrium systems route contributions need not add; intervention masks
must specify which couplings change and require re-solving the equilibrium.
The present number-systems network has no literal input/output bypass: every
output passes through core. Direct versus mediated routes therefore require
a new operational definition, not merely a new scalar in the existing loss.
Layers must not be declared abstraction levels by definition.

Prior-work search, 2026-10-28: “Jacobs adaptive mixtures local experts
responsibility error gating”; “Guerguiev segregated dendrites”; “Hinton
distilling knowledge neural network”.
[Jacobs et al., 1991](https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts.pdf)
[read: abstract/opening] provides modular expert responsibility learning;
[Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
[abstract] provides prediction distillation. Neither establishes abstraction
in SRI. The proposed direction combines related ingredients; novelty and
transfer benefit are unestablished. A new scalar weighting alone should not
be presented as a novel abstraction principle.

### Meta pilot: retention aggregation and adaptation credit

2026-10-28. Read `docs/number-systems-meta-v1-pilot-results.md`, its runner,
saved results and checkpoint label assignments [read: project artifacts].
The pilot shows modest accuracy-only adaptation gains at approximately matched
compute; it does not demonstrate retained image-task reuse or a compatible-order
advantage. A failed preservation term does not identify its failure mechanism.

Retention audit [derived from saved curves]: with interface-only Hand adaptation,
all ten image sources are unchanged; only Morse shares the updated sequence
satellite. Thus Morse's absolute accuracy change equals eleven times the
aggregate source accuracy change. At k=32 and 200 updates, the saved curves imply
a Morse accuracy decrease of 0.667 in ordinary, ordinary+dev and meta, and 0.680
in meta-I. The aggregate ratios of approximately 0.93 therefore conceal severe
Morse forgetting. Source evaluation reports perfect initial Morse accuracy.
Report affected-source and per-source retention alongside the overall mean.
These arithmetic checks read existing artifacts; no training was performed.

Gradient audit [derived from runner]: inner Adam updates are not differentiated.
For adapted satellite phi'=U(phi,theta), with core parameters theta unchanged
inside the episode, the exact outer derivative is
`dJ/dtheta = partial_theta J + (dphi'/dtheta)^T partial_phi' J`.
The runner keeps the first term and omits the second. It also approximates the
initial-satellite adaptation Jacobian by identity. This is the declared
first-order approximation, not an implementation/spec mismatch. In particular,
it omits the explicit credit for how core changes alter satellite learning.

Prior-work search, 2026-10-28: “On First-Order Meta-Learning Algorithms”.
[Nichol et al., 2018](https://arxiv.org/abs/1803.02999) [abstract] identifies
FOMAML as omitting second derivatives. Its results do not establish that the
approximation is responsible for this pilot's forgetting. A small development
episode comparing exact and approximate gradients, with a finite-difference
check, could discriminate that hypothesis before a larger run [proposed].

### Gradient diagnostic follow-up: scope and usable adaptation

2026-10-28. Read the meta pilot's added gradient diagnostic, its runner and
saved results [read: project artifacts]. The finite-difference checks support
the exact derivative implementation for the tested five-step SGD episodes.
First-order and exact directions differ substantially there. This does not
directly quantify the error of the original ten-step Adam episodes.

The report's claim that smooth activations are necessary is too strong.
ReLU gradient steps can make the adapted objective discontinuous across
activation boundaries, but finite-scale differences alone do not isolate
that cause or prove unusable meta-gradients. An activation-change or smoothing
control is required. Search/read, 2026-10-28: direct reading of
[Finn et al., 2017](https://proceedings.mlr.press/v70/finn17a/finn17a.pdf),
sections 5.1–5.2 [read]. Their successful MAML models use ReLU; this is a
counterexample to universal necessity, not a prediction for SRI.

Preservation interpretation [derived from saved curves]: distinguish final
source accuracy from the loss caused by adaptation, since an outer update
changes pre-adaptation source accuracy too. From the meta checkpoint at
outer length 0.03, exact updates reduce the before-minus-after source drop
by approximately 0.019 and 0.047 for inner SGD rates 0.003 and 0.01.
At 0.01, the first-order direction also reduces that drop slightly (0.008),
despite lowering final source accuracy. Thus “only exact reduces forgetting”
depends on which preservation quantity is intended.

Next obligation [proposed]: demonstrate useful support fitting and query
improvement under a stable, bounded inner adaptation on development tasks.
Then audit gradients for that same update rule. Chance query accuracy after
five steps does not distinguish no learning from memorization or preliminary
progress; record before/after support and query losses separately. Keeping
parameters unchanged must not satisfy a claimed learning-efficiency success.

### Working Adam episode: interpretation update

2026-10-28. Read the episode audit and selected-Adam gradient audit in the
meta pilot report, the audit runner, and its saved results [read: project
artifacts]. Support and query now both improve during adaptation, while old
source performance declines. This resolves the earlier acquisition-versus-
generalization ambiguity in favour of useful learning with interference.
Requiring an inner episode to already meet the final retention constraint
would prejudge the trade-off the outer learner is supposed to improve;
stable useful acquisition is the prerequisite, preservation is an outcome.

Metric correction [derived]: query-after minus query-before measures the
effect of adaptation, not learning efficiency by itself. Raising initial
accuracy can shrink that gain without harming performance at a fixed budget.
Saved fresh-episode means at outer step length 0.03 from ordinary: exact
raises query-before by 0.0123 and changes query-after by -0.0008; first-order
raises query-before by 0.0139 and query-after by 0.0012. Both lower the
combined fresh objective by about 0.35–0.36. From meta, the same steps lower
query-after and raise the combined objective. These are single-step,
development-only observations, not evidence of sustained improvement.

Numerical language [derived]: along the negative exact gradient, its
infinitesimal directional derivative cannot be positive. A positive central
finite difference at a specified epsilon is a finite-scale disagreement,
not a positive exact slope. The report's local checks and roughness evidence
must be distinguished accordingly.

Decision implication [hypothesis]: no demonstrated benefit currently pays
for the approximately threefold exact-gradient cost. This does not establish
that full training with exact gradients would fail. Any subsequent comparison
should prioritize absolute query accuracy at matched adaptation cost and
per-source preservation, rather than positive adaptation gain alone.

### Separate context-sensitive perception from relational use

2026-10-28. User distinguishes learning to group writing systems from pressure
to use core relations. Interpretation [hypothesis]: perceptual grouping should
learn which inputs can share interpretation machinery and which need private
adjustments, without equating visual similarity with numeric equivalence.
This is separate from reusing a learned relation to determine a new answer.

Targeted search, 2026-10-28: “Domain Separation Networks”; “sparsely gated
mixture of experts”; “Abstractors relational cross attention”.
[Bousmalis et al., 2016](https://arxiv.org/abs/1608.06019) [abstract] studies
shared/private domain representations;
[Shazeer et al., 2017](https://arxiv.org/abs/1701.06538) [abstract] learns
sparse expert routing. These are relevant baselines for conditional perception,
not proofs of automatic writing-family discovery or preserved continual
learning [analogy]. Learned routing must be distinguished from supplying a
separate module and identity for every writing system.

Implementation deduction [derived]: `MultiSRI.hessian` does not use the learned
successor A,t or the larger-relation direction. At fixed remaining parameters,
changing those relation parameters cannot change the label prediction. Their
training losses can shape the representations indirectly, but the present
classifier never executes those operators to obtain labels. That distinction
must precede another experiment on pressure to use relations. It does not
rule out compiled relational knowledge in a classifier.

SRI hypothesis [untested]: make relational predictions participate in inference
over available anchors/context and unresolved interpretations, then evaluate
under withheld local mappings. Evidence must actually identify the target;
unknown arbitrary symbols cannot be decoded without grounding. Compare intact
and perturbed learned relations at fixed perceptual interfaces, on held-out
compositions, to test dependence on correct relations rather than simply
passing through the core. Relation geometry alone is not this test.

Novelty qualification: [Altabaa et al., ICLR 2024](https://arxiv.org/abs/2304.00195)
[abstract] explicitly separates relational information from object features.
Thus pressure for relational use already has established approaches; a new SRI
mechanism would need comparison with these, not a claim that the general
problem has no standard solutions.

### User direction: coherent teaching and confidence-dependent inference

2026-10-28. User proposes teaching counts in order, grouped by language, and
invoking core relations when a direct answer lacks confidence. This changes
both the experience supplied to the learner and its allocation of inference.
It does not select a frozen contract or authorize a new training campaign.

Curriculum hypothesis [untested]: use coherent within-language counting
episodes, with object additions/removals linked to successive spoken/written
forms, then bridge languages through common quantities and relations.
Provide language context initially if needed and label that supervision
explicitly; do not claim it was discovered. Compare coherent episodes with
shuffled presentations of the same examples, targets, context and compute to
separate scheduling effects from additional grounding information. Mix in
retention checks; sequential language blocks alone can still cause forgetting.

Confidence hypothesis [untested]: obtain an initial prediction and an estimate
of its reliability, then use executable core relations to revise uncertain
interpretations when available context supports a better answer. A controller
should ultimately estimate expected improvement relative to computational
cost: uncertainty alone cannot tell whether additional processing will help.
Ambiguous evidence may warrant seeking more evidence rather than more steps.

The current deterministic quadratic solve has no separate deliberation
operation: additional settling to the same equilibrium cannot invoke an
unused successor. Relation-conditioned inference and a stopping/routing rule
must therefore be defined. During development, compare initial and revised
predictions against actual outcomes, including cases where revision harms
an initially correct answer. Calibrate confidence on held-out contexts;
cosine similarity and low equilibrium residual are not probabilities of
semantic correctness [derived design requirements].

Source search, 2026-10-28: “Sarnecka Carey counting cardinal principle”;
“Adaptive Computation Time Graves”; “Guo calibration neural networks”.
[Sarnecka and Carey, 2008](https://escholarship.org/uc/item/8dh3972h)
[abstract and selected source passages] separates count-list knowledge from
cardinal understanding; it does not establish the proposed language-block
curriculum. [Graves, 2016](https://arxiv.org/abs/1603.08983) [abstract] learns
adaptive computation time, not SRI's proposed relation-based controller.
[Guo et al., 2017](https://arxiv.org/abs/1706.04599) [abstract] studies confidence
calibration; it does not establish that calibration transfers to unseen scripts.
These are established ingredients, not evidence of a novel or successful SRI
mechanism.

No training experiment was run for this note. Any implementation must
declare construction, proposal/search, fitting, validation and deployment
complexity and measure their costs under `computation-reporting.md`.
