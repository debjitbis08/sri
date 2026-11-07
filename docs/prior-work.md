# Prior work

Living survey of established methods related to this project, maintained under
[literature review for contracts](literature-review.md). It covers the
continuous three-module line: the three-module, graded, local consolidation and
local contrastive studies. The discrete core-mediation and graph-reachability
studies are not yet reviewed.

## Search record and evidence levels

2026-11-07: [learning to apply relations](relations-application-notes.md) follows relations-as-objects v1:
fresh-code introduction episodes, query-conditioned use of partner evidence, and counterfactual application
checks. Local diagnosis **[read: local code and results]**; design arguments **[derived]** and proposed reader
**[analogy]**; MAML and Neural LP **[abstract]**, with searches and limits recorded in the note. No new run or contract.

2026-11-07: [rule application v1](rule-application-v1.md) (draft) maps the design to the Edge Transformer (pair states,
triangular attention; [Bergen et al. 2021](https://arxiv.org/abs/2112.00578)) looped with shared weights ([Looped
Transformers, Fan et al. 2024](https://arxiv.org/abs/2409.15647)), with rules told as formulas as in RuleTaker
([Clark et al. 2020](https://arxiv.org/abs/2002.05867)) and size generalization as in [Neural Logic
Machines](https://arxiv.org/abs/1904.11694) (all **[abstract]**). ∂ILP, NTP and Neural LP **[familiar]** are the
excluded alternative with modus ponens built in. Queries recorded in the contract.

2026-11-07: [amortized introduction v1](amortized-introduction-v1.md) (draft) maps the introduction encoder to
amortized inference and the Conditional Neural Process (set encoder with attention pooling, Set Transformer PMA)
**[familiar]**; zero-shot KG relations from an ontological schema, [OntoZSL](https://arxiv.org/abs/2102.07339)
**[abstract]**, as the established counterpart (named axioms with text there, arbitrary learned tokens here);
regression onto stored codes as [Attentive Mimicking](https://arxiv.org/abs/1904.01617) **[abstract]**; and the
meta-learning memorization failure ([Yin et al. 2020](https://arxiv.org/abs/1912.03820) **[abstract]**: a task
identifiable without its data is solved without adapting), answered by virtual relations (task augmentation,
**[familiar]**) and a held-out partner split. Queries recorded in the contract.

2026-11-05: [drift remedies v1](drift-remedies-v1.md) (draft) maps the moving-representation remedies onto the continual
learning taxonomy (replay, regularization, parameter isolation): EWC and synaptic intelligence, latent and generative
replay, residual adapters; plus drift compensation, relative representations, shift detection, hubness and savings.
Sources **[abstract]** or search excerpts; section below.

2026-11-05: [moving-representation notes](moving-representation-notes.md) separate coherent relabelling (the one shift internal checks cannot see) from unconstrained movement of kinds far from the loss, and place every remedy in one of three families: stop the movement, keep evidence, or make consumers coordinate-independent. Reviewed twice the same day: senses are frozen and memory keyed on them does not drift, so the problem sits in the core's object encoder; within-kind resolution survives in the codes but comes mostly from the frozen senses; context normalization half helps and the rest of the bend is not affine; routing by sense code is partly tested, with known failures on blended streams. Local results **[read: local results]**; representational-drift, penalty and adapter sources **[familiar]**, no search run; the three-family argument, the savings test, the core frame with per-kind registration and the population-level drift signal **[derived]**; the collision check **[proposal]**. Section below. No experiment run.

2026-11-05: [consistency critic v1](consistency-critic-v1.md) maps the critic to dependency networks and masked
prediction, probe selection to top-k gating and active feature acquisition, repair to BeliefBank/ConCoRD (symbolic
solvers, given relations), self-training and Learning without Forgetting, and asking to active learning and its sampling
bias. Sources **[abstract]** from searches recorded in the section below; implications **[derived]**.

2026-11-05: [learned consistency critic](learned-consistency-critic-notes.md)
records the proposed operator-output monitor and offline encoder repair.
SPEN and confidence calibration are **[abstract]** analogues; limits of pair-only
inputs, anchor preservation and positive-only energy training are **[derived]**.
Masked conditional prediction is an untested alternative. No experiment run.

2026-11-05: [equality goal analysis](equality-goal-analysis.md) separates learned
pair classification from geometric alignment and numerical substitutability.
Implementation implications are **[derived]**, v2 observations **[read: local
results]**, and Khosla et al.'s supervised contrastive learning **[abstract]**.
Reader diagnostics and alignment controls are proposals, not tested remedies.
No new contract or training run.

2026-11-04: [shared output v1 contract](shared-output-v1-contract.md) (draft) maps a query
encoder with one shared candidate scorer to 1-N knowledge-graph scoring (ConvE), cross-encoders
and energy-based scoring, next-item scoring to contrastive predictive coding, abstention to
selective classification and open-set rejection, and naming-only transfer to acquired
equivalence and SCAN's add-primitive split. All sources are **[familiar]**; no new search was
run. The prediction that a shared scorer alone does not make a name substitutable is
**[derived]**. Section below.

2026-11-04: [shared relational expression interface](relational-expression-core-design.md)
maps the existing tasks to relational queries and distinguishes learned grounding
from supplied inverse/composition rules. miniKanren's project introduction is
**[read]**; λProlog's project search summary and the LTN, Neural Theorem Provers,
and TensorLog abstracts are **[abstract]**. Finite-query semantics and cost bounds
are **[derived]**; improved transfer/retention through shared scoring is
**[analogy]**, not an established result. The note proposes an unchanged-comparator
baseline before a naming-only transfer contract. No training was run.

2026-11-04: [mathematical analysis of the relearning loss](relearning-loss-analysis-v1.md)
derives the distance-loss gradients, tests scale invariance counterexamples, and
proposes a local geometry-constrained update. NormFace is **[abstract]** and OGD
is **[read: abstract and section 3 excerpt]**; transfer to this architecture is
**[analogy]**, not an established remedy. Deterministic toy checks and their
separate computation receipt accompany the derivations. No training was run.

2026-11-03: [general frozen encoder](general-frozen-encoder.md) records Baradad et al. 2021, Asano et al. 2019, Yosinski et al. 2014 and DINOv2 at **[abstract]** level on which training images give general features; a pretrained general encoder is named there as a cheaper alternative to test.

2026-11-03: [reading notes on memory through a learning encoder](natural-memory-prior-work.md) cover MoCo and SDC at **[read]** level, self-healing codes through a summarizing tool, and Expert Gate, Progressive Networks and REMIND at **[abstract]** level, for the [natural-image memory pilot](natural-memory-v1-pilot.md).

2026-11-01: [completed core versus language models](completed-core-versus-language-models.md)
compares the proposed system with documented dense, MoE and test-time-memory
model families. Llama 3, DeepSeek-V3, Titans and ATLAS are reviewed at
**[abstract]** level. Feature-level sparsity is distinguished from expert routing;
stable continual learning and modular reuse remain hypotheses, not established
advantages or novelty claims.

2026-11-01: [core generality, context and sensory features](core-generality-context-and-perception.md)
consolidates the exploratory discussion of code spaces, sparse readers,
counting specialization, grammar, geometric relations and perception. It links
the topic-specific theory and source reviews, records the user's context/memory
and feature-extraction observations, and distinguishes architectural possibility
from learned capability. Geometry and perception claims are task-sufficiency
arguments (**[derived]**), not a survey establishing solved perception. No new
experiment or architecture decision is recorded.

On 2026-10-29, [two-module graded expressivity analysis](two-module-graded-expressivity-analysis.md)
compared reciprocal quadratic modules with linear bottleneck networks, and
nonlinear block recurrence with ordinary RNNs and deep equilibrium models.
The affine compilation, equilibrium communication-rank bound, and finite-step
unrolling equivalence are **[derived]**. Hopfield (1984), abstract and model
introduction **[read]**, and Bai et al. (2019) **[abstract]** provide architectural
context; neither source establishes a benefit from inward-graded coupling.
The note records assumptions, proofs, primary links, cheaper alternatives and
theoretical costs; it makes no measured speed or learning claim.

On 2026-10-29, [conceptual instinct notes](conceptual-instinct-notes.md)
reviewed predictive-coding novelty detection, compatibility-energy learning,
amortized inference and BALD for the number-construction discussion. The notes
separate learned inconsistency detection from answer reliability and question
selection; the proposed mechanism is untested and is not a novelty claim.

On 2026-10-27, user-supplied readings by Rudra (2022), Morton (2012), and
Abbott–Zardini (2026) were reviewed in selected sections for the edge-algebra
discussion. [Detailed notes and reading scope](edge-algebra-notes.md) record
their complementary roles: efficient structured linear maps, compositional
tensor diagrams, and algebraic model construction/compilation. The notes
separate source findings from SRI-specific stateful-operator proposals.

Searches on 2026-10-26 covered GLOM, multi-view and multimodal predictive
coding, prospective configuration, equilibrium-propagation bias, predictive
coding as associative memory, coupled dictionary learning, representation
alignment and stitching, and critiques of importance-based continual learning.
No search found a published model combining all this project's ingredients,
which is not evidence of novelty.

A second round on 2026-10-26, for [correlated nuisance v1](correlated-nuisance-v1-contract.md),
covered early versus late fusion in multimodal learning, reduced-rank
regression under distribution shift, and spurious correlations in linear
regression, and re-checked Kamai et al. 2026 at its arXiv page.

Each claim below carries a tag:

- **[read]**: the cited section was read in the source;
- **[abstract]**: abstract, proceedings page or search summary only;
- **[familiar]**: a standard result stated from prior knowledge, not re-read
  for this survey;
- **[derived]**: derived for this project's model, not taken from a source;
- **[analogy]**: a structural resemblance, not an established result about
  this model.

Entries separate **shown** (what the source demonstrated), **differs** (how
this architecture differs) and **hypothesis** (what the project infers, still
untested).

## Fan-in pressure

2026-11-01: [grammar learning extension notes](grammar-learning-extension-notes.md)
derive the two-item history obstruction and a distinguishable-prefix memory lower
bound for typed brackets (**[derived]**). Primary abstracts on finite-precision
RNNs, self-attention limits, differentiable stacks, BLiMP and COGS are recorded
as **[abstract]**, with architecture differences explicit. Contextual state and
appropriate outputs are distinguished from optional structured memory and the
untested sparse-disruption hypothesis. No scaling or grammar-learning result
is claimed.

2026-11-01: [sparse-reader hypothesis properties](sparse-reader-hypothesis-properties.md)
derives the exact first-layer-sparse/two-ReLU reader form, dense containment,
nullspace and mask-matching rank tests, finite line-region bounds, a realizable
three-cycle, coordinate reparameterizations, parameter/mask counts and conditional
Lipschitz bounds (**[derived]**). Bartlett et al. (2019), Theorem 7 statement,
was **[read]** for general capacity upper bounds; Montufar et al. (2014) was
reviewed at **[abstract]** level. The note distinguishes fixed codes from learned
encoders and representability from optimizer preference; no checkpoint audit or
generalization certificate is claimed.

2026-11-01: [counting specialization and generalization](counting-specialization-generalization.md)
audits earlier ordinal heads versus current generic codes and counting-specific
evidence. It records **[abstract]** reviews of capacity bounds, PAC-Bayes,
multi-task bias learning, domain adaptation and random-label fitting. A finite
class bound, evidence-ambiguity test, code-collision obstruction and scalar-order
counterexample are **[derived]** with explicit assumptions. Proposed whole-task
holdouts are unrun; no generalization certificate is claimed.

2026-11-01: [code space versus sparse readers](code-space-reader-analysis.md)
separates representation accessibility from reader effects and their interaction.
Orthogonal-basis invariance of dense readouts and basis dependence of sparse
readouts are **[derived]**. Primary abstracts of dropout, disentanglement limits
and random features were reviewed (**[abstract]**); their relevance to this
network remains **[analogy]**. The note proposes frozen-code reader refits and
rotations before new code-family training, and records alternatives, confounds
and computation requirements. No new results or contract are claimed.

2026-10-30 user-supplied reading: [forget-and-relearn and iterated learning](forget-relearn-iterated-learning-notes.md)
records selected primary-source sections of Zhou et al. (2022), Kirby et al.
(2008), Ren et al. (2020), and Li and Bowling (2019). This supplies established
context missing from the reset interpretation. Immediate robustness, recovery
speed and retained-encoder adaptation are distinct hypotheses. A simple
attenuation/recovery recurrence is **[derived]** under fixed-mode assumptions;
neither it nor the cited papers guarantees this network's transfer. Reading
scope, differences, cheaper controls and the limit of old-task relearning
during a form-free delay are recorded in the note.

2026-10-30 after reset controls: [code accessibility under changing readers](rigl-after-reset-analysis.md)
withdraws the P-centered explanation and distinguishes distributed scalar codes
from coordinate-concentrated wiring. The single-coordinate affine-risk identity
is **[derived]**; faster readout relearning is an untested hypothesis. Random
rewiring's substantial transfer does not establish equivalence to RigL, and
static unit-removal failure does not rule out dynamic attrition interactions.

2026-10-30 after timing: [RigL resets and predictor attrition](rigl-after-timing-analysis.md)
revises the coordinate-choice explanation following the failed timing controls.
It derives k=1's gradient-conditioned resets and a condition for global ReLU
inactivity, and audits all ten final RigL checkpoints. Selective forgetting and
predictor simplification remain hypotheses; final structural inactivity does
not identify the cause of transfer. Proposed locus/reset controls are not yet
frozen contracts or new runs.

2026-10-30 revision review: [decaying-reader theory](decaying-readers-theory.md)
supersedes external-store critiques for proposal revision `217701a`. It derives
the decay filter, a positive slow/fast handoff result and its timescale,
functional-forgetting conditions, a shared-ReLU rescaling loophole, and a
conditional-expectation limit for proposed internal replay. Fast-weight,
L2-SP and decoupled-decay abstracts were checked. It also corrects fixed75's
residual units: they were randomly sparse, not dense. Proofs are conditional,
with separate fixed-fixture checks and computation receipts; no neural
consolidation or transfer improvement is newly established.

2026-10-30 mathematical follow-up: [transfer and retention theory](fan-in-memory-theory.md)
gives conditional proofs for grounded-graph identifiability and sharp resistance
bounds, functional-bypass margins, scale-invariant quadratic growth, optimal-design
memory scores, quadratic interference and finite binding information. All are
**[derived]**, with numerical identity checks and separate computation receipts.
Graph Gaussian fields (Zhu et al.), optimal design (Lu and Pong), and GEM provide
established context; reading scopes and limits are recorded in the note. The
grounded scalar model is a stronger assumption than the pilot's generic readers,
not an established description of their learned dynamics.

2026-10-30 follow-up: [fan-in and limited-memory analytical review](fan-in-memory-analysis.md)
records the fixed-coordinate replication and reviews the planned decaying store.
The reader derivative restriction, k=1's constant full reconsideration rate,
FIFO replay-age law and retrieval/training separation are **[derived]** from
local code and stated assumptions. Adaptive alignment is a hypothesis, not an
identified mechanism. RigL, tiny replay, kNN-LM and grokking abstracts were
checked; selected sections of Kumaran et al. (2016) were read. The note separates
their findings from predictions for this network and identifies replay-refresh,
withheld-label and delay-protocol ambiguities before a memory contract is frozen.

2026-10-29: design review for the
[fan-in pressure proposal](fan-in-pressure-proposal.md), which replaced a
bounded learnable locality proposal the same day. Sources opened directly:
the Conv1d operator documentation and the four paper abstracts below. This
is a targeted component review, not a novelty search. The first number-network
study uses fixed random fan-in limits; the locality entries below bear on its
deferred ordered-input variant.

- **Fan-in limit as pressure [hypothesis].** A fan-in-limited network is a
  special case of the dense one **[derived]**, so the claim is about what
  training finds, not what can be represented. A dense layer can set weights
  to zero but has no reason to. Pruning, sparse-training and random-sparsity
  results **[familiar; not read today]** show sparse networks matching dense
  accuracy at moderate sparsity; none known to us shows better systematic
  transfer. The expected outcome is therefore no gain, with reader limits the
  likeliest exception. DropConnect, L1 and prune-after-training are the
  stage-2 controls for random, soft and late pressure.

- **Convolution and local connectivity.**
  [PyTorch Conv1d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv1d.html)
  **[read: operator definition]** applies a shared kernel across positions.
  Untied local weights are a locally connected layer **[familiar]**. The
  proposal instead learns static per-unit sampling positions within hard
  intervals. Locality alone is not convolution **[derived]**. Ordinary
  convolution and full-window untied local weights are required controls.
- **Learned kernel positions.**
  [Khalfaoui-Hassani, Pellegrini and Masquelier, DCLS: beyond bilinear interpolation (2023)](https://arxiv.org/abs/2306.00817)
  **[abstract]** learns the positions of a convolution kernel's nonzero
  elements by interpolation; the original bilinear version sees only the
  nearest sites, and longer-range (Gaussian) interpolation improves
  ImageNet-1k classification. This is the closest established mechanism.
  It differs in tied weights and 2D tasks. With linear interpolation,
  position gradients see only adjacent sites **[derived]**, so small
  learned displacements are the expected outcome.
- **Learned sparse connectivity.**
  [Evci et al., Rigging the Lottery (ICML 2020)](https://arxiv.org/abs/1911.11134)
  **[abstract]** trains sparse networks at fixed density, updating topology
  by magnitude and occasional gradients, and reports better accuracy per
  training cost than earlier sparse methods. Magnitude pruning with
  fine-tuning and sparse evolutionary training are established
  alternatives **[familiar]**. At an equal edge budget the candidate and a
  pruned full window have the same compiled family **[derived]**, so the
  pruned window is the decisive control.
- **One shared coordinate read by every relation** (added 2026-10-30, for the
  fan-in pilot's fixed-coordinate arm). This is the accumulator or mental-number-line
  account of magnitude (Gallistel and Gelman; Dehaene) **[familiar]**, and in learning
  terms a one-dimensional bottleneck or ordinal latent score **[familiar]**. Predicts
  alignment of the code on one axis, at a cost to tasks that need other axes. It is
  also the form of the linear `larger` head the project removed **[derived]**.
- **Gradual constraints** (added 2026-10-30, for the timing arms). Curriculum and
  continuation methods (Bengio et al. 2009) and gradual pruning schedules (Zhu and Gupta
  2017) **[familiar]** impose a constraint progressively. They predict that what is learned
  early is kept better than under a constraint present from the start. Neither speaks to
  transfer.
- **Learned sampling.**
  [Dai et al., Deformable Convolutional Networks (2017)](https://arxiv.org/abs/1703.06211)
  **[abstract]** learns sampling offsets from task supervision and reports
  improved detection/segmentation. The proposed bounded static sampling is
  related **[analogy]**, but differs in bounds, unit-specific parameters and
  scalar shrinking layers. Those vision results do not predict a gain here.
- **Differentiable spatial transformation.**
  [Jaderberg et al., Spatial Transformer Networks (2015)](https://arxiv.org/abs/1506.02025)
  **[abstract]** learns input-conditioned transformations and reports
  invariance to several transformation families. The proposal uses static
  positions shared across examples, so it cannot claim that per-example
  adaptation **[derived]**. Differentiable sampling is the connection
  **[analogy]**, not equivalent inference or an equivalent estimator.
- **Earlier project evidence [read: local analysis/results].**
  [Locality adaptation analysis](locality-adaptation-analysis.md) proves
  linear approximation gaps and limits of phase adaptation.
  [V2](locality-adaptation-v2-results.md) recovered shifted structure but
  missed its joint criterion through aligned reconstruction noninferiority,
  with substantial selection cost. This motivates explicit sample-budget
  and cost comparisons; it does not validate nonlinear learned neighborhoods.
- **Predictions and failure modes [derived].** Fixed learned interpolation
  can be compiled into sparse weighted edges. A linear stack remains affine;
  bounded paths cannot access excluded information. Smaller dimensionality
  alone neither bounds information in bits nor ensures nuisance removal.
  Meaningful ordering, predictive supervision, nonlinear responses and
  transfer measurements are distinct obligations. Saturated location
  parameters, coincident samples and boundary loss require diagnostics.

The proposed contribution is an internal test of whether limiting how many
units each unit combines arranges the shared code.
No novelty or efficiency advantage is established. Pruned full local
windows, ordinary CNNs, fixed local layers, dynamic sparse training and
dense bottlenecks are cheaper or simpler alternatives whose costs must be
measured rather than assumed.

## Where this project sits

- The energy is a positive-definite quadratic in the activities. It defines a
  Gaussian over activities given parameters, and exact masked completion is
  that Gaussian's conditional mean [derived].
- The clamped-energy update has the same local form as the predictive coding
  learning rule [familiar]. That is a similarity of update equations, not an
  equivalence of statistical estimators. The normalized negative
  log-likelihood is E − ½ log det Q up to constants, and Q depends on the
  parameters, so minimizing clamped E is not maximum likelihood. Marginal
  likelihood over latents would add further terms [derived].
- The fixture is not Gaussian: it has discrete structure identities and
  uniform amplitudes [derived].
- **Hypothesis:** the combination of role-free modules, learned translations at
  several levels and purely local learning is less explored than each part.
  GLOM is the nearest conceptual neighbour [analogy].

## Energy, inference and learning rules

- **Predictive coding networks.**
  - Shown: hierarchical generative models whose activities settle by
    descending summed squared prediction errors, with local error × activity
    weight updates. Rao & Ballard 1999; Whittington & Bogacz 2017 [familiar].
    A 2024 [tutorial and survey](https://arxiv.org/pdf/2407.04117) frames them
    as latent-variable models trained by EM-like alternation [abstract]. See
    also [Millidge et al., IJCAI 2022](https://www.ijcai.org/proceedings/2022/0774.pdf)
    [abstract].
  - Differs: those models are usually single-stream, directed and generative.
    Here the energy also has symmetric translation terms between modules, and
    learning uses the clamped energy rather than a normalized likelihood.
- **Factor analysis, probabilistic PCA and probabilistic CCA.**
  - Shown: maximum-likelihood linear Gaussian latent models with closed-form or
    EM solutions for one view ([Tipping & Bishop 1999](https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/)
    [abstract]) and two views ([Bach & Jordan 2005](https://people.eecs.berkeley.edu/~jordan/papers/688.pdf)
    [familiar]). Latents are identified only up to an invertible transform, or
    a rotation under an isotropic prior [familiar].
  - Differs: these are statistical estimators. This project's training
    procedure is not shown to be equivalent to any of them.
  - Use: relevant **comparators** for multi-view linear data, not equivalents.
- **Prospective configuration.**
  - Shown: settling activities before plasticity is argued to reduce
    interference and help in online, few-shot and continual settings
    (Song et al., *Nature Neuroscience* 2024: [author manuscript](https://www.mrcbndu.ox.ac.uk/sites/default/files/uploads/PMC7615830.pdf),
    [Oxford summary](https://www.ox.ac.uk/news/2024-01-03-new-research-shows-way-brain-learns-different-and-better-way-artificial-intelligence);
    [Millidge et al., ICLR 2023](https://arxiv.org/pdf/2207.12316)) [abstract].
  - Differs: those comparisons are with backpropagation on supervised tasks,
    not with an equilibrium-propagation masked loss on module completion.
  - Hypothesis: the clamped-energy rule's advantage in local contrastive v1 may
    be related. That is context, not an established explanation. The classic
    generative-versus-discriminative small-sample result (Ng & Jordan 2002)
    [familiar] is likewise context only.
- **Equilibrium propagation.**
  - Shown: the two-phase rule is EP (Scellier & Bengio 2017), a descendant of
    contrastive Hebbian learning (Movellan 1991) [familiar]. Finite nudging
    biases the gradient estimate, and symmetric ± nudging reduces the bias
    ([Laborieux et al. 2021](https://arxiv.org/pdf/2006.03824)) [abstract].
    Holomorphic EP removes finite-nudge bias in principle, while weight
    asymmetry remains a separate obstacle ([Zenke lab, ICLR 2024](https://zenkelab.org/?p=1864))
    [abstract]. EP has trained predictive coding networks on ImageNet
    ([2026 preprint](https://arxiv.org/pdf/2606.03584)) [abstract; not verified].
  - Differs: none material. Local contrastive v1 used symmetric EP.
  - Note: the finite-β deviation in local contrastive v1's verification is
    consistent with this known bias [derived from that run].
- **Bidirectional predictive coding.**
  - Shown: one energy supporting generative and discriminative inference,
    with reported gains on multimodal learning and inference with missing
    information ([Oliviers, Tang & Bogacz, ICLR 2026](https://arxiv.org/abs/2505.23415))
    [abstract].
  - Hypothesis: a candidate established alternative for the energy versus
    masked-prediction objective question.
- **Associative memory by predictive coding.**
  - Shown: retrieval of stored items from partial or corrupted inputs by
    energy minimization; the fully connected version is reported not to
    generalize to unseen complex images ([Salvatori et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/1fb36c4ccf88f7e67ead155496f02338-Abstract.html))
    [abstract].
  - Differs: single network and memorization of items. Here, whole-module
    masks on held-out variations.

## Modules, coupling and translations

- **GLOM** ([Hinton 2021](https://arxiv.org/pdf/2102.12627)).
  - Shown, as a proposal: columns with several levels, and attention-weighted
    averaging between columns at the same level to form "islands of agreement"
    [abstract]. Implementations struggle to obtain reliable agreement, and a
    contrastive agreement term is proposed in response ([Zeng et al., ICCV 2025](https://mlanthology.org/iccv/2025/zeng2025iccv-agreement/))
    [abstract]. See also [testing GLOM on ambiguous parts](https://arxiv.org/pdf/2211.16564)
    and an [unofficial implementation](https://github.com/lucidrains/glom-pytorch).
  - Differs: GLOM columns share weights and interact by near-identity
    averaging over spatial positions. Here, modules have private codes and
    learned translations per edge and layer, with quadratic coupling.
  - Hypothesis: agreement between independently coded modules may not emerge
    under identity coupling. GLOM's difficulty does not predict this project's
    F/O/L ranking; it motivates testing it.
- **Coupled and semi-coupled dictionary learning.**
  - Shown: coupled dictionary pairs for two domains. Early versions force
    identical codes (Yang et al. 2010) [familiar]. Semi-coupled versions learn
    a linear map between codes (Wang et al., CVPR 2012) [familiar], motivated by
    the limits of identical codes. Later versions learn map and dictionaries
    jointly ([cross-domain example](https://arxiv.org/pdf/2101.02362);
    [Huang & Wang, ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Huang_Coupled_Dictionary_and_2013_ICCV_paper.pdf))
    [abstract].
  - Differs: two domains, sparse codes, non-local alternating optimization.
  - Hypothesis: the closest established analogue of fixed versus learned
    translations [analogy].
- **Representation alignment and model stitching.**
  - Shown: orthogonal (Procrustes) and affine maps between learned
    representations are compared routinely. Orthogonal maps preserve geometry;
    unconstrained maps fit better but can distort it; CCA-like maps can beat
    strict isometry for retrieval ([multi-way alignment, ICML 2026](https://arxiv.org/pdf/2602.06205);
    [stitcher families, 2026](https://arxiv.org/pdf/2601.21568)) [abstract].
    Hyperalignment aligns brain representations with Procrustes rotations
    (Haxby et al. 2011) [familiar].
  - Differs: post-hoc alignment of trained networks, fitted globally. Here the
    map is learned locally, online, while the modules learn.
  - Hypothesis: the rotation-only (O) mode is a local, online analogue of
    Procrustes alignment [analogy].
- **Multimodal predictive coding and shared latents.**
  - Shown: a shared top layer predicting visual and tactile hierarchies
    ([MuPNet, 2019](https://arxiv.org/pdf/1909.07201)) [abstract]. Shared and
    private latents in multimodal VAEs ([2024](https://arxiv.org/pdf/2403.06338))
    [abstract]. Multimodal deep Boltzmann machines (Srivastava & Salakhutdinov
    2012) and cross-stitch networks (Misra et al. 2016) [familiar].
  - Differs: designated modalities with a shared latent or learned mixing,
    trained non-locally, except MuPNet's predictive-coding updates.
  - Hypothesis: inner-only coupling resembles a shared top latent [analogy].
- **Cross-prediction versus alignment.**
  - Shown: in a linear spiked model, cross-modal alignment fails under strongly
    correlated cross-modal nuisance, and cross-prediction depends on
    source-modality quality. The paper gives a phase diagram of regimes where
    alignment, prediction, both or neither recover the signal, and reports
    support in nonlinear experiments ([Kamai et al. 2026](https://arxiv.org/abs/2606.11190))
    [abstract].
  - Differs: objectives for learned encoders, not local energy learning.
  - Hypothesis: the current fixture's nuisance is independent across modules,
    so a correlated-nuisance fixture is a cheap, theory-guided way to make
    objectives differ.
- **Early, intermediate and late fusion.**
  - Shown: multimodal surveys classify fusion as early (feature level), late
    (decision level) or hybrid and intermediate, and report trade-offs
    rather than a general ranking. Early fusion captures low-level
    interactions; late fusion tolerates missing modalities
    ([Baltrušaitis et al. 2017](https://arxiv.org/pdf/1705.09406);
    [Zhang et al. 2019](https://arxiv.org/pdf/1911.03977);
    [biomedical intermediate-fusion review, 2024](https://arxiv.org/abs/2408.02686))
    [abstract].
  - Differs: designated modalities trained end to end, and fusion is a
    combination step, not symmetric coupling at several layers.
  - Hypothesis: uniform versus inner coupling resembles early versus late
    fusion [analogy]. The surveys do not predict which wins here.
- **Rank constraints and spurious correlation.**
  - Shown: reduced-rank regression constrains a multivariate predictor's rank
    (Izenman 1975) [familiar]. Anchor regularization has been added to it for
    out-of-distribution robustness ([Durand et al., AISTATS 2025](https://proceedings.mlr.press/v258/durand25a.html))
    [abstract]. Invariant risk minimization (Arjovsky et al. 2019) and anchor
    regression (Rothenhäusler et al. 2021) target correlations that do not
    persist across environments [familiar]. In high-dimensional ridge
    regression, learned spurious correlation and in-distribution loss trade
    off, and the loss-minimizing regularization lies where learned spurious
    correlation is increasing ([Bombari & Mondelli 2025](https://arxiv.org/abs/2502.01347))
    [abstract].
  - Differs: estimators with explicit targets and, for invariance methods,
    environment labels. Here, local energy learning without environment
    labels.
  - Hypothesis: inner-only coupling acts like a rank-2 channel per module.
    It should filter nuisance only while signal dominates predictable
    variance [derived]; tested in correlated nuisance v1.
- **Thousand Brains / Monty.**
  - Shown: learning modules vote laterally on object identity and pose to
    reach consensus without a strict hierarchy ([Thousand-Brains systems, 2025](https://arxiv.org/pdf/2507.04494);
    [project overview](https://arxiv.org/pdf/2412.18354)) [abstract].
  - Differs: votes are discrete hypotheses with evidence, not continuous
    learned translations.

## Continual learning and consolidation

The [continuous-learning notes](continuous-learning-notes.md) record the
2026-10-27 reading on representation sharing versus interference (Holton et al.),
hybrid predictive coding, wake–sleep, generative replay, eligibility traces and
streaming learning components. Each entry separates reading depth, published
evidence, differences from SRI and untested program implications. The central
distinction is between making inference cheap and deciding which knowledge
to share or preserve; neither mechanism supplies the other automatically.

- **Synaptic Intelligence** ([Zenke, Poole & Ganguli 2017](https://arxiv.org/pdf/1703.04200)).
  - Shown [read, Section 3]: importance ω_k accumulates −g_k θ'_k along the
    trajectory. Ω_k = Σ ω_k / ((Δ_k)² + ξ) is updated at task ends. Protection
    is a **quadratic surrogate penalty** c Σ_k Ω_k (θ̃_k − θ_k)², which anchors
    parameters to their values at the end of the previous task.
  - Differs: local contrastive v1's arm 8 used an SI-**inspired** importance
    estimate to **scale post-Adam updates** by 1/(1 + λΩ). It has no anchor
    penalty, so it reduces plasticity rather than pulling parameters back.
    These are materially different protection mechanisms, and arm 8 was not a
    test of SI.
- **Evidence on importance regularizers.**
  - Shown [read, Appendix G.1 of [Bricken et al., ICLR 2023](https://arxiv.org/html/2303.11934v1)]:
    Hsu et al. 2018 reported that EWC, MAS and SI catastrophically forget
    class-incremental Split MNIST, and tuning the loss coefficient alone did not
    help EWC or SI. Bricken et al. traced this to models reaching 100% accuracy
    per split, which leaves no gradient for importance estimation. Scaling
    logits by β < 1 when estimating importance raised validation accuracy
    from 19.8% to 63.2% for EWC and from 19.7% to 35.8% for SI (L2 baseline
    36.8%). Their Split CIFAR10 results show the same pattern.
  - Shown: SI's link to the Fisher information, and its performance, are
    attributed to a previously unrecognized bias ([Benzing, AISTATS 2022](https://proceedings.mlr.press/v151/benzing22a.html))
    [abstract]. Training regimes that widen minima, through learning-rate
    decay, dropout and batch size, are reported to outperform other baselines
    ([Mirzadeh et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/518a38cc9a0173d0b2dc088166981cf8-Paper.pdf))
    [abstract].
  - Hypothesis: importance estimates can fail for identifiable reasons, such as
    vanishing gradients, and can be repaired. Simple slowing is a strong
    baseline. Local contrastive v1's result, where uniform slowing beat
    importance-scaled updates, is consistent with this, but its variant
    differs from SI. Further work should implement the anchor-penalty form,
    check whether importance estimates are starved of gradient, and use
    movement-matched controls.
- **Replay and sleep.** Complementary learning systems (McClelland et al.
  1995), wake-sleep (Hinton et al. 1995) and generative replay [familiar].
  [Foundations](foundations.md) commits to dreaming without a replay store,
  which places it in the generative-replay family. That family is not yet
  reviewed here.
- **Limited, decaying memory** (2026-10-30). The [limited-memory proposal](limited-memory-proposal.md#prior-work-and-expected-limits)
  places memory in the network's dense reader units, with weights that decay. It
  reviews fast and slow weights (Hinton and Plaut 1987; Ba et al. 2016, familiar),
  complementary learning systems, small episodic replay memories
  ([Chaudhry et al. 2019](https://arxiv.org/abs/1902.10486), abstract), nearest-neighbour
  memory beside a network ([Khandelwal et al. 2020](https://arxiv.org/abs/1911.00172),
  abstract) and generalization after memorization
  ([Power et al. 2022](https://arxiv.org/abs/2201.02177), abstract).

## Results that bear on this survey

- [Clamped-energy ablation v1](clamped-energy-ablation-v1-results.md):
  - The [derived] gauge prediction held. Inner-family fixed and rotation-only
    translations were indistinguishable, with no W/V clipping in that family.
  - The project hypothesis drawn from coupled dictionary learning and GLOM,
    that fixed identity coupling fails where codes are private, was
    consistent with the result. Uniform coupling with fixed identity
    translations failed at every budget, but that condition fixes all
    coupled layers, so the outer layer was not isolated.
  - Rotation-only alignment came within 1.31× of free translations, in line
    with the stitching literature's finding that isometries do much of the
    work [analogy]. The free translations kept unit gain on the pattern span
    and contracted nuisance directions, which resembles the regression-like
    maps of semi-coupled dictionary learning more than a pure rotation
    [analogy; post-hoc].

- [Correlated nuisance v1](correlated-nuisance-v1-results.md):
  - Inner-only coupling filtered weak correlated nuisance. In the strong
    regime its nuisance gain rose, the direction the variance-ranking
    hypothesis expects, but those conditions did not learn, so this is
    descriptive only. The inner channel did not behave like rank-2
    regression, which transmitted in the weak regime. A derived post-hoc
    explanation accounts for the comparator (it ranks centered variance);
    it does not establish how the energy objective selects directions.
  - Consistent with the fusion surveys' "trade-offs, not a ranking": the
    late-fusion-like inner channel was robust to the correlation shift but
    lost signal; the early-fusion-like surface channel was accurate in
    distribution and transmitted the shifted correlation [analogy].
  - As in the spurious-correlation literature, the learners that fit best
    in distribution absorbed the most non-persistent correlation.

- [Inner-channel diagnostic v1](inner-channel-diagnostic-v1-results.md):
  uncentered reduced-rank regression filtered weak correlated nuisance and
  kept the signal, as derived from second-moment ranking, while the
  centered version transmitted. The rank constraint itself is therefore not
  what costs inner coupling its signal. As in local contrastive v1, the
  exact masked-completion objective did not beat clamped energy.

- [Prior-strength training v1](prior-strength-training-v1-results.md):
  the activity priors play the role of the latent prior in factor analysis
  and probabilistic PCA, and their shrinkage behaved as ridge-like
  posterior shrinkage would predict. Weakening them restored completion
  gain without the weights drifting in scale, with weight decay as the
  remaining scale anchor.

## Relational abstraction and cross-domain transfer

Search record, 2026-10-26: searched MDP homomorphisms (Ravindran/Barto),
learning abstract models for transfer, Tolman-Eichenbaum structural
generalization, and bisimulation representations. Sources below inform the
[learning proposal](relational-abstraction-learning-proposal.md); this is a
targeted starting review, not a novelty survey.

- **MDP homomorphisms and learned abstraction.**
  - Shown: the [originator's research overview](https://www.cse.iitm.ac.in/~ravi/research-themes/mdp-homomorphisms/)
    describes homomorphisms as a framework for abstraction and transfer
    [read, overview]. [Online Abstraction with MDP Homomorphisms](https://arxiv.org/abs/1811.12929)
    reports a learned abstraction approach [abstract].
  - Differs: our initial fixture is a deterministic transition system with
    observable outcomes, without policy optimization or a claim of full MDP
    homomorphism learning.
  - Hypothesis: state/action mappings that preserve outcomes and transitions
    provide an operational target for SRI abstraction [analogy].
- **Tolman-Eichenbaum Machine (TEM).**
  - Shown: [Whittington et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/)
    learns shared graph structure across sensory remappings and uses it for
    prediction of unseen relationships [read, Results: Spatial and Relational
    Inferences; high-level algorithmic description].
  - Differs: TEM has architectural factorization of structural and sensory
    variables; it is not this project's local quotient learner. Its relation
    types have consistent meanings, whereas the proposal also tests learning
    action correspondence.
  - Hypothesis: separate structure from sensory bindings as a strong comparator
    [analogy]. This source also shows why prediction versus understanding is
    too strong a dichotomy: predictive training can support structural transfer
    [read, same sections].
- **Bisimulation and its limits.**
  - Shown: [Zhang et al. 2021](https://arxiv.org/abs/2006.10742) reports control
    representations invariant to distractors without pixel reconstruction
    [abstract]. [Zang et al. 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5a1667459d0cdeb2fe6b2f0dffc5cb9d-Abstract-Conference.html)
    identifies missing transitions as harmful for offline bisimulation
    estimation [abstract].
  - Differs: reward-based continuous control versus small observable transition
    systems with a declared outcome-query family.
  - Hypothesis: preserve distinguishing outcomes and audit coverage rather than
    treating unobserved transitions as agreement [analogy]. A constant map
    satisfies composition alone; observable consequences must rule it out
    [derived].
- **Cheaper alternatives and predictable results.**
  - Finite-state partition refinement and MDL/state merging are established
    reference families [familiar; implementation-specific primary-source review
    remains required before freezing]. Complete exact tables can be checked
    directly; doing so is a feasibility control, not evidence of learned
    abstraction [derived].
  - Arbitrary renaming can leave several equally valid alignments. Without
    calibration evidence distinguishing them, unique named-output transfer is
    unidentifiable [derived]. Evaluate modulo those symmetries or supply measured
    calibration equally to every arm.
  - The proposed contribution is an internal test of discovered behavioral
    reduction and transfer. Novelty, sleep-specific benefit, and an advantage
    over ordinary predictive model transfer remain unestablished.

## Linear action-conditioned dynamics and transfer

Search record, 2026-10-26, for the linear fixture of the revised
[relational proposal](relational-abstraction-learning-proposal.md): searched
joint learning of multiple linear systems, switched linear system
identification, Ho-Kalman/subspace identification finite-sample analysis, DMD
with control, Embed to Control, multi-step prediction training, provable
multi-task linear representation learning, and transfer across observation
spaces with shared latent dynamics. Targeted starting review, not a novelty
survey. Derived checks for the rotation-and-reflection pair were run as short
numerical scripts during this review and are not stored.

- **Linear system identification up to similarity.**
  - Shown: Ho-Kalman and subspace methods recover a state-space realization
    only up to an invertible state transformation [familiar]. Recent
    finite-sample analyses state their realization bounds up to similarity,
    and [He et al. 2025](https://arxiv.org/abs/2501.16639) bound open-loop
    MOESP/CVA-type estimates [abstract]. [Finite Sample Analysis of System Poles for Ho-Kalman Algorithm](https://arxiv.org/abs/2503.16331)
    reports that accurate pole estimation from Ho-Kalman can need sample sizes
    growing super-polynomially in the state-to-output dimension ratio
    [abstract].
  - Differs: single systems with known inputs, not several renderings of
    shared dynamics.
  - Hypothesis: evaluate learned dynamics only modulo one basis change shared
    by all actions, as the proposal already requires [familiar]. A target-only
    subspace method is a cheap direct baseline [familiar].
- **Joint learning of systems with shared structure.**
  - Shown: [Modi et al. 2024](https://arxiv.org/abs/2112.10955) model fully
    observed VAR(1) transition matrices as system-specific combinations of
    shared unknown basis matrices and prove that pooling improves estimation
    when the basis count is small relative to both d² and the number of
    systems. Misspecification erodes the gain, sharply so for unit-eigenvalue
    systems, and in their simulations the joint estimator does worse than
    individual ones when every system is misspecified [read, Sections 2-5].
  - Differs: the shared object there is the transition operator in fixed,
    observed coordinates. Here each domain has its own rendering and nuisance,
    so the target must also learn an encoder.
  - Hypothesis: broken-structure targets are the misspecified case; expect
    negative transfer there, worst with norm-preserving operators [analogy].
- **Provable representation transfer.**
  - Shown: with a shared k-dimensional linear representation,
    [Du et al. 2021](https://arxiv.org/abs/2002.09434) bound target risk by
    O(dk/(n₁T) + k/n₂), against O(d/n₂) without source tasks [abstract].
    [Tripuraneni et al. 2021](https://arxiv.org/abs/2002.11684) give
    estimators, lower bounds and a bias-variance analysis of transferring an
    imperfect representation [abstract].
  - Differs: there the representation is shared and the task head is new.
    Here the dynamics are shared and the encoder is new, so the roles are
    reversed.
  - Hypothesis: source benefit scales with the target parameters that
    transfer removes, and shrinks as the calibration budget grows [analogy;
    see the derived count below].
- **Action-conditioned linear latent models.**
  - Shown: [DMD with control](https://arxiv.org/abs/1409.6358) fits
    x' = Ax + Bu, separating unforced dynamics from additive actuation
    [abstract]. [Embed to Control](https://arxiv.org/abs/1506.07365) learns an
    encoder into a latent space with locally linear, action-conditioned
    dynamics and reports long-horizon prediction from images [abstract].
    Switched-linear identification alternates between per-mode models and
    mode assignment, for example [Lopes et al. 2023](https://repositorio.unb.br/handle/10482/47430),
    which keeps submodels in a common state basis [abstract].
  - Differs: modes are inferred there; here action identity is supplied.
    E2C is nonlinear and variational.
  - Hypothesis: additive control cannot make the transition matrix depend on
    the action. That matches the proposal's statement that an action
    indicator appended to a fixed linear map is insufficient, and it supports
    a bank of per-action operators in one shared coordinate system [derived].
- **Transfer across observation spaces.**
  - Shown: [Sun et al. 2022](https://arxiv.org/abs/2201.00248) transfer a
    latent dynamics model learned in a source task to a target task with a
    different observation space, as a regularizer on target representation
    learning, without an inter-task mapping [abstract].
  - Differs: reinforcement learning with a nonlinear encoder and reward.
  - Hypothesis: the closest published analogue of the proposal's frozen
    shared dynamics plus learned target binding; a strong conceptual
    comparator rather than one to implement [analogy].
- **Multi-step objectives.**
  - Shown: [Venkatraman et al. 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9590)
    note that one-step training lets errors compound in rollout and shift the
    input distribution. Their DaD reuses training trajectories to correct
    multi-step errors, with a bound relating training to multi-step error
    [abstract]. Equation-error versus simulation-error fitting is the
    corresponding distinction in system identification [familiar].
  - Differs: their models are fitted predictors, not equilibrium modules.
  - Hypothesis: the proposal's multi-step predictive training control
    corresponds to simulation-error or DaD-style training. A structural
    consolidation rule that selects on multi-step error must beat it to claim
    anything beyond the objective [analogy].
- **What follows without running, for the direct linear references.**
  - With a rotation R (angle not 0 or π) and a reflection Q fixed, the target
    binding constraint E x' = F_a E x determines the encoder on the excited
    observation span up to a single scalar; with one rotation alone, a
    rotation-and-scale ambiguity remains (commutant dimension 1 against 2)
    [derived, numerically checked]. The outcome probe fixes the scalar.
  - R and Q have determinants +1 and −1, so no basis change maps one onto the
    other: swapping them is a behaviorally detectable scramble. Two rotations
    commute, so cannot test order [derived].
  - Effective target parameter counts, for 2-D latent state, 8-D observations
    and two actions: a target-only shared-encoder latent model has about 16 +
    8 − 4 = 20 (encoder, operators, minus the GL(2) gauge); frozen-dynamics
    transfer has about 16 − 1 = 15. Per-action reduced-rank and full linear
    maps have about 56 and 128. The source benefit over the best target-only
    latent model should therefore be modest at large budgets [derived,
    counting heuristic].
  - Norm-preserving operators do not damp errors. A perturbation ε in an
    estimated operator gives rollout error growing roughly linearly in
    horizon (about 1.1×10⁻³ at L = 1 and 8.6×10⁻² at L = 128 for ε = 10⁻³)
    [derived, numerically checked]. Transfer of exact source operators should
    therefore help most at small budgets and long horizons; choose budgets
    and horizons where the direct references predict a measurable gap.
  - Composing the population-optimal one-step map R F_a K, with
    Kx = E[z | x], inserts KR between transitions, while the oracle applies
    K once; on a draft world it keeps 0.73 of the oracle's gain at
    L = 128. One-step fitting is therefore not a long-horizon capacity test
    [derived, numerically checked].
  - For the three-module SRI window at prior 0.001 and κ = 1, each two-step
    window keeps about 0.983 of gain (about 0.34 after 128 steps), which a
    scalar rescaling of the translations removes. A constructed encoder
    rejects nuisance only with nuisance weights beyond the ±2 clip (relative
    leakage 0.14 at the clip) [derived, numerically checked; see the
    [relational linear preflight](relational-linear-preflight-v1-contract.md)
    and its draft-check script].

## Post-preflight originality review and GDL corrections

Search/review date: 2026-10-27. See the
[detailed review](relational-originality-review.md) for reading scope,
derivations, corrections to the GDL notes, and the revised research decision.
Searches covered multi-step prediction, latent-dynamics transfer, NRI,
time-warping gates, unitary recurrence, switching state-space models,
multiplicative transformation learning, Augerino, equilibrium models, and
local temporal credit assignment. This is a targeted overlap review, not
proof that no original SRI contribution is possible.

- **GDL representation theory** [read, current book Chapter 3, §3.6.10].
  [The authors' chapter](https://geometricdeeplearning.com/book/algebraicpriors.html)
  explains irreducible decomposition and Schur's lemma, including the real
  1/2/4-dimensional possibilities. Our joint alignment constraints are an
  application. This does not establish noisy finite-sample encoder recovery.
  Domain coarsening in [Chapter 4](https://geometricdeeplearning.com/book/geometricpriors.html)
  is distinct from reducing channel width [read, §4.4.4].
- **Multi-step prediction and latent transfer** [read, selected sections].
  [Venkatraman et al.](https://cdn.aaai.org/ojs/9590/9590-13-13118-1-2-20201228.pdf)
  explicitly contrast linear one-step and simulation objectives. [Sun et al.](https://arxiv.org/pdf/2201.00248)
  give a transferable-dynamics result under deterministic-transition and
  sufficient-representation assumptions (Theorem 8). These explain why the
  broad objectives are established, but do not prove success of our noisy
  fixture's estimator or local learning rule.
- **Inferred relation types** [read, NRI §§3–3.4].
  [Kipf et al.](https://proceedings.mlr.press/v80/kipf18a/kipf18a.pdf)
  infer latent edge types and use type-specific prediction functions. They
  train over multiple steps specifically to discourage ignoring relations.
  SRI's proposed equilibrium edge state differs in inference and objective;
  it is not a new relation-discovery capability merely because it is an
  energy variable. Switching linear regimes are also established:
  [Ghahramani and Hinton](https://www.cs.utoronto.ca/~hinton/absps/switch.html)
  [abstract].
- **Multiplicative transformation codes** [read, opening only].
  [Memisevic and Hinton 2010](https://www.cs.toronto.edu/~rfm/pubs/factored.pdf)
  use factored higher-order Boltzmann machines to represent transformations
  and demonstrate visual analogy. This is a close conceptual predecessor of
  content-independent relation codes. Their probabilistic estimator is not
  equated with SRI's quadratic-residual construction.
- **Gating** [read, §1].
  [Tallec and Ollivier](https://arxiv.org/pdf/1804.11188) derive gates from
  time transformations through a continuous-time model and discretization.
  Our two-quadratic weighted-average identity is elementary [derived], not
  an independent gating invention or an exact discrete time-warp theorem.
- **Local temporal learning is also an active established area.**
  The main text of both papers was later read for the
  [fixture assessment](relational-linear-fixture-assessment.md).
  - [Williams et al. 2026](https://proceedings.mlr.press/v306/williams26a.html)
    [read, main text]. In data-aligned linear RNNs, one-step tBPTT shares
    BPTT's fixed-point structure. RFLO's weight changes have rank at most
    the output dimension, and it converges more slowly.
  - [Potter and Rhodes 2026](https://arxiv.org/abs/2602.18131)
    [read, main text; preprint, appendix proofs not audited]. Standard tPC
    has one-step truncated credit. tPC-RTRL is local in space and time and
    matches BPTT on four tasks. Propagating inferred states during training
    worsens open-loop rollout.
  - Differs: tPC-RTRL's exactness is for a single-layer element-wise
    recurrence; its multilayer extension is layer-local and approximate
    (§5.5). Williams et al. study a single linear recurrence with no
    energy-based inference.
  - Hypothesis: they supply machinery and controls for local long-horizon
    learning. Neither is equivalent to SRI's procedure, and neither shows
    that SRI's coupled-module combination is redundant.
- **Cheap deductions** [derived, detailed in the review]: implicit-Euler
  errors equal (I+hH)^−t e_0; product errors telescope and can accumulate
  with horizon; one-step conditional prediction repeatedly inserts KR;
  the Sylvester residual inherits local singular-value stability away from
  zero normalization. None needs an experimental mechanism-discovery study.

**Research decision:** the preflight adds implementation-specific feasibility
and cost evidence. Before a further substantive run, specify a concrete
advantage over the closest established method, its constraints, and a minimal
falsifying test. The detailed review supersedes treating the GDL notes'
follow-up list as an originality-ranked research agenda.

## Number concept network

Search date: 2026-10-28. Starting point: the user-supplied *Number concept
network: design doc* (2026-10-28). Queries: “Trabelsi Deep Complex Networks
2018 arxiv”, “Wang Isola Understanding Contrastive Representation Learning
Alignment Uniformity 2020”, and “Locatello unsupervised disentanglement
impossible inductive biases 2019”; also opened Khosla et al. directly.
The [implementation specification](number-concept-network-v1-spec.md) records
the architecture, loss changes, data boundaries, controls and computation plan.
No experiment has been implemented or run for this entry.

- **Complex network components.** [Trabelsi et al., 2018,
  §§3.4, 3.6 and reported activation comparisons](https://arxiv.org/html/1705.09792v4)
  [read, selected sections]. Shown: complex network building blocks and
  initialization constructions; modReLU was not consistently competitive in
  their feed-forward experiments. Differs: no digit-to-dice one-anchor transfer
  experiment. Hypothesis: these components may implement a useful phase-coded
  succession representation here; feasibility must be measured. The spec uses
  the Rayleigh/uniform-phase construction described in §3.6, not the
  semi-unitary initialization used in that paper's experiments.
- **Supervised contrastive learning.** [Khosla et al., 2020](https://arxiv.org/abs/2004.11362)
  [abstract]. Shown: class labels support multiple positives and class-based
  contrastive training. Differs: here a query is transformed by a learned
  successor operator and positives belong to its assigned *target* class.
  Hypothesis: a successor-conditioned multi-positive variant avoids the
  original PDF's same-number false negatives [analogy; not a reproduction of
  the published method]. The PDF's prose says negatives are other numbers,
  but its single-positive batch formula does not enforce that [derived].
- **Alignment and uniformity.** [Wang & Isola, 2020](https://proceedings.mlr.press/v119/wang20k.html)
  [abstract]. Shown: an asymptotic analysis connects contrastive objectives
  with positive alignment and distributional uniformity. Differs: no theorem
  identifying a common coordinate system for independently trained modalities
  sharing a frozen successor. Hypothesis: contrastive separation can help
  prevent a useful solution from collapsing, but it does not prove semantic
  alignment or guarantee optimization avoids collapse [derived limitation].
- **Identifiability context.** [Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html)
  [abstract]. Shown: unsupervised disentanglement requires assumptions about
  models/data; their empirical study also questions unsupervised model selection.
  Differs: this proposal has within-modality relation supervision and an anchor.
  Do not apply their impossibility result directly. The relevant warning about
  unacknowledged inductive biases is an analogy, not a proof for this model.

Project-specific deductions [derived, no experimental evidence]:

1. A diagonal unitary successor preserves each coordinate magnitude, so an
   exact orbit can encode distinct numbers entirely in phase. A magnitude-only
   probe is an inappropriate primary feasibility requirement.
2. Every diagonal unitary Q commutes with the successor R. Transforming dice
   codes by Q and precomposing their decoder with Q inverse preserves cosine
   succession scores and reconstruction. This symmetry is available within
   the proposed architecture by changing its final encoder affine map and first
   decoder affine map. It permits ambiguities beyond an integer count offset.
   Equal frequencies allow additional complex-unitary mixing. For the real
   orthogonal commutant relevant to cosine, opposite frequencies also allow
   swaps with conjugation, and frequencies 0 or pi allow reflections within
   their real 2×2 blocks. Those conjugate-linear maps are not generally
   absorbable into the architecture's complex-linear boundary layers, so
   relation-score symmetry need not be full-objective symmetry. Independently,
   every orbit start defines an exact recurrence, including different
   coordinate magnitudes; this does not imply equal contrastive/reconstruction
   losses but rules out a recurrence-only preference for integer offsets.
3. One exact anchor plus exact shared recurrence identifies a finite orbit,
   but neither a contrastive ordering loss nor one anchored image guarantees
   those equalities at class level. Approximate alignment is the experiment.
4. A ten-centroid readout has a 10% uniform-guess baseline even when scoring
   only true counts 2–6. A 20% baseline requires a different, restricted
   candidate set. Empirical controls are more informative than either nominal
   chance rate because predictions may be strongly nonuniform.
5. New random layouts and new counts change two factors simultaneously.
   Random layouts of already trained counts are required to interpret an
   unseen-count failure. Even a success leaves possible area-based shortcuts.
6. Contradictory-pair and no-pair controls do not isolate compatibility of the
   frozen operator with the recipient digit representation. The revised spec
   therefore includes a fixed cyclic donor-seed operator swap, holding the
   recipient anchor and readout fixed. This probes operator/representation
   compatibility; it is not proof of a unique semantic operator. Donor fit and
   within-dice recurrence diagnostics are needed to interpret a negative result,
   and cyclic reuse makes the five seed differences statistically dependent.
7. Rejecting a whole dice configuration, including its newly drawn radius,
   changes the accepted radius density to a distribution proportional to
   `p(radius) * P(valid placement | radius, count)`. Thus count-independent
   proposal radii do not imply count-independent rendered radii. The corrected
   sampler preassigns immutable radii and resamples positions only, for both
   standard and random layouts. Exhaustion aborts pool construction rather than
   replacing the sample. Per-ID metadata invariants and rejection-rate audits
   are required. This removes radius-selection bias, not geometry/radius
   dependence or total-area counting shortcuts [derived; no simulation run].

Cheaper alternatives: an exact known-code orbit is an algebraic sanity check,
not an image-learning baseline; reconstruction plus the same single anchor is
the included learned baseline. Explicit pip counting, real-valued encoders and
decoder-free learning are sensible follow-ups, but this v1 claims neither
general counting nor architectural/computational superiority. Its contribution
would be controlled practical evidence for or against one-anchor relation
transfer, not a novelty claim.

## Number cross-relation pilot: originality assessment

Search date: 2026-10-28, after reading
[the pilot](number-cross-relation-pilot.md). Targeted queries included
“knowledge graph entity alignment seed alignment TransE”, “MNIST dice successor
shared representation alignment”, “On the Transferability of VAE Embeddings”,
and “Coherent and Consistent Relational Transfer Learning”. This is a targeted
assessment, not an exhaustive novelty search. No experimental artifacts changed.

- [Strömfelt et al., NeurIPS 2022, *Formalizing Consistency and Coherence of
  Representation Learning*](https://proceedings.neurips.cc/paper_files/paper/2022/file/2d95270d763751439626d91f57e9a750-Paper-Conference.pdf)
  [read: introduction, decoder description, §§6–7]. Shown: MNIST-to-BlockStacks
  partial relation transfer, training a target autoencoder through frozen
  successor judgments and evaluating other relations; decoder consistency
  relates to transfer performance. Differs: separate autoencoders, staged
  transfer and within-target relation evaluation, rather than joint digit/dice
  training with sparse cross-modal equality pairs. Implication: coherent
  relation geometry aiding visual cross-domain transfer already has direct
  precedent; the pilot is not a reproduction of their method [analogy].
- [Strömfelt et al., 2020, *On the Transferability of VAE Embeddings using
  Relational Knowledge with Semi-Supervision*](https://arxiv.org/pdf/2011.07137)
  [read: abstract and main-text experiment description]. Shown: MNIST equality,
  ordering and successor supervision, comparing structured and higher-capacity
  relation decoders, including equality transfer to excluded classes. Differs:
  a VAE and within-MNIST transfer rather than digit/dice correspondence.
  Implication: even the structured-versus-flexible relation-decoder contrast
  and held-out equality evaluation have close precedent [analogy].
- [Zhu et al., IJCAI 2017, *Iterative Entity Alignment via Joint Knowledge
  Embeddings*](https://www.ijcai.org/Proceedings/2017/0595.pdf)
  [read: abstract, introduction and joint-embedding description]. Shown:
  shared relations and seed correspondences support further entity alignment
  across knowledge graphs. Differs: symbolic entity embeddings and iterative
  alignment, not learned image encodings. Implication: extending sparse
  correspondences using relational structure is established [analogy].
- [Bordes et al., 2013, TransE](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)
  [abstract]. Shown: relational scoring through translations in embedding
  space. Differs: the pilot uses a nonlinear residual successor and binary
  judgments, not TransE's fixed relation-vector model. A shared translation
  is therefore a simpler comparator, not an equivalent implemented method.

Assessment [inference from these sources and the pilot]: the general principle
is established; the particular MNIST/dice ablation is useful local evidence,
but no new algorithm or scientific principle has been established. These
searches did not establish whether the exact fixture/protocol already exists;
absence of an exact match would not itself establish a substantive contribution.
The strong numerical result supports feasibility, not originality. Before
claiming novelty, compare against a simpler translation-based relational model
and the closest structured-decoder approach, and identify a new capability,
efficiency advantage or demonstrated limitation of those approaches. Merely
adding seeds would strengthen reliability without resolving originality.

## Number core network

Search date: 2026-10-28, for the [number core network v1 specification](number-core-network-v1-spec.md).
Queries: “Deep Supervised Cross-modal Retrieval Zhen CVPR 2019”, “Mind the Gap
modality gap Liang 2022”, “Papyan Han Donoho neural collapse”, “Srivastava
Salakhutdinov multimodal deep Boltzmann machines”, “Piantadosi Tenenbaum Goodman
bootstrapping language of thought”, “Sarnecka Carey 2008 how counting represents
number”, “Stoianov Zorzi visual number sense”, “Nasr Viswanathan Nieder number
detectors”. Builds on the two preceding number entries and on
[modules, coupling and translations](#modules-coupling-and-translations).
This is a targeted contract review, not a novelty search.

- **Shared label space versus explicit invariance across modalities.**
  [Zhen et al., CVPR 2019, DSCMR](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.html)
  [read: §3 losses, §4.4.3 ablation, Tables 6–7]. Shown: image and text
  sub-networks with a shared second layer and one shared linear classifier.
  Label loss alone (their DSCMR4) gives cross-modal retrieval mAP 0.685 against
  0.716 for the full model on Pascal Sentence, and 0.524 against 0.613 on
  NUS-WIDE-10K. Removing only the explicit invariance loss `‖U − V‖` gives 0.694
  and 0.538. Differs: dense labels for every class in both modalities, retrieval
  rather than coincidence of codes, pretrained features, no relation structure.
  Implication: shared labels through a shared classifier align modalities
  substantially but not fully, and explicit invariance adds alignment even with
  dense labels [analogy]. This supports the spec's framing of label-only alignment
  as a hypothesis, and its equivalence-loss control arms.
- **Modality gap.** [Liang et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/702f4db7543a7432431df588d57bc7c9-Abstract.html)
  [abstract]. Shown: in CLIP-style models, image and text embeddings occupy
  separate regions, arising from initialization cones and kept by contrastive
  training, and the gap affects downstream results. Differs: contrastive
  image–text pretraining, not classification into shared labels or energy
  coupling. Implication: even objectives that explicitly match pairs can leave
  modalities in offset regions, so coincidence must be measured directly, not
  inferred from readout accuracy [analogy].
- **Neural collapse.** [Papyan, Han and Donoho, PNAS 2020](https://arxiv.org/abs/2008.08186)
  [abstract]. Shown: late in classification training, last-layer features
  collapse to class means that align with the classifier, and classification
  becomes nearest class mean. Differs: one modality, a learned classifier,
  training to zero error. Implication: shared fixed label codes push labelled
  classes of both input kinds toward the same *output* activity [analogy]. Whether
  that reaches the core is a separate question (deduction 1 below).
- **Multimodal models with a shared latent layer.**
  [Srivastava and Salakhutdinov, NIPS 2012; JMLR 2014](https://jmlr.org/papers/v15/srivastava14b.html)
  [abstract]. Shown: a multimodal deep Boltzmann machine with a joint top layer
  over modality-specific pathways learns a fused representation that helps
  classification and retrieval, including when only one modality is present at
  test. Differs: generative, sampled training on paired image–text data; no
  sparse labels or relation structure. Implication: the image ↔ core ↔ Morse
  layout is a standard shared-latent multimodal design, here as a quadratic
  energy with learned translations [analogy]. MuPNet (above) is the
  predictive-coding version.
- **Relations as translations and relation transfer.** Covered in the
  [cross-relation originality assessment](#number-cross-relation-pilot-originality-assessment):
  TransE (relation vectors), seed-based entity alignment (Zhu et al. 2017) and
  structured relation decoders for MNIST successor relations (Strömfelt et al.
  2020, 2022). The affine core successor `S(c) = A c + t` generalizes TransE's
  fixed translation; with A = I it is TransE's form [derived].
- **Numerosity in networks.** [Stoianov and Zorzi, Nature Neuroscience 2012](https://doi.org/10.1038/nn.2996)
  [abstract]: numerosity detectors emerge in a deep generative model of dot
  images. [Nasr, Viswanathan and Nieder, Science Advances 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6506249)
  [abstract]: number-tuned units emerge in a CNN trained only for object
  recognition. Differs: approximate numerosity of dot arrays, not exact counts
  or symbol grounding. Implication: dice cardinality may be partly available to
  a generic image encoder; the spec's dice still use fixed face layouts, so face
  recognition can stand in for counting. Set-function encoders (Deep Sets,
  Zaheer et al. 2017) [familiar] remain the structural alternative for a later
  random-layout study.
- **Children's number-word learning.** [Sarnecka and Carey, Cognition 2008](https://doi.org/10.1016/j.cognition.2008.05.007)
  [abstract and secondary citations]: cardinal-principle knowers judge that
  adding one item moves one step in the count list better than subset knowers,
  though modestly (about 67% for sets of 4 and 5 against 50% chance), and later
  work disputes the direction of dependence. [Piantadosi, Tenenbaum and
  Goodman, Cognition 2012](https://dspace.mit.edu/handle/1721.1/98844)
  [abstract]: a Bayesian program-learning model reproduces the subset-knower to
  cardinal-principle transition from Carey's bootstrapping account, arguing that
  co-occurrence learners cannot. Carey (2009) and Gelman and Gallistel (1978)
  [familiar]. Differs: these describe children and symbolic program induction,
  not learned image encoders. Implication: "exact meanings for a few small
  numbers plus a successor rule" is a documented developmental pattern, which
  the sparse-label design resembles [analogy]. It is motivation, not a
  prediction for this architecture.
- **Training through exact equilibria.** Backpropagation through the exact
  equilibrium of a quadratic energy is implicit differentiation of a linear
  solve, as in differentiable optimization layers (OptNet, Amos and Kolter 2017)
  and deep equilibrium models (Bai et al. 2019) [familiar]. With clamped
  inputs, the equilibrium is the conditional mean of the corresponding
  linear-Gaussian model [derived]. It is not a test of the program's local
  learning rule, which the spec defers to a later stage.

Project-specific deductions [derived, no experimental evidence]:

1. **Labels constrain the core only through what the output sees.** The label
   depends on the core's equilibrium state through the linear core → output path.
   Two core states that differ only in directions that path does not pass to the
   output give the same label. With a 28-unit core and a 16-unit output layer,
   those directions always exist. So label supervision cannot by itself force
   same-number core codes to coincide. Coincidence must come from the
   within-modality relations, the coupling geometry or the training dynamics,
   and must be measured. This sharpens the reviewer's point.
2. **Succession transfers labels only through the core inner layer.** The
   successor and judge act on the core inner layer. If labelled codes coincide
   there (dice 1 with digit 1, and so on), then `S` carries the coincidence to
   unlabelled counts, as in the pilot. If they coincide only in the part the
   output sees, nothing ties unlabelled counts across input kinds. The S-X
   versus S-X-E contrast measures which happens.
3. **Chance.** A ten-way readout has a 1/10 uniform baseline on any subset of
   true classes (as in the number concept entry, deduction 4).
4. **Standard Morse partly encodes number on its surface.** Codes 1–5 are n
   dots then dashes, and 6–9 are n − 5 dashes then dots. An encoder could
   interpolate unlabelled Morse counts from the surface alone. Scrambled codes
   remove that; the comparison of N-M with N-X measures it.

Cheaper alternatives: a shared encoder with a structured judge (the cross-relation
pilot) already answers whether succession extends sparse equivalents, for two image
kinds. A DSCMR-style late-fusion model (two encoders, shared classifier, optional
invariance loss) is the standard baseline for the label-versus-invariance question
and could be added at small cost. The spec includes its two ingredients (label
loss, equivalence loss) inside the SRI architecture, not the DSCMR model itself.

Contribution: no new general principle. Sparse-correspondence extension through
relational structure and shared-label multimodal alignment are both established.
The study tests whether the program's coupled-module architecture supports them,
whether shared labels suffice as the bridge, and how coupling topology affects it.

## Abstraction learning beyond coordinate alignment

2026-10-28 follow-up to number-core pilot 2: see
[learning reusable structure](abstraction-learning-principles.md) for the
source/search record and claim-level evidence tags. The note distinguishes
shared parameters from reused mechanisms, and predictive sufficiency from
compression. It connects multi-view information bottleneck, predictive state
representations, and DreamCoder's library refactoring to an untested SRI
consolidation hypothesis. It also records the limits of importing
identifiability results from unsupervised disentanglement. No new method,
contract, or successful abstraction result is claimed.

User-supplied follow-up: [An et al., 2023, v1](https://arxiv.org/html/2302.11978v1)
is discussed in the same note, including a revision of research priority toward
transfer dynamics and training history. Explicit consolidation remains a
hypothesis rather than an established missing ingredient.

## Training for learning efficiency (number systems meta)

Search date: 2026-10-28, for the [number systems meta v1 specification](number-systems-meta-v1-spec.md).

Sources:
- abstract pages fetched for MAML (arXiv 1703.03400), first-order meta-learning and
  Reptile (1803.02999), OML (1905.12588), ANML (2002.09571), ANIL (1909.09157) and
  Chollet (1911.01547);
- the others are stated from memory.

This is a targeted contract review, not a novelty search.

- **Value of an abstraction as learning cost.**
  [Chollet 2019, *On the Measure of Intelligence*](https://arxiv.org/abs/1911.01547)
  [abstract].
  - *Shown (an argument, not an experiment):* skill alone can be bought with priors or
    experience. Intelligence is skill-acquisition efficiency over a scope, accounting
    for priors, experience and generalization difficulty.
  - *Differs:* a definition and benchmark proposal (ARC), not a training method.
  - *Implication:* Stage A's accounting (source cost reported beside reuse benefit) and
    the external review's "reuse benefit" are this definition applied to one study
    [analogy].
- **Optimizing for performance after bounded adaptation.**
  [Finn, Abbeel and Levine, ICML 2017, MAML](https://arxiv.org/abs/1703.03400)
  [abstract]. First-order MAML and Reptile:
  [Nichol, Achiam and Schulman 2018](https://arxiv.org/abs/1803.02999) [abstract].
  - *Shown:* initial parameters trained so that a few gradient steps on a new task work
    well. First-order variants that drop second derivatives perform well on few-shot
    benchmarks.
  - *Differs:* tasks there are new classes for one input space. Here the new task is a
    new *input representation* of known numbers. Here the adapted part is the input
    satellite, and the outer gradient reaches a core and output that the inner loop never
    changes [derived].
  - *Implication:* the episode in the spec is first-order MAML with a preservation term.
    Known difficulties: sensitivity to the inner learning rate and step count, and
    instability of the outer loop [familiar].
- **Meta-learning against forgetting.**
  [Javed and White, NeurIPS 2019, OML](https://arxiv.org/abs/1905.12588) [abstract].
  [Beaulieu et al. 2020, ANML](https://arxiv.org/abs/2002.09571) [abstract].
  Related: MER (Riemer et al., ICLR 2019) and La-MAML (Gupta et al., NeurIPS 2020)
  [familiar].
  - *Shown:* representations meta-trained for fast online learning *and* low
    interference. OML reports naturally sparse representations, with plain online
    updating competitive with rehearsal methods. ANML meta-learns a gating network that
    controls where learning happens, and learns 600 classes sequentially.
  - *Differs:* OML adapts a head on a fixed meta-learned representation. Here the input
    interface adapts in front of a meta-trained core, the reverse placement [derived].
  - *Implication:* the preservation term λ L_preservation is the same idea as OML's
    interference objective. ANML is the established form of "a learned signal decides
    which connections learn", the reviewer's credit-assignment remark [analogy].
- **Rapid learning or feature reuse.**
  [Raghu et al., ICLR 2020, ANIL](https://arxiv.org/abs/1909.09157) [abstract].
  - *Shown:* MAML's gains come mainly from reusing features. Adapting only the head
    (ANIL) matches MAML.
  - *Related:* strong multi-task or supervised pretraining often matches meta-learning on
    few-shot benchmarks (Tian et al., ECCV 2020) [familiar].
  - *Implication, a prediction for this study:* meta may help mainly by improving generic
    features. That would show up as meta ≈ ordinary + dev, the data-matched control,
    which the spec includes for this reason.

**Failure modes to watch [derived].**
- **Meta-overfitting:** only three development representations drive the outer loop.
  The final representations are reserved to detect this.
- **Bias of the first-order gradient:** gradients taken at adapted satellite values are
  applied to the pre-adaptation values.
- **Unequal budgets:** episode encodings are counted, and CPU time is measured
  separately.
- **Hand:** the sequence medium has no episodes, so Hand tests transfer to an
  untrained medium.

**Cheaper alternatives.**
- ANIL-style episodes that adapt only part of the satellite.
- Reptile, which needs no outer loss.
- The ordinary + dev multi-task control, already an arm.

Contribution: no new method. The study tests whether the established
learning-efficiency objective, applied to the SRI number network, turns structure that
is visible to probes into lower learning cost.

## Senses, varied contexts, rehearsal and imagery (number construction Part D)

Search date: 2026-10-29, for the Part D protocol and imagery probe in the
[number construction v1 specification](number-construction-v1-spec.md).

Sources:
- search-result summaries checked for Rolnick et al. 2019, Dosovitskiy and Brox 2016,
  Izard et al. 2009 and Hinton et al. 1995;
- the others are stated from memory.

This is a targeted contract review, not a novelty search.

- **Rehearsal against forgetting.** Ratcliff 1990 and Robins 1995 (rehearsal and
  pseudo-rehearsal) [memory];
  [Rolnick et al. 2019, *Experience Replay for Continual Learning*](https://arxiv.org/abs/1811.11682)
  [search summary].
  - *Shown:* replaying stored earlier experience greatly reduces catastrophic forgetting.
    A small buffer with random discarding does almost as well as an unbounded one.
  - *Predicts:* D-rehearsal should retain Chinese and English far better than D-base.
    This is a standard component, the "standard solution" side of problem 1, not a
    contribution.
- **One quantity across senses.**
  [Izard, Sann, Spelke and Streri 2009, *Newborn infants perceive abstract numbers*](https://pubmed.ncbi.nlm.nih.gov/19520833/)
  [search summary].
  - *Shown:* newborns familiarized with auditory sequences of a number look longer at
    visual arrays with the matching number (4–18).
  - *Differs:* innate approximate number, not learned exact counts; a behavioural
    result, not a mechanism.
  - *Implication:* taps–collection pairing as a second sense is a natural context, and
    taps → form matching without any taps–form pairing is the cross-sense composition
    to test.
- **Varied versus repeated practice.** Shea and Morgan 1979 (contextual interference);
  interleaving studies [memory].
  - *Predicts:* varied practice can slow acquisition and improve transfer and
    retention; acquisition speed is not the criterion.
- **Multi-view and multi-task learning.** Shared representations trained on several
  views or tasks generalize better when the tasks share structure (Caruana 1997;
  multi-view contrastive learning) [memory].
  - *Differs:* those set the views and tasks in advance with shared labels or paired
    views; here the compositions are untaught pairs.
- **Imagery by reverse inference.** A clamped quadratic energy is a Gaussian model:
  clamping a core layer and solving the rest gives the conditional mean of every other
  state [derived]. Generative top-down passes are the Helmholtz machine's sleep phase
  ([Hinton, Dayan, Frey and Neal 1995](https://www.cs.toronto.edu/~hinton/absps/ws.htm),
  [search summary]) and the predictive-coding generative direction (Rao and Ballard
  1999).
  - *Predicts:* the conditional mean shrinks toward zero, so solved features have small
    norm. Rescaling is a readout convenience, not part of the model.
- **Reading features as images.**
  [Dosovitskiy and Brox 2016, *Inverting Visual Representations with Convolutional Networks*](https://arxiv.org/abs/1506.02753)
  [search summary]; Mahendran and Vedaldi 2015 (optimization-based inversion) [memory].
  - *Shown:* a trained decoder predicts the average of the images that could have
    produced a feature vector.
  - *Predicts (confirmed in a smoke check):* features invariant to dot positions decode
    to blurred collections, so connected-component counting fails on reconstructions. A
    count reader trained on reconstructions of real inputs, and nearest real inputs in
    feature space, are the usable readouts.
  - *Known failure:* a decoder can produce plausible inputs from its own training prior
    regardless of the clamped state. A random-state control is needed.
- **Hidden-outcome arithmetic.** Wynn 1992 (infants shown objects added behind a screen
  look longer at impossible outcomes) [memory].
  - *Differs:* the outcome is shown, so it tests expectation of a single operation (the
    composition test's level 1), not a comparison of a hidden result with a reference.
- **Systematic composition in neural networks.** Lake and Baroni 2018 (SCAN): sequence
  models often fail to compose learned primitives in new combinations [memory].
  - *Predicts:* composed relations may work on familiar step counts and fail on new
    crossings, so composition must be tested on strata where shortcuts fail.
- **Coherence detection from successor residuals.** Prediction-error novelty detection
  (Rao and Ballard; Li, Tang and Bogacz 2025, in
  [conceptual instinct notes](conceptual-instinct-notes.md)).
  - *Implication:* Part D records the untrained residual detector as a baseline only.

## Memorization, interference and sparse representations (arbitrary-pairings design)

Search and read date: 2026-10-30. Abstracts or arXiv listing summaries of the four arXiv
papers below were read; the older interference literature is from memory. Written for the
[arbitrary-pairings contract](arbitrary-pairings-v1-contract.md) before freezing.

- **Networks can memorize arbitrary pairings.** [Zhang et al. 2017](https://arxiv.org/abs/1611.03530)
  **[abstract]**: large convolutional networks trained by stochastic gradient methods fit
  random labelings, and random-noise inputs, and removing regularization does not change
  this qualitatively; a depth-two network can fit any finite sample once parameters exceed
  samples. Differs: their setting is image classification at scale with one head. Hypothesis:
  capacity to memorize unstructured pairs is not in doubt here; what varies across arms is
  how fast and at what cost to structure.
- **Memorization can be needed for accuracy.** [Feldman 2020](https://arxiv.org/abs/1906.05271)
  **[abstract]**: on a long-tailed mixture of subpopulations, memorizing rare examples is
  required for near-optimal generalization, and limiting memorization (compression,
  privacy) costs some subgroups more than others. Differs: a theoretical model plus
  supporting experiments, not a number task. Hypothesis: a network with a memory part may
  hold specific associations that a structure-only network cannot, so a deficit of
  `sparse48` on arbitrary pairs is the expected sign.
- **Compression forgets the rare, atypical cases first.** [Hooker et al. 2019](https://arxiv.org/abs/1911.05248)
  **[abstract]**: pruned and quantized networks keep aggregate accuracy while losing
  disproportionately on a small set of atypical or noisy exemplars (PIEs). Differs: post
  hoc compression of a trained model; here sparsity is built in. Hypothesis: a sparse
  reader will show the same pattern on pairings with no shared structure.
- **Sparse representations are more robust to interference.** [Ahmad and Scheinkman 2019](https://arxiv.org/abs/1903.11257)
  **[abstract]**: highly sparse weights and activations were more robust to noise and
  interference than dense ones with competitive accuracy on MNIST and Google Speech Commands.
  Differs: the interference there is input noise, not the learning of later tasks. Hypothesis:
  the sparse part may suffer less interference between pairings; the memory part more.
- **Catastrophic interference and its remedies.** McCloskey and Cohen (1989), Ratcliff
  (1990) and French (1999, a review that includes semi-distributed representations)
  **[familiar]**: learning new associations in a shared distributed network overwrites
  old ones, and less overlapping representations reduce this. Complementary learning
  systems and fast weights are in the [limited-memory proposal](limited-memory-proposal.md#prior-work-and-expected-limits).

**What the literature does and does not support (corrected after external review).**
Zhang et al. establish that large networks fit random labels, not that dense networks
memorize better than sparse ones; Ahmad and Scheinkman study robustness of sparse
high-dimensional representations to noise and interference in image and speech classifiers,
not sequential interference between learned pairings in readers like these. Feldman and
Hooker support asking whether limiting capacity costs rare, specific cases. So three things
are **hypotheses**, none a prediction from the literature: that dense-with-memory arms
acquire arbitrary pairs faster than `sparse48`; that sparse arms interfere less; that decay
costs retention unless rehearsal restores it (Hinton and Plaut 1987 **[abstract]** describe
rehearsal restoring fast-weight associations). The size and direction of each effect here
are open.

**What the pairings design adds.** The earlier arms differ in connectivity and decay but
were scored on structured relations only. Scoring arbitrary pairings in the same curriculum
tests the memory/structure trade directly; it is an internal control for the limited-memory
study, not a new method.

## Memorized facts and relational inference

2026-10-30 targeted search: [memory and relational prior work](memory-relational-prior-work.md)
records TEM, complementary structure-learning, DNC, Memory Networks, Go-CLS,
Wide & Deep and shared/routed experts, with evidence levels and architecture differences.
The central distinction is coexistence of recall and reasoning versus inference that
actually uses newly memorized facts; connectivity sparsity must also be distinguished
from sparse representations and expert routing. These are reading notes, not a new contract.

## Established remedies after the forgetting-mechanism study

2026-10-31: [forgetting remedies](forgetting-remedies-prior-work.md) records the
representation/readout distinction, Dark Experience Replay, Learning without Forgetting,
and REMIND, with reading depth, storage requirements, and limits of transfer to our task.
The proposed next emphasis is established retention baselines, not assuming the observed
failure needs a new architectural solution.

## Implications for contracts

- Use standard names where the mechanism is the same: EP for the two-phase
  rule; "SI-inspired update scaling" rather than SI for arm 8. Describe the
  clamped-energy rule as having predictive-coding form, not as factor analysis.
- Declare activity-prior strength as a design choice. Topology rankings in
  this project changed when it changed.
- On the current linear fixture, avoidable error can be measured exactly
  against the noiseless mean
  ([floor analysis](local-contrastive-consolidation-v1-floor-analysis.md)).
  Declare criteria on it.
- Directions with documented difficulties: identity-style consensus between
  modules (GLOM agreement); importance estimation without gradient signal (SI
  and EWC on saturated tasks); finite-β EP (use symmetric or holomorphic
  nudging).
- For the linear relational fixture: evaluate dynamics modulo one shared
  basis change; use noncommuting operators of different determinant so action
  order and scrambles are detectable; compare against a target-only
  shared-encoder latent model, not only per-action regressions; and include
  simulation-error (DaD-style) multi-step training as the objective control.
- Established alternatives to consider before inventing new mechanisms:
  - bidirectional predictive coding for the objective question;
  - correlated-nuisance fixtures to separate prediction from alignment;
  - anchor-penalty SI and training-regime controls for forgetting.
- For forgetting-as-pressure designs (2026-10-30): the established counterparts are
  forget-and-relearn (Zhou et al. 2022), warm starting and its fix (Ash and Adams 2019),
  DropConnect (Wan et al. 2013) and neural iterated learning (Ren et al. 2020); abstracts
  read 2026-10-30, full texts not read.
  In this project disrupting sparse reader connections helps, dense readers do not gain,
  and the count sits on most code coordinates, so measure transfer on the average
  coordinate ([analysis](count-coordinate-analysis.md),
  [limited-memory proposal](limited-memory-proposal.md)).

## Forgetting mechanism (design for the forgetting-mechanism contract)

Search and read date: 2026-10-31. Abstracts of the arXiv papers below were read via their arXiv listing pages; Hinton and Plaut
(1987) from a web-search summary of the paper and a later reproduction, not the full text. Written for the
[forgetting-mechanism contract](forgetting-mechanism-v1-contract.md) before freezing.

- **Forgetting concentrates in deeper layers.** [Ramasesh, Dyer and Raghu 2020](https://arxiv.org/abs/2007.07400) **[abstract]**: deeper layers are
  disproportionately the source of forgetting; mitigation methods work by stabilizing them; forgetting peaks at intermediate task similarity.
  Differs: image classification across tasks, not a matching reader after a pairing block. Hypothesis: the reader-side freeze should prevent
  more of the loss than the encoder-side freeze; this is an expectation the branches can contradict.
- **Representation can change without losing the knowledge.** [Davari et al. 2022](https://arxiv.org/abs/2203.13381) **[abstract]**: forgetting measured by
  the drop in task performance conflates head change with feature change; an optimal linear probe on the features measures representation forgetting
  separately and often shows it to be small even without anti-forgetting methods. Differs: supervised and unsupervised task sequences, not arbitrary pairs.
  Hypothesis: the item code stays separable (the 1-NN result so far) while the readout fails; the contract's frozen-bank decoder and component swap
  are aimed at the same distinction, and a linear probe trained on the current code is a cheaper established form worth adding as a descriptive measure.
- **Regime matters for forgetting.** [Mirzadeh et al. 2020](https://arxiv.org/abs/2006.06958) **[abstract]**: learning-rate decay, batch size and dropout widen
  minima and reduce forgetting; the optimizer is not named in the abstract. Differs: task sequences with a classifier. Hypothesis: lowering the step size
  would reduce the loss, which is why the earlier draft's learning-rate branch was dropped as a confound of gradient and momentum effects, not because it is uninformative.
- **Protecting important weights.** [Kirkpatrick et al. 2017](https://arxiv.org/abs/1612.00796) **[abstract]**: elastic weight consolidation slows learning on weights
  important to earlier tasks. Differs: needs a task boundary and an importance estimate. It is the non-rehearsal alternative to `rehearse-10`; not included, since the question here is
  where the loss arises, not a remedy.
- **Fast weights and recovery by rehearsing a subset.** Hinton and Plaut (1987) **[secondary]**: a slow and a fast weight per connection; after set B degrades set A, brief
  rehearsal of a subset of A restores the whole set. This is the nearest published form of a savings test; it motivates the relearning branches, and in the original rehearsal
  hits a subset, which the contract's relearning does not (it retrains all of A).
- **Catastrophic interference, rehearsal, complementary learning systems.** McCloskey and Cohen (1989), French (1999), Robins (1995) and McClelland, McNaughton and
  O'Reilly (1995) **[familiar]**: not re-read for this entry; the contract treats them as context only and draws no prediction from them.

**What this supports and does not.** The literature leads to expecting forgetting in the deeper, reader-side parameters and recoverability of representations; both are
hypotheses for this setting (a matching reader, arbitrary pairs, Adam, a small network). None of it predicts the early 25-update drop or the role of inherited Adam moments, which the
`zero-grad` branch tests directly. Nothing here is a ranking of the arms.

**External note on remedies (2026-10-31):** [forgetting-remedies-prior-work.md](forgetting-remedies-prior-work.md) compares the forgetting-mechanism findings with established remedies
(representation-versus-readout forgetting, Dark Experience Replay, Learning without Forgetting, REMIND). It is a targeted comparison, not a survey; its program implication is to
prioritize established retention baselines and their retention–new-learning–compute–storage tradeoffs before another broad diagnosis, with a first-moment reset kept as a focused
mechanistic diagnostic.

## Retention remedies and consolidation (added 2026-10-31, for retention remedies v1)

Search date 2026-10-31; sources were abstract pages and search summaries unless tagged otherwise. Full texts unread.

- **Rehearsal and pseudorehearsal.** Robins (1995, Connection Science 7) **[abstract: search summary]**: rehearsal of earlier items sharply reduces forgetting; pseudorehearsal replays
  internally generated pseudo-patterns, giving rehearsal's benefit without stored old data. Differs: small feed-forward nets and generic pattern sets, not matched arbitrary pairs.
  Hypothesis: the random-latent cue arm is a pseudorehearsal control; whether random cues land on the stored pairs is exactly what is open.
- **Deep generative replay.** [Shin et al. 2017](https://arxiv.org/abs/1705.08690) **[abstract]**: a generative model supplies synthetic old-task data mixed with new data, avoiding storage of past
  data. Differs: image classification tasks and a trained generator. Hypothesis: a cue cache (stored codes) is the storage-based counterpart of a generator; a learned generator is not tested here.
- **Distillation.** [Hinton, Vinyals and Dean 2015](https://arxiv.org/abs/1503.02531) **[abstract]**: transfers a model's learned behavior into another by training on its outputs. Differs: ensemble to
  single model, not recent memory to protected store. Hypothesis: destination training on the recent memory's soft outputs; the destination cannot exceed the teacher's fidelity.
- **Complementary learning systems.** McClelland, McNaughton and O'Reilly (1995, Psychological Review 102) **[abstract: search summary]**: fast sparse learning plus slow interleaved consolidation to avoid interference
  in overlapping distributed representations. Differs: a theory with partial empirical support, not a specification for this network. Hypothesis [analogy]: the project's second protected store follows this
  division; the evidence here can neither confirm nor refute the theory.
- **Dark Experience Replay, REMIND, Learning without Forgetting, representation versus readout forgetting:** see [forgetting-remedies-prior-work.md](forgetting-remedies-prior-work.md).
- **Not found / unverified:** an identifier I had in mind for a "brain-inspired replay" paper (van de Ven et al.) resolved to an unrelated paper and is not cited; its claims are unverified.

## Addressed memory and writing facts into weights (added 2026-10-31, for the translation-memory draft)

Search date 2026-10-31; abstract pages only.
- **Product-key memory layers.** [Lample et al. 2019](https://arxiv.org/abs/1907.05242) **[abstract]**: a structured memory layer with fast exact nearest-neighbor lookup adds up to about a billion parameters at small compute; a 12-layer transformer with it beat a 24-layer one and ran about twice as fast at inference. Differs: trained end to end, not written by consolidation. Hypothesis: addressing and retrieval cost for a bounded protected store.
- **Fast weights.** [Ba et al. 2016](https://arxiv.org/abs/1610.06258) **[abstract]**: weights on a time scale between activities and slow weights hold temporary memories of recent inputs without storing activity copies. Hypothesis: a source of replay cues with no separate buffer.
- **Mass-editing memory (MEMIT).** [Meng et al. 2022](https://arxiv.org/abs/2210.07229) **[abstract]**: thousands of associations written directly into GPT-J and GPT-NeoX weights. Differs: large pretrained models and known subject representations. Hypothesis: closed-form writes into existing weights as the alternative to a separate store.

Review follow-up, 2026-10-31: opened the linked primary sources for the
translation-memory draft review; no new experiment or contract.

- **Product-key constraints [read: sections 3.1–3.2].**
  [Lample et al.](https://arxiv.org/html/1907.05242) use a Cartesian product of
  two shared sub-key codebooks; fast exact lookup depends on that structure.
  **[derived]** Independently stored observed codes do not automatically obtain
  this lookup bound. Changing shared sub-keys can change multiple addresses;
  touching one value slot does not guarantee preservation of other retrievals.
  **[hypothesis]** Use exhaustive lookup as the small-scale reference before
  testing structured addressing and its collision/allocation policy.
- **Fast-weight accounting [abstract].**
  [Ba et al.](https://arxiv.org/abs/1610.06258) describe temporary memory without
  explicit copies of activity patterns. **[derived]** This replaces a storage
  representation, not the need to count retained state. **[hypothesis]** Using
  it to generate consolidation cues requires a separate retrieval/generation
  mechanism; autonomous enumeration is not established by the abstract.

### Neuroscience hints for translation-memory addressing

Search date 2026-10-31. Queries covered hippocampal pattern separation and
completion, memory indexing/cortical reinstatement, representational drift,
Tse schema consolidation, and Girardeau ripple suppression. Selected evidence,
not an exhaustive survey or a frozen design:

- [Neunuebel and Knierim, 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3904133/)
  **[read: summary]**: simultaneous rat DG/CA3 recordings under conflicting spatial
  cues showed separated DG representations and coherent CA3 completion.
  Difference: spatial population activity, not arbitrary visual translations.
  **[analogy]** Separate similar cues before associative storage, then recover
  an established pattern from degraded cues. **[hypothesis]** Sparse encoding
  plus recurrent retrieval may reduce interference; measure capacity and
  convergence rather than assuming sublinear lookup.
- [Horner et al., 2015](https://www.nature.com/articles/ncomms8462)
  **[read: indexed results excerpts; full page inaccessible]**: human event
  recall linked hippocampal activity to reinstatement of associated cortical
  elements. Difference: multi-element episodic recollection, not a software
  address table. **[analogy]** An associative index can reinstate content used
  by another subsystem; this does not establish a particular indexing algorithm.
- [Girardeau et al., 2009](https://pubmed.ncbi.nlm.nih.gov/19749750/)
  **[abstract]**: suppressing hippocampal ripples impaired spatial memory in rats.
  Difference: physiological intervention, not a specified replay algorithm.
  **[hypothesis]** Test internally generated replay and its causal contribution;
  the result does not establish replay as a solution to encoder drift.
- [Tse et al., 2007](https://pubmed.ncbi.nlm.nih.gov/17412951/)
  **[abstract]**: established flavor-place schemas supported rapid assimilation
  of new paired associates in rats. Difference: structured prior knowledge,
  not mere familiarity with an output symbol. **[hypothesis]** Distinguish
  schema-compatible bindings from familiar-target bindings in future tasks.

**[derived]** These findings motivate sparse associative retrieval and replay,
not an O(1) lookup guarantee. Parallel neural activity must not be equated
with constant total computational work. Software evaluation must count
encoding, active connections, settling iterations, retained synapses and replay.

Revised-draft review, 2026-10-31. Searches: Willshaw sparse capacity and
winner-take-all retrieval; modern Hopfield attention; PackNet iterative pruning.

- [Graham and Willshaw, 1994](https://papers.neurips.cc/paper_files/paper/1994/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)
  **[abstract]** analyze partial connectivity, noisy cues and winner-take-all
  recall; modifying dendritic sums using activity and unit usage changes
  capacity. Difference: not the draft's specified joint source/target dynamics
  (which remain undecided). **[derived]** A capacity estimate must name the
  retrieval rule, coding distribution and error criterion before it is used to
  size this experiment. **[hypothesis]** Compare one-step associative readout
  with recurrent settling to isolate the benefit of completion.
- [Ramsauer et al., 2020](https://arxiv.org/abs/2008.02217)
  **[abstract: search excerpt]** identify their update rule with transformer
  attention. **[derived]** This does not equate the draft's nearest-key lookup
  and duplicate/eviction policy with their full method; an attention reference
  needs its own specified read operation.
- [Mallya and Lazebnik, 2018](https://arxiv.org/abs/1711.05769)
  **[read: indexed paper excerpts]** use iterative pruning and retraining to
  allocate parameters. **[derived]** Newly allocated blocks without pruning
  should be called a parameter-isolation baseline inspired by that literature,
  not a reproduction of PackNet.

### Translation-memory pilot: identity objective follow-up

Search date 2026-10-31. Queries: supervised contrastive learning; continual
representation learning and forgetting. Prompted by the failed storage gate
in [the translation-memory pilot](translation-memory-v1-pilot.md).

- [Khosla et al., 2020](https://arxiv.org/abs/2004.11362)
  **[abstract: search excerpt]** study supervised contrastive representation
  learning for image classification. Difference: not this sequential symbol
  stream, anchored interface or associative store. **[hypothesis]** Train
  keys to identify symbols across renderings using known source identity;
  assess retention of this identity across subsequent batches. This is
  supervised identity learning when same-symbol labels define positives,
  not automatically instance discrimination or a remedy for forgetting.
- **Pilot interpretation [derived].** Nearest-prototype accuracy is the
  performance of one decoder, not an upper bound on information in a code.
  Concentration of variance in three directions does not alone establish
  that identity is absent. The post-hoc diagnosis uses current prototypes
  from eight training renderings at batch 6; it does not directly evaluate
  the historical single-rendering keys used by the deployed memory.
  **[hypothesis]** Separate current-code identity, historical-key matching,
  one-step storage recovery and recurrent recovery before attributing all
  failure to perception. Independent random source codes with controlled
  corruption provide a diagnostic of the memory rule without visual encoding.

Stable-key follow-up, 2026-10-31. Search: prototypical networks for few-shot
learning; primary abstract below.

- [Snell et al., 2017](https://arxiv.org/abs/1703.05175)
  **[abstract: search excerpt]** learn a metric for classification of classes
  absent from training using a small support set. Difference: this does not
  establish 120-way recognition from single rendered-symbol keys, nor lifelong
  adaptability. **[hypothesis]** A separate encoder pretrained on disjoint
  background symbols and then fully frozen can provide stable memory keys
  while the host continues learning. Test unseen-symbol recognition before
  integrating with the host; count pretraining and model selection costs.
- **[derived]** Freezing one independent key pathway has constant model
  storage, unlike retaining a snapshot per batch. Its preprocessing,
  normalization statistics and projection must also be fixed to guarantee
  identical codes for identical inputs. Slowing updates does not provide this
  guarantee. Stable keys can still be confused as more competing keys arrive;
  evaluate candidate-set growth separately from time with candidates fixed.

### Input-driven retrieval (Betteti et al., 2025)

Reading date 2026-10-31. User supplied DOI 10.1126/sciadv.adu6991;
publisher PDF denied access, so read the
[primary full text on PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12017325/).
**[read: main text including equations and methods; supplement not read]**.
Input continually modulates retrieval couplings using squared overlaps with
stored patterns. Simulations examine ambiguity, noise and transient glitches;
long-term learning remains future work. Inputs are constructed mixtures of
stored patterns. The general sensory-to-memory mapping is left open.

**[derived]** Our unclamped synchronous completion rule is not this model;
its failure does not test input-driven retrieval. Conversely, successful
tracking is not evidence of continual acquisition, bounded retention or
compatibility with changing encoders. For P patterns of width D, the explicit
factorization requires O(PD) pattern storage and dense matrix-vector work per
dynamics step; forming a D-by-D input-dependent matrix directly costs
O(PD²). These are implementation orders, not measured speed results.

**[hypothesis]** Persistent sensory evidence could stabilize retrieval under
brief corruption while permitting genuine changes. A small diagnostic should
compare input-driven dynamics with independent nearest-memory retrieval and
temporal smoothing/hysteresis on the same similarities. Measure false switches,
switch delay, failure to reject unfamiliar inputs and compute at matched
latency. Test image/audio evidence and conflicting cues only as an extension,
not a reproduction. No new contract or run is established by this note.

### LSTM gates for continuous sensory memory

Search date 2026-10-31: original LSTM and Learning to Forget.
[Gers, Schmidhuber and Cummins, 2000](https://pubmed.ncbi.nlm.nih.gov/11032042/)
**[abstract]** introduce adaptive forget gates for continuous streams without
externally marked resets, demonstrating benefits on their continual prediction
benchmarks. Difference: retention in recurrent state, not a demonstration of
lifelong multimodal storage or preservation during online parameter updates.
**[derived]** Persistent state and persistent learned parameters are different
memory mechanisms; gating the former does not automatically protect the latter.
**[hypothesis]** Test an ordinary LSTM receiving sensory embeddings as a bounded
memory baseline, before building a custom store controller. Teach delayed
cross-modal recall, distractor resistance and revisions through sensory
experience; measure old/new recall and interference at fixed state size.
Separate state-only adaptation with fixed parameters from continual parameter
learning. A gated associative matrix is a possible later extension, not an
LSTM reproduction. Count state, parameters and training workspace separately.

### Retrieval as a memory-changing experience

Search date 2026-10-31: Roediger/Karpicke testing effect and Anderson/Bjork
retrieval-induced forgetting.
- [Roediger and Karpicke, 2006](https://pubmed.ncbi.nlm.nih.gov/16507066/)
  **[abstract/search record]** report improved long-term retention from testing.
  Difference: human learning, not evidence that an LSTM automatically implements
  retrieval-dependent strengthening. **[hypothesis]** Sensory queries and the
  resulting retrieval activity should be allowed to alter persistent state.
- [Anderson, Bjork and Bjork, 1994](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/Anderson_RBjork_EBjork_1994.pdf)
  **[abstract: indexed excerpt]** find impaired delayed recall for unpracticed
  related items after selective retrieval practice. **[derived]** Human evidence
  does not imply every retrieval preserves its item or all unused items simply
  decay independently; competitive effects matter.

**[derived]** LSTM state can change through recurrent feedback without parameter
updates, but output gating alone is not a rule that reinforces the retrieved
association. **[hypothesis]** Train/test use-dependent retention at fixed state
capacity, separating query-only retrieval from sensory feedback/re-exposure.
Do not use evaluator correctness to select updates. Compare identical initial
states exposed to retrieval practice, matched re-exposure, or distractors;
measure delayed retention and incorrect-answer persistence. Natural queries
may update state; audit evaluations use discarded copies to avoid secretly
rehearsing memories through repeated measurement. Earlier frozen results keep
their original state-preserving evaluation protocol.

Streaming retrieval review, 2026-10-31, for
[streaming retrieval v1](streaming-retrieval-v1-contract.md). Searches: Betteti
preprint parameters; leaky competing accumulators and multi-alternative
sequential tests; quickest change detection; temporal smoothing and hysteresis
for online recognition; open-set rejection in streams.

- [Betteti et al., preprint](https://arxiv.org/abs/2411.05849)
  **[read: HTML, first 100,000 characters]** gives dt = 0.01, x(0) ~ N(0, I)
  and orthogonal or large-N i.i.d. binary patterns, and gives no capacity result
  for the input-driven model. It differs from the published version: weight
  linear in overlap, ψ = tanh(z), σ = 8. **[derived]** A contract must name the
  version it follows.
- [Xie et al., 2021](https://arxiv.org/abs/2104.04186)
  **[read: sections II–III.C]**: with known distributions CUSUM and the Shiryaev
  procedure are optimal in their senses; first-order delay is |log α|/D; EWMA is
  a simple alternative that needs no knowledge of the distributions. Transient
  dynamics change the problem. Difference: detecting one change between known
  distributions, not retrieval among P items with unfamiliar input.
  **[derived]** Delay and false alarms trade off; any comparison has to hold one
  fixed. A forward filter is the reference for independent frames.
- [Usher and McClelland, 2001](https://pubmed.ncbi.nlm.nih.gov/11488378/)
  **[abstract]**: leaky, competing nonlinear accumulators for perceptual choice.
  Difference: fits behaviour, one decision at a time, additive input.
  **[derived]** Input-driven dynamics in overlap coordinates also leak, but input
  multiplies the state; the two are not equated.
- Bogacz and Gurney, 2007 (Neural Computation)
  **[search excerpt only]**: the multihypothesis sequential
  probability ratio test as a multi-alternative decision benchmark. Context for
  using a filter as the reference.
- [Zhao and Krähenbühl, 2022](https://arxiv.org/abs/2209.09236)
  **[search excerpt]**: temporal smoothing kernels with constant-time updates for
  online video detection. Difference: learned features with supervision.
  **[derived]** Constant-time smoothing is established practice, so a recurrent
  mechanism has to beat it, not merely per-frame classification.
- **No match found** in these searches for hysteresis or open-set rejection
  evaluated together on a stream of retrievals from memory. Not a novelty claim;
  the searches were few.

2026-10-31: [Test-time neural memory: Titans, MIRAS and ATLAS](test-time-memory-prior-work.md)
records a mechanism-level reading prompted by the sensory-memory discussion.
It distinguishes online neural-memory updates, retention and windowed rehearsal
from retrieval-induced strengthening and lifelong adaptation of perception.

2026-10-31: [Continuous sensory memory: research direction](continuous-sensory-memory-notes.md)
records the user's sensory-only multimodal requirements, imperfect/use-dependent
retention, and the proposed intermediate memory for ongoing work across days.
It separates immediate context, active work and durable memory as hypotheses
and lists interruption/resumption and retrieval-practice tests before a contract.

Follow-up in that note: assimilation versus reorganization, linked back to
the sparse-reader transfer/acquisition/retention tradeoff and Go-CLS selective
consolidation. Surprise and gradient conflict are not sufficient criteria for
reorganizing shared representations; the proposed criterion is future
predictive benefit with measured retention and resource costs.

Further discussion records the user's [accumulated-surprise proposal](continuous-sensory-memory-notes.md#accumulating-surprises-before-changing-shared-knowledge):
temporary sensory traces gain support from related independent experiences,
drive association or representation updates, and otherwise fade. It separates
retrieval strengthening from independent confirmation, connects the hypothesis
to sparse-network retention, and identifies bounded replay and immediate
surprise-weighted updates as comparisons. This is not a novelty claim or a
frozen experiment contract.

### Surprise lifecycle: unresolved experience as evidence

Search date 2026-11-01, for the [surprise lifecycle draft](surprise-lifecycle-v1-contract.md).
Queries: MINAS novelty detection in streams; latent-cause memory modification;
Anderson and Schooler 1991; semantic drift compensation; storage and retrieval
strength; coherent gradients; noisy TV and learning progress. Sources also
read: Titans Eqs. 11–15, ATLAS Eq. 9 and its memorization note.

- [MINAS, de Faria, Carvalho and Gama, 2016](https://repositorio.inesctec.pt/handle/123456789/5309)
  **[abstract/search record]**: stream examples the model cannot explain are marked
  unknown, and groups of them become new patterns. **[familiar, full text not
  obtained]** a short-term memory holds the unknowns and they age out. Difference:
  clusters unlabeled examples to find classes; no durable-update routing.
  **[hypothesis]** The closest stream-mining relative of the accumulated-surprise
  proposal. Read in full before freezing.
- [Gershman et al., 2017](https://elifesciences.org/articles/23763) **[read: model
  section]**: a time-sensitive Chinese restaurant process, with a power-law
  temporal kernel, decides whether a trial shares a latent cause with old trials;
  delta-rule updates are scaled by the posterior. Longer gaps make assignment to
  an old cause less likely. Difference: Pavlovian conditioning model with few
  stimuli. **[hypothesis]** Far-apart recurrence is inherently ambiguous between
  "same cause" and "new cause", and its theory treats that as inference.
- [Titans](https://arxiv.org/html/2501.00663v1) **[read: Eqs. 11–15]**:
  `M_t = (1 − α_t)M_{t−1} + S_t`, `S_t = η_t S_{t−1} − θ_t∇‖M(k_t) − v_t‖²`, with
  α, η, θ data-dependent. [ATLAS](https://arxiv.org/pdf/2505.23735) **[read: Eq. 9,
  arXiv v1]**: minimizes `Σ γ_i‖M(k_i) − v_i‖²` over the last c tokens, c = 1 being
  the delta rule. Its γ are input-dependent gates that can prune inside the window,
  so selection there is learned, within a window fixed by recency. **[read]** The
  authors state that no persistent learning typically carries over once the memory
  is cleared. Difference from our use: constants replace learned controls.
  [Proceedings version](https://raw.githubusercontent.com/mlresearch/v306/main/assets/behrouz26b/behrouz26b.pdf)
  **[read: pp. 5-6, Appendix G]**: the same rule is Eq. 5 there, and OmegaNet's update
  is `M_t = alpha_t M_{t-1} - grad sum gamma_i ||M(phi(k_i)) - v_i||^2` with phi a
  polynomial kernel. So `omega` in the lifecycle contract is this update with
  gamma = 1, phi the identity and constant alpha and rate; it lacks the gates and the
  kernel, and ATLAS's Muon optimizer. Appendix G: with c = 1 and gamma = 1 plus momentum
  this is the Titans memory; with a linear memory, gamma = 1 and c the whole context it
  is least squares, which the authors expect to suffer from having no pruning.
- [Anderson and Schooler, 1991](https://www.psychologicalscience.org/journals/psychological-science/j.1467-9280.1991.tb00174.x/)
  **[abstract/search record]**: memory availability relates to frequency, recency
  and pattern of exposure, and environmental need falls as a power of the retention
  interval. **[hypothesis]** A single store with such a priority is the baseline any
  intermediate level must beat.
- [Semantic drift compensation, Yu et al., 2020](https://arxiv.org/abs/2004.00440)
  **[search record]**: estimates old prototypes' drift from current data, without
  exemplars. **[hypothesis]** Baseline for retrieval-driven key refresh.
- [Coherent gradients, Chatterjee, 2020](https://arxiv.org/abs/2002.10657) and
  [Chatterjee and Zielinski, 2020](https://deepai.org/publication/making-coherence-out-of-nothing-at-all-measuring-the-evolution-of-gradient-alignment)
  **[abstract]**: gradients from similar examples reinforce; per-example coherence
  also reaches high values on random labels. **[derived]** Alignment alone does not
  separate shared structure from memorization.
- [Learning progress against irreducible noise](https://arxiv.org/abs/2509.25438)
  **[abstract]**: unlearnable transitions give prediction error but no progress.
  Difference: exploration reward, not memory consolidation. **[hypothesis]** Scoring
  candidate updates on later experience should reject repeating noise.
- [Bjork's new theory of disuse](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2021/01/RABjorkELBjorkJARMAC2020ForPostingSingleSpaced.pdf)
  **[search record]**: storage and retrieval strength are separate; unused memories
  lose accessibility rather than storage.

**[derived]** Under FIFO eviction a singleton's reach is traces held / surprise
rate, and any online rule must treat a cause's first occurrence like a one-off. No
match was found in these searches for residual grouping with prequentially scored
shadow updates routed to binding or shared stores. Not a novelty claim; the
searches were few and MINAS's full text is unread.

### Sleep: consolidation into persistent parameters

Reading date 2026-11-01. User supplied the
[Google Research overview](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/);
followed its link to [Language Models Need Sleep](https://arxiv.org/pdf/2606.03979)
(v2). **[read: introduction, sections 3.1–3.4 excerpts, section 4 results
excerpts; not a full proof, implementation or compute audit]**.

**Source evidence [read].** Sleep transfers fast memory into slower parameters
through distillation and imitation learning, activates additional low-rank
experts, and rehearses synthetic examples. Faster experts are reset after
transfer. Expansion can use preallocated masked parameters. The paper reports
retention improvements in sequential language learning and long-context tasks.

**Difference [derived from the read mechanism and our contract].** This is
direct prior work for temporary-to-durable consolidation. It does not by itself
establish our narrower claim: that independent later sensory outcomes can
validate unresolved surprises under a fixed total byte budget. Synthetic
rehearsal can transfer an existing belief without supplying independent
evidence that it is correct. Neither observation is a claim that Sleep fails.

**Program implication [hypothesis].** Before freezing the lifecycle contract,
consider a periodic consolidation control using the same bounded buffer,
reader, observations and total update allowance. Compare scheduled replay or
distillation with evidence-triggered consolidation, so a gain cannot be
attributed solely to delaying updates. This would be a small control inspired
by consolidation, not a reproduction or test of Sleep. Count dormant capacity,
teacher state, synthetic generation and consolidation compute if implementing
a closer variant. Record quality versus total cost, including offline work.

**Open reading.** Audit the paper's capacity trajectory, consolidation schedule,
training budgets and controls before making any quantitative comparison. The
broad consolidation motivation is established prior work; any contribution
claim here must concern the specific evidence-selection mechanism and fixture.

### Sleep: scheduled consolidation and dreaming

Reading date 2026-11-01. [Behrouz et al., "Language Models Need Sleep", arXiv
2606.03979v2](https://arxiv.org/abs/2606.03979) **[read: introduction, §3.2–3.4,
§4 tables, §4.2, appendix on self-distillation; appendix B and figure values not
read]**, from the [Google Research post](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/).

- **Shown.** A chain of memory blocks with ordered update frequencies. At step
  multiples of chunk sizes, a new low-rank expert is added to the slower block, all
  other parameters are frozen, the faster block is distilled into it (on-policy
  distillation plus imitation reward), and the sender's experts are reset. Then
  dreaming: generate synthetic samples with a randomly added expert, keep top-k by
  a gradient-based score plus b random, fine-tune an isolated copy on each, and
  reward generation if downstream measure τ improves (Eq. 5). Reports on sequential
  Manchu and Kalamang translation that performance rises with consolidation stages
  (Figure 5, values not read). No-context SQuAD: base 31.9, SEAL 46.7/43.2, Sleep
  48.9/46.2 (n = 1 / n = 200); no dreaming 35.7/36.2; without gradient selection
  47.1/45.2; without the random expert 48.0/44.7. ARC: 80% on 8 held-out tasks after
  filtering unsolvable ones. Math reasoning gains of 1–3 points. The tables read show
  no intervals.
- **Differs.** LLMs with task measures; a text world without bounded total memory;
  trigger is a fixed schedule. Selection applies to dreams, not to which experience
  is consolidated. No exhaustion behaviour is given for the pre-allocated experts.
  It frames forgetting as "insufficient capacity". Whether τ uses labels is not
  stated in the sections read, and it follows SEAL (not yet read).
- **[derived]** In the SQuAD ablation, nearly all gain comes from dreaming, and little
  from gradient selection or memory levels, on one benchmark without intervals. So
  it does not establish that the consolidation hierarchy causes retention there.
  Upward distillation into frozen-protected new parameters resembles our
  [retention remedies](retention-remedies-v1-results.md) destination. There, on M,
  `consolidate-cache` 0.459 against `rehearse-M` 0.471 and `frozen-snapshot` 0.502,
  and `consolidate-random` 0.273: uninformative cues transferred far less. Different
  scale and task, so context only. **[hypothesis]** Dream quality may matter for the
  same reason.
- **[hypothesis]** Scheduled unselective consolidation is the alternative the
  accumulated-surprise proposal must beat. Resetting the fast block at each sleep
  also ends any evidence that has not resolved yet, so it should struggle with
  evidence spanning sleeps, unless the slow block accumulates it. Its frequency chain
  is an established form of the intermediate level in the
  [notes](continuous-sensory-memory-notes.md), and should be a matched-resources
  comparator in any test of whether a middle level earns its place.

**Sleep against merging units (2026-11-02, §§2.2–3.4 reread).** **[read]** Sleep's
consolidation moves knowledge between time scales: it adds a low-rank expert to the
slower block, freezes all else, distills from samples the teacher generates, not raw
data, then resets the sender's experts. Capacity grows at each sleep. **[derived]** It
never asks whether two stored things are the same. The
[consolidation pilot](sense-consolidation-v1-pilot.md) asks only that, in one store,
at fixed capacity, keeping raw traces. So the two are complementary, and the word
covers both. **[hypothesis]** Merging could prepare a store for transfer to a long-term
network, which this project has not combined. Generated cues in place of stored traces
would cut the pilot's main storage cost, but `consolidate-random` above warns that
uninformative cues transfer poorly.

### Relevant update: several experiences together

Search date 2026-11-01, for the [relevant update draft](relevant-update-v1-contract.md). Sources: arXiv pages of the works below; ATLAS proceedings pp. 5-6 and Appendix G, recorded above.

- [Maximally interfered retrieval, Aljundi et al., 2019](https://arxiv.org/abs/1908.04742) **[abstract]**: replays stored samples whose predictions a coming update would most worsen, with less forgetting than random replay reported. Difference: it chooses memories to protect from an update. **[hypothesis]** A candidate for what to rehearse, and a contrast with choosing experiences to form an update.
- [A-GEM, Chaudhry et al., 2018](https://arxiv.org/abs/1812.00420) **[abstract]**: an efficient form of GEM, reported about as cheap as EWC. The constraint itself was not on the page read. **[hypothesis]** Baseline family for protecting old knowledge; read before use.
- [Orthogonal weight modification, Zeng et al., 2018](https://arxiv.org/abs/1810.01256) **[abstract]**: learns context-dependent mapping rules online from about ten samples per rule without catastrophic forgetting. Difference: the setting and networks differ, and a context module supplies context. **[hypothesis]** Directly relevant to few-example rules; a low-rank form suits a bounded budget. Read before use.
- [von Oswald et al., 2022](https://arxiv.org/abs/2212.07677) **[abstract]**: linear self-attention can implement gradient descent on a regression loss. ATLAS Appendix G attributes a recursive least-squares memory to them; the abstract does not describe one, so this is **secondary**.

**[derived]** Omega's window, Titans' single step and least squares on a whole context are one family in ATLAS Appendix G, varying c and optimizer. So an update-rule test can vary them inside one rule, with γ the published slot for experience selection.

### Long-term memory: a network beside sparse relations

Reading date 2026-11-01, for the [long-term memory draft](long-term-memory-v1-contract.md).

- [Nested Learning and Hope, Behrouz et al., 2025](https://arxiv.org/abs/2512.24695) **[abstract; §7 through a page reader, not the full paper]**: memory as a chain of blocks with a spectrum of update frequencies. Block l updates every C(l) steps, gradients summed over the chunk (Eq. 71), with blocks nested, sequential or independent and aggregated (Eqs. 72-74). §7.1 argues that forgetting in one block is mitigated because older knowledge persists in lower-frequency blocks. No forgetting numbers were in the part read. Difference: Transformers on language tasks, and blocks in one gradient flow; ours is detached and trained on residuals. **[hypothesis]** A low update frequency may slow interference in a long-term network.
- **[derived]** Detaching memory and training it on the readers' errors leaves the host's weights bitwise unchanged, which removes the transfer cost seen with dense units inside the host, and leaves only the combined answer at risk.
- **[familiar; read before use]** Momentum encoders (MoCo) for stable keys; sparse distributed activation and k-winners for limiting interference.

No match found in these few searches for a neural memory trained on sparse readers' residuals beside a relational core. Not a novelty claim.

### Interference protection after the long-term-memory pilot

Reading date 2026-11-01. Revisited primary sources for retaining A while
learning B: Sleep §§2.2, 3.2–3.3; Hope §7.1; and the OGD proceedings abstract.
No new experiment is frozen by these notes.

- **[read]** [Sleep](https://arxiv.org/html/2606.03979v1#S3.S2)
  allocates a new low-rank expert to avoid interference with previously
  consolidated knowledge. Knowledge seeding updates expanded parameters;
  fast experts are subsequently reset. Preallocated masked capacity is an
  implementation option. **[derived]** This is protection and allocation,
  not merely a larger fully plastic MLP. It consumes capacity and does not
  establish indefinite retention at fixed bytes. Protected weights also need
  a preserved input representation and retrieval path.
- **[read]** [Hope §7.1](https://arxiv.org/html/2512.24695v1#S7.SS1)
  uses multiple memory blocks with distinct frequencies and knowledge transfer;
  its retention argument depends on knowledge surviving in other blocks.
  **[derived]** Sweeping C for one memory does not test that mechanism. Sleep
  §2.2 notes that multiple timescales can still forget when blocks update,
  motivating additional consolidation.
- **[abstract]** [Orthogonal Gradient Descent, Farajtabar et al. 2020](https://proceedings.mlr.press/v108/farajtabar20a.html)
  projects new-task gradients into directions intended to preserve old-task
  outputs without retaining old raw examples. **[derived]** This is a direct
  alternative to parameter isolation; retained gradient-subspace state must
  count toward memory. **[open]** Read the full algorithm and approximation
  limits before specifying it for this nonlinear network.
- **[derived from implementation]** Top-k activation does not reserve disjoint
  parameters for earlier associations: units can be reused, and learned
  key/value projections and output bias/scale remain shared. Zero explicit
  decay does not constrain B's gradient to preserve A. **[hypothesis]** Compare
  protected allocation and constrained updates on old recall during new
  acquisition, with replay as a reference, rather than another width/frequency
  sweep. This targets interference directly.

### Interference in a long-term network

Search date 2026-11-01, after the [long-term memory pilot](long-term-memory-v1-pilot.md), where pairs in a network fell from 0.52-0.59 to 0.16-0.23 after a second set. Search records and abstracts only, unless stated; read before use.

- **Sparse local memory.** [Bricken et al., 2023, SDMLP](https://arxiv.org/abs/2303.11934) **[search record]**: Top-K activation, no bias terms, L2 normalization and non-negativity give continual learning with no task labels, boundaries or replay; momentum optimizers go stale with high sparsity; ablation finds each component necessary. [Shen, Dasgupta and Navlakha, 2021](https://arxiv.org/abs/2107.07617) **[search record]**: fly circuit: sparse high-dimensional random expansion, then only synapses from active neurons to the associated output change, without backpropagation. [Lan and Mahmood, 2023](https://arxiv.org/abs/2310.01365) **[search record]**: sparse gradients matter beside sparse representations. **[derived]** Our `k-sparse` had biases, signed tanh units, no normalization, momentum 0.9 and moving learned projections, so it was not a test of these. Translation memory's one-step associative memory, a random sparse expansion with local writes, kept 0.907 of old words at 1,024 units, 41 active, and lost 0.02 on batch 1 over the schedule, on frozen keys: project evidence for this family.
- **Gating to non-overlapping units.** [Masse, Grant and Freedman, 2018](https://arxiv.org/abs/1802.01569) **[search record]**: mostly non-overlapping units per task; gating alone only partly helps across 100 tasks; with synaptic stabilization, little forgetting. [Iyer et al., 2021](https://arxiv.org/abs/2201.00042) **[abstract]**: active dendrites and sparsity route by context. Difference: context signals are given or inferred at task level. **[hypothesis]** A frozen key could serve as context per item.
- **Protecting allocated units.** [Serra et al., 2018, HAT](https://arxiv.org/abs/1801.01423) **[abstract]**: learned hard masks conditioned on previous tasks' masks. Task-based. **[familiar]** PackNet, SupSup. Difference: task identity and boundaries; a stream has neither.
- **Importance of weights.** [Kirkpatrick et al., 2016, EWC](https://arxiv.org/abs/1612.00796) **[abstract]**: slows learning on weights important for earlier tasks. **[familiar]** Synaptic intelligence and memory aware synapses estimate importance online, and task-free variants exist. Bricken et al. report combining SDMLP with EWC on CIFAR-10.
- **Multi-timescale synapses.** [Benna and Fusi, 2016](https://www.nature.com/articles/nn.4401) **[search record]**: synapses with several internal variables store many memories over long timescales, capacity roughly linear in synapses. **[hypothesis]** Hope's low-frequency blocks are a coarser relative.
- **Replay.** [Shin et al., 2017](https://arxiv.org/abs/1705.08690) **[abstract]**: a generator interleaves synthetic past data, storing none. Retention remedies found bounded rehearsal effective here, on marked pairs.

Follow-up reading, 2026-11-01: [sparse local memory notes](sparse-local-memory-notes.md)
records primary-paper excerpts for SDMLP, FlyModel and Elephant, and maps the
user's unit-vector routing proposal to address/content memory. Distinguishes
sparse activation from local parameter updates, records static-manifold and
optimizer limitations, and states the conditions under which disjoint writes
protect old reads. No experiment frozen.

The [shared-core sparse memory draft](shared-core-sparse-memory-v1-contract.md)
first asks whether one core can hold both relations and the identity of new
symbols, since a code trained only on relations read 0.11 and 0.14 on 90-way
identity on existing caches (seeds 392, 393; exploratory). It then tests local
content writes with learned addresses on an address code that passes: units are
allocated on mismatch by a vigilance threshold, and addresses follow the cues
they take. Protection of old reads is derived and conditional. Associative
retrieval is deferred to its own contract. The closest established methods are
adaptive resonance theory and leader clustering for the addresses and Kanerva's
sparse distributed memory for the fixed-address reference, all **[familiar; read
before use]**.

## Shared output scorer and naming-only transfer

Added 2026-11-04, for the [shared output v1 contract](shared-output-v1-contract.md). Every
source is **[familiar]** (recalled, not re-read for this entry).

**What the sources showed.**
- **1-N scoring in knowledge-graph completion.** ConvE (Dettmers et al. 2018) encodes
  (head, relation) once and scores every entity through one scorer. TransE (Bordes et al. 2013)
  and NTN (Socher et al. 2013) define other shared scoring functions over triples.
- **Cross-encoders** (Humeau et al. 2020) score a query and a candidate jointly. They are more
  accurate than separate embeddings and cost one pass per candidate, hence retrieve-then-rerank.
- **Contrastive predictive coding** (van den Oord et al. 2018) scores candidates for the next
  item instead of regressing its embedding.
- **Selective classification** (Chow 1970; Geifman and El-Yaniv 2017) and **open-set rejection**
  (Scheirer et al. 2013): abstention at a threshold. Softmax scores are overconfident off the
  training distribution (Hendrycks and Gimpel 2017).
- **Acquired equivalence** (Honey and Hall 1989) and **stimulus equivalence** (Sidman and Tailby
  1982): stimuli trained to a common outcome come to share other associations.
- **SCAN add-primitive** (Lake and Baroni 2018): sequence models taught a primitive in isolation
  failed to compose it. Meta-learning across many such episodes succeeded (Lake and Baroni
  2023).
- **Inductive entity embedding** from a few links (Hamaguchi et al. 2017).
- **Key–value memory reading** (Sukhbaatar et al. 2015; Miller et al. 2016) and **modern Hopfield networks** (Ramsauer et al. 2020):
  a read returns a weighted stored value in the space of the stored patterns, which can feed the next hop. Entity linking resolves a
  mention to an entity before querying its relations.
- **Representation-mediated learning** (Holland 1981): a stimulus that evokes another's representation inherits its associations,
  the proposed mechanism of acquired equivalence.

**How the setting differs.** These are symbolic entities or text tokens, not perceptual codes
through a sparse core. There is no continual relearning in these sources. The equivalence
studies are behavioural.

**Program implications.**
- **[derived]** A shared scorer gives recall and verification one logit, so their agreement
  holds by construction. It does not tie the slot a name was taught in (the `same` source) to the
  slot it is tested in (the history's current item). Transfer therefore still depends on the
  name's code being substitutable. The contract measures this with a substitution bound and a
  nearest-count diagnostic.
- **[derived]** A scorer removes the squared-distance pull of candidates toward a predicted
  point, which is the relearning contraction route identified locally. It may move
  interference into the shared scorer.
- **[derived]** Routing every relational slot through identification (a grounded participant state) holds substitutivity by
  structure: a name inherits relations to the degree it is identified. Stage 0 supports this as local evidence: grounded 0.56 / 0.50
  against raw 0.39 / 0.27 ([stage 0](shared-output-v1-stage0.md)).
- **[analogy]** Acquired equivalence suggests that equality training can carry other
  associations to the new item. Whether it does in this core is the open question.

### Pilot follow-up: preserve identification before redesigning training

Discussion 2026-11-04; proposals, not a frozen contract or authorization to run.

- **[read: local pilot write-up]** The [pilot](shared-output-v1-pilot.md) reports
  that replacing names with recalled participants at evaluation raises block-3
  `gp` N1 from 0.47 to 0.77. Grounding throughout training damages familiar
  continuation. Gradient starvation and soft-read blurring remain confounded.
- **[derived]** First compare direct, learned-participant and harness-supplied
  correct-participant reads on the same existing checkpoints at every read point.
  The last is a diagnostic control, not an available model capability. This can
  separate identification failures from failures downstream without retraining.
- **[abstract]** Chaudhry et al., [On Tiny Episodic Memories in Continual
  Learning](https://arxiv.org/abs/1902.10486), studies replay from small episodic
  memories. Search on 2026-11-04: `site.arxiv.org continual learning experience
  replay tiny episodic memories 2019`; abstract-level search result reviewed.
  This is a standard retention baseline, not evidence that replay solves this
  core's naming problem or rejection of unfamiliar symbols.
- **[analogy / untested]** Compare raw training with and without replay of
  naming-only matching evidence, using frozen sensory traces rather than images.
  Evaluate both direct and grounded reads; keep names out of relational teaching.
  Match total update and row budgets, disclose replay exposures and retained bytes,
  and report selection/training/read costs separately under the computation rules.
  Diagnose grounded-training failures separately if that architecture remains a
  research objective. No confirmation criteria are set by this discussion.

**Added 2026-11-04 for the [K arms](shared-output-v1-k-arms.md)** (all **[familiar]**).
- **Bi-encoders** (DSSM, Huang et al. 2013; dense passage retrieval, Karpukhin et al. 2020) score ⟨Q(query), K(candidate)⟩:
  cheaper over many candidates than cross-encoders and less accurate (Humeau et al. 2020).
- **Shared embeddings across positions.** Siamese verification (Bromley et al. 1993) and contrastive metric learning (Chopra,
  Hadsell and LeCun 2005) use one encoder on both sides of a sameness judgment. TransE, DistMult (Yang et al. 2015) and ComplEx
  (Trouillon et al. 2016) share an entity's embedding across head, tail and candidate positions, which is how evidence about an
  entity in one position reaches a query in another.
- **[derived]** A key tower over candidates alone does not reach the query slot a name is tested in. One K applied to slots
  and candidates does: the sameness loss then acts in the space every relation reads.
- **Known failures:** collapse of a shared encoder under sameness alone; symmetric scores when the query ignores slot order; a
  dot product accepts one half-space per query, which limits relations with separated valid fillers.

## Relation codes, relational bottleneck and context normalization

Added 2026-11-05 for [relation codes check v1](relation-codes-check-v1.md). [familiar] sources, recalled.
- **Relations as offsets in entity space:** TransE (Bordes et al. 2013), word2vec analogies. A relation code computed from two
  entity codes inherits any distortion of those codes; part A measured this (pair relation reads equal identity reads).
- **Separate relational processing:** Relation Networks (Santoro et al. 2017); ESBN (Webb, Sinha and Cohen 2021) keeps a key
  pathway that never sees object features; CoRelNet (Kerg et al. 2022) and the relational bottleneck (Webb et al. 2023) pass on
  only similarities between objects; Abstractors (Altabaa et al. 2023). All report out-of-distribution generalization of
  relational rules to new objects.
- **Temporal context normalization** (Webb et al. 2020): standardizing embeddings over a task context lets relational rules
  extrapolate. It removes a per-context shift and scale, which matches the bend measured in the two-script reads.
- **One space or two:** tensor product representations (Smolensky 1990) bind roles and fillers from separate spaces; holographic
  reduced representations (Plate 1995) bind in one. The Tolman–Eichenbaum Machine (Whittington et al. 2020) learns a structural
  code once and binds new sensory objects to it.
- **[program implication, measured]** what survives interference in the pilot is order within a kind's set, not pair relations;
  context-level structure is the candidate to store.

## Consistency critic, probe selection and asking

Added 2026-11-05 for [consistency critic v1](consistency-critic-v1.md); design notes in
[learned consistency critic](learned-consistency-critic-notes.md). Searches 2026-11-05: `Settles active learning literature
survey 2009 query by committee`; `Seung Opper Sompolinsky 1992 query by committee COLT`; `Dasgupta "Two faces of active
learning" sampling bias`; `Arazo pseudo-labeling confirmation bias`; `Li Hoiem Learning without Forgetting`; `Shazeer sparsely-gated
mixture-of-experts`; `Shim joint active feature acquisition`; `Heckerman dependency networks JMLR 2000`; `BeliefBank Kassner`;
`ConCoRD Mitchell 2022`; `Set Transformer Lee`; `Neural LP Yang Cohen 2017`. Abstracts and search excerpts inspected; no full
texts read.

- **Critic as masked conditional prediction = a dependency network.** **[abstract]** [Heckerman et al.
  2000](https://jmlr.org/papers/v1/heckerman00a.html) learn one conditional distribution per variable given the others (graph
  possibly cyclic) instead of a joint model, and extract joint predictions by Gibbs sampling. The critic is that estimator
  over operator outputs. **[familiar]** It is also pseudo-likelihood (Besag 1975) and BERT-style masked prediction. **[derived]**
  Conditionals learned this way need not be consistent with one joint distribution; a critic used only to flag one masked
  output at a time does not need them to be.
- **Repair from consistency = BeliefBank and ConCoRD, with the constraints learned.** **[abstract]** [BeliefBank (Kassner et
  al. 2021)](https://arxiv.org/abs/2109.14723) keeps a language model's answers in a memory, revises clashing beliefs with
  weighted MaxSAT over given constraints, and re-queries with known beliefs as context; it reports better accuracy and
  consistency. [ConCoRD (Mitchell et al. 2022)](https://arxiv.org/abs/2211.11875) scores pairwise compatibility of candidate
  answers with an NLI model and solves weighted MaxSAT, with no retraining. **[derived] Differences:** both use a symbolic
  solver and hand-given or NLI-given relations; here the dependencies are learned from examples and repair stays neural (the
  user's purely-neural requirement). Their improvement is context, not a prediction for this system.
- **Probe selection = active feature acquisition and sparse routing.** **[abstract]** [Shim, Hwang and Yang
  (NeurIPS 2018)](https://papers.neurips.cc/paper/2018/hash/e5841df2166dd424a57127423d276bbe-Abstract.html) choose features
  one at a time at test time, trading accuracy against acquisition cost, with an order-invariant set encoding of acquired
  features and a reinforcement-learned policy. [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538) route each input to a
  sparse top-k of experts with a learned gate, for capacity without proportional compute. **[derived]** The v1 selector is a
  one-shot top-k gate distilled from leave-one-out influence, not a sequential policy; a sequential acquisition policy is the
  stronger known alternative and is deferred.
- **Set input for the critic.** **[abstract]** [Set Transformer (Lee et al. 2019)](https://arxiv.org/abs/1810.00825) models
  interactions among set elements with attention, in time linear in set size via inducing points. v1 uses the simpler
  DeepSets mean pool (**[familiar]**, Zaheer et al. 2017); interactions across units then come only through the head.
- **Learned rules over relations.** **[abstract]** [Neural LP (Yang, Yang and Cohen 2017)](https://arxiv.org/abs/1702.08367)
  learns first-order rules end to end, with attention as rule confidences, compiled from a differentiable logic (TensorLog).
  **[derived]** It builds in the chain-rule form; the critic builds in only which outputs are grouped into one unit.
- **Asking = active learning.** **[abstract]** [Seung, Opper and Sompolinsky
  (1992)](https://www.sciweavers.org/node/177772) query where a committee trained on the same data disagrees most; in their toy
  models error falls exponentially in the number of queries, against an inverse power law for random sampling.
  [Settles (2009)](https://minds.wisconsin.edu/handle/1793/60660) surveys query strategies (uncertainty sampling,
  query-by-committee, expected model change). [Dasgupta (2009/2011)](https://mlanthology.org/alt/2009/dasgupta2009alt-two)
  treats sampling bias as the basic obstacle: adaptive querying can converge to a worse classifier even with unlimited labels;
  remedies are importance weighting or managing sampling regions. **[derived]** Here the disagreement is among one system's
  operator outputs, not among models; uncertainty sampling (the reader's own margin) is the cheaper established baseline.
- **Repair towards critic predictions = self-training.** **[abstract]** [Arazo et al.](https://arxiv.org/abs/1908.02983) find
  naive pseudo-labelling fits its own wrong predictions (confirmation bias); mixup, label noise and a minimum number of
  labelled samples per batch reduce it. **[derived]** Repair needs retained teaching evidence in every update.
- **Output preservation = Learning without Forgetting.** **[abstract]** [Li and Hoiem](https://arxiv.org/abs/1606.09282) keep
  old-task outputs by distillation on new-task data while fine-tuning. It is the standard form of the anchor-preservation loss.
- **Confidence.** **[abstract]** Guo et al. 2017 (above, critic notes): neural confidence is poorly calibrated; this does not
  make margin a weak error detector on in-distribution data. **[derived]** v1 therefore reads whether conflict adds to margin.
- **Error detection baselines.** **[abstract]** [Hendrycks and Gimpel 2017](https://arxiv.org/abs/1610.02136): the maximum
  softmax probability separates misclassified from correct examples (AUROC, AUPR) across vision, language and speech, though
  poorly calibrated; it is the baseline the critic's margin comparison stands for. [Jiang et al. 2018, trust
  score](https://arxiv.org/abs/1805.11783): agreement with a modified nearest-neighbour classifier flags errors with higher
  precision than the classifier's confidence in their tests. [Lakkaraju et al. 2017](https://arxiv.org/abs/1610.09064):
  high-confidence errors ("unknown unknowns") arise from systematic training bias and cannot be found by confidence-based
  active learning; they search data partitions with a bandit under an oracle budget. Searches 2026-11-05: `Hendrycks Gimpel
  baseline detecting misclassified`; `Lakkaraju unknown unknowns AAAI 2017`; `Jiang Kim Guan Gupta To Trust Or Not To Trust`.
  For [consistency critic v1 follow-up](consistency-critic-v1-confident.md).

## Moving representations, representational drift and registration (added 2026-11-05)

For the [moving-representation notes](moving-representation-notes.md). All sources **[familiar]**, recalled and not
re-read; read before a contract cites them.

- **Representational drift with stable behaviour.** Driscoll et al. 2017 (parietal cortex, Cell), Schoonover et al. 2021
  (piriform, Nature), Deitch, Rubin and Ziv 2021 (visual cortex, Current Biology) and Marks and Goard 2021 (visual cortex,
  Nature Communications) report single-unit tuning changing over days to weeks while the animal's behaviour is stable.
  Rule et al. 2020 (eLife) show the drift concentrates in directions orthogonal to a stable linear readout, so task
  information survives in a changing population. Masset, Qin and Zavatone-Veth 2022 review proposed functions.
  **[derived]** The analogue here: relations learned on a slow core frame are the readout, and what moves is the core's
  own object encoder on frozen sense codes. Locally the [two-script reads](two-script-reads-v1.md) and
  [relation codes check](relation-codes-check-v1.md) show the reading against collections failing while the codes
  stay separable, and order surviving one block back only, so the analogue is partial.
- **Drift as implicit regularization.** Ratzon, Derdikman and Barak 2024 (eLife): noise in learning walks the solution
  along flat directions of the loss, so pinned directions are those the loss constrains. **[derived]** Kinds far from
  the present loss are flat; near kinds are partly pinned through shared features, which matches the
  [drift measurement](sense-drift-structure-v1.md) (near-family maps predict 0.52–0.86 of movement after a change, far
  ones none) and the [natural-image pilot](natural-memory-v1-pilot.md).
- **Self-healing codes.** Rule and O'Leary 2022: a Hebbian and homeostatic readout tracks drift if the coarse geometry of
  the variable is preserved (in [reading notes](natural-memory-prior-work.md)). **[derived]** Coarse geometry is kept
  one block back and not two in the relation codes check.
- **Weight-importance against output-distillation penalties.** Kirkpatrick et al. 2017 (EWC) and Zenke, Poole and
  Ganguli 2017 (SI) act on weights; Li and Hoiem 2016 (LwF) distils outputs on present data. **[derived]** They should be
  predicted differently here: a weight penalty can pin a far absent kind, output distillation only where present data
  shares its features. Not run in this program; the [retention remedies contract](retention-remedies-v1-contract.md)
  excluded them. Known failure: protection degrades as the number of tasks grows, and importance estimates assume the
  old data distribution. The target is the core's object encoder, since the senses are frozen.
- **Adapters and routing.** Rebuffi, Bilen and Vedaldi 2017 (residual adapters) and Aljundi et al. 2017 (Expert Gate,
  in reading notes): per-domain parameters on a shared trunk avoid forgetting but need the domain at test time, chosen
  by a gate. **[read: local results]** Routing by frozen sense code is partly tested here: label-free centres read like
  told ones on streams where kinds arrive in runs ([adapt pilot](sense-adapt-v1-pilot.md)), and no growth rule matches
  one centre per kind on all four stress streams, with batch-mean rules failing on interleaved and gradual streams
  ([growth pilot](sense-growth-v1-pilot.md)). **[derived]** A registration per kind is this family and inherits that
  cost.
- **Registration of embedding spaces.** Conneau et al. 2018 (Procrustes alignment of word-embedding spaces, with a small
  seed dictionary or none); Mikolov, Le and Sutskever 2013 (linear maps between languages); Webb et al. 2020 (context
  normalization, the unsupervised per-kind affine case already in `src/relation_codes.py`); Shen et al. 2020 (backward
  compatibility as a training objective, in reading notes). **[read: local results]** Context normalization half helps
  in the relation codes check (`two-ctx` 0.62 against `two` 0.50; `count-ctx` 0.46 against `count` 0.69), which says the
  rest of the bend is not affine; codes stay 0.99 separable with labels, and 0.84–0.87 for untaught Thai
  ([equality diagnostics v2](equality-diagnostics-v2.md)). **[derived]** The informative test is savings: labels needed
  by a per-kind map for a taught, drifted script against an untaught one, from core and from sense codes.
- **Unmatched versus stale.** **[derived]** For one item, novelty and staleness give the same internal signal (nothing
  matches well); open-set rejection baselines (Hendrycks and Gimpel 2017, in the critic section) detect the signal but
  cannot name its cause. At the population level they separate: many once well-matched readings losing margin together
  is drift. With memory keyed on frozen senses ([sense stable pilot](sense-stable-v1-pilot.md): 0 of 5,400 reads drift),
  the place to measure it is the core's readings. The single case needs outside evidence: a second sense, rehearsal, or
  asking.

## Drift remedies: regularization, rehearsal and per-kind adapters

Added 2026-11-05 for [drift remedies v1](drift-remedies-v1.md), from the [moving-representation notes](moving-representation-notes.md).
Searches 2026-11-05: `Kirkpatrick 2017 PNAS elastic weight consolidation Fisher diagonal`; `Zenke Poole Ganguli synaptic
intelligence`; `Pellegrini latent replay 1912.01100`; `van de Ven Siegelmann Tolias brain-inspired replay`; `van de Ven Tolias
three scenarios 1904.07734`; `Rebuffi Bilen Vedaldi residual adapters`; `De Lange continual learning survey`; `Yu semantic drift
compensation`; `Moschella relative representations`; `Rabanser Failing Loudly`; `hubness CSLS cross-lingual`; `savings relearning
catastrophic forgetting`. Abstracts and search excerpts inspected; no full texts read.

- **Taxonomy.** **[abstract]** [De Lange et al. 2021](https://arxiv.org/abs/1909.08383) group continual-learning methods into
  replay, regularization and parameter isolation, and compare them on memory, computation and storage. The notes' three
  remedy places are this taxonomy.
- **EWC.** **[abstract / search excerpts]** [Kirkpatrick et al. 2017](https://core.ac.uk/download/81116384.pdf) add a quadratic
  penalty anchoring parameters to the previous task's solution, weighted by a diagonal Fisher estimate. Summaries report the
  constraints accumulate and performance degrades over many tasks. **Synaptic intelligence** ([Zenke, Poole and Ganguli
  2017](https://arxiv.org/abs/1703.04200)) estimates importance online along the training path instead of from a Fisher at the
  end. **[derived]** Storage is constant in the number of kinds but proportional to parameters: here about 2 × 105,610 floats
  (anchor and importance), 845 KB, more than rehearsal stores at small scale.
- **Regularization fails when task identity must be inferred.** **[abstract]** [van de Ven and Tolias
  2019](https://arxiv.org/abs/1904.07734): in class-incremental learning EWC-type methods fail and replay of previous
  experiences seems required; only replay does well in all three scenarios. **[derived]** Here the relation readers are shared
  and are not told the kind; only context normalization uses the kind. This is closer to domain-incremental learning, so the
  literature predicts partial protection from EWC at best.
- **Latent replay.** **[abstract]** [Pellegrini et al. 2020](https://arxiv.org/abs/1912.01100) store activations at an
  intermediate layer instead of inputs; with frozen lower layers this is equivalent to rehearsal from the input, with storage
  and compute savings. If the layers below keep training, stored activations age, which is not disruptive when they train
  slowly. **[derived]** A buffer of frozen sense codes is exactly the frozen-lower-layers case: no aging. It is rehearsal.
- **Generative replay without stored data.** **[abstract]** [van de Ven, Siegelmann and Tolias
  2020](https://doi.org/10.1038/s41467-020-17866-2) replay internal representations generated by the network's own
  context-modulated feedback connections, with no stored data, and report state-of-the-art class-incremental results on
  CIFAR-100. The stronger storage-free alternative; deferred.
- **Residual adapters.** **[abstract]** [Rebuffi, Bilen and Vedaldi 2017](https://arxiv.org/abs/1705.08045) switch one
  network between visual domains with small residual adapter modules, under 10% of the parameters per domain. The domain is
  given. **[read: local results]** Routing without a given domain is the cost measured by the sense growth pilot.
- **Near inputs predict an absent kind's drift.** **[abstract]** [Semantic drift compensation (Yu et al.
  2020)](https://arxiv.org/abs/2004.00440) estimates the drift of old class prototypes from the drift of current-task
  embeddings, exemplar-free. That is the notes' "near present inputs carry partial information".
- **Invariance by anchors.** **[abstract]** [Relative representations (Moschella et al.
  2023)](https://arxiv.org/abs/2209.15430) code each sample by its similarity to a fixed set of anchors, which gives invariance
  to latent isometries and rescalings. **[derived]** That cancels a movement shared by all items, not drift specific to one
  kind while the anchors stay pinned, which is the drift measured here.
- **Detection.** **[abstract]** [Rabanser, Günnemann and Lipton 2019](https://proceedings.neurips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html):
  two-sample tests on representations from a pretrained classifier detect dataset shift best in their study. The
  population-level drift signal of the notes is this family, applied to the core's own margins.
- **Collisions = hubness.** **[search excerpts]** In cross-lingual embedding mapping a few targets become nearest neighbours of
  many sources (hubness); CSLS ([Conneau et al. 2018](https://arxiv.org/abs/1710.04087)) penalizes hubs. The collision check
  (two forms judged equal to one count) is a hubness signal.
- **Savings.** **[search excerpts]** The relearning savings measure (Hetherington and Seidenberg, in [French
  1999](https://lead.ube.fr/wp-content/uploads/2023/09/000282-catastrophic-forgetting-in-connectionist-networks.pdf)) asks how
  quickly forgotten associations are retrained. French notes not all forgetting is "shallow" in this sense.

## Everyday relations: knowledge-base completion and semantic development (added 2026-11-05)

Survey for [everyday relations v1](everyday-relations-v1.md); the sources and evidence tags are in that contract's prior work
section. In short: the study is small closed-world knowledge-base completion (Hinton's family trees; Socher et al. 2013;
ComplEx). Progressive differentiation and overgeneralization of exceptions are predicted by Rogers and McClelland (2004) and,
exactly for deep linear networks, by Saxe, McClelland and Ganguli (2019). Kemp and Tenenbaum (2008) is the built-in-form
alternative the program excludes. Composition (CLUTRR) is the expected weak point of single-hop scorers. Locally, generic
relations v1 predicts weak fast mapping through a generic reader.

## Backtracking search over middle things (added 2026-11-06, for the conflict search v1 draft)

No search run on 2026-11-06; sources **[familiar]**, local results **[read: local results]**, implications **[derived]**.

- **Conflict loop v1's defects [read: local code]:** the loop's step used the same G_L, S_L and E_L as step 0 with
  gradients on, so the loss on withheld rows trained the one-pass reader; the trigger was recomputed from step 0's
  disagreement each training step and fired on 6% of withheld rows by the end; rows were withheld at random across
  relations, so the relation code was still taught one-pass. Implication: readers must be detached inside any loop or
  search, and whole relations withheld, before a loop can be said to have had to reason.
- **MINERVA** (Das et al. 2018) and **DeepPath** (Xiong et al. 2017): a learned agent walks a knowledge graph
  choosing a relation and next entity per step, trained by reinforcement on reaching the answer, with a readable path.
  Known failures: sparse reward and a beam needed at test. Implication: the forward walk is MINERVA's shape; the draft
  adds backtracking with a trace, two learned routes checking each hop, and a ranking loss from the search's own checks
  as reward shaping. (Added on external review of the first draft, which had missed both.)
- **Neural Theorem Provers** (Rocktäschel and Riedel 2017) and **Conditional Theorem Provers** (Minervini et al. 2020):
  differentiable backward chaining with soft unification and learned goal reformulation; exponential without k-max
  pruning. **Neural LP** (Yang et al. 2017) and **DRUM** (Sadeghian et al. 2019): learned sequences of base relations
  as matrix products over all entities. Difference from the draft: one committed middle at a time, a trace of failures,
  a universal learned reader as the scorer, no rule templates.
- **Expert iteration** (Anthony et al. 2017) and AlphaZero-style training: a proposal network improved from its own
  search's outcomes. Known failure: the evaluator's errors become the proposer's labels. The draft keeps the final truth
  label as the only ground truth and reads whether found paths are true paths.
- **Language-model reasoning [familiar]:** one pass is pattern completion over trained closures; the compositionality
  gap (Press et al. 2022) and out-of-distribution failure of grokked composition (Wang et al. 2024) predict that a
  one-pass reader does not compose beyond what it saw. Chain of thought works because the intermediate is discrete,
  committed, fed back through fixed weights and checkable; Tree of Thoughts (Yao et al. 2023) and
  reinforcement-learned backtracking (DeepSeek-R1, 2025) add search and revision on a frozen base. Unfaithful chains
  (Turpin et al. 2023) motivate reading the path, not only the answer. Implication: freeze the readers, train the
  controller on outcomes, commit the middle, read the program.
- **Negation as failure** (Prolog): exhaustion treated as false. The draft learns what exhaustion means, since the
  world has unknowns.
- **Neural stacks** (Grefenstette et al. 2015; Joulin and Mikolov 2015) and **inhibition of return** (Posner and Cohen
  1984): a fixed stack discipline with learned contents, and exclusion of tried candidates.
- **CLUTRR** (Sinha et al. 2019): facts given at question time force composition; lookup-table models fail on held-out
  chain lengths. Not the draft; the study after it, where the search step becomes a memory hop.
- **Path Ranking Algorithm** (Lao and Cohen 2010) **[familiar]** and the learned-walker literature's exhaustive
  baselines: unguided path enumeration between two entities finds many spurious paths, and with a known rule and a
  correct graph a bounded exhaustive search finds every proof within its depth. **[read: local results]** Conflict
  search diagnostics v1 confirms both on the frozen v1 reader graph: unguided BFS within 5 hops connects 0.97 of false
  pairs; the right program with exhaustive search finds valid proofs for 0.77 of true withheld rows, against 0.13–0.36
  for the trained walkers. Implication: separate learning the program from executing it, and measure the reader graph
  before judging a controller.
