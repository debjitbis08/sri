# Prior work

Living survey of established methods related to this project, maintained under
[literature review for contracts](literature-review.md). It covers the
continuous three-module line: the three-module, graded, local consolidation and
local contrastive studies. The discrete core-mediation and graph-reachability
studies are not yet reviewed.

## Search record and evidence levels

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
