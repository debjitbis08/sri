# Prior work

Living survey of established methods related to this project, maintained under
[literature review for contracts](literature-review.md). It covers the
continuous three-module line: the three-module, graded, local consolidation and
local contrastive studies. The discrete core-mediation and graph-reachability
studies are not yet reviewed.

## Search record and evidence levels

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
- Established alternatives to consider before inventing new mechanisms:
  - bidirectional predictive coding for the objective question;
  - correlated-nuisance fixtures to separate prediction from alignment;
  - anchor-penalty SI and training-regime controls for forgetting.
