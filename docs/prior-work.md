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
