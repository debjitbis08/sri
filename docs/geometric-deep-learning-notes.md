# Notes on Geometric Deep Learning

2026-10-26. Reading notes, not a contract or result.

Source: Bronstein, Bruna, Cohen and Veličković,
[Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)
(v2, 2021). Read: §3.4 scale separation, §3.5 the blueprint, §4.5 gauges,
§5.7 recurrent networks and §5.8 LSTMs. Everything else was skimmed or is
cited from memory.

Evidence tags follow [prior work](prior-work.md):

- **[read]**: the section was read for these notes;
- **[familiar]**: a standard result stated from prior knowledge;
- **[derived]**: derived here for SRI, numerically checked where noted;
- **[analogy]**: a structural resemblance only;
- **[hypothesis]**: untested.

## Why this book is useful to us

The book is a program in the spirit of Klein's Erlangen program. It
describes CNNs, group-equivariant CNNs, GNNs, Deep Sets, Transformers, RNNs
and LSTMs in one vocabulary, with symmetry at the centre: a domain Ω,
signals on it, a symmetry group G, and three building blocks [read, §3.5]:

- local equivariant maps;
- coarsening;
- global invariant pooling.

It explains existing architectures from these principles. It does not use
them to propose new ones, since that is not its aim.

For SRI, the value is translation. If our components can be stated in this
vocabulary *with the conditions checked*, we can use the book's standard
results instead of rederiving them. We can also see which parts of SRI fall
outside it: that is where SRI would have to make its own case.

## The bridge: invariant energies give equivariant equilibria

GDL describes feed-forward compositions. SRI computes by minimizing an
energy, so the translation needs one lemma [derived, elementary]:

> Let a group act linearly on the activities, mapping clamped coordinates to
> clamped and free to free. If E(g·s) = E(s) for every g, and the minimizer
> over the free activities is unique, then the equilibrium map is
> equivariant: the equilibrium for clamped values g·x is g times the
> equilibrium for x.

Our Q is positive definite, so uniqueness holds. The design question "which
group should the layers be equivariant to?" therefore becomes "which group
should the energy be invariant to?".

The reparametrization version, E(g·s; g·θ) = E(s; θ), is the gauge
statement in [§4.5 below](#gauges-45). The
[energy scaling analysis](energy-scaling-analysis.md) already used a special
case of it.

## Topic notes

Each topic gives what the book says, why it is interesting, whether it
applies to us, and how.

### Linear invariants are group averages (§3.5)

- **What.** If f is linear and G-invariant, it depends on x only through
  the group average of x. Linear *equivariant* maps are rich, but
  expressive invariants need a nonlinearity between equivariant layers
  [read].
- **Why interesting.** It is a precise version of the checkpoint's "linear
  modules cannot hold structure their surfaces do not linearly contain".
- **Applies?** Yes, to every linear SRI module.
- **How.**
  - A linear module can remove an additive nuisance *subspace* by
    projection, which is what the linear relational fixture tests.
  - It cannot become invariant to nuisance that acts as a group on the
    signal, or to arbitrary token partitions as in the finite quotient
    fixture. Either would amount to averaging.
  - This supports the proposal's demand for a separate capacity audit
    before the finite fixture, and it gives the nonlinear line a target:
    nonlinearity between equivariant maps.

### Representations, intertwiners and Schur's lemma (§3.1, standard theory)

- **What.** Linear maps that commute with group actions are intertwiners.
  For an irreducible representation, the self-intertwiners form a division
  algebra over R: R, C or H. They are only scalars when the
  representation is absolutely irreducible [familiar].
- **Why interesting.** The certified mismatch tests in
  [relational linear preflight v1](relational-linear-preflight-v1-contract.md)
  (P0 and P2) ask whether some S satisfies S F′_a = F_a S. That is an
  intertwiner question, so our draft-check results are textbook statements
  [derived, numerically checked]:
  - **Commutant dimension 1 for a rotation plus a reflection.** The pair
    acts absolutely irreducibly on R².
  - **Dimension 2 for a single rotation.** It is irreducible over R but not
    absolutely; its commutant is a copy of C.
  - **Same-eigenvalue scrambles are barely detectable** (bound 0.009). For
    pairs of 2 × 2 matrices, the joint similarity class is fixed by tr A,
    tr B, det A, det B and tr AB [familiar]. Such a scramble keeps the first
    four and changes only tr AB.
- **Applies?** Directly.
- **How.**
  - **Design rule:** a target binding is identifiable up to scale exactly
    when the action representation is absolutely irreducible. A reducible
    action set leaves a separate basis freedom for each block. The number
    of free binding parameters follows from the multiplicities of the
    irreducible parts.
  - **Evaluation rule:** relational descriptors must include joint
    invariants such as tr AB, not only per-operator spectra.
  - **Framing:** cross-domain transfer of relational structure is
    recognizing two representations as isomorphic. The shared abstraction
    is the isomorphism class, and a domain binding is an intertwiner. This
    is the continuous counterpart of the proposal's MDP-homomorphism target
    [analogy].

### Deformation stability: Fourier against wavelets (§3.3-3.4)

- **What.** The Fourier modulus is exactly shift-invariant but unstable:
  for deformations ε away from a shift, the change is O(1) however small ε
  is. Wavelet decompositions are stable, O(ε), because their atoms are
  localized in space and frequency. The local measurements they produce are
  not yet invariant and must be combined progressively towards coarse
  scales [read].
- **Why interesting.** Exact invariance to an idealized group is not
  enough. A useful representation must also degrade gracefully under
  transformations *near* the group.
- **Applies?** Partly.
  - The observation index set of the current fixture has no geometry,
    because renderings are random, so spatial Fourier and wavelet bases
    mean nothing there.
  - The idea carries over by analogy to relational descriptors. Per-operator
    eigenvalues behave like a Fourier modulus: invariant but blind to the
    joint structure [analogy].
- **How.** When we define "the same relational pattern", ask whether the
  descriptor is stable under small changes of the operators, not only
  invariant under basis change. Whether the certified mismatch bound is
  Lipschitz in the operators is a question to settle before using it as a
  similarity measure [hypothesis].

### Scale separation and coarsening (§3.4)

- **What.** A function is locally stable at scale j if f ≈ f_j ∘ P_j, where
  P_j is a coarse-graining that merges nearby points. Interactions can then
  be separated across scales, as in the renormalization group and the fast
  multipole method. Coarsening needs a metric on the domain [read].
- **Why interesting.** It is the book's main answer to the curse of
  dimensionality, and it reads naturally as a definition of abstraction.
- **Applies?**
  - **Not to 8 → 4 → 2.** That is channel compression, not coarsening of a
    domain.
  - **Yes to time.** Sequences live on a one-dimensional grid.
- **How.**
  - **A temporal hierarchy:** coarse operators trained to equal products of
    fine ones (dyadic, G^(2^j)) would answer an L-step query with about
    log₂ L re-encodings instead of L/2. That directly attacks the measured
    per-window gain loss [derived from the draft numbers].
  - **The catch:** with mixed actions the number of distinct action words
    grows exponentially. A coarse level cannot store one operator per word;
    it must abstract over sequences. It could chunk recurring sequences
    (macro-actions, as in the options framework) or carry coarse state that
    does not depend on fine actions [familiar, not yet reviewed]. That is
    where genuine abstraction, and almost certainly nonlinearity, enters.
  - **Link to the original note:** "Tension → Persistence → Transformation
    → Release" is a coarse-time description of a long, detailed trajectory.
    The shared pattern sits at a coarse temporal scale and the sensory
    detail at a fine one [analogy].

### The blueprint (§3.5)

- **What.** f = A ∘ σ_J ∘ B_J ∘ P_{J−1} ∘ … ∘ P_1 ∘ σ_1 ∘ B_1:
  - B: linear equivariant layers;
  - σ: pointwise nonlinearities;
  - P: coarsening;
  - A: global invariant pooling.

  One wide local layer cannot represent long-range interactions, but a
  composition of local equivariant layers with interleaved coarsening can,
  while keeping stability. Shallow global invariants trade off against
  deformation stability [read].
- **Why interesting.** It gives a checklist for any architecture: what is
  Ω, what is G, and which blocks are present.
- **Applies?** It sorts SRI's pieces: three candidate domains, and two of
  the four blocks.
  - **Candidate domains:**
    - **The observation index set:** no structure, as above.
    - **The module graph** (chain 0-1-2): each edge has its own untied
      translation, so in GNN terms it is message passing without weight
      sharing and no permutation symmetry applies.
    - **Time:** the natural domain for the temporal window.
  - **Blocks:** SRI currently has B (linear maps) and channel compression.
    It has no σ and no explicit A.
- **How.** Every proposed extension should state its Ω, G and blocks. The
  nonlinear line should state what σ is and what it must be equivariant to.

### Gauges (§4.5)

- **What.** A gauge is a choice of frame at each point. Gauge
  transformations act separately on each point's feature vector. Gauge
  equivariant maps have coordinate expressions that do not depend on the
  gauge [read].
- **Why interesting.** Each module's latent coordinates are such a frame,
  and each domain's binding is a frame too.
- **Applies?** Yes, as follows [derived, plus analogy where marked].
  - **Inner translations act as connections** carrying coordinates between
    module frames [analogy].
  - **Decode-and-re-encode fixes the gauge**, because every step returns to
    module 0's frame.
  - **The activity priors fix most of the gauge.** Isotropic priors reduce
    each module's freedom from GL(k) to O(k) and pin the scale. That is the
    shrinkage and gain-loss mechanism measured in the
    [energy scaling analysis](energy-scaling-analysis.md).
  - **Earlier results already fit this reading.** Rotation-only
    translations came within 1.31× of free ones, and inner-family fixed and
    rotation-only translations matched "as the gauge argument predicts"
    ([clamped-energy ablation](clamped-energy-ablation-v1-results.md)).
- **How.** Score only gauge-invariant quantities, as the preflight already
  does (X_L, and mismatch modulo S). Treat the priors' prior-strength
  dependence as a gauge-fixing term rather than an incidental
  regularizer.

### Recurrence, weight tying and fixed points (§5.7)

- **What.** An RNN applies one shared update R along a one-dimensional time
  grid. It is shift-equivariant only with that sharing and with an initial
  state that is a fixed point of R(0, ·), which can be found by iterating a
  contraction (Banach). A single RNN layer has depth equal to the sequence
  length, which is the source of vanishing and exploding gradients [read].
- **Why interesting.** Our rollouts are recurrences, and our settling is a
  contraction.
- **Applies?** Yes.
  - **The temporal window breaks shift symmetry.** In
    [relational linear preflight v1](relational-linear-preflight-v1-contract.md),
    modules 0, 1 and 2 have untied weights. Decode-and-re-encode partly
    restores shift structure by reusing module 0. In GDL terms, tying the
    weights is the standard prior, so untied weights are a choice that the
    learning contract should declare.
  - **The prior strength may set both gain and speed** [hypothesis].
    Settling converges at a rate set by the smallest eigenvalue of the free
    Hessian, which weak priors make small. The preflight smoke test needed
    270-830 settling steps to reach 1e-3. If this holds, prior strength
    controls both gain (loss grows with λ) and settling speed (slower as λ
    shrinks).
- **How.**
  - Decide tying explicitly in the learning contract: tied weights give the
    standard symmetry prior and fewer parameters, while untied weights match
    the preflight.
  - Check the prior trade-off against the preflight's saved parameters: the
    smallest free-Hessian eigenvalue against steps needed, and against λ.

### Gating and time warping (§5.8)

- **What.** Requiring a model class to be invariant to time warping
  τ(t) leads to the update h(t+1) = Γ·R(z, h) + (1 − Γ)·h, with
  Γ ≈ dτ/dt in (0, 1). That is the gated RNN, and its gate biases set the
  characteristic memory time (the chrono initialization). Class invariance
  is not zero-shot transfer; a pure rescaling τ₂ = ατ₁, however, transfers
  by rescaling the gates alone, Γ₂ = αΓ₁, with R unchanged [read].
- **Why interesting.** It derives gating from a symmetry rather than
  presenting it as a trick. It also names our gain decay: a slightly
  contracting recurrence is the forward form of the vanishing-signal
  problem.
- **Applies?** Strongly.
  - **Known remedies for our gain decay.** Rescaling T so that |eig G| = 1
    is the linear version of orthogonal or unitary recurrent networks
    [familiar]. LSTMs instead keep a retained cell path with forget-gate
    bias near 1.
  - **The local learner lacks a long-horizon mechanism.** The clamped-energy
    learner sees only windows. Like a network trained on short sequences, it
    has no pressure towards long-horizon accuracy. That is the review's
    objective mismatch, which the energy-scaling analysis predicts.
  - **Our energies can express a gate** [derived, by direct minimization].
    Minimizing a/2·‖c′ − c‖² + b/2·‖c′ − r‖² over c′ gives
    c′ = (a·c + b·r)/(a + b): the gated form with Γ = b/(a + b).
    - The constraint 0 < Γ < 1 follows from the precisions being positive,
      which a positive-definite energy already requires.
    - Our current energies have constant κ and priors, so Γ is fixed: they
      behave like a fixed-gate RNN, which assumes no warping.
    - Gating in SRI therefore means precisions that depend on the input or
      state, with a persistence term (an identity coupling between time
      steps) competing against the relational term. This resembles
      precision weighting in predictive coding [familiar]. It makes
      inference nonlinear, so it belongs to the nonlinear line.
  - **Tempo acts as a binding parameter.** Shared R with rescaled gates has
    the same split as our transfer design: shared relation, domain-specific
    binding. Here the domain-specific part is speed. In the original note's
    example, a drive and a painting session share a pattern at very
    different time scales [analogy].
  - **Two limits for linear dynamics** [derived and familiar]:
    - **The convex gate is first-order and contracts.** For a rotation F,
      the eigenvalues of ΓF + (1 − Γ)I have squared modulus
      1 − 2Γ(1 − Γ)(1 − cos θ) < 1. Exact warping of linear flows goes
      through the continuous-time generator, F = exp(A) with warped steps
      exp(ΓA).
    - **Reflections cannot be warped.** A matrix with determinant −1 has no
      real logarithm. In family A the rotation is a continuous flow and the
      reflection a discrete event.
- **How.**
  - The learning contract should expect window-only local learning to
    decay at long horizons, and should consider an orthogonality or gain
    constraint as a declared arm.
  - The nonlinear line could take precision-gated persistence as its first,
    principled nonlinearity.
  - Any tempo or duration experiment should use generator-based flows for
    the warpable relations and treat reflections as discrete actions.

## Fourier bases for communication between layers within a module

This section is not from a single book section. It applies the book's
harmonic-analysis thread (§3.4, and the generalized Fourier transform on
groups in §4) to communication between the x, h and c layers of one module.

**What the theory gives** [familiar, derived where marked]:

- **Convolution theorem.** Maps that commute with a commutative group
  action are diagonal in the Fourier basis. For non-commutative groups they
  are block diagonal, one block per irreducible representation (Schur).
- **The whole energy decouples** [derived, by the lemma above]. If each
  layer carries a representation of one group G, and W, V and the coupling
  translations are all intertwiners, then Q commutes with the group action.
  Inference then decouples into independent problems, one per frequency or
  irreducible component.
- **A numerical check** [derived, numerically checked]. Let the cyclic group
  Z₈ act on x by shifts, on h through Z₈ → Z₄, and on c through Z₈ → Z₂.
  - Equivariant W has 4 free parameters instead of 32, and V has 2 instead
    of 8.
  - Q commutes with the action.
  - Of x's frequencies, exactly {0, 2, 4, 6} reach h and {0, 4} reach c.
    The others are filtered deterministically.
- **Equivariant coarsening is not low-pass.** The quotient maps select which
  frequencies pass. With the natural quotients here, the core layer carries
  the constant mode and the alternating (Nyquist) mode, not the two lowest
  frequencies. Low-pass coarsening needs an additional smoothness prior, or
  a different construction such as projection onto the lowest |k| modes.

**Why it is interesting for SRI:**

- **Inference becomes exact per mode, and cheap.** The
  [energy scaling analysis](energy-scaling-analysis.md)'s scalar chain model
  becomes *exact* for each frequency instead of approximate from singular
  values. Exact inference costs O(n log n) by FFT, instead of the O(d³) of a
  dense factorization.
- **Learned singular vectors already play this role.** For a linear chain
  aligned with its singular vectors, everything decouples by mode. Fourier
  is the special case where the modes are fixed by a symmetry instead of
  learned, and so are shared by every layer and module. Translations
  between modules would then be diagonal phase shifts, just as a 2-D
  rotation is a phase shift of a complex coordinate.
- **Interpretability.** Each frequency channel carries one kind of
  structure, and the filtering above makes explicit which kinds reach the
  core.

**Does it apply to us?**

- **Not over unit indices in the current fixture.** No group acts on the 8
  observation coordinates, because renderings are random.
- **Yes over the action group.** The generalized Fourier transform of the
  action representation is its decomposition into irreducible parts.
  - In family A, the rotation's complex eigenbasis is the Fourier basis of
    SO(2). The reflection couples conjugate frequencies into one
    two-dimensional block, an irreducible representation of a dihedral
    group.
  - Nuisance is not acted on by the actions. Filtering it therefore
    amounts to keeping the isotypic components on which the actions act
    nontrivially [analogy, worth making precise].
- **Not to nonlinear layers without further work.** Pointwise
  nonlinearities mix frequencies (aliasing). §3.4 also warns that global
  Fourier features are unstable under deformations, which favours local,
  wavelet-like bases once a domain has geometry.

**How we could use it:**

1. **As an analysis tool now.** Decompose learned W, V and T into the
   isotypic components of the action representation. Report how much of
   each layer's gain sits in the components the actions act on, against
   the trivial component. This would sharpen the "signal gain" and
   "nuisance gain" measures we already use.
2. **As a design for a structured fixture.** Use observations with a real
   domain, for example a ring of 8 sensors with a pattern that moves around
   it. Compare modules parametrized in the Fourier basis (diagonal per
   frequency, about n parameters per layer) against dense ones at equal
   evidence, with the computation reported. GDL predicts gains in sample
   efficiency and cost when the symmetry is real, and a mismatch penalty
   when it is not.
3. **As a coarsening choice.** If a hierarchy is built on such a domain,
   decide deliberately which frequencies each layer should keep: the
   quotient-selected set, or low-pass with a smoothness prior. Do not
   assume equivariance yields low-pass.

## Edge features: relations as inferred states

**What the book says** [read]:

- **Node features only.** The book develops GNNs with node features. §4.1
  notes that edges or whole graphs can carry features, "but as this does
  not alter the main findings of this section, we will defer discussing it
  to future work".
- **§5.3 message passing** computes a vector per edge, m_uv = ψ(x_u, x_v),
  from the two node features, but takes no edge features as input. It cites
  Gilmer et al. 2017 (MPNN, whose messages also take edge features e_uv) and
  Battaglia et al. 2018 (Graph Networks) without developing them.
- **§5.5 equivariant message passing** uses one edge-derived quantity, the
  squared distance ‖x_u − x_v‖², which is computed from node coordinates.
- **§6** mentions edge-type-conditioned weights (Merkwirth and Lengauer
  2005).
- **§5.4 latent graph inference** learns which edges exist, not edge
  feature vectors.

The standard framework with node, edge and global attributes is Battaglia et
al. 2018,
[Relational inductive biases, deep learning, and graph networks](https://arxiv.org/abs/1806.01261).
It has explicit edge updates φᵉ(e, v_s, v_r, u) beside the node and global
updates [familiar, not yet read for us].

**Why it is interesting.** In SRI the relation between two modules lives on
an edge, and the relational proposal is about learning such relations.

**Does it apply to us?** Yes.

- **Our edges carry parameters, not states.** The inner translations T⁰¹ and
  T¹² live on the edges of the module chain. Indexing them by action, as in
  [relational linear preflight v1](relational-linear-preflight-v1-contract.md),
  gives edge-type-conditioned weights, as in Merkwirth and Lengauer or
  relational GCNs [familiar]. These are slow learned parameters, and in the
  preflight they are selected by a supplied action label.
- **An edge feature vector would be an edge activity:** a fast state
  inferred at equilibrium, like the node activities.

**How** [derived]:

    E_edge = κ/2 · ‖c_v − T(r_uv) c_u‖² + (prior on r_uv),   T(r) = Σ_k r_k T_k

- **What it would give.** The edge latent r_uv softly selects among the
  bank's relations, so inference decides which relation holds instead of
  being told. The proposal notes that a supplied action selector does not
  demonstrate autonomous relation selection; inferring r from data would
  address that directly. It is also the natural first step towards the
  proposal's later "learned action correspondence" condition.
- **What it costs.** The energy becomes bilinear in (r, c).
  - Each block is quadratic given the other, so the energy is biconvex and
    can be minimized by alternating exact solves.
  - Uniqueness is lost: sign symmetries such as (r, c) → (−r, −c), and
    multiple minima, appear. The bridge lemma's uniqueness condition
    therefore fails, and symmetry statements must be made about the set of
    minima, or after fixing those symmetries.
- **The relation to gating.** A precision gate (see
  [gating and time warping](#gating-and-time-warping-58)) is a scalar edge
  latent controlling *how strongly* an edge couples. This is a vector edge
  latent controlling *which relation* it carries. Both are edge activities,
  and both make inference nonlinear in the same way. They could be designed
  together as the first nonlinear extension.
- **What a first test would need.**
  - A fixture where the relation on an edge changes from episode to
    episode and must be inferred from the clamped evidence, for example
    which action produced the observed transition.
  - The audits this line already uses: a capacity witness, identifiability
    of r up to its symmetries, and settling against the alternating exact
    solve.
  - A review of Battaglia et al. 2018, Gilmer et al. 2017, and
    mixture-of-experts or switching models for the selection step, before a
    contract.

## What the book does not cover

These are the parts SRI must supply or justify on its own terms:

- **Inference as energy minimization.** The nearest framework is implicit
  layers and deep-equilibrium models [familiar, not reviewed]. The bridge
  lemma above carries symmetry statements across.
- **Local learning rules, replay, consolidation and sleep.** The book
  characterizes function classes, not how they are learned.
- **Learning the symmetry.** The book assumes G is known. The relational
  proposal asks for the action representation itself to be learned and
  aligned across domains. The relevant neighbours are symmetry discovery
  and learned invariance (for example Augerino, and Lie-algebra discovery)
  [familiar, not reviewed].

## Candidate follow-ups

In rough order of value:

1. Record these mappings in the [prior-work survey](prior-work.md),
   pointing here, with the evidence tags.
2. In the learning contract: decide time-weight tying explicitly; expect
   long-horizon decay from window-only local learning; consider a declared
   orthogonality or gain arm.
3. After the preflight run: check the hypothesized trade-off between prior
   strength, gain and settling speed against the saved parameters.
4. Analysis: decompose learned maps into isotypic components of the action
   representation.
5. Proposal candidates, each needing its own literature review first:
   - a temporal multiscale hierarchy (review options and hierarchical
     multiscale RNNs);
   - precision-gated persistence as the first nonlinearity (review
     precision weighting and gated RNNs);
   - inferred edge relations, T(r) = Σ_k r_k T_k, possibly together with
     gating (review Battaglia et al. 2018, Gilmer et al. 2017 and switching
     or mixture-of-experts models);
   - a structured-domain fixture for Fourier-parametrized modules.
