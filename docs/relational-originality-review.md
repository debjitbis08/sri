# Relational line: originality review after the linear preflight

2026-10-27. Literature and analytical review, not an experiment or contract.
No new fitting runs. Frozen contracts and results are unchanged.

## Assessment

The linear preflight is a useful implementation and feasibility audit. It
does not establish a new abstraction method. Most of its qualitative outcomes
were predicted in the existing prior-work entry; several proposed follow-ups
would repeat textbook calculations or established methods. Passing the project's
proceed rule is not sufficient reason to fund a broad learning/transfer study.

The earlier suggestion to proceed directly to such a study is superseded by
this review. First specify what SRI would add over an established estimator or
learning procedure. A novel combination of names or components is insufficient.

## Reading scope and sources

Read the project's GDL notes, relational proposal, preflight contract/results,
runner's quadratic construction, energy-scaling analysis, and prior-work survey.
Revisited selected sections of the authors' current online book rather than
claiming to have reread the entire 2021 proto-book:

- [GDL Chapter 3](https://geometricdeeplearning.com/book/algebraicpriors.html),
  representation decomposition and Schur's lemma, especially §3.6.10 [read].
- [GDL Chapter 4](https://geometricdeeplearning.com/book/geometricpriors.html),
  §4.4.4 and the blueprint: domain coarsening and local/global maps [read].
- [2021 proto-book HTML](https://ar5iv.labs.arxiv.org/html/2104.13478),
  selected passages on linear invariants and the blueprint [read]. The HTML
  rendering available during review is incomplete; the full PDF exceeded the
  web tool's size limit. Recurrence/gating was checked against the original
  Tallec–Ollivier paper instead.
- [Venkatraman et al. 2015](https://cdn.aaai.org/ojs/9590/9590-13-13118-1-2-20201228.pdf),
  introduction and one-step versus simulation objectives [read].
- [Sun et al. 2022](https://arxiv.org/pdf/2201.00248), introduction and
  Theorem 8 on transferable dynamics [read].
- [Kipf et al. 2018, NRI](https://proceedings.mlr.press/v80/kipf18a/kipf18a.pdf),
  §§3–3.4, latent edge types, decoder and multi-step training [read].
- [Tallec and Ollivier 2018](https://arxiv.org/pdf/1804.11188), §1,
  continuous-time derivation and discretization of gating [read].
- [Memisevic and Hinton 2010](https://www.cs.toronto.edu/~rfm/pubs/factored.pdf),
  abstract/introduction on multiplicative transformation representations
  and visual analogy [read, opening only].
- [Arjovsky et al. 2016](https://proceedings.mlr.press/v48/arjovsky16.pdf),
  orthogonal/unitary recurrence [abstract and introductory excerpt].

Targeted searches also covered switching state-space models, Augerino,
deep equilibrium models, and local temporal credit assignment. Sources used
only at abstract level are identified below and in the shared survey. This is
enough to reject broad originality claims, not an exhaustive novelty search
for a future precisely specified algorithm.

## What is known, and what the preflight actually adds

| Observation or proposal | Established explanation / method | Remaining contribution here |
| --- | --- | --- |
| Rotation/reflection binding ambiguity | Intertwiners and commutants; solve simultaneous linear constraints [read, derived] | Generator-specific numerical conditioning and threshold checks |
| Linear equilibrium prediction | Eliminate free variables in a quadratic: a linear conditional map [derived] | Feasibility inside this architecture's clips and priors |
| One-step fits fail in long rollout | Equation-error versus simulation-error fitting; compounding error [read] | Magnitudes for this implementation, not discovery of the phenomenon |
| Frozen dynamics help a new rendering | Shared latent dynamics transfer across observation spaces [read] | Whether this learner offers any advantage at matched data and compute |
| Slow settling | Spectral convergence of a linear iteration [derived below] | Which modes are excited in the saved fits; no new convergence principle |
| Orthogonality/gain correction | Norm-preserving recurrence [abstract; algebra below] | A task-specific implementation choice, not a new long-memory mechanism |
| Precision-weighted persistence | Gated/leaky recurrence; quadratic weighted averaging [read, derived] | A distinct learning/inference advantage would need to be demonstrated |
| Inferring the relation rather than receiving its label | Switching models, NRI, multiplicative transformation models [read/abstract] | Equilibrium inference is a variant, not by itself a new capability |
| Fourier-parametrized ring fixture | Known equivariance imposed on a known symmetry [read] | Internal control unless it tests a separate, unexpected hypothesis |

The precise success of the clipped P1 search was not guaranteed by a generic
GDL theorem. That is legitimate engineering information. But reducing a known
linear predictor to a more elaborate parametrization is not an expressivity
advance. Similarly, Sun et al.'s theorem does not supply finite-sample guarantees
for our noisy partially observed fixture or our optimizer; it does establish
that the broad shared-dynamics/new-observation-space idea is already occupied.

NRI is especially relevant to the proposed nonlinear direction: it infers
discrete edge types and uses type-specific functions, with weighted sums under
a continuous relaxation. Its §3.4 explicitly uses multi-step prediction because
a short-term predictor can ignore latent relations. Its encoder/decoder and
training differ from SRI's equilibrium inference. That difference must become
a testable advantage before this is a research direction rather than a port.

## Questions to discharge analytically

### Settling speed is not an open mechanism

For fixed parameters and clamp mask, write the free energy as
E(s) = 1/2 sᵀHs − bᵀs + const, with H positive definite. Implicit Euler
with step h gives [derived]:

    s_(t+1) = (I + hH)^−1 (s_t + hb)
    e_t = (I + hH)^−t e_0
    ||e_t||₂ ≤ (1 + h λ_min(H))^−t ||e_0||₂.

With zero start, a sufficient iteration count for relative error ε is
ceil(log(1/ε) / log(1 + h λ_min(H))). The actual count depends on the
projection of the initial error onto the eigenmodes. The same argument applies
to the matrix of right-hand sides used in P3, with Frobenius norm.

Consequently, no sweep is needed to discover that the free Hessian controls
settling. At fixed weights, increasing a hidden activity prior adds a positive
semidefinite term to H, so λ_min cannot decrease. Exact observed convergence
counts need not be monotone, because the equilibrium and mode weights change.
Retraining changes H again. Prediction and training clamp masks also differ.
Use this formula if a solver diagnostic is needed; do not sell it as new theory.

### Small effective-operator errors can compound

For operators expressed in one common aligned basis, the difference between
two L-fold products is a sum of L terms, each inserting one operator error
between prefix and suffix products [derived by telescoping]. If true operators
have norm one and each estimated operator differs by at most δ, the product
error is at most (1 + δ)^L − 1, approximately Lδ when Lδ is small.

The preflight J_G is a normalized squared aggregate intertwining residual,
not a horizon-uniform rollout bound. Passing its tolerance does not exclude
consequential dynamics differences. A further ablation may quantify the error
budget, but it would diagnose an implementation, not establish a new principle.

### The noisy one-step oracle already explains objective mismatch

Let Kx = E[z|x]. Composing the optimal one-step observation predictor RF_aK
inserts KR at each intermediate step. The long-horizon conditional predictor
RF_(a_L)…F_(a_1)K conditions only once [derived; already in the contract].
Repeated shrinkage is therefore expected even with population-optimal
one-step fitting. Estimating latent dynamics while modelling observation noise,
or optimizing simulation error, are established alternatives. Merely beating
an attenuating least-squares reference does not establish an SRI advantage.

### Mismatch stability needs assumptions, not a new experiment

For the stacked Sylvester matrix A, J = σ_min(A)² / D, where
D = Σ||F_a||². The singular-value perturbation inequality
|σ_min(A+ΔA) − σ_min(A)| ≤ ||ΔA||₂ gives local Lipschitz control of J
on bounded operator sets with D bounded away from zero [derived/familiar].
This does not make J a metric on joint similarity classes. Singular minimizers,
conditioning, nonorthogonal coordinate changes and the probe-constrained
problem require separate treatment. In particular, a stable scalar residual
does not imply a stable recovered alignment near degeneracy.

## Corrections to the GDL notes

These qualifications supersede the corresponding informal statements in the
2026-10-26 notes; they are not reasons to launch additional studies.

1. **Linear invariant maps can retain signal.** For a finite or compact group,
   a linear invariant map factors through the average/projector onto the fixed
   subspace. It is not generally incapable of removing group-acting nuisance:
   it can retain a nonzero fixed signal subspace. The limitation is inability
   to retain richer orbit information outside that subspace [derived].
2. **Five scalar invariants do not classify arbitrary matrix pairs.** Extra
   hypotheses such as the appropriate irreducible/semisimple setting matter.
   For example, (I,I) and (I+N,I), with nonzero 2×2 nilpotent N, have the same
   tr A, tr B, det A, det B and tr AB but are not simultaneously similar
   [derived]. Keep the actual simultaneous linear-system test.
3. **Schur's lemma has conditions.** The compact/orthogonal, completely
   reducible setting supports the notes' representation-based ambiguity
   reasoning. It does not automatically prove finite-data identification of
   an encoder with unknown nuisance and incomplete excitation [read/derived].
4. **Unit-modulus eigenvalues are weaker than orthogonality.** A Jordan block
   with eigenvalue one grows under powers; individually benign operators can
   have unstable mixed products. Orthogonality in one common metric controls
   every mixed product. Scalar rescaling of general banks is not equivalent
   to a unitary-RNN constraint [derived].
5. **Isotropic residuals already restrict coordinate symmetry.** Transforming
   h−Vc or c_v−Tc_u by a nonorthogonal basis change changes its Euclidean
   norm, even without activity priors. GL freedom of a latent input-output
   realization is distinct from symmetry of this fixed-metric energy.
   Priors add restrictions; they are not the sole source [derived].
6. **Equivariance requires compatible metrics and masks.** For an orthogonal
   representation, an invariant quadratic has a commuting Hessian. For general
   linear representations the condition is ρᵀQρ=Q instead. A clamp mask must
   preserve the decomposition if the conditional solve is to decouple by mode.
   Fourier inference also includes per-mode solves and multiplicity blocks;
   O(n log n) is conditional on bounded block sizes and suitable structure
   [derived].
7. **No observation-space action representation has been supplied.** Random
   renderings need not forbid an induced representation, but one must define
   or learn it. Fresh nuisance draws are not a deterministic identity action
   on a trajectory. Arbitrary learned W,V,T cannot simply be decomposed into
   shared isotypic blocks without establishing compatible actions [derived].
8. **Dyadic composition reduces depth, not arbitrary-sequence work.** A
   balanced tree computes an L-letter matrix product with O(log L) parallel
   depth but O(L) work. Repeated identical operators permit repeated squaring;
   arbitrary mixed words still require reading their actions. Compression or
   reusable chunks need additional assumptions. This does not establish a
   need for nonlinear abstraction by itself [derived].

The elementary invariant-energy/unique-minimizer lemma remains useful and
correct with its stated feasible-set conditions. It is a bridge for applying
known symmetry results, not an originality claim. The weighted-average gate
identity is likewise elementary. Gate rescaling inherits continuous-time and
discretization qualifications, not exact arbitrary discrete-time warping.

## Direction and stopping rule

Do not prioritize more capacity sweeps, a Hessian-versus-settling correlation
study, a ring/Fourier demonstration, or a generic "multi-step beats one-step"
experiment as research contributions. Retain them only as cheap controls when
required for a concrete new claim. Do not treat inferred edges, gating or
multiscale composition as automatically original alternatives.

The potentially defensible question is narrower: **does a specified SRI
learning/inference mechanism provide a capability or resource advantage that
the closest established method lacks under the same constraints?** Candidate
constraints include local information, online operation, retained memory,
partial observations, or adaptation without source replay. None establishes
novelty by itself, and choosing one should follow the project's motivation.

Even local temporal learning is occupied. [Williams et al. 2026](https://proceedings.mlr.press/v306/williams26a.html)
analyze local approximations in linear RNNs [abstract]. [Potter and Rhodes
2026](https://arxiv.org/abs/2602.18131) report temporal predictive coding with
RTRL and conditional gradient equivalence [abstract; preprint, proof not
audited]. These must be reviewed in full before claiming that local
long-horizon learning is the missing contribution. Their equations are not
assumed equivalent to SRI's.

Before a new contract, write a one-page claim with:

- the precise established comparator and its relevant assumptions;
- the single mathematical or algorithmic difference;
- why the claimed effect is not already implied by known theory;
- the resource constraint or capability that makes the difference valuable;
- the smallest decisive test and a failure condition that ends the direction.

First try to answer that claim analytically. If no defensible difference
survives, close this fixture as an internal feasibility result and reconsider
the research question. If a difference survives, implement only its minimal
comparison, with computation accounting under the existing policy. A broad
learning/transfer/consolidation suite should follow evidence of that difference,
not precede its definition.
