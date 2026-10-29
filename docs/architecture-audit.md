# Architecture audit: what is required, what is standard

Status: **draft for discussion, revision 2 (2026-10-29), after an external review.
Nothing run; not an experiment contract.** Follows the
[number-line checkpoint](number-line-checkpoint.md). Proposes the next step: list every
component of the number-line architecture, name the standard component it corresponds
to, and find a simpler architecture that is *sufficient* for the results already
obtained. It also checks the line against the program's founding hypothesis. Changes
from revision 1 are listed under [Review and corrections](#review-and-corrections).

## Researchers' decisions (2026-10-29)

1. **Building specific rule heads was a mistake.** The *next*, *same* and *larger* heads,
   each with its own loss, install the relations instead of letting them arise. Future
   designs do not add relation-specific heads.
2. **Not reading answers from the output module was a mistake.** Collection matching on
   core codes moved the answer into the shared space (fact 5). Future designs read answers
   from the output module.
3. **Stage 1 is a standalone standard network** trained on the same pilots, to check
   whether the results still hold without SRI. The audit keeps the current heads and core
   readout for this, because it audits the results already obtained. That is not an
   endorsement of either. The compiled-equivalent checks move to stage 2.
4. **After stage 1: the standard network is the default** for extending the number line.
   SRI gets one focused confirmation experiment on its candidate cross-sense advantage
   ([cross-sense confirmation v1](cross-sense-confirmation-v1-contract.md)).
5. **After the confirmation:** the relation-specific heads are removed, and the design is
   made fully generic (to be designed with the researchers).
6. **Imagery is a probe, not a result to audit.**

## The hypothesis to hold the line against

From the researchers (2026-10-29), restating [research program §1–2](research-program.md):

> **H.** Reusable knowledge emerges from interactions among modules, rather than being
> explicitly trained into a shared feature space.

The audit asks two questions:
1. Which components do the established results actually need?
2. Do the results bear on H at all?

## What the current network computes

Number-line network: `src/number_systems_net.py` (`MultiSRI`), solved in
`experiments/number_construction_part_a.py` (`Coupled`) and
`experiments/number_construction_part_d.py` (`Sequence`).

Notation: `f` is the front-end feature (the clamped outer layer), `y` the free states,
`Q_yy` and `Q_yx` the blocks of the energy's Hessian, `c` the core-inner code (16 dims).

**Derived facts** (from the code, for fixed parameters; no experiment needed):

1. **Perception compiles to one linear map per sense.** The energy is quadratic and
   `f` is its only clamp. So `c = M_r f`, with `M_r = −(Q_yy⁻¹ Q_yx)` restricted to the
   core-inner rows. `M_r` is 16 × 64. The CNN front ends remain nonlinear.
   - At inference, perception is equivalent to: CNN front end → one linear layer →
     16-dim code.
   - Training through the exact equilibrium is implicit differentiation of that linear
     solve (OptNet, deep equilibrium models; noted in
     [prior work](prior-work.md#number-core-network)).
   - Compiling the equilibrium for fixed parameters says nothing about learning. The
     coupled parameterization still determines how `M_r` changes during training.
2. **The output module has no readout role, but it is not inert.** Answers are read
   from core codes, and the output module is never clamped or read. Eliminating its
   states exactly leaves a structured, learned quadratic penalty on the core states. That
   penalty changes `M_r`, the code-space precision `P` (fact 3) and the training
   gradients. Deleting the module deletes the penalty, so removal is a real change, not
   an identity.
3. **Revision is affine least squares with a Schur-complement precision.** Minimizing
   the energy over every state except `c` leaves `½ (c − c₀)ᵀ P (c − c₀)`, with
   `c₀ = M_r f` and `P = ((Q_yy⁻¹)_cc)⁻¹`, the Schur complement onto the core-inner
   block. With the context terms added, the exact revised code is

   `c_rev = (P + μ(I + AᵀA))⁻¹ (P c₀ + μ(A c_prev + t) + μ Aᵀ(c_next − t))`.

   - Because of the offset `t`, this is affine, not linear, in `(f, c_prev, c_next)`.
   - It is the posterior mean of a linear-Gaussian model, i.e. Gaussian belief
     propagation on a three-node chain.
   - `P` weights the evidence differently in different directions. Replacing it by a
     scalar is a different estimator, not a compiled equivalent.
4. **Imagery is the model's own reverse conditional.** Clamping a core state and
   solving outward gives the conditional mean of the features under the joint Gaussian
   defined by `Q`. That is a fixed linear (affine with `t`) decoder, but it is not
   determined by `M_r` alone. A regression decoder fitted on data, or nearest neighbours
   in code space, estimates something else. They are baselines, not equivalents.
5. **All task losses act on the shared code.** Matching, *next*, *larger*, multi-step
   and equality losses are applied to core-inner codes directly. Encoders and couplings
   are trained by backpropagation from those losses.

## Component inventory

| # | Component | Used by results | Standard equivalent | Exact compiled form (fixed parameters) | Simpler substitute (not equivalent) |
| --- | --- | --- | --- | --- | --- |
| 1 | Front ends: image CNN, 1-D sequence CNN | all | standard encoders | — | none (already standard) |
| 2 | Per-sense satellite module (`W`, `V`), three layer translations `T`, graded coupling `κ` | all | per-modality projection head of a shared-embedding model (CLIP, ImageBind) | part of `M_r` | one directly trained linear layer per sense |
| 3 | Core module (`W`, `V`), activity prior `λ` | all | same projection; `λ` acts as shrinkage | part of `M_r` and `P` | absorbed into 2 |
| 4 | Output module with fixed label codes | no readout role; shapes `M_r`, `P` and gradients (fact 2) | learned quadratic regularizer on the core | penalty term in `M_r` and `P` | remove (a real change) |
| 5 | Exact Cholesky equilibrium with implicit gradients | all | implicit layer (OptNet / DEQ) | `M_r` | explicit forward pass |
| 6 | *Same* judge: `b − softplus(s)·‖a − b‖²` | matching (all) | metric-learning logits with learned temperature (Snell et al. 2017; CLIP) | itself | cosine with temperature (optional) |
| 7 | Successor `S(c) = Ac + t`, judged by distance | 1, 2, 6, 8 | affine generalization of TransE ([prior work](prior-work.md#number-cross-relation-pilot-originality-assessment)) | itself | TransE, `A = I` (optional) |
| 8 | *Larger* head: linear score on `c_a − c_b` | 5, 8 | pairwise linear ranking (RankNet-style) | itself | — |
| 9 | Losses: 9-way matching cross-entropy, *next* BCE with sampled negatives, comparison BCE | all | InfoNCE-style contrastive alignment; knowledge-graph negative sampling | — | — |
| 10 | Multi-step objective (`S^k` matches the count `n + k`) | 8 | multi-step / simulation-error training (DaD, Venkatraman et al. 2015; latent overshooting, Hafner et al. 2019) | — | — |
| 11 | Rehearsal: bounded buffer, 500 episodes per language | 3, 4, 5 | experience replay (Rolnick et al. 2019) | — | keep, labelled as scaffolding (see below) |
| 12 | Curriculum: language blocks, ordered counting episodes | 3 | blocked curriculum | — | Part A found no order advantage |
| 13 | Revision through relational energy terms | 2 | Gaussian posterior mean / belief propagation | affine solve with `P` (fact 3) | scalar-weight least squares in code space |
| 14 | Imagery by outward solve | 7 | linear-Gaussian reverse conditional; predictive-coding generative direction | compiled decoder from `Q` (fact 4) | fitted regression decoder; code-space nearest neighbour |
| 15 | Violation detection by successor residuals | 6 | prediction-error anomaly scoring | itself | — |
| 16 | Persistent state between steps | none (never trained stably) | recurrent state / Kalman filtering | — | dropped |

Components 1, 6–12 and 15 are already standard. For fixed parameters, components
2–5, 13 and 14 compile exactly to linear algebra over the shared code: `M_r`, `P` and
the reverse conditional. Whether the coupled parameterization matters *during
learning* is open.

## Standard explanations for each result

From the checkpoint's "Established in pilots". Tags follow [prior work](prior-work.md).
A standard explanation shows the result does not *require* SRI. It does not show that a
standard model would reproduce it. That has to be tested.

| Result | Standard explanation | Tested without SRI? |
| --- | --- | --- |
| 1. Relations give meaning to ungrounded items | Relational structure aligns sparse correspondences in a shared space (TransE completion, seed-based entity alignment) [read, prior work] | no |
| 2. Revision recovers answers | Affine Gaussian posterior combining item evidence with neighbours' predicted codes (fact 3) [derived] | no |
| 3. Rehearsal against forgetting | Experience replay [search summary] | no |
| 4. Varied contexts help | Multi-view / multi-task learning [familiar] | no |
| 5. Untaught compositions (taps ↔ forms; *larger* on forms) | Alignment through a shared anchor modality ([ImageBind, Girdhar et al. 2023](https://arxiv.org/abs/2305.05665)) [familiar; link from the external review]; a head on a shared space applies to anything mapped into it | no |
| 6. Residuals flag violations | Prediction-error anomaly detection [familiar] | no |
| 7. Imagery near the right count | Reverse conditional of a linear-Gaussian model, read by nearest real input (fact 4) [derived] | no |
| 8. Composition after multi-step teaching | Multi-step objective removes short-step solutions (DaD) [familiar] | no |

**Hypothesis [plausible, untested]:** a standard shared-embedding model reproduces the
eight results within predeclared margins. It would have CNN encoders, one linear
projection per sense, the same three relation heads, the same losses, rehearsal and
curriculum. Optimization through the coupled parameterization could change what is
learned (deep linear networks train differently from direct maps: Saxe et al. 2014;
Arora et al. 2018 [familiar]). So this is a real open question. Checkpoint caveat 2
already names the missing comparison.

## Bearing on H

**Assessment: the current results do not establish H, and the experiments are not
designed to isolate it.**
- Every concept (quantity, successor, order) is shaped by losses applied directly to the
  shared code (fact 5). So whatever the system knows could have been put there by those
  losses.
- Compiling the equilibrium into `M_r` (fact 1) does not show that the coupling had no
  learning role. It shows only that, after training, perception is linear in the
  features.

The results are evidence that **a shared space with relational heads composes across
senses**. That matters for the program's later goals (instinct, curiosity). They do not
bear on H either way.

**How the line moved away from testing H (traceable decisions):**
1. Number concept / core network: backpropagation through exact equilibria, with the
   program's local learning rule explicitly deferred
   ([prior work](prior-work.md#number-core-network): "It is not a test of the program's
   local learning rule").
2. The two-problem framing (2026-10-28) accepted standard grouping for problem 1.
   Problem 2 (pressure to use relations) is orthogonal to H. Both were reasonable, but
   since then no experiment has had H as its question.
3. Collection matching moved the readout into the core code. The output module and the
   couplings now act only as shaping terms (fact 2), not as paths to an answer.
4. Rehearsal uses a bounded episode buffer. Research program §2.4 rules out a separate
   historical replay store in the target architecture. The Part D rehearsal is
   experimental scaffolding and is labelled as such from now on. Rehearsal from current
   structure (generated or dreamed replay) is the program's own commitment.

**What a test of H needs.** "Agreement between modules" against "a loss puts knowledge
there" is not a precise distinction, because agreement can itself be enforced by a loss.
A test has to be defined as an intervention on learning signals and interactions:
- **Signals:** which parameters receive which learning signal, and from where. For
  example, relation losses applied to the shared code, against relation losses applied
  only inside each module, with cross-module influence only through coupling at
  inference.
- **Interactions:** the same modules with coupling present, removed, or replaced by a
  fixed map. Include a **core-free control** (modules coupled to each other directly, or
  not at all).
- **Reuse:** measured on held-out material, such as a new sense, a new relation or an
  untaught composition, not on the trained tasks.

Candidate designs, to be made precise before any contract:
1. **Separate rules that converge.** Each sense gets its own successor and *larger*
   heads, trained only within that sense, with no shared rule parameters. Test whether
   the rules agree across senses and predict across them.
2. **A rule without its loss.** Train grounding and coupling, never a *larger* loss.
   Test whether an order relation is usable from the core without a fitted probe.
3. **One perspective against many.** Train the rule on one sense or on three, at
   matched examples. Measure it on a held-out sense and on hidden-outcome compositions.

Established names for knowledge from interaction between modules, which an H-test must
beat or name itself as:
- agreement between views: co-training (Blum and Mitchell 1998), deep CCA (Andrew et
  al. 2013), BYOL / SimSiam-style cross-view prediction;
- coordination through a shared workspace: Goyal et al. 2022; recurrent independent
  mechanisms, Goyal et al. 2021;
- consensus between part-whole columns: GLOM (Hinton 2021), with documented
  difficulties noted in [prior work](prior-work.md#implications-for-contracts).

All [familiar], not yet read for this audit.

## Proposed audit

Two stages. Stage 1 separates mathematical equivalence from stage 2's empirical
question: does the coupled parameterization help learning?

### Stage 1 (decided 2026-10-29, run): the standalone standard network

Results: [stage 1 results](architecture-audit-stage1-results.md).

`src/standard_net.py`, run by `experiments/standard_baseline.py`:
- **Model:** the same CNN front ends feed one `nn.Linear(64, 16)` per sense (standard
  initialization) into the shared code, with the same relation heads. No modules, no
  energy, no output module.
- **Shared with the pilots:** the runners' data, losses, curriculum, rehearsal and
  evaluations run unchanged. Seeds, data streams and encoder initialization are the same,
  so every run pairs with its SRI pilot run. The *larger* head's initial vector differs
  (it is drawn first from the generator).
- **Revision** is code-space least squares with scalar evidence weight 1 (against μ = 1),
  fixed before running.
- **Imagery** uses a ridge decoder from codes to features, fitted on 4,000 training inputs
  per sense. The SRI core-middle-layer target has no counterpart and is dropped.
- **The energy confidence feature** is undefined (reported as zero).
- **Runs:** Part D (5 arms), multi-step (4 arms), then the composition, projection and
  imagery analyses, seeds 100–102.
- **Comparison:** `experiments/standard_baseline_compare.py`, using paired differences
  and the proposed margins (0.05 for accuracies, 0.03 for AUROCs; still to be confirmed).
  It also compares the arm-to-arm effects within each architecture.

**Complexity.** Per encoded item, perception costs one 16 × 64 product against the SRI
Cholesky solve of a 272-state system. The factorization is O(272³) per update and per
evaluation solver; the triangular solves are O(272²) per item. The CNN front ends are the
same in both. Measured costs are in the runners' runtime files.

### Stage 2: compiled equivalents on frozen checkpoints (no training)

Checkpoints: the saved Part D (D-varied+rehearsal) and multi-step pilot models, seeds
100–102.

1. **Verify exactness.** Compile, per sense, `M_r`, `P`, the exact affine revision
   (fact 3) and the compiled imagery decoder (fact 4). Check that they reproduce the
   original solves numerically, and that every readout in the eight results is
   unchanged. Deterministic; a failure here is a bug or a wrong derivation.
2. **Measure the substitutes on the same frozen codes.** These are baselines, not
   equivalents:
   - revision with a scalar evidence weight (weight tuned on development items, tuning
     cost reported) against the `P`-weighted revision;
   - the output-module penalty removed without retraining: the size of its contribution
     to `P` and `M_r`, and the effect on readouts.

This stage shows how much each fixed-parameter mechanism depends on its exact form, at
the cost of evaluations only. The imagery comparison is dropped: imagery was a probe, not
a result to audit (researchers, 2026-10-29).

### Stage 3: the remaining training rungs

Run on the Part D (D-varied+rehearsal) and multi-step designs, with the same data
streams and evaluations. Each rung changes exactly one thing, relative to T0.

| Rung | Change from T0 | Tests whether ... matters for learning |
| --- | --- | --- |
| T0 | current SRI network (reproduces the pilot exactly) | (control) |
| T1 | output module removed | the output module's penalty |
| T2 | perception as one directly trained linear layer per sense, initialized to T0's initial `M_r` | the coupled parameterization, as an optimizer |
| T3 | T2 with standard initialization | SRI-derived initialization (one change from T2); this is stage 1's network |

The optional head-variant rungs (TransE successor, cosine *same*) are dropped: the
researchers decided against relation-specific heads (decision 1). T1 and T2 are needed
only if stage 1 finds a difference to localize.

**Revision and imagery at every rung.** T2 and T3 have no energy, so their native
revision and imagery are undefined. Every rung, T0 included, is therefore evaluated
with one **common code-space revision** (scalar weight) and one **common imagery
readout** (fitted regression decoder and code-space nearest neighbour). T0 and T1 also
report their native, exact forms. The common readouts compare training regimes. Stage 1
compares inference mechanisms.

**Decision rule (to fix before running).** Proposed:
- For each headline result, a **predeclared practical margin** δ (for example 0.05 for
  accuracies, 0.03 for AUROCs; to be decided).
- **Paired comparisons** against T0: same seed, same streams, difference per seed.
- Three outcomes per result and rung:
  - **equivalent:** every paired difference within ±δ;
  - **worse** or **better:** the mean difference beyond δ, with every seed in the same
    direction;
  - **inconclusive:** otherwise.
- **Seeds:** three cannot show equivalence with any confidence. Proposed: five seeds for
  T0 and T3 (the main comparison), three for the others.
- **Interpretation:** an equivalent simpler rung shows that the simpler model is
  *sufficient* for these tasks. It does not show that the architecture is minimal, or
  that coupling cannot help elsewhere.

Confirming whichever architecture survives needs fresh seeds and a frozen contract, which
is the checkpoint's "consolidate" direction.

**Cost:** stage 1 costs only evaluations. Each stage-2 pilot arm took about 24 minutes
on 11 workers for the multi-step design. Four required rungs, with the extra seeds, is
on the order of four to six hours. T2 and T3 should be cheaper per step (no Cholesky
solve). Costs are measured and reported as required by
[computation reporting](computation-reporting.md).

## Open decisions

1. Run stage 1 now (cheap, deterministic) and decide on stage 2 from it, or only T3
   first?
2. If T3 is equivalent: continue the number line on the standard model (instinct,
   Part C) and open a separate H line, or redesign the number line around H?
3. Margins δ for each headline result, and the seed counts.
4. Rehearsal: keep the buffer as labelled scaffolding for now, or move to rehearsal from
   current structure?

## Review and corrections

External review, 2026-10-29. Revision 1 overstated several claims. The main diagnosis
stands: SRI-specific benefit is unestablished, and the shared-embedding baseline is
overdue. Corrected in revision 2:
1. **Output module.** Revision 1 said it does no work and that R1 was a near identity.
   Eliminating it leaves a learned penalty on the core; removing it is a real change
   (fact 2).
2. **Revision.** Revision 1 gave the evidence weight as `Q_yy` and called a scalar-weight
   replacement near-identical. The exact precision is the Schur complement `P`, and
   revision is affine because of `t` (fact 3). The scalar form is a baseline.
3. **Imagery.** A fitted decoder or code-space nearest neighbours are baselines, not
   substitutions. The reverse conditional is not determined by the forward map (fact 4).
4. **Prediction.** "A standard model reproduces all eight results" was tagged
   [derived]. It is a plausible hypothesis, and the optimization caveat is now part of
   the conclusion.
5. **H.** "The opposite of H" was too categorical. The results do not isolate H. The
   agreement-versus-loss distinction is replaced by an intervention on learning signals
   and interactions, with held-out reuse and a core-free control.
6. **Ladder.** Revision 1's R2 left revision and imagery undefined, and R4 and R5 each
   changed two things. Now there is a stage of exact equivalents on frozen checkpoints,
   one change per rung, common readouts at every rung, and a decision rule with margins,
   paired comparisons and an inconclusive outcome. Success means sufficiency, not
   minimality.

## Search record

2026-10-29. No new searches. The mappings come from existing [prior work](prior-work.md)
entries (number core network, cross-relation originality, Part D). The ImageBind link
comes from the external review. The others are from memory [familiar]: deep linear
network dynamics, prototypical networks, RankNet, DaD, latent overshooting, co-training,
deep CCA, BYOL/SimSiam, shared global workspace, RIMs and GLOM. Before a contract is
frozen on any rung or on an H-test, these need reading and recording in
[prior work](prior-work.md), starting with ImageBind and the agreement-between-views family.
