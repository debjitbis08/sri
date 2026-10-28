# Number core network v1: experiment specification

Status: **draft, 2026-10-28, revised after external review the same day; prior-work
review done. Not frozen.** Before freezing: a pipeline pilot and a contract with
implementation hashes.

Builds on the [cross-relation pilot](number-cross-relation-pilot.md). There, sparse
cross-modal **equivalence pairs** plus learned succession aligned unpaired counts
exactly, but only when both relations were read off one shared representation.
This study moves the question into the program's own architecture: separate input
modules coupled through a **core**, with an output module, coupling at corresponding layers
through learned translations, and exact equilibria of a quadratic energy. It adds a
second kind of input, Morse, so that alignment between input kinds can only happen
through the core.

**It also changes the bridge between input kinds.** The pilot tied modalities with
cross-modal equivalence losses, which pull same-number codes together. Here the
primary bridge is a **shared classification label** at counts 1, 3 and 5. Two core
states can predict the same label without being close, and within-modality relations
do not directly close that gap. So whether label supervision is enough to make codes
coincide is a **new hypothesis**, not something the pilot established. Equivalence-loss
control arms (below) diagnose a failure.

## Question

Two input modules, one for images (MNIST digits and dice faces) and one for Morse
signals, are coupled to a core, which is coupled to an output module that produces
a number label. Labels are dense for MNIST and sparse for dice and Morse (counts 1,
3, 5 only). With succession learned in the core:

1. Do unlabelled dice and Morse counts get the correct label (transfer through the
   core)?
2. Do image and Morse inputs of the same number land in the same core state? Does
   shared-label supervision suffice for that, or does it need the pilot's
   cross-modal equivalence loss?
3. How much of either comes from succession, and how much from Morse's own visible
   counting structure?

## Architecture

```
 MNIST ┐
       ├─ image front end ─ image module ─┐
 Dice  ┘                                  ├─ core ── output module (number label)
 Morse ── Morse front end ─ Morse module ─┘
```

- **Front ends (nonlinear, trained by backprop):**
  - Image: the pilot CNN. Conv 1→16 (3×3, stride 2), conv 16→32 (3×3, stride 2),
    linear 1,568→128, linear 128→16, ReLU between layers (padding 1). 207,696
    parameters; about 0.45 M multiply-adds per image.
  - Morse: conv 1→16 (kernel 5, stride 2, padding 2), conv 16→32 (kernel 5,
    stride 2, padding 2), linear 1,024→128, linear 128→16, ReLU between layers.
    Input length 128. 135,952 parameters; about 0.22 M multiply-adds per signal.
  - Each produces the outer activity of its input module (16 units).
- **SRI modules (linear, quadratic energy):**
  - **Image and Morse modules:** outer 16 → middle 8 → inner 4, with generative maps
    `x ≈ W h` and `h ≈ V c`, as in the linear line. The outer layer is clamped to the
    front-end features.
  - **Core:** a full module, outer 16 → middle 8 → inner 4, with the same
    generative maps. It receives no direct evidence; all its layers are free and
    are driven through coupling with the input modules and the output module.
  - **Output module:** outer 16 → middle 8 → inner 4, the same shape as every other
    module, so all coupled layers correspond. The layer sizes only follow the
    halving pattern of the earlier 8 → 4 → 2 modules; the exact numbers are not
    meaningful. Each number k has a fixed label code `q_k`: ten orthonormal vectors
    in the 16-dimensional outer layer, drawn once from a named seeded stream and
    never learned. The outer layer is free at inference. The label is the
    `argmax_k ⟨x_out, q_k⟩`. It is not clamped during training; the label loss acts
    on its equilibrium activity.
- **Coupling between adjacent modules at corresponding layers**, as in the
  [graded three-module](graded-three-module-v1-results.md) and
  [correlated-nuisance](correlated-nuisance-v1-contract.md) studies: for each edge
  (image–core, Morse–core, core–output) and each layer l ∈ {outer, middle, inner},
  `κ_l/2 ‖z_b,l − T_l z_a,l‖²` with a learned translation per edge and layer.
  All translations are square (16 × 16, 8 × 8, 4 × 4).
  - **Primary topology: uniform per-unit coupling**, κ_l = b/28 on every layer,
    with budget b = 16κ_outer + 8κ_middle + 4κ_inner = 4 per edge. Uniform per-unit
    coupling gave the lowest error of six profiles in the graded study.
  - **Topology control: inner-only coupling**, κ_inner = b/4 = 1, at the same
    budget. In the [weak-prior topology](weak-prior-topology-v1-results.md) study,
    surface coupling transmitted surface-specific structure while inner coupling
    filtered it. Here the surfaces of the input modules carry kind-specific
    appearance (stroke style, pip layout, Morse timing) that should not reach the
    shared number code, so the topology may matter.
- **Activity prior** λ = 0.001 on every hidden layer, and on all three core layers
  (the core outer is free), following the [checkpoint](linear-line-checkpoint.md)
  finding that strong priors cause gain loss. Symmetric energy.
- **Succession in the core:** an affine operator `S(c) = A c + t` on core codes,
  shared by all input kinds. Translation on a number line is the structural form
  argued earlier, so S is affine rather than a free MLP. This is a stronger
  structural prior than the pilot used, and a risk if the core code is not line-like.
- **Inference for one input:** clamp that input module's outer layer. The
  equilibrium is over that input module's hidden layers, the core and the output
  module. **The other input module is left out** of the energy for that example. If
  included without evidence, it would match the core exactly except for its priors,
  adding only a small prior-driven attenuation. The core code `c(x)` is the core's
  equilibrium activity: all three layers (28 units) for coupling measures, and the
  inner layer for the relation judge and succession. The label is the nearest label code
  to the output outer activity (by inner product).
- **Solve:** exact, by Cholesky factorization of the free block of the energy
  Hessian. Image-clamped and Morse-clamped examples have different free blocks
  (68 free activities each: input module hidden 8 + 4, core 16 + 8 + 4, output
  module 16 + 8 + 4). Each step therefore needs **two factorizations**, one per
  input kind, shared across that kind's batch.

## Data and labels

- **MNIST:** v1 split (54,000 training, 6,000 validation, 10,000 test).
- **Dice:** v1 standard faces 1–6 (the jitter caveat carries over; random layouts
  are a later study).
- **Morse (new generator), digits 0–9 in International Morse code:**
  - Window of 128 samples. Unit length `u ~ U(3, 5)` samples, drawn per example.
  - Dot = 1 unit, dash = 3 units, gap between elements = 1 unit. Each element and
    gap duration is `u · d · (1 + e)` with `e ~ U(−0.2, 0.2)` drawn independently,
    rounded to the nearest integer, minimum 1 sample.
  - Start offset: integer drawn uniformly from [0, 4], the same range for every
    class, so the offset carries no class information. The code's end position does
    vary with code length; that is part of the signal.
  - Length bound: the longest code (five dashes, four gaps, 19 units) lasts at most
    19 × 5 × 1.2 = 114 samples before rounding. Rounding adds at most 0.5 per
    duration, 4.5 over nine durations; the minimum of 1 sample never binds because
    every duration is at least 3 × 0.8 = 2.4. With the maximum offset of 4, the end
    is at most 122.5 < 128. **No signal is truncated or rejected.** A runtime
    assertion checks this for every generated example.
  - Amplitude `U(0.7, 1)` on the "on" samples, plus Gaussian noise of SD 0.1 on
    every sample.
  - Pools: 2,000 per number for training, 300 for validation, 500 for test, from
    named independent streams.
- **Scrambled Morse (control):** a fixed seeded permutation reassigns the ten codes
  to numbers. No number keeps its own code, and the number of dots is no longer
  monotone in number. Standard Morse makes 1–5 "n dots then dashes", so number is
  partly visible in the surface; scrambling removes that shortcut.
- **Label availability:** MNIST 0–9; dice 1, 3, 5; Morse 1, 3, 5. Unlabelled counts
  are still used in within-modality relation pairs.
- **No cross-modal pairs**, except in the equivalence-control arms.

## Training objective (backprop through exact equilibria)

Per step, the loss is a sum of per-task means, as in the pilot:

- **Label:** softmax cross-entropy over logits `β ⟨x_out, q_k⟩` at equilibrium, with
  a learned scalar β > 0.
  Batches of 64 per step for each of MNIST (all digits), dice (1, 3, 5) and Morse
  (1, 3, 5), balanced over the available classes.
- **Within-modality relations on core codes** (the pilot's structured judge):
  - *same*: logit `b_s − s_s ‖c(B) − c(A)‖²`, for MNIST, dice and Morse pairs;
  - *next* (succession arms only): logit `b_n − s_n ‖c(B) − S(c(A))‖²`, for
    MNIST, dice and Morse pairs.
  - 64 pairs per task per step, half positive, negatives uniform over non-matching
    class pairs. Relation pairs use class identity within a modality; they say
    "these two dice differ by one", not which number either one is.
- **Equivalence control** (E arms only): the pilot's cross-modal *same* on core
  codes, among fixed equivalents (10 per modality per labelled count), for
  image ↔ Morse pairs and MNIST ↔ dice pairs, in both orders. 64 pairs per order per
  step; negatives are equivalents of different labelled counts.
- Adam, learning rate 1e-3, 3,000 steps (pilot budget; the pipeline pilot checks
  convergence only and does not select a budget). Parameter clipping, if any, is
  fixed in the pilot and reported.

## Arms

| Arm | Succession in core | Morse | Cross-modal equivalence loss | Coupling |
| --- | --- | --- | --- | --- |
| S-M | on | standard | no | uniform |
| N-M | off | standard | no | uniform |
| S-X | on | scrambled | no | uniform |
| N-X | off | scrambled | no | uniform |
| S-X-E | on | scrambled | yes | uniform |
| N-X-E | off | scrambled | yes | uniform |
| S-X-I | on | scrambled | no | inner only |
| N-X-I | off | scrambled | no | inner only |

Five seeds per arm (40 fits). The E and I arms use scrambled Morse, the condition
without the surface shortcut. Labelled counts, equivalents and training images are
identical across arms within a seed.

## Measures (held-out test sets)

**Primary: label accuracy on unlabelled counts**, macro over counts and reported by
group:

- dice: 2, 4 (interpolation), 6 (extrapolation);
- Morse: 2, 4 (interpolation), 6–9 (extrapolation upward), 0 (extrapolation
  downward).

Prediction is the ten-way argmax over label codes, so the uniform-guess baseline is **1/10**,
including on unlabelled classes. A restricted prediction set is not used. The
no-succession arms give the separate, empirical learned baseline.

**Coupling between input modules in the core** (the "are the cores coupled"
question):

- paired linear CKA and raw coordinate correlation of core codes, image versus
  Morse, for same-number pairs, with shuffled-pair and within-modality references
  (as in the v1 alignment analysis);
- each Morse and dice core code read out by its nearest MNIST core centroid;
- matched-centroid distances relative to between-class distances;
- the same measures on the input modules' own inner codes, before translation, to
  show where the alignment arises.

**Cross-modal relations**, never trained across modalities in the non-E arms: *same*
and *next* from core codes, for Morse ↔ MNIST, Morse ↔ dice and dice ↔ MNIST,
scored as in the pilot (20 references per class).

**Health:** MNIST label accuracy; within-modality relation accuracy; core
composition `S^k` for k = 1–3 against MNIST core centroids; solve residuals;
signal gain through each translation.

## Predictions (to be fixed in the contract)

1. S-X transfers labels to unlabelled interpolation counts (dice 2, 4; Morse 2, 4)
   above N-X. The succession effect in the pilot came with equivalence losses, so
   for the label-only arms this is a hypothesis.
2. N-M does better on Morse than N-X, because standard Morse carries visible
   counting structure. S-X versus N-X is the clean succession effect for Morse.
3. **Hypothesis under test:** with succession and labels only (S-X), image and Morse
   core codes for the same number coincide (high cross-modal CKA and coordinate
   correlation). Alternative: labels align only the label readout, and codes for
   unlabelled counts sit apart. S-X-E is expected to show coincidence, as in the
   pilot. If S-X-E aligns and S-X does not, labels are insufficient.
4. Extrapolation degrades with distance: Morse 6–9 is worse than 2 and 4, with 9
   worst. Count 0 depends on whether S is learned to run below 1.
5. MNIST accuracy stays near the pilot's level (≥ 0.95).
6. Topology (S-X versus S-X-I): no directional prediction. Uniform coupling passes
   more signal from the surfaces into the core, which can help transfer; inner-only
   coupling should keep kind-specific appearance out of the core, which can help the
   codes coincide. Reported both ways: transfer accuracy and coupling measures.

## Prior work

Reviewed 2026-10-28 under [literature review](literature-review.md); sources,
evidence tags and deductions are in the
[number core network entry](prior-work.md#number-core-network) of the shared survey.

- **Established methods the components correspond to.**
  - The image ↔ core ↔ Morse layout is a shared-latent multimodal model (multimodal
    deep Boltzmann machines; MuPNet in predictive-coding form), here with a quadratic
    energy and learned translations at corresponding layers.
  - The core successor `S(c) = A c + t` generalizes TransE's relation translation.
  - Extending sparse correspondences through shared relations is established
    (seed-based entity alignment; structured relation decoders for MNIST successor
    relations, Strömfelt et al.).
  - Shared label space plus optional invariance loss is DSCMR's design.
  - Backprop through exact equilibria is implicit differentiation of a linear solve
    (OptNet, deep equilibrium models); with clamped inputs the equilibrium is the
    conditional mean of a linear-Gaussian model. It is not the program's local
    learning rule.
- **What theory or earlier results predict.**
  - Labels constrain core codes only through what the core → output path passes on.
    With a 28-unit core and 16-unit output, label supervision alone cannot force
    same-number core codes to coincide [derived]. DSCMR's ablation found label loss
    alone gives substantial but incomplete cross-modal alignment, even with dense
    labels [read]. The modality gap shows pair-matching objectives can still leave
    modalities offset [abstract]. So prediction 3 is genuinely open, and the S-X-E
    arm is the needed control.
  - Succession can carry label alignment to unlabelled counts only through the core
    inner layer, where S acts [derived].
  - Standard Morse lets an encoder interpolate counts 1–5 from its surface
    [derived]; the scrambled arms remove that.
- **Known failure modes and how the design handles them.**
  - Labels align only the readout: measured directly (CKA, coordinate correlation,
    centroid distances on core codes), with the equivalence control.
  - Prior-driven gain loss along chains (program result): weak prior λ = 0.001.
  - Dice face recognition standing in for counting (numerosity results show
    generic encoders pick up number, but fixed layouts allow shortcuts): accepted;
    v1 claims no counting. Random layouts and set encoders are a later study.
  - An affine successor may not fit a non-linear core code: reported through
    composition accuracy; a free successor is a possible follow-up.
- **Baselines and cheaper alternatives.** The cross-relation pilot already answers
  the shared-encoder version. A DSCMR-style late-fusion baseline (two encoders,
  shared classifier, optional invariance loss) is the standard comparator for the
  label-versus-invariance question. Its two ingredients are included as arms inside
  the SRI architecture; the DSCMR model itself is not, because the question here is
  about this architecture rather than about retrieval performance.
- **What the study adds.** No new general principle. It tests whether the program's
  coupled-module architecture supports sparse-label transfer through a shared
  successor, whether shared labels suffice as the cross-modal bridge, and whether
  coupling topology changes that.
- **Motivation, not prediction.** Children learn exact meanings for a few small
  number words before generalizing with a successor rule (Carey; Sarnecka and Carey
  2008; modelled by Piantadosi et al. 2012). The sparse-label design resembles that
  pattern; it does not test it.

## Later stages (not in v1)

1. Finite settling budgets instead of exact equilibria (repo evidence: 175–898
   steps to converge in the relational preflight).
2. Local clamped-energy learning for the SRI modules, with front ends frozen.
3. Random dice layouts and a blank (zero) face; beep sequences as a cardinal
   temporal modality.

## Computation

### Declared dimensions and counts

- **Fits:** 40 (8 arms × 5 seeds), plus a pipeline pilot.
- **Steps:** 3,000 per fit. Adam; float32 throughout.
- **Encodings per step**, from batch sizes above:
  - labels: 3 × 64 = 192;
  - relations: *same* for 3 modalities, plus *next* in succession arms, each
    64 pairs = 128 encodings. That is 384 (no succession) or 768 (succession);
  - equivalence (E arms): 2 orders × 2 modality pairings × 64 pairs × 2 = 512.
  - Maximum: 192 + 768 + 512 = 1,472 encodings per step (S-X-E); minimum 576 (N-M,
    N-X).
- **Solve:** state per single-input example = input hidden 12 + core 28 + output
  module 28 = 68 free activities (outer input clamped). Two Cholesky factorizations
  of a 68 × 68 block per step (one per input kind), and triangular solves for each
  example. Exact; no iterations, tolerance not applicable. A solve residual is
  recorded at evaluation.
- **Evaluation:** MNIST 10,000, dice 3,000, Morse 5,000 (and scrambled) test
  encodings; relation scoring as in the pilot (about 2 M judge evaluations);
  coupling measures on core codes (closed-form CKA, centroids).

### Time orders (estimates)

- **Training, per step:** front ends O(E · F) with E encodings and F ≈ 0.45 M (image)
  or 0.22 M (Morse) multiply-adds each, times about 3 for forward and backward. At
  E ≈ 1,000, about 1.2 × 10⁹ multiply-adds per step, or about 3.6 × 10¹² per fit.
  SRI part O(K · S³ + E · S²) with K = 2 and S = 68: about 6 × 10⁵ + 5 × 10⁶, which is
  negligible next to the front ends.
- **Inference, per example:** one front-end pass plus one 68-dimensional triangular
  solve pair (the factorization is reused across examples of the same input kind).
- **Construction:** dataset generation (dice, as in v1; Morse is O(samples) per
  signal, about 128 operations).
- **Selection:** none planned. The pipeline pilot is the only pre-confirmation run
  and does not select hyperparameters; its full cost is reported as selection cost.

### Storage

- **Parameters:** image front end ≈ 0.21 M, Morse front end ≈ 0.14 M, SRI weights
  ≈ 1,670: module maps W, V 640 (four modules); translations 1,008 (336 per
  edge); successor A, t 20; judge and label scalars 5. Label codes: 160 fixed values. About 1.4 MB at float32.
- **Optimizer state:** Adam keeps two moments per parameter, about 2.8 MB.
- **Persistent learned state beyond parameters:** none (no replay, no covariances).
- **Data held in memory:** MNIST training images ≈ 170 MB, dice ≈ 38 MB, Morse
  (both codes) ≈ 10 MB at float32, per worker (fork-shared until written).
- **Peak workspace:** activations for at most 1,472 encodings per step, a few MB.
- **Checkpoints:** final parameters per fit, about 1.4 MB, kept in gitignored
  `runs/` with SHA-256 values in the results, for the coupling analysis.

### Measured at execution (runtime JSON, separate from results)

Following [computation reporting](computation-reporting.md):

- source and contract hashes, revision and dirty status, command, seed policy;
- interpreter and library versions, OS, CPU, dtype, BLAS backend, thread and worker
  counts;
- wall and CPU seconds by phase: data generation, training (per fit), evaluation,
  coupling analysis, writing. Aggregate worker CPU across the pool;
- peak RSS per worker and for the parent, with units;
- counts: fits, optimizer steps, encodings, factorizations, solves, inference
  calls, convergence or numerical failures;
- costs split into pilot (selection) and confirmation, plus a short deployment
  timing (encoding plus solve, batch 1 and batch 256, warmed up, repeated,
  median and spread).

Expected wall time: about 60–120 minutes for 40 fits on 11 workers, from the cross-
relation pilot's step timing. Measured in the pipeline pilot.
