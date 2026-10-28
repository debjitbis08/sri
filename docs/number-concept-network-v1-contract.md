# Number concept network v1: contract

Status: **frozen 2026-10-28**. Substantive protocol changes require a versioned
successor contract; this file is not rewritten. Draft history:
[specification](number-concept-network-v1-spec.md) (SHA-256 at freeze, before its
status line was changed to point here:
`e3b18c9ac954e49d34a91b69847ae8da6f04017eaea490b8af60f85ffe926c73`). The protocol
below is that specification's agreed text with no pilot-driven amendment.
Source: Debjit Biswas, *Number concept network: design doc*, 2026-10-28,
five-page supplied PDF, not modified.

## Freeze record

**Implementation:** revision `a719ffd`. SHA-256:

- `src/number_concept.py`: `d8af3f35a76ece96a09f34da5d451cdbbb57a2ddc06592b41b96b89fd02c5479`
- `experiments/number_concept_network.py`: `29a180fb9cd7af22f9a77739924ca145e2d88de11d18364e023eddcbeb312ed7`
- `tests/test_number_concept.py`: `d494fd5ce35ab8d7a59b107bb6c33e38e7de6c8b22e9b9b413bf16199d274bde`

The confirmation run must use these files unchanged; its runtime JSON records
their hashes. Command: `.venv/bin/python experiments/number_concept_network.py
confirmation` (11 workers, one thread each).

**Environment:** Python 3.12.3, numpy 2.5.4, torch 2.14.1+cpu, Linux
7.0.11 x86_64 (glibc 2.39), 12 CPUs, CPU only, deterministic algorithms on,
single-threaded BLAS. MNIST: official IDX archives in `data/mnist/`, verified by the
SHA-256 values in `src/number_concept.py` (MD5s match torchvision's published list).

**Pilot:** [attempt 1](number-concept-network-v1-pilot-results.md), the only
attempt. Pipeline and stability checks passed; no amendment is made. Its diagnostic
accuracies did not inform any protocol choice.

**Implementation choices where the specification was silent** (fixed here):

1. Each seed draws its own count-1 anchor from stream `anchor/<seed>`, from that
   run's dice training pool; anchors therefore differ across seeds.
2. Digit centroids, probes and the stage-2 digit-1 centroid use that run's digit
   training pool (all 54,000 training-partition images in confirmation).
3. The dice pools and the MNIST split are fixed across seeds; seeds change
   initialization, sampling, anchors and SH shuffles.
4. U offsets are reported for counts 2–6 and 1–6; the "largest offset fraction"
   rule uses counts 2–6, matching the primary score.
5. Composition, recurrence and latent diagnostics are computed for every arm;
   the k=2 composition criterion uses arm A only.
6. Gradient clipping rescales the global real/imaginary-view gradient norm to 5
   (equivalent to `clip_grad_norm_`) and counts every clipped step.
7. Probes are fitted once per stage-1 checkpoint and evaluation split. Digit gates
   use the MNIST test set; validation metrics are diagnostics only.
8. Deployment timing uses seed 0's A arm for each variant. Grids use one fixed
   image per count from the first evaluation split (validation).
9. Checkpoints stay in gitignored `runs/`; results record their SHA-256 values.

## Question and scope

Can a latent succession operator learned from handwritten digits help a separate
dice encoder align with digit representations using one cross-modal anchor?
The practical target is **relation-guided, one-anchor transfer**, not a claim of
unsupervised number discovery, general counting, or an advantage of complex
networks over real networks. Number labels construct within-modality successor
pairs and positive masks. Digit labels also construct the evaluation centroids.
Only one dice-to-digit correspondence is available during anchored training.

Retain three modules, eight complex latent coordinates, separate image encoders
and decoders, two-stage training, and rotation versus MLP succession. Change the
original magnitude-only primary probe, false-negative contrastive denominator,
offset-only interpretation, undefined anchor loss, and chance baseline. Add
anchor-only and swapped-operator controls and seen-count random-layout evaluation.

## Prior work and what theory already predicts

Shared source notes and evidence tags are in
[prior work](prior-work.md#number-concept-network).

- Complex affine layers, modReLU and variance-scaled complex initialization are
  established components, not contributions of this experiment [read: Trabelsi
  et al., §§3.4, 3.6]. That paper also reports poor modReLU performance in some
  feed-forward settings; stable learning here is an empirical feasibility gate.
- The objective below is a successor-conditioned, multi-positive supervised
  contrastive objective [analogy to Khosla et al.]. Contrastive discrimination
  does not enforce exact successor equality [derived]. Alignment/uniformity
  theory does not establish alignment between separately trained modalities
  [abstract: Wang & Isola; derived limitation here].
- For rotation `R = diag(exp(i alpha))`, `|Rz| = |z|`. Exact successor codes on
  one orbit have identical coordinate magnitudes. Consequently a probe on
  `|Z|` alone may fail precisely when phase represents number [derived].
- Every diagonal unitary `Q = diag(exp(i beta))` commutes with `R`. Replacing
  a dice encoder by `Q E` and its decoder by `D Q^-1` preserves reconstruction
  and cosine successor scores, yet can change digit-centroid predictions.
  This gives a generic eight-angle symmetry, not merely an integer offset
  [derived]. Equal frequencies permit additional complex-unitary mixing.
  In the real/imaginary view, cosine is invariant under real orthogonal maps:
  opposite frequencies `alpha_j = -alpha_k (mod 2*pi)` permit a coordinate
  swap combined with conjugation, and `alpha_k = 0 or pi (mod 2*pi)` makes
  that coordinate's real 2×2 block `+I` or `-I`, permitting reflections.
  These commuting real maps preserve relation scores, but conjugate-linear
  maps cannot in general be absorbed into the complex-linear encoder/decoder
  boundary layers. Distinguish relation-score symmetry from exact symmetry
  of the full architecture/objective.
  More broadly, any orbit start `z_0` defines an exact recurrence
  `z_n=R^n z_0`, including starts with different coordinate magnitudes.
  Such orbits need not have equal contrastive or reconstruction loss, but
  recurrence alone gives no special preference to integer-offset solutions.
- Under **exact** class-level recurrence and exact anchor equality,
  `z_dice(1) = c_digit(1)` implies `z_dice(n) = R^(n-1)c_digit(1)`.
  Matching digit codes then follows only if they obey that same recurrence.
  Approximate contrastive training, within-class variation and one anchored
  image do not meet those assumptions automatically [derived]. For a rotation,
  recurrence errors accumulate at most additively in norm along a chain;
  classification also requires a sufficient centroid margin.

Cheaper references are the exact anchored orbit construction (an algebraic
sanity check, not an image learner) and reconstruction plus one anchor (included
as a trained control). A real-valued encoder, explicit pip counter, supervised
classifier and decoder-free model are possible later baselines. They are outside
v1 because v1 makes no compute-superiority, complex-superiority or general
counting claim. This study adds practical feasibility and controlled transfer
evidence on images; it does not establish a new learning principle.

## Architecture and numerical conventions

Use float32 pixels and complex64 activations/weights; real scalar losses.
An input image enters as `x + 0i`. Each modality has independent parameters:

1. Split 28×28 into 196 non-overlapping 2×2 blocks, row-major. Apply one shared
   complex affine map 4→4 to each block, then modReLU with four shared biases.
2. Flatten the 196×4 values; complex affine 784→128, then modReLU.
3. Complex affine 128→8 produces `Z`; no output activation or normalization.
4. Decoder: complex affine 8→128, modReLU, complex affine 128→784,
   modReLU, reshape to 196×4, shared complex affine 4→4 per block,
   reassemble pixels. Real parts are BCE logits; sigmoid is for display.

Encoder and decoder weights are untied. No skip connections, shared modality
weights, batch normalization, auxiliary label heads or latent norm penalty.
Affine biases are complex and initialized to zero. Each hidden modReLU has a
learned real bias initialized to -0.01 (shared over blocks in the first layer;
per coordinate elsewhere). In particular, the decoder's 128→784 output has
**784 independent modReLU biases**, not four shared over blocks. This is
intentional: that dense layer has position-specific outputs; only the final
4→4 block affine map shares its weights and four complex affine biases.
Use
`modReLU(z) = relu(abs(z)+b) * z / (abs(z)+1e-6)`.

Initialize each complex weight as independent Rayleigh magnitude with
`sigma = 1/sqrt(fan_in+fan_out)` and uniform phase on `[-pi, pi)`.
This is the complex Glorot magnitude/phase construction in Trabelsi §3.6,
not that paper's semi-unitary experimental initialization.

Succession variants:

- **Rotation:** `S(z)=exp(i alpha) * z`, eight learned real angles, initialized
  independently uniform on `[-pi, pi)`; no additive bias.
- **MLP:** complex affine 8→32, modReLU, complex affine 32→8; same
  initialization conventions, no output activation or residual path.

Flatten complex vectors to concatenated real and imaginary parts for cosine
similarity. Normalize by `max(norm, 1e-6)` only inside similarity/readout.
The anchor and reconstruction use raw latents. Record zero/near-zero norms,
non-finite values, gradient norms and learned angle distributions.

## Data, splits and information boundaries

**MNIST:** retain the official 10,000-image test set untouched until confirmation.
From the official 60,000 training examples, reserve 600 per digit for validation
using one fixed stratified split seed, leaving 54,000 training images. Store
source checksums and exact split indices. Scale pixels to [0,1]; no digit
augmentation in v1. Digit centroids and probe fits use training examples only.
In confirmation, report MNIST and standard-dice validation metrics at the final
checkpoint for diagnostics only; they cannot select checkpoints, change budgets,
stop runs early, exclude seeds or change the frozen decision rules. Confirmation
feasibility gates and headline scores use the held-out test sets.

**Standard dice:** 28×28 bright circular pips on a dark background; no box border,
text or class-dependent brightness/radius. At nominal coordinates 7, 13.5, 20:
one = center; two = top-left/bottom-right; three = two plus center;
four = corners; five = four plus center; six = left/right columns at all three
vertical coordinates. **Before any placement attempts**, assign every image ID
one radius uniformly in [1.4,2.2], one brightness uniformly in [0.7,1.0], and
one uniform quarter-turn, using count-independent distributions and RNG streams
separate from placement. Keep these assignments immutable. For each placement
attempt, propose all center jitters independently uniformly in [-2,2] pixels
per axis. If any center separation is less than `2*radius+1`, discard only
that set of center positions and redraw all jitters for the **same image ID
and radius**. Apply its fixed quarter-turn to the accepted layout.
Cap at 1,000,000 full-layout proposals per image (draw proposals in
vectorized batches; record the count used). On exhaustion, abort dataset
construction with the image ID and assigned metadata; do not drop the image,
redraw its radius, substitute a new ID, or retry with a different dataset seed.
Use 4×4 subpixel coverage per pixel, multiply foreground by the assigned
brightness, add independent Gaussian noise with SD 0.02, and clip to [0,1].

This samples positions conditional on the assigned radius and count, while
preserving the preassigned count-independent radius marginal in a completed
pool. Redrawing radius along with a rejected layout is prohibited: it would
weight the radius distribution by the count-dependent placement acceptance
probability, preferentially retaining smaller radii for crowded faces.
Accepted position jitters are conditional, not independent uniform draws;
radius/geometry dependence and total-pip-area cues remain limitations.

Generate fixed pools of 2,000 train, 300 validation and 500 test images **per
count** 1–6, with independent split RNG streams. No on-demand augmentation of
the anchor. Record generator version, RNG policy, sample identifiers, assigned
radius/brightness/rotation, accepted centers and placement-attempt counts.
Preassign metadata for the entire pool before placement; index placement and
noise streams by split and image ID so retries cannot change another image's
metadata. Save the assignment manifest even if construction fails.
Split test renders must not be reused as training or validation images.

**Random-layout diagnostic:** independent validation and test pools, respectively
300 and 500 images per count 1–9. Each full-layout proposal draws all centers
independently uniformly in [4,23]^2; accept only if every separation is at least
`2*radius+1`. Use the same immutable per-image radius, brightness and rotation
assignments and the same rendering/noise rules as standard dice. Rejection
redraws **positions only**, never the radius or image ID. The same 1,000,000
full-layout-proposal cap and whole-construction failure rule apply. Accepted
centers are conditional on separation, not independent uniform positions.
Random-layout data never enter fitting or hyperparameter selection.
Expose their validation results only after pilot decisions are recorded.
Counts 1–6 measure layout shift; 7–9 measure layout plus count shift.
The cap is sized for the hardest case: estimated per-proposal acceptance for
nine pips is about 1.0e-2 at radius 1.4 and 3.3e-5 at 2.2, so a 10,000 cap
would exhaust for about 12% of count-9 images, while 1,000,000 gives roughly
5e-15 per image at radius 2.2 [derived, Monte Carlo]. This keeps exact
rejection sampling; sequential placement would change the position distribution.

**Anchor:** choose one count-1 image from the dice training pool by a fixed
seeded index before training. Match it to the raw digit-1 training centroid,
computed after stage 1 and detached. It is one paired *dice example*, but its
digit target summarizes many labeled training images; do not describe this as
one labeled example in total. The same anchor image is repeated without changes.
Test count-1 images are distinct and reported separately from the primary 2–6
score. Within each seed all conditions receive identical data and anchor.

Stage 2 accesses only dice pixels, within-dice relation/class identities,
the frozen S and (in anchored arms) the saved digit-1 centroid. It cannot read
digit images, other digit centroids, probe weights or cross-modal test labels.
The evaluator receives all frozen centroids after training. Decoders participate
only in their own reconstruction; no cross-modal image loss is allowed.

## Losses and sampling

At each step draw an independent reconstruction batch and a successor-pair batch,
each of size `B=256`, sampling images with replacement from the fixed train pool.
Reconstruction class counts are as equal as possible across all available
classes; shuffle which classes receive the remainder. Successor-source classes
are similarly balanced over digits 0–8 or dice 1–5. Sample source and target
images independently from classes `n` and `n+1`. No 9→0 or 6→1 transition.
Keep reconstruction batches independent of relation batches so endpoints are
not systematically underrepresented in reconstruction.

`L_rec` is BCE-with-logits averaged over all reconstruction images and pixels.
For pair b, let `q_b=S(E(x_b))`, `k_c=E(x_c_plus)`, and let `t_b` be its
assigned target-class identity. With `P_b={c: label(x_c_plus)=t_b}`:

```text
l_bc   = cosine(q_b, k_c) / 0.1
L_succ = -(1/B) sum_b (1/|P_b|) sum_{c in P_b}
                         [l_bc - logsumexp_j(l_bj)]
```

Every positive set is nonempty. All same-target examples are positives; none
is an instance-discrimination negative. Target encodings receive gradients
during training; no momentum encoder or stop-gradient. Duplicate training
images are allowed by replacement and logged. Loss values include the
positive-multiplicity entropy floor and must not be interpreted as raw error.

For raw digit centroid `c_1` and the single fixed dice anchor `a`:

```text
L_anchor = ||E_dice(a)-c_1||_2^2 / max(||c_1||_2^2, 1e-6)
L_stage1 = L_rec + L_succ
L_stage2 = L_rec + lambda_succ*L_succ + lambda_anchor*L_anchor
```

No additional losses in v1. Reporting recurrence residuals does not introduce
an unregistered equality loss. An equality-loss variant would be a new arm.

## Training protocol and controls

Adam: learning rate 1e-3, betas (0.9,0.999), epsilon 1e-8, no weight decay or
scheduler; global gradient norm clipping at 5, with clip frequency reported.
For complex Adam specify its equivalence to optimizing the real/imaginary
views and pin the implementation version. Numerical failure stops that run;
record it, do not replace its seed. No test-based checkpoint selection.

Stage 1 jointly trains digit encoder/decoder and S. Freeze all of them for
stage 2, but retain gradients **through S's input** to the dice encoder.
For each S variant, train the following five dice arms from the same initial
dice weights and matched sampling streams, with fresh optimizer states:

| Arm | Succession pairs | lambda_succ | lambda_anchor | Purpose |
| --- | --- | ---: | ---: | --- |
| U | Correct | 1 | 0 | Unanchored transfer and ambiguity |
| A | Correct | 1 | 1 | Primary one-anchor transfer |
| SH | Shuffled | 1 | 1 | Does the correct relation matter? |
| AO | Unused | 0 | 1 | Can reconstruction plus anchor explain transfer? |
| SW | Correct, using donor S | 1 | 1 | Does S need to match the recipient digit representation? |

For SH independently permute the target images across the source rows each
step, preserving the target marginal distribution. Set each row's `t_b` to
its permuted assigned target label, then rebuild positive masks accordingly.
Do not retain a hidden correct-target mask. Random fixed points are allowed:
this is independence in expectation, not forced anti-succession. The condition
tests inconsistent relation supervision, not an alternative consistent ordering.
AO omits the succession forward/backward pass and its cost; otherwise match
reconstruction and anchor exposure. Record that control's cheaper compute.

For SW in confirmation seed s, substitute the frozen stage-1 S from seed
`(s+1) mod 5` of the **same variant**, while retaining seed s's dice initialization,
data streams, anchor image, digit-1 centroid and all evaluation digit centroids.
Keep the donor operator fixed throughout training and evaluation, including
composition and recurrence diagnostics. Do not align, conjugate or otherwise
fit the donor operator to the recipient's latent coordinates. Complete all
stage-1 fits before stage 2; record recipient/donor checkpoint hashes. If a
donor fit fails numerically, mark dependent SW runs unavailable with that reason;
do not substitute another donor or claim a passed control comparison.

SW supplies a fixed operator and correct dice relation pairs, in contrast to
SH's inconsistent pairs. It tests compatibility with this seed's digit geometry,
not whether every arbitrary operator is equally learnable. Report donor digit
feasibility and SW's within-dice recurrence and anchor residuals. The evaluator
also reports operator discrepancy on recipient MNIST test latents z as
`mean(||S_donor(z)-S_recipient(z)||^2) / max(mean(||S_recipient(z)||^2),1e-6)`;
this cannot select donors or tune training. Poor SW training can limit
the interpretation of a gap; a swapped operator may also be compatible by
coincidence. No post-hoc donor selection is allowed. The cyclic donor assignment
couples seed comparisons; five differences are descriptive paired observations,
not five independent operator-swap replicates for a significance test.

**Pilot:** seed 100, 500 training digits per class drawn from the train partition,
200 training dice per count, 5 epochs/stage, both S variants and all five arms.
Fit seed 101 stage 1 only, at the same pilot budget, to supply SW donors for
seed 100; this gives four stage-1 fits and ten dice fits in the pilot. Seed 101
uses the same fixed training subsets with separate initialization/sampling.
Each epoch has `ceil(train_pool_size/B)` optimizer steps, so stage 1 has 100
steps total and each stage-2 arm 25. Use the full standard validation pools,
not official test data. Check finite gradients, falling losses, clip frequency,
probe convergence and the implementation acceptance checklist. With only 100
stage-1 steps, this pilot cannot settle digit feasibility or the training budget;
missing the 95% gates is not a reason to expand it. Save every attempt and cost.

**Confirmation defaults:** seeds 0–4, full training pools, 20 epochs/stage,
211 steps/epoch in stage 1 and 47 in stage 2: 4,220 and 940 optimizer steps.
Use final checkpoints. There are 10 stage-1 fits and 50 dice fits: SW adds ten
dice fits (+25% relative to the former 40), with no extra confirmation stage-1
fits. Reuse the frozen checkpoint bank for all arms and donor assignments.
Initial digit **and dice** weights and data streams are paired across S variants
within each seed; S-specific initialization is separate. All five dice arms
within a variant also share their initial dice weights.
Use named independent RNG streams for splits, rendering, initialization,
sampling and shuffling so an arm's extra random draws cannot change its data.

No hyperparameter sweep is included in this specification. Pilot-driven
amendments are limited to documented pipeline or numerical-stability problems;
they cannot tune the training budget or objective weights for better accuracy.
Choosing a different budget requires a separately specified and recorded
budget-selection pilot, with fresh development seeds, validation-only decisions,
explicit candidate budgets and stopping/selection rules, and all selection cost
retained. That study is not included here. Freeze any resulting revision before
confirmation/test evaluation. Retain rejected pilots in the cost record.
Confirmation seeds cannot select the revision. Do not silently run until five
favorable seeds have been collected.

## Evaluation and decision rules

Form raw digit centroids `c_j=mean(E_digit(x) | label=j)` using training images.
Prediction is the largest cosine similarity to these ten centroids; ties choose
the smallest digit. All primary cross-modal decisions use candidates **0–9**,
including tests restricted to true dice counts 2–6. Uniform ten-way guessing
has 10% accuracy, not 20%. Actual controls need not be uniform; report their
measured predictions rather than requiring exactly chance performance.

1. **Digit feasibility.** Ten-way nearest-cosine raw-centroid accuracy and a
   frozen linear softmax probe on `[Re Z, Im Z]` are co-primary gates: **both
   must reach ≥95% on the MNIST test set in every seed** of the variant.
   Use overall accuracy for both gates and additionally report per-digit and
   macro accuracy. The centroid gate uses exactly the headline readout.
   `|Z|` is an auxiliary diagnostic.
   Fit both probes on digit training features using train-only feature
   standardization (SD floor 1e-6), zero-initialized weights/bias, full-batch
   Adam at 0.01 for 500 steps, no regularization, final weights. No gradients
   into encoders. Report probe convergence curves and cost. Failure of either
   gate limits interpretation of downstream failure; probe success cannot
   compensate for poor centroid accuracy.
2. **Primary transfer.** Macro accuracy over held-out standard faces 2–6 for A;
   report each count, the ten-way confusion matrix, and count 1 separately.
   A practical positive result for a variant requires mean accuracy ≥90%,
   no seed below 80%, and mean paired improvement ≥20 percentage points over
   each of SH, AO and SW. Report all five paired differences and sample SD, not
   only averages. These are engineering criteria, not a significance theorem.
   A numerically stopped A run fails the “no seed below 80%” rule explicitly;
   store accuracy as unavailable, not as a fabricated numerical score. Missing
   or failed control runs prevent a passed five-pair control comparison;
   they cannot be treated as zero-accuracy controls to inflate the gap.
3. **Unanchored structure.** For U report exact accuracy, histogram of
   `predicted_digit-count`, and largest offset fraction. Compute that fraction
   per seed before aggregation. An offset fraction ≥90% is descriptive evidence
   of offset-like behavior, not a required or theoretically guaranteed outcome.
   No per-class permutation or post-hoc alignment can count as primary success.
4. **Composition.** Apply frozen S k times to encoded test images for k=1,2,3;
   classify with the same digit centroids. Digits: source counts 0 through 9-k.
   Dice: source counts 1 through 6-k for the primary within-support test.
   Report macro accuracy, valid sample denominators, and errors by source count.
   Target ≥90% for k=2 in both modalities for A. Dice transitions into 7–9
   are separate extrapolation diagnostics, not pooled with this score.
5. **Layout/count transfer.** Report random-layout accuracies separately for
   1–6 and 7–9, per count and macro. A failure on 7–9 does not isolate counting
   when 1–6 also fail under layout shift. Success suggests generalization but
   does not establish general counting: pip area and other shortcuts remain.
6. **Mechanism diagnostics.** Report raw latent norms, within-class RMS spread,
   centroid cosine margins, and relative class-centroid recurrence residuals
   `||S(c_n)-c_(n+1)|| / max(||c_(n+1)||,1e-6)` within each modality.
   For the MLP also report mean per-example successor error, since S of a mean
   need not equal mean S. Record raw single-anchor residual and held-out count-1
   centroid residual; fitting one image is not evidence of class-wide alignment.
7. **Decoded images.** Save a fixed-seed grid of dice→digit and digit→dice
   reconstructions for supported counts 1–6, alongside within-modality
   reconstructions. These are qualitative diagnostics, not image-translation
   success metrics. The quantitative headline is centroid-based recognition.

The headline is **per variant**: “rotation passes” and “MLP passes” are separate
claims, each requiring both digit gates and all primary-transfer criteria.
One passing variant is enough for a positive one-anchor-transfer result for
that named variant; it does not establish the other variant or a rotation
advantage. Composition is a separately reported claim, requiring mean k=2
accuracy ≥90% in each modality across all five completed A runs; a primary
transfer pass does not imply a composition pass.

Report results for both S variants and all arms, including failures. A rotation
advantage is a paired measured gap, not an assumed result. If either digit
gate fails for any seed, that variant's strict five-seed transfer claim is not
established; report downstream results without dropping that seed. If digit
feasibility passes but A fails, distinguish within-dice recurrence failure from recurrence
without cross-modal alignment using the diagnostics. If SH or AO matches A,
the intended relation-transfer explanation is unsupported even at high accuracy.
If SW matches A, the necessity of the recipient's digit-learned S is not
established: an anchor plus another fixed operator may suffice on this fixture.

## Computation contract for the later implementation

Follow [computation reporting](computation-reporting.md). Let P=196 blocks,
C=4 block channels, H=128 hidden units, d=8 complex coordinates, K=10 classes,
B=256, and h=32 for S-MLP. One encoder or decoder costs
`O(B*(P*4*C + P*C*H + H*d))`; parameter storage is
`O(4*C + P*C*H + H*d)` plus biases, with constants for real/imaginary arithmetic.
Rotation costs `O(B*d)` time and `O(d)` parameters; MLP costs `O(B*d*h)`
time and `O(d*h)` parameters. Neither is an iterative inference solver.

Each training step uses three image encodings (reconstruction, source, target),
one reconstruction decoding, and, where applicable, one anchor encoding.
The contrastive similarity matrix costs `O(B^2*d)` time and `O(B^2)` storage;
multi-positive masks have the same storage order. For T steps multiply training
time by T, including backward/optimizer constants. Temporary activations cost
`O(B*(P*C+H+d)+B^2)` in addition to parameters/gradients/optimizer state.

Data construction is `O(N*28^2*16 + A)` for fixed 4×4 raster supersampling
and placement-attempt work A (including pairwise collision checks). Retained
raw images are `O(N*28^2)`; report actual uint8 versus float32 representation.
Centroid construction costs N encoder evaluations plus `O(N*d)` aggregation.
Readout costs one encoder plus `O(K*d)` per image; k-step composition adds
k operator evaluations. A linear probe with F=16 (or 8) features and 500
full-batch updates costs `O(500*N*F*K)`; feature caching costs `O(N*F)`.
There is no grouping search, sparse code, covariance estimator or solver
tolerance in this design; mark those fields not applicable rather than omit.

For complex64 parameters use 8 bytes/scalar and for real float32 4 bytes/scalar.
Report parameter, gradient, Adam moment, feature-cache, image-pool, checkpoint
and peak-workspace bytes separately; persistent inference state excludes Adam
but includes the two modality modules, S and digit centroids. Report both the
full model and the smaller recognition-only deployment without decoders.
Record actual allocations, serialized bytes and process/device peak memory;
parameter count is not a FLOP estimate. Training retains raw data: this is not
a no-replay or streaming experiment.

Write deterministic scientific results separately from a runtime JSON with:

- Source/spec/data hashes, git revision/dirty status, exact command, seed
  streams, package versions, OS, CPU/device, BLAS, threads/workers, dtype,
  determinism settings and relevant memory limits. State scope of any remaining
  device nondeterminism; timing never belongs in scientific result comparisons.
- Wall and process CPU seconds for data download/preparation, each stage-1 fit,
  each dice fit, centroid/features, each probe, validation, test evaluation,
  compilation and artifact writing. For parallel work aggregate worker CPU
  time or mark unavailable with a reason. Synchronize accelerator phase timings.
- Independent examples actually sampled, reconstruction/source/target/anchor
  exposures separately, optimizer steps, encoded images, S applications,
  probe fits, clips, failures, all pilot/tuning candidates and discarded runs.
  Sampling with replacement means pool size is not the number actually seen.
- Peak RSS with platform units and parent/whole-job scope, accelerator peak
  allocations if applicable, retained training and inference bytes. Missing
  measurements are null with reasons, never fabricated zeros.

For deployment timing warm up 10 batches, then measure 100 repetitions at
batch sizes 1 and 256, reporting median and interquartile range. Include
allocation and synchronization; separately name any excluded host/device
transfer or compilation. Measure encoding-plus-readout and cross-decoding
separately. Report quality versus total training/selection time and retained
bytes, with shared stage-1 cost shown both once per seed/variant and amortized
over its arms. Use `total_cost(Q)=preparation+selection+training+Q*inference`.
Do not infer equal-compute superiority from equal-epoch comparisons. Historical
SRI experiments lack comparable image-study costs and are context only.

## Implementation acceptance checklist and future artifacts

Before generating study data, verify that forced placement rejections preserve
the image's assigned radius, brightness and rotation; that extra retries do
not change other image IDs' metadata; and that forced exhaustion aborts pool
construction without replacements or dropped IDs. For every completed pool,
require exact equality of the preassigned and rendered metadata by image ID
and the requested number of images per count. Record per-count radius means,
SDs, quantiles and fixed-bin histograms, plus acceptance rates by count and
radius bin, separately for each split and layout family. Audit these against
the preassigned manifest: unequal finite-sample means alone are not proof of
bias, and a distribution test alone cannot replace the metadata invariants.
Any mismatch blocks training. No search over dataset seeds to improve these
diagnostics is allowed. Include rejected placement work in data-generation cost.

Before a pilot, verify shape/real-imag conventions, finite complex gradients,
frozen stage-1 parameters with live gradients through S, split isolation,
single-anchor access, positive masks (including SH), SW donor identity and
recipient-centroid retention, identical initial dice weights across variants,
candidate sets and
composition boundaries. Verify an exact synthetic rotation orbit has constant
magnitudes and that a generic diagonal phase change preserves relation scores
while possibly changing unanchored predictions; exact anchor propagation should
work when both orbits satisfy the exact-recurrence assumptions. These are
implementation checks, not evidence that learned image codes obey the model.

The implementation produces per-run scientific JSON, separate runtime JSON,
split/sample manifests, checkpoints, fixed qualitative grids and a results
document in docs/. Pilot, amended-spec and confirmation artifacts have distinct
names and hashes.
