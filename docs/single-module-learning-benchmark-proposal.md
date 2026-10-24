# Single-module learning: comparison with established architectures

**Status:** proposal, 2026-10-24. Not a frozen experimental contract. No
learning experiment or benchmark has been run. This records the requirement
to test faster learning and resilience against established alternatives,
not merely demonstrate a successful module in isolation.

## 1. Primary research claim

Can a learned, coarse-interior module acquire useful broad structure from
fewer independent observations, and preserve that usefulness on new related
inputs, better than appropriate established architectures?

Sample efficiency and resilience are primary. Execution time is secondary.
Multi-module interactions are outside scope. An architectural advantage is
a hypothesis to test, not an outcome to guarantee by selecting favourable
baselines or tasks.

The earlier studies supplied the relevant bands or averaging maps. The next
study must learn its broad representation without receiving the generating
subspace, latent variables, or test-set relationships during training.

## 2. Analytical preflight before a learning sweep

Begin with paired observations

```
h_1 = U c + d_1
h_2 = U c + d_2.
```

`U` is unknown. Independent, conditionally zero-mean detail terms give
`E[h_1 h_2^T]=U Cov(c) U^T`. This permits an exact population reference
and finite-sample analysis of recovery from related examples. Pair membership
is an explicit form of supervision shared by all learned methods.

First determine which hypothesis classes are actually distinct. A linear
32 -> 8 -> 4 factorization with disjoint local restrictions can represent
the same coarse maps as a suitably constrained compact 32 -> 4 encoder.
Likewise, some equilibrium and skip-connected feed-forward surface maps are
exactly equivalent. Such comparisons are controls for learning dynamics or
regularization, not evidence of different representational power.

Derive or calculate the attainable population solution, sensitivity to
nuisance, and dependence on sample size before selecting numerical budgets.
The spectral reference is especially important: on a linear synthetic task,
a neural implementation should not be credited for solving a problem that
a small covariance eigensolve handles more reliably.

## 3. Baseline suite

Include a small, strong suite rather than a large set of weak defaults:

| Comparator | Purpose |
| --- | --- |
| Matched feed-forward 32 -> 8 -> 4 encoder with a skip-connected surface decoder | Isolate reciprocal settling from hierarchy, locality and learning objective |
| Matched compact 32 -> 4 model | Isolate the benefit of an intermediate layer |
| Small convolutional/residual encoder | Compare with an established spatial inductive bias |
| Denoising autoencoder | Compare learned robustness through reconstruction of related/corrupted observations |
| Paired-view encoder trained with VICReg | Compare with an established invariance-learning method that prevents collapsed representations |
| Fixed averaging and spectral cross-covariance estimator | Sanity and analytical controls; not substitutes for neural baselines |

The denoising baseline must use only observations available to the other
methods. If trained from one noisy related view to another, label that as
the paired-view adaptation; do not silently give it clean latent targets.
Similarly, do not withhold pair information or permissible augmentations
from it merely because its usual implementation differs.

VICReg and Barlow Twins are learning objectives, not standalone network
architectures. Use explicit backbones when naming their results. VICReg is
the initial proposed paired-view method; Barlow Twins is an optional
replacement/additional check to decide during development, before freezing
confirmation. Do not substitute methods after inspecting test outcomes.

Relevant primary references:

- [Vincent et al., Stacked Denoising Autoencoders (2010)](https://www.jmlr.org/papers/v11/vincent10a.html).
- [Bardes et al., VICReg (ICLR 2022)](https://arxiv.org/abs/2105.04906).
- [Zbontar et al., Barlow Twins (ICML 2021)](https://proceedings.mlr.press/v139/zbontar21a.html).

## 4. Separate architecture from training method

Use two complementary comparisons:

1. **Architecture-controlled:** apply the same compatible invariance
   objective and the same pair stream to the coarse module and matched
   feed-forward/compact architectures. Match locality and latent dimension.
   This asks whether the organization itself helps.
2. **Method comparison:** allow established baselines their appropriate
   objectives and tuning within equal development resources. This asks
   whether the resulting module is competitive in practice.

A win only in the second comparison does not identify architecture as the
cause. A win only against a generic dense network can reflect the supplied
locality prior; the spatially matched comparator is mandatory.

At least one capacity-controlled comparison should match trainable parameter
counts within a declared tolerance, while a small prespecified capacity
range checks that an exact parameter match did not cripple a baseline.
Keep the evaluated core dimension fixed for the primary comparison, and
account for decoder and projection-head parameters. Do not mechanically
force every method to use an unsuitable hidden width or optimizer.

## 5. Tasks and held-out generalization

Start with an identifiable synthetic task, then require a less tailored
confirmation before making a broad architecture claim:

- **Aligned local structure:** unknown shared components mostly respect the
  coarse module's locality assumptions.
- **Misaligned structure:** shift group boundaries or use shared components
  spanning groups. This measures the cost of the architectural assumption.
- **New detail:** unseen fine variations around known broad situations.
- **New combinations:** held-out broad coefficient combinations, with no
  duplicates of their training views.
- **Changed nuisance:** altered magnitude or distribution of detail/noise.
- **Genuine broad change:** the representation must change appropriately;
  an unresponsive core cannot receive credit for resilience.

Include a development check with a persistent nuisance component: stability
across views is not automatically relevance. If both signal and distractor
persist and the observations provide no way to distinguish them, report the
identifiability limit instead of claiming that architecture alone can infer
the intended semantics.

Choose the secondary, less tailored task and its transformations before
confirmation. Passing only a block-aligned synthetic generator supports a
narrow conclusion about that inductive bias, not superiority over established
architectures generally.

## 6. Outcomes that would support improvement

Measure learning curves at independently counted training-pair budgets,
initially proposed as `{4,8,16,32,64,128}` subject to pre-confirmation
feasibility checks. Repeated augmentations and epochs do not count as new
independent examples. Report unique underlying situations, pair count,
augmentation exposures, and optimizer updates separately.

Primary outcomes should be:

- **Low-data generalization:** area under a held-out representation-quality
  curve versus log training-pair count, using a metric shared by all models.
- **Resilience:** performance on prespecified new related variations at the
  same training budget, with genuine broad distinctions still recoverable.

Potential common metrics include held-out paired retrieval among different
underlying situations, or a fixed-budget downstream probe with its labelled
calibration data counted explicitly. Constant or collapsed representations
must fail. Synthetic subspace recovery is an additional diagnostic where
defined; it cannot be the sole primary metric for nonlinear baselines.

Examples-to-threshold is a useful accompanying measure once an attainable
threshold is fixed on development tasks. A target such as using half as many
examples would be a meaningful effect, but is not a result promised in advance.
Methods that never reach the threshold are reported as such rather than
assigned an arbitrary favourable sample count.

Freeze the exact primary metric, practical effect margin, shift conditions,
and statistical comparison before confirmation. Compare against all declared
primary baselines with multiplicity handled, or select a single champion
using independent validation only. Do not choose the weakest comparator or
best sample budget after viewing test results.

## 7. Fairness and uncertainty

Every learned method receives the same independent examples, pair identities,
augmentations, and permitted supervision. No method receives clean targets,
generating modes, or test identities unless that information is shared and
explicitly part of the experiment. Fixed averaging is a disclosed prior,
not a learned competitor with hidden privileged information.

Use equal hyperparameter-search resources on disjoint development data.
Keep all learning-rate schedules, early stopping, model selection and
normalization statistics separate from test data. Report convergence failures
and confirm that apparent low-data superiority is not undertraining a baseline.

Use multiple independently generated tasks/train sets as well as
initialization seeds. Statistical replication should reflect independent
tasks or training samples, not count many correlated corruptions of one
pattern as independent evidence. Pair comparisons across shared task instances
and report uncertainty on the differences.

## 8. Decisions after the benchmark

- **Clear gain over the strongest relevant baselines:** identify whether it
  comes from locality, hierarchy, learning objective, or reciprocal dynamics
  using the controlled comparisons; then optimize the single-module design.
- **Tie with a simpler matched model:** report equivalence or comparable
  learning, without claiming an architectural advantage.
- **Gain only on aligned synthetic tasks:** report the domain restriction
  and evaluate whether it matches the intended application.
- **Loss:** use the analytical and optimization controls to identify the
  limitation; do not weaken the benchmark or relabel efficiency as learning.

The next implementation should follow a frozen contract that resolves the
candidate learning rule, data generator, evaluation metric, development
budget, and statistical criterion. This proposal intentionally does not
claim those design choices have already been validated or that superiority
has been demonstrated.
