# Masked capacity v1: the split improves; the declared module has a structural obstruction

**Completion:** complete, 2026-10-25. All 18 correctness tests passed. Both
task audits were byte-identical. All eight capacity candidates completed
1,000 updates, and every selected serialized checkpoint passed two identical
independent NumPy/Schur verifications. No witness was found. A separate
analytical argument derived during this study establishes a stronger conclusion:
the declared masked module cannot meet the 0.10 worst-scene target.

**Decision:** keep the explicit observation-mask semantics, but do not launch
a full learning study with this exact energy and parameterization for this
task. The masked module has a worst-scene error lower bound of **0.1369346**
within every relation–amplitude group, even with input-dependent routing.
This is a limitation of the specified radial capacity and coupling strengths,
not a general impossibility result for circular or reciprocal architectures.

[Contract](single-module-masked-capacity-v1-contract.md) ·
[manifest](../reports/single-module-masked-capacity-v1-manifest.json) ·
[task audit](../reports/single-module-masked-capacity-v1-audit.json) ·
[search results](../reports/single-module-masked-capacity-v1-results.json) ·
[runtime](../reports/single-module-masked-capacity-v1-runtime.json) ·
[proof](single-module-masked-capacity-obstruction.md) ·
[proof receipt](../reports/single-module-masked-capacity-v1-obstruction.json) ·
[proof runtime](../reports/single-module-masked-capacity-v1-obstruction-runtime.json)

## 1. What was tested

The architecture retains four rings, sector-supported orthonormal radial maps,
unit terminal vectors, reciprocal energy inference, the original gate formula
and the actual outer state as output. The only inference change is the explicit
observation mask in both passes: observed sectors have data anchors; missing
sectors do not. There is no added decoder, output gain or learned cross-sector
connection.

The new task has 72 clean scenes: three angular relations, three visible-part
amplitude ratios {1/3,1,3}, and eight common orientations. Hidden parts continue
the angular relation and reuse the two visible amplitudes. Both visible and
hidden signals have squared norm two. The holdout comprises three entire
relation–ratio combinations (24 scenes), so their sign reversals and common
rotations are also withheld. Each individual relation and ratio occurs in
the other 48 scenes.

Two renderings were fixed prospectively: identity and independent orthogonal
changes of coordinates inside each part. They are capacity fixtures, not
independent evidence of broad generalization. The study has no texture,
observation noise, trained representation probe or abstraction claim.

Capacity fitting deliberately uses **all 72 clean targets**, including the
nominal holdout. It asks whether suitable parameters exist. Its fitted weights
must not initialize a future learner. The full gradient through gates and
both solves is used only for this privileged search; it does not replace the
plasticity study's frozen-route learning rule.

## 2. The revised split excludes the declared simple shortcuts

The narrow split gate passes in both renderings. No held-out input is a
training input up to sign or common rotation/scale. The smallest held-out
visible squared discrepancy is 0.117157 for signed lookup and 0.105442 for
common rotation/scale lookup, both well above the declared 1e-10 exact-match
threshold. Both controls also fail the completion target.

Held-out mean normalized error and fraction with both missing orientations
correct are identical across renderings to reported precision:

| Reference | Held-out mean loss | Worst held-out loss | Joint orientation accuracy |
| --- | --- | --- | --- |
| Zero / training mean | 1.0000 | 1.0000 | 0/24 |
| First-part affine, trained on 48 scenes | 1.1469 | 1.5971 | 0/24 |
| Two-part affine, trained on 48 scenes | 1.2909 | 1.9233 | 0/24 |
| Ordinary or signed training-template lookup | 1.4037 | 2.1414 | 8/24 |
| Privileged common rotation/scale lookup | 1.9160 | 1.9473 | 0/24 |
| Privileged factored-template reference | numerical zero | <9e-31 | 24/24 |
| Modular-rule oracle | numerical zero | <7e-31 | 24/24 |

The factored reference knows the signal-plane rendering and the true amplitude
correspondence between visible and hidden parts. It normalizes parts separately,
matches an angular template, then restores the appropriate amplitudes. Its
success is expected and establishes that a simple supplied factorization solves
the task. It is not a fair learned comparator and does not demonstrate that a
learner discovers that factorization.

The revised split therefore excludes the named unfactored lookup mechanisms;
it does not exclude all template representations. A later abstraction claim
would require acquiring a reusable factorization from permitted experience,
with appropriate controls, rather than receiving it from the evaluator.

The all-scene affine capacity oracle achieves mean error 0.702027 over all
72 scenes. On the equal-amplitude subset alone, the exact optimal affine
average remains (8−2√2)/21 = 0.246265. Consequently any fixed-route predictor
is excluded from the <=0.10-on-every-scene target, independently of optimization.

## 3. The bounded nonlinear capacity search

The search used two Gaussian constrained starts and Adam step sizes 0.01 and
0.05 per rendering, at most 1,000 full-batch updates per candidate. Both
solves and gate computation were differentiated, gradients projected onto the
constraint tangents, and parameters retracted by QR/unit normalization. Every
candidate exhausted its declared budget; none passed early stopping.

Final checkpoint errors use all 72 scenes, including fitting targets:

| Rendering | Start | Step size | Final mean loss | Final worst-scene loss | Candidate wall seconds |
| --- | --- | --- | --- | --- | --- |
| Identity | 0 | 0.01 | 0.886809 | 1.226186 | 39.82 |
| Identity | 0 | 0.05 | 0.886852 | 1.226446 | 38.44 |
| Identity | 1 | 0.01 | 0.886819 | 1.226361 | 38.16 |
| Identity | 1 | 0.05 | 0.886827 | 1.227293 | 38.04 |
| Rotated | 0 | 0.01 | 0.883807 | 1.207376 | 38.15 |
| Rotated | 0 | 0.05 | 0.883753 | 1.208812 | 38.64 |
| Rotated | 1 | 0.01 | 0.883798 | 1.209839 | 38.24 |
| Rotated | 1 | 0.05 | 0.888458 | 1.182954 | 38.55 |

Optimizing mean error worsened the worst scene. Under the prospective
worst-scene checkpoint rule, every candidate selected step zero. The best
verified worst errors were 1.006322 for identity and 1.007027 for rotated,
both from start 1. Selecting these checkpoints is not evidence of successful
learning; it records the lack of improvement under the capacity criterion.

For example, the first candidate in the declared order follows this quality–cost
curve. All candidates' curves are retained in scientific and runtime JSON.

| Updates | Scene exposures | Mean loss | Worst loss | Cumulative candidate wall seconds |
| --- | --- | --- | --- | --- |
| 0 | 0 | 1.003517 | 1.009592 | 0.015 |
| 100 | 7,200 | 0.911086 | 1.079128 | 4.28 |
| 250 | 18,000 | 0.887246 | 1.225726 | 10.75 |
| 500 | 36,000 | 0.886860 | 1.225718 | 20.22 |
| 1,000 | 72,000 | 0.886809 | 1.226186 | 39.48 |

The 72 finite scenes are reused throughout; exposures are not independent
examples. This fitting trajectory is not a sample-efficiency or transfer
learning curve. Its stagnation does not, on its own, prove an optimum.

## 4. A separate proof resolves this configuration's capacity question

The [full derivation](single-module-masked-capacity-obstruction.md) was developed
after seeing search progress and then checked independently against saved
states. It is not a prospective statistical finding and does not infer
impossibility from failed optimization.

Each sector has a single unpenalized radial direction, whose outer expression
is `w_j=D_0,j D_1,j D_2,j v_j`. The gates change positive penalty strengths
but cannot rotate that direction with parameters fixed. Decomposing the radial
energy into orthogonal scalar chains bounds its response outside that direction
by `sqrt(30)/a_min` times the applied force, where a_min=0.325.

For an entirely missing sector, that force comes through the lateral coupling.
The minimized energy is no larger than the zero-state energy, and the ring
Laplacian has spectral norm at most four. These facts bound the combined hidden
output perpendicular to the two passthrough lines by

```
C = sqrt(240 * alpha * lambda)/a_min
  = 8 sqrt(15)/65
  = 0.476674873379.
```

Within every relation–ratio group, averaging the eight orientations leaves
at least unit squared target energy perpendicular to those lines. At least
one scene therefore requires perpendicular target norm >=1. Its normalized
error is at least

```
(1-C)^2/2 = 0.136934594076 > 0.10.
```

The result permits arbitrary input dependence of routes within their existing
positive bounds. It applies to the prescribed two-pass matched routing and
both renderings. It excludes the exact worst-scene target for every allowed
parameter setting, rather than only the starts and optimizer used here.

The proof receipt checks the chain inverse, source Laplacian, gate floor,
energy bound, radial nullspace dimension, radial response operator, hidden
stationarity and target projection inequalities. It covers all 12 distinct
saved initial/selected and final parameter states, all 72 scenes per state,
and both hidden sectors. No checks failed. The largest observed perpendicular
output norm was 0.007298, below the conservative universal bound 0.476675.
The largest checked radial-response operator norm was 11.2896, below
sqrt(30)/0.325. These finite checks supplement the proof; they do not establish
its universal quantifier by sampling.

The main search JSON retains `unresolved_after_budget`, correctly describing
what the optimizer established. The separate proof receipt records
`proved_obstruction_for_declared_masked_module`. The combined conclusion is a
structural obstruction at the declared target. The lower bound is conservative
and is not an estimate of an attainable optimum; raising the acceptance
threshold slightly above it would not establish feasibility.

## 5. Implications for our architecture

The observation mask is still the appropriate missing-data interface. It
removes the false instruction to reconstruct missing measurements as zero.
The new obstruction concerns the internal representation and communication
available after that correction.

The original signal-extraction generator supplied one signal direction per
sector, matching the one-dimensional terminal nullspace. The orientation task
uses a two-dimensional signal plane per part. With this energy, positive gate
floor and weak lateral coupling, the hidden response cannot move far enough
away from its single passthrough line to cover that plane accurately.

This does not imply that a scalar latent can never encode an angle through
a different nonlinear decoder. The proof uses the actual radial maps,
terminal penalty, coupling strengths and direct outer-state output here.

A natural next design question is the retained terminal subspace dimension.
Allowing a two-dimensional retained subspace would remove an assumption of
this proof while preserving the circular reciprocal organization. It would
still need its own capacity witness and noise/retention checks. Changing the
relative coupling strengths or gate floor would likewise invalidate parts
of this bound, but would not guarantee success and could change denoising
behavior. None of these revisions was made or tested in this study.

Further optimization of the unchanged configuration cannot produce the
declared witness. The next work should be a deliberate, narrowly specified
representational revision and another feasibility check, before a learning
and abstraction experiment.

## 6. Correctness, reproducibility and computation

Contract commit `a620c59` preceded implementation. Implementation and manifest
commit `dd95316` preceded tests, audits and fits. The proof text and receipt
source were committed as `41c26ae` after the search and before their numerical
audit. All prior frozen dependencies remain hash-identical. The separate
mechanism-diagnostics contract was present but untracked and was not executed.

All 18 tests passed, including full derivatives through the gates against
independent finite differences, off-manifold curvature, masking, tangent
projection/retraction, target-free template selection and direct/Schur
equivalence. The selected serialized candidates' maximum Torch/NumPy output
difference was 1.16e-14, and maximum full/Schur state disagreement 1.66e-14.
No nonfinite value, rank failure, constraint violation, solver repair or
replacement candidate occurred.

Both complete task audits were byte-identical. Each selected serialized
checkpoint was independently evaluated twice with byte-identical verification
records. The complete optimizer trajectory was run once, as declared; its
re-execution reproducibility was not tested.

Measured on Ryzen 5 3600, float64, NumPy 2.5.4, SciPy 1.18.1 and CPU PyTorch
2.14.1, with one process, one BLAS thread and one Torch intra/inter-op thread:

| Phase | Wall seconds | CPU seconds | Peak RSS KiB, process scope |
| --- | --- | --- | --- |
| CPU dependency installation | 44.24 | 3.63 | 59,620 installer |
| Manifest freeze command | 5.93 | 5.89 | 395,972 parent |
| Tests subprocess, including imports | 2.39 | 2.39 | 368,484 child |
| Task audit command, two executions | 2.62 | 2.57 | 378,852 parent |
| Capacity search command, all eight candidates | 311.18 | 311.05 | 441,936 parent |
| Supplementary proof audit | 3.00 | 2.99 | 109,164 parent |

The tests command also used 2.32 s parent CPU and 4.76 s total wall time;
do not add its child wall time to that total. Candidate and checkpoint times
are contained in the search total. Process RSS is not summed whole-job memory.

The first freeze attempt stopped at import because Torch was absent; no
numerical work ran. An installation attempt was blocked by sandbox DNS, then
the repository's pinned CPU dependency was installed with approval. The
[setup runtime](../reports/single-module-masked-capacity-v1-setup-runtime.json)
records all attempts, the 7.6 s reported retry duration, and nulls/reasons for
unmeasured complete startup-attempt costs. Requirements and methods were not
changed. Initial imports account for much of the short commands' elapsed time.

Each task-audit execution used eight affine fits across two renderings, two
constant means and 1,584 comparator-scene scores, including the privileged
references. Both audit executions are charged. Search used exactly 8,000
updates, 576,000 fitting-scene exposures, 8,000 backward passes, 96,000 QR
retractions and 32,000 unit-vector retractions. Its 8,056 Torch forward batches
include optimization, 40 scheduled checkpoints and 16 verification forwards;
these used 8,056 pilot and 580,032 second-pass factorizations. Independent
selected-checkpoint verification added 2,304 NumPy/Schur equilibrium checks.
The separate proof audit added 1,728 equilibrium checks, 1,728 radial
eigendecompositions and 1,728 radial-response SVDs. Tests have additional
small fixtures, timed separately rather than folded into search counts.

Complexity orders are in the contract: dense O(N M^3) forward/backward work,
O(N M^2) autograd state, and O(P) Adam state. Parameters occupy 2,624 bytes;
actual Adam state occupies 5,264 bytes (5,248 moment bytes and 16 counter
bytes). The retained tensor inputs/targets occupy 27,648 bytes per rendering,
a dense curvature tensor 819,200 bytes, and a 72x80x80 tensor 3,686,400 bytes.
These are named allocations, not a bound on simultaneous native workspace.
Native autograd workspace, FLOPs and backward primitive-solve counts are
unavailable with reasons in runtime JSON. No scaling or inference-latency
claim is made.

For reproduction, use an isolated checkout at `dd95316` with the pinned
dependencies and that revision's initial manifest/runtime, then execute
`python -m experiments.masked_capacity --phase tests`, `--phase audit`, and
`--phase search`. The supplementary proof source is in `41c26ae` and runs as
`python -m experiments.masked_capacity_obstruction` after the search. Runners
refuse changed frozen inputs and already recorded phases; preserve the
archived artifacts rather than overwrite them.
