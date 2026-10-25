# Continuous local learning v1: checkpoint diagnosis

Status: post-run exploratory analysis, 2026-10-25. The main run is complete;
this note declares read-only diagnostics before calculating them. No training,
new trajectories, hyperparameter changes, or modifications to archived results.
Sources: [contract](continuous-local-learning-v1-contract.md) and
[scientific results](../reports/continuous-local-learning-v1-results.json).

## 1. A structural distinction the decoder must learn

Let u=f(0,0), v=(f(0,1)+f(1,0))/2, and w=f(1,1) be vectors of local detector
responses. Shared edge evaluation and averaging imply these exact pooled states:

| Outer family | Possible pooled edge states |
| --- | --- |
| Uniform | u, w |
| One different | (u+v)/2, (w+v)/2 |
| Adjacent pairs | (u+w+2v)/4 |
| Alternating | v |

All 16 outer patterns therefore occupy at most a two-dimensional affine subspace
of the four edge channels. This does not by itself prevent separation: the two
coordinates can retain all four relational shapes. The problem is how the decoder
uses them, and whether u,v,w are sufficiently distinct.

The adjacent-pair point is exactly the midpoint of the two one-different points.
A linear class-score decoder cannot label both endpoints one-different and their
midpoint adjacent-pairs with strict correct margins. Its difference between the
two class scores at the midpoint is the average difference at the endpoints.
Thus this task requires a nonlinearity that assigns the two ends to one class
and the middle to another. In the earlier count-table circuit this behavior was
supplied by the decoder family; continuous learning must acquire it here.

The class-mean pooled state of one-different is also exactly the adjacent-pair
state. Comparing only class means would miss the needed distinction. The supplied
sigmoid decoder can in principle bend the response, but its actual finite-weight
curvature and trained capacity were not proved by the previous run.

Useful diagnostic directions are polarity u-w and relational contrast
(u+w)/2-v. With approximately affine local detectors, the latter is small:
averaging local input coordinates loses most neighbor-relation information.
Small nonzero contrast is not equivalent to impossibility; it may instead create
a weak learning signal or poor conditioning.

For each decoder neuron with logit a at the adjacent-pair midpoint, the two
one-different logits are a +/- d, where d=A_edge*(u-w)/4. The nonlinear gap is

```
0.5*(sigma(a+d)+sigma(a-d)) - sigma(a).
```

Since max |sigma''|=1/(6 sqrt(3)), its absolute value is at most
d^2/(12 sqrt(3)). This is a curvature bound, not an empirical fit. It explains
why nearly linear operating regions can provide extremely weak separation of
these classes even though a nonlinear activation is present.

## 2. Read-only diagnostic menu

For all 6 saved trajectories, at initialization and post-A/post-B:

- Reconstruct parameters from the specified initialization or saved arrays;
  verify all main-run source/contract/result hashes first.
- Compute u,v,w, the polarity/relational-contrast norms, centered pooled singular
  values, and the midpoint identity. Use NumPy full SVD only on a 16x4 matrix;
  this is a representation diagnostic, not a learned projection.
- Compute the one-different/adjacent-pair mean gap in pooled features, decoder
  activities, and predicted outputs. Check the sigmoid-curvature bound with
  absolute tolerance 1e-12. Calculate the minimum of the correct pairwise class
  margins over the two one-different endpoints and the adjacent midpoint.
- Derive the final presented label of each block from the declared schedule and
  compare it with the saved predicted classes. This checks consistency with a
  recency interpretation; it does not prove recency is the sole cause.
- For the direct-prediction squared-error objective, average gradients over the
  eight original training patterns. Decompose the average into the gradient for
  the constant mean target mu=(1/4,...,1/4), plus its difference from the gradient
  for actual labels. The latter is the label-dependent contrast contribution;
  regularization cancels in the difference. Report layerwise norms. This is NOT
  reconstruction of the joint arm's actual online gradients: its training latent
  states were not saved. It is a shared predictive-function diagnostic only.

For the existing completion records, count exactly how many thresholded final
patterns equal their initialized default with observations inserted. Calculate
the default-only class-valid count for the declared 48-query battery and compare
with model scores. These are descriptive post-run controls, not newly trained
arms or basis for selecting a winner.

At the declared completion initialization, z=f(x), h=g(mean z), y is the supplied
class. Consequently ez=eh=0. Hidden x is at a binary stored default, so its prior
and binary-penalty derivatives also vanish. The initial hidden-input gradient
is zero, while output prediction error acts immediately on h and on B. Verify
this at all 12 post-A/post-B checkpoints and all 48 queries: 576 gradient checks,
no update steps. This identifies an available error-reduction route, not which
route dominates a complete unrecorded trajectory.

## 3. Interpretation limits and possible explanations

An energy decrease can be achieved by revising hidden explanations or the output
predictor, without learning a useful dependence on observed input. The initial
completion gradients make that distinction explicit. Establishing a causal
remedy would need a new prospective intervention, not retrospective retuning.

The direct-gradient comparator also failed under the declared stream and budget.
Therefore a negative result cannot be attributed uniquely to simultaneous
inference/learning or local residual updates. Architecture, weak feature geometry,
online presentation dwell, regularization, step sizes, and optimization all remain
possible contributors. No one of these has been isolated by the main experiment.

## 4. Complexity and cost

There are 18 checkpoint analyses, 288 supervised direct-gradient evaluations
(8 patterns x 2 target choices x 18), and 576 completion-initialization joint
gradient evaluations. No fits, parameter updates, model selection, or inference
trajectories. One additional direct prediction per completion initialization
constructs its latent state. Report actual counts from the imported evaluators.

The feature/gradient cost per checkpoint is O(T*(nk+km+mc)), with T=8 and the
dimensions from the main contract. Full SVD of an Mxk pooled matrix for M>=k costs
O(M*k^2+k^3) and O(Mk+k^2) storage; M=16,k=4 here. The 576 initializations have
the same O(nk+km+mc) factor-gradient cost each. Stored main artifacts are read as
inputs; runtime RSS includes their parsed JSON, not just deployed model state.

Run `experiments/continuous_local_learning_diagnosis.py` once and record separate
deterministic diagnostic JSON and runtime receipt. Hash this note, the diagnosis
script, imported main source and all consumed main artifacts. Measure wall/CPU
for the complete post-provenance diagnosis through scientific serialization;
report environment, counts, RSS and missing allocator workspace. No inference
latency, scalability, general learning, or intervention-effect claim follows.
