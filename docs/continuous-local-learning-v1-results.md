# Continuous local learning v1: descent without useful grouping

Completed 2026-10-25. The declared six trajectories and all diagnostics ran once.
No hyperparameters, seeds, or budgets were extended after seeing the results.

[Contract](continuous-local-learning-v1-contract.md) ·
[runner](../experiments/continuous_local_learning.py) ·
[scientific results](../reports/continuous-local-learning-v1-results.json) ·
[runtime receipt](../reports/continuous-local-learning-v1-runtime.json) ·
[post-run diagnosis specification](continuous-local-learning-v1-diagnosis.md) ·
[diagnostic results](../reports/continuous-local-learning-v1-diagnosis.json) ·
[diagnostic runtime](../reports/continuous-local-learning-v1-diagnosis-runtime.json)

## 1. Outcome

The simultaneous local-factor update rule was implemented and passed gradient,
clamp, and fixed-evidence energy checks. It did not learn useful grouping within
the declared budget. Direct gradient training of the same predictive functions
also failed. Therefore this result does not isolate simultaneous learning as
the cause, and does not establish that continuous local learning is impossible.

| Arm | Replica | All-pattern classification after A | After B | Original training after A |
| --- | --- | --- | --- | --- |
| Joint continuous | 0 | 5/16 | 4/16 | 3/8 |
| Joint continuous | 1 | 8/16 | 4/16 | 2/8 |
| Joint continuous | 2 | 2/16 | 4/16 | 2/8 |
| Direct gradient | 0 | 4/16 | 4/16 | 2/8 |
| Direct gradient | 1 | 8/16 | 4/16 | 2/8 |
| Direct gradient | 2 | 2/16 | 4/16 | 2/8 |

Eight correct is not a strong result on this domain: always predicting the
one-different class gives 8/16. Replica 1 already made that constant prediction
at initialization. Most post-A models predicted one class for every input; only
joint replica 0 predicted two. All post-B models predicted adjacent pairs for
every input. Classification was read from direct prediction with labels absent,
never from the y activity clamped to the training target.

Block B introduced two rotated patterns and a frequency shift toward two existing
classes, not new relational semantics. Original-training accuracy fell from 3/8
to 2/8 in joint replica 0 and remained 2/8 in the other runs. Since grouping was
not first learned reliably, these are interference/recency observations rather
than a successful-learning-then-forgetting demonstration. Two of the original
held-out patterns became training evidence in B; only the remaining six remain
unseen afterward.

## 2. Completion: the observed success was entirely default-compatible

Every checkpoint/model/plasticity setting matched the semantic completion oracle
on 18/48 queries. The post-run default-only control also gets 18/48: insert the
observed bits into 0011 and leave all hidden bits unchanged. Across all 1,152
queries, every thresholded final answer equals that initialized pattern.

There were continuous changes: maximum hidden-coordinate displacement was 0.23335.
But none crossed the binary decision threshold. Thus the matching score does not
demonstrate relational reconstruction, even though some continuous errors/energies
changed. The class was supplied as explicit context; even that did not produce
the required hidden-bit changes.

| Training arm | Probe weights | Queries across A/B and replicas | Oracle matches | Activity settled at 1e-5 |
| --- | --- | --- | --- | --- |
| Joint continuous | Frozen | 288 | 108 | 288 |
| Joint continuous | Adapting | 288 | 108 | 0 |
| Direct gradient | Frozen | 288 | 108 | 48 |
| Direct gradient | Adapting | 288 | 108 | 0 |

All adapting probes exhausted 300 steps without reaching the activity tolerance.
This is a truncation result, not evidence of a perpetual oscillation. Parameters
continued moving: mean norm drift per 48-query group ranged from 0.154 to 0.188
for joint-trained models and from 0.494 to 0.576 for directly trained models.
All accepted fixed-evidence steps satisfied the declared descent tolerance;
no backtracking rejection was needed anywhere in the run. A descending energy
and finite trajectories are not sufficient tests of useful inference or learning.

## 3. Diagnosis: weak relational features and inadequate nonlinear separation

The post-run diagnostic made no updates or refits. It verified source/result
hashes, reconstructed all checkpoint predictions, and checked exact structural
identities. Its scope was declared separately after the negative run was known;
the diagnosis is explanatory/exploratory, not a fresh confirmation study.

Let u=f(00), v=(f(01)+f(10))/2, and w=f(11) be local channel responses. Shared
pooling represents the one-different class at two points, (u+v)/2 and (w+v)/2.
The adjacent-pair representation is their exact midpoint, (u+w+2v)/4. A linear
decoder cannot correctly assign the two endpoints to one-different and the middle
to adjacent-pairs. The continuous decoder must learn a nonlinear separation.

The sigmoid decoder technically supplies nonlinearity, but its observed response
to this distinction was weak. Post-A output-vector differences between the
one-different class mean and adjacent-pair prediction were:

| Arm | Replica 0 | Replica 1 | Replica 2 |
| --- | --- | --- | --- |
| Joint continuous | 1.04e-5 | 4.33e-6 | 3.15e-8 |
| Direct gradient | 1.86e-5 | 3.27e-4 | 1.18e-5 |

These mean gaps are not complete separability tests. The diagnostic also checked
the actual pairwise correct-class margins at both endpoints and the midpoint:
their minimum was negative for every post-A model. No model achieved this
necessary distinction. The sigmoid-curvature bound derived in the diagnosis
held at every checkpoint.

In the joint arm, local relational contrast ||(u+w)/2-v|| decreased strongly:

| Replica | Initial | Post-A |
| --- | --- | --- |
| 0 | 0.01861 | 0.001348 |
| 1 | 0.04821 | 0.0003515 |
| 2 | 0.004674 | 0.000003618 |

Thus there is evidence of diminished input relation signal as well as a decoder
that fails to exploit the remaining nonlinear distinction. The direct arm did
not uniformly lose this feature contrast—replica 1 increased it—yet still failed
classification. Feature collapse alone therefore does not explain both arms.

## 4. Diagnosis: recency and alternative ways to reduce error

At post-A, 15/16 predictions in joint replica 0 and 16/16 in every other model
equaled the final presented label of the block. After B, all 16 predictions in
all six models equaled its final label, adjacent pairs. This is strong consistency
with recency dominance, but no counterfactual schedule intervention was performed.

For the shared direct-prediction objective, the diagnostic decomposed the mean
gradient on the balanced eight-pattern training set into (a) a gradient toward
the constant mean target and (b) a label-contrast contribution. At post-A the
output-layer constant-target gradient norm exceeded the label-contrast norm by
roughly hundreds to hundreds of thousands in the joint arm. This measures a
weak discriminatory training signal relative to adjustment of overall output;
it is not reconstruction of the joint arm's unsaved online latent-state gradients.

The completion initialization exposes a second issue exactly. Because z and h
start at their feedforward predictions, their local residuals are zero. Hidden
x starts at a binary stored default, so its default/binary derivatives are zero
too. At all 576 checked post-A/post-B query initializations:

- The hidden-input gradient was exactly zero.
- Data-dependent W and A gradients were zero (only weight decay remained).
- The data-dependent B gradient had norm 0.615–2.279, while h also received an
  immediate prediction-error signal.

The system can initially reduce class error by modifying its decoder or hidden
explanation before evidence-driven corrections reach the missing input. When
weights adapt during completion, that is an available alternative to changing the
answer. This identifies a mechanism worth isolating, not proof that it accounts
for every later trajectory. Neither reducing an energy nor making a label easier
to explain guarantees learning to predict it from input.

## 5. Numerical correctness and computation

Central finite differences checked all joint state/parameter derivatives and
direct predictive parameter derivatives. Maximum absolute disagreements were
2.03e-10 and 9.37e-11 respectively, below the declared 2e-7 tolerance. Clamps
were preserved and no nonfinite value, line-search exhaustion, or correctness
stop occurred. Both the main run and diagnosis passed on their first execution.

The main run used 408,000 training update attempts, 10,200 presentations of
10 distinct labeled patterns, and 345,600 completion probe steps. There were
594,001 accepted joint steps including probes, benchmarks and the check step;
204,000 accepted direct training steps; 1,234,949 joint objective calls;
639,554 joint gradient calls; and 204,001 direct gradient calls. All six
trajectories and all diagnostic costs are included, with no best-seed selection.

| Main scope | Wall seconds |
| --- | --- |
| Learning, both arms and all replicas | 79.123 |
| Completion diagnostics | 83.276 |
| Inference benchmark batches | 16.427 |
| Other named phases, including correctness/serialization | 0.098 |
| Whole run, including driver overhead | 178.930 |

Whole-run CPU time was 178.903 s. Peak process RSS was 44,784 KiB. Each model
uses 704 learned-array bytes and 256 activity-array bytes; no optimizer, covariance,
or replay. Gradient/candidate arrays and Python bookkeeping are extra workspace,
not included in those model-byte counts. All checkpoint arrays/logs and shadow
copies are separately scoped in the receipt. NumPy 2.5.4, float64, one CPU process,
one BLAS/OMP thread, Ryzen 5 3600. This is not a scaling claim.

The post-run diagnosis cost 0.2280 s wall / 0.2279 s CPU, with peak RSS 41,320 KiB
including parsed artifacts. It used 18 small full SVDs, 288 direct gradients,
576 joint initialization gradients, and no parameter updates. A preliminary
read-only summary extraction also took approximately 0.0119 s wall/CPU; it
performed no model updates and is not included in the main or diagnosis timing.

## 6. Inference comparator and interpretation of settling cost

For fully observed x and free y, the direct function is the exact zero-prediction-
residual solution. The replica-0 post-A benchmarks used one warmup plus three
measured batches of all 16 patterns. Median amortized direct times were about
45.0 microseconds/query (joint-trained weights) and 42.3 microseconds/query
(direct-trained weights). Batch min/max, CPU timings and all samples are archived.

Iterative settling of the joint-trained model took about 183.5 ms/query and
reached the 500-step budget on all 16 patterns without satisfying the activity
tolerance. Maximum output disagreement with the direct function was 0.000928.
Do not compare this as equal-tolerance inference. Direct-trained weights settled
in 3,100 steps across 16 patterns (all met the activity tolerance), at about
71.3 ms/query, with maximum direct-output disagreement 0.0000204. Settled-path
timing includes one direct-reference call, allocation, gradients and line search.
Neither iterative benchmark supplies a practical advantage over direct inference.

## 7. Decision after analysis

This version demonstrates implementable simultaneous state/weight descent, but
not useful continuously learned grouping or completion. Both arms fail, so the
result cannot decide whether the main bottleneck is the continuous mechanism,
the representation, or its optimization under this stream.

Before a larger learning run, the most informative separation is to establish
a constructive classifier in the SAME bounded sigmoid architecture, then isolate
the midpoint-separation nonlinearity and the presentation/learning timescale.
For completion, an intervention must distinguish correcting the missing state
from adapting the decoder to explain the supplied label. These need prospective
tests; no revised rates, extra passes, or frozen-weight success should be passed
off as a solution to continuously adaptive inference.
