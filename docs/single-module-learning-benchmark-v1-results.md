# Single-module learning benchmark v1 — the locality prior wins on aligned tasks and pays when misaligned

**Completion:** complete, 2026-10-24. Development selection ran first. The
manifest (source hashes, selected configurations) was written next.
Confirmation then ran on 32 fresh worlds; its hash check passed. One
post-hoc exploratory check (section 5) used development worlds only.

[Contract](single-module-learning-benchmark-v1-contract.md) ·
[manifest](../reports/single-module-learning-benchmark-v1-manifest.json) ·
[development](../reports/single-module-learning-benchmark-v1-development.json) ·
[results](../reports/single-module-learning-benchmark-v1-results.json) ·
[training check](../reports/single-module-learning-benchmark-v1-training-check.json) ·
[runtime](../reports/single-module-learning-benchmark-v1-runtime.json).

## 1. Main findings

1. **Primary outcome, aligned regime.** The candidate is **better** than all four
   comparators on both H1 (low-data) and H2 (resilience). The comparators are
   dense linear, MLP, conv and paired DAE, all Holm p < 1e-16. By the declared
   rule this meets "faster learning than the established arms", but only
   within this task.
2. **The source is the locality prior.** The candidate is the
   learned coarse module, but the preflight proved its settled map and
   gradients equal a feed-forward local encoder. Its advantage over
   `dense_linear_vicreg` (same objective, same code dimension, no locality)
   is +0.116 low-data and +0.606 resilience. Reciprocal settling contributes
   nothing that this experiment can detect.
3. **An analytical estimator beats every learned method, the candidate included.**
   Local ridge CCA exceeds the candidate by 0.081 low-data and 0.103
   resilience on aligned tasks. Global CCA, with no locality, is within 0.02
   of the candidate on low-data.
4. **Locality costs when it is wrong.** On dense-rotated tasks the
   candidate is **worse** than dense linear (−0.130) and conv (−0.039) on
   low-data. At 256 pairs it reaches R² 0.527, against 0.725 for dense
   linear and 0.772 for global CCA. Shifted tasks fall in between.
5. **H2 combines two effects.** At 32 pairs the candidate starts higher
   (in-distribution +0.27 to +0.48 over each comparator), and it also loses
   less under the shifted conditions (mean drop smaller by 0.13 to 0.29).
   Both are significant, and section 4 separates them.
6. **Undertraining is less likely but not ruled out.** Lower learning rates
   and four times as many steps did not close the gaps (section 5). Conv and
   DAE get worse with longer training. That fits overfitting, but it does not
   prove it, and it does not explain the loss oscillations. Early stopping and
   stronger regularization remain untested controls.

## 2. Learning curves, in-distribution probe R² (32 confirmation worlds)

Aligned:

| Method | 4 | 8 | 16 | 32 | 64 | 128 | 256 | Low-data score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **coarse_local_vicreg** | 0.251 | 0.404 | 0.584 | 0.719 | 0.757 | 0.776 | 0.785 | **0.611** |
| dense_linear_vicreg | 0.240 | 0.345 | 0.446 | 0.407 | 0.608 | 0.689 | 0.730 | 0.495 |
| mlp_vicreg | 0.127 | 0.228 | 0.318 | 0.414 | 0.473 | 0.544 | 0.531 | 0.377 |
| conv_vicreg | 0.231 | 0.305 | 0.400 | 0.452 | 0.516 | 0.518 | 0.621 | 0.435 |
| dae_paired | 0.115 | 0.167 | 0.228 | 0.242 | 0.242 | 0.260 | 0.421 | 0.239 |
| local_cca (reference) | 0.462 | 0.597 | 0.690 | 0.752 | 0.772 | 0.782 | 0.787 | 0.692 |
| global_cca (reference) | 0.294 | 0.447 | 0.553 | 0.624 | 0.702 | 0.748 | 0.771 | 0.591 |
| local_spectral | 0.265 | 0.378 | 0.490 | 0.582 | 0.668 | 0.711 | 0.757 | 0.550 |
| fixed_averaging | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 | 0.482 |

These are mean R² values, and 95% intervals over worlds are stored in the
results file. First budget with mean R² ≥ 0.6: local CCA 16; candidate
and global CCA 32; dense linear 64; conv 256; MLP and DAE not reached by 256.

Fixed averaging beats every learned method at 4 pairs. Its prior happens to
overlap the smooth templates, but it never improves.

`dense_linear_vicreg` dips at 32 pairs (0.407, below 0.446 at 16). The
selected learning rate changes between those budgets (3e-2 to 3e-3).
The development table shows the same ordering, so this reflects per-budget
selection in this arm, not a confirmation artefact.

Dense regime, 256 pairs: candidate 0.527, dense linear 0.725, conv 0.613,
MLP 0.549, global CCA 0.772, local CCA 0.518.

## 3. Primary and secondary tests (candidate minus comparator, mean [95% CI])

| Regime | Comparator | H1 low-data | Outcome | H2 resilience | Outcome |
| --- | --- | --- | --- | --- | --- |
| aligned | dense linear | +0.116 [0.102, 0.130] | better | +0.606 [0.567, 0.645] | better |
| aligned | MLP | +0.234 [0.222, 0.247] | better | +0.437 [0.400, 0.474] | better |
| aligned | conv | +0.176 [0.163, 0.189] | better | +0.441 [0.401, 0.480] | better |
| aligned | DAE | +0.372 [0.356, 0.387] | better | +0.613 [0.573, 0.653] | better |
| aligned | local CCA | −0.081 [−0.091, −0.071] | descriptive | −0.103 [−0.122, −0.084] | descriptive |
| aligned | global CCA | +0.020 [0.009, 0.030] | descriptive | +0.200 [0.167, 0.234] | descriptive |
| shifted | dense linear | −0.019 [−0.036, −0.002] | no practical difference | +0.326 | better |
| shifted | conv | +0.034 | better | +0.157 | better |
| dense | dense linear | −0.130 [−0.144, −0.116] | worse | +0.085 | better |
| dense | conv | −0.039 [−0.051, −0.027] | worse | −0.075 | worse |
| dense | MLP | +0.013 | no practical difference | −0.078 | worse |

Shifted and dense rows are secondary. The full table, including Holm-adjusted
p-values, is in the results file.

## 4. What resilience measured

At 32 pairs, aligned, by condition:

| Method | In-distribution | Held-out combination | Changed nuisance (texture ×4) | New detail |
| --- | --- | --- | --- | --- |
| coarse_local_vicreg | 0.719 | 0.642 | 0.690 | 0.312 |
| dense_linear_vicreg | 0.407 | 0.218 | 0.329 | −0.720 |
| mlp_vicreg | 0.414 | 0.171 | 0.113 | 0.049 |
| conv_vicreg | 0.452 | 0.221 | 0.219 | −0.118 |
| local_cca | 0.752 | 0.687 | 0.723 | 0.541 |
| local_spectral | 0.582 | 0.457 | 0.315 | 0.550 |

**What "new detail" is.** Its directions are not absent from training. Every
training view has isotropic noise of variance 0.25 in all 32 directions. The
condition adds variance 2 in the directions orthogonal to the signal and
texture subspaces, so their total variance goes from 0.25 to 2.25. This tests
resilience to a ninefold increase of nuisance in weakly excited directions,
not to entirely unseen ones. An unconstrained code can put weight on those
directions, because the weak training variance penalized them only a little.
When they are amplified, R² can go far below zero. A local code has only
eight nonzero weights per unit, so it is less exposed.

**Starting accuracy and degradation.** The resilience score is a level (mean
R² over the shifted conditions). It therefore contains the in-distribution
lead as well as resistance to degradation. Exactly, H2 = (start difference)
+ (difference in mean drop from in-distribution). Paired over the 32
confirmation worlds at 32 pairs, candidate advantage [95% CI], with drop
columns as comparator drop minus candidate drop:

| Comparator | Start (in-distribution) | Mean drop | Held-out combination drop | Changed nuisance drop | New detail drop |
| --- | --- | --- | --- | --- | --- |
| dense linear | +0.312 [0.282, 0.341] | +0.294 [0.264, 0.324] | +0.113 | +0.049 | +0.720 [0.641, 0.799] |
| MLP | +0.305 [0.278, 0.332] | +0.132 [0.105, 0.158] | +0.166 | +0.272 | −0.042 [−0.101, 0.016] |
| conv | +0.267 [0.242, 0.293] | +0.173 [0.147, 0.200] | +0.153 | +0.205 | +0.162 [0.094, 0.231] |
| DAE | +0.477 [0.449, 0.504] | +0.136 [0.104, 0.168] | +0.101 | +0.210 | +0.098 [0.020, 0.176] |

Both parts matter. Against dense linear they are about equal, and the
smaller drop comes almost entirely from new detail (candidate drop 0.407,
dense linear 1.127). Against MLP, conv and DAE, the starting lead is the
larger part. The candidate degrades no less than the MLP under new detail;
its smaller drops there come from held-out combinations and changed nuisance.
So "most of the H2 margin comes from new detail" holds only for the dense
linear comparison. The candidate's smaller drops still come from its
supports, not from its learning dynamics. These figures are computed from
the per-member rows of the results file. They are a post-hoc
decomposition, not a declared test.

Every method loses accuracy on held-out combinations, linear ones
included. The probe is fitted on correlated in-distribution coefficients,
so linear models are affected as well. This condition does not isolate
nonlinear failure to generalize.

## 5. Training-sufficiency check (post hoc, exploratory)

The confirmation flagged many members as still changing by more than 5% over
the last 10% of steps: MLP 672/672, conv 645, DAE 480, dense linear 368,
candidate 162. Proposal section 7 requires ruling out undertraining. I
retrained each arm on the eight aligned development worlds at learning rates
{3e-4, 1e-3} × {1500, 6000} steps, for the larger widths.

| Arm (aligned, dev) | 64 pairs: best at 1500 steps | 64 pairs: best at 6000 steps | 256 pairs: best at 1500 | 256 pairs: best at 6000 |
| --- | --- | --- | --- | --- |
| coarse_local_vicreg | 0.742 | 0.758 | 0.758 | 0.782 |
| dense_linear_vicreg | 0.652 | 0.614 | 0.720 | 0.716 |
| mlp_vicreg | 0.450 | 0.469 | 0.538 | 0.547 |
| conv_vicreg | 0.479 | 0.472 | 0.656 | 0.515 |
| dae_paired | 0.201 | 0.196 | 0.336 | 0.270 |

Longer training raises the MLP by at most 0.02. It *lowers* conv and DAE.
That fits overfitting, but these numbers alone do not show it. Smaller learning rates help conv
at 256 pairs (0.656, against 0.621 for the selected configuration) but leave
a gap of 0.12. The candidate gains slightly more from longer training than
the baselines do. The 5% tail flag remains set even when performance has
plateaued, so it measures VICReg's oscillation near its minimum, not lack
of progress.

**Conclusion:** within the declared search space and a fourfold-longer
schedule, more training does not close the gaps. This narrows the
optimization concern but does not settle it. The check does not separate
overfitting from other causes, such as optimizer instability, and it does
not explain the loss oscillations. Two meaningful controls are still
untested: early stopping on held-out pairs, and stronger regularization.
They could narrow the gaps for the nonlinear arms.

## 6. Interpretation and scope

- **Supported:** with a correctly aligned locality prior, the learned coarse
  module learns broad structure from fewer pairs than generic dense, MLP,
  convolutional and denoising paired-view learners. It starts more accurate
  at 32 pairs and also degrades less on average under the shifted
  conditions. Its new-detail drop is much smaller than dense linear's, but
  not smaller than the MLP's.
- **Not supported:** any advantage from the module's reciprocal or
  shared-surface organization. Those are exactly equivalent to a local
  feed-forward encoder here. Nor is there an advantage over the
  analytical reference, local CCA, which is better.
- **Cost of the prior:** a fixed block layout loses to dense learners once
  the structure rotates away from it, and its asymptote stays low.
  Proposal section 8 applies: "gain only on aligned synthetic tasks: report
  the domain restriction".
- **Limits:** one synthetic linear-Gaussian generator. No less tailored
  secondary task (required by proposal section 5 before a broad claim), no
  early stopping, VICReg without an expander head, and parameter counts not
  matched for nonlinear arms. The conv arm's spatial bias (k=5 translation
  equivariance) is a different, weaker prior than block locality.

The next substantive question remains the one the preflight named. What
candidate makes the module *distinct* from its feed-forward control? A
retained-context or local learning rule, or adaptive grouping that removes
the misaligned-locality cost. This benchmark's harness, arms and references
can be reused as its controls.

## 7. Reproduction

```
.venv/bin/python -m unittest tests.test_learning_benchmark -v
.venv/bin/python -m experiments.learning_benchmark --phase all --workers 8
.venv/bin/python -m experiments.learning_benchmark_training_check
```

Runtime with 8 worker processes and one thread each: development 48 min,
confirmation 32 min.
