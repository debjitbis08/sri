# Analysis 2: useful retained structure versus stale retained activations

**Status:** analytical development study, 2026-10-24. Extends the existing
[related-input analysis](related-input-analysis.md), which already derived
the bias/variance trade-off for a supplied old coarse activation. The new
question is when a learned representation helps *fresh coefficients* and
how to evaluate retained structure without confusing it with activation reuse.

## A direct usefulness criterion

Let a frozen rank-k orthogonal projector P be learned from past observations.
A new clean signal s has covariance S_new and zero mean, and the current
observation is `h=s+e`, with independent zero-mean noise covariance N.
Assume P is independent of this fresh noise. Orthogonal reconstruction gives

```
E ||P h - s||² = tr((I-P) S_new) + tr(P N).
E ||h-s||² = tr(N).
```

Thus retained structure helps exactly when
`tr((I-P) S_new) < tr((I-P) N)`. Derivation: signal residual and projected
noise occupy orthogonal subspaces, so their cross term vanishes; take traces
of their second moments. Under isotropic noise `N=sigma² I`, the threshold
is `(d-k)sigma²`. This is a downstream reconstruction result, not merely
similarity between representations. It does not require the new coefficients
to resemble any previous activation.

If `S_new=tau² Q`, for another rank-k projector Q with principal angles
theta_i to P, then

```
tr((I-P) S_new) = tau² sum_i sin²(theta_i).
```

This gives a continuous, computable transfer boundary. For d=32, k=4,
tau²=1 and equal angles, with sigma²=0.01, reuse helps only when
`sin²(theta) < 0.07` (theta below about 15.34 degrees). At sigma²=0.25
the threshold is 7, larger than the maximum residual 4: even a wrong rank-four
projection beats raw noisy input. Therefore **beating raw input is too weak
a success criterion** at that noise level. Include newly fitted global/local
CCA and an oracle current subspace; report signal distortion as well as
total denoising error. For learned P the same conditional identity holds,
then averages over training randomness. No oracle P may be credited as
learned performance.

## Retaining a previous activation is a different hypothesis

Within a known correct k-dimensional subspace, suppose old and new clean
coefficients are independent, zero-mean, covariance tau² I. The old estimate
has independent noise w I, the new estimate v I. Blending them with old
weight a gives risk per coordinate

```
R_old(a) = (1-a)² v + a² (2 tau² + w).
```

This extends the previous fixed-change formula by averaging over independent
fresh coefficients. However, comparison to current input alone is misleading:
shrinking the new estimate toward zero with the same a has risk

```
R_zero(a) = (1-a)² v + a² tau².
R_old(a)-R_zero(a) = a² (tau²+w) >= 0.
```

Old-state blending may beat a noisy current estimate through shrinkage while
being worse than a memoryless shrinkage control. For correlated coefficients
with `Cov(c_old,c_new)=rho tau² I`, the difference becomes
`a² ((1-2rho) tau²+w)`. Genuine temporal relatedness can reverse
the comparison. Correlation, uncertainty, and a must be declared or learned
from permitted training history; test coefficients cannot set them.

Consequently separate three regimes: fresh combinations in a stable subspace,
temporally correlated coefficients, and changing subspaces. Do not use one
aggregate score to credit activation persistence for representation transfer.
Paired observations also cannot identify a persistent distractor with the
same statistics as signal (the preflight's identifiability counterexample).
If a task requires that distinction it needs additional permitted information.

## What to retain and what it costs

A fixed orthonormal basis needs O(dk) weights and O(dk) projection work
(O(d) for one disjoint feature per block). An old activation adds O(k)
state and blend work. Reconstructing a full surface also costs O(dk), and
materializing d outputs costs at least O(d). A full streaming cross-covariance
needs O(d²) persistent statistics and O(d²) work per pair; local block
statistics need O(db). An online rank-k basis can use O(dk) state, but must
pay orthogonalization/update costs and demonstrate preservation empirically.

A sufficient-statistic reference can retain cross-products and marginal
covariances without raw-example replay. Give it the same past pairs and count
its memory. A neural local rule must compete against that reference, a frozen
basis, a freshly fitted basis, and a history-matched direct encoder. None of
the equations implies reciprocal settling is necessary.

## Decision and experiment order

1. First run the bounded locality-adaptation experiment from
   [analysis 1](locality-adaptation-analysis.md). Its obstruction and proposed
   correction are explicit; a new neural mechanism is unnecessary to test it.
2. Next test **representation reuse across fresh combinations and controlled
   subspace drift**. Learn on a source stream, freeze or update on counted
   target pairs, and evaluate fresh target observations. Cross coefficient
   correlation with subspace drift; keep nuisance changes separate. Compare
   reconstruction, clean signal loss, target acquisition, and source retention
   at fixed example and compute budgets. The source-retention test is evaluator
   data, not feedback available to a no-replay learner.
3. Add a local plasticity/retained-context candidate only after specifying how
   it uses history differently from the direct and sufficient-statistic controls.
   An improvement that comes solely from shrinkage or a supplied grouping must
   be attributed to that mechanism.

The second study's analytical predictions use known/frozen P as a ceiling.
The proposed experiment must measure errors from learning P and choosing its
update rate. Selection uses development data; confirmation uses fresh worlds.
Early stopping and regularization controls from the corrected v1 report remain
required when neural baselines return. Neither analysis justifies repeating
the full neural benchmark before these narrower questions are resolved.

## Verification receipt

Run `.venv/bin/python -m experiments.adaptation_analysis`. The
[report](../reports/adaptation-analysis.json) checks ten angle/noise cases
using full linear error maps against the principal-angle risk formula, plus
twelve activation-weight/correlation cases using the full joint covariance.
All checks pass at absolute tolerance 1e-12. The low-noise transfer boundary
is 15.342 degrees. No sampled examples, fitted models, or confirmation runs
were used. Cost measurements are in the separate
[runtime receipt](../reports/adaptation-analysis-runtime.json).
