# Phase selection: uncertainty, refit mismatch and reconstruction scoring

Status: analytical and retrospective exploratory study, 2026-10-24.
This does not change v1/v2 outcomes or launch a v3 confirmation experiment.

## Scope declared before calculation

Derive Gaussian pairwise score moments for fixed symmetric projector
differences, and a clean-target-free reconstruction score. Verify identities
on explicit noncommuting matrices through independent Gaussian fourth-moment
expansion. Analyze the limitations of applying held-out identities to CV.

Replay only the 64 existing v2 aligned worlds at n=16 and 32, chosen because
the published result locates the remaining deficit there. Recover fold fits
and verify their CV scores against the frozen artifact. Use the already
recorded all-n candidate losses; do not regenerate confirmation outcomes.
Compare actual CV, oracle mean fold signal loss, oracle mean fold reconstruction
loss, and minimum all-n fitted loss. These are evaluator diagnostics, not
deployable selectors or independent evidence of improvement. No alternative
empirical selector, hyperparameter search, new worlds or significance tests.

## Computation contract

With W=64, two budgets, F=4, K=8, d=32, b=8 and k=4, replay costs
O(W sum_n FK[(3n/4)db+db²]) for fitting and O(W sum_n Knd) for scores.
Dense diagnostic projector/moment calculations cost at most O(W FK d³)
per budget. Raw pairs need O(256d), one local fit O(db) statistics and O(d)
weights, and dense diagnostic matrices O(d²); output rows need O(WK) per
budget. Loading the historical JSON additionally uses O(artifact size).
The independent fourth-moment check uses O((2d_check)^4) time and O(d_check²)
workspace, with d_check=4. Direct Cholesky/eigh solves, float64, one process
and one BLAS thread; no optimizer, tuning or deployment benchmark.

All regenerated fits and scoring exposures count as this study's diagnostic
cost. Historical all-n fits are reused, not credited as newly executed or free
learner work; their original costs are in v2's runtime receipt. Scientific
JSON is deterministic. A separate runtime receipt records phase wall/CPU
time, RSS scope, source/input hashes, environment and counters. No frozen
historical file is modified.

## Pairwise moments: correlation between phases matters

Let x,y be zero-mean jointly Gaussian views with marginal covariance T=S+N
and cross-covariance S, and let A=P-Q for fixed symmetric projectors.
The difference in signal-validation scores is z_A=xᵀAy. Then

```
E[z_A] = tr(AS)
Var(z_A) = tr(ATAT) + tr(ASAS).
Cov(z_P,z_Q) = tr(PTQT) + tr(PSQS).
```

These follow by expanding the fourth Gaussian moment and subtracting the
product of means. For m independent held-out pairs, divide the variance by
m. This retains the covariance of competing phase scores rather than treating
them as independent. If Δ=tr(AS)>0, Cantelli's inequality gives
`Pr(mean(z_A) <= 0) <= V/(V+m Δ²)`. This is a sufficient upper bound on
one comparison's error, not an impossibility result or a practical sample
size prescription. When A=0 the variance and gap are both zero, so the
bound with Δ>0 does not apply.

For independent fitting data, these identities hold conditional on both
fitted candidates. Their use for CV needs care: each fold's held-out score
has the stated conditional moments, but other fold fits use those same
held-out pairs. Conditioning on all fold fits does not preserve held-out
independence. Unconditionally,

```
Var(CV difference) = (1/F²) [sum_f Var(Z_f)
                              + 2 sum_{f<g} Cov(Z_f,Z_g)],
```

where Z_f is the mean difference on fold f. Each Var(Z_f) itself contains
both conditional validation variance and variation in its conditional mean.
Neither four independent folds nor n independent scores can be assumed.
This study does not estimate the cross-fold terms or derive a CV error bound.

## A reconstruction-aligned score, and its variance cost

For rank-k orthogonal P, normalized reconstruction risk is
`E(P)=[tr(S)+tr(PT)-2tr(PS)]/k`. The unknown tr(S) is common to every phase.
Thus minimizing the following held-out score requires no clean targets:

```
r_P = (xᵀPx + yᵀPy)/2 - 2 xᵀPy
E[r_P] = tr(P(T-2S)).
```

The difference r_A has the same expression with A=P-Q. Set
u=(x+y)/sqrt(2), v=(x-y)/sqrt(2). These Gaussian vectors are independent,
with covariances T+S and T-S=N. Since
`r_A = -uᵀAu/2 + 3vᵀAv/2`,

```
Var(r_A) = (1/2) tr(A(T+S)A(T+S)) + (9/2) tr(ANAN)
Var(r_A) - Var(z_A) = 4 tr(ANAN) >= 0.
```

The final term equals four times the squared Frobenius norm of
N^(1/2) A N^(1/2). This comparison is for unnormalized single-pair scores;
divide both variances by k² for normalized scores, and by m for independent
held-out means. With isotropic nuisance and equal rank, tr(AN)=0, so r_A
and -z_A have the same mean, but r_A has greater variance unless the extra
term vanishes. This is not a general misselection-probability ordering.
With anisotropic nuisance, their target gaps differ: reconstruction alignment
can still be worth its added variance. It does not automatically help.

## Retrospective diagnostics of v2

The replay reproduces all 128 stored eight-phase CV score vectors and selected
phases within 1e-10. No all-n model is fitted again; its recorded signal loss
and projected noise are reused. The oracle fold-signal choice minimizes the
mean *true* signal loss of the four fitted fold candidates. The oracle fold-risk
choice instead minimizes their mean true reconstruction risk. Neither uses
noisy validation scores; both are forbidden evaluator information for a learner.

| Diagnostic, 64 aligned worlds per budget | n=16 | n=32 |
| --- | ---: | ---: |
| Actual CV versus oracle fold-signal phase disagreements | 24 | 1 |
| Oracle fold-signal versus best all-n signal phase disagreements | 3 | 1 |
| Oracle fold-signal versus oracle fold-risk disagreements | 1 | 0 |
| Actual reconstruction excess over fixed phase 0 | 0.071626 | 0.000570 |
| Reconstruction excess with oracle fold-signal choice and all-n fit | 0.000973 | 0 |
| Reconstruction excess with oracle fold-risk choice and all-n fit | −0.000353 | 0 |

Phase disagreements alone are not loss measures; tied or near-tied choices
can differ harmlessly. To quantify their cost, let a be actual CV's phase,
o the oracle fold-signal phase, and b the best all-n signal phase. The exact
retrospective identity is

```
L_all(a)-L_all(b)
  = [L_all(a)-L_all(o)] + [L_all(o)-L_all(b)].
```

The first term is signed, not a guaranteed nonnegative regret. At n=16 its
mean is 0.055456, versus 0.000963 for the second term: about 98.3% of mean
excess over the best all-n fit lies in the actual-versus-oracle fold choice.
At n=32 these terms are 0.000463 and 0.000121. The best all-n fitted phase
can outperform fixed phase 0, so this decomposition is not identical to the
primary fixed-baseline contrast.

As a separate conditional diagnostic, across each fold's phases 1–7 versus
phase 0, mean single-fresh-pair difference variance is 2.234 for signal scoring
and 4.181 for reconstruction scoring at n=16 (2.208 and 3.365 at n=32).
These are hypothetical independent-validation moments conditional on each
fit, averaged over existing worlds. They are **not CV variances** and are
not a new selector performance comparison.

The oracle fold-signal choice almost removes the reconstruction deficit, and
changing its oracle objective to reconstruction changes only one n=16 choice.
On these already inspected worlds, noisy score selection is therefore the
main diagnostic explanation; fold-to-refit mismatch and objective mismatch
are small. This does not prove an information-theoretic limit or establish
that a better estimator cannot work. No fresh confirmatory evidence is added.

## Decision

There is no strong mechanism here for a v3 that simply changes to a
reconstruction score or changes refitting. Such a score has a demonstrable
variance cost, and oracle objective mismatch was rare. More complex selection
would need a specific variance-reduction argument and counted cost before
another confirmation run.

The next experiment should instead test source-to-target representation reuse
and controlled drift, following the existing transfer analysis: learn phase
and weights on a counted source stream, then compare frozen reuse, counted
updates, fresh local/global fits and history-matched sufficient-statistic
controls on target pairs. Separate acquisition from reuse cost; include
source retention and fresh target reconstruction. A new prospective contract
must explicitly state that proceeding revises the earlier pass-contingent
research sequence: v2 remains an overall failure, and its local reference
has a disclosed n=16 acquisition deficit. This study does not authorize
claiming that reuse succeeds before that experiment.

## Verification and artifacts

Run `.venv/bin/python -m experiments.phase_selection_analysis`.
Three explicit cases (a projector, a nonzero difference of projectors, and
zero difference) verify score expectations, both variances and their difference
against independent fourth-moment expansion. The matrices include noncommuting
signal and noise covariances. Replay checks historical CV scores and deployed
metrics and verifies the signed decomposition. A second run must reproduce
the scientific JSON byte-for-byte.

[Scientific report](../reports/phase-selection-analysis.json) records all
128 diagnostic rows and moment checks; [runtime receipt](../reports/phase-selection-analysis-runtime.json)
records actual computation separately. One execution performs 4,096 fold
fits, 16,384 Cholesky and 16,384 eigh solves of size 8, 73,728 fitting-row
and 24,576 validation-row exposures. It regenerates 64 length-256 historical
streams but uses only the first 32 pairs of each (2,048 distinct pairs).
It reuses 1,024 historical all-n fits. Tuning trials and new confirmation
worlds are zero. RSS is a whole-process peak, not model memory; FLOPs and
isolated temporary allocation measurements are unavailable. Timings include
diagnostic matrix operations and are not deployment benchmarks.
