"""Folds, cross-validated phase selection, refit and diagnostics for locality adaptation v2.

Contract: docs/locality-adaptation-v2-contract.md. Estimators, generator and metrics
are imported unchanged from v1's frozen module.
"""
import numpy as np

from src.locality_adaptation import (B, Meter, encode_local, finite, fit_adaptive, fit_global,
    fit_local, local_basis, orthonormal_basis, select)


FOLDS = 4
ROOT_SEED = 2026102408
ARMS = ('cv_refit_cca', 'fixed_cca_all', 'global_cca_all', 'holdout_refit_cca', 'holdout_cca',
        'oracle_structural_phase_cca_all')
REFIT_ARMS = ('cv_refit_cca', 'holdout_refit_cca', 'oracle_structural_phase_cca_all')


def folds(n):
    """Fold of the pair with one-based index i is i mod 4; fold 0 is v1's validation split."""
    return np.arange(1, n+1) % FOLDS


def score_sum(xv, yv, phase, weights):
    """sum_i h1_i^T P h2_i through the deployed O(d) local code."""
    return float(np.sum(encode_local(xv, phase, weights)*encode_local(yv, phase, weights)))


def cv_scores(x, y, b=B, meter=None):
    """cv_score_p = (1/n) sum_i h1_i^T P_{p,-f(i)} h2_i; fold candidates are discarded after scoring."""
    meter = meter or Meter()
    fold, totals = folds(len(x)), np.zeros(b)
    for f in range(FOLDS):
        train, held = fold != f, fold == f
        for p in range(b):
            w = fit_local(x[train], y[train], p, 'cca', b, meter)
            with meter.time('validation'):
                totals[p] += score_sum(x[held], y[held], p, w)
            meter.counts['candidate_scores'] += 1
            meter.counts['validation_row_exposures'] += int(held.sum())
    return finite(totals/len(x))


def refit(x, y, phase, b=B, meter=None):
    """Selected phase fitted on all n pairs; timed under refit_* stages."""
    meter = meter or Meter()
    sub = Meter()
    w = fit_local(x, y, phase, 'cca', b, sub)
    for stage, seconds in sub.seconds.items():
        meter.seconds[f'refit_{stage}'] += seconds
    meter.counts.update(sub.counts)
    meter.counts['refits'] += 1
    return w


def fit_cv_refit(x, y, b=B, meter=None):
    meter = meter or Meter()
    scores = cv_scores(x, y, b, meter)
    with meter.time('validation'):
        chosen = select(scores)
    return chosen, refit(x, y, chosen, b, meter), list(scores)


def fit_holdout_refit(x, y, b=B, meter=None):
    meter = meter or Meter()
    chosen, _, scores = fit_adaptive(x, y, 'cca', b, meter)
    return chosen, refit(x, y, chosen, b, meter), scores


def fit_arm(arm, x, y, b=B, meter=None, phase=None):
    """Returns (orthonormal basis, local phase or None, record or None).

    `phase` is a declared phase, used only by the oracle structural-phase reference,
    whose phase the evaluator supplies; no fitter receives world truth."""
    meter = meter or Meter()
    d = x.shape[1]
    if arm == 'cv_refit_cca':
        chosen, w, scores = fit_cv_refit(x, y, b, meter)
        return local_basis(w, chosen, d), chosen, dict(scores=scores, weights=w)
    if arm == 'holdout_refit_cca':
        chosen, w, scores = fit_holdout_refit(x, y, b, meter)
        return local_basis(w, chosen, d), chosen, dict(scores=scores, weights=w)
    if arm == 'holdout_cca':
        chosen, candidates, scores = fit_adaptive(x, y, 'cca', b, meter)
        return local_basis(candidates[chosen], chosen, d), chosen, dict(
            scores=scores, weights=candidates[chosen],
            candidates=[local_basis(w, p, d) for p, w in enumerate(candidates)])
    if arm == 'fixed_cca_all':
        w = fit_local(x, y, 0, 'cca', b, meter)
        return local_basis(w, 0, d), 0, dict(weights=w)
    if arm == 'oracle_structural_phase_cca_all':
        w = fit_local(x, y, phase, 'cca', b, meter)
        return local_basis(w, phase, d), phase, dict(weights=w)
    if arm == 'global_cca_all':
        return orthonormal_basis(fit_global(x, y, 'cca', d//b, meter)), None, None
    raise ValueError(arm)


def all_n_candidates(x, y, b=B, meter=None):
    """Evaluator diagnostic: every phase fitted on all n pairs (not candidate cost)."""
    return [fit_local(x, y, p, 'cca', b, meter) for p in range(b)]
