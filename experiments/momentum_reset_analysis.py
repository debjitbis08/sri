"""Analysis of momentum reset v1 under the frozen contract (docs/momentum-reset-v1-contract.md, "Readings").

Per arm over ten seeds (340-349). D(t) = recall(t) - recall(0); branch differences are D(branch) - D(reference), positive meaning less
loss. Intervals and outcomes use fan_in_pilot_analysis.classify_ci (margin 0.05) and, for "noninferior", a lower bound above -0.05.
Timing goes to a separate -analysis-runtime.json.

Usage: .venv/bin/python -m experiments.momentum_reset_analysis
"""
import gzip
import json
import time

import numpy as np

from experiments import fan_in_pilot_analysis as AN
from experiments import number_systems as X

ROOT = X.ROOT
SEEDS = list(range(340, 350))
ARMS = ('sparse48', 'dense', 'mixed', 'mixed-h125')
CHANCE, MARGIN = 1 / 9, .05
TIMES = (50, 250, 2000)
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
LOSS_WINDOWS = ((1, 50), (51, 250), (251, 2000))
LOSS_NAMES = ('match', 'next', 'compare', 'transition')
OUT = ROOT / 'reports' / 'momentum-reset-v1-analysis.json'
RUNTIME = ROOT / 'reports' / 'momentum-reset-v1-analysis-runtime.json'


def summarize(d):
    d = np.asarray(d, float)
    ci = AN.interval(d)
    return {'mean_ci': ci, 'outcome': AN.classify_ci(d, 'acc'), 'noninferior': bool(ci[1] > -MARGIN)}


def interval(d):
    return {'mean_ci': AN.interval(np.asarray(d, float))}


def point(rec, t, key='recall'):
    return next(p[key] for p in rec['curve'] if p['offset'] == t)


def delta(R, arm, branch, t):
    return np.array([point(R[f'{arm}/seed{s}']['branches'][branch], t) - point(R[f'{arm}/seed{s}']['branches'][branch], 0) for s in SEEDS])


def final(R, arm, branch):
    return np.array([point(R[f'{arm}/seed{s}']['branches'][branch], 2000) for s in SEEDS])


def a_of(snap):
    return float(np.mean([np.mean(snap['probe'][k]) for k in TRANSFER_SETS]))


def context_perf(snap):
    e = snap['evaluate']
    return float(np.mean([e['senses']['taps_to_collection'], e['senses']['comparison']['collections'], e['succession_pairwise']['collections']]))


def per_seed(R, arm, branch, f):
    return np.array([f(R[f'{arm}/seed{s}']['branches'][branch]) for s in SEEDS])


def loss_window(rec, lo, hi, k):
    return float(np.mean([u[k] for u in rec['losses'][lo - 1:hi]]))


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    R = json.load(gzip.open(ROOT / 'reports' / 'momentum-reset-v1-results.json.gz', 'rt'))['results']
    missing = [f'{a}/seed{s}' for a in ARMS for s in SEEDS if f'{a}/seed{s}' not in R]
    if missing:
        raise SystemExit(f'incomplete cohort: {missing[:3]}')
    out = {'seeds': SEEDS, 'readings': {}, 'descriptive': {}}
    for arm in ARMS:
        rd = {}
        rd['1_zero_grad_vs_0'] = {t: summarize(delta(R, arm, 'zero-grad', t)) for t in TIMES}
        zm = [R[f'{arm}/seed{s}']['branches']['zero-grad-m0'] for s in SEEDS]
        rd['2_checks'] = {
            'adam_reset_unchanged_and_zeroed': bool(all(R[f'{arm}/seed{s}']['branches'][b]['adam_check']['unchanged_and_zeroed']
                                                         for s in SEEDS for b in ('ctx-m0', 'hold-m0', 'zero-grad-m0'))),
            'n_uninitialized_states': int(max(R[f'{arm}/seed{s}']['branches'][b]['adam_check']['n_uninitialized']
                                              for s in SEEDS for b in ('ctx-m0', 'hold-m0', 'zero-grad-m0'))),
            'zero_grad_m0_weights_unchanged': bool(all(max(r['movement_final'].values()) == 0 for r in zm)),
            'zero_grad_m0_recall_constant': bool(all(p['recall'] == r['curve'][0]['recall'] for r in zm for p in r['curve']))}
        rd['3_ctx-m0_minus_ctx'] = {t: summarize(delta(R, arm, 'ctx-m0', t) - delta(R, arm, 'ctx', t)) for t in TIMES}
        ccp = {b: per_seed(R, arm, b, lambda r: context_perf(r['snapshot'])) for b in ('ctx', 'ctx-m0')}
        aa = {b: per_seed(R, arm, b, lambda r: a_of(r['snapshot'])) for b in ('ctx', 'ctx-m0')}
        gates = {'i_context_performance': summarize(ccp['ctx-m0'] - ccp['ctx']),
                 'ii_transfer_a': summarize(aa['ctx-m0'] - aa['ctx']),
                 'iii_hold-m0_minus_hold_D250': summarize(delta(R, arm, 'hold-m0', 250) - delta(R, arm, 'hold', 250)),
                 'iii_hold-m0_minus_hold_recall_end': summarize(final(R, arm, 'hold-m0') - final(R, arm, 'hold'))}
        rd['4_cost_gates'] = gates
        rd['5_remedy_candidate'] = bool(rd['3_ctx-m0_minus_ctx'][50]['outcome'] == 'better' and all(g['noninferior'] for g in gates.values()))
        out['readings'][arm] = rd
        ds = {'recall_mean': {}, 'recall_train_mean': {}, 'ctx-fresh_minus_ctx': {t: interval(delta(R, arm, 'ctx-fresh', t) - delta(R, arm, 'ctx', t)) for t in TIMES},
              'loss_diff_m0_minus_ctx': {}, 'loss_means': {}, 'context_performance_end': {b: float(v.mean()) for b, v in ccp.items()},
              'context_components_end': {}, 'a_transfer_end': {}, 'movement_final': {}}
        for b in R[f'{arm}/seed{SEEDS[0]}']['branches']:
            recs = [R[f'{arm}/seed{s}']['branches'][b] for s in SEEDS]
            offs = [p['offset'] for p in recs[0]['curve']]
            ds['recall_mean'][b] = {t: float(np.mean([point(r, t) for r in recs])) for t in offs}
            ds['recall_train_mean'][b] = {t: float(np.mean([point(r, t, 'recall_train') for r in recs])) for t in offs}
            ds['movement_final'][b] = {g: float(np.mean([r['movement_final'][g] for r in recs])) for g in recs[0]['movement_final']}
            if 'snapshot' in recs[0]:
                ds['a_transfer_end'][b] = float(np.mean([a_of(r['snapshot']) for r in recs]))
                e = [r['snapshot']['evaluate'] for r in recs]
                ds['context_components_end'][b] = {'taps_to_collection': float(np.mean([x['senses']['taps_to_collection'] for x in e])),
                                                   'comparison_collections': float(np.mean([x['senses']['comparison']['collections'] for x in e])),
                                                   'successor_collections': float(np.mean([x['succession_pairwise']['collections'] for x in e])),
                                                   'coherence_auroc_collections': float(np.mean([x['senses']['coherence_auroc_next_judgment']['collections'] for x in e]))}
            if 'losses' in recs[0]:
                ds['loss_means'][b] = {f'{lo}-{hi}/{n}': float(np.mean([loss_window(r, lo, hi, k) for r in recs]))
                                       for lo, hi in LOSS_WINDOWS for k, n in enumerate(LOSS_NAMES)}
        for b0, b1 in (('ctx', 'ctx-m0'), ('hold', 'hold-m0')):
            ds['loss_diff_m0_minus_ctx'][b1] = {f'{lo}-{hi}/{n}': interval([loss_window(R[f'{arm}/seed{s}']['branches'][b1], lo, hi, k)
                                                                           - loss_window(R[f'{arm}/seed{s}']['branches'][b0], lo, hi, k) for s in SEEDS])
                                                for lo, hi in LOSS_WINDOWS for k, n in enumerate(LOSS_NAMES)}
        out['descriptive'][arm] = ds
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'command': 'momentum_reset_analysis', 'wall_seconds': time.perf_counter() - t0,
                                   'process_cpu_seconds': time.process_time() - c0,
                                   'scope': 'nondeterministic timing; the analysis JSON holds the scientific content only',
                                   'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}, indent=1) + '\n')
    print('wrote', OUT.relative_to(ROOT))
    for arm, rd in out['readings'].items():
        print(f'== {arm}  checks {rd["2_checks"]}')
        print('  1 zero-grad vs 0', {t: (round(x['mean_ci'][0], 3), x['outcome']) for t, x in rd['1_zero_grad_vs_0'].items()})
        print('  3 ctx-m0 - ctx  ', {t: (round(x['mean_ci'][0], 3), x['outcome']) for t, x in rd['3_ctx-m0_minus_ctx'].items()})
        print('  4 gates         ', {k: (round(x['mean_ci'][0], 3), x['outcome'], 'NI' if x['noninferior'] else '') for k, x in rd['4_cost_gates'].items()})
        print('  5 remedy candidate:', rd['5_remedy_candidate'])


if __name__ == '__main__':
    main()
