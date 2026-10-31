"""Analysis of forgetting mechanism v1 under the frozen contract (docs/forgetting-mechanism-v1-contract.md, "Readings").

Per arm over ten seeds. Accuracy change D(t) = recall(t) - recall(0) on independent renderings (training renderings reported
beside); branch differences are D(branch) - D(reference), positive meaning less loss. Intervals and outcomes use the earlier
studies' rule (fan_in_pilot_analysis.classify_ci, margin 0.05) and, for "noninferior", a lower bound above -0.05.
Timing goes to a separate -analysis-runtime.json.

Usage: .venv/bin/python -m experiments.forgetting_mechanism_analysis
"""
import gzip
import json
import time

import numpy as np

from experiments import fan_in_pilot_analysis as AN
from experiments import number_systems as X

ROOT = X.ROOT
SEEDS = list(range(330, 340))
ARMS = ('sparse48', 'dense', 'mixed', 'mixed-h125')
CHANCE, MARGIN = 1 / 9, .05
TIMES = (50, 250, 2000)
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
OUT = ROOT / 'reports' / 'forgetting-mechanism-v1-analysis.json'
RUNTIME = ROOT / 'reports' / 'forgetting-mechanism-v1-analysis-runtime.json'


def summarize(d):
    d = np.asarray(d, float)
    ci = AN.interval(d)
    return {'mean_ci': ci, 'outcome': AN.classify_ci(d, 'acc'), 'noninferior': bool(ci[1] > -MARGIN)}


def point(rec, t, key):
    return next(p[key] for p in rec['curve'] if p['offset'] == t)


def delta(R, arm, branch, t, key='recall'):
    return np.array([point(R[f'{arm}/seed{s}']['branches'][branch], t, key) - point(R[f'{arm}/seed{s}']['branches'][branch], 0, key) for s in SEEDS])


def a_of(snap):
    return float(np.mean([np.mean(snap['probe'][k]) for k in TRANSFER_SETS]))


def window(curve, lo, hi):
    pts = sorted((c['offset'], c['recall']) for c in curve if lo <= c['offset'] <= hi)
    x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts]) - CHANCE
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    R = json.load(gzip.open(ROOT / 'reports' / 'forgetting-mechanism-v1-results.json.gz', 'rt'))['results']
    missing = [f'{a}/seed{s}' for a in ARMS for s in SEEDS if f'{a}/seed{s}' not in R]
    if missing:
        raise SystemExit(f'incomplete cohort: {missing[:3]}')
    out = {'seeds': SEEDS, 'readings': {}, 'descriptive': {}}
    for arm in ARMS:
        has = lambda b: b in R[f'{arm}/seed{SEEDS[0]}']['branches']
        rd = {}
        # 1. evaluation stability
        rd['1_noupdate_exact'] = bool(all(
            p[k] == R[f'{arm}/seed{s}']['branches']['noupdate']['curve'][0][k]
            for s in SEEDS for p in R[f'{arm}/seed{s}']['branches']['noupdate']['curve']
            for k in ('recall', 'recall_train', 'rebuilt/identity', 'frozen_bank/identity', 'probe/count')))
        # 2. optimizer state and decay
        rd['2_zero_grad_vs_0'] = {t: summarize(delta(R, arm, 'zero-grad', t)) for t in TIMES}
        if has('decay-only'):
            rd['2_decay_only_vs_0'] = {t: summarize(delta(R, arm, 'decay-only', t)) for t in TIMES}
            rd['2_ctx_minus_ctx_nodecay'] = {t: summarize(delta(R, arm, 'ctx', t) - delta(R, arm, 'ctx-nodecay', t)) for t in TIMES}
        # 3. exposure
        for b in ('hold', 'rehearse-10'):
            rd[f'3_{b}_minus_ctx'] = {t: summarize(delta(R, arm, b, t) - delta(R, arm, 'ctx', t)) for t in TIMES}
            rd[f'3_{b}_vs_0'] = {t: summarize(delta(R, arm, b, t)) for t in TIMES}
        # 4. which updates are needed
        for b in ('ctx-enc-frozen', 'ctx-read-frozen'):
            ref = 'ctx-nodecay' if (b == 'ctx-read-frozen' and has('ctx-nodecay')) else 'ctx'
            rd[f'4_{b}_minus_{ref}'] = {t: summarize(delta(R, arm, b, t) - delta(R, arm, ref, t)) for t in TIMES}
            rd[f'4_{b}_vs_0'] = {t: summarize(delta(R, arm, b, t)) for t in TIMES}
        # 5. savings: early window (0-100) of relearning recall above chance
        w = {n: np.array([window(R[f'{arm}/seed{s}']['relearn'][n], 0, 100) for s in SEEDS]) for n in ('original', 'reassigned', 'B')}
        rd['5_original_minus_reassigned'] = summarize(w['original'] - w['reassigned'])
        rd['5_reassigned_minus_B'] = summarize(w['reassigned'] - w['B'])
        rd['5_window_means'] = {n: float(v.mean()) for n, v in w.items()}
        # 6. recognition and representation movement (descriptive intervals, no outcome label)
        rd['6_ctx_end_minus_start'] = {k: {'mean_ci': AN.interval(delta(R, arm, 'ctx', 2000, k))}
                                       for k in ('rebuilt/identity', 'frozen_bank/identity', 'rebuilt/count', 'frozen_bank/count', 'probe/count')}
        out['readings'][arm] = rd
        # descriptive
        ds = {'recall_mean': {}, 'recall_train_mean': {}, 'loss_fraction': {}, 'movement_final': {}, 'a_transfer_end': {}, 'swap': {},
              'decoders_end': {}}
        for b in R[f'{arm}/seed{SEEDS[0]}']['branches']:
            recs = [R[f'{arm}/seed{s}']['branches'][b] for s in SEEDS]
            offs = [p['offset'] for p in recs[0]['curve']]
            ds['recall_mean'][b] = {t: float(np.mean([point(r, t, 'recall') for r in recs])) for t in offs}
            ds['recall_train_mean'][b] = {t: float(np.mean([point(r, t, 'recall_train') for r in recs])) for t in offs}
            fr = {}
            for t in TIMES:
                v = [1 - (point(r, t, 'recall') - CHANCE) / (point(r, 0, 'recall') - CHANCE) for r in recs if point(r, 0, 'recall') - CHANCE > .10]
                fr[t] = {'n_defined': len(v), 'mean': float(np.mean(v)) if v else None}
            ds['loss_fraction'][b] = fr
            ds['movement_final'][b] = {g: float(np.mean([r['movement_final'][g] for r in recs])) for g in recs[0]['movement_final']}
            if 'snapshot' in recs[0]:
                ds['a_transfer_end'][b] = float(np.mean([a_of(r['snapshot']) for r in recs]))
            ds['decoders_end'][b] = {k: float(np.mean([point(r, offs[-1], k) for r in recs])) for k in ('rebuilt/identity', 'frozen_bank/identity', 'rebuilt/count', 'frozen_bank/count', 'probe/count')}
        ds['a_transfer_end']['b4'] = float(np.mean([a_of(R[f'{arm}/seed{s}']['block4']['snapshot']) for s in SEEDS]))
        for k in ('swap/current_enc_b4_readers', 'swap/b4_enc_current_readers'):
            ds['swap'][k] = {t: float(np.mean([point(R[f'{arm}/seed{s}']['branches']['ctx'], t, k) for s in SEEDS])) for t in (50, 250, 2000)}
        out['descriptive'][arm] = ds
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'command': 'forgetting_mechanism_analysis', 'wall_seconds': time.perf_counter() - t0,
                                   'process_cpu_seconds': time.process_time() - c0,
                                   'scope': 'nondeterministic timing; the analysis JSON holds the scientific content only',
                                   'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}, indent=1) + '\n')
    print('wrote', OUT.relative_to(ROOT))
    for arm, rd in out['readings'].items():
        print(f'== {arm}  noupdate exact: {rd["1_noupdate_exact"]}')
        for k, v in rd.items():
            if k.startswith(('2', '3', '4')) and isinstance(v, dict) and 250 in map(int, v):
                print(' ', k, {t: (round(x['mean_ci'][0], 3), x['outcome'], 'NI' if x['noninferior'] else '') for t, x in v.items()})
            elif k.startswith('5_') and 'mean_ci' in v:
                print(' ', k, round(v['mean_ci'][0], 3), v['outcome'])


if __name__ == '__main__':
    main()
