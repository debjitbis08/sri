"""Analysis of arbitrary pairings v1 under the frozen contract (docs/arbitrary-pairings-v1-contract.md, "Readings").

Option A (primary architecture comparison): the contrasts `mixed` - `sparse48`, `mixed-h125` - `mixed`, `mixed-h125` - `mixed-reset`
on acquisition, retained recall, interference and structure; everything else descriptive. Option B (disruption test): per arm the
paired difference B - A. Intervals and outcomes use the earlier pilots' rule (fan_in_pilot_analysis.classify_ci, margin 0.05) and,
for "noninferior", a lower bound above -0.05. Both options and all ten seeds of every arm are required; with NAP_DESCRIPTIVE=1 an
incomplete cohort gives per-arm means only, in a separate labeled file, with no intervals or outcomes.

Usage: .venv/bin/python -m experiments.arbitrary_pairings_analysis      (NFI_TAG, NAP_SEEDS and NAP_N/NAP_L as for the runner)
Timing goes to a separate -analysis-runtime.json.
"""
import gzip
import json
import os
import time

import numpy as np

from experiments import fan_in_pilot_analysis as AN
from experiments import number_systems as X

ROOT = X.ROOT
N, L = int(os.environ.get('NAP_N', '45')), int(os.environ.get('NAP_L', '2000'))
TAG = os.environ.get('NFI_TAG', '')
SEEDS = [int(s) for s in os.environ.get('NAP_SEEDS', ','.join(map(str, range(320, 330)))).split(',')]
ARMS = ('sparse48', 'dense', 'mixed', 'mixed-h125', 'mixed-h500', 'mixed-reset')
DESCRIPTIVE = os.environ.get('NAP_DESCRIPTIVE') == '1'
CHANCE, MARGIN = 1 / 9, .05
EARLY = 250
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
OUT = ROOT / 'reports' / f'arbitrary-pairings-v1-analysis{TAG}{"-descriptive" if DESCRIPTIVE else ""}.json'
RUNTIME = ROOT / 'reports' / f'arbitrary-pairings-v1-analysis{TAG}{"-descriptive" if DESCRIPTIVE else ""}-runtime.json'
CONTRASTS = (('mixed', 'sparse48'), ('mixed-h125', 'mixed'), ('mixed-h125', 'mixed-reset'))


def load(option):
    name = f'arbitrary-pairings-v1-pairs-{option}-N{N}-L{L}{TAG}'
    return json.load(gzip.open(ROOT / 'reports' / f'{name}-results.json.gz', 'rt'))['results']


def a_of(snap):
    """(a): order accuracy of the collection-fitted count map, averaged over the 16 coordinates within taps, Chinese forms and
    Devanagari forms, then over the three sets (as in the limited-memory study)."""
    return float(np.mean([np.mean(snap['probe'][k]) for k in TRANSFER_SETS]))


def window(curve, key, lo, hi):
    pts = sorted((c['offset'], c[key]) for c in curve if lo <= c['offset'] <= hi)
    x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts]) - CHANCE
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def measures(res):
    br = res['branches']
    b4, ab, pa, c = br['block4_A'], br['P-AB'], br['P-A'], br['C']
    out = {'acq_early': window(b4['curve'], 'A', 0, EARLY), 'acq_block': window(b4['curve'], 'A', 0, L),
           'acq_early_B': window(ab['curve'], 'B', 0, EARLY),
           'recall_A_b4': b4['pairs']['A']['recall'], 'recall_A_b4_train': b4['pairs']['A']['recall_on_training_renderings'],
           'recall_A_b5': ab['pairs']['A']['recall'], 'recall_A_b5_train': ab['pairs']['A']['recall_on_training_renderings'],
           'recall_A_b5_noB': pa['pairs']['A']['recall'], 'recall_B_b5': ab['pairs']['B']['recall'],
           'interference': pa['pairs']['A']['recall'] - ab['pairs']['A']['recall'],
           'interference_train': pa['pairs']['A']['recall_on_training_renderings'] - ab['pairs']['A']['recall_on_training_renderings'],
           'a_b3': a_of(res['parent']), 'a_b4': a_of(b4['snapshot']), 'a_b5': a_of(ab['snapshot']), 'a_b5_C': a_of(c['snapshot'])}
    out['a_damage'] = out['a_b5'] - out['a_b5_C']
    t5, tc = ab['snapshot']['four_way']['full']['transfer8'], c['snapshot']['four_way']['full']['transfer8']
    for k in t5:
        out[f'transfer8_b5/{k}'] = t5[k]
        out[f'transfer8_damage/{k}'] = t5[k] - tc[k]
    out['rescored_shuffled'] = b4['pairs']['A']['rescored_shuffled_mean']
    out['identity_nn_b4'] = b4['pairs']['A']['lookup']['identity']
    out['lookup_count_b4'] = b4['pairs']['A']['lookup']['count']
    if "P-A'" in br:
        out['learning_check_A_minus_Aprime'] = b4['pairs']['A']['recall'] - br["P-A'"]['pairs']["A'"]['recall']
    for part in ('mem_silenced', 'struct_silenced', 'both_silenced'):
        out[f'recall_A_b4_{part}'] = b4['pairs']['A'][part]
    return out


def summarize(d):
    d = np.asarray(d, float)
    out = {'mean_ci': AN.interval(d), 'outcome': AN.classify_ci(d, 'acc')}
    out['noninferior'] = bool(out['mean_ci'][1] > -MARGIN)
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    runs = {o: load(o) for o in 'AB'}
    missing = [f'{o}/{a}/seed{s}' for o in 'AB' for a in ARMS for s in SEEDS if f'{a}/seed{s}' not in runs[o]]
    if missing and not DESCRIPTIVE:
        raise SystemExit(f'incomplete cohort: {len(missing)} of {2 * len(ARMS) * len(SEEDS)} runs missing (first: {missing[:3]}). '
                         'Rerun the study, or set NAP_DESCRIPTIVE=1 for a labeled descriptive summary.')
    M = {o: {k: measures(v) for k, v in runs[o].items() if int(k.split('seed')[1]) in SEEDS} for o in 'AB'}
    names = list(next(iter(M['A'].values())))
    out = {'status': 'descriptive: incomplete cohort' if missing else 'complete cohort', 'seeds': SEEDS, 'N': N, 'L': L, 'missing_runs': missing,
           'means': {o: {a: {m: float(np.mean([M[o][f'{a}/seed{s}'][m] for s in SEEDS if f'{a}/seed{s}' in M[o] and m in M[o][f'{a}/seed{s}']]))
                             for m in names if any(m in M[o].get(f'{a}/seed{s}', {}) for s in SEEDS)} for a in ARMS
                         if any(f'{a}/seed{s}' in M[o] for s in SEEDS)} for o in 'AB'}}
    if not DESCRIPTIVE:
        contrast_measures = [m for m in names if not m.startswith(('recall_A_b4_mem', 'recall_A_b4_struct', 'recall_A_b4_both', 'learning_check'))
                             and m not in ('rescored_shuffled', 'identity_nn_b4', 'lookup_count_b4')]
        out['option_A_contrasts'] = {f'{x} - {y}': {m: summarize([M['A'][f'{x}/seed{s}'][m] - M['A'][f'{y}/seed{s}'][m] for s in SEEDS])
                                                    for m in contrast_measures} for x, y in CONTRASTS}
        disrupt_measures = ('recall_A_b4', 'recall_A_b4_train', 'recall_A_b5', 'recall_A_b5_train', 'acq_early', 'acq_block', 'a_b5', 'a_damage')
        out['option_B_minus_A'] = {a: {m: summarize([M['B'][f'{a}/seed{s}'][m] - M['A'][f'{a}/seed{s}'][m] for s in SEEDS]) for m in disrupt_measures}
                                   for a in ARMS}
        out['dense_B_equals_A'] = bool(all(abs(M['B'][f'dense/seed{s}'][m] - M['A'][f'dense/seed{s}'][m]) == 0
                                           for s in SEEDS for m in disrupt_measures))
        out['sanity'] = {o: {'rescored_shuffled_mean_over_arms_and_seeds': float(np.mean([v['rescored_shuffled'] for v in M[o].values()])),
                             'learning_check_A_minus_Aprime': {a: summarize([M[o][f'{a}/seed{s}']['learning_check_A_minus_Aprime'] for s in SEEDS])
                                                               for a in ('sparse48', 'mixed')}} for o in 'AB'}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'command': 'arbitrary_pairings_analysis', 'wall_seconds': time.perf_counter() - t0,
                                   'process_cpu_seconds': time.process_time() - c0, 'descriptive': DESCRIPTIVE,
                                   'scope': 'nondeterministic timing; the analysis JSON holds the scientific content only',
                                   'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}, indent=1) + '\n')
    print('wrote', OUT.relative_to(ROOT), '|', out['status'])
    for o in 'AB':
        print(f'option {o}:')
        for a, m in out['means'][o].items():
            print(f"  {a:12s} acq_early {m['acq_early']:+.3f}  recall A b4 {m['recall_A_b4']:.2f} (train {m['recall_A_b4_train']:.2f})  "
                  f"b5 {m['recall_A_b5']:.2f}  interference {m['interference']:+.2f}  a_b5 {m['a_b5']:.3f}  a_damage {m['a_damage']:+.3f}")
    if not DESCRIPTIVE:
        for k, v in out['option_A_contrasts'].items():
            print(k, {m: (v[m]['outcome'], round(v[m]['mean_ci'][0], 3)) for m in ('acq_early', 'recall_A_b4', 'recall_A_b4_train', 'recall_A_b5', 'interference', 'a_b5')})
        print('dense B == A:', out['dense_B_equals_A'])


if __name__ == '__main__':
    main()
