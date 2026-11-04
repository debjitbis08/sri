"""Tables for experiments.relearning_dynamics (docs/relearning-dynamics-v1.md).
Usage: python -m experiments.relearning_dynamics_summary reports/relearning-dynamics-v1.json[.gz]
"""
import sys

import numpy as np

from experiments import relearning_branches_summary as RBS
from experiments import sense_core_summary as SS

GROUPS = ('image projection', 'core projection', 'comparator g', 'taps encoder', 'predictor P', 'relation tokens', 'scale')
LOSSES = ('match', 'next', 'compare', 'sequence')


def cell(xs):
    return SS.cell(xs)


def main(path):
    d = SS.load(path)
    R = d['results']
    seeds = sorted(R)
    print('reproduction:', {s: (R[s]['delay_state_max_abs_difference_from_step8'], R[s]['as-run-1000']['update200_state_max_abs_difference_from_step8_relearn']) for s in seeds})

    print('\n## first updates: total gradient norm, clipping factor, and update / parameter norm per group (mean (range) over seeds)')
    for b in ('as-run-1000', 'no-forms-20', 'no-form-seq', 'no-form-match'):
        L = {s: R[s][b]['log_first_updates'] for s in seeds}
        print(f'\n### {b}')
        for t in (0, 1, 2, 5, 10, 19):
            row = [f'grad {cell([L[s][t]["total_grad_norm"] for s in seeds])}', f'clip {cell([L[s][t]["clip_factor"] for s in seeds])}']
            row += [f'{g[:10]} {cell([L[s][t]["groups"][g]["update_over_param"] for s in seeds])}' for g in GROUPS[:4]]
            print(f'update {t:2d}: ' + ' | '.join(row))
        cum = {g: [sum(L[s][t]['groups'][g]['update_over_param'] for t in range(20)) for s in seeds] for g in GROUPS}
        print('sum over updates 0-19 of update/param: ' + ' | '.join(f'{g} {cell(v)}' for g, v in cum.items()))

    print('\n## per-loss gradient norms by group, as-run, mean over updates 0-19 (before clipping)')
    for g in GROUPS:
        print(f'{g:16s} ' + ' '.join(f'{n} {cell([np.mean([R[s]["as-run-1000"]["log_first_updates"][t]["groups"][g][f"grad_{n}"] for t in range(20)]) for s in seeds])}' for n in LOSSES))
    print('\n## per-loss gradient norms by group, update 0 only, as-run and no-forms reference')
    for b in ('as-run-1000', 'no-forms-20'):
        for g in GROUPS[:3]:
            print(f'{b:12s} {g:16s} ' + ' '.join(f'{n} {cell([R[s][b]["log_first_updates"][0]["groups"][g][f"grad_{n}"] for s in seeds])}' for n in LOSSES))
    print('\n## cosines between losses\' gradients (all parameters, and match|sequence within groups), as-run')
    keys = list(R[seeds[0]]['as-run-1000']['log_first_updates'][0]['cosines'])
    for k in keys:
        print(f'{k:40s} ' + ' '.join(f'u{t}: {cell([R[s]["as-run-1000"]["log_first_updates"][t]["cosines"][k] for s in seeds])}' for t in (0, 2, 5, 10, 19)))

    def track(b, f, t):
        return [f(R[s][b]['track'][str(t)]) for s in seeds]

    print('\n## loss ablations against as-run (first 200 updates)')
    for label, f in RBS.MEASURES:
        print(f'\n### {label}')
        for b in ('as-run-1000', 'no-form-seq', 'no-form-match'):
            print(f'{b:14s} ' + ' '.join(f'{t}: {cell(track(b, f, t))}' for t in (0, 10, 20, 50, 100, 200)))

    print('\n## the long as-run continuation: recovery against updates and against English reacquisition')
    pts = (0, 20, 50, 100, 200, 300, 400, 500, 700, 1000)
    for label, f in RBS.MEASURES:
        if label.startswith(('collections ->', 'taps ->', 'English', 'memory: A symbol', 'same-count', 'collection class', 'symmetry: same', 'AUROC g(c,t)')):
            print(f'{label:46s} ' + ' '.join(f'{t}: {cell(track("as-run-1000", f, t))}' for t in pts))
    eng = lambda r: r['english_matching']
    ct = lambda r: r['collections_taps']['collections_to_taps']['fresh']
    print('\nper seed, (English matching, collections -> taps) at ' + ', '.join(map(str, pts)))
    for s in seeds:
        print(s, ' '.join(f'({eng(R[s]["as-run-1000"]["track"][str(t)]):.2f}, {ct(R[s]["as-run-1000"]["track"][str(t)]):.2f})' for t in pts))


if __name__ == '__main__':
    main(*sys.argv[1:])
