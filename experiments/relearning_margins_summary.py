"""Tables for experiments.relearning_margins (docs/relearning-margins-v1.md).
Checks (exact split of each update, approximation error against the actual changes), then for each read and window (updates 0-9, 0-19): accuracy,
the lost decisions' mean actual and attributed change of the correct score, the max wrong score and the margin, by parameter group and by loss path,
against decisions that stay correct. Means over seeds (range).
Usage: python -m experiments.relearning_margins_summary reports/relearning-margins-v1.json[.gz] [path|first]
       python -m experiments.relearning_margins_summary reports/relearning-margins-v1.json[.gz] counterfactual
       python -m experiments.relearning_margins_summary reports/relearning-margins-stop-v1.json[.gz] window   (updates 10-19, by decision fate)
The second form needs the per-query attributions in runs/relearning-margins-v1/margins-seed*.npz: accuracy at updates 10 and 20 with a loss path's
attributed (path-integral) margin change subtracted, holding every other part of each actual step fixed (a linear counterfactual of removing that
part of the displacement, with clipping and Adam's normalization unchanged).
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

READS = ('c->t fresh', 'c->t stored', 't->c', 'seq collections up 1', 'seq collections down 1', 'seq taps up 1', 'comp collections up 2',
         'comp collections up 3', 'comp taps up 2', 'memory A symbol->taps', 'memory B symbol->taps')


ORDER = ('c->t fresh', 'c->t stored', 't->c', 'seq collections up 1', 'seq collections down 1', 'seq taps up 1', 'comp collections up 2',
         'comp collections up 3', 'comp taps up 2', 'memory A symbol->taps', 'memory B symbol->taps')  # the layout order of relearning_margins.outputs


def counterfactual(path):
    from experiments import relearning_margins as RMg
    d = SS.load(path)
    R, paths = d['results'], d['paths']
    for remove in (['seq form via candidates'], ['seq form via prediction'], ['seq form via candidates', 'seq form via prediction'], ['match form']):
        print('without', ' + '.join(remove))
        ki = [paths.index(k) for k in remove]
        for name in ORDER:
            rows = []
            for s in sorted(R):
                z = np.load(RMg.RUNS / f'margins-{s}.npz')
                traj, A = z['traj'], z['path']
                pos = sum(2 * R[s]['reads'][nm]['queries'] for nm in ORDER[:ORDER.index(name)])
                n = R[s]['reads'][name]['queries']
                m = traj[:, pos:pos + n] - traj[:, pos + n:pos + 2 * n]
                ok = np.array(R[s]['reads'][name].get('retrieval_ok', [True] * n))  # memory reads: abstentions count wrong
                assert abs(((m[0] > 0) & ok).mean() - R[s]['reads'][name]['accuracy'][0]) < 1e-9
                cm = np.cumsum(A[:, ki][..., pos:pos + n].sum((1, 2)) - A[:, ki][..., pos + n:pos + 2 * n].sum((1, 2)), 0)
                acc = lambda x: ((x > 0) & ok).mean()
                rows.append([acc(m[0]), acc(m[10]), acc(m[10] - cm[9]), acc(m[20]), acc(m[20] - cm[19])])
            r = np.array(rows).mean(0)
            print(f'  {name:24s} at 0 {r[0]:.2f} | at 10 actual {r[1]:.2f}, without {r[2]:.2f} | at 20 actual {r[3]:.2f}, without {r[4]:.2f}')


def window(path, key='10-20'):
    """Updates 10-19: decisions that become wrong (correct at 10, wrong at 20), decisions already wrong at 10 (correct at 0), by whether they
    recover, and decisions that stay correct; their mean attributed change of the correct score, max wrong score and margin, by group and path."""
    d = SS.load(path)
    R = d['results']
    seeds = sorted(R)
    paths, groups = d['paths'], R[seeds[0]]['groups']
    print('trajectory:', d.get('arm', 'unchanged'))
    for name in READS:
        acc = np.array([R[s]['reads'][name]['accuracy'] for s in seeds])
        print(f'\n######## {name}: accuracy at 0, 10, 20:', ' '.join(f'{acc[:, t].mean():.2f}' for t in (0, 10, 20)))
        for sname in ('lost', 'already_wrong_still_wrong', 'already_wrong_recovered', 'kept'):
            cs = [R[s]['reads'][name]['cumulative'][key]['path'].get(sname) for s in seeds]
            cs = [c for c in cs if c]
            if not cs:
                continue
            print(f'  {sname}: n {[c["n"] for c in cs]}')
            for q in ('correct', 'max_wrong', 'margin'):
                act = [c[q]['actual_mean'] for c in cs]; est = [c[q]['estimated_mean'] for c in cs]; err = [c[q]['relative_error'] for c in cs]
                B = np.array([c[q]['by_path_and_group'] for c in cs])
                byg, byp = B.sum(1).mean(0), B.sum(2).mean(0)
                print(f'    {q:9s} actual {np.mean(act):+8.2f} est {np.mean(est):+8.2f} (rel err {np.mean(err):.2f})')
                print('      by group: ' + ', '.join(f'{g} {v:+.2f}' for g, v in zip(groups, byg)))
                print('      by path:  ' + ', '.join(f'{p} {v:+.2f}' for p, v in zip(paths, byp)))
                if q != 'max_wrong':
                    M = B.mean(0)
                    top = np.dstack(np.unravel_index(np.argsort(M.ravel()), M.shape))[0][:5]
                    print('      most negative cells: ' + ', '.join(f'{paths[i]} x {groups[j]} {M[i, j]:+.2f}' for i, j in top))


def main(path, label='path'):
    if label == 'counterfactual':
        return counterfactual(path)
    if label == 'window':
        return window(path)
    d = SS.load(path)
    R = d['results']
    seeds = sorted(R)
    paths, groups = d['paths'], R[seeds[0]]['groups']
    print('delay reproduction:', {s: R[s]['delay_state_max_abs_difference_from_step8'] for s in seeds})
    ck = [r for s in seeds for r in R[s]['per_update_checks']]
    print('max over updates: gradient split error', f'{max(r["gradient_sum_relative_error"] for r in ck):.1e}',
          '| displacement reconstruction error', f'{max(r["displacement_reconstruction_relative_error"] for r in ck):.1e}',
          '| loss split value error', f'{max(abs(r["component_loss_value_minus_total"]) for r in ck):.1e}')
    print('\n## per-update relative error of the estimated change, median (max) over the 20 updates and seeds: first order | path integral')
    for name in READS:
        cells = []
        for q in ('correct', 'max_wrong', 'margin'):
            for lab in ('first', 'path'):
                v = [r['relative_error'][lab][name][q] for r in ck]
                cells.append(f'{np.median(v):.3f} ({max(v):.2f})')
        print(f'{name:24s} correct {cells[0]} | {cells[1]}   wrong {cells[2]} | {cells[3]}   margin {cells[4]} | {cells[5]}')
    for name in READS:
        acc = np.array([R[s]['reads'][name]['accuracy'] for s in seeds])
        print(f'\n######## {name}: accuracy at 0, 5, 10, 15, 20:', ' '.join(f'{acc[:, t].mean():.2f}' for t in (0, 5, 10, 15, 20)))
        for T_ in ('10', '20'):
            print(f'-- updates 0-{int(T_) - 1} ({label})')
            for sname in ('lost', 'kept'):
                cs = [R[s]['reads'][name]['cumulative'][T_][label].get(sname) for s in seeds]
                cs = [c for c in cs if c]
                if not cs:
                    continue
                n = [c['n'] for c in cs]
                print(f'  {sname}: n {n}')
                for q in ('correct', 'max_wrong', 'margin'):
                    act = [c[q]['actual_mean'] for c in cs]; est = [c[q]['estimated_mean'] for c in cs]; err = [c[q]['relative_error'] for c in cs]
                    B = np.array([c[q]['by_path_and_group'] for c in cs])  # (seeds, paths, groups)
                    byg = B.sum(1).mean(0); byp = B.sum(2).mean(0)
                    print(f'    {q:9s} actual {np.mean(act):+8.2f} est {np.mean(est):+8.2f} (rel err {np.mean(err):.2f})')
                    print('      by group: ' + ', '.join(f'{g} {v:+.2f}' for g, v in zip(groups, byg) if abs(v) >= 0.005 * max(1e-9, np.abs(byg).max())))
                    print('      by path:  ' + ', '.join(f'{p} {v:+.2f}' for p, v in zip(paths, byp)))
                    if sname == 'lost' and q == 'margin':
                        top = np.dstack(np.unravel_index(np.argsort(B.mean(0).ravel()), B.shape[1:]))[0][:6]
                        print('      most negative cells: ' + ', '.join(f'{paths[i]} x {groups[j]} {B.mean(0)[i, j]:+.2f}' for i, j in top))


if __name__ == '__main__':
    main(*sys.argv[1:])
