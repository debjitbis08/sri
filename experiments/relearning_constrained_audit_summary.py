"""Tables and the declared gate for experiments.relearning_constrained_audit (docs/relearning-constrained-audit-v1.md).
Usage: python -m experiments.relearning_constrained_audit_summary reports/relearning-constrained-audit-v1.json[.gz]
       python -m experiments.relearning_constrained_audit_summary reports/relearning-constrained-continuation-v1.json[.gz] reports/relearning-remedies-a-v1.json.gz
The second form: worst transient and level at 200 against update 0, mean (range) over seeds, beside stage A's unchanged and seq-cand-stop arms.
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

FO = ('sequence', 'english_matching', 'matching', 'total')


def continuation(path, stage_a):
    from experiments import relearning_remedies_summary as RS
    R, A = SS.load(path)['results'], SS.load(stage_a)['results']
    seeds = sorted(R)
    src = {'unchanged': lambda s: A[s]['unchanged']['track'], 'seq-cand-stop': lambda s: A[s]['seq-cand-stop']['track'],
           'constrained': lambda s: R[s]['corrected']['track']}
    extra = (('AUROC -distance(c,t)', lambda r: r['relation']['discrimination_auroc']['-distance(c,t)']),
             ('AUROC g(c,t)', lambda r: r['relation']['discrimination_auroc']['g(c,t)']),
             ('AUROC g(t,c)', lambda r: r['relation']['discrimination_auroc']['g(t,c)']),
             ('symmetry correlation', lambda r: r['relation']['symmetry']['correlation']))
    for label, f in (*RS.MEASURES, *extra):
        print(f'\n## {label}')
        for a, g in src.items():
            v = {s: [f(g(s)[str(t)]) for t in range(0, 201, 10)] for s in seeds}
            worst = [min(v[s]) - v[s][0] for s in seeds]; end = [v[s][-1] - v[s][0] for s in seeds]
            traj = ' '.join(f'{np.mean([v[s][t // 10] for s in seeds]):.2f}' for t in (10, 20, 50, 100, 200))
            print(f'{a:14s} {SS.cell([v[s][0] for s in seeds]):>18s} {RS.signed(worst):>24s} {RS.signed(end):>24s}   {traj}')
    print('\n## constrained path: retained norm, mismatch constraint active, realized batch dlogS and dE (means over seeds) by window')
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 200)):
        rows = [r for s in seeds for r in R[s]['corrected']['per_update'][lo:hi]]
        print(f'{lo:3d}-{hi:3d}: kept {np.median([r["retained"] for r in rows]):.3f}, active {np.mean([r["constraints"]["mismatch_constraint_active"] for r in rows]):.2f}, '
              f'|dlogS| {np.mean([abs(r["finite"]["delta"]["1.0"]["log_S"]) for r in rows]):.2e}, dE {np.mean([r["finite"]["delta"]["1.0"]["E"] for r in rows]):+.3f}, '
              f'seq-loss first-order kept {sum(r["first_order"]["delta"]["sequence"] for r in rows) / sum(r["first_order"]["delta0"]["sequence"] for r in rows):.2f}')
    print('held-out log collection spread at 0/50/100/200:', ' '.join(f'{np.mean([R[s]["corrected"]["held_out"][str(t)]["log_collection_spread"] for s in seeds]):.2f}' for t in (0, 50, 100, 200)))
    print('sequence loss (as trained) at 0, 10, 50, 199:', ' '.join(f'{np.mean([R[s]["corrected"]["per_update"][t]["losses"]["sequence"] for s in seeds]):.3f}' for t in (0, 10, 50, 199)))


def main(path, stage_a=None):
    if stage_a:
        return continuation(path, stage_a)
    d = SS.load(path)
    R = d['results']
    seeds = sorted(R)
    print('delay reproduction:', {s: R[s]['delay_state_max_abs_difference_from_step8'] for s in seeds}, 'parameters:', R[seeds[0]]['parameters'])
    gate = {}
    for s in seeds:
        A, C = R[s]['audit']['per_update'], R[s]['corrected']['per_update']
        print(f'\n## {s}: audit path, per update')
        print(f'{"t":>2s} {"|d0|":>9s} {"kept":>5s} {"cos":>5s} {"cosab":>6s} {"E-act":>5s} {"a.d0":>9s} {"b.d0":>9s}'
              + ''.join(f' {k[:8]+" d0/d":>17s}' for k in FO) + f' {"dlogS fin d0/d":>17s} {"dE fin d0/d":>17s}')
        for t, r in enumerate(A):
            f0, f1 = r['first_order']['delta0'], r['first_order']['delta']
            fin0, fin1 = r['finite']['delta0']['1.0'], r['finite']['delta']['1.0']
            print(f'{t:2d} {r["delta0_norm"]:9.2e} {r["retained"]:5.2f} {r["cos_delta_delta0"]:5.2f} {r["constraints"]["cos_a_b"]:6.2f} '
                  f'{"y" if r["constraints"]["mismatch_constraint_active"] else "n":>5s} {f0["a"]:9.2e} {f0["b"]:9.2e}'
                  + ''.join(f' {f0[k]:8.1e}/{f1[k]:8.1e}' for k in FO)
                  + f' {fin0["log_S"]:8.1e}/{fin1["log_S"]:8.1e} {fin0["E"]:8.1e}/{fin1["E"]:8.1e}')
        kept = np.median([r['retained'] for r in A])
        g2 = {}
        for k in ('sequence', 'english_matching'):
            p0 = sum(r['first_order']['delta0'][k] for r in A); p1 = sum(r['first_order']['delta'][k] for r in A)
            g2[k] = (p0, p1, (p1 / p0 if p0 < 0 else None))
        sumS = lambda rows: sum(abs(r['finite']['delta0' if rows is A else 'delta']['1.0']['log_S']) for r in rows)
        sumE = lambda rows: sum(r['finite']['delta0' if rows is A else 'delta']['1.0']['E'] for r in rows)
        hA, hC = R[s]['audit']['held_out'], R[s]['corrected']['held_out']
        fall = lambda h: h['0']['log_collection_spread'] - h['20']['log_collection_spread']
        G1 = kept >= 0.5
        G2 = all(v[2] is None or v[2] >= 0.5 for v in g2.values())
        G3 = sumS(C) <= 0.25 * sumS(A) and sumE(C) <= max(0., 0.25 * sumE(A)) and fall(hC) <= 0.25 * max(fall(hA), 0.)
        gate[s] = (G1, G2, G3)
        print(f'G1 median kept {kept:.3f} -> {G1}')
        print('G2 summed first-order change (proposal, corrected, ratio):', {k: tuple(None if x is None else round(x, 4) for x in v) for k, v in g2.items()}, '->', G2)
        print(f'G3 sum|dlogS| audit {sumS(A):.4f} corrected {sumS(C):.4f}; sum dE audit {sumE(A):.3f} corrected {sumE(C):.3f}; '
              f'held-out log-spread fall audit {fall(hA):.3f} corrected {fall(hC):.3f} -> {G3}')
        print('held out at 0/10/20 (audit | corrected):')
        for k in ('log_collection_spread', 'collection_taps_mismatch', 'log_taps_spread'):
            print(f'  {k:26s}', ' '.join(f'{hA[str(t)][k]:7.3f}' for t in (0, 10, 20)), '|', ' '.join(f'{hC[str(t)][k]:7.3f}' for t in (0, 10, 20)))
        print('corrected path, active mismatch constraint in', sum(r['constraints']['mismatch_constraint_active'] for r in C), 'of 20; median kept',
              f'{np.median([r["retained"] for r in C]):.3f}')
    print('\n## corrected path reads, mean (range) over seeds')
    T = lambda s, t: R[s]['corrected']['track'][str(t)]
    for label, f in (('collections -> taps, fresh', lambda r: r['collections_taps']['collections_to_taps']['fresh']),
                     ('collections -> taps, stored', lambda r: r['collections_taps']['collections_to_taps']['stored']),
                     ('taps -> collections', lambda r: r['collections_taps']['taps_to_collections']),
                     ('memory A symbol -> taps', lambda r: r['memory_present']['A']['end_to_end']['r3']),
                     ('English matching', lambda r: r['english_matching']),
                     ('sequence collections up 1', lambda r: r['composition']['collections/up/1/distance']),
                     ('composition collections up 2', lambda r: r['composition']['collections/up/2/distance']),
                     ('spacing', lambda r: r['relation']['geometry']['collection_class_spacing']),
                     ('same-count c-t / spacing', lambda r: r['relation']['geometry']['same_count_c_t'])):
        print(f'{label:30s}', ' '.join(f'{t}: {SS.cell([f(T(s, t)) for s in seeds])}' for t in (0, 10, 20)))
    print('\nseconds per update (mean): extra gradients, correction, finite checks:',
          [round(float(np.mean([r['seconds'][k] for s in seeds for r in R[s]['audit']['per_update']])), 4) for k in ('extra_gradients', 'correction', 'finite_checks')])
    print('\nGATE (G1, G2, G3):', gate, '-> continuation warranted' if all(all(v) for v in gate.values()) else '-> not warranted')


if __name__ == '__main__':
    main(*sys.argv[1:])
