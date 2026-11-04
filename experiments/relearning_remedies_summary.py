"""Tables for experiments.relearning_remedies (docs/relearning-remedies-plan.md, stage A).
For each measure and arm: the value at update 0; the worst transient (minimum over updates 0-200) and the level at the deadline (update 200), each as a
change from update 0. Mean (range) over seeds.
Usage: python -m experiments.relearning_remedies_summary reports/relearning-remedies-a-v1.json[.gz]
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

C = lambda key: (lambda r: r['composition'][key])
MEASURES = (
    ('collections -> taps, fresh', lambda r: r['collections_taps']['collections_to_taps']['fresh']),
    ('collections -> taps, stored values', lambda r: r['collections_taps']['collections_to_taps']['stored']),
    ('collections -> taps, trained order', lambda r: r['collections_taps']['collections_to_taps_swapped_order']['fresh']),
    ('taps -> collections', lambda r: r['collections_taps']['taps_to_collections']),
    ('memory A: symbol -> taps', lambda r: r['memory_present']['A']['end_to_end']['r3']),
    ('memory B: symbol -> taps', lambda r: r['memory_present']['B']['end_to_end']['r3']),
    ('memory A: taps -> symbol', lambda r: r['memory_present']['A']['end_to_end']['r3_taps_to_symbol']),
    ('English matching', lambda r: r['english_matching']),
    ('sequence: collections, up 1, distance', C('collections/up/1/distance')),
    ('sequence: collections, down 1, distance', C('collections/down/1/distance')),
    ('sequence: taps, up 1, distance', C('taps/up/1/distance')),
    ('sequence: English forms, up 1, distance', C('forms/english/up/1/distance')),
    ('sequence: Devanagari forms, up 1, distance', C('forms/devanagari_digits/up/1/distance')),
    ('composition: collections, up 2, distance', C('collections/up/2/distance')),
    ('composition: collections, up 3, distance', C('collections/up/3/distance')),
    ('composition: collections, up 1, same', C('collections/up/1/same')),
    ('composition: taps, up 2, distance', C('taps/up/2/distance')),
    ('collection class spacing', lambda r: r['relation']['geometry']['collection_class_spacing']),
    ('same-count c-t distance / spacing', lambda r: r['relation']['geometry']['same_count_c_t']),
)


def main(path):
    d = SS.load(path)
    R, arms = d['results'], d['arms']
    seeds = sorted(R)
    print('delay reproduction:', {s: R[s]['delay_state_max_abs_difference_from_step8'] for s in seeds})
    print('s0:', {s: round(R[s]['seq-relative']['s0'], 1) for s in seeds})
    steps = sorted(int(t) for t in R[seeds[0]][arms[0]]['track'])
    for label, f in MEASURES:
        print(f'\n## {label}')
        print(f'{"arm":16s} {"update 0":>18s} {"worst transient":>24s} {"at 200":>24s}   trajectory (mean) at 10, 20, 50, 100, 200')
        for a in arms:
            v = {s: [f(R[s][a]['track'][str(t)]) for t in steps] for s in seeds}
            v0 = [v[s][0] for s in seeds]
            worst = [min(v[s]) - v[s][0] for s in seeds]
            end = [v[s][-1] - v[s][0] for s in seeds]
            traj = ' '.join(f'{np.mean([v[s][steps.index(t)] for s in seeds]):.2f}' for t in (10, 20, 50, 100, 200))
            print(f'{a:16s} {SS.cell(v0):>18s} {signed(worst):>24s} {signed(end):>24s}   {traj}')
    print('\n## training rows\' mean candidate spread (seq arms), mean over seeds at updates 0, 20, 50, 100, 199')
    for a in arms[1:]:
        sp = [R[s][a]['training_candidate_spread'] for s in seeds]
        print(a, ' '.join(f'{np.mean([x[t] for x in sp]):.1f}' for t in (0, 20, 50, 100, 199)))
    print('\n## sequence loss (as trained), mean over seeds at updates 0, 1, 10, 50, 199')
    for a in arms:
        L = [R[s][a]['loss_every_update'] for s in seeds]
        print(a, ' '.join(f'{np.mean([x[t][3] for x in L]):.3f}' for t in (0, 1, 10, 50, 199)))


def signed(xs):
    return f'{np.mean(xs):+.2f} ({min(xs):+.2f} to {max(xs):+.2f})'


if __name__ == '__main__':
    main(*sys.argv[1:])
