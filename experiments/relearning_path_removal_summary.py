"""Tables for experiments.relearning_path_removal (docs/relearning-path-removal-v1.md): each read at updates 0, 5, 10, 15, 20 per arm, mean (range)
over seeds; the unchanged arm against stage A's unchanged track (same state and batches); clip factors.
Usage: python -m experiments.relearning_path_removal_summary reports/relearning-path-removal-v1.json[.gz] [reports/relearning-remedies-a-v1.json.gz]
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

C = lambda key: (lambda r: r['composition'][key])
MEASURES = (
    ('collections -> taps, fresh', lambda r: r['collections_taps']['collections_to_taps']['fresh']),
    ('collections -> taps, stored', lambda r: r['collections_taps']['collections_to_taps']['stored']),
    ('taps -> collections', lambda r: r['collections_taps']['taps_to_collections']),
    ('memory A: symbol -> taps', lambda r: r['memory_present']['A']['end_to_end']['r3']),
    ('memory B: symbol -> taps', lambda r: r['memory_present']['B']['end_to_end']['r3']),
    ('memory A: taps -> symbol', lambda r: r['memory_present']['A']['end_to_end']['r3_taps_to_symbol']),
    ('English matching', lambda r: r['english_matching']),
    ('sequence: English forms, up 1', C('forms/english/up/1/distance')),
    ('sequence: English forms, down 1', C('forms/english/down/1/distance')),
    ('sequence: Devanagari forms, up 1', C('forms/devanagari_digits/up/1/distance')),
    ('sequence: Chinese forms, up 1', C('forms/chinese/up/1/distance')),
    ('sequence: collections, up 1', C('collections/up/1/distance')),
    ('sequence: collections, down 1', C('collections/down/1/distance')),
    ('sequence: taps, up 1', C('taps/up/1/distance')),
    ('composition: collections, up 2', C('collections/up/2/distance')),
    ('composition: collections, up 3', C('collections/up/3/distance')),
    ('composition: taps, up 2', C('taps/up/2/distance')),
    ('collection class spacing', lambda r: r['relation']['geometry']['collection_class_spacing']),
    ('same-count c-t distance / spacing', lambda r: r['relation']['geometry']['same_count_c_t']),
)


def main(path, stage_a=None):
    d = SS.load(path)
    R, arms = d['results'], d['arms']
    seeds = sorted(R)
    print('delay reproduction:', {s: R[s]['delay_state_max_abs_difference_from_step8'] for s in seeds})
    if stage_a:
        A = SS.load(stage_a)['results']
        diff = max(abs(f(R[s]['unchanged']['track'][str(t)]) - f(A[s]['unchanged']['track'][str(t)])) for s in seeds for t in (0, 10, 20) for _, f in MEASURES)
        print('unchanged against stage A unchanged at 0, 10, 20, largest difference over all measures:', diff)
    ts = d['reads_at']
    for label, f in MEASURES:
        print(f'\n## {label}')
        for a in arms:
            print(f'{a:18s}', ' '.join(f'{t}: {SS.cell([f(R[s][a]["track"][str(t)]) for s in seeds])}' for t in ts))
    print('\n## clip factor (mean over seeds) at updates 0, 5, 10, 15, 19')
    for a in arms:
        print(f'{a:18s}', ' '.join(f'{np.mean([R[s][a]["per_update"][t]["clip"] for s in seeds]):.3f}' for t in (0, 5, 10, 15, 19)))
    print('path share of the clipped full gradient (removed arm):',
          ' '.join(f'{np.mean([R[s]["form-cand-removed"]["per_update"][t]["path_share_of_gradient"] for s in seeds]):.2f}' for t in (0, 5, 10, 15, 19)))


if __name__ == '__main__':
    main(*sys.argv[1:])
