"""Tables for experiments.relearning_branches (docs/relearning-branches-v1.md).
Usage: python -m experiments.relearning_branches_summary reports/relearning-branches-v1.json[.gz]
"""
import sys

import numpy as np

from experiments import sense_core_summary as SS

STEPS = ('0', '20', '50', '100', '150', '200')
MEASURES = (
    ('collections -> taps, fresh (evaluated order)', lambda r: r['collections_taps']['collections_to_taps']['fresh']),
    ('collections -> taps, stored values', lambda r: r['collections_taps']['collections_to_taps']['stored']),
    ('collections -> taps, fresh, trained order', lambda r: r['collections_taps']['collections_to_taps_swapped_order']['fresh']),
    ('taps -> collections', lambda r: r['collections_taps']['taps_to_collections']),
    ('c -> t ranking, candidate-centred', lambda r: r['relation']['ranking']['c -> t by g(c,t) (evaluated)']['candidate_centred']),
    ('symmetry: correlation g(c,t), g(t,c)', lambda r: r['relation']['symmetry']['correlation']),
    ('symmetry: same-count violation', lambda r: r['relation']['symmetry']['same_count_pairs']),
    ('AUROC g(c,t)', lambda r: r['relation']['discrimination_auroc']['g(c,t)']),
    ('AUROC g(t,c)', lambda r: r['relation']['discrimination_auroc']['g(t,c)']),
    ('collection class spacing', lambda r: r['relation']['geometry']['collection_class_spacing']),
    ('taps spacing / collection spacing', lambda r: r['relation']['geometry']['taps_class_spacing']),
    ('same-count c-t distance / spacing', lambda r: r['relation']['geometry']['same_count_c_t']),
    ('English forms nearest same-count collections', lambda r: r['relation']['geometry']['english_forms_to_collections_nearest']),
    ('English matching (relearned script)', lambda r: r['english_matching']),
    ('memory: A symbol -> taps, end to end', lambda r: r['memory_present']['A']['end_to_end']['r3']),
    ('memory: A symbol -> taps, stored value', lambda r: r['memory_present']['A']['bypass']['r3']),
    ('memory: B symbol -> taps, end to end', lambda r: r['memory_present']['B']['end_to_end']['r3']),
    ('memory: A taps -> symbol, end to end', lambda r: r['memory_present']['A']['end_to_end']['r3_taps_to_symbol']),
)


def main(path):
    d = SS.load(path)
    R, branches = d['results'], d['branches']
    seeds = sorted(R)
    print('reproduction:', {s: (R[s]['delay_state_max_abs_difference_from_step8'], R[s]['as-run']['relearn_state_max_abs_difference_from_step8']) for s in seeds})
    for label, f in MEASURES:
        print(f'\n## {label}: mean (range) over seeds, by update')
        print(f'{"branch":14s} ' + ' '.join(f'{t:>18s}' for t in STEPS))
        for b in branches:
            print(f'{b:14s} ' + ' '.join(f'{SS.cell([f(R[s][b]["track"][t]) for s in seeds]):>18s}' for t in STEPS))
        print(f'{"per seed, 200":14s} ' + ' | '.join(f'{b}: ' + ' '.join(f'{f(R[s][b]["track"]["200"]):.2f}' for s in seeds) for b in branches))


if __name__ == '__main__':
    main(*sys.argv[1:])
