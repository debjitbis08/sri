"""Tables for experiments.relearning_predictor_stop (docs/relearning-predictor-stop-v1.md): reads at 0, 10, 15, 20 per arm, mean (range) over
seeds; the two stop arms' equality at update 10; and form sequence prediction read with update-10 parts swapped in.
Usage: python -m experiments.relearning_predictor_stop_summary reports/relearning-predictor-stop-v1.json[.gz]
"""
import sys

from experiments import sense_core_summary as SS
from experiments import relearning_path_removal_summary as PR

FORMS = [f'forms/{L}/{d}/1/distance' for L in ('english', 'chinese', 'devanagari_digits') for d in ('up', 'down')]


def main(path):
    d = SS.load(path)
    R, arms = d['results'], d['arms']
    seeds = sorted(R)
    print('delay reproduction:', {s: R[s]['delay_state_max_abs_difference_from_step8'] for s in seeds})
    eq = max(abs(f(R[s]['form-cand-stop']['track'][str(t)]) - f(R[s]['form-cand-stop+P']['track'][str(t)])) for s in seeds for t in (0, 10) for _, f in PR.MEASURES)
    print('form-cand-stop and form-cand-stop+P at 0 and 10, largest difference:', eq)
    for label, f in PR.MEASURES:
        print(f'\n## {label}')
        for a in arms:
            print(f'{a:18s}', ' '.join(f'{t}: {SS.cell([f(R[s][a]["track"][str(t)]) for s in seeds])}' for t in d['reads_at']))
    print('\n## form sequence prediction: as read | current codes + P at 10 | codes at 10 + current P   (at 15, 20; update-10 value first)')
    for k in FORMS:
        for a in arms[1:]:
            v10 = SS.cell([R[s][a]['track']['10']['composition'][k] for s in seeds])
            cells = []
            for t in ('15', '20'):
                tr = lambda s: R[s][a]['track'][t]
                cells.append(f'{t}: {SS.cell([tr(s)["composition"][k] for s in seeds])} | {SS.cell([tr(s)["composition_current_codes_P_at_10"][k] for s in seeds])}'
                             f' | {SS.cell([tr(s)["composition_codes_at_10_current_P"][k] for s in seeds])}')
            print(f'{k:36s} {a:18s} 10: {v10}   ' + '   '.join(cells))
    print('\n## collection composition, same three reads')
    for k in ('collections/up/2/distance', 'collections/up/3/distance', 'taps/up/2/distance', 'collections/down/1/distance'):
        for a in arms[1:]:
            tr = lambda s: R[s][a]['track']['20']
            print(f'{k:30s} {a:18s} 20: {SS.cell([tr(s)["composition"][k] for s in seeds])} | {SS.cell([tr(s)["composition_current_codes_P_at_10"][k] for s in seeds])}'
                  f' | {SS.cell([tr(s)["composition_codes_at_10_current_P"][k] for s in seeds])}')


if __name__ == '__main__':
    main(*sys.argv[1:])
