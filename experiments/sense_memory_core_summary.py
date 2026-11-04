"""Tables for experiments.sense_memory_core (docs/sense-memory-core-v1-design.md).
Usage: python -m experiments.sense_memory_core_summary reports/sense-memory-core-v1-pilot.json[.gz]
Cells are means over seeds and both orders, with the range over those (seed, order) runs in brackets.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

READS = ('r2', 'r3', 'r4', 'r3_taps_to_symbol')
NAMES = {'r2': 'dots', 'r3': 'taps', 'r4': 'comparison', 'r3_taps_to_symbol': "taps -> symbol"}


def load(p):
    p = Path(p)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes())


def cell(xs):
    xs = [x for x in xs if x is not None]
    return '—' if not xs else f'{np.mean(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})'


def get(R, key, arm=None, read=None):
    """Values over seeds and orders for results key 'host|point|{order}|state|role'."""
    out = []
    for seed, r in R.items():
        for order in ('1', '2'):
            v = r['results'].get(key.replace('{order}', order))
            if v is None:
                continue
            if arm is not None:
                v = v.get(arm)
                if v is None:
                    continue
                v = v.get(read) if read else v
            out.append(v)
    return out


def main(path):
    R = load(path)['results']
    cal, ex = 'joined-mlp-calibrated', 'existing'
    print('## memory')
    for seed, r in R.items():
        for order, m in r['memory'].items():
            print(seed, 'order', order, {s: (st['units'], st['forced'], round(st['units_per_fact'], 2)) for s, st in m['states'].items()},
                  'centres opened', r['centres_opened'], f"read {1e3 * m['batch_one_read_seconds']:.2f} ms", 'hosts unchanged', r['hosts_unchanged_by_reads'])

    print('\n## block 3, `both` state, both sets (first and second written): end to end, bypass, retrieved-only')
    rows = [('R0 retrieval', lambda role: get(R, f'{cal}|block3|{{order}}|both|{role}', 'R0_retrieval'))]
    for role in ('first', 'second'):
        print(f'\n### {role} set   R0 {cell(get(R, f"{cal}|block3|{{order}}|both|{role}", "R0_retrieval"))}   answered {cell(get(R, f"{cal}|block3|{{order}}|both|{role}", "answered"))}'
              f'   sight-only dots {cell([v["r1"] for v in get(R, f"sight|{{order}}|both|{role}", "sense-only")])}')
        print(f'   at 0.70: R0 {cell(get(R, f"{cal}|block3|{{order}}|both|{role}", "R0_retrieval@0.70"))}   answered {cell(get(R, f"{cal}|block3|{{order}}|both|{role}", "answered@0.70"))}')
        arms = [(cal, 'trace'), (cal, 'trace@0.70'), (cal, 'trace|retrieved'), (cal, 'trace-bypass'), (cal, 'trace-single'), (cal, 'trace-single-bypass'), (cal, 'copy'),
                (cal, 'core-only'), (cal, 'fresh-dots'), (ex, 'copy-existing'), (ex, 'core-only'), (ex, 'fresh-dots')]
        print(f'{"arm":40s} ' + ' '.join(f'{NAMES[k]:>18s}' for k in READS))
        for host, arm in arms:
            print(f'{host[:8] + " " + arm:40s} ' + ' '.join(f'{cell(get(R, f"{host}|block3|{{order}}|both|{role}", arm, k)):>18s}' for k in READS))

    print('\n## host change: first set on `first`, through each checkpoint')
    for host, arm in ((cal, 'trace'), (cal, 'trace-bypass'), (cal, 'copy'), (cal, 'fresh-dots'), (ex, 'copy-existing'), (ex, 'fresh-dots')):
        for k in READS[:3]:
            print(f'{host[:8] + " " + arm:30s} {NAMES[k]:12s} ' + ' '.join(f'{p}: {cell(get(R, f"{host}|{p}|{{order}}|first|first", arm, k))}' for p in ('block1', 'block3', 'delay', 'relearn')))

    print('\n## interference: first set, `first` against `both`, same checkpoint (paired difference both - first)')
    for p in ('block3', 'delay', 'relearn'):
        for arm in ('trace', 'trace-bypass'):
            for k in ('R0',) + READS[:3]:
                if k == 'R0':
                    a, b = get(R, f'{cal}|{p}|{{order}}|first|first', 'R0_retrieval'), get(R, f'{cal}|{p}|{{order}}|both|first', 'R0_retrieval')
                    if arm != 'trace':
                        continue
                else:
                    a, b = get(R, f'{cal}|{p}|{{order}}|first|first', arm, k), get(R, f'{cal}|{p}|{{order}}|both|first', arm, k)
                d = [y - x for x, y in zip(a, b)]
                print(f'{p:8s} {arm:14s} {k if k == "R0" else NAMES[k]:12s} {np.mean(d):+.3f} ({min(d):+.3f} to {max(d):+.3f})')

    print('\n## answer dependence on the interpreter: same stored value and candidates, share of answers that change between checkpoints (first set, `first`)')
    for arm in ('trace-bypass', 'trace'):
        for k in ('r2', 'r3'):
            for a, b in (('block1', 'block3'), ('block3', 'delay'), ('block3', 'relearn')):
                ch = []
                for seed, r in R.items():
                    for order in ('1', '2'):
                        pa, pb = r['picks'].get(f'{cal}|{a}|{order}|first|first|{arm}'), r['picks'].get(f'{cal}|{b}|{order}|first|first|{arm}')
                        if pa and pb:
                            ch.append(float(np.mean(np.array(pa[k]) != np.array(pb[k]))))
                print(f'{arm:14s} {NAMES[k]:6s} {a} -> {b}: {cell(ch)}')

    print('\n## unseen symbols answered at tau')
    for state in ('first', 'both'):
        print(state, cell(get(R, f'sight|{{order}}|{state}|unseen', 'R5_unseen_answered')), 'at 0.70', cell(get(R, f'sight|{{order}}|{state}|unseen', 'R5_unseen_answered@0.70')))


if __name__ == '__main__':
    main(*sys.argv[1:])
