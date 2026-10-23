"""Run phase C of core mediation v1: structured internals, recovery, input and output."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import tracemalloc

from src.core_structured import (
    CORE_CODES, Module, SAT_CODE, System, distance_one_code, mode_recovery,
    pulse_response, recovery,
)


ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (1, 2, 4, 8)
OUTPUTS = (None, 1, 4)
PULSES = (1, 2, 4, 8, 16, 32)
MODULES = ('A', 'B', 'K')


def witnesses():
    report = json.loads((ROOT/'reports/core-mediation-v1.json').read_text())
    for case in report['cases']:
        if not case['strict']:
            continue
        for row in case['results']:
            if row['outcome'] == 'realizable':
                tables = {tuple(int(p) if p.isdigit() else p for p in k.split('.')): v
                          for k, v in row['witness'].items()}
                yield case['cls'], case['n'], tuple(map(tuple, row['targets'])), tables


def summarize(rows):
    counts = Counter()
    for r in rows:
        key = f"{r['kind']}/{MODULES[r['module']]}"
        if r['status'] == 'unresolved':
            counts[key+'/unresolved'] += 1
            continue
        counts[key+'/trials'] += 1
        counts[key+'/readout_recovered'] += r['readout_recovered']
        counts[key+'/state_recovered'] += r['state_recovered']
        counts[key+'/other_module_readout_changed'] += r['other_module_readout_changed']
    return dict(sorted(counts.items()))


def pulse_case(cls, n, tables, lam, targets, mu):
    rows, selected = [], 0
    for x in targets:
        for h in PULSES:
            out = {e: pulse_response(cls, n, tables, lam, x, e, h, mu) for e in (0, 1)}
            finals = {e: out[e].get('final') for e in (0, 1)}
            ok = (all(out[e]['status'] == 'resolved' and len(finals[e]) == 1 for e in (0, 1))
                  and finals[0] != finals[1]
                  and all(finals[e][0][0] == e and tuple(finals[e][0]) in targets for e in (0, 1))
                  and any(finals[e][0][2] != x[2] for e in (0, 1)))
            selected += ok
            rows.append(dict(start=list(x), h=h, input_selected_mediation=ok,
                             final={str(e): [list(r) for r in finals[e] or []] for e in (0, 1)},
                             status={str(e): out[e]['status'] for e in (0, 1)}))
    return dict(trials=len(rows), input_selected_mediation=selected, rows=rows)


def run_witness(cls, n, targets, tables):
    out = dict(cls=cls, n=n, targets=[list(x) for x in targets])
    out['mode_level_control'] = [dict(target=list(x), moves=mode_recovery(cls, n, tables, x))
                                 for x in targets]
    out['core_separation'] = Module(CORE_CODES[n]).min_separation()
    out['satellite_separation'] = Module(SAT_CODE).min_separation()
    configs = []
    for lam in LAMBDAS:
        for code_name, code in (('separated', None), ('distance_one', distance_one_code(n))):
            for mu in (OUTPUTS if code_name == 'separated' else (None,)):
                system = System(cls, n, tables, lam, core_code=code, mu=mu)
                encoded = [system.encode(x) for x in targets]
                terminal = [not system.successors(s) for s in encoded]
                readout_ok = [system.readouts(s) == x for s, x in zip(encoded, targets)]
                rows = [r for x in targets for r in recovery(system, x)]
                row = dict(lam=lam, core_code=code_name, output_mu=mu,
                           targets_terminal=all(terminal), targets_read_correctly=all(readout_ok),
                           all_flip1_recovered=all(r.get('readout_recovered') for r in rows
                                                   if r['kind'] == 'flip1'),
                           counts=summarize(rows),
                           max_visited=max(r.get('visited', 0) for r in rows))
                if code_name == 'separated':
                    row['pulses'] = pulse_case(cls, n, tables, lam, targets, mu)
                configs.append(row)
    out['configs'] = configs
    return out


def render():
    results = [run_witness(*w) for w in witnesses()]
    files = ['src/core_structured.py', 'experiments/core_structured.py',
             'docs/core-mediation-v1-contract.md', 'reports/core-mediation-v1.json']
    return dict(
        experiment='core-mediation-v1-phase-C',
        contract='docs/core-mediation-v1-contract.md',
        grid=dict(lambdas=LAMBDAS, output_mu=OUTPUTS, pulses=PULSES),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        witnesses=results,
    )


def main():
    tracemalloc.start()
    start = time.perf_counter()
    report = render()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    (ROOT/'reports/core-mediation-v1-phase-c.json').write_text(json.dumps(report, indent=1) + '\n')
    (ROOT/'reports/core-mediation-v1-phase-c-runtime.json').write_text(json.dumps(dict(
        seconds=round(elapsed, 3), peak_traced_bytes=peak, python=platform.python_version(),
        machine=platform.machine(), witnesses=len(report['witnesses'])), indent=1) + '\n')
    print(len(report['witnesses']), 'witnesses', round(elapsed, 1), 's')


if __name__ == '__main__':
    main()
