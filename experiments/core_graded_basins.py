"""Run the declared graded-basin grid: 14 witnesses x sigma x (lambda_S, lambda_K), plus a tie control."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

from src.core_basins import analyze
from src.core_graded import evaluate


ROOT = Path(__file__).resolve().parents[1]
SIGMAS = (5, 9, 13)
BARRIERS = (1, 2, 3)
H = 32
TIE_CONTROL = dict(cls='T', sigma=8, lam=(1, 1), targets=((0, 0, 0), (0, 1, 1), (1, 0, 1)),
                   tables={('uA', 1): 1, ('ABK', 0, 0, 1): 1, ('ABK', 0, 1, 0): 1,
                           ('ABK', 1, 1, 0): 1, ('ABK', 1, 0, 1): -1})


def witnesses():
    report = json.loads((ROOT/'reports/core-mediation-v1.json').read_text())
    for case in report['cases']:
        if case['strict'] and case['n'] == 2 and case['cls'] in 'SFTG':
            for row in case['results']:
                if row['outcome'] == 'realizable':
                    tables = {tuple(int(p) if p.isdigit() else p for p in k.split('.')): v
                              for k, v in row['witness'].items()}
                    yield case['cls'], case['level'], tuple(map(tuple, row['targets'])), tables


def setting(cls, tables, targets, sigma, lam):
    basin = analyze(cls, tables, targets, sigma, lam)
    criteria = evaluate(cls, tables, targets, sigma, lam, H, with_census=False)
    return dict(sigma=sigma, lam_satellite=lam[0], lam_core=lam[1],
                P1=criteria['P1'], P2=criteria['P2'], joint=criteria['joint'],
                pulses=criteria['pulses'], basins=basin)


def render():
    rows = []
    for cls, level, targets, tables in witnesses():
        for sigma in SIGMAS:
            for lam in ((s, k) for s in BARRIERS for k in BARRIERS):
                rows.append(dict(cls=cls, level=level, targets=[list(x) for x in targets],
                                 **setting(cls, tables, targets, sigma, lam)))
    c = TIE_CONTROL
    tie = setting(c['cls'], c['tables'], c['targets'], c['sigma'], c['lam'])
    files = ['src/core_basins.py', 'src/core_graded.py', 'src/core_mediation.py',
             'experiments/core_graded_basins.py', 'docs/core-graded-basins-contract.md',
             'reports/core-mediation-v1.json']
    return dict(
        experiment='core-graded-basins',
        contract='docs/core-graded-basins-contract.md',
        grid=dict(sigmas=SIGMAS, barriers=BARRIERS, pulse=H),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        propositions_confirmed_everywhere=all(r['basins']['propositions_confirmed'] for r in rows)
                                         and tie['basins']['propositions_confirmed'],
        tie_control=tie,
        settings=rows,
    )


def main():
    start = time.perf_counter()
    report = render()
    elapsed = time.perf_counter() - start
    (ROOT/'reports/core-graded-basins.json').write_text(json.dumps(report, indent=1) + '\n')
    (ROOT/'reports/core-graded-basins-runtime.json').write_text(json.dumps(dict(
        seconds=round(elapsed, 3), python=platform.python_version(), machine=platform.machine(),
        settings=len(report['settings'])), indent=1) + '\n')
    print(len(report['settings']), 'settings; propositions confirmed everywhere:',
          report['propositions_confirmed_everywhere'], round(elapsed, 1), 's')
    print(Counter((r['cls'], r['joint']) for r in report['settings']))


if __name__ == '__main__':
    main()
