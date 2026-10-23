"""Run phase B of core mediation v1: exhaustive bounded realizability by orbit."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time
import tracemalloc

import numpy
import scipy

from src.core_mediation import (
    BOUND, BRANCH_LIMIT, CLASSES, MARGIN, configurations, orbits, solve, stable_set,
)


ROOT = Path(__file__).resolve().parents[1]
STRONG_SCALE = 64


def cases():
    for cls in CLASSES:
        for n in ((2,) if cls == 'G' else (2, 3, 4)):
            for level in (1, 2):
                for strict in (False, True):
                    yield cls, n, level, strict


def run_case(cls, n, level, strict):
    rows = []
    for targets, size in sorted(orbits(cls, n, level).items()):
        result = solve(cls, n, targets, strict)
        result['orbit_size'] = size
        if result['outcome'] == 'realizable':
            tables = {tuple(int(p) if p.isdigit() else p for p in key.split('.')): v
                      for key, v in result['witness'].items()}
            result['stable_at_scale_1'] = [list(x) for x in stable_set(cls, n, tables)]
            result[f'stable_at_scale_{STRONG_SCALE}'] = [
                list(x) for x in stable_set(cls, n, tables, STRONG_SCALE)]
        rows.append(result)
    counts = Counter(r['outcome'] for r in rows)
    return dict(cls=cls, n=n, level=level, strict=strict, orbits=len(rows),
                target_sets=sum(r['orbit_size'] for r in rows),
                outcomes=dict(sorted(counts.items())), results=rows)


def summary(blocks):
    table = {}
    for b in blocks:
        key = f"{b['cls']}/L{b['level']}/{'strict' if b['strict'] else 'base'}"
        table.setdefault(key, {})[str(b['n'])] = b['outcomes']
    smallest = {}
    for key, by_n in table.items():
        realizable = [int(n) for n, o in by_n.items() if o.get('realizable')]
        smallest[key] = min(realizable) if realizable else None
    return dict(outcomes=table, smallest_core_with_any_realizable_orbit=smallest)


def render():
    blocks = [run_case(*case) for case in cases()]
    files = ['src/core_mediation.py', 'experiments/core_mediation.py',
             'docs/core-mediation-v1-contract.md']
    return dict(
        experiment='core-mediation-v1-phase-B',
        contract='docs/core-mediation-v1-contract.md',
        bounds=dict(entry_bound=BOUND, margin=MARGIN, branch_limit=BRANCH_LIMIT,
                    strong_scale_reported=STRONG_SCALE),
        versions=dict(numpy=numpy.__version__, scipy=scipy.__version__),
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        summary=summary(blocks),
        cases=blocks,
    )


def main():
    tracemalloc.start()
    start = time.perf_counter()
    report = render()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    (ROOT/'reports/core-mediation-v1.json').write_text(json.dumps(report, indent=1) + '\n')
    (ROOT/'reports/core-mediation-v1-runtime.json').write_text(json.dumps(dict(
        seconds=round(elapsed, 3), peak_traced_bytes=peak, python=platform.python_version(),
        machine=platform.machine(),
        solves=sum(b['orbits'] for b in report['cases'])), indent=1) + '\n')
    for key, by_n in report['summary']['outcomes'].items():
        print(key, by_n)


if __name__ == '__main__':
    main()
