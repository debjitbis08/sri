"""Run the graded core interface v1 feasibility example; no broader grid."""
import hashlib
import json
from pathlib import Path

from src.core_graded import evaluate


ROOT = Path(__file__).resolve().parents[1]
WITNESS = {('uA', 1): 1, ('ABK', 0, 0, 1): 1, ('ABK', 0, 1, 0): 1,
           ('ABK', 1, 1, 0): 1, ('ABK', 1, 0, 1): -1}
TARGETS = ((0, 0, 0), (0, 1, 1), (1, 0, 1))
SIGMA, H, LAMBDAS = 8, 8, (1, 2, 3)


def render():
    files = ['src/core_graded.py', 'src/core_mediation.py', 'experiments/core_graded.py',
             'docs/core-graded-v1-contract.md']
    return dict(
        experiment='core-graded-v1-feasibility',
        contract='docs/core-graded-v1-contract.md',
        witness={'.'.join(map(str, k)): v for k, v in WITNESS.items()},
        targets=[list(x) for x in TARGETS],
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        settings=[evaluate('T', WITNESS, TARGETS, SIGMA, lam, H) for lam in LAMBDAS],
    )


def main():
    report = render()
    (ROOT/'reports/core-graded-v1.json').write_text(json.dumps(report, indent=1) + '\n')
    for s in report['settings']:
        print(s['lam'], 'P1', s['P1'], 'P2', s['P2'], 'joint', s['joint'])


if __name__ == '__main__':
    main()
