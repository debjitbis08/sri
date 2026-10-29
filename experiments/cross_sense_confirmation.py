"""Cross-sense confirmation v1 (docs/cross-sense-confirmation-v1-contract.md): the Part D D-varied+rehearsal arm on fresh
seeds 200-209, for the SRI network and the standard network, to test whether SRI's form -> taps advantage for withheld
Chinese forms (architecture audit stage 1) is real.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.cross_sense_confirmation pilot sri
       NSY_WORKERS=11 .venv/bin/python -m experiments.cross_sense_confirmation pilot standard
Mode 'pilot' keeps the Part D pilot's settings. Then experiments.cross_sense_confirmation_analysis.
"""
import os
import sys

ARCH = sys.argv[2] if len(sys.argv) > 2 else 'sri'
SEEDS = tuple(range(200, 210))
os.environ['NCD_SEEDS'] = ','.join(map(str, SEEDS))  # read at import by the Part D runner

from experiments import number_systems as X  # noqa: E402
from experiments import number_construction_part_d as D  # noqa: E402
from experiments import standard_baseline as SB  # noqa: E402

ARM = 'D-varied+rehearsal'


def name(arch, mode=None):
    return f'cross-sense-confirmation-v1-{arch}-{mode or X.MODE}' + ('-smoke' if X.SMOKE else '')


def main():
    if ARCH not in ('sri', 'standard'):
        raise SystemExit(f'unknown architecture {ARCH}')
    if ARCH == 'standard':
        SB.install()
    D.ARMS = {ARM: D.ARMS[ARM]}
    D.SEEDS = SEEDS[:2] if X.SMOKE else SEEDS
    D.NAME = name(ARCH)
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / D.NAME, X.ROOT / 'runs' / D.NAME
    D.main()


if __name__ == '__main__':
    main()
