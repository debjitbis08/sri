"""Architecture audit stage 1: the number-line pilots rerun on a standard shared-embedding network
(docs/architecture-audit.md; model in src/standard_net.py).

The Part D and multi-step pilots, and the composition, projection and imagery analyses, run unchanged except that the
SRI network is replaced by StandardNet: encoders -> one linear projection per sense -> the same code and relation heads.
Same seeds, data streams, losses, curriculum, rehearsal and evaluations, and the same encoder initialization, so every
run pairs with its SRI pilot run. Outputs are written under '-standard-' names; the frozen pilots are not touched.

Usage (mode 'pilot' keeps the pilots' settings):
  NSY_WORKERS=11 .venv/bin/python -m experiments.standard_baseline pilot part-d
  NSY_WORKERS=11 .venv/bin/python -m experiments.standard_baseline pilot multistep
  NSY_WORKERS=11 .venv/bin/python -m experiments.standard_baseline pilot composition-d   (also projection-d, imagery,
  composition-m, projection-m)
"""
import os
import sys

STEP = sys.argv[2] if len(sys.argv) > 2 else 'part-d'
MODE, SUFFIX = (sys.argv[1] if len(sys.argv) > 1 else 'pilot'), ('-smoke' if os.environ.get('NSY_SMOKE') == '1' else '')
RUNS_D, RUNS_M = (f'number-construction-v1-standard-{k}-{MODE}{SUFFIX}' for k in ('part-d', 'multistep'))
os.environ['NCC_RUNS'] = RUNS_M if STEP.endswith('-m') else RUNS_D  # read at import by the composition/projection runners
os.environ['NCI_RUNS'] = RUNS_D

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments import number_systems as X  # noqa: E402
from experiments import number_construction_b1 as B  # noqa: E402
from experiments import number_construction_part_a as P  # noqa: E402
from experiments import number_construction_part_d as D  # noqa: E402
from src import number_core as ncore  # noqa: E402
from src import number_systems_net as net  # noqa: E402
from src import standard_net as std  # noqa: E402

ROOT = X.ROOT
FIT_ITEMS = 4000  # training inputs per sense for the imagery decoder


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}  # as X.make_model
    model = std.StandardNet(('image', 'sequence'), torch.Generator().manual_seed(seed))
    model._enc = enc  # for the imagery decoder fit (state dicts are loaded into these modules in place)
    return enc, model


@torch.no_grad()
def reverse_solve(model, sense, clamp, value):
    """Imagery: decode a clamped code to front-end features by ridge regression fitted on training inputs. Only the
    core-inner clamp exists in the standard network; other clamps (the SRI core middle layer) return zeros."""
    from experiments import number_construction_imagery as I
    if clamp != I.CORE_INNER:
        return torch.zeros(len(value), net.SIZES[0])
    decs = model.__dict__.setdefault('_dec', {})
    if sense not in decs:
        rng = np.random.default_rng(0)
        if sense == 'image':
            h = FIT_ITEMS // 2
            x = [P.DOTS['train'][rng.integers(len(P.DOTS['train']), size=h), rng.integers(9, size=h)]]
            for lang in P.LANGS:
                pool = X.DATA[(lang, 'train')][0]
                x.append(pool[rng.integers(len(pool), size=(FIT_ITEMS - h) // len(P.LANGS))])
            x = np.concatenate(x)
        else:
            x = D.TAPS['train'][rng.integers(len(D.TAPS['train']), size=FIT_ITEMS)]
        f = model._enc[sense](torch.from_numpy(np.ascontiguousarray(x)))
        decs[sense] = std.LinearDecoder(model.proj[sense](f), f)
    return decs[sense](value)


def install():
    X.make_model = make_model
    B.Solver = std.StandardSolver
    P.Coupled = lambda model: std.StandardSolver(model, 'image')
    D.Sequence = lambda model: std.StandardSolver(model, 'sequence')
    P.total_energy = std.total_energy


def main():
    install()
    if STEP == 'part-d':
        D.NAME = RUNS_D
        D.PREFIX, D.RUNS = ROOT / 'reports' / D.NAME, ROOT / 'runs' / D.NAME
        D.main()
    elif STEP == 'multistep':
        from experiments import number_construction_multistep as M
        M.NAME = RUNS_M
        M.PREFIX, M.RUNS = ROOT / 'reports' / M.NAME, ROOT / 'runs' / M.NAME
        M.main()
    elif STEP in ('composition-d', 'composition-m'):
        from experiments import number_construction_composition as C
        C.main()
    elif STEP in ('projection-d', 'projection-m'):
        from experiments import number_construction_projection as R
        R.main()
    elif STEP == 'imagery':
        from experiments import number_construction_imagery as I
        I.reverse_solve = reverse_solve
        I.NAME = f'number-construction-v1-standard-imagery-{X.MODE}' + ('-smoke' if X.SMOKE else '')
        I.PREFIX = ROOT / 'reports' / I.NAME
        I.main()
    else:
        raise SystemExit(f'unknown step {STEP}')


if __name__ == '__main__':
    main()
