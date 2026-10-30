"""Where the count lives in the code of the erasure-stage checkpoints (post hoc; docs/count-coordinate-analysis.md).

Per configuration and seed, from the saved final checkpoints (no training):
1. Axis alignment: the unit direction of the least-squares map from collection codes to count, and its largest absolute
   component (1 = the count is carried by one coordinate; 1/sqrt(16) = 0.25 = spread evenly).
2. Best coordinate j*: the coordinate with the highest collection R^2 (chosen on collections only), its R^2, and its
   pairwise order accuracy on taps and on each script's forms; whether the coordinate that would be best on a set (an
   oracle choice) is j*.
3. Readers: reads per coordinate (both operands) in the first layer of g and of P; whether the most-read coordinate is
   j*, and the rank correlation of reads with collection R^2 across coordinates.
Coordinates are not comparable across seeds (each seed trains its own code); only within-seed quantities are compared.
Retention cannot be examined: only final checkpoints are saved.

Usage: .venv/bin/python -m experiments.fan_in_count_coordinate pilot erasure
"""
import gzip
import json
import sys
import time

import numpy as np
from scipy import stats
import torch

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import generic_relations as G
from experiments import fan_in_pilot as FI
from experiments import fan_in_pilot_analysis as A

ROOT, ns = X.ROOT, X.ns
SETS = ('collections', 'taps', *[f'forms/{L}' for L in P.LANGS])


@torch.no_grad()
def direction(enc, model):
    """Unit direction of the least-squares count map from collection codes (centred), and its largest component."""
    code = G.codes_fn(enc, model)
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS]).double()
    x = colls.reshape(-1, colls.shape[-1])
    y = torch.arange(1., 10., dtype=torch.float64)[:, None].expand(-1, colls.shape[1]).reshape(-1)
    x, y = x - x.mean(0), y - y.mean()
    w = torch.linalg.solve(x.T @ x + A.RIDGE * len(x) * torch.eye(x.shape[1], dtype=torch.float64), x.T @ y)
    w = w / w.norm()
    return float(w.abs().max()), int(w.abs().argmax())


def reads(layer):
    """Reads per code coordinate over both operands, or None for a dense layer."""
    if not hasattr(layer, 'mask') or bool(layer.mask.all()):
        return None
    m = layer.mask[:, :2 * FI.CODE]
    return (m[:, :FI.CODE] + m[:, FI.CODE:]).sum(0).double().numpy()


def one(seed, cfg):
    enc, model, withheld, _, _ = A.load(seed, cfg)
    cp = A.coordinate_probe(enc, model)
    r2 = np.array(cp['r2'])
    j = int(r2.argmax())
    align, wj = direction(enc, model)
    out = {'best_coordinate': j, 'best_r2': float(r2[j]), 'linear_axis_alignment': align, 'linear_axis_coordinate': wj,
           'order_at_best': {k: cp[k][j] for k in SETS}, 'oracle_is_best': {k: int(np.argmax(cp[k])) == j for k in SETS[1:]},
           'order_best_possible': {k: float(np.max(cp[k])) for k in SETS}, 'order_mean_over_coordinates': {k: float(np.mean(cp[k])) for k in SETS},
           'coordinates_r2_over_half': int((r2 >= .5).sum())}
    for name in ('g', 'P'):
        rd = reads(getattr(model, name)[0])
        if rd is not None:
            out[f'reads_{name}'] = {'most_read_is_best': int(rd.argmax()) == j, 'reads_at_best_share': float(rd[j] / rd.sum()),
                                    'spearman_reads_r2': float(stats.spearmanr(rd, r2)[0]) if rd.std() > 0 and r2.std() > 0 else None}
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    run_rt = json.loads((ROOT / 'reports' / f'{FI.NAME}-runtime.json').read_text())
    FI.CHANNELS_LAST = run_rt.get('channels_last_image_encoder', False)
    torch.set_num_threads(run_rt.get('threads_per_worker', 1))
    X.prepare(); P.prepare(); D.prepare()
    res = {f'{cfg}/seed{s}': one(s, cfg) for cfg in FI.CONFIGS for s in A.SEEDS}
    out = ROOT / 'reports' / f'fan-in-count-coordinate-{FI.STAGE}'
    with gzip.open(f'{out}.json.gz', 'wt') as f:
        json.dump(res, f)
    (ROOT / 'reports' / f'fan-in-count-coordinate-{FI.STAGE}-runtime.json').write_text(json.dumps(
        {'command': ' '.join(sys.argv), 'wall_seconds': time.perf_counter() - t0, 'process_cpu_seconds': time.process_time() - c0,
         'threads': torch.get_num_threads(), 'runs': len(res)}, indent=1))
    cfgs = list(FI.CONFIGS)
    print(f'{"config":34s} align  bestR2 nR2>.5  ord(best): ' + ' '.join(k.split('/')[-1][:4] for k in SETS[1:]) + '  | oracle=best  | g reads: most=best share spearman')
    for c in cfgs:
        rs = [res[f'{c}/seed{s}'] for s in A.SEEDS]
        m = lambda f: np.mean([f(r) for r in rs])
        g = [r['reads_g'] for r in rs if 'reads_g' in r]
        print(f'{c:34s} {m(lambda r: r["linear_axis_alignment"]):.2f}   {m(lambda r: r["best_r2"]):.2f}   {m(lambda r: r["coordinates_r2_over_half"]):.1f}    '
              + ' '.join(f'{m(lambda r, k=k: r["order_at_best"][k]):.2f}' for k in SETS[1:]) + '  | '
              + ' '.join(f'{m(lambda r, k=k: r["oracle_is_best"][k]):.1f}' for k in SETS[1:])
              + (f'  | {np.mean([x["most_read_is_best"] for x in g]):.1f} {np.mean([x["reads_at_best_share"] for x in g]):.2f} '
                 f'{np.nanmean([x["spearman_reads_r2"] if x["spearman_reads_r2"] is not None else np.nan for x in g]):.2f}' if g else ''))


if __name__ == '__main__':
    main()
