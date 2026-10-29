"""Comparison-direction diagnostic of composed states on Part D checkpoints (docs/number-construction-v1-spec.md,
"Comparison-direction diagnostic of composed states").

How far does the learned successor, applied k times (or its exact inverse), move a code along the comparison direction
w, against how far real items of the target count lie? No training.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_projection pilot
Environment: NCC_RUNS (runs directory name; default the Part D runs).
"""
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_construction_composition as C

nc, ns = X.nc, X.ns
ROOT = X.ROOT
NAME = f'number-construction-v1-projection-{X.MODE}{C.TAG}' + ('-smoke' if X.SMOKE else '')
PREFIX = ROOT / 'reports' / NAME


@torch.no_grad()
def run(path):
    t0, c0 = time.perf_counter(), time.process_time()
    ck = torch.load(path, weights_only=False)
    arm, seed = ck['arm'], ck['seed']
    enc, sri = X.make_model(seed)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); sri.load_state_dict(ck['sri'])
    sv = {m: B.Solver(sri, m) for m in ('image', 'sequence')}
    A, t, w = sri.A, sri.t, sri.larger
    Ainv = torch.linalg.inv(A)
    q = lambda c: (c @ w).numpy()
    out = {'eigenvalue_abs': sorted(float(x) for x in torch.linalg.eigvals(A).abs()), 'kinds': {}}
    for kind in C.KINDS:
        m = 'sequence' if kind == 'taps' else 'image'
        codes = {n: sri.split(sv[m].solve(enc[m](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner'] for n, x in C.items(kind).items()}
        qbar = {n: float(q(codes[n]).mean()) for n in ns.NUMBERS}
        step = float(np.mean([qbar[n + 1] - qbar[n] for n in range(1, 9)]))
        res = {'real_position': qbar, 'real_step': step,
               'per_step_advance_on_real': {str(n): float(q(codes[n] @ A.T + t).mean() - qbar[n]) for n in ns.NUMBERS}}
        for action, sgn in (('add', 1), ('remove', -1)):
            for k in C.KS:
                ratios, disc = [], []
                for n in ns.NUMBERS:
                    tgt = n + sgn * k
                    if not 1 <= tgt <= 9:
                        continue
                    c = codes[n]
                    for _ in range(k):
                        c = c @ A.T + t if sgn > 0 else (c - t) @ Ainv.T
                    real = qbar[tgt] - qbar[n]
                    ratios.append(float((q(c) - q(codes[n])).mean() / real))
                    disc.append(float(sgn * (q(c).mean() - qbar[tgt]) / step))
                res[f'{action}/{k}'] = {'advance_ratio': float(np.mean(ratios)), 'signed_discrepancy_steps': float(np.mean(disc)),
                                        'advance_ratio_by_start': ratios}
        out['kinds'][kind] = res
    return f'{arm}/seed{seed}', out, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                       'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    P.prepare(); D.prepare()
    phases = {'prepare_wall_seconds': time.perf_counter() - wall0}
    paths = sorted(C.RUN_DIR.glob('*.pt'))
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(paths))) as pool:
        done = pool.map(run, paths)
    phases['analysis_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(P.SPEC.relative_to(ROOT)), 'status': 'comparison-direction diagnostic on Part D checkpoints (no training)',
           'config': {'runs': str(C.RUN_DIR.relative_to(ROOT)), 'kinds': C.KINDS, 'steps': C.KS},
           'checkpoint_sha256': {p.name: nc.sha256(p) for p in paths}, 'results': {k: r for k, r, _ in sorted(done)}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'spec_sha256': nc.sha256(P.SPEC),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'phases': phases, 'tasks': worker,
        'data_bytes': data['data_bytes'], 'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
