"""Architecture audit stage 1: computation costs of the SRI and standard networks (docs/architecture-audit.md;
docs/computation-reporting.md).

Deterministic part (reports/architecture-audit-stage1-costs.json): parameter counts and bytes per network, from the
saved checkpoints, and the paired training and evaluation cost totals copied from each runner's runtime file.
Timing part (reports/architecture-audit-stage1-costs-runtime.json): a perception microbenchmark per sense, from front-end
features to the 16-dim code, for one checkpoint of each network (one thread, float32, batch of 270 as in an evaluation
call, 20 warm-up and 200 timed repeats; median and quartiles). Excludes the CNN front ends, which are identical, and
for SRI includes the Cholesky factorization once per solver construction, timed separately.

Usage: .venv/bin/python -m experiments.standard_baseline_costs
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments import standard_baseline as SB
from src import number_concept as nc
from src import standard_net as std

ROOT = X.ROOT
PAIRS = {'part-d': ('number-construction-v1-part-d-pilot', SB.RUNS_D), 'multistep': ('number-construction-v1-multistep-pilot', SB.RUNS_M)}
ANALYSES = {
    'composition-d': ('number-construction-v1-composition-pilot', 'number-construction-v1-composition-pilot-standard-part-d-pilot'),
    'projection-d': ('number-construction-v1-projection-pilot', 'number-construction-v1-projection-pilot-standard-part-d-pilot'),
    'composition-m': ('number-construction-v1-composition-pilot-multistep-pilot', 'number-construction-v1-composition-pilot-standard-multistep-pilot'),
    'projection-m': ('number-construction-v1-projection-pilot-multistep-pilot', 'number-construction-v1-projection-pilot-standard-multistep-pilot'),
    'imagery': ('number-construction-v1-imagery-pilot', 'number-construction-v1-standard-imagery-pilot'),
}
BATCH, WARM, REPS = 270, 20, 200


def runtime(name):
    return json.loads((ROOT / 'reports' / f'{name}-runtime.json').read_text())


def task_totals(rt):
    keys = ('cpu_seconds', 'train_cpu_seconds', 'eval_cpu_seconds', 'transfer_cpu_seconds', 'wall_seconds')
    return {k: float(sum(t.get(k, 0.) for t in rt['tasks'].values())) if any(k in t for t in rt['tasks'].values()) else None
            for k in keys} | {'tasks': len(rt['tasks']), 'updates_per_task': sorted({t.get('updates') for t in rt['tasks'].values()}),
                              'wall_seconds_job': rt.get('wall_seconds'), 'peak_rss_children_max_kb': rt.get('peak_rss_children_max_kb')}


def param_counts(state):
    groups = {}
    for k, v in state.items():
        g = 'encoders' if k.startswith(('image.', 'sequence.')) else 'relation heads' if k.split('.')[-1] in ('A', 't', 'larger', 'judge_bias', 'judge_scale') else 'perception (SRI modules or projections)'
        groups.setdefault(g, 0)
        groups[g] += v.numel()
    return groups


def bench(fn):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    q = np.percentile(ts, [25, 50, 75]) * 1e6
    return {'median_us': float(q[1]), 'q25_us': float(q[0]), 'q75_us': float(q[2]), 'repeats': REPS, 'warmup': WARM}


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    det, timing = {'pilots': {}, 'analyses': {}, 'parameters': {}}, {}
    for k, (a, b) in PAIRS.items():
        det['pilots'][k] = {'sri': task_totals(runtime(a)), 'standard': task_totals(runtime(b))}
    for k, (a, b) in ANALYSES.items():
        det['analyses'][k] = {arch: {'wall_seconds': runtime(n).get('wall_seconds'), 'worker_cpu_seconds_total': runtime(n).get('worker_cpu_seconds_total')}
                              for arch, n in (('sri', a), ('standard', b))}
    ck = {'sri': torch.load(ROOT / 'runs/number-construction-v1-part-d-pilot/D-varied+rehearsal-seed100.pt', weights_only=False),
          'standard': torch.load(ROOT / 'runs' / SB.RUNS_D / 'D-varied+rehearsal-seed100.pt', weights_only=False)}
    for arch, c in ck.items():
        state = {**{f'image.{k}': v for k, v in c['image'].items()}, **{f'sequence.{k}': v for k, v in c['sequence'].items()}, **c['sri']}
        n = param_counts(state)
        det['parameters'][arch] = {'counts': n, 'total': sum(n.values()), 'float32_bytes': 4 * sum(n.values()),
                                   'adam_state_bytes': 8 * sum(n.values())}
    det['selection_and_tuning'] = {'cost': 'none: the revision evidence weight (1.0) and the imagery ridge (1e-3) were fixed before running; '
                                           'no candidates were scored', 'rejected_candidates': 0}
    det['notes'] = ['Rehearsal keeps a buffer of 500 episodes per language in both networks (index records, not rendered images).',
                    'The standard network has no energy; the energy confidence feature is unavailable (null in the comparison).']
    # perception microbenchmark (features -> code), one sense, one checkpoint of each network
    enc, sri = X.make_model(100)
    sri.load_state_dict(ck['sri']['sri'])
    _, sm = SB.make_model(100)
    sm.load_state_dict(ck['standard']['sri'])
    f = torch.randn(BATCH, sri.sizes[0], generator=torch.Generator().manual_seed(0))
    for m in ('image', 'sequence'):
        sv, ss = B.Solver(sri, m), std.StandardSolver(sm, m)
        timing[m] = {'sri_solver_construction_incl_cholesky': bench(lambda: B.Solver(sri, m)),
                     'sri_solve_batch': bench(lambda: sv.solve(f)),
                     'standard_projection_batch': bench(lambda: ss.code(f)),
                     'batch': BATCH, 'state_dim_sri': int(sv.Qyy.shape[0]), 'code_dim': sri.sizes[2]}
    out = ROOT / 'reports' / 'architecture-audit-stage1-costs.json'
    out.write_text(json.dumps(det, indent=1, sort_keys=True) + '\n')
    (ROOT / 'reports' / 'architecture-audit-stage1-costs-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment',
        'module_sha256': {p: nc.sha256(ROOT / p) for p in ('src/standard_net.py', 'experiments/standard_baseline.py',
                                                              'experiments/standard_baseline_compare.py', 'experiments/standard_baseline_costs.py')},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__,
        'numpy': np.__version__, 'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'threads': 1, 'dtype': 'float32',
        'blas': torch.__config__.show().split('BLAS_INFO=')[1].split(',')[0] if 'BLAS_INFO=' in torch.__config__.show() else None,
        'perception_microbenchmark': timing}, indent=1, sort_keys=True) + '\n')
    print(json.dumps(det['parameters'], indent=1)); print(json.dumps(timing, indent=1))


if __name__ == '__main__':
    main()
