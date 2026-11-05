"""Consistency critic v1 follow-up: confident errors and combination with margin (docs/consistency-critic-v1-confident.md).
Retrains v1's critic and sel-seen selector with the same code and seeds and draws the same evaluation queries.

Usage: .venv/bin/python -m experiments.consistency_critic_v1_confident            (CC_WORKERS, default 8)
       .venv/bin/python -m experiments.consistency_critic_v1_confident summary
"""
import json
import os
import platform
import resource
import sys
import time

from experiments import consistency_critic_v1 as V1
from experiments import durable as DUR
import numpy as np
import torch

DG, RP, EQ, X, nc = V1.DG, V1.RP, V1.EQ, V1.X, V1.nc
C, KINDS, SHORT, SEEN = V1.C, V1.KINDS, V1.SHORT, V1.SEEN
NAME = 'consistency-critic-v1-confident'
CONTRACT = X.ROOT / 'docs' / 'consistency-critic-v1-confident.md'
REPORT = X.ROOT / 'reports' / NAME
RUNS = X.ROOT / 'runs' / NAME
BUDGET, ABS_T = .10, 2.0
auroc, topk_present = V1.auroc, V1.topk_present


@torch.enable_grad()
def fit_logistic(F_, y, steps=500):
    mu, sd = F_.mean(0), F_.std(0).clamp_min(1e-6)
    w = torch.zeros(F_.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps)
    Z = (F_ - mu) / sd

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y.float())
        loss.backward()
        return loss

    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda G: ((G - mu) / sd) @ w + b, {'w': w.tolist(), 'b': float(b)}


def features(critic, b, present):
    p = torch.sigmoid(critic(b['q'], b['vals'], b['mask'], b['r'], b['typ'], present))
    o = (b['reader_logit'] > 0).float()
    conf = torch.where(o > 0, 1 - p, p).clamp(1e-6, 1 - 1e-6)
    return conf, -b['reader_logit'].abs()


@torch.no_grad()
def evaluate(T, critic, sel, seed):
    R = T['R']
    # fitting queries for the combination: collections, separate generator, teaching labels
    gf = torch.Generator().manual_seed(seed + 7)
    combos, out = {}, {}
    for qi in range(R):
        b = V1.sample(T, (C, 'test'), V1.EVAL_Q, gf, ops_from=qi)
        conf, mg = features(critic, b, torch.ones(b['vals'].shape[:2], dtype=torch.bool))
        err = (b['reader_logit'] > 0).float() != b['truth']
        combos[qi], out[f'_fit/{EQ.OPS[qi]}'] = fit_logistic(torch.stack([mg, torch.logit(conf)], 1), err)
    g = torch.Generator().manual_seed(seed + 2)  # as V1.evaluate: sample, then the random scores, per kind and operator
    for kind in KINDS:
        for qi, op in enumerate(EQ.OPS[:R]):
            b = V1.sample(T, (kind, 'test'), V1.EVAL_Q, g, ops_from=qi)
            Q, U = b['vals'].shape[:2]
            torch.rand(Q, U, generator=g)
            err = (b['reader_logit'] > 0).float() != b['truth']
            full, mg = features(critic, b, torch.ones(Q, U, dtype=torch.bool))
            k3, _ = features(critic, b, topk_present(sel(*V1.sel_inputs(T, b)), 3))
            absl = b['reader_logit'].abs()
            r = {'reproduction': {'full': auroc(full, err), 'margin': auroc(mg, err)}, 'errors': int(err.sum())}
            med = absl[err].median() if err.any() else absl.max() + 1  # labels define the scoring stratum only
            for name, strat in (('confident_half_errors', absl > med), ('top_third', absl >= absl.quantile(2 / 3)), ('abs_gt_2', absl > ABS_T)):
                e = err[strat]
                r[name] = {'n': int(strat.sum()), 'errors': int(e.sum()), 'error_rate': float(e.float().mean()) if strat.any() else None,
                           'share_of_all_errors': float(e.sum() / err.sum().clamp_min(1)),
                           'auroc_full': auroc(full[strat], e) if strat.any() else None,
                           'auroc_sel_seen_k3': auroc(k3[strat], e) if strat.any() else None,
                           'auroc_margin': auroc(mg[strat], e) if strat.any() else None}
            nb, nh = int(round(BUDGET * Q)), int(round(BUDGET * Q / 2))
            top = lambda s, n: torch.zeros(Q, dtype=torch.bool).index_fill_(0, s.topk(n).indices, True)
            m_all, c_all = top(mg, nb), top(full, nb)
            split = top(mg, nh) | top(full, nh)
            ne = err.sum().clamp_min(1)
            r['budget'] = {'margin': float((m_all & err).sum() / ne), 'critic': float((c_all & err).sum() / ne),
                           'split': float((split & err).sum() / ne), 'split_size': float(split.float().mean()),
                           'critic_half_only': float((top(full, nh) & ~top(mg, nh) & err).sum() / ne)}
            fn = combos[qi]
            r['fitted'] = {'combined': auroc(fn(torch.stack([mg, torch.logit(full)], 1)), err), 'margin': auroc(mg, err)}
            out[f'{SHORT[kind]}/{op}'] = r
    return out


def run(t):
    arm, seed, block = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    T = V1.tables(DG.load(arm, seed, block), block)
    critic, _ = V1.train_critic(T, seed)
    V1.train_selector(T, critic, [(C, 'self')], seed)  # v1 trained sel-coll first; kept so the RNG state matches
    sel, _ = V1.train_selector(T, critic, [(C, 'self')] + [(k, 'fit') for k in SEEN], seed)
    tt = (time.perf_counter() - t0, time.process_time() - c0)
    res = evaluate(T, critic, sel, seed)
    rt = {'train': {'wall_seconds': tt[0], 'cpu_seconds': tt[1]},
          'evaluate': {'wall_seconds': time.perf_counter() - t0 - tt[0], 'cpu_seconds': time.process_time() - c0 - tt[1]},
          'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return '{}/seed{}/{}'.format(*t), res, rt


def reproduction(res):
    v1 = json.loads((X.ROOT / 'reports' / 'consistency-critic-v1.json').read_text())['results']
    bad = []
    for key, r in res.items():
        for cell, c in r.items():
            if cell.startswith('_'):
                continue
            a = v1[key][cell]['auroc']
            if abs(a['full'] - c['reproduction']['full']) > 1e-9 or abs(a['margin'] - c['reproduction']['margin']) > 1e-9:
                bad.append(f'{key}/{cell}')
    return {'equal_to_v1': not bad, 'differing_cells': bad}


def main():
    t0 = time.perf_counter()
    RP.prepare()
    tasks = [(a, s, b) for a in V1.ARMS for s in V1.SEEDS for b in V1.BLOCKS]
    workers = int(os.environ.get('CC_WORKERS', 8))
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(run, RUNS / 'done', '{0}-seed{1}-{2}'), tasks, chunksize=1)
    results = {k: r for k, r, _ in res}
    RP.S0.save_json(f'{REPORT}.json', {'status': 'consistency critic v1: confident errors and margin combination',
                                        'budget': BUDGET, 'abs_threshold': ABS_T, 'reproduction': reproduction(results),
                                        'results': results})
    RP.S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
        'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(),
        'workers': workers, 'threads_per_worker': 1, 'dtype': 'float32', 'device': 'cpu', 'wall_seconds': time.perf_counter() - t0,
        'aggregate_worker_cpu_seconds': float(sum(sum(p['cpu_seconds'] for p in t.values() if isinstance(p, dict)) for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary():
    rep = json.loads(REPORT.with_suffix('.json').read_text())
    print('reproduction:', rep['reproduction']['equal_to_v1'], len(rep['reproduction']['differing_cells']))
    res = rep['results']
    f = lambda v: f'{np.mean(v):.2f}' if v else '-'
    for block in V1.BLOCKS:
        ops = EQ.OPS[:3 if block == 'block3' else 2]
        for op in ops:
            print(f'\n## {block} {op}: confident half of errors: n errors | auroc full/k3/margin ## top third errors(share) | auroc full/k3/margin || abs>2 errors(share) full/margin || '
                  'budget margin/critic/split (critic-only) || fitted vs margin')
            for arm in V1.ARMS:
                ks = [k for k in res if k.startswith(arm + '/') and k.endswith(block)]
                cells = []
                for kd in ('coll', 'zh', 'dev', 'en', 'thai'):
                    rs = [res[k][f'{kd}/{op}'] for k in ks]
                    g = lambda *p: f([v for v in (V1._dig(x, p) for x in rs) if v is not None])
                    cells.append(f"{kd} {g('confident_half_errors', 'errors')} {g('confident_half_errors', 'auroc_full')}/"
                                 f"{g('confident_half_errors', 'auroc_sel_seen_k3')}/{g('confident_half_errors', 'auroc_margin')} ## {g('top_third', 'errors')}({g('top_third', 'share_of_all_errors')}) "
                                 f"{g('top_third', 'auroc_full')}/{g('top_third', 'auroc_sel_seen_k3')}/{g('top_third', 'auroc_margin')} || "
                                 f"{g('abs_gt_2', 'errors')}({g('abs_gt_2', 'share_of_all_errors')}) {g('abs_gt_2', 'auroc_full')}/{g('abs_gt_2', 'auroc_margin')} || "
                                 f"{g('budget', 'margin')}/{g('budget', 'critic')}/{g('budget', 'split')} ({g('budget', 'critic_half_only')}) || "
                                 f"{g('fitted', 'combined')}/{g('fitted', 'margin')}")
                print(f'### {arm}\n' + '\n'.join(cells))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
