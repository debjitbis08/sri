"""Episode audit for number systems meta v1 (docs/number-systems-meta-v1-spec.md, "Episode audit").

Development representations only. For each inner adaptation procedure (optimizer, learning rate, adapted
parameters), adapt the pilot checkpoints to a few-shot support set and score support accuracy, query accuracy
and image-source retention at several budgets, to find a procedure that learns stably before the
learning-efficiency gradient is judged.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_episode pilot
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
from experiments import number_systems_meta as M

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
PREFIX = ROOT / 'reports' / f'{M.NAME}-episode'
CHECKPOINTS = ('ordinary', 'meta')
EPISODES = 2 if not X.SMOKE else 1
OPTIMIZERS = {'adam': (1e-4, 3e-4, 1e-3, 1e-2), 'sgd': (1e-3, 1e-2, 3e-2, 1e-1)} if not X.SMOKE else {'adam': (1e-3,)}
PARTS = ('satellite', 'sri', 'front_end')
SCORE_AT = (0, 5, 10, 25, 50) if not X.SMOKE else (0, 5)
QUERY_PER_NUMBER, SOURCE_PER_NUMBER = (50, 20) if not X.SMOKE else (5, 5)
QUERY_GAIN, RETAIN = 0.15, 0.9
IMAGES = [s for s in M.SOURCES if X.MEDIUM[s] == 'image']


def load(name):
    enc, sri = X.make_model(X.SEED)
    ck = torch.load(M.RUNS / f'{name}.pt', weights_only=False)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    sri.load_state_dict(ck['sri'])
    return enc, sri, {r: tuple(v) for r, v in ck['labelled'].items()}


def parameters(enc, sri, part):
    if part == 'satellite':
        return [*enc['image'].parameters(), *sri.satellite_parameters('image')]
    return list(sri.satellite_parameters('image')) if part == 'sri' else list(enc['image'].parameters())


def subset(r, per, key):
    y = X.DATA[(r, 'eval')][1]
    return np.concatenate([nc.stream('meta_episode_eval', key, r, int(c)).permutation(np.flatnonzero(y == c))[:per] for c in X.numbers(r)])


def accuracy(st, y, codes, keep=None):
    pred = (st['output_outer'] @ codes.T).argmax(1).numpy()
    sel = np.ones(len(y), bool) if keep is None else np.isin(y, keep)
    return float((pred[sel] == y[sel]).mean())


@torch.no_grad()
def score(enc, sri, codes, d, support, q_idx, src_idx, labelled):
    sup = np.concatenate(list(support.values()))
    groups = {d: q_idx, **src_idx}
    st = X.encode(enc, sri, groups, 'eval')
    st_sup = X.encode(enc, sri, {d: sup}, 'train')[d]
    per = {s: accuracy(st[s], X.DATA[(s, 'eval')][1][src_idx[s]], codes, labelled[s]) for s in IMAGES}
    return {'support': accuracy(st_sup, X.DATA[(d, 'train')][1][sup], codes),
            'query': accuracy(st[d], X.DATA[(d, 'eval')][1][q_idx], codes),
            'sources': float(np.mean(list(per.values()))), 'sources_per': per}


def run(job):
    name, d, i, opt_name, lr, part = job
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(0)
    enc, sri, labelled = load(name)
    codes = ncore.label_codes(net.SIZES[0])
    rng = nc.stream('meta_episode', name, d, i)
    idx = {c: np.flatnonzero(X.DATA[(d, 'train')][1] == c) for c in X.numbers(d)}
    support = {c: rng.choice(v, M.K_SHOT, replace=False) for c, v in idx.items()}
    q_idx = subset(d, QUERY_PER_NUMBER, 'query')
    src_idx = {s: subset(s, SOURCE_PER_NUMBER, 'source') for s in IMAGES}
    params = parameters(enc, sri, part)
    opt = (torch.optim.Adam if opt_name == 'adam' else torch.optim.SGD)(params, lr)
    curve, diverged = [], False
    enc['image'].eval()
    curve.append({'step': 0, **score(enc, sri, codes, d, support, q_idx, src_idx, labelled)})
    enc['image'].train()
    nums, order = tuple(X.numbers(d)), {n: n for n in X.numbers(d)}
    for step in range(1, SCORE_AT[-1] + 1):
        req, tasks = X.task_batch(rng, d, support, nums, order)
        try:
            loss = X.batch_loss(sri, codes, X.encode(enc, sri, {d: req}, 'train')[d], tasks)
        except torch.linalg.LinAlgError:
            diverged = True
            break
        if not torch.isfinite(loss):
            diverged = True
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in SCORE_AT:
            enc['image'].eval()
            try:
                curve.append({'step': step, 'loss': round(loss.item(), 4), **score(enc, sri, codes, d, support, q_idx, src_idx, labelled)})
            except torch.linalg.LinAlgError:
                diverged = True
                break
            enc['image'].train()
    return (name, d, i, opt_name, lr, part), {'curve': curve, 'diverged': diverged}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def summarize(results):
    """Per checkpoint and procedure: mean gains over episodes at each budget, and the functioning rule."""
    out = {}
    procs = sorted({k.split('/', 3)[3] for k in results})
    for name in CHECKPOINTS:
        for proc in procs:
            runs = [r for k, r in results.items() if k.startswith(f'{name}/') and k.split('/', 3)[3] == proc]
            row = {'diverged': sum(r['diverged'] for r in runs), 'budgets': {}}
            for b in SCORE_AT[1:]:
                pts = [(r['curve'][0], next((p for p in r['curve'] if p['step'] == b), None)) for r in runs]
                if any(p is None for _, p in pts):
                    continue
                row['budgets'][str(b)] = {
                    'support_gain': float(np.mean([p['support'] - p0['support'] for p0, p in pts])),
                    'query_gain': float(np.mean([p['query'] - p0['query'] for p0, p in pts])),
                    'query_after': float(np.mean([p['query'] for _, p in pts])),
                    'retention': float(np.mean([p['sources'] / p0['sources'] for p0, p in pts]))}
            ok = [(int(b), v) for b, v in row['budgets'].items() if v['query_gain'] >= QUERY_GAIN and v['retention'] >= RETAIN]
            row['functioning_budgets'] = [b for b, _ in ok]
            out[f'{name}/{proc}'] = row
    best = {}
    for name in CHECKPOINTS:
        cands = [(v['budgets'][str(b)]['query_gain'], -b, k, b) for k, v in out.items() if k.startswith(f'{name}/')
                 for b in v['functioning_budgets'] if v['diverged'] == 0]
        best[name] = None if not cands else {'procedure': max(cands)[2].split('/', 1)[1], 'budget': max(cands)[3],
                                             'query_gain': max(cands)[0]}
    return {'procedures': out, 'selected': best}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    jobs = [(n, d, i, o, lr, p) for n in CHECKPOINTS for d in M.DEV for i in range(EPISODES)
            for o, lrs in OPTIMIZERS.items() for lr in lrs for p in PARTS]
    done = []
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(jobs))) as pool:
        for item in pool.imap_unordered(run, jobs):
            done.append(item)
            if len(done) % 24 == 0 or len(done) == len(jobs):
                print(f'done {len(done)}/{len(jobs)} at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['audit_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    results = {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}
    res = {'spec': str(M.SPEC.relative_to(ROOT)), 'status': 'development-only episode audit: not headline evidence',
           'config': {'checkpoints': CHECKPOINTS, 'dev': M.DEV, 'episodes': EPISODES, 'optimizers': OPTIMIZERS, 'parts': PARTS,
                      'score_at': SCORE_AT, 'k_shot': M.K_SHOT, 'query_per_number': QUERY_PER_NUMBER,
                      'source_per_number': SOURCE_PER_NUMBER, 'query_gain_rule': QUERY_GAIN, 'retain_rule': RETAIN,
                      'image_sources': IMAGES},
           'checkpoint_sha256': {n: nc.sha256(M.RUNS / f'{n}.pt') for n in CHECKPOINTS},
           'summary': summarize(results), 'results': results}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(M.SPEC), 'runner_sha256': nc.sha256(__file__), 'meta_sha256': nc.sha256(M.__file__),
        'experiment_sha256': nc.sha256(X.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
