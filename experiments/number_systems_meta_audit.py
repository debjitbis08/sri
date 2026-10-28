"""Gradient audit through the selected inner procedure (docs/number-systems-meta-v1-spec.md, "Decision after the
episode audit"). Development representations only.

Same measurements as experiments/number_systems_meta_gradient.py, with the inner loop replaced by the procedure
the episode audit selected: Adam at 1e-3 on the whole image satellite for 25 updates, unrolled differentiably.
Fresh-episode scores add support and query accuracy before and after adaptation, and separate the outer step's
own effect on source accuracy from forgetting during adaptation.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_audit pilot
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
from torch.func import functional_call

from experiments import number_systems_meta_gradient as G

X, M, nc, ncore, ns, net = G.X, G.M, G.nc, G.ncore, G.ns, G.net
ROOT = X.ROOT
PREFIX = ROOT / 'reports' / f'{M.NAME}-audit'
CHECKPOINTS = ('ordinary', 'meta')
LR_INNER, B = 1e-3, (25 if not X.SMOKE else 2)
BETAS, ADAM_EPS, SQRT_FLOOR = (0.9, 0.999), 1e-8, 1e-16
ETAS = (0.003, 0.01, 0.03) if not X.SMOKE else (0.01,)
G.B = B  # make_episode draws B inner batches


def objective(model, phi, theta, ep, create_graph):
    """J = L_query + lambda * L_preservation after B differentiable Adam steps on the satellite."""
    p = dict(phi)
    m = {k: torch.zeros_like(v) for k, v in p.items()}
    s = {k: torch.zeros_like(v) for k, v in p.items()}
    b1, b2 = BETAS
    for t, (x, tasks) in enumerate(ep['inner'], 1):
        loss = functional_call(model, {**p, **theta}, (x, [tasks], [len(x)]))[0]
        g = torch.autograd.grad(loss, list(p.values()), create_graph=create_graph)
        new = {}
        for (k, v), gk in zip(p.items(), g):
            m[k] = b1 * m[k] + (1 - b1) * gk
            s[k] = b2 * s[k] + (1 - b2) * gk * gk
            mh, sh = m[k] / (1 - b1 ** t), s[k] / (1 - b2 ** t)
            new[k] = v - LR_INNER * mh / (torch.sqrt(sh + SQRT_FLOOR) + ADAM_EPS)
        p = new
    if not create_graph:
        p = {k: v.detach().requires_grad_() for k, v in p.items()}
    x, ts, sizes = ep['outer']
    losses = functional_call(model, {**p, **theta}, (x, ts, sizes))
    new_l, keep = losses[0], sum(losses[1:])
    return new_l + M.LAMBDA * keep, new_l, keep, p


def gradients(model, phi, theta, ep):
    phi = {k: v.clone().requires_grad_() for k, v in phi.items()}
    theta = {k: v.clone().requires_grad_() for k, v in theta.items()}
    t0 = time.process_time()
    J, _, _, _ = objective(model, phi, theta, ep, True)
    g = torch.autograd.grad(J, list(phi.values()) + list(theta.values()))
    full_cpu = time.process_time() - t0
    full = dict(zip(list(phi) + list(theta), g))
    t0 = time.process_time()
    J1, _, _, p = objective(model, phi, theta, ep, False)
    g = torch.autograd.grad(J1, list(p.values()) + list(theta.values()))
    fo_cpu = time.process_time() - t0
    fo = dict(zip(list(phi) + list(theta), g))
    return float(J.detach()), full, fo, {'full_cpu_seconds': full_cpu, 'first_order_cpu_seconds': fo_cpu}


def value(model, phi, theta, ep):
    try:
        with torch.enable_grad():
            phi = {k: v.clone().requires_grad_() for k, v in phi.items()}
            J, new, keep, _ = objective(model, phi, theta, ep, False)
    except torch.linalg.LinAlgError:
        return None
    return float(J) if torch.isfinite(J) else None


def support_accuracy(model, params, ep):
    d = ep['dev']
    sup = np.concatenate(list(ep['support'].values()))
    x = torch.from_numpy(X.DATA[(d, 'train')][0][sup]).to(G.DTYPE)
    with torch.no_grad():
        st = functional_call(G.StatesView(model), {f'm.{k}': v for k, v in params.items()}, (x,))
    pred = (st['output_outer'] @ model.codes.T).argmax(1).numpy()
    return float((pred == X.DATA[(d, 'train')][1][sup]).mean())


def adapt_and_score(model, phi, theta, ep, sets, labelled):
    try:
        with torch.enable_grad():
            p = {k: v.clone().requires_grad_() for k, v in phi.items()}
            J, new, keep, adapted = objective(model, p, theta, ep, False)
    except torch.linalg.LinAlgError:
        return None
    if not torch.isfinite(J):
        return None
    after = {**{k: v.detach() for k, v in adapted.items()}, **theta}
    before = {**phi, **theta}
    d = ep['dev']
    src = lambda prm: float(np.mean([G.label_accuracy(model, prm, sets, s, labelled[s]) for s in M.SOURCES if X.MEDIUM[s] == 'image']))
    sb, sa = src(before), src(after)
    return {'J': float(J), 'query_loss': float(new), 'preservation_loss': float(keep),
            'support_before': support_accuracy(model, before, ep), 'support_after': support_accuracy(model, after, ep),
            'query_before': G.label_accuracy(model, before, sets, d, X.numbers(d)),
            'query_after': G.label_accuracy(model, after, sets, d, X.numbers(d)),
            'sources_before': sb, 'sources_after': sa, 'forgetting': sb - sa}


def run(job):
    name, d, i = job
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(0)
    model, phi, theta, labelled = G.load(name)
    model.train()
    ep = G.make_episode(name, d, i, labelled, 'meta_audit_episode')
    J, full, fo, cpu = gradients(model, phi, theta, ep)
    groups = {'satellite': list(phi), 'shared': list(theta), 'all': list(phi) + list(theta)}
    cmp = {}
    for gname, keys in groups.items():
        a, b = G.flat(full, keys), G.flat(fo, keys)
        cmp[gname] = {'cosine': float(a @ b / (a.norm() * b.norm() + 1e-300)), 'norm_full': float(a.norm()),
                      'norm_first_order': float(b.norm()), 'relative_difference': float((a - b).norm() / (a.norm() + 1e-300))}
    keys = groups['all']
    gen = torch.Generator().manual_seed(i * 100 + len(d))
    dirs = {}
    for j in range(G.RANDOM_DIRECTIONS):
        v = {k: torch.randn(full[k].shape, generator=gen, dtype=G.DTYPE) for k in keys}
        n = G.flat(v, keys).norm()
        dirs[f'random{j}'] = {k: t / n for k, t in v.items()}
    nf, no = G.flat(full, keys).norm(), G.flat(fo, keys).norm()
    dirs['full'] = {k: -full[k] / nf for k in keys}
    dirs['first_order'] = {k: -fo[k] / no for k in keys}
    fd = {}
    for dname, v in dirs.items():
        vp, vt = {k: v[k] for k in phi}, {k: v[k] for k in theta}
        num = {}
        for eps in G.FD_EPS:
            jp = value(model, G.shift(phi, vp, eps), G.shift(theta, vt, eps), ep)
            jm = value(model, G.shift(phi, vp, -eps), G.shift(theta, vt, -eps), ep)
            num[str(eps)] = None if jp is None or jm is None else (jp - jm) / (2 * eps)
        fd[dname] = {'finite_difference': num, 'full': float(sum((full[k] * v[k]).sum() for k in keys)),
                     'first_order': float(sum((fo[k] * v[k]).sum() for k in keys))}
    sets = G.eval_sets(labelled)
    fresh = [G.make_episode(name, dd, j, labelled, 'meta_audit_fresh') for dd in M.DEV for j in range(G.FRESH_PER_DEV)]
    steps = {}
    for dname in ('full', 'first_order'):
        for eta in (0.0, *ETAS):
            if eta == 0.0 and dname == 'first_order':
                continue
            v = dirs[dname]
            p1, t1 = G.shift(phi, {k: v[k] for k in phi}, eta), G.shift(theta, {k: v[k] for k in theta}, eta)
            scores = [adapt_and_score(model, p1, t1, f, sets, labelled) for f in fresh]
            ok = [s for s in scores if s is not None]
            mean = {k: float(np.mean([s[k] for s in ok])) for k in ok[0]} if ok else None
            steps['baseline' if eta == 0.0 else f'{dname}/{eta}'] = {'same_episode_J': value(model, p1, t1, ep), 'fresh': mean,
                                                                     'fresh_diverged': len(scores) - len(ok)}
    return (name, d, i), {'J': J, 'gradient_comparison': cmp, 'finite_difference': fd, 'outer_steps': steps,
                          'preserve_sources': ep['preserve']}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **cpu,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    jobs = [(n, d, i) for n in CHECKPOINTS for d in M.DEV for i in range(G.EPISODES_PER_DEV)]
    done = []
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(jobs))) as pool:
        for item in pool.imap_unordered(run, jobs):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(jobs)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['audit_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    res = {'spec': str(M.SPEC.relative_to(ROOT)), 'status': 'development-only gradient audit: not headline evidence',
           'config': {'checkpoints': CHECKPOINTS, 'inner': {'optimizer': 'adam (unrolled)', 'lr': LR_INNER, 'steps': B,
                                                            'betas': BETAS, 'eps': ADAM_EPS, 'sqrt_floor': SQRT_FLOOR,
                                                            'parameters': 'whole image satellite'},
                      'episodes_per_dev': G.EPISODES_PER_DEV, 'fresh_per_dev': G.FRESH_PER_DEV, 'etas_unit_direction': ETAS,
                      'fd_eps': G.FD_EPS, 'dtype': 'float64', 'lambda': M.LAMBDA, 'k_shot': M.K_SHOT, 'preserve': M.PRESERVE,
                      'dev': M.DEV, 'eval_per_number': G.EVAL_PER_NUMBER},
           'checkpoint_sha256': {n: nc.sha256(M.RUNS / f'{n}.pt') for n in CHECKPOINTS},
           'results': {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(M.SPEC), 'runner_sha256': nc.sha256(__file__), 'gradient_sha256': nc.sha256(G.__file__),
        'meta_sha256': nc.sha256(M.__file__), 'experiment_sha256': nc.sha256(X.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'gradient_cpu_seconds': {'full_total': sum(r['full_cpu_seconds'] for r in worker.values()),
                                 'first_order_total': sum(r['first_order_cpu_seconds'] for r in worker.values())},
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
