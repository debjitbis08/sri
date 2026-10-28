"""Gradient diagnostic for number systems meta v1 (development representations only).

The pilot's episode (experiments/number_systems_meta.py) is first-order: the outer gradient is taken at the
adapted satellite phi' and applied as if phi' did not depend on the shared parameters. With theta the
parameters outside the satellite and phi' = U(phi, theta) the adapted satellite,

    dJ/dtheta = dJ/dtheta|phi' + (dphi'/dtheta)^T dJ/dphi'        (and likewise for the initial phi)

and the first-order estimate keeps only the first term: it changes the core so the already-adapted satellite
does better, not so the satellite adapts differently. This diagnostic, on the pilot's checkpoints and only the
development representations:

1. computes the exact gradient by differentiating through a short inner episode (plain SGD, so the unrolled
   update is differentiable) and the first-order estimate on the same episode;
2. checks both against central finite differences of the episode objective (float64). With a ReLU front end the
   objective after any inner step is piecewise smooth with jumps (kinks change the inner gradient abruptly), so
   finite differences agree with the exact gradient only at very small steps; several are reported;
3. takes small outer steps along each and measures, on fresh development episodes, the query error after
   adaptation and the forgetting of the affected (image) sources.

The inner optimizer differs from the pilot's (Adam at 1e-2); that difference is reported, not hidden.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta_gradient pilot
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

from experiments import number_systems as X
from experiments import number_systems_meta as M

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
PREFIX = ROOT / 'reports' / f'{M.NAME}-gradient'
CHECKPOINTS = ('meta', 'ordinary')
ALPHAS = (0.003, 0.01) if not X.SMOKE else (0.01,)  # inner SGD learning rates
B = 5 if not X.SMOKE else 2
EPISODES_PER_DEV = 2 if not X.SMOKE else 1
FRESH_PER_DEV = 2 if not X.SMOKE else 1
ETAS = (0.01, 0.03, 0.1) if not X.SMOKE else (0.03,)  # outer step lengths along unit-norm directions
FD_EPS = (1e-8, 1e-7, 1e-5)  # the episode objective is piecewise smooth with jumps (ReLU kinks moved by inner steps)
RANDOM_DIRECTIONS = 3 if not X.SMOKE else 1
EVAL_PER_NUMBER = 20 if not X.SMOKE else 5
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)  # the SRI Hessian builds its matrices in the default dtype


class Net(torch.nn.Module):
    """Image front end and the SRI network, with a functional loss over several task batches."""

    def __init__(self, image, sri, codes):
        super().__init__()
        self.image, self.sri, self.codes = image, sri, codes

    def forward(self, x, tasks, sizes):
        st = self.sri.split(self.image(x) @ self.sri.equilibrium_map('image')[0].T)
        out, start = [], 0
        for t, n in zip(tasks, sizes):
            out.append(X.batch_loss(self.sri, self.codes, {k: v[start:start + n] for k, v in st.items()}, t))
            start += n
        return out

    def states(self, x):
        return self.sri.split(self.image(x) @ self.sri.equilibrium_map('image')[0].T)


def load(name):
    enc, sri = X.make_model(X.SEED)
    ck = torch.load(M.RUNS / f'{name}.pt', weights_only=False)
    enc['image'].load_state_dict(ck['image']); sri.load_state_dict(ck['sri'])
    model = Net(enc['image'], sri, ncore.label_codes(net.SIZES[0]).to(DTYPE)).to(DTYPE)
    phi = {k: v.detach().clone() for k, v in model.named_parameters()
           if k.startswith('image.') or k in {'sri.W.image', 'sri.V.image'} or k.startswith('sri.T.image-core-')}
    theta = {k: v.detach().clone() for k, v in model.named_parameters()
             if k not in phi and not k.startswith('sri.T.sequence-') and k not in {'sri.W.sequence', 'sri.V.sequence'}}
    labelled = {r: tuple(v) for r, v in ck['labelled'].items()}
    return model, phi, theta, labelled


def batch(r, req):
    return torch.from_numpy(X.DATA[(r, 'train')][0][req]).to(DTYPE)


def make_episode(name, d, i, labelled, key):
    """Fixed batches: B inner batches on the support set, then query and preservation batches."""
    rng = nc.stream(key, name, d, i)
    idx = {c: np.flatnonzero(X.DATA[(d, 'train')][1] == c) for c in X.numbers(d)}
    support = {c: rng.choice(v, M.K_SHOT, replace=False) for c, v in idx.items()}
    query = {c: np.setdiff1d(v, support[c]) for c, v in idx.items()}
    nums, order = tuple(X.numbers(d)), {n: n for n in X.numbers(d)}
    inner = []
    for _ in range(B):
        req, tasks = X.task_batch(rng, d, support, nums, order)
        inner.append((batch(d, req), tasks))
    req, tasks = X.task_batch(rng, d, query, nums, order)
    xs, ts, sizes = [batch(d, req)], [tasks], [len(req)]
    images = [s for s in M.SOURCES if X.MEDIUM[s] == 'image']
    pres = rng.choice(images, M.PRESERVE, replace=False).tolist()
    for s in pres:
        sidx = {c: np.flatnonzero(X.DATA[(s, 'train')][1] == c) for c in X.numbers(s)}
        req, tasks = X.task_batch(rng, s, sidx, labelled[s], {n: n for n in X.numbers(s)})
        xs.append(batch(s, req)); ts.append(tasks); sizes.append(len(req))
    return {'dev': d, 'inner': inner, 'outer': (torch.cat(xs), ts, sizes), 'preserve': pres, 'support': support}


def objective(model, phi, theta, ep, alpha, create_graph):
    """Episode objective J = L_query + lambda * L_preservation after B inner SGD steps on the satellite."""
    p = dict(phi)
    for x, tasks in ep['inner']:
        loss = functional_call(model, {**p, **theta}, (x, [tasks], [len(x)]))[0]
        g = torch.autograd.grad(loss, list(p.values()), create_graph=create_graph)
        p = {k: v - alpha * gk for (k, v), gk in zip(p.items(), g)}
    if not create_graph:
        p = {k: v.detach().requires_grad_() for k, v in p.items()}
    x, ts, sizes = ep['outer']
    losses = functional_call(model, {**p, **theta}, (x, ts, sizes))
    new, keep = losses[0], sum(losses[1:])
    return new + M.LAMBDA * keep, new, keep, p


def gradients(model, phi, theta, ep, alpha):
    phi = {k: v.clone().requires_grad_() for k, v in phi.items()}
    theta = {k: v.clone().requires_grad_() for k, v in theta.items()}
    t0 = time.process_time()
    J, _, _, _ = objective(model, phi, theta, ep, alpha, True)
    g = torch.autograd.grad(J, list(phi.values()) + list(theta.values()))
    full_cpu = time.process_time() - t0
    full = dict(zip(list(phi) + list(theta), g))
    t0 = time.process_time()
    J1, _, _, p = objective(model, phi, theta, ep, alpha, False)
    g = torch.autograd.grad(J1, list(p.values()) + list(theta.values()))
    fo_cpu = time.process_time() - t0
    fo = dict(zip(list(phi) + list(theta), g))  # the phi' gradient is applied to the initial phi, as in the pilot
    return float(J.detach()), full, fo, {'full_cpu_seconds': full_cpu, 'first_order_cpu_seconds': fo_cpu}


def flat(g, keys):
    return torch.cat([g[k].reshape(-1) for k in keys])


def value(model, phi, theta, ep, alpha):
    with torch.enable_grad():
        phi = {k: v.clone().requires_grad_() for k, v in phi.items()}
        J, new, keep, _ = objective(model, phi, theta, ep, alpha, False)
    return float(J), float(new), float(keep)


def shift(params, direction, eps):
    return {k: v + eps * direction[k] for k, v in params.items()}


def eval_sets(labelled):
    sets = {}
    for r in [*M.DEV, *[s for s in M.SOURCES if X.MEDIUM[s] == 'image']]:
        y = X.DATA[(r, 'eval')][1]
        sel = np.concatenate([nc.stream('meta_grad_eval', r, int(c)).permutation(np.flatnonzero(y == c))[:EVAL_PER_NUMBER]
                              for c in X.numbers(r)])
        sets[r] = (torch.from_numpy(X.DATA[(r, 'eval')][0][sel]).to(DTYPE), y[sel])
    return sets


def label_accuracy(model, params, sets, r, numbers):
    x, y = sets[r]
    with torch.no_grad():
        # states() is not forward, so swap parameters with functional_call on a bound wrapper.
        st = functional_call(StatesView(model), {f'm.{k}': v for k, v in params.items()}, (x,))
    pred = (st['output_outer'] @ model.codes.T).argmax(1).numpy()
    keep = np.isin(y, numbers)
    return float((pred[keep] == y[keep]).mean())


class StatesView(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.states(x)


def adapt_and_score(model, phi, theta, ep, alpha, sets, labelled):
    """Run the episode's inner adaptation (no graph) and score query accuracy and affected-source forgetting.

    Returns None when the adaptation diverges (non-finite parameters make the factorization fail)."""
    try:
        with torch.enable_grad():
            p = {k: v.clone().requires_grad_() for k, v in phi.items()}
            J, new, keep, adapted = objective(model, p, theta, ep, alpha, False)
    except torch.linalg.LinAlgError:
        return None
    if not torch.isfinite(J):
        return None
    images = [s for s in M.SOURCES if X.MEDIUM[s] == 'image']
    before = {s: label_accuracy(model, {**phi, **theta}, sets, s, labelled[s]) for s in images}
    after = {s: label_accuracy(model, {**{k: v.detach() for k, v in adapted.items()}, **theta}, sets, s, labelled[s]) for s in images}
    d = ep['dev']
    return {'J': float(J), 'query_loss': float(new), 'preservation_loss': float(keep),
            'dev_accuracy_after': label_accuracy(model, {**{k: v.detach() for k, v in adapted.items()}, **theta}, sets, d, X.numbers(d)),
            'image_source_accuracy_before': float(np.mean(list(before.values()))),
            'image_source_accuracy_after': float(np.mean(list(after.values())))}


def run(job):
    name, alpha, d, i = job
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(0)
    model, phi, theta, labelled = load(name)
    model.train()
    ep = make_episode(name, d, i, labelled, 'meta_grad_episode')
    J, full, fo, cpu = gradients(model, phi, theta, ep, alpha)
    groups = {'satellite': list(phi), 'shared': list(theta), 'all': list(phi) + list(theta)}
    cmp = {}
    for gname, keys in groups.items():
        a, b = flat(full, keys), flat(fo, keys)
        cmp[gname] = {'cosine': float(a @ b / (a.norm() * b.norm() + 1e-300)), 'norm_full': float(a.norm()), 'norm_first_order': float(b.norm()),
                      'relative_difference': float((a - b).norm() / (a.norm() + 1e-300))}
    # Finite-difference checks of directional derivatives.
    keys = groups['all']
    gen = torch.Generator().manual_seed(i * 100 + len(d))
    dirs = {f'random{j}': {k: torch.randn(full[k].shape, generator=gen, dtype=DTYPE) for k in keys} for j in range(RANDOM_DIRECTIONS)}
    nf, no = flat(full, keys).norm(), flat(fo, keys).norm()
    dirs['full'] = {k: -full[k] / nf for k in keys}
    dirs['first_order'] = {k: -fo[k] / no for k in keys}
    for k_, v in list(dirs.items()):
        if k_.startswith('random'):
            n = flat(v, keys).norm()
            dirs[k_] = {k: t / n for k, t in v.items()}
    fd = {}
    for dname, v in dirs.items():
        vp, vt = {k: v[k] for k in phi}, {k: v[k] for k in theta}
        num = {}
        for eps in FD_EPS:
            jp = value(model, shift(phi, vp, eps), shift(theta, vt, eps), ep, alpha)[0]
            jm = value(model, shift(phi, vp, -eps), shift(theta, vt, -eps), ep, alpha)[0]
            num[str(eps)] = (jp - jm) / (2 * eps)
        fd[dname] = {'finite_difference': num, 'full': float(sum((full[k] * v[k]).sum() for k in keys)),
                     'first_order': float(sum((fo[k] * v[k]).sum() for k in keys))}
    # Outer steps along each unit direction, scored on fresh development episodes.
    sets = eval_sets(labelled)
    fresh = [make_episode(name, dd, j, labelled, 'meta_grad_fresh') for dd in M.DEV for j in range(FRESH_PER_DEV)]
    steps = {}
    for dname in ('full', 'first_order'):
        for eta in (0.0, *ETAS):
            if eta == 0.0 and dname == 'first_order':
                continue
            v = dirs[dname]
            p1, t1 = shift(phi, {k: v[k] for k in phi}, eta), shift(theta, {k: v[k] for k in theta}, eta)
            try:
                same = value(model, p1, t1, ep, alpha)[0]
            except torch.linalg.LinAlgError:
                same = None
            scores = [adapt_and_score(model, p1, t1, f, alpha, sets, labelled) for f in fresh]
            ok = [s for s in scores if s is not None]
            mean = {k: float(np.mean([s[k] for s in ok])) for k in ok[0]} if ok else None
            steps['baseline' if eta == 0.0 else f'{dname}/{eta}'] = {'same_episode_J': same, 'fresh': mean,
                                                                     'fresh_diverged': len(scores) - len(ok)}
    return (name, alpha, d, i), {'J': J, 'gradient_comparison': cmp, 'finite_difference': fd, 'outer_steps': steps,
                                 'preserve_sources': ep['preserve']}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **cpu,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    jobs = [(n, a, d, i) for n in CHECKPOINTS for a in ALPHAS for d in M.DEV for i in range(EPISODES_PER_DEV)]
    done = []
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(jobs))) as pool:
        for item in pool.imap_unordered(run, jobs):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(jobs)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['diagnostic_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    res = {'spec': str(M.SPEC.relative_to(ROOT)), 'status': 'development-only diagnostic of a pilot: not headline evidence',
           'config': {'checkpoints': CHECKPOINTS, 'alphas_inner_sgd': ALPHAS, 'inner_steps': B, 'episodes_per_dev': EPISODES_PER_DEV,
                      'fresh_per_dev': FRESH_PER_DEV, 'etas_unit_direction': ETAS, 'fd_eps': FD_EPS, 'dtype': 'float64',
                      'lambda': M.LAMBDA, 'k_shot': M.K_SHOT, 'preserve': M.PRESERVE, 'dev': M.DEV,
                      'eval_per_number': EVAL_PER_NUMBER,
                      'pilot_inner_optimizer': 'Adam 1e-2, B = 10 (differs from this diagnostic)'},
           'checkpoint_sha256': {n: nc.sha256(M.RUNS / f'{n}.pt') for n in CHECKPOINTS},
           'results': {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'meta_sha256': nc.sha256(M.__file__), 'experiment_sha256': nc.sha256(X.__file__),
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
