"""Number construction v1, B1-ambiguous (docs/number-construction-v1-spec.md, "B1-ambiguous protocol").

One adjacent grounded neighbour (dots of count m = n +- 1), side unknown: the context narrows the answer to two
counts and only the item can choose. Compares immediate recognition, restricted recognition (the item's scores over
the two context-consistent candidates), energy hypothesis selection (revise under both sides, keep the lower total
energy) and the oracle side, with private-order, perturbed-successor and swapped-item controls.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_b1_ambiguous pilot
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

from experiments import number_construction_b1 as B

X, nc, ncore, ns, net = B.X, B.nc, B.ncore, B.ns, B.net
ROOT = X.ROOT
NAME = f'number-construction-v1-b1-ambiguous-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX = ROOT / 'reports' / NAME
PER, MUS, CHECKPOINTS = B.PER, B.MUS, B.CHECKPOINTS
ITEMS, CAND, CTX = {}, {}, {}


def side(n, r, i):
    if n <= 2:
        return 1
    if n >= 8:
        return -1
    return 1 if nc.stream('b1a_side', r, n, i).random() < .5 else -1


def render_item(args):
    r, n, i = args
    order = nc.stream('b1a_order', r, n, i).permutation(9)
    cand = np.stack([ns.render_dots(B.COUNTS[k], nc.stream('b1a_cand', r, n, i, int(k)), mode='area') for k in order])
    m = n + side(n, r, i)
    return (r, n, i), cand, np.array([B.COUNTS[k] for k in order]), m, ns.render_dots(m, nc.stream('b1a_ctx', r, n, i), mode='area')


def prepare(sources):
    for r in sources:
        y = X.DATA[(r, 'eval')][1]
        for n in X.numbers(r):
            ITEMS[(r, n)] = nc.stream('b1a_items', r, n).permutation(np.flatnonzero(y == n))[:PER]
    jobs = [(r, n, i) for r in sources for n in X.numbers(r) for i in range(PER)]
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        for key, cand, counts, m, img in pool.imap_unordered(render_item, jobs, chunksize=64):
            CAND[key] = (cand, counts)
            CTX[key] = (m, img)


def total_energy(sv, f, y, mu, A, t, cm, h):
    E = .5 * ((y @ sv.Qyy) * y).sum(1) + ((f @ sv.Qyx.T) * y).sum(1) + .5 * ((f @ sv.Qxx) * f).sum(1)
    c = y[:, sv.inner]
    rel = c - (cm @ A.T + t) if h > 0 else cm - (c @ A.T + t)
    return E + .5 * mu * rel.pow(2).sum(1)


def run(name):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(0)
    enc, sri = X.make_model(X.SEED)
    ck = torch.load(ROOT / CHECKPOINTS[name], weights_only=False)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); sri.load_state_dict(ck['sri'])
    enc['image'].eval(); enc['sequence'].eval()
    sources, labelled = ck['sources'], {r: tuple(v) for r, v in ck['labelled'].items()}
    A_true, t_true = sri.A.detach(), sri.t.detach()
    R, _ = torch.linalg.qr(torch.from_numpy(nc.stream('b1_rotation', name).normal(size=A_true.shape)).float())
    succ = {'true': (A_true, t_true), 'perturbed': (R @ A_true, R @ t_true)}
    out = {k: [] for k in ('rep', 'num', 'half', 'm', 'nswap', 'medium', 'immediate', 'restricted', 'restricted_swap')}
    sel = {}  # (successor, mu, kind) -> list of arrays; kind in energy / oracle / swap
    with torch.no_grad():
        solvers = {m: B.Solver(sri, m) for m in ('image', 'sequence')}

        def states(med, x):
            f = enc[med](torch.from_numpy(np.ascontiguousarray(x)))
            return f, solvers[med].solve(f)

        def scores(c, cc):
            return sri.relation_logits(c[:, None, :].expand_as(cc).reshape(-1, cc.shape[-1]), cc.reshape(-1, cc.shape[-1]), 1).reshape(cc.shape[:2])

        def pick9(c, cc, counts):
            return counts[np.arange(len(counts)), scores(c, cc).argmax(1).numpy()]

        def pick2(c, cc, counts, m):
            s = scores(c, cc).numpy()
            lo = s[np.arange(len(m)), [list(row).index(k) for row, k in zip(counts, m - 1)]]
            hi = s[np.arange(len(m)), [list(row).index(k) for row, k in zip(counts, m + 1)]]
            return np.where(hi > lo, m + 1, m - 1)

        def select(sv, f, cm, mu, A, t, tag):
            ys = {h: sv.revise(f, mu, A, t, cm if h > 0 else [None] * len(f), [None] * len(f) if h > 0 else cm, (tag, h))
                  for h in (1, -1)}
            E = {h: total_energy(sv, f, ys[h], mu, A, t, cm, h) for h in (1, -1)}
            choose_up = E[1] <= E[-1]
            return torch.where(choose_up[:, None], ys[1], ys[-1]), choose_up.numpy()

        for r in sources:
            med = X.MEDIUM[r]
            sv = solvers[med]
            for n in X.numbers(r):
                f, y = states(med, X.DATA[(r, 'eval')][0][ITEMS[(r, n)]])
                cand = np.stack([CAND[(r, n, i)][0] for i in range(PER)])
                counts = np.stack([CAND[(r, n, i)][1] for i in range(PER)])
                cc = sri.split(states('image', cand.reshape(-1, 28, 28))[1])['core_inner'].reshape(PER, 9, -1)
                m = np.array([CTX[(r, n, i)][0] for i in range(PER)])
                cm = sri.split(states('image', np.stack([CTX[(r, n, i)][1] for i in range(PER)]))[1])['core_inner']
                n_swap = int(nc.stream('b1a_swap', r, n).choice([k for k in X.numbers(r) if k != n]))
                f_s, y_s = states(med, X.DATA[(r, 'eval')][0][ITEMS[(r, n_swap)]])
                c_item, c_swap = sri.split(y)['core_inner'], sri.split(y_s)['core_inner']
                out['rep'] += [r] * PER; out['num'] += [n] * PER; out['m'] += list(m); out['nswap'] += [n_swap] * PER
                out['half'] += [0] * (PER // 2) + [1] * (PER - PER // 2); out['medium'] += [med] * PER
                out['immediate'].append(pick9(c_item, cc, counts))
                out['restricted'].append(pick2(c_item, cc, counts, m))
                out['restricted_swap'].append(pick2(c_swap, cc, counts, m))
                true_side = np.sign(n - m)
                for sname, (A, t) in succ.items():
                    for mu in MUS:
                        yr, up = select(sv, f, cm, mu, A, t, sname)
                        sel.setdefault((sname, mu, 'energy'), []).append(pick9(sri.split(yr)['core_inner'], cc, counts))
                        sel.setdefault((sname, mu, 'side'), []).append(np.where(up, m + 1, m - 1))
                        if sname == 'true':
                            yo = torch.stack([sv.revise(f[i:i + 1], mu, A, t, cm[i:i + 1] if true_side[i] > 0 else [None],
                                                        [None] if true_side[i] > 0 else cm[i:i + 1], (sname, int(true_side[i])))[0]
                                              for i in range(PER)])
                            sel.setdefault((sname, mu, 'oracle'), []).append(pick9(sri.split(yo)['core_inner'], cc, counts))
                            ysw, _ = select(sv, f_s, cm, mu, A, t, sname)
                            sel.setdefault((sname, mu, 'swap'), []).append(pick9(sri.split(ysw)['core_inner'], cc, counts))
    rep, num, half, m, nswap = (np.array(out[k]) for k in ('rep', 'num', 'half', 'm', 'nswap'))
    medium = np.array(out['medium'])
    imm, res2, res2s = (np.concatenate(out[k]) for k in ('immediate', 'restricted', 'restricted_swap'))
    withheld = np.array([n not in labelled[r] for r, n in zip(rep, num)])
    notdots = rep != 'dots'
    groups = {'all': notdots, 'withheld': notdots & withheld, 'labelled': notdots & ~withheld,
              'image': notdots & (medium == 'image'), 'sequence': notdots & (medium == 'sequence')}
    acc = lambda pick, mask: float((pick == num)[mask].mean())
    picks = {k: np.concatenate(v) for k, v in sel.items()}
    choose = {}
    for sname in succ:
        a = {mu: acc(picks[(sname, mu, 'energy')], notdots & (half == 0)) for mu in MUS}
        choose[sname] = max(MUS, key=lambda u: (a[u], -u))
    res = {'selected_mu': choose, 'groups': {}, 'mu_sweep': {}, 'swapped': {}}
    for g, mask in groups.items():
        b = mask & (half == 1)
        if not b.any():
            continue
        mu = choose['true']
        e = acc(picks[('true', mu, 'energy')], b)
        res['groups'][g] = {
            'n': int(b.sum()), 'immediate': acc(imm, b), 'context_only': 0.5, 'restricted': acc(res2, b),
            'energy': e, 'energy_side_choice': acc(picks[('true', mu, 'side')], b), 'oracle_side': acc(picks[('true', mu, 'oracle')], b),
            'energy_perturbed': acc(picks[('perturbed', choose['perturbed'], 'energy')], b),
            'integration': e - max(acc(res2, b), 0.5)}
    b = groups['all'] & (half == 1)
    other = 2 * m - num  # the other context-consistent count
    for mu in MUS:
        res['mu_sweep'][str(mu)] = {'energy': acc(picks[('true', mu, 'energy')], b), 'energy_side_choice': acc(picks[('true', mu, 'side')], b),
                                    'oracle_side': acc(picks[('true', mu, 'oracle')], b),
                                    'energy_perturbed': acc(picks[('perturbed', mu, 'energy')], b)}
        p = picks[('true', mu, 'swap')]
        res['swapped'][str(mu)] = {'true_side': float((p == num)[b].mean()), 'other_side': float((p == other)[b].mean()),
                                   'follows_item': float((p == nswap)[b].mean())}
    res['swapped_restricted'] = {'true_side': float((res2s == num)[b].mean()), 'other_side': float((res2s == other)[b].mean())}
    return name, res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                       'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       'factorizations': {k: len(v.cache) + 1 for k, v in solvers.items()}}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    t = time.perf_counter()
    union = sorted({s for p in CHECKPOINTS.values() for s in torch.load(ROOT / p, weights_only=False)['sources']})
    prepare(union)
    phases = {'data_wall_seconds': t - wall0, 'render_wall_seconds': time.perf_counter() - t}
    t = time.perf_counter()
    done = []
    with mp.get_context('fork').Pool(min(X.WORKERS, len(CHECKPOINTS))) as pool:
        for item in pool.imap_unordered(run, list(CHECKPOINTS)):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(CHECKPOINTS)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['b1a_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(B.SPEC.relative_to(ROOT)), 'status': 'B1-ambiguous on frozen pilot checkpoints: not headline evidence',
           'config': {'checkpoints': CHECKPOINTS, 'per_number': PER, 'mus': MUS, 'context': 'one adjacent fixed-area dots image, side unknown, m in 2-8'},
           'checkpoint_sha256': {k: nc.sha256(ROOT / p) for k, p in CHECKPOINTS.items()},
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(B.SPEC), 'runner_sha256': nc.sha256(__file__), 'b1_sha256': nc.sha256(B.__file__),
        'experiment_sha256': nc.sha256(X.__file__), 'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
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
