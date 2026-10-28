"""Number construction v1, B1: do the existing relations carry usable information? (docs/number-construction-v1-spec.md,
"B1 protocol").

Frozen checkpoints from number systems v1 and the meta pilot. Each item (a source representation's evaluation image)
is answered by collection matching against 9 fresh fixed-area dots candidates. Relational revision adds successor
couplings to grounded context (dots of n - 1 and n + 1, or same-representation forms, or oracle centroids) and solves
again; a controller trained on half A decides when to revise; selectors are compared on half B.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_b1 pilot
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
import torch.nn.functional as F

from experiments import number_systems as X

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
SPEC = ROOT / 'docs/number-construction-v1-spec.md'
NAME = f'number-construction-v1-b1-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX = ROOT / 'reports' / NAME
CHECKPOINTS = {'v1/C14': 'runs/number-systems-v1-pilot/C14-3000.pt', 'v1/I14': 'runs/number-systems-v1-pilot/I14-3000.pt',
               'meta/ordinary': 'runs/number-systems-meta-v1-pilot/ordinary.pt',
               'meta/ordinary+dev': 'runs/number-systems-meta-v1-pilot/ordinary+dev.pt',
               'meta/meta': 'runs/number-systems-meta-v1-pilot/meta.pt', 'meta/meta-I': 'runs/number-systems-meta-v1-pilot/meta-I.pt'}
PER = 4 if X.SMOKE else 40  # items per number per representation; first half is A, second half B
MUS = (1., 10.) if X.SMOKE else (0.1, 0.3, 1., 3., 10., 30., 100.)
C_COST, ORACLE_PER = 0.02, (5 if X.SMOKE else 50)
COUNTS = tuple(range(1, 10))
CONTEXTS = ('collections', 'forms', 'oracle')
SIDE = {-1: 'prev', 1: 'next'}
CAND, CTX, ORACLE, ITEMS = {}, {}, {}, {}


# ---------------------------------------------------------------- rendering (parent, before fork)

def render_item(args):
    r, n, i = args
    order = nc.stream('b1_order', r, n, i).permutation(9)
    cand = np.stack([ns.render_dots(COUNTS[k], nc.stream('b1_cand', r, n, i, int(k)), mode='area') for k in order])
    ctx = {s: ns.render_dots(n + s, nc.stream('b1_ctx', r, n, i, SIDE[s]), mode='area') if 1 <= n + s <= 9 else None for s in (-1, 1)}
    return (r, n, i), cand, np.array([COUNTS[k] for k in order]), ctx


def prepare(sources):
    for r in sources:
        y = X.DATA[(r, 'eval')][1]
        for n in X.numbers(r):
            ITEMS[(r, n)] = nc.stream('b1_items', r, n).permutation(np.flatnonzero(y == n))[:PER]
            forms = {}
            for s in (-1, 1):
                pool = np.flatnonzero(y == n + s)
                forms[s] = nc.stream('b1_forms', r, n, SIDE[s]).choice(pool, PER) if len(pool) else None
            ITEMS[(r, n, 'forms')] = forms
    jobs = [(r, n, i) for r in sources for n in X.numbers(r) for i in range(PER)]
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        for key, cand, counts, ctx in pool.imap_unordered(render_item, jobs, chunksize=64):
            CAND[key] = (cand, counts)
            CTX[key] = ctx
    for k in COUNTS:
        ORACLE[k] = np.stack([ns.render_dots(k, nc.stream('b1_oracle', k, j), mode='area') for j in range(ORACLE_PER)])


# ---------------------------------------------------------------- model pieces

class Solver:
    def __init__(self, sri, med):
        Q = sri.hessian(med)
        o = sri.sizes[0]
        self.Qyy, self.Qyx, self.Qxx, self.o = Q[o:, o:], Q[o:, :o], Q[:o, :o], o
        self.L = torch.linalg.cholesky(self.Qyy)
        m, i = sri.sizes[1], sri.sizes[2]
        start = m + i + o + m  # core block starts after the satellite's middle and inner layers
        self.inner = slice(start, start + i)
        self.cache = {}

    def solve(self, f):
        return torch.cholesky_solve(-(f @ self.Qyx.T).T, self.L).T

    def energy(self, f, y):
        return .5 * ((f @ self.Qxx) * f).sum(1) + .5 * ((f @ self.Qyx.T) * y).sum(1)

    def revise(self, f, mu, A, t, prev, nxt, key):
        """Solve with 1/2 mu ||c - (A prev + t)||^2 and 1/2 mu ||nxt - (A c + t)||^2 added (rows without context: None)."""
        n = len(f)
        has_p = torch.tensor([p is not None for p in prev]) if isinstance(prev, list) else torch.ones(n, dtype=torch.bool)
        has_n = torch.tensor([q is not None for q in nxt]) if isinstance(nxt, list) else torch.ones(n, dtype=torch.bool)
        out = torch.empty(n, self.Qyy.shape[0])
        d = A.shape[0]
        P = torch.stack([p if p is not None else torch.zeros(d) for p in prev]) if isinstance(prev, list) else prev
        N = torch.stack([q if q is not None else torch.zeros(d) for q in nxt]) if isinstance(nxt, list) else nxt
        for hp in (False, True):
            for hn in (False, True):
                rows = (has_p == hp) & (has_n == hn)
                if not rows.any():
                    continue
                K = self.Qyy.clone()
                if hp:
                    K[self.inner, self.inner] += mu * torch.eye(d)
                if hn:
                    K[self.inner, self.inner] += mu * A.T @ A
                ck = (key, mu, hp, hn)
                if ck not in self.cache:
                    self.cache[ck] = torch.linalg.cholesky(K)
                rhs = -(f[rows] @ self.Qyx.T)
                b = torch.zeros(int(rows.sum()), d)
                if hp:
                    b += mu * (P[rows] @ A.T + t)
                if hn:
                    b += mu * ((N[rows] - t) @ A)
                rhs[:, self.inner] += b
                out[rows] = torch.cholesky_solve(rhs.T, self.cache[ck]).T
        return out


def auroc(score, positive):
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_controller(Xf, target, seed=0, steps=500):
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(Xf.shape[1], 3)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * .01); lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), 0.05)
    for _ in range(steps):
        opt.zero_grad()
        (F.cross_entropy(lin(Xf), target) + 1e-3 * lin.weight.pow(2).sum()).backward()
        opt.step()
    return lin


# ---------------------------------------------------------------- one checkpoint

def run(name):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(0)
    enc, sri = X.make_model(X.SEED)
    ck = torch.load(ROOT / CHECKPOINTS[name], weights_only=False)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); sri.load_state_dict(ck['sri'])
    enc['image'].eval(); enc['sequence'].eval()
    sources, labelled = ck['sources'], {r: tuple(v) for r, v in ck['labelled'].items()}
    codes = ncore.label_codes(net.SIZES[0])
    phase = {}
    with torch.no_grad():
        solvers = {m: Solver(sri, m) for m in ('image', 'sequence')}
        sp = lambda med, y: sri.split(y)

        def states(med, x):
            f = enc[med](torch.from_numpy(np.ascontiguousarray(x)))
            y = solvers[med].solve(f)
            return f, y

        # sanity: the solver reproduces the experiment's encode()
        r0 = sources[0]
        ref = X.encode(enc, sri, {r0: np.arange(5)}, 'eval')[r0]['core_inner']
        _, y0 = states(X.MEDIUM[r0], X.DATA[(r0, 'eval')][0][:5])
        sanity = float((sp(None, y0)['core_inner'] - ref).abs().max())
        e0 = time.process_time()
        oracle = {k: sp(None, states('image', ORACLE[k])[1])['core_inner'].mean(0) for k in COUNTS}
        rows = []
        for r in sources:
            med = X.MEDIUM[r]
            for n in X.numbers(r):
                idx = ITEMS[(r, n)]
                f, y = states(med, X.DATA[(r, 'eval')][0][idx])
                st = sp(None, y)
                cand = np.stack([CAND[(r, n, i)][0] for i in range(PER)])
                counts = np.stack([CAND[(r, n, i)][1] for i in range(PER)])
                _, yc = states('image', cand.reshape(-1, 28, 28))
                cc = sp(None, yc)['core_inner'].reshape(PER, 9, -1)
                ctx = {}
                for s in (-1, 1):
                    if 1 <= n + s <= 9:
                        ctx[('collections', s)] = sp(None, states('image', np.stack([CTX[(r, n, i)][s] for i in range(PER)]))[1])['core_inner']
                        ctx[('oracle', s)] = oracle[n + s].expand(PER, -1)
                    fi = ITEMS[(r, n, 'forms')][s]
                    if fi is not None:
                        ctx[('forms', s)] = sp(None, states(med, X.DATA[(r, 'eval')][0][fi])[1])['core_inner']
                # swapped-item control: same context, item replaced by another number of the same representation
                n_swap = int(nc.stream('b1_swap', r, n).choice([k for k in X.numbers(r) if k != n]))
                f_swap, _ = states(med, X.DATA[(r, 'eval')][0][ITEMS[(r, n_swap)]])
                rows.append({'r': r, 'n': n, 'med': med, 'f': f, 'y': y, 'st': st, 'cc': cc, 'counts': counts, 'ctx': ctx,
                             'f_swap': f_swap, 'n_swap': n_swap,
                             'energy': solvers[med].energy(f, y)})
        phase['encode_cpu_seconds'] = time.process_time() - e0

        def answer(c, cc, counts):
            logits = sri.relation_logits(c[:, None, :].expand_as(cc).reshape(-1, cc.shape[-1]), cc.reshape(-1, cc.shape[-1]), 1).reshape(cc.shape[:2])
            top = logits.topk(2, dim=1)
            pick = counts[np.arange(len(counts)), top.indices[:, 0].numpy()]
            return pick, (top.values[:, 0] - top.values[:, 1]).numpy(), top.values[:, 0].numpy()

        meta_rows = []
        for row in rows:
            pick, margin, best = answer(row['st']['core_inner'], row['cc'], row['counts'])
            lab = (row['st']['output_outer'] @ codes.T).argmax(1).numpy()
            meta_rows.append({'pick': pick, 'margin': margin, 'best': best, 'label': lab})
        A_true, t_true = sri.A.detach(), sri.t.detach()
        R, _ = torch.linalg.qr(torch.from_numpy(nc.stream('b1_rotation', name).normal(size=A_true.shape)).float())
        succ = {'true': (A_true, t_true), 'perturbed': (R @ A_true, R @ t_true)}
        e0 = time.process_time()
        revised = {}  # (context, successor, mu) -> per-row (pick, label)
        for cname in CONTEXTS:
            for sname, (A, t) in succ.items():
                for mu in MUS:
                    out = []
                    for row in rows:
                        prev = row['ctx'].get((cname, -1)); nxt = row['ctx'].get((cname, 1))
                        if prev is None and nxt is None:
                            out.append(None)
                            continue
                        P = [None] * PER if prev is None else prev
                        N = [None] * PER if nxt is None else nxt
                        yr = solvers[row['med']].revise(row['f'], mu, A, t, P, N, sname)
                        sr = sp(None, yr)
                        pick, _, _ = answer(sr['core_inner'], row['cc'], row['counts'])
                        out.append({'pick': pick, 'label': (sr['output_outer'] @ codes.T).argmax(1).numpy()})
                    revised[(cname, sname, mu)] = out
        for mu in MUS:  # swapped items with collection context and the true successor
            out = []
            for row in rows:
                prev = row['ctx'].get(('collections', -1)); nxt = row['ctx'].get(('collections', 1))
                yr = solvers[row['med']].revise(row['f_swap'], mu, A_true, t_true, [None] * PER if prev is None else prev,
                                                [None] * PER if nxt is None else nxt, 'true')
                imm_pick, _, _ = answer(sp(None, solvers[row['med']].solve(row['f_swap']))['core_inner'], row['cc'], row['counts'])
                pick, _, _ = answer(sp(None, yr)['core_inner'], row['cc'], row['counts'])
                out.append({'pick': pick, 'immediate': imm_pick})
            revised[('swap', mu)] = out
        phase['revision_cpu_seconds'] = time.process_time() - e0

    # ---------------------------------------------------------------- flatten to items
    e0 = time.process_time()
    rep = np.concatenate([[row['r']] * PER for row in rows])
    num = np.concatenate([[row['n']] * PER for row in rows])
    half = np.concatenate([np.r_[np.zeros(PER // 2, int), np.ones(PER - PER // 2, int)] for _ in rows])
    medium = np.concatenate([[row['med'] == 'sequence'] * PER for row in rows]).astype(float)
    withheld = np.array([n not in labelled[r] for r, n in zip(rep, num)])
    interior = (num >= 2) & (num <= 8)
    notdots = rep != 'dots'
    imm = np.concatenate([m['pick'] for m in meta_rows]) == num
    imm_label = np.concatenate([m['label'] for m in meta_rows]) == num
    margin = np.concatenate([m['margin'] for m in meta_rows]); best = np.concatenate([m['best'] for m in meta_rows])
    fnorm = np.concatenate([row['f'].norm(dim=1).numpy() for row in rows])
    energy = np.concatenate([row['energy'].numpy() for row in rows]) / fnorm ** 2

    def rev_vec(key, field):
        return np.concatenate([o[field] if o is not None else np.full(PER, -1) for o in revised[key]])

    has_ctx = {c: np.concatenate([[(row['ctx'].get((c, -1)) is not None) or (row['ctx'].get((c, 1)) is not None)] * PER for row in rows])
               for c in CONTEXTS}
    one_sided = {c: np.concatenate([[(row['ctx'].get((c, -1)) is None) != (row['ctx'].get((c, 1)) is None)] * PER for row in rows]).astype(float)
                 for c in CONTEXTS}
    groups = {'main_interior': notdots & interior, 'main_interior_withheld': notdots & interior & withheld,
              'main_interior_labelled': notdots & interior & ~withheld, 'boundary': notdots & ~interior,
              'dots': ~notdots, 'image': notdots & interior & (medium == 0), 'sequence': notdots & interior & (medium == 1)}
    res = {'sanity_max_abs_diff': sanity, 'n_items': int(len(rep)), 'immediate': {}, 'label_readout': {}, 'mu_sweep': {}, 'selection': {}}
    for g, m in groups.items():
        for h in (0, 1):
            sel = m & (half == h)
            if sel.any():
                res['immediate'][f'{g}/{"AB"[h]}'] = float(imm[sel].mean())
    for c in CONTEXTS:
        for s in succ:
            for mu in MUS:
                rv = rev_vec((c, s, mu), 'pick') == num
                rv = np.where(has_ctx[c], rv, imm)
                res['mu_sweep'][f'{c}/{s}/{mu}'] = {f'{g}/{"AB"[h]}': float(rv[m & (half == h)].mean())
                                                    for g, m in groups.items() for h in (0, 1) if (m & (half == h)).any()}
    for c in CONTEXTS:
        for s in succ:
            selA = notdots & (half == 0) & has_ctx[c]
            accA = {mu: float(np.where(has_ctx[c], rev_vec((c, s, mu), 'pick') == num, imm)[selA].mean()) for mu in MUS}
            mu = max(MUS, key=lambda u: (accA[u], -u))
            rv = np.where(has_ctx[c], rev_vec((c, s, mu), 'pick') == num, imm)
            target = np.where(~imm & rv, 2, np.where(imm & ~rv, 0, 1))
            feats = np.stack([margin, best, energy, fnorm, one_sided[c], medium], 1)
            trainA = notdots & (half == 0) & has_ctx[c]
            mu_f, sd_f = feats[trainA].mean(0), feats[trainA].std(0) + 1e-9
            Z = torch.from_numpy((feats - mu_f) / sd_f).float()
            lin = fit_controller(Z[torch.from_numpy(trainA)], torch.from_numpy(target[trainA]))
            with torch.no_grad():
                p = torch.softmax(lin(Z), 1).numpy()
            gain = np.where(has_ctx[c], p[:, 2] - p[:, 0], -np.inf)
            cand = np.unique(np.r_[gain[trainA], -np.inf, np.inf])
            obj = lambda th, sel: (np.where(gain > th, rv, imm)[sel].mean() - C_COST * (gain > th)[sel].mean())
            theta = max(cand, key=lambda th: obj(th, notdots & (half == 0)))
            label_rev = np.where(has_ctx[c], rev_vec((c, s, mu), 'label') == num, imm_label)
            out = {'mu': mu, 'mu_accuracy_A': accA, 'threshold': float(theta), 'controller_weights': lin.weight.detach().tolist()}
            for g, m in groups.items():
                sel = m & (half == 1)
                if not sel.any():
                    continue
                go = gain[sel] > theta
                rate = float(go.mean())
                a_imm, a_all = float(imm[sel].mean()), float(rv[sel].mean())
                help_ = ~imm[sel] & rv[sel]; hurt = imm[sel] & ~rv[sel]
                curve = []
                for q in np.linspace(0, 1, 21):
                    th = np.quantile(gain[sel], q) if np.isfinite(gain[sel]).any() else np.inf
                    gg = gain[sel] > th
                    curve.append([float(gg.mean()), float(np.where(gg, rv[sel], imm[sel]).mean())])
                bins = np.clip((p[sel, 2] * 10).astype(int), 0, 9)
                ece = float(sum(abs(p[sel, 2][bins == b].mean() - help_[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
                out[g] = {'immediate': a_imm, 'always': a_all, 'gated': float(np.where(go, rv[sel], imm[sel]).mean()), 'gated_rate': rate,
                          'random_at_gated_rate': a_imm + rate * (a_all - a_imm), 'oracle': float((imm[sel] | rv[sel]).mean()),
                          'oracle_rate': float(help_.mean()),
                          'always_corrected': float(help_.mean()), 'always_damaged': float(hurt.mean()),
                          'gated_corrected': float((help_ & go).mean()), 'gated_damaged': float((hurt & go).mean()),
                          'curve_rate_accuracy': curve, 'ece_p_help': ece,
                          'label_readout_immediate': float(imm_label[sel].mean()), 'label_readout_always': float(label_rev[sel].mean())}
            res['selection'][f'{c}/{s}'] = out
    n_swap = np.concatenate([[row['n_swap']] * PER for row in rows])
    res['swapped_item'] = {}
    for mu in MUS:
        pk = np.concatenate([o['pick'] for o in revised[('swap', mu)]])
        ip = np.concatenate([o['immediate'] for o in revised[('swap', mu)]])
        sel = groups['main_interior'] & (half == 1)
        res['swapped_item'][str(mu)] = {'revised_follows_context': float((pk == num)[sel].mean()),
                                        'revised_follows_item': float((pk == n_swap)[sel].mean()),
                                        'immediate_follows_item': float((ip == n_swap)[sel].mean())}
    res['error_auroc'] = {}
    for medname, mm in (('image', 0.), ('sequence', 1.)):
        sel = notdots & (medium == mm)
        err = ~imm[sel]
        res['error_auroc'][medname] = {'energy_over_fnorm2': auroc(energy[sel], err), 'fnorm': auroc(fnorm[sel], err),
                                       'negative_margin': auroc(-margin[sel], err), 'n': int(sel.sum()), 'error_rate': float(err.mean())}
    phase['analysis_cpu_seconds'] = time.process_time() - e0
    n_items = len(rep)
    cost = {'item_solves': n_items, 'candidate_solves': 9 * n_items, 'context_solves_collections': int(2 * n_items),
            'revision_solves_per_config': n_items, 'factorizations': {m: len(s.cache) + 1 for m, s in solvers.items()}}
    return name, res, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **phase, 'counts': cost,
                       'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


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
    phases['b1_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(SPEC.relative_to(ROOT)), 'status': 'B1 on frozen pilot checkpoints: not headline evidence',
           'config': {'checkpoints': CHECKPOINTS, 'per_number': PER, 'mus': MUS, 'c_cost': C_COST, 'contexts': CONTEXTS,
                      'oracle_per_count': ORACLE_PER, 'candidates': 'fixed-area dots, 9 per item, shuffled'},
           'checkpoint_sha256': {k: nc.sha256(ROOT / p) for k, p in CHECKPOINTS.items()},
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__), 'experiment_sha256': nc.sha256(X.__file__),
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
