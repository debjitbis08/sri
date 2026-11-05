"""Equality diagnostics on v2 snapshots (docs/equality-diagnostics-v2.md). Frozen weights; nothing in the system changes.

Usage: .venv/bin/python -m experiments.equality_diagnostics_v2            (DIAG_WORKERS, default 12)
       .venv/bin/python -m experiments.equality_diagnostics_v2 summary
"""
import json
import platform
import resource
import sys
import time

from experiments import equality_relations_v2 as V2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src import relation_codes as RC

RP, EQ, X, nc, ns = V2.RP, V2.EQ, V2.X, V2.nc, V2.ns
NAME = 'equality-diagnostics-v2'
CONTRACT = X.ROOT / 'docs' / 'equality-diagnostics-v2.md'
WEIGHTS = X.ROOT / 'runs' / ('equality-relations-v2-snap' + ('-smoke' if V2.SMOKE else '')) / 'weights'
REPORT = X.ROOT / 'reports' / NAME
ARMS, SEEDS, BLOCKS = V2.SNAP_ARMS, V2.SEEDS, ('block2', 'block3')
C, ZH, DEV, EN, TH = V2.C, V2.ZH, V2.DEV, V2.EN, V2.TH
KINDS = (C, ZH, DEV, EN, TH)
SHORT, N = V2.SHORT, V2.N
REFIT_STEPS, REFIT_BATCH, PROBE_STEPS = 500, 8192, 300


def load(arm, seed, block):
    RP.prepare()
    model = EQ.RelNet(arm, RP.DATA['coll_train'].shape[-1], torch.Generator().manual_seed(seed))
    model.load_state_dict(torch.load(WEIGHTS / f'{arm}-seed{seed}-{block}.pt'))
    return model.eval()


def pair_feature(model, ca, cb):
    return model.G(torch.cat([ca, cb], 1)) if model.arm in RC.PAIR else torch.cat([ca, cb], 1)


def count_scores(S, labr):
    """(items, refs) score matrix -> (items, 9) mean score per reference count."""
    return torch.stack([S[:, labr == v].mean(1) for v in range(1, N + 1)], 1)


def stats(pred, lab, wh, score=None):
    err = pred - lab
    r = {'acc': float((err == 0).float().mean()), 'within1': float((err.abs() <= 1).float().mean()),
         'adjacent_error_share': float(((err.abs() == 1).sum() / (err != 0).sum().clamp_min(1))),
         'acc_taught': float((err[~wh] == 0).float().mean()), 'acc_withheld': float((err[wh] == 0).float().mean()) if wh.any() else None}
    if score is not None:
        top = score.topk(2, 1).values
        r['margin'] = float((top[:, 0] - top[:, 1]).mean())
    return r


def probe(x, y, xt, seed):
    torch.manual_seed(seed)
    lin = nn.Linear(x.shape[1], N)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(PROBE_STEPS):
        opt.zero_grad()
        F.cross_entropy(lin(x), y - 1).backward()
        opt.step()
    with torch.no_grad():
        return lin(xt).argmax(1) + 1


def refit_eq(model, pairs, seed):
    """Fresh eq reader (architecture of S plus a fresh operator code) on frozen pair features; pairs: list of (ca, ya, cb, yb)."""
    torch.manual_seed(seed)
    fin = RC.D if model.arm in RC.PAIR else 2 * RC.D
    S = nn.Sequential(nn.Linear(fin + EQ.OPD, 64), nn.ReLU(), nn.Linear(64, 1))
    code = nn.Parameter(torch.randn(EQ.OPD) * .1)
    with torch.no_grad():
        feats, ts = [], []
        for ca, ya, cb, yb in pairs:
            a, b, _ = RC.all_pairs(ca, ya, cb, yb)
            feats.append(pair_feature(model, a, b))
            ts.append((ya.repeat_interleave(len(yb)) == yb.repeat(len(ya))).float())
        f, t = torch.cat(feats), torch.cat(ts)
    pos = t.mean()
    opt = torch.optim.Adam([*S.parameters(), code], lr=1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(REFIT_STEPS):
        idx = torch.randint(len(f), (min(REFIT_BATCH, len(f)),), generator=g)
        opt.zero_grad()
        lg = S(torch.cat([f[idx], code.expand(len(idx), -1)], 1))[:, 0]
        F.binary_cross_entropy_with_logits(lg, t[idx], pos_weight=(1 - pos) / pos).backward()
        opt.step()
    return lambda ca, cb: S(torch.cat([pair_feature(model, ca, cb), code.expand(len(ca), -1)], 1))[:, 0]


@torch.no_grad()
def mat(fn, xa, xb):
    """Operator outputs for all (xa[i], xb[j]), reshaped to (len(xa), len(xb), ...)."""
    out = fn(xa.repeat_interleave(len(xb), 0), xb.repeat(len(xa), 1))
    return out.reshape(len(xa), len(xb), *out.shape[1:])


def bal(p, t):
    return float(.5 * (p[t].float().mean() + (~p[~t]).float().mean())) if t.any() and (~t).any() else None


def diagnose(arm, seed, block):
    torch.set_num_threads(1)
    model = load(arm, seed, block)
    withheld = {k: tuple(int(x) for x in v) for k, v in V2.D.withheld_counts(seed).items()}
    ev = RP.DATA['eval']
    with torch.no_grad():
        codes = {}
        for g, kind in enumerate(ev):
            x = torch.from_numpy(np.concatenate([ev[kind][v] for v in ns.NUMBERS]))
            codes[kind] = model.codes(x, torch.full((len(x),), g))
    k = len(ev[C][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    test = (torch.arange(len(lab)) % k) >= k // 2
    fit = ~test
    M, labM = codes[C][fit], lab[fit]  # references: the collection fit half
    mu = torch.stack([M[labM == v].mean(0) for v in range(1, N + 1)])
    yt = lab[test]
    out = {'identity': {}, 'judgement': {}, 'substitution': {}, 'conflict': {}}
    ops = lambda a, b: model.op_logits(a, b)
    refit_coll = refit_eq(model, [(M, labM, M, labM)], seed)
    for kind in KINDS:
        x = codes[kind][test]
        wh = torch.tensor([int(v) in set(withheld.get(kind, ())) for v in yt.tolist()])
        L1, L2 = mat(ops, x, M), mat(ops, M, x).transpose(0, 1)  # [i, j] = op(x_i, m_j), op(m_j, x_i)
        E1, E2 = L1[..., 0], L2[..., 0]
        zero = mat(lambda a, b: model.rel_logits(a, b).log_softmax(1)[:, N - 1], x, M)
        idr = {}
        with torch.no_grad():
            near = -((x[:, None] - mu[None]) ** 2).sum(2)
            idr['nearest'] = stats(near.argmax(1) + 1, yt, wh, near)
            for name, S in (('eq_first', E1), ('eq_second', E2), ('zero_class', zero)):
                cs = count_scores(S, labM)
                idr[name] = stats(cs.argmax(1) + 1, yt, wh, cs)
            idr['eq_item'] = stats(labM[E1.argmax(1)], yt, wh)
            idr['relational'] = stats(RC.relational_identity(model, x, mu), yt, wh)
            cs = count_scores(mat(refit_coll, x, M), labM)
            idr['refit_eq_collections'] = stats(cs.argmax(1) + 1, yt, wh, cs)
        idr['probe_collections'] = stats(probe(M, labM, x, seed), yt, wh)
        if kind != C:
            xf, yf = codes[kind][fit], lab[fit]
            idr['probe_kind'] = stats(probe(xf, yf, x, seed), yt, wh)
            up = refit_eq(model, [(M, labM, M, labM), (xf, yf, M, labM), (M, labM, xf, yf)], seed)
            with torch.no_grad():
                cs = count_scores(mat(up, x, M), labM)
            idr['refit_eq_upper'] = stats(cs.argmax(1) + 1, yt, wh, cs)
        out['identity'][SHORT[kind]] = idr
        # judgement
        same = yt[:, None] == labM[None]
        dist = (yt[:, None] - labM[None]).abs()
        acc1 = E1 > 0
        cs = count_scores(E1, labM)
        judged = cs.argmax(1) + 1
        accepted = cs.max(1).values > 0
        correct = judged == yt
        out['judgement'][SHORT[kind]] = {
            'positive_recall': float(acc1[same].float().mean()), 'false_accept_adjacent': float(acc1[dist == 1].float().mean()),
            'false_accept_far': float(acc1[dist >= 2].float().mean()), 'accepted_share': float(accepted.float().mean()),
            'acc_accepted': float(correct[accepted].float().mean()) if accepted.any() else None,
            'acc_rejected': float(correct[~accepted].float().mean()) if (~accepted).any() else None,
            'sym_pos': float((E2[acc1] > 0).float().mean()) if acc1.any() else None,
            'sym_neg': float((E2[~acc1] <= 0).float().mean())}
        # substitution against truth and against own judgement
        sub = {}
        for grp, gm in (('accepted_correct', accepted & correct), ('accepted_wrong', accepted & ~correct), ('rejected', ~accepted)):
            if not gm.any():
                continue
            r = {'n': int(gm.sum())}
            for tgt, yy in (('truth', yt), ('own', judged)):
                y = yy[gm][:, None].expand(-1, len(labM))
                m_ = labM[None].expand(len(y), -1)
                for oi, op, t1, t2 in ((1, 'more', m_ > y, y > m_), (2, 'sum10', (y + m_) == 10, (y + m_) == 10)):
                    v = [bal(L1[gm][..., oi] > 0, t1), bal(L2[gm][..., oi] > 0, t2)]
                    v = [w for w in v if w is not None]
                    r[f'{tgt}_{op}'] = float(np.mean(v)) if v else None
            sub[grp] = r
        out['substitution'][SHORT[kind]] = sub
        m1, m2 = L1[..., 1] > 0, L2[..., 1] > 0
        out['conflict'][SHORT[kind]] = {
            'more_given_eq': float((m1 | m2)[acc1].float().mean()) if acc1.any() else None,
            'more_both_ways': float((m1 & m2).float().mean()), 'eq_accept_rate': float(acc1.float().mean())}
    # triples zh -> coll -> dev
    zt, dt = codes[ZH][test], codes[DEV][test]
    A = mat(ops, zt, M)[..., 0] > 0
    B = mat(ops, M, dt)[..., 0] > 0
    Cz = mat(ops, zt, dt)[..., 0] > 0
    path = (A.float() @ B.float()) > 0
    truth = yt[:, None] == yt[None]
    out['triples'] = {'path_share': float(path.float().mean()),
                      'violation_given_path': float((path & ~Cz).sum() / path.sum().clamp_min(1)),
                      'true_same_given_path': float(truth[path].float().mean()) if path.any() else None,
                      'eq_given_no_path': float((~path & Cz).sum() / (~path).sum().clamp_min(1)),
                      'true_same_given_no_path': float(truth[~path].float().mean()) if (~path).any() else None}
    return out


def task(t):
    t0, c0 = time.perf_counter(), time.process_time()
    r = diagnose(*t)
    return '{}/seed{}/{}'.format(*t), r, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}


def reproduction():
    a = json.loads((X.ROOT / 'reports' / 'equality-relations-v2.json').read_text())['results']
    b = json.loads((X.ROOT / 'reports' / 'equality-relations-v2-snap.json').read_text())['results']
    return {k: a[k] == b[k] for k in b}


def main():
    t0 = time.perf_counter()
    RP.prepare()
    tasks = [(a, s, b) for a in ARMS for s in SEEDS for b in BLOCKS]
    with __import__('multiprocessing').get_context('fork').Pool(int(__import__('os').environ.get('DIAG_WORKERS', 12)), maxtasksperchild=1) as pool:
        res = pool.map(task, tasks, chunksize=1)
    RP.S0.save_json(f'{REPORT}.json', {'status': 'equality diagnostics on v2 snapshots', 'arms': ARMS, 'seeds': SEEDS,
                                        'reproduction_equal_to_v2': reproduction(), 'results': {k: r for k, r, _ in res}})
    RP.S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
        'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
        'python': platform.python_version(), 'torch': torch.__version__, 'cpu': platform.processor(), 'threads_per_worker': 1,
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary():
    rep = json.loads(REPORT.with_suffix('.json').read_text())
    res = rep['results']
    print('reproduction equal to v2:', all(rep['reproduction_equal_to_v2'].values()), rep['reproduction_equal_to_v2'])
    f = lambda v: f'{np.mean([w for w in v if w is not None]):.2f}' if any(w is not None for w in v) else '-'
    get = lambda arm, block, *path: [_dig(res[f'{arm}/seed{s}/{block}'], path) for s in SEEDS]
    reads = ('nearest', 'eq_first', 'eq_second', 'eq_item', 'zero_class', 'relational', 'refit_eq_collections', 'refit_eq_upper',
             'probe_collections', 'probe_kind')
    for block in BLOCKS:
        for kd in ('zh', 'dev', 'en', 'thai'):
            print(f'\n## {block} {kd} identity acc (adjacent share of errors)')
            print('| arm | ' + ' | '.join(reads) + ' |')
            for arm in ARMS:
                print(f'| {arm} | ' + ' | '.join(f"{f(get(arm, block, 'identity', kd, r, 'acc'))} ({f(get(arm, block, 'identity', kd, r, 'adjacent_error_share'))})"
                                               if _dig(res[f'{arm}/seed{SEEDS[0]}/{block}'], ('identity', kd, r)) else '-' for r in reads) + ' |')
        print(f'\n## {block} judgement: recall / false adj / false far / accepted / acc acc / acc rej / sym+ / sym-')
        keys = ('positive_recall', 'false_accept_adjacent', 'false_accept_far', 'accepted_share', 'acc_accepted', 'acc_rejected', 'sym_pos', 'sym_neg')
        for arm in ARMS:
            print(f'| {arm} | ' + ' | '.join(kd + ' ' + '/'.join(f(get(arm, block, 'judgement', kd, q)) for q in keys) for kd in ('coll', 'zh', 'dev', 'en', 'thai')) + ' |')
        print(f'\n## {block} substitution truth/own more, sum10 by group')
        for arm in ARMS:
            cells = []
            for kd in ('zh', 'dev', 'thai'):
                for grp in ('accepted_correct', 'accepted_wrong', 'rejected'):
                    v = lambda q: f(get(arm, block, 'substitution', kd, grp, q))
                    cells.append(f"{kd} {grp[:10]} n={f(get(arm, block, 'substitution', kd, grp, 'n'))} {v('truth_more')}/{v('own_more')} {v('truth_sum10')}/{v('own_sum10')}")
            print(f'| {arm} | ' + ' | '.join(cells) + ' |')
        print(f'\n## {block} conflict: more|eq / more both / eq accept rate ; triples path, viol|path, true|path, eq|nopath, true|nopath')
        for arm in ARMS:
            c = ' | '.join(kd + ' ' + '/'.join(f(get(arm, block, 'conflict', kd, q)) for q in ('more_given_eq', 'more_both_ways', 'eq_accept_rate'))
                           for kd in ('coll', 'zh', 'dev', 'en', 'thai'))
            tr = '/'.join(f(get(arm, block, 'triples', q)) for q in ('path_share', 'violation_given_path', 'true_same_given_path', 'eq_given_no_path', 'true_same_given_no_path'))
            print(f'| {arm} | {c} | triples {tr} |')


def _dig(d, path):
    for p in path:
        if d is None or p not in d:
            return None
        d = d[p]
    return d


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
