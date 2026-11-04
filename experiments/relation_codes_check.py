"""Relation codes check v1, parts A and B (docs/relation-codes-check-v1.md). No training of the system; probes only.

Per checkpoint and code level (projection output, core_inner):
  R-diff     linear softmax on c(b) - c(a) -> d = y - x (17 classes)
  R-pair     MLP on [c(a), c(b), c(b) - c(a)] -> d
  via-id     MLP identity probe, d = id(b) - id(a)
  rank code  axis position (ridge on collections) -> rank within kind -> count by quantile
  system     the trained next and more readers (core level only)
All probes are fitted on collection/taps pairs (first half of items) only.
Part B: counts derived from anchors through R-pair / R-diff: one same-kind anchor, three same-kind anchors, one collection anchor.

Usage: .venv/bin/python -m experiments.relation_codes_check            (TS_WORKERS, default 4; rerun to resume)
       .venv/bin/python -m experiments.relation_codes_check summary [arm]
"""
import os
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import two_script_reads as T
sys.argv = _argv
import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import two_script_reads_v4 as V4

SOX, S0, P, D, X, LM, nc, ns = T.SOX, T.S0, T.P, T.D, T.X, T.LM, T.nc, T.ns
NAME = 'relation-codes-check-v1'
CONTRACT = X.ROOT / 'docs' / 'relation-codes-check-v1.md'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
ARMS = ('gp', 'common-k-cand')
POINTS = ('block1', 'block3', 'delay', 'relearn')
LEVELS = ('projection', 'core_inner')
N = 9
FIT_PAIRS = (('collection', 'collection'), ('taps', 'taps'), ('collection', 'taps'), ('taps', 'collection'))
TEST_PAIRS = FIT_PAIRS + (('english', 'english'), ('chinese', 'chinese'), ('devanagari_digits', 'devanagari_digits'),
                          ('thai_digits', 'thai_digits'), ('chinese', 'devanagari_digits'), ('devanagari_digits', 'chinese'),
                          ('collection', 'chinese'), ('chinese', 'collection'), ('collection', 'devanagari_digits'))
SCRIPT_KINDS = ('english', 'chinese', 'devanagari_digits', 'thai_digits')
STEPS, LR, WD, HID = 400, 1e-2, 1e-4, 64


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    if pos.all() or not pos.any():
        return None
    r = np.argsort(np.argsort(score, kind='stable'), kind='stable') + 1.0
    # average ranks for ties
    order = np.argsort(score, kind='stable'); s = score[order]; rr = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rr[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s)); r[order] = rr
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def train(net, x, y, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).long()
    for _ in range(STEPS):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
    net.eval()
    return lambda z: torch.log_softmax(net(torch.from_numpy(z).float()), 1).detach().numpy()


def pair_rows(ca, cb, la, lb, sel_a=None, sel_b=None):
    """Item-index-aligned pairs over all 81 count pairs. c*: dict count -> (k, d) codes."""
    A, B, dd = [], [], []
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            a, b = ca[x], cb[y]
            idx = np.arange(min(len(a), len(b)))
            if sel_a is not None:
                idx = idx[sel_a[:len(idx)]]
            A.append(a[idx]); B.append(b[idx]); dd.append(np.full(len(idx), y - x))
    return np.concatenate(A), np.concatenate(B), np.concatenate(dd)


def scores_d(logp_d, x_counts=None):
    """Most likely d (index 0..16 -> -8..8)."""
    return logp_d.argmax(1) - 8


@torch.no_grad()
def system_reads(model, A, B):
    A, B = torch.from_numpy(A), torch.from_numpy(B)
    nxt = model.relation_logits(A, B, 0).numpy()
    more = ((model.relation_logits(A, B, 2) - model.relation_logits(B, A, 2)) / 2).numpy()
    return nxt, more


def relation_metrics(dhat, d, next_score=None, sign_score=None):
    r = {'exact': float((dhat == d).mean()), 'within1': float((np.abs(dhat - d) <= 1).mean())}
    nz = d != 0
    r['sign'] = float((np.sign(dhat[nz]) == np.sign(d[nz])).mean()) if sign_score is None else float((np.sign(sign_score[nz]) == np.sign(d[nz])).mean())
    if next_score is not None:
        r['next_auroc'] = auroc(next_score, d == 1)
    r['next_exact_recall'] = float((dhat[d == 1] == 1).mean())
    return r


def analyse(codes, model, seed, lv, system):
    """codes: kind -> count -> (k, d) array at level lv."""
    k = min(len(codes[kd][n]) for kd in codes for n in codes[kd])
    half = np.arange(k) < k // 2
    dim = codes['collection'][1].shape[1]
    # standardization from fit kinds
    fit_items = np.concatenate([codes[kd][n][:k][half] for kd in ('collection', 'taps') for n in range(1, N + 1)])
    mu, sd = fit_items.mean(0), fit_items.std(0) + 1e-6
    z = {kd: {n: (codes[kd][n][:k] - mu) / sd for n in codes[kd]} for kd in codes}
    # fit pair data
    FA, FB, FD = zip(*[pair_rows(z[a], z[b], a, b, sel_a=half) for a, b in FIT_PAIRS])
    FA, FB, FD = np.concatenate(FA), np.concatenate(FB), np.concatenate(FD)
    torch.manual_seed(seed)
    rdiff = train(torch.nn.Linear(dim, 17), FB - FA, FD + 8, seed)
    torch.manual_seed(seed + 1)
    rpair_net = torch.nn.Sequential(torch.nn.Linear(3 * dim, HID), torch.nn.ReLU(), torch.nn.Linear(HID, 17))
    rp = train(rpair_net, np.concatenate([FA, FB, FB - FA], 1), FD + 8, seed + 1)
    rpair = lambda a, b: rp(np.concatenate([a, b, b - a], 1))
    rd = lambda a, b: rdiff(b - a)
    torch.manual_seed(seed + 2)
    id_net = torch.nn.Sequential(torch.nn.Linear(dim, HID), torch.nn.ReLU(), torch.nn.Linear(HID, N))
    IX = np.concatenate([z[kd][n][half] for kd in ('collection', 'taps') for n in range(1, N + 1)])
    IY = np.concatenate([np.full(half.sum(), n - 1) for kd in ('collection', 'taps') for n in range(1, N + 1)])
    idp = train(id_net, IX, IY, seed + 2)
    ident = lambda c: idp(c).argmax(1) + 1
    # rank code: ridge axis on collection first half (raw codes, as v4), rank within kind
    C = np.concatenate([z['collection'][n][half] for n in range(1, N + 1)]); Cy = np.repeat(np.arange(1, N + 1), half.sum())
    w = np.linalg.solve(C.T @ C + np.eye(dim), C.T @ (Cy - 5.0))
    rank_count, axis_stats = {}, {}
    for kd in codes:
        sel = ~half if kd in ('collection', 'taps') else np.ones(k, bool)
        allc = np.concatenate([z[kd][n] for n in range(1, N + 1)]); lab = np.repeat(np.arange(1, N + 1), k)
        pos = allc @ w + 5.0
        q = np.argsort(np.argsort(pos, kind='stable'), kind='stable')  # rank over all items of the kind
        rc = np.minimum(q * N // len(q), N - 1) + 1
        rank_count[kd] = {n: rc[lab == n] for n in range(1, N + 1)}
        s = np.tile(sel, N)
        axis_stats[T.SHORT[kd]] = {'spearman': spearman(pos[s], lab[s]), 'slope': float(np.polyfit(lab[s], pos[s], 1)[0]),
                                   'rank_identity': float((rc == lab)[s].mean()), 'rank_within1': float((np.abs(rc - lab) <= 1)[s].mean()),
                                   'probe_identity': float((ident(allc[s]) == lab[s]).mean())}
    out = {'axis': axis_stats, 'pairs': {}, 'anchored': {}}
    for a, b in TEST_PAIRS:
        sel = ~half if (a, b) in FIT_PAIRS else None
        A, B, d = pair_rows(z[a], z[b], a, b, sel_a=sel)
        r = {}
        lp = rpair(A, B); r['R-pair'] = relation_metrics(scores_d(lp), d, next_score=lp[:, 9])
        lp = rd(A, B); r['R-diff'] = relation_metrics(scores_d(lp), d, next_score=lp[:, 9])
        r['via-id'] = relation_metrics(ident(B) - ident(A), d)
        # rank code d: aligned the same way as pair_rows
        RA, RB, _ = pair_rows({n: rank_count[a][n][:, None] for n in range(1, N + 1)}, {n: rank_count[b][n][:, None] for n in range(1, N + 1)}, a, b, sel_a=sel)
        r['rank'] = relation_metrics((RB - RA)[:, 0], d)
        if system:
            rawA, rawB, _ = pair_rows(codes[a], codes[b], a, b, sel_a=sel)
            nxt, more = system_reads(model, rawA.astype(np.float32), rawB.astype(np.float32))
            r['system'] = {'next_auroc': auroc(nxt, d == 1), 'sign': float((np.sign(-more[d != 0]) == np.sign(d[d != 0])).mean())}
        out['pairs'][f'{T.SHORT[a]}>{T.SHORT[b]}'] = r
    # part B: anchored derivation
    rng = np.random.default_rng(seed)
    for kd in SCRIPT_KINDS + ('collection',):
        sel = ~half if kd == 'collection' else np.ones(k, bool)
        items = np.concatenate([z[kd][n][sel] for n in range(1, N + 1)]); lab = np.repeat(np.arange(1, N + 1), sel.sum())
        m = len(items)
        res = {'probe_identity': float((ident(items) == lab).mean())}
        for name, f in (('R-pair', rpair), ('R-diff', rd)):
            # logp[j, i, :] = log p(d | anchor j, item i)
            LP = np.stack([f(np.repeat(items[j:j + 1], m, 0), items) for j in range(m)])  # (m, m, 17)
            cand = np.arange(1, N + 1)
            # score[j, i, n] = logp(d = n - lab[j])
            S = np.take_along_axis(LP, (cand[None, None] - lab[:, None, None] + 8).repeat(m, 1), 2)
            one = S.argmax(2) + 1
            mask = ~np.eye(m, dtype=bool)
            res[f'{name}_one_anchor'] = float((one == lab[None])[mask].mean())
            hits = []
            for _ in range(200):
                js = rng.choice(m, 3, replace=False)
                while len(set(lab[js])) < 3:
                    js = rng.choice(m, 3, replace=False)
                pred = S[js].sum(0).argmax(1) + 1
                keep = np.ones(m, bool); keep[js] = False
                hits.append((pred == lab)[keep].mean())
            res[f'{name}_three_anchors'] = float(np.mean(hits))
            # one collection anchor (held-out collections)
            if kd != 'collection':
                cs = np.concatenate([z['collection'][n][~half] for n in range(1, N + 1)]); cl = np.repeat(np.arange(1, N + 1), (~half).sum())
                acc = []
                for j in range(len(cs)):
                    lpj = f(np.repeat(cs[j:j + 1], m, 0), items)
                    sc = lpj[:, np.clip(cand - cl[j] + 8, 0, 16)]
                    acc.append((sc.argmax(1) + 1 == lab).mean())
                res[f'{name}_collection_anchor'] = float(np.mean(acc))
        out['anchored'][T.SHORT[kd]] = res
    return out


@torch.no_grad()
def item_codes(enc, model):
    raw = {'collection': ({n: P.DOTS['succ'][n][:T.E] for n in ns.NUMBERS}, 'image'),
           'taps': ({n: D.TAPS[('eval', n)][:T.E] for n in ns.NUMBERS}, 'sequence')}
    for lang in T.SCRIPTS:
        raw[lang] = ({n: X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:T.E] for n in ns.NUMBERS}, 'image')
    out = {lv: {} for lv in LEVELS}
    for kind, (v, medium) in raw.items():
        per = {n: V4.levels(enc, model, v[n], medium) for n in ns.NUMBERS}
        for lv in LEVELS:
            out[lv][kind] = {n: per[n][lv].astype(np.float32) for n in ns.NUMBERS}
    return out


def task(t):
    arm, seed = t
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    res, ck = {}, {}
    for point in POINTS:
        enc, model, ck[point] = T.load(arm, seed, point)
        codes = item_codes(enc, model)
        res[point] = {lv: analyse(codes[lv], model, seed, lv, system=(lv == 'core_inner')) for lv in LEVELS}
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'{arm}/seed{seed}', {'reads': res, 'checkpoints': ck}, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    tasks = [(a, s) for a in ARMS for s in T.SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(T.WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'relation codes check parts A and B; probes on frozen checkpoints', 'arms': ARMS, 'seeds': T.SEEDS,
                                     'points': POINTS, 'levels': LEVELS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': T.WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
        'probe_steps': STEPS, 'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary(arm='gp', lv='core_inner'):
    import json
    res = json.loads(REPORT.with_suffix('.json').read_text())['results']
    m = lambda f: (lambda v: f'{np.mean(v):.2f}' if v else '-')([x for x in (f(res[f'{arm}/seed{s}']['reads']) for s in T.SEEDS) if x is not None])
    for point in POINTS:
        print(f'\n## {arm}, {point}, {lv}')
        print('axis: ' + ', '.join(f"{kd} rho {m(lambda r: r[point][lv]['axis'][kd]['spearman'])} slope {m(lambda r: r[point][lv]['axis'][kd]['slope'])}"
                                    f" rank-id {m(lambda r: r[point][lv]['axis'][kd]['rank_identity'])} probe-id {m(lambda r: r[point][lv]['axis'][kd]['probe_identity'])}"
                                    for kd in ('coll', 'en', 'zh', 'dev', 'thai')))
        print('| pair | read | exact | within1 | sign | next AUROC | next recall |')
        for pk in res[f'{arm}/seed300']['reads'][point][lv]['pairs']:
            for rd in ('R-pair', 'R-diff', 'via-id', 'rank', 'system'):
                if rd not in res[f'{arm}/seed300']['reads'][point][lv]['pairs'][pk]:
                    continue
                g = lambda key: m(lambda r: r[point][lv]['pairs'][pk][rd].get(key))
                print(f'| {pk} | {rd} | {g("exact")} | {g("within1")} | {g("sign")} | {g("next_auroc")} | {g("next_exact_recall")} |')
        print('anchored:')
        for kd, v in res[f'{arm}/seed300']['reads'][point][lv]['anchored'].items():
            print(kd, {key: m(lambda r: r[point][lv]['anchored'][kd].get(key)) for key in v})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary(*sys.argv[2:])
    else:
        main()
