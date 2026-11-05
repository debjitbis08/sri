"""Consistency critic with learned probe selection v1 (docs/consistency-critic-v1.md). Frozen v2 snapshots; the system
does not change. A critic is trained on collection teaching labels; a selector learns which probes to evaluate.

Usage: .venv/bin/python -m experiments.consistency_critic_v1            (CC_WORKERS, default 12)
       .venv/bin/python -m experiments.consistency_critic_v1 summary
       CC_SMOKE=1 ... for a short check (one snapshot, few steps)
"""
import json
import os
import platform
import resource
import sys
import time

from experiments import durable as DUR
from experiments import equality_diagnostics_v2 as DG
import numpy as np
import torch
import torch.nn.functional as F

from src import consistency_critic as CC

V2, RP, EQ, X, nc, ns = DG.V2, DG.RP, DG.EQ, DG.X, DG.nc, DG.ns
SMOKE = bool(os.environ.get('CC_SMOKE'))
NAME = 'consistency-critic-v1' + ('-smoke' if SMOKE else '')
CONTRACT = X.ROOT / 'docs' / 'consistency-critic-v1.md'
REPORT = X.ROOT / 'reports' / NAME
RUNS = X.ROOT / 'runs' / NAME
ARMS, SEEDS, BLOCKS = DG.ARMS, DG.SEEDS, DG.BLOCKS
C, ZH, DEV, EN, TH = DG.C, DG.ZH, DG.DEV, DG.EN, DG.TH
KINDS, SHORT, N = DG.KINDS, DG.SHORT, DG.N
SEEN = (ZH, DEV, EN)  # kinds whose fit halves the 'sel-seen' selector may use, label-free
NEAR, RAND = 3, 2  # candidate stage: 3 nearest to a, 3 nearest to b (cosine), 2 random; K = 8 contexts
K = 2 * NEAR + RAND
CRITIC_STEPS, SEL_STEPS, BATCH, SEL_QUERIES = (30, 30, 64, 256) if SMOKE else (2000, 1000, 256, 4096)
EVAL_Q = 300 if SMOKE else 3000  # queries per (kind, operator)
KS = (1, 2, 3, 4, 6, 9)
SELECTORS = ('sel-coll', 'sel-seen')


@torch.no_grad()
def tables(model, block):
    ev = RP.DATA['eval']
    codes = {}
    for g, kind in enumerate(ev):
        x = torch.from_numpy(np.concatenate([ev[kind][v] for v in ns.NUMBERS]))
        codes[kind] = model.codes(x, torch.full((len(x),), g))
    k = len(ev[C][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    test = (torch.arange(len(lab)) % k) >= k // 2
    R = 3 if block == 'block3' else 2
    M, labM = codes[C][~test], lab[~test]
    ops = lambda a, b: model.op_logits(a, b)[:, :R]
    MM = DG.mat(ops, M, M)
    T = {'M': M, 'labM': labM, 'R': R, 'MM': MM, 'kinds': {}}
    for kind in KINDS:
        for half, sel in (('test', test), ('fit', ~test)):
            if half == 'fit' and kind not in SEEN:
                continue
            x = codes[kind][sel]
            T['kinds'][(kind, half)] = {'x': x, 'lab': lab[sel], 'XM': DG.mat(ops, x, M),
                                        'MXt': DG.mat(ops, M, x).transpose(0, 1), 'cos': F.normalize(x, dim=1) @ F.normalize(M, dim=1).T}
    T['kinds'][(C, 'self')] = {'x': M, 'lab': labM, 'XM': MM, 'MXt': MM.transpose(0, 1), 'cos': F.normalize(M, dim=1) @ F.normalize(M, dim=1).T}
    return T


def contexts(T, src, xi, ci, g, self_set):
    """Candidate stage: NEAR nearest memory items to a (cosine), NEAR nearest to b, RAND random; distinct, excluding a, b."""
    Cm = T['kinds'][(C, 'self')]['cos']
    ar = torch.arange(len(xi))
    neg = torch.full((len(xi), len(T['M'])), 0.)
    neg[ar, ci] = -1e9
    if self_set:
        neg[ar, xi] = -1e9
    n1 = (src['cos'][xi] + neg).topk(NEAR, 1).indices
    neg2 = neg.clone()
    neg2.scatter_(1, n1, -1e9)
    n2 = (Cm[ci] + neg2).topk(NEAR, 1).indices
    neg2.scatter_(1, n2, -1e9)
    n3 = (torch.rand(neg2.shape, generator=g) + neg2).topk(RAND, 1).indices
    return torch.cat([n1, n2, n3], 1)


def units(T, src, xi, ci, ctx, q):
    """Unit tensors for queries q(a = x[xi], b = M[ci]). Order: same-pair r = 0..R-1, then context (r, m), r major."""
    R, MM, XM, MXt = T['R'], T['MM'], src['XM'], src['MXt']
    Q = len(xi)
    s = lambda t: torch.sign(t).clamp(-1, 1) + (t == 0).float()  # logit > 0 -> +1, else -1
    same = torch.stack([s(XM[xi, ci]), s(MXt[xi, ci])], 2)  # (Q,R,2)
    same = torch.cat([same, torch.zeros(Q, R, 2)], 2)
    smask = torch.cat([torch.ones(Q, R, 2), torch.zeros(Q, R, 2)], 2)
    smask[torch.arange(Q), q, 0] = 0  # the query's own value is masked
    xm = XM[xi[:, None], ctx]  # (Q,K,R)
    mx = MXt[xi[:, None], ctx]
    cm = MM[ci[:, None], ctx]
    mc = MM[ctx, ci[:, None]]
    cv = torch.stack([s(xm), s(mx), s(cm), s(mc)], 3).permute(0, 2, 1, 3).reshape(Q, R * K, 4)
    vals = torch.cat([same, cv], 1)
    mask = torch.cat([smask, torch.ones(Q, R * K, 4)], 1)
    r = torch.cat([torch.arange(R), torch.arange(R).repeat_interleave(K)])
    typ = torch.cat([torch.zeros(R, dtype=torch.long), torch.ones(R * K, dtype=torch.long)])
    return vals * mask, mask, r, typ


def sample(T, key, n, g, ops_from=None):
    src = T['kinds'][key]
    self_set = key[1] == 'self'
    xi = torch.randint(len(src['x']), (n,), generator=g)
    ci = torch.randint(len(T['M']), (n,), generator=g)
    if self_set:  # a != b
        ci = torch.where(ci == xi, (ci + 1) % len(T['M']), ci)
    q = torch.randint(T['R'], (n,), generator=g) if ops_from is None else torch.full((n,), ops_from)
    ctx = contexts(T, src, xi, ci, g, self_set)
    vals, mask, r, typ = units(T, src, xi, ci, ctx, q)
    truth = EQ.op_targets(src['lab'][xi], T['labM'][ci])[torch.arange(n), q]
    reader = src['XM'][xi, ci, q]
    return {'src': src, 'xi': xi, 'ci': ci, 'q': q, 'ctx': ctx, 'vals': vals, 'mask': mask, 'r': r, 'typ': typ,
            'truth': truth, 'reader_logit': reader}


def train_critic(T, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    critic = CC.Critic(T['R'])
    lM = T['labM']
    off = ~torch.eye(len(lM), dtype=torch.bool)
    ya, yb = lM[:, None].expand(len(lM), len(lM)), lM[None].expand(len(lM), len(lM))
    tt = EQ.op_targets(ya[off], yb[off])[:, :T['R']]
    pos = tt.mean(0)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(CRITIC_STEPS):
        b = sample(T, (C, 'self'), BATCH, g)
        p = torch.rand(BATCH, 1, generator=g) * .9 + .1
        present = torch.rand(b['vals'].shape[:2], generator=g) < p
        lg = critic(b['q'], b['vals'], b['mask'], b['r'], b['typ'], present)
        w = torch.where(b['truth'] > 0, ((1 - pos) / pos)[b['q']], torch.ones(BATCH))
        loss = (F.binary_cross_entropy_with_logits(lg, b['truth'], reduction='none') * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return critic.eval(), float(loss.detach())


@torch.no_grad()
def influence(critic, b):
    """Leave-one-out influence of each unit on the full critic's prediction: |sigma(full) - sigma(full without u)|."""
    Q, U = b['vals'].shape[:2]
    full = torch.sigmoid(critic(b['q'], b['vals'], b['mask'], b['r'], b['typ'], torch.ones(Q, U, dtype=torch.bool)))
    out = torch.empty(Q, U)
    for s in range(0, Q, 256):
        e = slice(s, min(Q, s + 256))
        n = e.stop - e.start
        pres = ~torch.eye(U, dtype=torch.bool)[None].expand(n, U, U).reshape(n * U, U)
        rep = lambda t: t[e].repeat_interleave(U, 0)
        lo = torch.sigmoid(critic(rep(b['q']), rep(b['vals']), rep(b['mask']), b['r'], b['typ'], pres)).view(n, U)
        out[e] = (full[e, None] - lo).abs()
    return full, out


def sel_inputs(T, b):
    return b['q'], b['src']['x'][b['xi']], T['M'][b['ci']], T['M'][b['ctx']], b['r'], b['typ']


def train_selector(T, critic, keys, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 1)
    parts = [sample(T, key, SEL_QUERIES // len(keys), g) for key in keys]
    data = []
    for b in parts:
        _, inf = influence(critic, b)
        data.append((sel_inputs(T, b), inf))
    sel = CC.Selector(T['R'], T['M'].shape[1])
    opt = torch.optim.Adam(sel.parameters(), lr=1e-3)
    cat = lambda i: torch.cat([d[0][i] for d in data])
    q, ca, cb, cm = cat(0), cat(1), cat(2), cat(3)
    r, typ = data[0][0][4], data[0][0][5]
    inf = torch.cat([d[1] for d in data])
    tgt = torch.where(inf.sum(1, keepdim=True) > 0, inf / inf.sum(1, keepdim=True).clamp_min(1e-12), torch.full_like(inf, 1 / inf.shape[1]))
    for _ in range(SEL_STEPS):
        idx = torch.randint(len(q), (BATCH,), generator=g)
        sc = sel(q[idx], ca[idx], cb[idx], cm[idx], r, typ)
        loss = -(tgt[idx] * sc.log_softmax(1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sel.eval(), float(loss.detach())


def auroc(score, y):
    y = y.bool()
    if y.all() or (~y).any() is False or not y.any():
        return None
    order = score.argsort()
    ranks = torch.empty(len(score))
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float)
    # average ranks over ties
    s_sorted = score[order]
    uniq, inv, cnt = torch.unique_consecutive(s_sorted, return_inverse=True, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    avg = (start + (cnt + 1) / 2).float()
    ranks[order] = avg[inv]
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def topk_present(scores, k):
    U = scores.shape[1]
    k = min(k, U)
    pres = torch.zeros_like(scores, dtype=torch.bool)
    pres.scatter_(1, scores.topk(k, 1).indices, True)
    return pres


@torch.no_grad()
def evaluate(T, critic, selectors, seed):
    g = torch.Generator().manual_seed(seed + 2)
    R = T['R']
    out = {}
    for kind in KINDS:
        key = (kind, 'test')
        for qi, op in enumerate(EQ.OPS[:R]):
            b = sample(T, key, EVAL_Q, g, ops_from=qi)
            Q, U = b['vals'].shape[:2]
            o = (b['reader_logit'] > 0).float()
            err = o != b['truth']

            def conflict(present):
                p = torch.sigmoid(critic(b['q'], b['vals'], b['mask'], b['r'], b['typ'], present))
                return torch.where(o > 0, 1 - p, p)

            full_p, inf = influence(critic, b)
            full = torch.where(o > 0, 1 - full_p, full_p)
            r = {'n': Q, 'error_rate': float(err.float().mean()), 'units': U,
                 'auroc': {'full': auroc(full, err), 'margin': auroc(-b['reader_logit'].abs(), err)}}
            scores = {'random': torch.rand(Q, U, generator=g), 'oracle': inf}
            for name, sel in selectors.items():
                scores[name] = sel(*sel_inputs(T, b))
            for name, sc in scores.items():
                r['auroc'][name] = {str(k): auroc(conflict(topk_present(sc, k)), err) for k in KS}
            # does conflict add to the reader's own margin? mean of the two ranks, no fitting
            rank = lambda v: v.argsort().argsort().float()
            mg = rank(-b['reader_logit'].abs())
            r['auroc']['margin+full'] = auroc(mg + rank(full), err)
            for name in selectors:
                r['auroc'][f'margin+{name}_k3'] = auroc(mg + rank(conflict(topk_present(scores[name], 3))), err)
            pattern = {}
            for name in ('oracle', *selectors):
                pres = topk_present(scores[name], 3).float()
                pattern[name] = {f"{'same' if t == 0 else 'ctx'}:{EQ.OPS[rr]}": float(pres[:, (b['typ'] == t) & (b['r'] == rr)].sum(1).mean())
                                 for t in (0, 1) for rr in range(R)}
            r['selection_k3'] = pattern
            if op == 'eq':
                acc = o > 0
                hand = acc & ((b['src']['XM'][b['xi'], b['ci'], 1] > 0) | (b['src']['MXt'][b['xi'], b['ci'], 1] > 0))
                nh = int(hand.sum())
                e = {'accepted': int(acc.sum()), 'false_share_of_accepted': float(err[acc].float().mean()) if acc.any() else None,
                     'hand_flags': nh, 'hand_precision': float(err[hand].float().mean()) if nh else None}
                for name, sc in (('full', full), *((f'{nm}_k3', conflict(topk_present(scores[nm], 3))) for nm in selectors)):
                    if nh and acc.any():
                        s_ = torch.where(acc, sc, torch.full_like(sc, -1.))
                        top = s_.topk(min(nh, int(acc.sum()))).indices
                        e[f'{name}_precision_at_hand_rate'] = float(err[top].float().mean())
                r['eq_accepted'] = e
            out[f'{SHORT[kind]}/{op}'] = r
    return out


def run(t):
    arm, seed, block = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    model = DG.load(arm, seed, block)
    T = tables(model, block)
    tp = (time.perf_counter() - t0, time.process_time() - c0)
    critic, closs = train_critic(T, seed)
    tc = (time.perf_counter() - t0, time.process_time() - c0)
    selectors, slosses = {}, {}
    for name, keys in (('sel-coll', [(C, 'self')]), ('sel-seen', [(C, 'self')] + [(k, 'fit') for k in SEEN])):
        selectors[name], slosses[name] = train_selector(T, critic, keys, seed)
    ts = (time.perf_counter() - t0, time.process_time() - c0)
    res = evaluate(T, critic, selectors, seed)
    te = (time.perf_counter() - t0, time.process_time() - c0)
    R = T['R']
    res['_train'] = {'critic_final_loss': closs, 'selector_final_loss': slosses,
                     'critic_parameters': sum(p.numel() for p in critic.parameters()),
                     'selector_parameters': sum(p.numel() for p in selectors['sel-coll'].parameters()),
                     'reader_S_calls_per_query_full': 2 * R + 4 * R * K, 'pair_features_per_query_full': 2 + 4 * K}
    phases = {'tables': tp, 'critic': (tc[0] - tp[0], tc[1] - tp[1]), 'selectors': (ts[0] - tc[0], ts[1] - tc[1]),
              'evaluate': (te[0] - ts[0], te[1] - ts[1])}
    rt = {ph: {'wall_seconds': w, 'cpu_seconds': c} for ph, (w, c) in phases.items()}
    rt['peak_rss_kb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return '{}/seed{}/{}'.format(*t), res, rt


def main():
    t0 = time.perf_counter()
    RP.prepare()
    tasks = [(a, s, b) for a in ARMS for s in SEEDS for b in BLOCKS]
    if SMOKE:
        tasks = [('two-ctx-enbuf', SEEDS[0], 'block3')]
    workers = int(os.environ.get('CC_WORKERS', 12))
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(run, RUNS / 'done', '{0}-seed{1}-{2}'), tasks, chunksize=1)
    RP.S0.save_json(f'{REPORT}.json', {'status': 'consistency critic with learned probe selection v1' + (' (smoke)' if SMOKE else ''),
                                        'arms': ARMS, 'seeds': SEEDS, 'blocks': BLOCKS, 'K_contexts': K, 'ks': KS,
                                        'critic_steps': CRITIC_STEPS, 'selector_steps': SEL_STEPS, 'eval_queries': EVAL_Q,
                                        'results': {k: r for k, r, _ in res}})
    RP.S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
        'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT) if CONTRACT.exists() else None,
        'runner_sha256': nc.sha256(__file__), 'module_sha256': nc.sha256(CC.__file__),
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(),
        'workers': workers, 'threads_per_worker': 1, 'dtype': 'float32', 'device': 'cpu',
        'wall_seconds': time.perf_counter() - t0,
        'aggregate_worker_cpu_seconds': float(sum(sum(p['cpu_seconds'] for k, p in t.items() if isinstance(p, dict)) for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary():
    rep = json.loads(REPORT.with_suffix('.json').read_text())
    res = rep['results']
    keys = sorted(res)
    f = lambda v: f'{np.mean(v):.2f}' if v else '-'
    seeds = lambda arm, block: [k for k in keys if k.startswith(arm + '/') and k.endswith(block)]
    for block in BLOCKS:
        arms = [a for a in ARMS if seeds(a, block)]
        if not arms:
            continue
        ops = EQ.OPS[:3 if block == 'block3' else 2]
        for op in ops:
            print(f'\n## {block} {op} queries: AUROC for reader error | full / margin / learned k=1,3 (coll, seen) / random k=3 / oracle k=3')
            for arm in arms:
                cells = []
                for kd in ('coll', 'zh', 'dev', 'en', 'thai'):
                    rs = [res[k][f'{kd}/{op}']['auroc'] for k in seeds(arm, block)]
                    g = lambda *p: f([_dig(x, p) for x in rs if _dig(x, p) is not None])
                    cells.append(f"{kd} {g('full')}/{g('margin')}/{g('sel-coll', '1')},{g('sel-coll', '3')} {g('sel-seen', '3')}/{g('random', '3')}/{g('oracle', '3')}")
                print(f'| {arm} | ' + ' | '.join(cells) + ' |')
        print(f'\n## {block} eq accepted: false share / hand precision / full / sel-coll k3 / sel-seen k3 precision at hand rate')
        for arm in arms:
            cells = []
            for kd in ('coll', 'zh', 'dev', 'en', 'thai'):
                rs = [res[k][f'{kd}/eq']['eq_accepted'] for k in seeds(arm, block)]
                g = lambda q: f([x[q] for x in rs if x.get(q) is not None])
                cells.append(f"{kd} {g('false_share_of_accepted')}/{g('hand_precision')}/{g('full_precision_at_hand_rate')}/"
                             f"{g('sel-coll_k3_precision_at_hand_rate')}/{g('sel-seen_k3_precision_at_hand_rate')}")
            print(f'| {arm} | ' + ' | '.join(cells) + ' |')
        print(f'\n## {block} k-curve (mean over kinds coll, zh, dev, en), AUROC by k: sel-coll | sel-seen | random | oracle ; full')
        for arm in arms:
            for op in ops:
                rs = [res[k][f'{kd}/{op}']['auroc'] for k in seeds(arm, block) for kd in ('coll', 'zh', 'dev', 'en')]
                row = []
                for nm in ('sel-coll', 'sel-seen', 'random', 'oracle'):
                    row.append(','.join(f([x[nm][str(kk)] for x in rs if x[nm][str(kk)] is not None]) for kk in KS))
                print(f"| {arm} {op} | " + ' | '.join(row) + f" ; {f([x['full'] for x in rs if x['full'] is not None])} |")
        print(f'\n## {block} selection at k=3 (mean units of each kind), zh queries: oracle | sel-coll | sel-seen')
        for arm in arms:
            for op in ops:
                ps = [res[k][f'zh/{op}']['selection_k3'] for k in seeds(arm, block)]
                row = []
                for nm in ('oracle', 'sel-coll', 'sel-seen'):
                    cats = ps[0][nm].keys()
                    row.append(' '.join(f"{c}={np.mean([p[nm][c] for p in ps]):.2f}" for c in cats))
                print(f'| {arm} {op} | ' + ' | '.join(row) + ' |')


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
