"""Drift remedies v1 (docs/drift-remedies-v1.md): rehearsal, online EWC and per-kind registration (comparison only) on
equality v2's two-ctx-enbuf, in two script orders.

Usage: .venv/bin/python -m experiments.drift_remedies_v1 select     (EWC lambda on seeds 310, 311; DR_WORKERS, default 8)
       .venv/bin/python -m experiments.drift_remedies_v1            (main; rerun to resume)
       .venv/bin/python -m experiments.drift_remedies_v1 summary
       DR_SMOKE=1 ... for a short check (20 steps per block)
"""
import json
import os
import platform
import resource
import sys
import time

from experiments import equality_relations_v2 as V2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments import durable as DUR

RP, EQ, RC, D, X, nc, ns = V2.RP, V2.EQ, V2.RC, V2.D, V2.X, V2.nc, V2.ns
SMOKE = bool(os.environ.get('DR_SMOKE'))
NAME = 'drift-remedies-v1' + ('-smoke' if SMOKE else '')
CONTRACT = X.ROOT / 'docs' / 'drift-remedies-v1.md'
RUNS, REPORT = X.ROOT / 'runs' / NAME, X.ROOT / 'reports' / NAME
STEPS = 20 if SMOKE else RP.STEPS
C, ZH, DEV, EN, TH = V2.C, V2.ZH, V2.DEV, V2.EN, V2.TH
N, SHORT, LR, COLL_PER, FORM_PER = V2.N, V2.SHORT, V2.LR, V2.COLL_PER, V2.FORM_PER
BUF_STORE, BUF_PER = V2.BUF_STORE, V2.BUF_PER
ORDERS = {'A': (EN, ZH, DEV), 'B': (EN, DEV, ZH)}
ARMS = ('none', 'reh1', 'reh2', 'reh4', 'ewc', 'ewc-reh1', 'reg')
SEEDS, SEL_SEEDS, LAMBDAS = V2.SEEDS, (310, 311), (0.1, 1.0, 10.0, 100.0, 1000.0)
FISHER_BATCHES = 4 if SMOKE else 64
ADAPTER_H = 16
SELECTED = RUNS / 'selected_lambda.json'


def reh_n(arm):
    return {'reh1': 1, 'reh2': 2, 'reh4': 4, 'ewc-reh1': 1}.get(arm, 0)


class Adapters(nn.Module):
    """Residual registration per kind: c = E(x) + B_k ReLU(A_k x), B_k zero-initialized (comparison arm only)."""

    def __init__(self, kinds, width, generator):
        super().__init__()
        self.mods = nn.ModuleDict()
        for k in kinds:
            a, b = nn.Linear(width, ADAPTER_H), nn.Linear(ADAPTER_H, RC.D)
            with torch.no_grad():
                a.weight.copy_(torch.randn(a.weight.shape, generator=generator) / width ** .5)
                a.bias.zero_(), b.weight.zero_(), b.bias.zero_()
            self.mods[k] = nn.Sequential(a, nn.ReLU(), b)


def attach_registration(model, adapters):
    """Replace model.codes so that rows of a kind with an adapter get its residual. model.group_kinds maps group id -> kind."""
    base_codes = model.codes

    def codes(x, groups):
        c = model.E(x)
        add = torch.zeros_like(c)
        for gid in torch.unique(groups).tolist():
            kind = model.group_kinds.get(gid)
            if kind in adapters.mods:
                m = groups == gid
                add[m] = adapters.mods[kind](x[m])
        c = c + add
        return RC.context_norm(c, groups) if model.arm in RC.CTX else c

    model.codes, model.base_codes = codes, base_codes


def eval_group_kinds(model):
    model.group_kinds = {g: kind for g, kind in enumerate(RP.DATA['eval'])}


@torch.no_grad()
def judgement(model, kind, block):
    """As the equality diagnostics: test half of `kind` against the collection fit half."""
    ev = RP.DATA['eval']
    eval_group_kinds(model)
    codes = {}
    for g, kd in enumerate(ev):
        if kd in (C, kind):
            x = torch.from_numpy(np.concatenate([ev[kd][v] for v in ns.NUMBERS]))
            codes[kd] = model.codes(x, torch.full((len(x),), g))
    k = len(ev[C][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    test = (torch.arange(len(lab)) % k) >= k // 2
    M, lM, x, y = codes[C][~test], lab[~test], codes[kind][test], lab[test]
    mat = lambda a, b: model.op_logits(a.repeat_interleave(len(b), 0), b.repeat(len(a), 1)).reshape(len(a), len(b), -1)
    L1, L2 = mat(x, M), mat(M, x).transpose(0, 1)
    acc = L1[..., 0] > 0
    dist = (y[:, None] - lM[None]).abs()
    out = {'false_accept_adjacent': float(acc[dist == 1].float().mean()), 'false_accept_far': float(acc[dist >= 2].float().mean()),
           'positive_recall': float(acc[dist == 0].float().mean())}
    # substitution against truth among accepted pairs: the form's own outputs with every collection item
    cs = torch.stack([L1[:, lM == v, 0].mean(1) for v in range(1, N + 1)], 1)
    accepted = cs.max(1).values > 0
    bal = lambda p, t: float(.5 * (p[t].float().mean() + (~p[~t]).float().mean())) if t.any() and (~t).any() else None
    if accepted.any():
        yy, mm = y[accepted][:, None].expand(-1, len(lM)), lM[None].expand(int(accepted.sum()), -1)
        ops = [(1, 'more', mm > yy, yy > mm)] + ([(2, 'sum10', (yy + mm) == 10, (yy + mm) == 10)] if block == 'block3' else [])
        for oi, op, t1, t2 in ops:
            v = [w for w in (bal(L1[accepted][..., oi] > 0, t1), bal(L2[accepted][..., oi] > 0, t2)) if w is not None]
            out[f'substitution_truth_{op}'] = float(np.mean(v)) if v else None
    out['accepted_share'] = float(accepted.float().mean())
    return out


@torch.no_grad()
def kind_codes(model, kind):
    ev = RP.DATA['eval']
    eval_group_kinds(model)
    g = list(ev).index(kind)
    x = torch.from_numpy(np.concatenate([ev[kind][v] for v in ns.NUMBERS]))
    return model.codes(x, torch.full((len(x),), g))


def run(arm, order, seed, lam=None):
    RP.prepare()
    DATA = RP.DATA
    blocks = [(f'block{i + 1}', lang) for i, lang in enumerate(ORDERS[order])]
    earlier = ORDERS[order][1]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    v2arm = 'two-ctx-enzhbuf' if reh_n(arm) else 'two-ctx-enbuf'  # v2's training stream for the matching design
    tag = 'equality_relations_v2' if order == 'A' else 'drift_remedies_v1_orderB'
    rng = nc.stream(tag, seed, EQ.ARMS.index(v2arm))
    brng = nc.stream('equality_buffer', seed)
    frng = nc.stream('drift_remedies_v1_fisher', seed)
    withheld = {k: tuple(int(x) for x in v) for k, v in D.withheld_counts(seed).items()}
    width = DATA['coll_train'].shape[-1]
    model = EQ.RelNet('two-ctx-enbuf', width, g)
    model.group_kinds = {}
    adapters = None
    if arm == 'reg':
        adapters = Adapters(ORDERS[order][1:], width, torch.Generator().manual_seed(seed + 1000))
        attach_registration(model, adapters)
    n_ep = len(DATA['coll_train']) * 5 // 6
    buf = {}
    for lang in (EN, earlier):  # as v2: English, then the block-2 script, from the same buffer stream
        fx, fy = DATA['form_train'][lang]
        idx = np.concatenate([brng.choice(np.flatnonzero(fy == v), BUF_STORE, replace=False)
                              for v in ns.NUMBERS if v not in set(withheld.get(lang, ()))])
        buf[lang] = (fx[idx], fy[idx])
    n = reh_n(arm)
    if n:  # keep the first n stored codes per count (nested buffers)
        bx, by = buf[earlier]
        keep = np.concatenate([np.flatnonzero(by == v)[:n] for v in np.unique(by)])
        buf[earlier] = (bx[keep], by[keep])
    fisher, anchor = None, None
    lam = lam if lam is not None else (json.loads(SELECTED.read_text())['lambda'] if arm.startswith('ewc') else None)
    snaps, steps, extra = {}, 0, {}
    prev_codes = None

    def batch(lang, by_n, taught, replay, coll_ops, r):
        ep = r.integers(n_ep, size=(N, COLL_PER))
        xc = DATA['coll_train'][ep, np.arange(N)[:, None]].reshape(-1, width)
        yc = np.repeat(np.arange(1, N + 1), COLL_PER)
        fx, fy = DATA['form_train'][lang]
        fi = np.concatenate([r.choice(by_n[v], FORM_PER, replace=False) for v in taught])
        parts, kinds = [(xc, yc), (fx[fi], fy[fi])], [C, lang]
        for name in replay:
            bx, by = buf[name]
            bsel = np.concatenate([r.choice(np.flatnonzero(by == v), min(BUF_PER, int((by == v).sum())), replace=False) for v in np.unique(by)])
            parts.append((bx[bsel], by[bsel]))
            kinds.append(name)
        x = torch.from_numpy(np.concatenate([p[0] for p in parts]))
        grp = torch.cat([torch.full((len(p[0]),), i, dtype=torch.long) for i, p in enumerate(parts)])
        model.group_kinds = dict(enumerate(kinds))
        c = model.codes(x, grp)
        cs = [c[grp == i] for i in range(len(parts))]
        ys = [torch.from_numpy(p[1].astype(np.int64)) for p in parts]
        L = RC.loss(model, cs[0], ys[0], cs[1], ys[1]) + EQ.op_loss(model, cs[0], ys[0], cs[0], ys[0], coll_ops)
        for kk in range(2, len(parts)):
            L = L + EQ.label_loss(model, cs[0], ys[0], cs[kk], ys[kk])
        return L

    named = lambda: [(nm, p) for nm, p in list(model.named_parameters()) + ([(f'ad.{a}', b) for a, b in adapters.named_parameters()] if adapters else [])]
    for bi, (block, lang) in enumerate(blocks):
        if arm == 'reg':
            if bi == 1:
                for p in model.E.parameters():
                    p.requires_grad_(False)
            for k, m in adapters.mods.items():
                for p in m.parameters():
                    p.requires_grad_(k == lang)
        params = [p for _, p in named() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=LR)
        fx, fy = DATA['form_train'][lang]
        taught = [v for v in ns.NUMBERS if v not in set(withheld.get(lang, ()))]
        by_n = {v: np.flatnonzero(fy == v) for v in taught}
        replay = [EN] if bi >= 1 else []
        if n and bi >= 2:
            replay.append(earlier)
        coll_ops = [0, 1, 2] if block == 'block3' else [0, 1]
        for _ in range(STEPS):
            L = batch(lang, by_n, taught, replay, coll_ops, rng)
            if fisher is not None:
                L = L + lam / 2 * sum((fisher[nm] * (p - anchor[nm]) ** 2).sum() for nm, p in named() if nm in fisher)
            opt.zero_grad()
            L.backward()
            opt.step()
            steps += 1
        if fisher is not None:
            with torch.no_grad():
                extra.setdefault('penalty_end_of_block', {})[block] = float(lam / 2 * sum(
                    (fisher[nm] * (p - anchor[nm]) ** 2).sum() for nm, p in named() if nm in fisher))
        if arm.startswith('ewc'):  # empirical Fisher of this block's own loss, summed over blocks; anchor at block end
            t0 = time.perf_counter()
            fb = {nm: torch.zeros_like(p) for nm, p in named()}
            for _ in range(FISHER_BATCHES):
                model.zero_grad()
                batch(lang, by_n, taught, replay, coll_ops, frng).backward()
                for nm, p in named():
                    if p.grad is not None:
                        fb[nm] += p.grad.detach() ** 2 / FISHER_BATCHES
            mean = torch.cat([v.flatten() for v in fb.values()]).mean().clamp_min(1e-12)
            fb = {k: v / mean for k, v in fb.items()}  # normalized: mean importance 1 per block
            fisher = fb if fisher is None else {k: fisher[k] + fb[k] for k in fb}
            anchor = {nm: p.detach().clone() for nm, p in named()}
            model.zero_grad()
            extra.setdefault('fisher_seconds', 0.)
            extra['fisher_seconds'] += time.perf_counter() - t0
        eval_group_kinds(model)
        snaps[block] = RP.evaluate(model, withheld)
        snaps[block]['final_loss'] = float(L.detach())
        eval_group_kinds(model)
        snaps[block]['operators'] = V2.op_eval(model, withheld)
        snaps[block]['judgement'] = {SHORT[k]: judgement(model, k, block) for k in (ZH, DEV, EN, TH)}
        cur = kind_codes(model, earlier)
        if prev_codes is not None:
            spread = ((prev_codes - prev_codes.mean(0)) ** 2).sum(1).mean().sqrt()
            snaps[block]['earlier_movement'] = float((cur - prev_codes).norm(dim=1).mean() / spread)
        prev_codes = cur if block == 'block2' else prev_codes
    bytes_ = {'stored_codes': int(sum(len(b[0]) for k, b in buf.items() if k == earlier) * width * 4) if n else 0,
              'ewc_state': int(2 * sum(p.numel() for _, p in named()) * 4) if arm.startswith('ewc') else 0,
              'adapters': int(sum(p.numel() for p in adapters.parameters()) * 4) if adapters else 0}
    return snaps, withheld, {'optimizer_steps': steps, 'retained_bytes_remedy': bytes_, 'lambda': lam, **extra}


def task(t):
    arm, order, seed = t[:3]
    lam = t[3] if len(t) > 3 else None
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    snaps, withheld, rt = run(arm, order, seed, lam)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    key = f'{arm}/{order}/seed{seed}' + (f'/lambda{lam:g}' if len(t) > 3 else '')
    return key, {'snapshots': snaps, 'withheld': withheld, 'retained_bytes_remedy': rt['retained_bytes_remedy'], 'lambda': rt['lambda']}, rt


def pool_run(tasks, fmt, out):
    t0 = time.perf_counter()
    RP.prepare()
    workers = int(os.environ.get('DR_WORKERS', 8))
    with __import__('multiprocessing').get_context('fork').Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(DUR.durable(task, RUNS / 'done', fmt), tasks, chunksize=1)
    rt = {'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
          'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
          'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(),
          'workers': workers, 'threads_per_worker': 1, 'dtype': 'float32', 'device': 'cpu', 'wall_seconds': time.perf_counter() - t0,
          'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
          'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}}
    RP.S0.save_json(f'{out}.json', {'status': 'drift remedies v1' + (' (smoke)' if SMOKE else ''), 'steps_per_block': STEPS,
                                    'results': {k: r for k, r, _ in res}})
    RP.S0.save_json(f'{out}-runtime.json', rt)
    return {k: r for k, r, _ in res}


def criterion(r):
    b3 = r['snapshots']['block3']['operators']['eq_identity']
    return .5 * (b3['zh'] + b3['dev'])  # order A: earlier script Chinese, new script Devanagari


def select():
    tasks = [('ewc', 'A', s, lam) for lam in LAMBDAS for s in SEL_SEEDS]
    res = pool_run(tasks, 'sel-{0}-{1}-seed{2}-lambda{3:g}', f'{REPORT}-selection')
    score = {lam: float(np.mean([criterion(res[f'ewc/A/seed{s}/lambda{lam:g}']) for s in SEL_SEEDS])) for lam in LAMBDAS}
    best = max(LAMBDAS, key=lambda l: (score[l], -l))
    RUNS.mkdir(parents=True, exist_ok=True)
    DUR.atomic_write(SELECTED, json.dumps({'lambda': best, 'scores': {str(k): v for k, v in score.items()}}).encode())
    print('scores', score, 'selected', best)


def main():
    if not SELECTED.exists():
        raise SystemExit('run `select` first')
    tasks = [(a, o, s) for o in ORDERS for a in ARMS for s in SEEDS]
    pool_run(tasks, '{0}-{1}-seed{2}', REPORT)


def reproduction(res):
    v2 = json.loads((X.ROOT / 'reports' / 'equality-relations-v2.json').read_text())['results']
    out = {}
    for arm, v2arm in (('none', 'two-ctx-enbuf'), ('reh4', 'two-ctx-enzhbuf')):
        for s in SEEDS:
            a = DUR.roundtrip(res[f'{arm}/A/seed{s}']['snapshots'])
            b = v2[f'{v2arm}/seed{s}']['snapshots']
            out[f'{arm}/seed{s}'] = all(a[bl][k] == b[bl][k] for bl in b for k in b[bl])
    return out


def summary():
    rep = json.loads(REPORT.with_suffix('.json').read_text())
    res = rep['results']
    print('reproduction of v2 (identity, relations, operators):', reproduction(res))
    f = lambda v: f'{np.mean(v):.2f}' if v else '-'
    for order in ORDERS:
        earlier, new = SHORT[ORDERS[order][1]], SHORT[ORDERS[order][2]]
        print(f'\n## order {order}: block 3 eq identity earlier {earlier} / new {new} / en / thai | earlier: block2 eq id, '
              f'movement, false adj, subst more | bytes')
        for arm in ARMS:
            rs = [res[f'{arm}/{order}/seed{s}'] for s in SEEDS if f'{arm}/{order}/seed{s}' in res]
            if not rs:
                continue
            g = lambda bl, kd: f([r['snapshots'][bl]['operators']['eq_identity'][kd] for r in rs])
            j = lambda q: f([r['snapshots']['block3']['judgement'][earlier][q] for r in rs if r['snapshots']['block3']['judgement'][earlier].get(q) is not None])
            print(f"| {arm} | {g('block3', earlier)} / {g('block3', new)} / {g('block3', 'en')} / {g('block3', 'thai')} | "
                  f"{g('block2', earlier)}, {f([r['snapshots']['block3']['earlier_movement'] for r in rs])}, {j('false_accept_adjacent')}, "
                  f"{j('substitution_truth_more')} | {sum(rs[0]['retained_bytes_remedy'].values())}")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'main'
    {'select': select, 'main': main, 'summary': summary}[cmd]()
