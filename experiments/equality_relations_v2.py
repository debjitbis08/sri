"""Equality as a learned relation v2 (docs/equality-relations-v2.md). Trained pilot on frozen sense codes, Part C network.

Usage: .venv/bin/python -m experiments.equality_relations_v2            (EQ_WORKERS, default 4; rerun to resume)
       .venv/bin/python -m experiments.equality_relations_v2 summary
       EQ_SMOKE=1 ... for a short check (20 steps per block)
"""
import os
import platform
import resource
import sys
import time

from experiments import relation_codes_pilot as RP  # sets up the pilot's data
import numpy as np
import torch

from experiments import durable as DUR
from src import equality_net_v2 as EQ
from src import relation_codes as RC

T, D, X, nc, ns = RP.T, RP.D, RP.X, RP.nc, RP.ns
SMOKE = bool(os.environ.get('EQ_SMOKE'))
NAME = 'equality-relations-v2' + ('-smoke' if SMOKE else '')
CONTRACT = X.ROOT / 'docs' / 'equality-relations-v2.md'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
SEEDS = RP.SEEDS
WORKERS = int(os.environ.get('EQ_WORKERS', 4))
STEPS = 20 if SMOKE else RP.STEPS
LR, COLL_PER, FORM_PER = RP.LR, RP.COLL_PER, RP.FORM_PER
BUF_STORE, BUF_PER = 4, 2  # stored codes per count (as v1); replayed per count per step
BLOCKS, SHORT, N = RP.BLOCKS, RP.SHORT, RP.N
C, ZH, DEV, EN, TH = 'collection', 'chinese', 'devanagari_digits', 'english', 'thai_digits'
ACC_PAIRS = ((C, C), (ZH, EN), (EN, ZH), (DEV, EN), (EN, DEV), (ZH, DEV), (DEV, ZH), (ZH, ZH), (DEV, DEV), (EN, EN),
             (ZH, C), (C, ZH), (DEV, C), (C, DEV), (EN, C), (TH, C))
PROP_PAIRS = ((C, C), (ZH, EN), (DEV, EN), (ZH, DEV), (ZH, ZH), (DEV, DEV), (ZH, C), (DEV, C), (TH, C))
SUBST = (ZH, DEV, EN, TH)


@torch.no_grad()
def op_eval(model, withheld):
    model.eval()
    ev = RP.DATA['eval']
    codes = {}
    for g, kind in enumerate(ev):
        x = torch.from_numpy(np.concatenate([ev[kind][n] for n in ns.NUMBERS]))
        codes[kind] = model.codes(x, torch.full((len(x),), g))
    k = len(ev[C][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    n = len(lab)
    ya, yb = lab.repeat_interleave(n), lab.repeat(n)
    t = EQ.op_targets(ya, yb).bool()
    out = {'balanced_accuracy': {}, 'properties': {}, 'substitution': {}, 'eq_identity': {}}
    pred = {}
    for a, b in set(ACC_PAIRS) | {(q, p) for p, q in PROP_PAIRS} | set(PROP_PAIRS):
        pred[(a, b)] = model.op_logits(codes[a].repeat_interleave(n, 0), codes[b].repeat(n, 1)) > 0
    bal = lambda p, tt: float(.5 * (p[tt].float().mean() + (~p[~tt]).float().mean()))
    for a, b in ACC_PAIRS:
        out['balanced_accuracy'][f'{SHORT[a]}>{SHORT[b]}'] = {op: bal(pred[(a, b)][:, i], t[:, i]) for i, op in enumerate(EQ.OPS)}
    # properties: compare out(a_i, b_j) with out(b_j, a_i); the reversed pair index is j * n + i
    rev = (torch.arange(n).repeat(n) * n + torch.arange(n).repeat_interleave(n))
    unequal = ya != yb
    for a, b in PROP_PAIRS:
        pab, pba = pred[(a, b)], pred[(b, a)][rev]
        r = {}
        for i, op in enumerate(EQ.OPS):
            if EQ.SYMMETRIC[op]:
                r[op + '_symmetric'] = float((pab[:, i] == pba[:, i]).float().mean())
            else:
                r[op + '_antisymmetric'] = float((pab[unequal, i] != pba[unequal, i]).float().mean())
                r[op + '_irreflexive'] = float((~pab[~unequal, i] & ~pba[~unequal, i]).float().mean())
        out['properties'][f'{SHORT[a]}|{SHORT[b]}'] = r
    # substitution, derived (no loss): x against a same-count collection item (true pairing), a different-count item (null),
    # and the collection item the system itself judges equal (highest eq logit)
    M = codes[C]
    shifted = (torch.arange(n) + 3 * k) % n

    def agree(xs, cs):
        m = len(xs)
        X1, C1, M1 = xs.repeat_interleave(n, 0), cs.repeat_interleave(n, 0), M.repeat(m, 1)
        r = {}
        f1, c1, f2, c2 = model.op_logits(X1, M1) > 0, model.op_logits(C1, M1) > 0, model.op_logits(M1, X1) > 0, model.op_logits(M1, C1) > 0
        for i, op in enumerate(EQ.OPS[1:], start=1):
            r[op] = float(np.mean([bal(px[:, i], pc[:, i]) for px, pc in ((f1, c1), (f2, c2))]))
        return r

    for kind in SUBST:
        x = codes[kind]
        eqlog = model.op_logits(x.repeat_interleave(n, 0), M.repeat(n, 1))[:, 0].reshape(n, n)
        judged = eqlog.argmax(1)
        out['eq_identity'][SHORT[kind]] = float((lab[judged] == lab).float().mean())
        r = {f'true_{key}': v for key, v in agree(x, M).items()}
        r.update({f'null_{key}': v for key, v in agree(x, M[shifted]).items()})
        r.update({f'judged_{key}': v for key, v in agree(x, M[judged]).items()})
        out['substitution'][SHORT[kind]] = r
    model.train()
    return out


def run(arm, seed):
    RP.prepare()
    DATA = RP.DATA
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = nc.stream('equality_relations_v2', seed, EQ.ARMS.index(arm))
    brng = nc.stream('equality_buffer', seed)  # the same stored codes as v1 for every arm of a seed
    withheld = {k: tuple(int(x) for x in v) for k, v in D.withheld_counts(seed).items()}
    width = DATA['coll_train'].shape[-1]
    model = EQ.RelNet(arm, width, g)
    n_ep = len(DATA['coll_train']) * 5 // 6
    buf = {}
    for lang in (EN, ZH):
        fx, fy = DATA['form_train'][lang]
        idx = np.concatenate([brng.choice(np.flatnonzero(fy == v), BUF_STORE, replace=False)
                              for v in ns.NUMBERS if v not in set(withheld.get(lang, ()))])
        buf[lang] = (fx[idx], fy[idx])
    en_mode, zh_mode = EQ.buffer_modes(arm)
    snaps, steps = {}, 0
    for bi, (block, lang) in enumerate(BLOCKS):
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        fx, fy = DATA['form_train'][lang]
        taught = [v for v in ns.NUMBERS if v not in set(withheld.get(lang, ()))]
        by_n = {v: np.flatnonzero(fy == v) for v in taught}
        replay = [(name, mode) for name, mode, first in ((EN, en_mode, 1), (ZH, zh_mode, 2)) if mode and bi >= first]
        coll_ops = [0, 1, 2] if block == 'block3' else [0, 1]
        for _ in range(STEPS):
            ep = rng.integers(n_ep, size=(N, COLL_PER))
            xc = DATA['coll_train'][ep, np.arange(N)[:, None]].reshape(-1, width)
            yc = np.repeat(np.arange(1, N + 1), COLL_PER)
            fi = np.concatenate([rng.choice(by_n[v], FORM_PER, replace=False) for v in taught])
            parts = [(xc, yc), (fx[fi], fy[fi])]
            for name, _ in replay:
                bx, by = buf[name]
                bsel = np.concatenate([rng.choice(np.flatnonzero(by == v), BUF_PER, replace=False) for v in np.unique(by)])
                parts.append((bx[bsel], by[bsel]))
            x = torch.from_numpy(np.concatenate([p[0] for p in parts]))
            grp = torch.cat([torch.full((len(p[0]),), i, dtype=torch.long) for i, p in enumerate(parts)])
            c = model.codes(x, grp)
            cs = [c[grp == i] for i in range(len(parts))]
            ys = [torch.from_numpy(p[1].astype(np.int64)) for p in parts]
            L = RC.loss(model, cs[0], ys[0], cs[1], ys[1]) + EQ.op_loss(model, cs[0], ys[0], cs[0], ys[0], coll_ops)
            for kk, (name, mode) in enumerate(replay, start=2):
                if mode == 'label':
                    L = L + EQ.label_loss(model, cs[0], ys[0], cs[kk], ys[kk])
                else:  # current script first; 'both' adds the reversed order
                    L = L + EQ.op_loss(model, cs[1], ys[1], cs[kk], ys[kk], EQ.CROSS_OPS)
                    if mode == 'both':
                        L = L + EQ.op_loss(model, cs[kk], ys[kk], cs[1], ys[1], EQ.CROSS_OPS)
            opt.zero_grad()
            L.backward()
            opt.step()
            steps += 1
        snaps[block] = RP.evaluate(model, withheld)
        snaps[block]['final_loss'] = float(L.detach())
        snaps[block]['operators'] = op_eval(model, withheld)
    return snaps, withheld, {'optimizer_steps': steps}


def task(t):
    arm, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    snaps, withheld, rt = run(arm, seed)
    rt.update(wall_seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0,
              peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/seed{seed}', {'snapshots': snaps, 'withheld': withheld}, rt


def main():
    t0 = time.perf_counter()
    RP.prepare()
    tp = time.perf_counter() - t0
    tasks = [(a, s) for a in EQ.ARMS for s in SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    RP.S0.save_json(f'{REPORT}.json', {'status': 'equality as a learned relation v2' + (' (smoke)' if SMOKE else ''), 'arms': EQ.ARMS,
                                        'seeds': SEEDS, 'steps_per_block': STEPS, 'results': {k: r for k, r, _ in res}})
    RP.S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'),
        'dirty': bool(X.git('status', '--porcelain')), 'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__),
        'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'cpu': platform.processor(),
        'workers': WORKERS, 'threads_per_worker': 1, 'dtype': 'float32', 'device': 'cpu', 'prepare_seconds': tp,
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary():
    import json
    res = json.loads(REPORT.with_suffix('.json').read_text())['results']
    f = lambda v: f'{np.nanmean(v):.2f}' if v else '-'
    for block in ('block2', 'block3'):
        print(f'\n## {block}: identity zh/dev/en (best of nearest, relational) | eq identity zh/dev/en')
        for arm in EQ.ARMS:
            s = [res[f'{arm}/seed{x}']['snapshots'][block] for x in SEEDS]
            best = lambda kd: f([max(v['identity'][kd]['nearest'], v['identity'][kd]['relational']) for v in s])
            eqi = lambda kd: f([v['operators']['eq_identity'][kd] for v in s])
            print(f"| {arm} | {best('zh')}/{best('dev')}/{best('en')} | {eqi('zh')}/{eqi('dev')}/{eqi('en')} |")
        print(f'\n## {block}: balanced accuracy eq/more/sum10')
        keys = ('coll>coll', 'zh>en', 'en>zh', 'dev>en', 'en>dev', 'zh>dev', 'dev>zh', 'zh>zh', 'zh>coll', 'thai>coll')
        print('| arm | ' + ' | '.join(keys) + ' |')
        for arm in EQ.ARMS:
            o = [res[f'{arm}/seed{x}']['snapshots'][block]['operators'] for x in SEEDS]
            print(f'| {arm} | ' + ' | '.join('/'.join(f([v['balanced_accuracy'][kk][op] for v in o]) for op in EQ.OPS) for kk in keys) + ' |')
        print(f'\n## {block}: properties eq_sym / more_antisym / more_irrefl / sum10_sym')
        pk = ('coll|coll', 'zh|en', 'dev|en', 'zh|dev', 'zh|zh', 'zh|coll', 'thai|coll')
        print('| arm | ' + ' | '.join(pk) + ' |')
        for arm in EQ.ARMS:
            o = [res[f'{arm}/seed{x}']['snapshots'][block]['operators']['properties'] for x in SEEDS]
            print(f'| {arm} | ' + ' | '.join('/'.join(f([v[kk][m] for v in o]) for m in ('eq_symmetric', 'more_antisymmetric', 'more_irreflexive', 'sum10_symmetric')) for kk in pk) + ' |')
    print('\n## block3 substitution more/sum10: true | null | judged')
    for arm in EQ.ARMS:
        o = [res[f'{arm}/seed{x}']['snapshots']['block3']['operators']['substitution'] for x in SEEDS]
        print(f'| {arm} | ' + ' | '.join(kd + ' ' + ' | '.join('/'.join(f([v[kd][f'{w}_{op}'] for v in o]) for op in ('more', 'sum10')) for w in ('true', 'null', 'judged')) for kd in ('zh', 'dev', 'en', 'thai')) + ' |')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
