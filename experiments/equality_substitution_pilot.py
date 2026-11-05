"""Equality as substitutability v1 (docs/equality-substitution-v1.md). Trained pilot on frozen sense codes, Part C network.

Usage: .venv/bin/python -m experiments.equality_substitution_pilot            (EQ_WORKERS, default 4; rerun to resume)
       .venv/bin/python -m experiments.equality_substitution_pilot summary
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
from src import equality_net as EQ
from src import relation_codes as RC

T, D, X, nc, ns = RP.T, RP.D, RP.X, RP.nc, RP.ns
SMOKE = bool(os.environ.get('EQ_SMOKE'))
NAME = 'equality-substitution-v1' + ('-smoke' if SMOKE else '')
CONTRACT = X.ROOT / 'docs' / 'equality-substitution-v1.md'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
SEEDS = RP.SEEDS
WORKERS = int(os.environ.get('EQ_WORKERS', 4))
STEPS = 20 if SMOKE else RP.STEPS
LR, COLL_PER, FORM_PER = RP.LR, RP.COLL_PER, RP.FORM_PER
BUF_STORE, BUF_PER = 4, 2  # stored codes per count; replayed per count per step
BLOCKS, SHORT, N = RP.BLOCKS, RP.SHORT, RP.N
OP_PAIRS = (('collection', 'collection'), ('chinese', 'collection'), ('collection', 'chinese'), ('chinese', 'chinese'),
            ('chinese', 'devanagari_digits'), ('devanagari_digits', 'collection'), ('devanagari_digits', 'devanagari_digits'),
            ('english', 'collection'), ('english', 'english'), ('thai_digits', 'collection'))
SUBST = (('chinese', 'collection'), ('devanagari_digits', 'collection'), ('english', 'collection'), ('thai_digits', 'collection'),
         ('chinese', 'devanagari_digits'))


@torch.no_grad()
def op_eval(model, withheld):
    model.eval()
    ev = RP.DATA['eval']
    codes = {}
    for g, kind in enumerate(ev):
        x = torch.from_numpy(np.concatenate([ev[kind][n] for n in ns.NUMBERS]))
        codes[kind] = model.codes(x, torch.full((len(x),), g))
    k = len(ev['collection'][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    n = len(lab)
    taught = lambda kind: torch.tensor([int(v) not in set(withheld.get(kind, ())) for v in lab.tolist()])
    out = {'balanced_accuracy': {}, 'substitution': {}}
    for a, b in OP_PAIRS:
        ya, yb = lab.repeat_interleave(n), lab.repeat(n)
        t = EQ.op_targets(ya, yb).bool()
        p = model.op_logits(codes[a].repeat_interleave(n, 0), codes[b].repeat(n, 1)) > 0
        out['balanced_accuracy'][f'{SHORT[a]}>{SHORT[b]}'] = {
            op: float(.5 * (p[t[:, i], i].float().mean() + (~p[~t[:, i], i]).float().mean())) for i, op in enumerate(EQ.OPS)}
    M = codes['collection']
    within = torch.arange(n) - torch.arange(n) % k + (torch.arange(n) + 1) % k  # another item of the same count
    shifted = (torch.arange(n) + 3 * k) % n  # an item of count n + 3 (mod 9)

    def agree(xs, cs):
        m = len(xs)
        X1, C1, M1 = xs.repeat_interleave(n, 0), cs.repeat_interleave(n, 0), M.repeat(m, 1)
        r = {}
        for i, op in enumerate(EQ.OPS):
            v = []
            for px, pc in ((model.op_logits(X1, M1)[:, i] > 0, model.op_logits(C1, M1)[:, i] > 0),
                           (model.op_logits(M1, X1)[:, i] > 0, model.op_logits(M1, C1)[:, i] > 0)):
                v.append(.5 * ((px[pc]).float().mean() + (~px[~pc]).float().mean()))  # balanced over the reference's output
            r[op] = float(torch.stack(v).mean())
        dx = model.rel_logits(X1, M1).argmax(1) == model.rel_logits(C1, M1).argmax(1)
        dy = model.rel_logits(M1, X1).argmax(1) == model.rel_logits(M1, C1).argmax(1)
        r['difference'] = float(torch.cat([dx, dy]).float().mean())
        return r

    for kind, ref in SUBST + (('collection', 'collection'),):
        x = codes[kind]
        c = codes[ref][within] if kind == ref else codes[ref]  # item i of x and of c have the same count
        mask = taught(kind) & taught(ref)
        r = {}
        for split, keep in (('all', torch.ones(n, dtype=torch.bool)), ('taught', mask)):
            r.update({f'{split}_{key}': v for key, v in agree(x[keep], c[keep]).items()})
        r.update({f'null_{key}': v for key, v in agree(x, codes[ref][shifted]).items()})  # reference of a different count
        out['substitution'][f'{SHORT[kind]}~{SHORT[ref]}'] = r
    model.train()
    return out


def run(arm, seed):
    RP.prepare()
    DATA = RP.DATA
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = nc.stream('equality_substitution', seed, EQ.ARMS.index(arm))
    brng = nc.stream('equality_buffer', seed)  # the same stored codes for every arm of a seed
    withheld = {k: tuple(int(x) for x in v) for k, v in D.withheld_counts(seed).items()}
    width = DATA['coll_train'].shape[-1]
    model = EQ.EqNet(arm, width, g)
    n_ep = len(DATA['coll_train']) * 5 // 6
    buf = {}
    for lang in ('english', 'chinese'):
        fx, fy = DATA['form_train'][lang]
        idx = np.concatenate([brng.choice(np.flatnonzero(fy == n), BUF_STORE, replace=False)
                              for n in ns.NUMBERS if n not in set(withheld.get(lang, ()))])
        buf[lang] = (fx[idx], fy[idx])
    en_mode, zh_mode = EQ.en_mode(arm), EQ.zh_mode(arm)
    snaps, steps = {}, 0
    for bi, (block, lang) in enumerate(BLOCKS):
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        fx, fy = DATA['form_train'][lang]
        taught = [n for n in ns.NUMBERS if n not in set(withheld.get(lang, ()))]
        by_n = {n: np.flatnonzero(fy == n) for n in taught}
        replay = [(name, mode) for name, mode, first in (('english', en_mode, 1), ('chinese', zh_mode, 2)) if mode and bi >= first]
        for _ in range(STEPS):
            ep = rng.integers(n_ep, size=(N, COLL_PER))
            xc = DATA['coll_train'][ep, np.arange(N)[:, None]].reshape(-1, width)
            yc = np.repeat(np.arange(1, N + 1), COLL_PER)
            fi = np.concatenate([rng.choice(by_n[n], FORM_PER, replace=False) for n in taught])
            parts = [(xc, yc), (fx[fi], fy[fi])]
            for name, _ in replay:
                bx, by = buf[name]
                bsel = np.concatenate([rng.choice(np.flatnonzero(by == n), BUF_PER, replace=False) for n in np.unique(by)])
                parts.append((bx[bsel], by[bsel]))
            x = torch.from_numpy(np.concatenate([p[0] for p in parts]))
            grp = torch.cat([torch.full((len(p[0]),), i, dtype=torch.long) for i, p in enumerate(parts)])
            c = model.codes(x, grp)
            cs = [c[grp == i] for i in range(len(parts))]
            ys = [torch.from_numpy(p[1].astype(np.int64)) for p in parts]
            L = RC.loss(model, cs[0], ys[0], cs[1], ys[1])
            for k, (name, mode) in enumerate(replay, start=2):
                if mode == 'label':
                    L = L + EQ.label_loss(model, cs[0], ys[0], cs[k], ys[k])
                else:
                    i, j = EQ.same_count_pairs(ys[1], ys[k])
                    L = L + EQ.substitution_loss(model, cs[0], cs[1][i], cs[k][j], with_ops=block == 'block3')
            if block == 'block3':
                L = L + EQ.op_loss(model, cs[0], ys[0])
            opt.zero_grad()
            L.backward()
            opt.step()
            steps += 1
        snaps[block] = RP.evaluate(model, withheld)
        snaps[block]['final_loss'] = float(L.detach())
        if block == 'block3':
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
    RP.S0.save_json(f'{REPORT}.json', {'status': 'equality as substitutability v1' + (' (smoke)' if SMOKE else ''), 'arms': EQ.ARMS,
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
    f = lambda v: f'{np.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})' if v else '-'
    for block, _ in BLOCKS:
        print(f'\n## {block} identity (best of nearest/relational): zh | dev | en | thai')
        for arm in EQ.ARMS:
            s = [res[f'{arm}/seed{x}']['snapshots'][block] for x in SEEDS]
            best = lambda kd: f([max(v['identity'][kd]['nearest'], v['identity'][kd]['relational']) for v in s])
            print(f"| {arm} | {best('zh')} | {best('dev')} | {best('en')} | {best('thai')[:4]} | zh>dev exact {f([v['relations']['zh>dev']['exact'] for v in s])} "
                  f"via id {f([v['relations']['zh>dev']['via_id_exact'] for v in s])[:4]} |")
    print('\n## block3 operators: substitution (balanced agreement, all counts) more/sum10/difference; balanced accuracy more/sum10')
    for arm in EQ.ARMS:
        o = [res[f'{arm}/seed{x}']['snapshots']['block3']['operators'] for x in SEEDS]
        cells = []
        for key in ('coll~coll', 'zh~coll', 'dev~coll', 'en~coll', 'thai~coll', 'zh~dev'):
            cells.append('/'.join(f([v['substitution'][key][f'all_{op}'] for v in o])[:4] for op in ('more', 'sum10', 'difference')))
        acc = ['/'.join(f([v['balanced_accuracy'][key][op] for v in o])[:4] for op in EQ.OPS) for key in ('coll>coll', 'zh>coll', 'zh>dev', 'en>coll')]
        print(f'| {arm} | ' + ' | '.join(cells) + ' | ' + ' | '.join(acc) + ' |')
        print('|   null | ' + ' | '.join('/'.join(f([v['substitution'][key][f'null_{op}'] for v in o])[:4] for op in ('more', 'sum10', 'difference'))
                                    for key in ('coll~coll', 'zh~coll', 'dev~coll', 'en~coll', 'thai~coll', 'zh~dev')) + ' |')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
