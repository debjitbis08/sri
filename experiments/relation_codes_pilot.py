"""Relation codes check v1, part C: one code space or two (docs/relation-codes-check-v1.md). Trained pilot on frozen sense codes.

Usage: .venv/bin/python -m experiments.relation_codes_pilot            (RC_WORKERS, default 4; rerun to resume)
       .venv/bin/python -m experiments.relation_codes_pilot summary
       RC_SMOKE=1 ... for a short check (20 steps per block)
"""
import os
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import two_script_reads as T  # sets up the pilot's data (P, X, D)
sys.argv = _argv
import numpy as np
import torch

from experiments import durable as DUR
from experiments import sense_core as SC
from src import relation_codes as RC

S0, P, D, X, nc, ns = T.S0, T.P, T.D, T.X, T.nc, T.ns
SMOKE = bool(os.environ.get('RC_SMOKE'))
NAME = 'relation-codes-pilot-v1' + ('-smoke' if SMOKE else '')
CONTRACT = X.ROOT / 'docs' / 'relation-codes-check-v1.md'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
SEEDS = (300, 301, 302, 303)
WORKERS = int(os.environ.get('RC_WORKERS', 4))
STEPS = 20 if SMOKE else 1500
LR = 1e-3
COLL_PER, FORM_PER = 8, 8
BLOCKS = (('block1', 'english'), ('block2', 'chinese'), ('block3', 'devanagari_digits'))
SCRIPTS = ('english', 'chinese', 'devanagari_digits', 'thai_digits')
SHORT = T.SHORT
N = 9
DATA = {}


def raw_codes(images):
    tab = SC.table(SC.MODELS)
    rows = np.array([tab['index'][SC.image_key(x)] for x in images])
    return np.concatenate([tab[n][rows] for n in SC.MODELS], 1).astype(np.float32)


def prepare():
    """Frozen codes standardized with the pilot's calibration statistics (English and dot training images)."""
    if DATA:
        return
    S0.prepare()
    dots_train = P.DOTS['train']  # (episodes, 9, 28, 28): count n at index n - 1
    n_ep = len(dots_train)
    coll_raw = raw_codes(dots_train.reshape(-1, 28, 28)).reshape(n_ep, N, -1)
    en_raw = raw_codes(X.DATA[('english', 'train')][0])
    calib = np.concatenate([en_raw, coll_raw.reshape(-1, coll_raw.shape[-1])])
    mu, sd = calib.mean(0), calib.std(0) + 1e-6
    z = lambda a: ((a - mu) / sd).astype(np.float32)
    DATA['coll_train'] = z(coll_raw)
    DATA['form_train'] = {}
    for lang in SCRIPTS[:3]:
        raw = en_raw if lang == 'english' else raw_codes(X.DATA[(lang, 'train')][0])
        DATA['form_train'][lang] = (z(raw), X.DATA[(lang, 'train')][1].astype(np.int64))
    DATA['eval'] = {'collection': {n: z(raw_codes(P.DOTS['succ'][n][:T.E])) for n in ns.NUMBERS}}
    for lang in SCRIPTS:
        DATA['eval'][lang] = {n: z(raw_codes(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:T.E])) for n in ns.NUMBERS}


@torch.no_grad()
def evaluate(model, withheld):
    model.eval()
    ev = DATA['eval']
    codes = {}
    for g, kind in enumerate(ev):
        x = torch.from_numpy(np.concatenate([ev[kind][n] for n in ns.NUMBERS]))
        codes[kind] = model.codes(x, torch.full((len(x),), g))
    k = len(ev['collection'][1])
    lab = torch.arange(1, N + 1).repeat_interleave(k)
    mu = torch.stack([codes['collection'][lab == n].mean(0) for n in ns.NUMBERS])
    out = {}
    for kind, c in codes.items():
        near = ((c[:, None] - mu[None]) ** 2).sum(2).argmin(1) + 1
        rel = RC.relational_identity(model, c, mu)
        wh = torch.tensor([int(n) in set(withheld.get(kind, ())) for n in lab.tolist()])
        r = {}
        for name, pred in (('nearest', near), ('relational', rel)):
            ok = (pred == lab)
            r[name] = float(ok.float().mean())
            r[name + '_within1'] = float(((pred - lab).abs() <= 1).float().mean())
            r[name + '_taught'] = float(ok[~wh].float().mean())
            r[name + '_withheld'] = float(ok[wh].float().mean()) if wh.any() else None
        out[SHORT[kind]] = r
    # script relations by the arm's relation read, item index aligned over all count pairs
    pairs = {}
    for a, b in (('chinese', 'chinese'), ('chinese', 'devanagari_digits'), ('english', 'english'), ('devanagari_digits', 'devanagari_digits'),
                 ('thai_digits', 'thai_digits'), ('collection', 'collection')):
        A, B, dd = [], [], []
        for x in ns.NUMBERS:
            for y in ns.NUMBERS:
                A.append(codes[a][lab == x]); B.append(codes[b][lab == y]); dd.append(torch.full((k,), y - x))
        A, B, dd = torch.cat(A), torch.cat(B), torch.cat(dd)
        dh = model.rel_logits(A, B).argmax(1) - (N - 1)
        nearest = lambda c: ((c[:, None] - mu[None]) ** 2).sum(2).argmin(1) + 1
        dv = nearest(B) - nearest(A)  # via identity: d from the two nearest-mean identities
        nz = dd != 0
        pairs[f'{SHORT[a]}>{SHORT[b]}'] = {'exact': float((dh == dd).float().mean()), 'within1': float(((dh - dd).abs() <= 1).float().mean()),
                                           'sign': float((torch.sign(dh[nz]) == torch.sign(dd[nz])).float().mean()),
                                           'via_id_exact': float((dv == dd).float().mean()), 'via_id_within1': float(((dv - dd).abs() <= 1).float().mean()),
                                           'via_id_sign': float((torch.sign(dv[nz]) == torch.sign(dd[nz])).float().mean())}
    model.train()
    return {'identity': out, 'relations': pairs}


def run(arm, seed):
    prepare()
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rng = nc.stream('relation_codes_pilot', seed, RC.ARMS.index(arm))
    withheld = {k: tuple(int(x) for x in v) for k, v in D.withheld_counts(seed).items()}
    width = DATA['coll_train'].shape[-1]
    model = RC.RelationNet(arm, width, g)
    params = lambda: [p for p in model.parameters() if p.requires_grad]
    n_ep = len(DATA['coll_train']) * 5 // 6  # last sixth of episodes unused (held back)
    snaps, steps = {}, 0
    t0 = time.perf_counter()
    for block, lang in BLOCKS:
        opt = torch.optim.Adam(params(), lr=LR)
        fx, fy = DATA['form_train'][lang]
        taught = [n for n in ns.NUMBERS if n not in set(withheld.get(lang, ()))]
        by_n = {n: np.flatnonzero(fy == n) for n in taught}
        for _ in range(STEPS):
            ep = rng.integers(n_ep, size=(N, COLL_PER))
            xc = DATA['coll_train'][ep, np.arange(N)[:, None]].reshape(-1, width)
            yc = np.repeat(np.arange(1, N + 1), COLL_PER)
            fi = np.concatenate([rng.choice(by_n[n], FORM_PER, replace=False) for n in taught])
            xf, yf = fx[fi], fy[fi]
            x = torch.from_numpy(np.concatenate([xc, xf]))
            grp = torch.cat([torch.zeros(len(xc), dtype=torch.long), torch.ones(len(xf), dtype=torch.long)])
            c = model.codes(x, grp)
            L = RC.loss(model, c[:len(xc)], torch.from_numpy(yc), c[len(xc):], torch.from_numpy(yf))
            opt.zero_grad()
            L.backward()
            opt.step()
            steps += 1
        snaps[block] = evaluate(model, withheld)
        snaps[block]['final_loss'] = float(L.detach())
        if block == 'block1':
            model.freeze_structure()
    rt = {'wall_seconds': time.perf_counter() - t0, 'optimizer_steps': steps}
    return snaps, withheld, rt


def task(t):
    arm, seed = t
    torch.set_num_threads(1)
    t0, c0 = time.perf_counter(), time.process_time()
    snaps, withheld, rt = run(arm, seed)
    rt.update(cpu_seconds=time.process_time() - c0, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return f'{arm}/seed{seed}', {'snapshots': snaps, 'withheld': withheld}, rt


def main():
    t0 = time.perf_counter()
    prepare()
    tp = time.perf_counter() - t0
    tasks = [(a, s) for a in RC.ARMS for s in SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'relation codes pilot, part C' + (' (smoke)' if SMOKE else ''), 'arms': RC.ARMS, 'seeds': SEEDS,
                                     'steps_per_block': STEPS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': 1, 'dtype': 'float32', 'device': 'cpu',
        'prepare_seconds': tp, 'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'tasks': {k: t for k, _, t in res}})


def summary():
    import json
    res = json.loads(REPORT.with_suffix('.json').read_text())['results']
    f = lambda v: f'{np.mean(v):.2f} ({min(v):.2f}-{max(v):.2f})' if v else '-'
    for block, _ in BLOCKS:
        print(f'\n## {block}\n| arm | zh nearest | zh relational | zh withheld (near/rel) | dev near/rel | en near/rel | thai near | zh>zh exact/sign | zh>dev exact/sign |')
        for arm in RC.ARMS:
            s = [res[f'{arm}/seed{x}']['snapshots'][block] for x in SEEDS]
            g = lambda kd, key: f([v['identity'][kd][key] for v in s if v['identity'][kd][key] is not None])
            h = lambda pk, key: f([v['relations'][pk][key] for v in s])
            print(f"| {arm} | {g('zh', 'nearest')} | {g('zh', 'relational')} | {g('zh', 'nearest_withheld')[:4]}/{g('zh', 'relational_withheld')[:4]} | "
                  f"{g('dev', 'nearest')[:4]}/{g('dev', 'relational')[:4]} | {g('en', 'nearest')[:4]}/{g('en', 'relational')[:4]} | {g('thai', 'nearest')[:4]} | "
                  f"{h('zh>zh', 'exact')[:4]}/{h('zh>zh', 'sign')[:4]} | {h('zh>dev', 'exact')[:4]}/{h('zh>dev', 'sign')[:4]} |")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary()
    else:
        main()
