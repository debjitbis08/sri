"""Two-script reads on the shared output pilot checkpoints (docs/two-script-reads-v1.md). No training.

For every ordered pair of kinds (collections, taps, English, Chinese, Devanagari, Thai), at the block-3, delay and relearn checkpoints of gp,
common-raw, common-k-cand and common-k-shared, seeds 300-303:
  matching      same(item of kind A, count n) against a gallery of kind B, one item per count (Part D's gallery format, item index aligned): the
                top candidate's count is n. Chance 1/9.
  comparison    more(item of A, count a; item of B, count b) > 0 iff a > b, over all a != b, item index aligned (Part D's comparison read).
  continuation  (prev of A, count n-1; cur of B, count n) -> n+1, and (n+1, n) -> n-1, n in 2..8: the top participant of M through the arm's
                cont_logits. No abstention (the pilot's thresholds are not re-selected here).
Rows are split by whether any count involved is withheld in its script (part D's withheld counts: Chinese and Devanagari, two each).
Thai digits are never taught: an untaught-script control. English is taught in block 1 and relearned at the end.

Usage:
  .venv/bin/python -m experiments.two_script_reads            (CPU workers, TS_WORKERS, default 4; rerun to resume)
  .venv/bin/python -m experiments.two_script_reads summary
"""
import os
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import shared_output as SOX  # sets up sense_core and limited_memory as the pilot does
sys.argv = _argv
import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from src import senses as SN

SC, S0, P, D, X, LM, nc, ns = SOX.SC, SOX.S0, SOX.P, SOX.D, SOX.X, SOX.LM, SOX.nc, SOX.ns
NAME = 'two-script-reads-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'two-script-reads-v1.md'
ARMS = ('gp', 'common-raw', 'common-k-cand', 'common-k-shared')
SEEDS = (300, 301, 302, 303)
POINTS = ('block3', 'delay', 'relearn')
WORKERS = int(os.environ.get('TS_WORKERS', 4))
E = 30  # evaluation items per count and kind, as P.EVAL_PER
SCRIPTS = (*P.LANGS, P.TRANSFER)  # english, chinese, devanagari_digits, thai_digits
KINDS = ('collection', 'taps', *SCRIPTS)
SHORT = {'collection': 'coll', 'taps': 'taps', 'english': 'en', 'chinese': 'zh', 'devanagari_digits': 'dev', 'thai_digits': 'thai'}


def load(arm, seed, point):
    """The pilot's model for (arm, seed), with the checkpoint's weights."""
    W = SOX.W
    z = dict(np.load(SOX.RUNS / f'names-seed{seed}.npz'))
    import pickle
    mem = pickle.load(open(SOX.MEMORY / f'memory-seed{seed}.pkl', 'rb'))
    W.update(arm=arm, names=z, inventory=SOX.inventory(seed), unseen_images=mem['query_images']['unseen'],
             unseen_codes=np.concatenate([mem['query_codes']['unseen'][n].reshape(-1, SN.get(n).dim) for n in SC.MODELS], 1))
    del mem
    enc, model = DEV.build(lambda: SOX.make_pilot_model(seed, arm))
    path = SOX.RUNS / 'pilot' / arm / 'checkpoints' / f'sparse48-seed{seed}-{point}.pt'
    ck = torch.load(path, weights_only=False)
    enc['image'].load_state_dict(ck['image'])
    enc['sequence'].load_state_dict(ck['sequence'])
    model.load_state_dict(ck['sri'])
    for m in (enc['image'], enc['sequence'], model):
        m.eval()
    model._M, model.hard = None, False
    return enc, model, {'file': str(path.relative_to(X.ROOT)), 'sha256': nc.sha256(path)}


@torch.no_grad()
def reads(enc, model, withheld):
    img = lambda x: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), 'image')
    items = {'collection': {n: img(P.DOTS['succ'][n][:E]) for n in ns.NUMBERS},
             'taps': {n: D.codes_for(enc, model, D.TAPS[('eval', n)][:E], 'sequence') for n in ns.NUMBERS}}
    for lang in SCRIPTS:
        items[lang] = {n: img(X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]][:E]) for n in ns.NUMBERS}
    k = min(len(v[n]) for v in items.values() for n in ns.NUMBERS)  # smoke data may hold fewer
    items = {kind: {n: v[n][:k] for n in ns.NUMBERS} for kind, v in items.items()}
    wh = {kind: set(withheld.get(kind, ())) for kind in KINDS}
    M = model.participants()
    inv_n = np.array([n for n, _ in SOX.W['inventory']])
    out = {}

    def split(rows):
        """rows: list of (hits array, touches withheld) -> all / paired only / touching withheld."""
        r = {'all': float(np.mean(np.concatenate([h for h, _ in rows])))}
        for name, sel in (('paired', False), ('withheld', True)):
            h = [h for h, t in rows if t == sel]
            r[name] = float(np.mean(np.concatenate(h))) if h else None
        return r

    for a in KINDS:
        for b in KINDS:
            # matching: item of a, count n, against a gallery of b
            gal = torch.stack([items[b][m] for m in ns.NUMBERS], 1)  # (k, 9, d)
            rows = [(SOX.D.pick(model, items[a][n], gal) == n - 1, n in wh[a] or n in wh[b]) for n in ns.NUMBERS]
            match = split(rows)
            # comparison
            rows = []
            for x in ns.NUMBERS:
                for y in ns.NUMBERS:
                    if x != y:
                        rows.append(((model.relation_logits(items[a][x], items[b][y], 2) > 0).numpy() == (x > y), x in wh[a] or y in wh[b]))
            comp = split(rows)
            # continuation over M, no abstention
            cont = {}
            for direction, sign in (('up', 1), ('down', -1)):
                rows = []
                for n in range(2, 9):
                    prev, cur = items[a][n - sign], items[b][n]
                    lg = model.cont_logits(prev, cur, M[None].expand(len(prev), -1, -1)).numpy()
                    rows.append((inv_n[lg.argmax(1)] == n + sign, (n - sign) in wh[a] or n in wh[b]))
                cont[direction] = split(rows)
            out[f'{SHORT[a]}>{SHORT[b]}'] = {'matching': match, 'comparison': comp, 'continuation': cont}
    return out


def task(t):
    arm, seed = t
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    t0, c0 = time.perf_counter(), time.process_time()
    withheld = {k: tuple(int(x) for x in v) for k, v in D.withheld_counts(seed).items()}
    res, ck = {}, {}
    for point in POINTS:
        enc, model, ck[point] = load(arm, seed, point)
        res[point] = reads(enc, model, withheld)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
          'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    return f'{arm}/seed{seed}', {'reads': res, 'checkpoints': ck, 'withheld': withheld}, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    tasks = [(a, s) for a in ARMS for s in SEEDS]
    fn = DUR.durable(task, RUNS / 'done', '{0}-seed{1}')
    with __import__('multiprocessing').get_context('fork').Pool(WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'reads on the shared output pilot checkpoints; no training', 'arms': ARMS, 'seeds': SEEDS,
                                     'points': POINTS, 'kinds': KINDS, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', {
        'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
        'torch': torch.__version__, 'cpu': platform.processor(), 'workers': WORKERS, 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
        'wall_seconds': time.perf_counter() - t0, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res)),
        'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, 'rss_convention': 'ru_maxrss in KiB; children is the largest single child',
        'tasks': {k: t for k, _, t in res}})


def summary():
    import json
    res = json.loads((REPORT.with_suffix('.json')).read_text())['results']
    short = [SHORT[k] for k in KINDS]

    def grid(arm, point, f):
        lines = [f'| {arm}, {point} | ' + ' | '.join(short) + ' |', '|---|' + '---|' * len(short)]
        for a in short:
            cells = []
            for b in short:
                v = [f(res[f'{arm}/seed{s}']['reads'][point][f'{a}>{b}']) for s in SEEDS]
                v = [x for x in v if x is not None]
                cells.append(f'{np.mean(v):.2f}' if v else '—')
            lines.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        return '\n'.join(lines)

    reads_ = {'matching (row: query kind, column: gallery kind)': lambda r: r['matching']['all'],
              'comparison (row: first item, column: second)': lambda r: r['comparison']['all'],
              'continuation, mean of directions (row: prev, column: cur)': lambda r: np.mean([r['continuation'][d]['all'] for d in ('up', 'down')]),
              'comparison, rows touching withheld counts': lambda r: r['comparison']['withheld'],
              'matching, rows touching withheld counts': lambda r: r['matching']['withheld']}
    for name, f in reads_.items():
        print(f'## {name}\n')
        for arm in ARMS:
            for point in POINTS:
                print(grid(arm, point, f) + '\n')


if __name__ == '__main__':
    (summary if len(sys.argv) > 1 and sys.argv[1] == 'summary' else main)()
