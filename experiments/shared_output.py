"""Shared output v1 (docs/shared-output-v1-contract.md), stage 0: naming-only transfer on the existing step 8 checkpoints.

From the step 8 `joined-mlp-calibrated` delay checkpoints (sparse48, g and P unchanged), 45 new symbols (names, 5 per count) are taught only by matching:
200 updates of delay-style context episodes, each with 9 naming rows added (one per count, name -> the update's 9 candidate collections through g's
`same`). Names never enter next, more or sequence rows; every update is audited. Fresh Adam (the checkpoints hold no optimizer state). RigL is off.

Reads at updates 0, 50, 100 and 200, with fixed draws per seed:
  name matching           a name's evaluation rendering against 9 evaluation collections, one per count, through g(same)
  continuation            (prev, cur) -> P -> nearest of the 72-item inventory (8 training collections per count, raw frozen codes re-read through
                          the current core), ascending (n-1, n) -> n+1 and descending (n+1, n) -> n-1, n in 2..8, prev an evaluation collection.
                          cur is a name (direct), the name's recall over the inventory through g(same) (grounded: softmax-weighted state, or the
                          top item's state), or a familiar item: an evaluation collection, Devanagari form (trained counts) or taps.
  substitutability        whether a name's nearest inventory count mean is its own count, for names and for familiar Devanagari forms
Recall counts the top inventory item's count; AUROC ranks the target count's items against the others, and against the reverse and repeated counts.

Usage:
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot names    (frozen codes of the names, GPU, once per seed)
  SO_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot stage0       (CPU)
"""
import os
import platform
import resource
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import sense_core as SC  # sets limited_memory's argv itself
from experiments import sense_memory_core as SMC
sys.argv = _argv
import numpy as np
import torch

from experiments import device as DEV
from experiments import durable as DUR
from experiments import fan_in_pilot as FI
from experiments import pairing_symbols as PS
from src import senses as SN

S0, P, D, X, LM = SC.S0, SC.P, SC.D, SC.X, SC.LM
SP, nc, ns = LM.SP, X.nc, X.ns
NAME = 'shared-output-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
CONTRACT = X.ROOT / 'docs' / 'shared-output-v1-contract.md'
HOST = 'joined-mlp-calibrated'
SEEDS = tuple(int(s) for s in os.environ.get('SO_SEEDS', '300,301,302,303').split(','))
WORKERS = int(os.environ.get('SO_WORKERS', 4))
N_NAMES, TRAIN_R, EVAL_R = 45, 8, 5
STEPS, READ_AT = 200, (0, 50, 100, 200)
INV_PER, HIST_PER, CUR_PER = 8, 3, 5
DEV_LANG = 'devanagari_digits'
NAMES_KEY = ('names', 'train')


def names(seed):
    """45 pool symbols disjoint from sense_memory_core's sets A, B and unseen for this seed, with balanced counts."""
    pool, sets, unseen, rest = SMC.symbols(seed)
    cps = rest[:N_NAMES]
    assert not set(cps) & (set(sets['A']) | set(sets['B']) | set(unseen))
    counts = nc.stream('so_names', seed).permutation(np.repeat(np.arange(1, 10), N_NAMES // 9))
    return pool, cps, counts


def runtime(extra):
    return {'scope': 'nondeterministic timing', 'command': ' '.join(sys.argv), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
            'contract_sha256': nc.sha256(CONTRACT), 'runner_sha256': nc.sha256(__file__), 'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cpu': platform.processor(), 'threads_per_worker': FI.THREADS, 'dtype': 'float32', **DEV.info(),
            'parent_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'children_peak_rss_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            'rss_convention': 'ru_maxrss in KiB, per process; children is the largest single child', **extra}


def names_stage():
    t0, c0 = time.perf_counter(), time.process_time()
    DEV.install()
    sense = SN.Sense(SC.MODELS, DEV.NAME if DEV.ACTIVE else 'cpu', chunk=64)
    out = {}
    for seed in SEEDS:
        path = RUNS / f'names-seed{seed}.npz'
        if path.exists():
            continue
        s0 = time.perf_counter()
        pool, cps, counts = names(seed)
        tr = np.stack([PS.renderings(cp, pool[cp], seed, 'train', TRAIN_R) for cp in cps]).astype(np.float32)
        ev = np.stack([PS.renderings(cp, pool[cp], seed, 'eval', EVAL_R) for cp in cps]).astype(np.float32)
        r0 = time.perf_counter()
        f = sense(np.concatenate([tr.reshape(-1, 28, 28), ev.reshape(-1, 28, 28)]))
        RUNS.mkdir(parents=True, exist_ok=True)
        tmp = RUNS / f'names-seed{seed}.tmp.npz'
        np.savez(tmp, cps=np.array(cps), counts=counts, train=tr, eval=ev, **{n: f[n] for n in SC.MODELS})
        os.replace(tmp, path)
        out[seed] = {'render_seconds': r0 - s0, 'encode_seconds': time.perf_counter() - r0, 'images': int(len(tr.reshape(-1, 28, 28)) + len(ev.reshape(-1, 28, 28)))}
        print(seed, out[seed], flush=True)
    S0.save_json(f'{REPORT}-names-runtime.json', runtime({'phase': 'frozen codes of the names', 'wall_seconds': time.perf_counter() - t0,
                                                         'cpu_seconds': time.process_time() - c0, 'seeds': out, 'senses': sense.describe()}))


# ---------------------------------------------------------------- stage 0

def load_host(seed, z):
    enc, model = SC.make_model(seed, 'sparse48', HOST)
    path = SC.RUNS / HOST / 'checkpoints' / f'sparse48-seed{seed}-delay.pt'
    ck = torch.load(path, weights_only=False)
    enc['image'].load_state_dict(ck['image'])
    enc['sequence'].load_state_dict(ck['sequence'])
    model.load_state_dict(ck['sri'])
    imgs = np.concatenate([z['train'].reshape(-1, 28, 28), z['eval'].reshape(-1, 28, 28)])
    enc['image'].add_codes([SC.image_key(np.ascontiguousarray(r, dtype=np.float32)) for r in imgs], np.concatenate([z[n] for n in SC.MODELS], 1))
    return enc, model, {'file': str(path.relative_to(X.ROOT)), 'sha256': nc.sha256(path)}


def material(crng, nrng, counts, audit):
    items, pairs, match, seqs = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
    first = len(items)
    for n in ns.NUMBERS:
        s = int(nrng.choice(np.flatnonzero(counts == n)))
        idx = s * TRAIN_R + int(nrng.integers(TRAIN_R))
        fi = len(items)
        items.append(('form', 'names', n, idx))
        items.append(('coll', None, n, int(nrng.integers(len(P.DOTS['train'])))))
        match.append((fi, fi + 1, n))
    named = {i for i in range(first, len(items)) if items[i][1] == 'names'}
    # the audit: names only as the left argument of matching. Comparison draws from collections and taps, next negatives from the kinds of pairs
    # (taps and collections here), sequence rows from the episodes' steps; none can reach a name.
    assert not named & {i for p in pairs for i in p}
    assert not named & {i for s in seqs for _, st in s for i in st.values()}
    assert not named & {m[1] for m in match}
    assert {it[0] for k, it in enumerate(items) if k not in named} <= {'taps', 'coll'}
    assert {items[p[0]][0] for p in pairs} <= {'taps', 'coll'}
    audit['naming_rows'] += len(named)
    audit['name_in_pairs'] += 0
    audit['pairs'] += len(pairs)
    audit['matching_rows'] += len(match)
    return items, pairs, match, seqs


def auroc(pos, neg):
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None]
    return float(((pos > neg).mean() + .5 * (pos == neg).mean()))


@torch.no_grad()
def read(enc, model, seed, z, inv):
    for m in (enc['image'], enc['sequence'], model):
        m.eval()
    img = lambda x: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), 'image')
    rng = nc.stream('so_stage0_draws', seed)  # the same draws at every read point
    counts = z['counts']
    inv_c = img(np.stack([P.DOTS['train'][e, n - 1] for n, e in inv]))
    inv_n = np.array([n for n, _ in inv])
    means = torch.stack([inv_c[inv_n == n].mean(0) for n in ns.NUMBERS])
    colls = {n: img(P.DOTS['succ'][n]) for n in ns.NUMBERS}
    taps = {n: D.codes_for(enc, model, D.TAPS[('eval', n)], 'sequence') for n in ns.NUMBERS}
    withheld = set(D.withheld_counts(seed)[DEV_LANG])
    forms = {n: img(X.DATA[(DEV_LANG, 'eval')][0][P.EVAL[(DEV_LANG, n, 'idx')]]) for n in ns.NUMBERS}
    nm = img(z['eval'].reshape(-1, 28, 28)).reshape(N_NAMES, EVAL_R, -1)
    out = {}

    # name matching against 9 evaluation collections; recall over the inventory through g(same)
    q = nm.reshape(-1, nm.shape[-1])
    qn = np.repeat(counts, EVAL_R)
    js = rng.integers(len(colls[1]), size=(len(q), 9))
    cands = torch.stack([torch.stack([colls[k + 1][js[i, k]] for k in range(9)]) for i in range(len(q))])
    out['name_matching'] = float((D.pick(model, q, cands) + 1 == qn).mean())
    lg = P.same_logits(model, q[:, None].expand(-1, len(inv_c), -1).reshape(-1, q.shape[-1]), inv_c[None].expand(len(q), -1, -1).reshape(-1, q.shape[-1])).reshape(len(q), -1)
    out['name_recall_inventory'] = float((inv_n[lg.argmax(1).numpy()] == qn).mean())
    grounded = {'soft': (lg.softmax(1) @ inv_c).reshape(N_NAMES, EVAL_R, -1), 'hard': inv_c[lg.argmax(1)].reshape(N_NAMES, EVAL_R, -1)}

    # substitutability: nearest inventory count mean
    def nearest(c, n):
        d = torch.cdist(c, means)
        own = d[:, n - 1]
        other = torch.cat([d[:, :n - 1], d[:, n:]], 1).min(1).values
        return (d.argmin(1).numpy() + 1 == n), (own / other).numpy()
    sub = {'names': [nearest(nm[i], int(counts[i])) for i in range(N_NAMES)],
           'devanagari_trained': [nearest(forms[n][:CUR_PER], n) for n in ns.NUMBERS if n not in withheld]}
    out['substitutability'] = {k: {'nearest_own_count': float(np.concatenate([a for a, _ in v]).mean()), 'own_over_other_distance_median': float(np.median(np.concatenate([b for _, b in v])))}
                               for k, v in sub.items()}

    # continuation through P over the inventory
    def cur_items(kind, n, k):
        if kind.startswith('name'):
            sel = np.flatnonzero(counts == n)
            src = nm if kind == 'name' else grounded[kind.split('-')[1]]
            return src[sel].reshape(-1, src.shape[-1])
        if kind == 'collection':
            return colls[n][rng.permutation(len(colls[n]))[:CUR_PER]]
        if kind == 'devanagari':
            return None if n in withheld else forms[n][rng.permutation(len(forms[n]))[:CUR_PER]]
        return taps[n][rng.permutation(len(taps[n]))[:CUR_PER]]
    cont = {}
    for kind in ('name', 'name-soft', 'name-hard', 'collection', 'devanagari', 'taps'):
        for direction, sign in (('up', 1), ('down', -1)):
            hit, rev, rep, au, au_hard = [], [], [], [], []
            for n in range(2, 9):
                cur = cur_items(kind, n, CUR_PER)
                if cur is None:
                    continue
                prev = colls[n - sign][rng.permutation(len(colls[n - sign]))[:HIST_PER]]
                pv, cu = prev.repeat_interleave(len(cur), 0), cur.repeat(len(prev), 1)
                lgc = -(model.predict(pv, cu)[:, None] - inv_c[None]).pow(2).sum(-1).numpy()
                top = inv_n[lgc.argmax(1)]
                tgt = n + sign
                hit.append(top == tgt); rev.append(top == n - sign); rep.append(top == n)
                for row in lgc:
                    au.append(auroc(row[inv_n == tgt], row[inv_n != tgt]))
                    au_hard.append(auroc(row[inv_n == tgt], row[(inv_n == n - sign) | (inv_n == n)]))
            cont[f'{kind}/{direction}'] = {'recall': float(np.concatenate(hit).mean()), 'top_is_reverse': float(np.concatenate(rev).mean()),
                                           'top_is_repeat': float(np.concatenate(rep).mean()), 'auroc': float(np.mean(au)), 'auroc_reverse_repeat': float(np.mean(au_hard)),
                                           'queries': int(len(np.concatenate(hit)))}
    out['continuation'] = cont

    # familiar matching, as a check that naming rows leave the trained reads in place
    def match_read(qs, n_of):
        js = rng.integers(len(colls[1]), size=(len(qs), 9))
        cands = torch.stack([torch.stack([colls[k + 1][js[i, k]] for k in range(9)]) for i in range(len(qs))])
        return float((D.pick(model, qs, cands) + 1 == n_of).mean())
    tq = torch.cat([taps[n][:CUR_PER] for n in ns.NUMBERS])
    tn = np.repeat(np.arange(1, 10), CUR_PER)
    fq = torch.cat([forms[n][:CUR_PER] for n in ns.NUMBERS])
    out['matching'] = {'taps_to_collections': match_read(tq, tn), 'devanagari_to_collections_trained': match_read(fq[[n not in withheld for n in tn]], tn[[n not in withheld for n in tn]]),
                       'devanagari_to_collections_withheld': match_read(fq[[n in withheld for n in tn]], tn[[n in withheld for n in tn]])}
    for m in (enc['image'], enc['sequence'], model):
        m.train()
    return out


def stage0_task(task):
    seed, = task
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(FI.THREADS)
    z = dict(np.load(RUNS / f'names-seed{seed}.npz'))
    X.DATA[NAMES_KEY] = (z['train'].reshape(-1, 28, 28), np.repeat(z['counts'], TRAIN_R))
    enc, model, ck = load_host(seed, z)
    irng = nc.stream('so_inventory', seed)
    inv = [(n, int(e)) for n in ns.NUMBERS for e in irng.choice(len(P.DOTS['train']), INV_PER, replace=False)]
    opt = torch.optim.Adam([*enc['image'].parameters(), *enc['sequence'].parameters(), *model.parameters()], P.LR)
    LM.CURRENT.update(arm='sparse48', seed=seed, step=10 ** 9, end=1, r=None)  # step past the RigL window: no rewiring
    rng, crng, nrng = nc.stream('so_stage0_train', seed), nc.stream('so_stage0_context', seed), nc.stream('so_stage0_names', seed)
    audit = {'naming_rows': 0, 'name_in_pairs': 0, 'pairs': 0, 'matching_rows': 0}
    reads, cpu = {}, {'train': 0., 'read': 0.}
    for t in range(STEPS + 1):
        if t in READ_AT:
            e0 = time.process_time()
            reads[str(t)] = read(enc, model, seed, z, inv)
            cpu['read'] += time.process_time() - e0
        if t == STEPS:
            break
        e0 = time.process_time()
        LM.train_update(enc, model, opt, material(crng, nrng, z['counts'], audit), rng, True)
        cpu['train'] += time.process_time() - e0
    return f'seed{seed}', {'checkpoint': ck, 'names': [int(c) for c in z['cps']], 'counts': [int(c) for c in z['counts']], 'inventory': inv,
                           'audit': audit, 'reads': reads}, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                                             **{f'{k}_cpu_seconds': v for k, v in cpu.items()}, 'updates': STEPS,
                                                             'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def stage0():
    t0 = time.perf_counter()
    assert not DEV.ACTIVE, 'stage 0 runs on the CPU'
    S0.prepare()
    LM.install()
    fn = DUR.durable(stage0_task, RUNS / 'done-stage0', 'seed{0}')
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, [(s,) for s in SEEDS], chunksize=1)
    S0.save_json(f'{REPORT}-stage0.json', {'status': 'stage 0 of the draft contract: naming-only teaching on the step 8 delay checkpoints, g and P unchanged',
                                           'host': HOST, 'seeds': SEEDS, 'steps': STEPS, 'read_at': READ_AT, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-stage0-runtime.json', runtime({'phase': 'stage 0', 'wall_seconds': time.perf_counter() - t0, 'workers': WORKERS,
                                                           'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    {'names': names_stage, 'stage0': stage0}[sys.argv[2]]()
