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

Pilot: the four readout arms of src.shared_output (gp, gp-grounded, common, common-raw), trained from scratch on the step 8 curriculum
(limited_memory run_main, frozen sight), with names taught only by matching in block 3, acceptance losses on matching and continuation, and the
participants M (72 raw frozen collection codes). Reads (N1-N3) are added to the block-3, delay and relearn snapshots. `reference` runs the unchanged
step 8 path at seed 300 and checks it against reports/sense-core-v1-calibrated.json.gz.

Usage:
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot names    (frozen codes of the names, GPU, once per seed)
  SO_WORKERS=4 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot stage0       (CPU)
  NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot reference (GPU, seed 300)
  NFI_DEVICE=cuda SO_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.shared_output pilot pilot
  (rerun the same command to resume: finished runs are kept by experiments/durable.py)
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
from src import shared_output as SO

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


# ---------------------------------------------------------------- pilot

PILOT_ARMS = tuple(os.environ.get('SO_ARMS', ','.join(SO.ARMS)).split(','))
READ_POINTS = ('block3', 'delay', 'relearn')
MEMORY = X.ROOT / 'runs' / 'sense-memory-core-v1'
STEP8 = X.ROOT / 'reports' / 'sense-core-v1-calibrated.json.gz'
TAG = os.environ.get('SO_TAG', '')  # e.g. '-smoke': separate checkpoints, records and reports
W = {}  # per worker: arm, names, unseen, inventory, audit


def inventory(seed):
    irng = nc.stream('so_inventory', seed)
    return [(n, int(e)) for n in ns.NUMBERS for e in irng.choice(len(P.DOTS['train']), INV_PER, replace=False)]


def make_pilot_model(seed, arm):
    enc, model = SC.make_model(seed, 'sparse48', HOST)
    fi = enc['image']
    z, mem = W['names'], W['unseen_images']
    imgs = np.concatenate([z['train'].reshape(-1, 28, 28), z['eval'].reshape(-1, 28, 28), mem.reshape(-1, 28, 28)])
    raw = np.concatenate([np.concatenate([z[n] for n in SC.MODELS], 1), W['unseen_codes']])
    fi.add_codes([SC.image_key(np.ascontiguousarray(r, dtype=np.float32)) for r in imgs], raw)
    if arm == 'gp-reference':
        return enc, model
    rows = [fi.index[SC.image_key(np.ascontiguousarray(P.DOTS['train'][e, n - 1], dtype=np.float32))] for n, e in W['inventory']]
    SO.convert(model, arm, enc, fi.raw[rows], int(nc.stream('so_readout_init', seed, arm).integers(2 ** 62)))
    return enc, model


ORIG_IMAGES, ORIG_BATCH, ORIG_SNAPSHOT = P.images, SP.batch_material, LM.snapshot


def images(items):
    """P.images, plus names (kind 'name', index into the seed's training renderings)."""
    named = [k for k, it in enumerate(items) if it[0] == 'name']
    if not named:
        return ORIG_IMAGES(items)
    x = ORIG_IMAGES([it if it[0] != 'name' else ('coll', None, 1, 0) for it in items])
    tr = W['names']['train'].reshape(-1, 28, 28)
    for k in named:
        x[k] = tr[items[k][3]]
    return x


def batch_material(lang_eps, ctx_eps, context):
    """SP.batch_material, plus 9 naming rows (one per count) in every block-3 update. Names have their own kind, so no loss that draws by kind reaches them."""
    items, pairs, match, seqs = ORIG_BATCH(lang_eps, ctx_eps, context)
    step = LM.CURRENT['step']  # updates done before this one
    if 2 * D.STEPS <= step < 3 * D.STEPS:
        nrng, counts = W['name_rng'], W['names']['counts']
        for n in ns.NUMBERS:
            s = int(nrng.choice(np.flatnonzero(counts == n)))
            fi = len(items)
            items.append(('name', None, n, s * TRAIN_R + int(nrng.integers(TRAIN_R))))
            items.append(('coll', None, n, int(nrng.integers(len(P.DOTS['train'])))))
            match.append((fi, fi + 1, n))
    return items, pairs, match, seqs


def train_update(enc, model, opt, material, rng, compare):
    """TP.train_update with the same losses and draws, plus: acceptance losses on matching and continuation, continuation through the arm's
    cont_logits, participants re-read with gradient for the grounded arms, and the naming audit."""
    TP, F = LM.TP, torch.nn.functional
    items, pairs, match, seqs = material
    named = {k for k, it in enumerate(items) if it[0] == 'name'}
    solvers = {'image': P.Coupled(model)}
    if any(it[0] == 'taps' for it in items):
        solvers['sequence'] = D.Sequence(model)
    c = D.encode_items(enc, model, items, solvers)
    model._M = model.encode_participants(solvers['image']) if model.arm in SO.GROUNDED else None
    ci = rng.integers(len(P.DOTS['train']), size=9)
    cand = solvers['image'].solve(enc['image'](torch.from_numpy(np.stack([P.DOTS['train'][ci[k], k] for k in range(9)]))))[:, P.INNER]
    fi = torch.tensor([m[0] for m in match]); cj = torch.tensor([m[1] for m in match]); nn_ = torch.tensor([m[2] for m in match]) - 1
    C = cand[None].expand(len(match), -1, -1).clone()
    C[torch.arange(len(match)), nn_] = c[cj]
    logits = P.same_logits(model, c[fi][:, None].expand_as(C).reshape(-1, D.SIZES[2]), C.reshape(-1, D.SIZES[2])).reshape(len(match), 9)
    loss_match = F.cross_entropy(logits, nn_) + SO.acceptance(logits, nn_)
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    kinds = [items[p[0]][0] for p in pairs]
    by_kind = {k: [i for i, it in enumerate(items) if it[0] == k] for k in ('form', 'coll', 'taps')}
    na, nb = [], []
    for k in kinds:
        pool = by_kind[k]
        for _ in range(20):
            u, v = rng.choice(pool, 2)
            if items[v][2] != items[u][2] + 1:
                na.append(u); nb.append(v)
                break
    a2 = torch.cat([a, torch.tensor(na, dtype=torch.long)]); b2 = torch.cat([b, torch.tensor(nb, dtype=torch.long)])
    target = torch.cat([torch.ones(len(a)), torch.zeros(len(na))])
    loss_next = F.binary_cross_entropy_with_logits(model.relation_logits(c[a2], c[b2], 0), target)
    loss_cmp, u, v = torch.zeros(()), np.zeros(0, int), np.zeros(0, int)
    if compare:
        pool = np.array(by_kind['coll'] + by_kind['taps'])
        u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
        cnt = np.array([it[2] for it in items])
        keep = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
        u, v = u[keep], v[keep]
        loss_cmp = F.binary_cross_entropy_with_logits(model.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
    rows = []  # SP.sequence_loss's rows: (prev, cur, target collection, count)
    for steps in seqs:
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
    loss_t = torch.zeros(())
    if rows:
        prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
        cnt_t = torch.tensor([r[3] for r in rows]) - 1
        Ct = cand[None].expand(len(rows), -1, -1).clone()
        Ct[torch.arange(len(rows)), cnt_t] = c[tgt]
        lg = model.cont_logits(c[prev], c[cur], Ct)
        loss_t = F.cross_entropy(lg, cnt_t) + SO.acceptance(lg, cnt_t)
    # the audit: a name is only ever the source of a matching row
    used = {*a2.tolist(), *b2.tolist(), *u.tolist(), *v.tolist(), *cj.tolist(), *(i for r in rows for i in r[:3])}
    if named & used or not named <= set(fi.tolist()):
        raise AssertionError('a name reached a loss other than matching')
    au = W['audit']
    au['updates_with_names'] += bool(named)
    au['naming_rows'] += len(named)
    au['next_rows'] += len(a2); au['comparison_rows'] += len(u); au['continuation_rows'] += len(rows); au['matching_rows'] += len(match)
    loss = loss_match + loss_next + loss_cmp + loss_t
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('shared output update')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    model._M = None
    return [round(float(x.detach()), 4) for x in (loss_match, loss_next, loss_cmp, loss_t)]


# ---------------------------------------------------------------- reads

def _metrics(lg, n_of, tgt, thr, inv_n):
    """lg (q, N) continuation logits over M; n_of current counts; tgt target counts."""
    top = lg.argmax(1)
    top_n, top_v = inv_n[top], lg[np.arange(len(lg)), top]
    is_t = inv_n[None] == tgt[:, None]
    rev = inv_n[None] == (2 * n_of - tgt)[:, None]
    rep = inv_n[None] == n_of[:, None]
    acc = lg >= thr
    other = np.where(is_t, -np.inf, lg).max(1)
    au, au_h = [], []
    for i in range(len(lg)):
        au.append(auroc(lg[i, is_t[i]], lg[i, ~is_t[i]]))
        au_h.append(auroc(lg[i, is_t[i]], lg[i, rev[i] | rep[i]]))
    return {'recall': float(((top_n == tgt) & (top_v >= thr)).mean()), 'recall_no_abstention': float((top_n == tgt).mean()),
            'abstained': float((top_v < thr).mean()), 'top_is_reverse': float((top_n == 2 * n_of - tgt).mean()), 'top_is_repeat': float((top_n == n_of).mean()),
            'accept_correct': float(acc[is_t].mean()), 'accept_reverse': float(acc[rev].mean()), 'accept_repeat': float(acc[rep].mean()),
            'accept_rest': float(acc[~(is_t | rev | rep)].mean()), 'removed_set_abstains': float((other < thr).mean()),
            'auroc': float(np.mean(au)), 'auroc_reverse_repeat': float(np.mean(au_h)), 'queries': int(len(lg))}


@torch.no_grad()
def pilot_reads(enc, model, seed):
    for m in (enc['image'], enc['sequence'], model):
        m.eval()
    model._M, model.hard = None, False
    img = lambda x: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), 'image')
    rng = nc.stream('so_pilot_draws', seed)  # the same draws at every point and in every arm
    inv = W['inventory']
    inv_n = np.array([n for n, _ in inv])
    used_e = {n: {e for m, e in inv if m == n} for n in ns.NUMBERS}
    M = model.participants()
    counts = W['names']['counts']
    nm = img(W['names']['eval'].reshape(-1, 28, 28)).reshape(N_NAMES, EVAL_R, -1)
    un = img(W['unseen_images'].reshape(-1, 28, 28))
    colls = {n: img(P.DOTS['succ'][n]) for n in ns.NUMBERS}
    taps = {n: D.codes_for(enc, model, D.TAPS[('eval', n)], 'sequence') for n in ns.NUMBERS}
    withheld = set(D.withheld_counts(seed)[DEV_LANG])
    forms = {n: img(X.DATA[(DEV_LANG, 'eval')][0][P.EVAL[(DEV_LANG, n, 'idx')]]) for n in ns.NUMBERS}
    out = {}

    def cont(prev, cur):
        return model.cont_logits(prev, cur, M[None].expand(len(prev), -1, -1)).numpy()

    # validation thresholds: training-split collections outside M (continuation); training-split Devanagari forms and taps (identification)
    tr = P.DOTS['train']
    free = {n: np.array(sorted(set(range(len(tr))) - used_e[n])) for n in ns.NUMBERS}
    vp, vn = [], []
    for sign in (1, -1):
        for n in range(2, 9):
            e1, e2 = rng.choice(free[n - sign], 4), rng.choice(free[n], 4)
            lg = cont(img(tr[e1, n - sign - 1]), img(tr[e2, n - 1]))
            vp.append(lg[:, inv_n == n + sign].ravel()); vn.append(lg[:, inv_n != n + sign].ravel())
    thr_c, bal_c = SO.threshold(np.concatenate(vp), np.concatenate(vn))
    dx, dy = X.DATA[(DEV_LANG, 'train')]
    sp, sn_ = [], []
    for n in ns.NUMBERS:
        qs = []
        if n not in withheld:
            qs.append(img(dx[rng.choice(np.flatnonzero(dy == n), 5)]))
        qs.append(D.codes_for(enc, model, D.TAPS['train'][rng.choice(D.TAPS['train_idx'][n], 5)], 'sequence'))
        lg = model.identify(torch.cat(qs), M).numpy()
        sp.append(lg[:, inv_n == n].ravel()); sn_.append(lg[:, inv_n != n].ravel())
    thr_s, bal_s = SO.threshold(np.concatenate(sp), np.concatenate(sn_))
    out['thresholds'] = {'continuation': thr_c, 'continuation_validation_balanced_accuracy': bal_c, 'same': thr_s, 'same_validation_balanced_accuracy': bal_s}
    if model.arm in SO.COMMON:
        out['thresholds']['pooled'], out['thresholds']['pooled_validation_balanced_accuracy'] = SO.threshold(np.concatenate(vp + sp), np.concatenate(vn + sn_))

    # identification
    q = nm.reshape(-1, nm.shape[-1])
    qn = np.repeat(counts, EVAL_R)
    li = model.identify(q, M).numpy()
    ident = inv_n[li.argmax(1)] == qn
    accepted = li.max(1) >= thr_s
    mid = (qn >= 2) & (qn <= 8)
    lu = model.identify(un, M).numpy()
    fq = torch.cat([forms[n][:CUR_PER] for n in ns.NUMBERS if n not in withheld])
    fn = np.concatenate([np.full(len(forms[n][:CUR_PER]), n) for n in ns.NUMBERS if n not in withheld])  # smoke data can hold fewer than CUR_PER forms
    out['identification'] = {'names': float(ident.mean()), 'names_counts_2_8': float(ident[mid].mean()), 'names_accepted': float(accepted.mean()),
                             'names_identified_and_accepted': float((ident & accepted).mean()), 'unseen_accepted': float((lu.max(1) >= thr_s).mean()),
                             'devanagari_trained': float((inv_n[model.identify(fq, M).numpy().argmax(1)] == fn).mean())}
    ident_by = ident.reshape(N_NAMES, EVAL_R)

    # continuation (N1, N2)
    def cur_set(kind, n):
        if kind == 'name':
            sel = np.flatnonzero(counts == n)
            return nm[sel].reshape(-1, nm.shape[-1]), ident_by[sel].ravel()
        if kind == 'collection':
            return colls[n][rng.permutation(len(colls[n]))[:CUR_PER]], None
        if kind == 'taps':
            return taps[n][rng.permutation(len(taps[n]))[:CUR_PER]], None
        return (None, None) if n in withheld else (forms[n][rng.permutation(len(forms[n]))[:CUR_PER]], None)

    def prev_set(hist, n):
        if hist == 'collection':
            return colls[n][rng.permutation(len(colls[n]))[:HIST_PER]]
        if hist == 'taps':
            return taps[n][rng.permutation(len(taps[n]))[:HIST_PER]]
        return nm[rng.choice(np.flatnonzero(counts == n))][:HIST_PER]  # one name of the previous count, three renderings

    plan = [('name', 'collection'), ('collection', 'collection'), ('taps', 'collection'), ('devanagari', 'collection'), ('name', 'taps'), ('name', 'name')]
    draws = {}
    for kind, hist in plan:  # draw every query set once, so modes read the same queries
        for direction, sign in (('up', 1), ('down', -1)):
            for n in range(2, 9):
                cur, idf = cur_set(kind, n)
                if cur is None:
                    continue
                draws[kind, hist, direction, n] = (prev_set(hist, n - sign), cur, idf)
    modes = ['soft', 'hard'] if model.arm in SO.GROUNDED else ['direct', 'external-hard']
    cont_out = {}
    for mode in modes:
        model.hard = mode == 'hard'
        model._M = None
        for kind, hist in plan:
            if mode == 'external-hard' and kind != 'name':
                continue
            for direction, sign in (('up', 1), ('down', -1)):
                L, NO, TG, ID = [], [], [], []
                for n in range(2, 9):
                    if (kind, hist, direction, n) not in draws:
                        continue
                    prev, cur, idf = draws[kind, hist, direction, n]
                    if mode == 'external-hard':  # stage 0's diagnostic: the name replaced by its top participant before the raw readout
                        cur = M[torch.as_tensor(model.identify(cur, M).argmax(1))]
                        if hist == 'name':
                            prev = M[torch.as_tensor(model.identify(prev, M).argmax(1))]
                    pv, cu = prev.repeat_interleave(len(cur), 0), cur.repeat(len(prev), 1)
                    L.append(cont(pv, cu)); NO.append(np.full(len(pv), n)); TG.append(np.full(len(pv), n + sign))
                    if idf is not None:
                        ID.append(np.tile(idf, len(prev)))
                lg, no, tg = np.concatenate(L), np.concatenate(NO), np.concatenate(TG)
                r = _metrics(lg, no, tg, thr_c, inv_n)
                if ID:
                    idv = np.concatenate(ID)
                    r['recall_given_identified'] = _metrics(lg[idv], no[idv], tg[idv], thr_c, inv_n)['recall'] if idv.any() else None
                    r['share_identified'] = float(idv.mean())
                cont_out[f'{mode}/{kind}/{hist}/{direction}'] = r
    model.hard, model._M = False, None
    out['continuation'] = cont_out

    # code geometry of collections
    means = torch.stack([colls[n].mean(0) for n in ns.NUMBERS])
    within = float(np.mean([(colls[n] - means[n - 1]).norm(dim=1).mean().item() for n in ns.NUMBERS]))
    between = float(torch.pdist(means).mean())
    out['collection_geometry'] = {'between_count_means': between, 'within_count': within, 'ratio': between / within}

    # substitutability (docs/shared-output-v1-k-arms.md): is a name's nearest per-count mean of M its own count, in core space and, for the K arms,
    # in K space, where common-k-shared's relations read it. Added with the K arms; reports from before lack it.
    qn5 = np.repeat(counts, EVAL_R)
    spaces = {'core': lambda x: x}
    if model.arm in SO.KEYED:
        spaces['key'] = model.key
    sub = {}
    for space, f in spaces.items():
        mu = torch.stack([f(M)[torch.as_tensor(inv_n == n)].mean(0) for n in ns.NUMBERS])
        fq = f(nm.reshape(-1, nm.shape[-1]))
        near = np.asarray(torch.cdist(fq, mu).argmin(1)) + 1
        sub[space] = {'names_nearest_own_count': float((near == qn5).mean()), 'names_nearest_own_count_2_8': float((near == qn5)[(qn5 >= 2) & (qn5 <= 8)].mean())}
    out['substitutability'] = sub
    for m in (enc['image'], enc['sequence'], model):
        m.train()
    return out


def snapshot(enc, model, withheld, seed, point, disrupt):
    if isinstance(model, SO.Readout):
        model._M = None
    out = ORIG_SNAPSHOT(enc, model, withheld, seed, point, disrupt)
    if point in READ_POINTS and isinstance(model, SO.Readout):
        e0 = time.process_time()
        out['shared_output'] = pilot_reads(enc, model, seed)
        W['read_cpu'] += time.process_time() - e0
    return out


def pilot_task(task):
    arm, seed = task
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    z = dict(np.load(RUNS / f'names-seed{seed}.npz'))
    import pickle
    mem = pickle.load(open(MEMORY / f'memory-seed{seed}.pkl', 'rb'))
    W.update(arm=arm, names=z, inventory=inventory(seed), name_rng=nc.stream('so_pilot_names', seed), read_cpu=0.,
             unseen_images=mem['query_images']['unseen'], unseen_codes=np.concatenate([mem['query_codes']['unseen'][n].reshape(-1, SN.get(n).dim) for n in SC.MODELS], 1),
             audit={'updates_with_names': 0, 'naming_rows': 0, 'next_rows': 0, 'comparison_rows': 0, 'continuation_rows': 0, 'matching_rows': 0})
    del mem
    LM.RUNS = RUNS / ('pilot' + TAG) / arm
    X.make_model = lambda s: DEV.build(lambda: make_pilot_model(s, arm))
    if arm != 'gp-reference':
        P.images, SP.batch_material, LM.TP.train_update, LM.snapshot = images, batch_material, train_update, snapshot
    holder = {}
    orig_setup = LM.setup

    def setup(a, s):
        enc, model, opt = orig_setup(a, s)
        holder.update(enc=enc, model=model)
        return enc, model, opt
    LM.setup = setup
    try:
        key, res, rt = LM.run_main(('sparse48', seed))
    finally:
        LM.setup = orig_setup
    model = holder['model']
    res.update(arm=arm, audit=W['audit'], inventory=W['inventory'], reader_parameters=SO.reader_parameters(model) if isinstance(model, SO.Readout) else None)
    rt.update(read_cpu_seconds=W['read_cpu'], image_lookups=holder['enc']['image'].lookups, image_cache_misses=holder['enc']['image'].misses)
    if DEV.ACTIVE:
        rt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return f'{arm}/seed{seed}', res, rt


def pilot_main(arms, seeds, out):
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    X.make_model = None
    tasks = [(a, s) for a in arms for s in seeds]
    fn = DUR.durable(pilot_task, RUNS / ('done-pilot' + TAG), '{0}-seed{1}')
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(WORKERS, maxtasksperchild=1) as pool:
        res = pool.map(fn, tasks, chunksize=1)
    out = out + TAG
    S0.save_json(f'{REPORT}-{out}.json', {'status': 'pilot of the draft contract (not frozen); four readout arms on the step 8 curriculum with names taught only by matching in block 3',
                                          'host': HOST, 'arms': arms, 'seeds': seeds, 'results': {k: r for k, r, _ in res}})
    S0.save_json(f'{REPORT}-{out}-runtime.json', runtime({'phase': out, 'wall_seconds': time.perf_counter() - t0, 'workers': WORKERS,
                                                         'tasks': {k: t for k, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))
    return res


def reference():
    """The unchanged step 8 path at seed 300, compared measure by measure with the step 8 calibrated report."""
    import gzip, json
    res = pilot_main(('gp-reference',), (300,), 'reference')
    new = res[0][1]
    old = json.loads(gzip.decompress(STEP8.read_bytes()))['results'][f'{HOST}/seed300']
    flat = lambda o, p='': ({p: o} if not isinstance(o, (dict, list)) else
                            {k: v for i, x in (o.items() if isinstance(o, dict) else enumerate(o)) for k, v in flat(x, f'{p}/{i}').items()})
    a, b = flat({k: new[k] for k in ('snapshots', 'series', 'relearning', 'loss_every_100')}), flat({k: old[k] for k in ('snapshots', 'series', 'relearning', 'loss_every_100')})
    diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    print(f'reference: {len(set(a) & set(b)) - len([k for k in diff if k in a and k in b])} of {len(set(a) | set(b))} measures equal; {len(diff)} differ', diff[:10])
    S0.save_json(f'{REPORT}-reference-check.json', {'measures': len(set(a) | set(b)), 'differing': sorted(diff)})


if __name__ == '__main__':
    {'names': names_stage, 'stage0': stage0, 'reference': reference,
     'pilot': lambda: pilot_main(PILOT_ARMS, SEEDS, os.environ.get('SO_OUT', 'pilot'))}[sys.argv[2]]()
