"""Number systems v1: many representations, complementary sparse labels, held-out adaptation.

Implements the draft docs/number-systems-v1-spec.md. Mode `pilot` (the only mode until a contract
is frozen): seed 100, validation pools, 1,000 training images per number per representation.

Phases:
  1. data: render every representation (src/number_systems.py; MNIST and dice from v1 code);
  2. source: C2 (2 compatible sources), C14 (all 14 compatible sources), I14 (14 sources whose
     relations each follow their own permutation of the numbers); checkpoints at NSY_CKPTS;
  3. adaptation: each held-out representation (tree, german, thai_digits, hand) goes through its
     medium's existing satellite (one satellite per medium: image, sequence), labelled at 1, 2, 5,
     with within-representation relations, for NSY_ADAPT steps, from every source checkpoint.
     Modes: zero-shot (no adaptation), interface (the medium satellite adapts; core, output,
     successor and judge frozen), whole (everything adapts); plus a no-source baseline (whole
     system from random initialization).

Deterministic results and runtime data go to separate files under reports/. Environment:
NSY_STEPS (3000), NSY_CKPTS (1000,3000), NSY_ADAPT (300), NSY_WORKERS, NSY_SMOKE=1.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import json
import multiprocessing as mp
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import number_concept as nc  # noqa: E402
from src import number_core as ncore  # noqa: E402
from src import number_systems as ns  # noqa: E402
from src import number_systems_net as net  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
SMOKE = os.environ.get('NSY_SMOKE') == '1'
SPEC = ROOT / 'docs/number-systems-v1-spec.md'
NAME = f'number-systems-v1-{MODE}' + ('-smoke' if SMOKE else '')
PREFIX, RUNS = ROOT / 'reports' / NAME, ROOT / 'runs' / NAME
WORKERS = int(os.environ.get('NSY_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))
STEPS = 20 if SMOKE else int(os.environ.get('NSY_STEPS', '3000'))
CKPTS = (10, 20) if SMOKE else tuple(int(c) for c in os.environ.get('NSY_CKPTS', '1000,3000').split(','))
ADAPT = 10 if SMOKE else int(os.environ.get('NSY_ADAPT', '300'))
PER_TRAIN, PER_EVAL = (40, 20) if SMOKE else (1000, 300)
SEED, K, BATCH, PAIRS, LR = 100, 4, 64, 64, 1e-3
HELD_OUT = ('tree', 'german', 'thai_digits', 'hand')
SOURCES = ('mnist', 'dice', 'dots', 'tally', 'banana', 'egyptian', 'roman', 'chinese', 'devanagari_digits',
           'arabic_digits', 'english', 'french', 'hindi', 'morse')
SOURCE_RUNS = {'C2': (2, False), 'C14': (14, False), 'I14': (14, True)}  # name: (N sources, incompatible)
ADAPT_LABELS = (1, 2, 5)
MEDIUM = {**{s: v[1] for s, v in ns.SYSTEMS.items()}, 'mnist': 'image', 'dice': 'image'}
DATA = {}


# ---------------------------------------------------------------- data

def numbers(r):
    return nc.FACES if r == 'dice' else ns.NUMBERS


def render_pool(args):
    system, split, per = args
    t = time.perf_counter()
    pool = ns.build(system, split, per)
    return system, split, pool, time.perf_counter() - t


def prepare():
    raw, checks = nc.load_mnist(ROOT / 'data/mnist')
    train, val = nc.mnist_split(raw['train_labels'])
    lab = raw['train_labels']
    timings = {}
    for split, idx, per in (('train', train, PER_TRAIN), ('eval', val, PER_EVAL)):
        keep = np.concatenate([nc.stream('mnist_pick', split, d).permutation(idx[lab[idx] == d])[:per] for d in ns.NUMBERS])
        DATA[('mnist', split)] = (raw['train_images'][keep].astype(np.float32) / 255, lab[keep])
        pool = nc.build_dice('standard', 'train' if split == 'train' else 'validation')
        keep = np.concatenate([np.flatnonzero(pool['labels'] == c)[:per] for c in nc.FACES])
        DATA[('dice', split)] = (pool['images'][keep], pool['labels'][keep])
    jobs = [(s, split, PER_TRAIN if split == 'train' else PER_EVAL) for s in ns.SYSTEMS for split in ('train', 'eval')]
    with mp.get_context('fork').Pool(WORKERS) as pool:
        for system, split, p, sec in pool.imap_unordered(render_pool, jobs):
            DATA[(system, split)] = (p['x'], p['y'])
            timings[f'{system}/{split}'] = sec
    return {'mnist_sha256': checks, 'render_seconds': timings,
            'sizes': {f'{k[0]}/{k[1]}': int(len(v[1])) for k, v in DATA.items()},
            'data_bytes': int(sum(v[0].nbytes for v in DATA.values()))}


def assign_labels(sources, seed):
    """Three labelled numbers per source, balancing how often each number is labelled; no repeated sets."""
    rng = nc.stream('label_assignment', seed, len(sources))
    count, used, out = {n: 0 for n in ns.NUMBERS}, set(), {}
    for r in sources:
        best = None
        for _ in range(400):
            s = tuple(sorted(rng.choice(numbers(r), 3, replace=False).tolist()))
            if s in used:
                continue
            cost = sum(count[n] for n in s) + rng.random() * .1
            if best is None or cost < best[0]:
                best = (cost, s)
        out[r] = best[1]
        used.add(best[1])
        for n in best[1]:
            count[n] += 1
    return out


def orders(sources, incompatible, seed):
    """Relation order per representation: identity, or (incompatible) a fixed random permutation of its numbers."""
    out = {}
    for r in sources:
        nums = list(numbers(r))
        perm = nc.stream('incompatible_order', seed, r).permutation(nums).tolist() if incompatible else nums
        out[r] = {n: perm.index(n) + 1 for n in nums}  # number -> position in the relation order
    return out


# ---------------------------------------------------------------- batches and losses

def encode(enc, sri, groups, split):
    """groups: {rep: index array}; one front-end pass and one equilibrium map per medium."""
    by_medium = {}
    for r, idx in groups.items():
        by_medium.setdefault(MEDIUM[r], []).append((r, DATA[(r, split)][0][idx]))
    feats = {}
    for med, items in by_medium.items():
        f = enc[med](torch.from_numpy(np.concatenate([x for _, x in items])))
        start = 0
        for r, x in items:
            feats[r] = f[start:start + len(x)]
            start += len(x)
    maps = {med: sri.equilibrium_map(med)[0] for med in by_medium}
    return {r: sri.split(f @ maps[MEDIUM[r]].T) for r, f in feats.items()}


def task_batch(rng, r, idx, labelled, order):
    """Index requests and targets for one representation: labels, then same / next / larger pairs."""
    nums = list(numbers(r))
    req, tasks = [], []
    cells = np.array(labelled)[rng.integers(0, len(labelled), BATCH)]
    req.append(np.array([rng.choice(idx[c]) for c in cells]))
    tasks.append(('label', torch.as_tensor(cells)))
    rels = {0: lambda a, b: order[b] == order[a] + 1, 1: lambda a, b: a == b, 2: lambda a, b: order[a] > order[b]}
    for head, rel in rels.items():
        pos = [(a, b) for a in nums for b in nums if rel(a, b)]
        neg = [(a, b) for a in nums for b in nums if not rel(a, b)]
        pc = [pos[i] for i in rng.integers(0, len(pos), PAIRS // 2)] + [neg[i] for i in rng.integers(0, len(neg), PAIRS // 2)]
        req.append(np.array([rng.choice(idx[a]) for a, _ in pc]))
        req.append(np.array([rng.choice(idx[b]) for _, b in pc]))
        tasks.append((head, torch.tensor([1.] * (PAIRS // 2) + [0.] * (PAIRS // 2))))
    return np.concatenate(req), tasks


def batch_loss(sri, codes, st, tasks):
    loss, pos = 0., 0
    for kind, target in tasks:
        if kind == 'label':
            out = st['output_outer'][pos:pos + BATCH]
            loss = loss + (1 - F.cosine_similarity(out, codes[target], dim=1)).mean()
            pos += BATCH
        else:
            ca, cb = st['core_inner'][pos:pos + PAIRS], st['core_inner'][pos + PAIRS:pos + 2 * PAIRS]
            loss = loss + F.binary_cross_entropy_with_logits(sri.relation_logits(ca, cb, kind), target)
            pos += 2 * PAIRS
    return loss


def make_model(seed):
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(net.SIZES[0]), 'sequence': ncore.MorseEncoder(net.SIZES[0])}
    return enc, net.MultiSRI(('image', 'sequence'), torch.Generator().manual_seed(seed))


def train_steps(enc, sri, opt, reps, idx, labelled, order, steps, rng, codes, on_step=None):
    losses, encodings = [], 0
    for step in range(steps):
        chosen = rng.choice(reps, min(K, len(reps)), replace=False).tolist() if len(reps) > K else list(reps)
        groups, tasks = {}, {}
        for r in chosen:
            groups[r], tasks[r] = task_batch(rng, r, idx[r], labelled[r], order[r])
        st = encode(enc, sri, groups, 'train')
        encodings += sum(len(g) for g in groups.values())
        loss = sum(batch_loss(sri, codes, st[r], tasks[r]) for r in chosen)
        if not torch.isfinite(loss):
            raise nc.NumericalFailure(f'step {step}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 99 or step == steps - 1:
            losses.append(round(loss.item(), 4))
        if on_step:
            on_step(step + 1)
    return losses, encodings


def source_fit(name):
    n, incompatible = SOURCE_RUNS[name]
    t0, c0 = time.perf_counter(), time.process_time()
    sources = list(SOURCES) if n == len(SOURCES) else sorted(nc.stream('source_subset', SEED, n).choice(SOURCES, n, replace=False).tolist())
    labelled, order = assign_labels(sources, SEED), orders(sources, incompatible, SEED)
    enc, sri = make_model(SEED)
    codes = ncore.label_codes(net.SIZES[0])
    params = [*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()]
    opt = torch.optim.Adam(params, LR)
    idx = {r: {c: np.flatnonzero(DATA[(r, 'train')][1] == c) for c in numbers(r)} for r in sources}
    RUNS.mkdir(parents=True, exist_ok=True)
    saved, results = {}, {}

    def checkpoint(step):
        if step in CKPTS:
            path = RUNS / f'{name}-{step}.pt'
            torch.save({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': sri.state_dict(),
                        'sources': sources, 'labelled': labelled}, path)
            saved[step] = {'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)}
            results[step] = evaluate_source(enc, sri, codes, sources, labelled)
    losses, encodings = train_steps(enc, sri, opt, sources, idx, labelled, order, STEPS, nc.stream('source_train', SEED, name), codes, checkpoint)
    return ('source', name), {'sources': sources, 'labelled': {r: list(v) for r, v in labelled.items()},
                              'incompatible_orders': {r: v for r, v in order.items()} if incompatible else None,
                              'loss_every_100': losses, 'checkpoints': saved, 'eval': results}, \
        {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'steps': STEPS,
         'training_encodings': encodings, 'factorizations': None,
         'factorizations_note': 'one per medium present in each step (1 or 2)',
         'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


# ---------------------------------------------------------------- evaluation

def probe(xtr, ytr, xte, yte, k, seed, steps=300):
    g = torch.Generator().manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    lin = torch.nn.Linear(xtr.shape[1], k)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * .01)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(xte).argmax(1) == yte).float().numpy()


@torch.no_grad()
def eval_codes(enc, sri, reps):
    groups = {r: np.arange(len(DATA[(r, 'eval')][1])) for r in reps}
    return {r: v for r, v in encode(enc, sri, groups, 'eval').items()}


def label_acc(st, y, codes):
    pred = (st['output_outer'] @ codes.T).argmax(1).numpy()
    return {int(c): float((pred[y == c] == c).mean()) for c in np.unique(y)}


def evaluate_source(enc, sri, codes, sources, labelled):
    enc['image'].eval(); enc['sequence'].eval()
    st = eval_codes(enc, sri, sources)
    out = {'label_accuracy': {}, 'complementary_transfer': {}}
    for r in sources:
        y = DATA[(r, 'eval')][1]
        acc = label_acc(st[r], y, codes)
        out['label_accuracy'][r] = {str(k): v for k, v in acc.items()}
        out['complementary_transfer'][r] = {'labelled': float(np.mean([acc[n] for n in labelled[r]])),
                                            'unlabelled': float(np.mean([v for n, v in acc.items() if n not in labelled[r]]))}
    out['complementary_transfer_mean'] = float(np.mean([v['unlabelled'] for v in out['complementary_transfer'].values()]))
    # Input-kind decodability (numbers 1-6, present in every representation), chance 1/N.
    rng = nc.stream('probe_split', len(sources))
    X, Y = [], []
    for j, r in enumerate(sources):
        y = DATA[(r, 'eval')][1]
        sel = np.flatnonzero(y <= 6)
        X.append(st[r]['core'][sel]); Y += [j] * len(sel)
    X, Y = torch.cat(X), torch.tensor(Y)
    p = torch.as_tensor(rng.permutation(len(Y)))
    h = len(p) // 2
    out['input_kind_probe'] = {'accuracy': float(probe(X[p[:h]], Y[p[:h]], X[p[h:]], Y[p[h:]], len(sources), 0).mean()),
                               'chance': 1 / len(sources)} if len(sources) > 1 else None
    # Cross-representation number readout: probe trained on one representation's core codes, tested on the others.
    cross = {}
    for a in sources:
        ya = torch.as_tensor(DATA[(a, 'eval')][1])
        for b in sources:
            if a == b:
                continue
            yb = torch.as_tensor(DATA[(b, 'eval')][1])
            cross[f'{a}->{b}'] = float(probe(st[a]['core'], ya, st[b]['core'], yb, 10, 1).mean())
    out['cross_number_probe'] = cross
    out['cross_number_probe_mean'] = float(np.mean(list(cross.values()))) if cross else None
    enc['image'].train(); enc['sequence'].train()
    return out


# ---------------------------------------------------------------- adaptation

def adapt(job):
    source, step, held, mode = job
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri = make_model(SEED)
    sources = []
    if source is not None:
        ck = torch.load(RUNS / f'{source}-{step}.pt', weights_only=False)
        sources = ck['sources']
        enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
        sri.load_state_dict(ck['sri'])
    codes = ncore.label_codes(net.SIZES[0])
    med = MEDIUM[held]
    losses, encodings = [], 0
    if mode != 'zero-shot':
        params = [*enc[med].parameters(), *sri.satellite_parameters(med)] if mode == 'interface' else \
            [*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()]
        opt = torch.optim.Adam(params, LR)
        idx = {held: {c: np.flatnonzero(DATA[(held, 'train')][1] == c) for c in numbers(held)}}
        order = {held: {n: n for n in numbers(held)}}
        losses, encodings = train_steps(enc, sri, opt, [held], idx, {held: ADAPT_LABELS}, order, ADAPT,
                                        nc.stream('adapt', SEED, str(source), step, held, mode), codes)
    enc['image'].eval(); enc['sequence'].eval()
    with torch.no_grad():
        st = eval_codes(enc, sri, [held] + sources)
    y = DATA[(held, 'eval')][1]
    acc = label_acc(st[held], y, codes)
    res = {'loss_every_100': losses, 'label_accuracy': {str(k): v for k, v in acc.items()},
           'labelled': float(np.mean([acc[n] for n in ADAPT_LABELS])),
           'interpolation_3_4': float(np.mean([acc[3], acc[4]])),
           'extrapolation_6_9': float(np.mean([acc[n] for n in (6, 7, 8, 9)]))}
    if sources:
        # Number readout trained on pooled source core codes, applied to the held-out representation.
        X = torch.cat([st[r]['core'] for r in sources])
        Y = torch.cat([torch.as_tensor(DATA[(r, 'eval')][1]) for r in sources])
        hit = probe(X, Y, st[held]['core'], torch.as_tensor(y), 10, 2)
        res['source_probe_on_held_out'] = {str(c): float(hit[y == c].mean()) for c in ns.NUMBERS}
        res['source_probe_unlabelled'] = float(hit[~np.isin(y, ADAPT_LABELS)].mean())
        res['retention_source_label_accuracy'] = float(np.mean([np.mean(list(label_acc(st[r], DATA[(r, 'eval')][1], codes).values()))
                                                                for r in sources]))
    return ('adapt', source, step, held, mode), res, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'steps': 0 if mode == 'zero-shot' else ADAPT, 'training_encodings': encodings,
        'factorizations': 0 if mode == 'zero-shot' else ADAPT, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(job):
    return source_fit(job[1]) if job[0] == 'source' else adapt(job[1:])


# ---------------------------------------------------------------- main

def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    if MODE != 'pilot':
        raise SystemExit('only the pilot mode exists until a contract is frozen')
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    done = []
    ctx = mp.get_context('fork')
    t = time.perf_counter()
    first = [('source', n) for n in SOURCE_RUNS] + [('adapt', None, 0, h, 'whole') for h in HELD_OUT]
    with ctx.Pool(min(WORKERS, len(first))) as pool:
        for item in pool.imap_unordered(run, first):
            done.append(item)
            print(f'done {item[0]} ({len(done)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['source_wall_seconds'] = time.perf_counter() - t
    t = time.perf_counter()
    second = [('adapt', s, c, h, m) for s in SOURCE_RUNS for c in CKPTS for h in HELD_OUT for m in ('zero-shot', 'interface', 'whole')]
    with ctx.Pool(WORKERS) as pool:
        for item in pool.imap_unordered(run, second):
            done.append(item)
            print(f'done {item[0]} ({len(done)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['adaptation_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    res = {'spec': str(SPEC.relative_to(ROOT)), 'mode': MODE, 'status': 'pilot: not headline evidence',
           'config': {'seed': SEED, 'steps': STEPS, 'checkpoints': CKPTS, 'adapt_steps': ADAPT, 'per_number': [PER_TRAIN, PER_EVAL],
                      'reps_per_step': K, 'batch': BATCH, 'pairs': PAIRS, 'lr': LR, 'held_out': HELD_OUT, 'sources': SOURCES,
                      'source_runs': SOURCE_RUNS, 'adapt_labels': ADAPT_LABELS, 'sizes': net.SIZES,
                      'label_loss': 'positive-only cosine pull for every representation'},
           'data': {k: v for k, v in data.items() if k != 'render_seconds'},
           'results': {key(k): r for k, r, _ in sorted(done, key=lambda d: key(d[0]))}}
    PREFIX.parent.mkdir(exist_ok=True)
    Path(f'{PREFIX}-results.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': WORKERS, 'threads_per_worker': 1, 'dtype': 'float32',
        'phases': phases, 'render_seconds': data['render_seconds'], 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process',
        'deployment_timing': None, 'deployment_timing_reason': 'not measured in the pilot'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
