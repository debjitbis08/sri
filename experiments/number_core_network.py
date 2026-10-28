"""Number core network v1: image and Morse modules coupled through a core to a label output.

Implements docs/number-core-network-v1-spec.md (draft). Modes:

  pilot         seed 100, all eight arms at narrow width (16 -> 8 -> 4), NCC_STEPS steps
                (default 1,000), validation pools only. Ran with the original softmax label
                loss for every modality (runner hash in its runtime file).
  tally         exploratory tally-mark test on the pilot2 structure: tally images (counts 1-10) go
                through the image encoder, labelled only at 1, 2 and 5 (cosine pull). Arms N-M,
                S-M and S-X-E at wide width, each with tally labels only and with labels plus
                within-tally same/next relations. Seed 100, validation pools.
  pilot2        after the pilot amendments: seed 100, all eight arms at wide width
                (64 -> 32 -> 16) plus S-M and S-X-E at narrow width, same steps and pools.
                Label loss: softmax for MNIST, positive-only cosine pull for dice and Morse.
  confirmation  refuses to run until docs/number-core-network-v1-contract.md exists.

Deterministic results and nondeterministic runtime data go to separate files under
reports/; checkpoints go to runs/ (gitignored). Environment: NCC_STEPS, NCC_WORKERS,
NCC_SMOKE=1 (20 steps, small evaluation).
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import hashlib
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

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

MODE = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
SMOKE = os.environ.get('NCC_SMOKE') == '1'
SPEC = ROOT / 'docs/number-core-network-v1-spec.md'
CONTRACT = ROOT / 'docs/number-core-network-v1-contract.md'
NAME = f'number-core-network-v1-{MODE}' + ('-smoke' if SMOKE else '')
PREFIX = ROOT / 'reports' / NAME
RUNS = ROOT / 'runs' / NAME
WORKERS = int(os.environ.get('NCC_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))
STEPS = 20 if SMOKE else int(os.environ.get('NCC_STEPS', '1000'))
PILOT = MODE in ('pilot', 'pilot2', 'tally')
TALLY_MODE = MODE == 'tally'
TALLY_LABELLED = (1, 2, 5)
SEEDS = (100,) if PILOT else (0, 1, 2, 3, 4)
SPLIT = 'validation' if PILOT else 'test'
ARMS = {  # arm: (succession, scrambled Morse, equivalence loss, topology)
    'S-M': (True, False, False, 'uniform'), 'N-M': (False, False, False, 'uniform'),
    'S-X': (True, True, False, 'uniform'), 'N-X': (False, True, False, 'uniform'),
    'S-X-E': (True, True, True, 'uniform'), 'N-X-E': (False, True, True, 'uniform'),
    'S-X-I': (True, True, False, 'inner'), 'N-X-I': (False, True, False, 'inner'),
}
BATCH, PAIRS, FEW, LR, REF = 64, 64, 10, 1e-3, 20
GROUPS = {'dice': {'labelled': (1, 3, 5), 'interpolation': (2, 4), 'extrapolation': (6,)},
          'morse': {'labelled': (1, 3, 5), 'interpolation': (2, 4), 'extrapolation_up': (6, 7, 8, 9),
                    'extrapolation_down': (0,)}}
GROUPS['tally'] = {'labelled': (1, 2, 5), 'interpolation': (3, 4), 'extrapolation': (6, 7, 8, 9), 'ten': (10,)}
CLASSES = {'mnist': tuple(range(10)), 'dice': nc.FACES, 'morse': tuple(range(10)), 'tally': ncore.TALLY_COUNTS}
KIND = {'mnist': 'image', 'dice': 'image', 'morse': 'morse', 'tally': 'image'}
DATA = {}


# ---------------------------------------------------------------- data

def prepare():
    raw, checks = nc.load_mnist(ROOT / 'data/mnist')
    train, val = nc.mnist_split(raw['train_labels'])
    lab = raw['train_labels']
    img = lambda a: (a.astype(np.float32) / 255)
    dtr, dev = nc.build_dice('standard', 'train'), nc.build_dice('standard', SPLIT)
    mtr, mev = ncore.build_morse('train'), ncore.build_morse(SPLIT)
    if PILOT:
        ev_images, ev_labels = img(raw['train_images'][val]), lab[val]
    else:
        ev_images, ev_labels = img(raw['test_images']), raw['test_labels']
    perm = ncore.scramble()
    DATA.update(mnist_train=img(raw['train_images'][train]), mnist_train_y=lab[train],
                mnist_eval=ev_images, mnist_eval_y=ev_labels,
                dice_train=dtr['images'], dice_train_y=dtr['labels'], dice_eval=dev['images'], dice_eval_y=dev['labels'],
                morse_train=mtr['signals'], morse_train_code=mtr['codes'],
                morse_eval=mev['signals'], morse_eval_code=mev['codes'], perm=perm)
    if TALLY_MODE:
        ttr, tev = ncore.build_tally('train'), ncore.build_tally(SPLIT)
        DATA.update(tally_train=ttr['images'], tally_train_y=ttr['labels'], tally_eval=tev['images'], tally_eval_y=tev['labels'])
    return {'mnist_sha256': checks, 'scramble_perm_number_to_code': perm.tolist(),
            'morse_max_end_sample': int(max(mtr['ends'].max(), mev['ends'].max())),
            'morse_sha256': {s: hashlib.sha256(m['signals'].tobytes()).hexdigest() for s, m in (('train', mtr), (SPLIT, mev))},
            'eval_split': SPLIT, 'mnist_test_loaded': not PILOT,
            'sizes': {'mnist_train': int(len(train)), 'mnist_eval': int(len(ev_labels)), 'dice_train': int(len(dtr['labels'])),
                      'dice_eval': int(len(dev['labels'])), 'morse_train': int(len(mtr['codes'])),
                      'morse_eval': int(len(mev['codes']))},
            'data_bytes': {'mnist_train_float32': int(DATA['mnist_train'].nbytes), 'dice_float32': int(dtr['images'].nbytes + dev['images'].nbytes),
                           'morse_float32': int(mtr['signals'].nbytes + mev['signals'].nbytes)}}


def labels(split, scrambled):
    """Number labels per modality for a split ('train' or 'eval')."""
    perm = DATA['perm'] if scrambled else np.arange(10)
    out = {'mnist': DATA[f'mnist_{split}_y'], 'dice': DATA[f'dice_{split}_y'],
           'morse': ncore.morse_numbers(DATA[f'morse_{split}_code'], perm)}
    if TALLY_MODE:
        out['tally'] = DATA[f'tally_{split}_y']
    return out


def by_class(y):
    return {int(c): np.flatnonzero(y == c) for c in np.unique(y)}


# ---------------------------------------------------------------- batches

class Batch:
    """Collects (modality, index) requests, encodes them in one pass per front end, returns states."""
    def __init__(self, split):
        self.split, self.req = split, {'image': [], 'morse': []}

    def add(self, modality, idx):
        kind = KIND[modality]
        start = sum(len(a) for a in self.req[kind])
        arr = DATA[f'{modality}_{self.split}'][idx]
        self.req[kind].append(arr)
        return kind, slice(start, start + len(idx))

    def run(self, enc, sri):
        out = {}
        for kind, arrs in self.req.items():
            if not arrs:
                continue
            x = torch.from_numpy(np.concatenate(arrs))
            feats = enc[kind](x)
            M, _ = sri.equilibrium_map(kind)
            out[kind] = sri.split(feats @ M.T)
        return out


def pair_cells(rng, ca, cb, relation, n):
    rel = (lambda a, b: b == a + 1) if relation == 'next' else (lambda a, b: b == a)
    pos = [(a, b) for a in ca for b in cb if rel(a, b)]
    neg = [(a, b) for a in ca for b in cb if not rel(a, b)]
    cells = [pos[i] for i in rng.integers(0, len(pos), n // 2)] + [neg[i] for i in rng.integers(0, len(neg), n - n // 2)]
    return cells, torch.tensor([1.] * (n // 2) + [0.] * (n - n // 2))


def logits_rel(sri, ca, cb, head):
    d = (cb - (sri.S(ca) if head == 0 else ca)).pow(2).sum(1)
    return sri.judge_bias[head] - F.softplus(sri.judge_scale[head]) * d


# ---------------------------------------------------------------- training

def jobs_for_mode():
    if TALLY_MODE:
        return [(s, a, 'wide', t) for t in ('labels', 'relations') for a in ('N-M', 'S-M', 'S-X-E') for s in SEEDS]
    if MODE == 'pilot2':
        return [(s, a, 'wide') for a in ARMS for s in SEEDS] + [(s, a, 'narrow') for a in ('S-M', 'S-X-E') for s in SEEDS]
    if MODE == 'pilot':
        return [(s, a, 'narrow') for a in ARMS for s in SEEDS]
    return [(s, a, 'wide') for a in ARMS for s in SEEDS]


def label_loss(modality, out, codes, beta, cells):
    """MNIST: softmax over the ten codes. Dice and Morse (sparse labels): positive-only cosine pull
    toward the correct code, so unlabelled numbers are never pushed away (pilot amendment 1)."""
    if MODE == 'pilot' or modality == 'mnist':
        return F.cross_entropy(beta * out @ codes.T, cells)
    return (1 - F.cosine_similarity(out, codes[cells], dim=1)).mean()


def fit(job):
    seed, arm, width, *rest = job
    tally = rest[0] if rest else None
    succession, scrambled, equivalence, topology = ARMS[arm]
    sizes = ncore.WIDTHS[width]
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(seed)
    enc = {'image': ncore.ImageEncoder(sizes[0]), 'morse': ncore.MorseEncoder(sizes[0])}
    sri = ncore.SRI(topology, torch.Generator().manual_seed(seed), sizes)
    codes = ncore.label_codes(sizes[0])
    params = [*enc['image'].parameters(), *enc['morse'].parameters(), *sri.parameters()]
    opt = torch.optim.Adam(params, LR)
    y = labels('train', scrambled)
    idx = {m: by_class(y[m]) for m in y}
    rng = nc.stream('core_train', seed, arm)
    label_classes = {'mnist': CLASSES['mnist'], 'dice': ncore.LABELLED, 'morse': ncore.LABELLED}
    if tally:
        label_classes['tally'] = TALLY_LABELLED
    few = {m: {k: nc.stream('core_few', seed, m, k).choice(idx[m][k], FEW, replace=False) for k in ncore.LABELLED}
           for m in ('mnist', 'dice', 'morse')}
    losses, encodings = [], 0
    for step in range(STEPS):
        b = Batch('train')
        label_req = []
        for m, cls in label_classes.items():
            cells = np.array(cls)[rng.integers(0, len(cls), BATCH)]
            label_req.append((m, b.add(m, np.array([rng.choice(idx[m][c]) for c in cells])), torch.as_tensor(cells)))
        rel_req = []
        for m in ('mnist', 'dice', 'morse') + (('tally',) if tally == 'relations' else ()):
            for head, relation in ((0, 'next'), (1, 'same')):
                if relation == 'next' and not succession:
                    continue
                cells, t = pair_cells(rng, CLASSES[m], CLASSES[m], relation, PAIRS)
                ra = b.add(m, np.array([rng.choice(idx[m][a]) for a, _ in cells]))
                rb = b.add(m, np.array([rng.choice(idx[m][c]) for _, c in cells]))
                rel_req.append((ra, rb, t, head))
        if equivalence:
            image_few = {k: [('mnist', i) for i in few['mnist'][k]] + [('dice', i) for i in few['dice'][k]] for k in ncore.LABELLED}
            morse_few = {k: [('morse', i) for i in few['morse'][k]] for k in ncore.LABELLED}
            mnist_few = {k: [('mnist', i) for i in few['mnist'][k]] for k in ncore.LABELLED}
            dice_few = {k: [('dice', i) for i in few['dice'][k]] for k in ncore.LABELLED}
            for fa, fb in ((image_few, morse_few), (morse_few, image_few), (mnist_few, dice_few), (dice_few, mnist_few)):
                cells, t = pair_cells(rng, ncore.LABELLED, ncore.LABELLED, 'same', PAIRS)
                pa = [fa[a][rng.integers(len(fa[a]))] for a, _ in cells]
                pb = [fb[c][rng.integers(len(fb[c]))] for _, c in cells]
                # Each side may mix MNIST and dice: add per element, then gather.
                ra = [b.add(m, np.array([i])) for m, i in pa]
                rb = [b.add(m, np.array([i])) for m, i in pb]
                rel_req.append((ra, rb, t, 1))
        st = b.run(enc, sri)
        encodings += sum(sum(len(a) for a in v) for v in b.req.values())

        def get(r, key):
            if isinstance(r, list):
                return torch.cat([st[k][key][s] for k, s in r])
            return st[r[0]][key][r[1]]
        loss = 0.
        for m, r, cells in label_req:
            loss = loss + label_loss(m, get(r, 'output_outer'), codes, F.softplus(sri.label_scale), cells)
        for ra, rb, t, head in rel_req:
            loss = loss + F.binary_cross_entropy_with_logits(logits_rel(sri, get(ra, 'core_inner'), get(rb, 'core_inner'), head), t)
        if not torch.isfinite(loss):
            raise nc.NumericalFailure(f'{seed}/{arm} step {step}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 49 or step == STEPS - 1:
            losses.append(round(loss.item(), 5))
    train_wall, train_cpu = time.perf_counter() - t0, time.process_time() - c0
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (f'{seed}-{arm}.pt' if MODE == 'pilot' else '-'.join(map(str, job)) + '.pt')
    torch.save({'image': enc['image'].state_dict(), 'morse': enc['morse'].state_dict(), 'sri': sri.state_dict()}, path)
    t1, c1 = time.perf_counter(), time.process_time()
    result = evaluate(enc, sri, codes, seed, arm, scrambled, succession)
    result.update(loss_every_50=losses, checkpoint={'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)},
                  max_abs_weight={k: float(v.detach().abs().max()) for k, v in sri.named_parameters()})
    runtime = {'train_wall_seconds': train_wall, 'train_cpu_seconds': train_cpu,
               'eval_wall_seconds': time.perf_counter() - t1, 'eval_cpu_seconds': time.process_time() - c1,
               'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'counts': {'optimizer_steps': STEPS, 'training_encodings': encodings, 'factorizations': 2 * STEPS + 2}}
    return job, result, runtime


# ---------------------------------------------------------------- evaluation

def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float((y.T @ x).pow(2).sum() / ((x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()).clamp_min(1e-12))


def coord_corr(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float(((x * y).sum(0) / (x.norm(dim=0) * y.norm(dim=0)).clamp_min(1e-12)).mean())


def per_class(pred, y, classes, target=lambda c: c):
    return {int(c): float((pred[y == c] == target(c)).mean()) for c in classes if (y == c).any()}


def grouped(per, modality):
    return {g: float(np.mean([per[c] for c in cs if c in per])) for g, cs in GROUPS[modality].items() if any(c in per for c in cs)}


@torch.no_grad()
def evaluate(enc, sri, codes, seed, arm, scrambled, succession):
    rng = nc.stream('core_eval', seed)
    y = labels('eval', scrambled)
    lim = 50 if SMOKE else None
    sel = {m: np.sort(np.concatenate([np.flatnonzero(y[m] == c)[:lim] for c in CLASSES[m]])) for m in y}
    st, residual = {}, {}
    for m in y:
        kind = KIND[m]
        feats = enc[kind](torch.from_numpy(DATA[f'{m}_eval'][sel[m]]))
        M, Q = sri.equilibrium_map(kind)
        free = feats @ M.T
        full = torch.cat([feats, free], 1)
        residual[m] = float(((full @ Q.T)[:, sri.sizes[0]:]).norm() / (feats.norm() * Q.norm()))
        st[m] = sri.split(free)
        y[m] = y[m][sel[m]]
    out = {'solve_relative_residual': residual, 'label_accuracy': {}, 'coupling': {}, 'nearest_mnist_centroid': {},
           'cross_relations': {}}
    for m in y:
        pred = (st[m]['output_outer'] @ codes.T).argmax(1).numpy()
        per = per_class(pred, y[m], CLASSES[m])
        out['label_accuracy'][m] = {'per_class': {str(k): v for k, v in per.items()}, 'mean': float(np.mean(list(per.values()))),
                                    'modal_label': {str(c): int(np.bincount(pred[y[m] == c], minlength=10).argmax()) for c in CLASSES[m]}}
        if m != 'mnist':
            out['label_accuracy'][m]['groups'] = grouped(per, m)
    # Coupling: same-number pairs across modalities; references within MNIST and shuffled.
    pairs = (('morse', 'mnist'), ('dice', 'mnist'), ('morse', 'dice'), ('mnist', 'mnist')) + ((('tally', 'mnist'),) if 'tally' in y else ())
    others = ('dice', 'morse') + (('tally',) if 'tally' in y else ())
    for key in ('core', 'core_inner', 'input_inner'):
        out['coupling'][key] = {}
        for a, b in pairs:
            ia = np.flatnonzero(np.isin(y[a], CLASSES[b]))
            ib = np.array([rng.choice(np.flatnonzero(y[b] == c)) for c in y[a][ia]])
            xa, xb = st[a][key][ia], st[b][key][ib]
            out['coupling'][key][f'{a}-{b}'] = {'paired_cka': cka(xa, xb), 'coordinate_correlation': coord_corr(xa, xb),
                                                'shuffled_cka': cka(xa, xb[torch.as_tensor(rng.permutation(len(ib)))])}
    for key in ('core', 'core_inner'):
        cents = torch.stack([st['mnist'][key][y['mnist'] == d].mean(0) for d in range(10)])
        out['nearest_mnist_centroid'][key] = {}
        for m in others:
            pred = (F.normalize(st[m][key], dim=1) @ F.normalize(cents, dim=1).T).argmax(1).numpy()
            per = per_class(pred, y[m], CLASSES[m])
            out['nearest_mnist_centroid'][key][m] = {'per_class': {str(k): v for k, v in per.items()}, 'groups': grouped(per, m)}
        if key == 'core':
            # Distance between matched cross-modal centroids relative to MNIST between-class distances.
            between = torch.cdist(cents, cents)[~torch.eye(10, dtype=bool)].mean()
            out['matched_centroid_distance_ratio'] = {
                m: {str(c): float((st[m][key][y[m] == c].mean(0) - cents[c]).norm() / between) for c in CLASSES[m] if c <= 9}
                for m in others}
    # Cross-modal relations from core inner codes against MNIST references (20 per digit).
    ref = {d: rng.choice(np.flatnonzero(y['mnist'] == d), REF, replace=False) for d in range(10)}
    heads = ((0, 'next'), (1, 'same')) if succession else ((1, 'same'),)
    for m in others:
        za = st[m]['core_inner']
        for head, rel in heads:
            P = torch.stack([torch.sigmoid(logits_rel(sri, za.repeat_interleave(REF, 0),
                                                      st['mnist']['core_inner'][ref[d]].repeat(len(za), 1), head)).reshape(len(za), REF).mean(1)
                             for d in range(10)], 1)
            pred = P.argmax(1).numpy()
            per = per_class(pred, y[m], [c for c in CLASSES[m] if (c + 1 if rel == 'next' else c) <= 9],
                            (lambda c: c + 1) if rel == 'next' else (lambda c: c))
            out['cross_relations'][f'{rel}/{m}->mnist'] = {'per_class': {str(k): v for k, v in per.items()}, 'groups': grouped(per, m)}
    # Composition on MNIST core inner codes, and translation gains.
    cents = torch.stack([st['mnist']['core_inner'][y['mnist'] == d].mean(0) for d in range(10)])
    if succession:
        comp = {}
        for k in (1, 2, 3):
            z = st['mnist']['core_inner']
            for _ in range(k):
                z = sri.S(z)
            ok = y['mnist'] + k <= 9
            pred = torch.cdist(z[torch.as_tensor(ok)], cents).argmin(1).numpy()
            comp[str(k)] = float((pred == y['mnist'][ok] + k).mean())
        out['mnist_composition'] = comp
    out['core_inner_norm'] = {m: float(st[m]['core_inner'].norm(dim=1).mean()) for m in y}
    out['input_inner_norm'] = {m: float(st[m]['input_inner'].norm(dim=1).mean()) for m in y}
    return out


# ---------------------------------------------------------------- main

def git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    if MODE == 'confirmation' and not CONTRACT.exists():
        raise SystemExit('confirmation requires the frozen contract docs/number-core-network-v1-contract.md')
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = prepare()
    prep = {'wall_seconds': time.perf_counter() - wall0, 'cpu_seconds': time.process_time() - cpu0}
    jobs = jobs_for_mode()
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(WORKERS, len(jobs))) as pool:
        done = []
        for item in pool.imap_unordered(fit, jobs):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(jobs)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    done.sort(key=lambda d: jobs.index(d[0]))
    fits_wall = time.perf_counter() - t
    res = {'spec': str(SPEC.relative_to(ROOT)), 'mode': MODE, 'status': 'pilot: pipeline check, not headline evidence'
           if PILOT else 'confirmation', 'data': data,
           'config': {'seeds': SEEDS, 'arms': ARMS, 'steps': STEPS, 'batch': BATCH, 'pairs': PAIRS, 'equivalents': FEW,
                      'lr': LR, 'widths': ncore.WIDTHS, 'lambda': ncore.LAMBDA, 'per_unit_coupling': ncore.TOPOLOGIES,
                      'label_loss': 'softmax for all modalities' if MODE == 'pilot' else
                      'softmax for MNIST; positive-only cosine pull for dice and Morse',
                      'labelled': ncore.LABELLED, 'eval_split': SPLIT, 'reference_per_class': REF},
           'runs': {'/'.join(map(str, j)): r for j, r, _ in done}}
    PREFIX.parent.mkdir(exist_ok=True)
    Path(f'{PREFIX}-results.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {'/'.join(map(str, j)): rt for j, _, rt in done}
    counts = {}
    for rt in worker.values():
        for k, n in rt['counts'].items():
            counts[k] = counts.get(k, 0) + n
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json',
        'spec_sha256': nc.sha256(SPEC), 'contract_sha256': nc.sha256(CONTRACT) if CONTRACT.exists() else None,
        'runner_sha256': nc.sha256(__file__), 'module_sha256': {'number_core': nc.sha256(ncore.__file__), 'number_concept': nc.sha256(nc.__file__)},
        'revision': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NCC_')},
        'seed_policy': f'named SeedSequence streams under ROOT_SEED {nc.ROOT_SEED} (morse, morse_scramble, label_codes, core_few, '
                       'core_train, core_eval); torch.manual_seed(seed) for front-end and SRI initialization',
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'workers': min(WORKERS, len(jobs)), 'threads_per_worker': 1,
        'dtype': 'float32', 'blas': 'single-threaded (OPENBLAS/OMP/MKL_NUM_THREADS=1, torch.set_num_threads(1))',
        'phases': {'data_preparation': prep, 'fits_wall_seconds': fits_wall},
        'worker_tasks': worker, 'worker_cpu_seconds_total': sum(r['train_cpu_seconds'] + r['eval_cpu_seconds'] for r in worker.values()),
        'counts_total': counts, 'data_bytes': data['data_bytes'],
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process',
        'deployment_timing': None, 'deployment_timing_reason': 'not measured in the pilot; planned for confirmation',
        'peak_workspace_bytes': None, 'peak_workspace_reason': 'torch CPU allocator peak is not exposed; see peak RSS'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
