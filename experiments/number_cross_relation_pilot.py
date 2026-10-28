"""Exploratory pilot: does succession extend sparse cross-modal number equivalents?

Not a frozen contract; see docs/number-cross-relation-pilot.md. One shared encoder sees
MNIST digits and standard dice faces (v1 generator). A judge answers two true/false
questions about an ordered image pair (A, B):

  same  - do A and B show the same number?  Trained on digit-digit and dice-dice pairs, and
          on cross-modal pairs among a fixed set of equivalents: 10 images per modality for
          each paired count (identical images in every arm of a seed).
  next  - is B's number A's number + 1?     Trained only within a modality, and only in
          arms with succession on.

Judges: `free` (MLP on the concatenated pair, one logit per question) and `structured`
(same = closeness of codes, next = closeness of B to S(A) for one shared operator S).
Arms: free/structured x succession on/off with pairs {1, 3, 5}; structured on/off with
pairs {1, 2, 3}. Primary measure: cross-modal accuracy on unpaired dice counts.

Writes reports/number-cross-relation-pilot-results.json (deterministic) and -runtime.json.
NCR_SMOKE=1 runs a few steps for pipeline checks; NCR_WORKERS sets the pool size.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import number_concept as nc  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

SMOKE = os.environ.get('NCR_SMOKE') == '1'
OUT = ROOT / ('reports/number-cross-relation-pilot' + ('-smoke' if SMOKE else ''))
SEEDS = (0, 1, 2)
PAIR_SETS = {'135': (1, 3, 5), '123': (1, 2, 3)}
ARMS = {  # arm: (judge, succession, pair set)
    'free-succ-135': ('free', True, '135'), 'free-nosucc-135': ('free', False, '135'),
    'struct-succ-135': ('structured', True, '135'), 'struct-nosucc-135': ('structured', False, '135'),
    'struct-succ-123': ('structured', True, '123'), 'struct-nosucc-123': ('structured', False, '123'),
}
GROUPS = {'135': {'interpolation': (2, 4), 'extrapolation': (6,), 'paired': (1, 3, 5)},
          '123': {'extrapolation': (4, 5, 6), 'paired': (1, 2, 3)}}
FEW, STEPS, PAIRS, LR, Z = 10, 20 if SMOKE else 3000, 64, 1e-3, 16  # PAIRS per task per step, half positive
REF, CENTROID_PER_DIGIT = 20, 1000  # reference images per class for image-level questions; train centroids
WORKERS = int(os.environ.get('NCR_WORKERS', str(max(1, (os.cpu_count() or 1) - 1))))
CLASSES = {'digit': nc.DIGITS, 'dice': nc.FACES}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 16, 3, 2, 1), nn.Conv2d(16, 32, 3, 2, 1)
        self.fc, self.out = nn.Linear(32 * 49, 128), nn.Linear(128, Z)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return self.out(F.relu(self.fc(h.flatten(1))))


class FreeJudge(nn.Module):
    """MLP on the ordered pair; logits for (next, same). No tie between the two answers."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * Z, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, za, zb):
        return self.net(torch.cat([za, zb], 1))


class StructuredJudge(nn.Module):
    """same: closeness of A and B; next: closeness of B to S(A), with S(z) = z + g(z) shared by both modalities."""
    def __init__(self):
        super().__init__()
        self.g = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, Z))
        self.bias, self.scale = nn.Parameter(torch.ones(2)), nn.Parameter(torch.zeros(2))

    def S(self, z):
        return z + self.g(z)

    def forward(self, za, zb):
        d = torch.stack([(zb - self.S(za)).pow(2).sum(1), (zb - za).pow(2).sum(1)], 1)
        return self.bias - F.softplus(self.scale) * d


def by_class(labels):
    return {int(c): np.flatnonzero(labels == c) for c in np.unique(labels)}


def pairs(rng, idx_a, idx_b, relation, n, classes_a, classes_b):
    """n index pairs (half positive) for `relation` between two class->index maps."""
    rel = (lambda a, b: b == a + 1) if relation == 'next' else (lambda a, b: b == a)
    pos = [(a, b) for a in classes_a for b in classes_b if rel(a, b)]
    neg = [(a, b) for a in classes_a for b in classes_b if not rel(a, b)]
    cells = [pos[i] for i in rng.integers(0, len(pos), n // 2)] + [neg[i] for i in rng.integers(0, len(neg), n - n // 2)]
    ia = np.array([rng.choice(idx_a[a]) for a, _ in cells])
    ib = np.array([rng.choice(idx_b[b]) for _, b in cells])
    return ia, ib, np.array([1.] * (n // 2) + [0.] * (n - n // 2), np.float32)


def few_set(seed, idx, K):
    """Fixed equivalents: keyed by (seed, modality, count) only, so arms of a seed share them."""
    return {m: {k: nc.stream('cross_few', seed, m, k).choice(idx[m][k], FEW, replace=False) for k in K} for m in idx}


def fit(job):
    seed, arm = job
    judge_kind, succession, pset = ARMS[arm]
    K = PAIR_SETS[pset]
    t0, c0 = time.perf_counter(), time.process_time()
    torch.manual_seed(seed)
    rng = nc.stream('cross_relation', seed, arm)
    enc, judge = Encoder(), (FreeJudge() if judge_kind == 'free' else StructuredJudge())
    opt = torch.optim.Adam([*enc.parameters(), *judge.parameters()], LR)
    imgs = {'digit': DATA['digit_train'], 'dice': DATA['dice_train']}
    idx = {m: by_class(DATA[f'{m}_train_labels']) for m in imgs}
    few = few_set(seed, idx, K)
    tasks = [(m, m, rel, idx[m], idx[m], CLASSES[m], CLASSES[m]) for m in imgs
             for rel in (('next', 'same') if succession else ('same',))]
    tasks += [(ma, mb, 'same', few[ma], few[mb], K, K) for ma, mb in (('digit', 'dice'), ('dice', 'digit'))]
    losses = []
    for step in range(STEPS):
        xa, xb, ys, head = [], [], [], []
        for ma, mb, rel, ia_map, ib_map, ca, cb in tasks:
            ia, ib, y = pairs(rng, ia_map, ib_map, rel, PAIRS, ca, cb)
            xa.append(imgs[ma][ia]); xb.append(imgs[mb][ib]); ys.append(y); head.append(0 if rel == 'next' else 1)
        logits = judge(enc(torch.from_numpy(np.concatenate(xa))), enc(torch.from_numpy(np.concatenate(xb))))
        logits = logits.reshape(len(tasks), PAIRS, 2)[torch.arange(len(tasks)), :, torch.as_tensor(head)]
        # Sum of per-task means: every task keeps the same weight whether or not succession tasks are present.
        loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(np.stack(ys)), reduction='none').mean(1).sum()
        if not torch.isfinite(loss):
            raise nc.NumericalFailure(f'{seed}/{arm} step {step}')
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 99 or step == STEPS - 1:
            losses.append(round(loss.item(), 5))
    train_seconds = time.perf_counter() - t0
    result = evaluate(enc, judge, seed, judge_kind, succession, pset)
    result['loss_every_100'] = losses
    return (seed, arm), result, {'train_wall_seconds': train_seconds, 'wall_seconds': time.perf_counter() - t0,
                                 'cpu_seconds': time.process_time() - c0, 'steps': STEPS,
                                 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def nearest(z, cents):
    return (F.normalize(z, dim=1) @ F.normalize(cents, dim=1).T).argmax(1).numpy()


def grouped(per_count, pset):
    """Means of a {dice count: accuracy} dict over the pair set's count groups."""
    return {g: float(np.mean([per_count[c] for c in cs if c in per_count])) for g, cs in GROUPS[pset].items()
            if any(c in per_count for c in cs)}


@torch.no_grad()
def evaluate(enc, judge, seed, judge_kind, succession, pset):
    rng = nc.stream('cross_eval', seed)  # same reference images for every arm of a seed
    enc.eval()
    z = {m: enc(torch.from_numpy(DATA[f'{m}_test'])) for m in CLASSES}
    lab = {m: DATA[f'{m}_test_labels'] for m in CLASSES}
    idx = {m: by_class(lab[m]) for m in CLASSES}
    ref = {m: {c: rng.choice(idx[m][c], REF, replace=False) for c in CLASSES[m]} for m in CLASSES}
    tr_idx = by_class(DATA['digit_train_labels'])
    cents = torch.stack([enc(torch.from_numpy(DATA['digit_train'][rng.choice(tr_idx[d], CENTROID_PER_DIGIT, replace=False)])).mean(0)
                         for d in nc.DIGITS])
    heads = ('next', 'same') if succession else ('same',)
    out = {'class_matrices': {}, 'image_level': {}, 'class_offsets': {}}

    def probs(za, mb):
        """Mean judge probabilities of each query code against each class of modality mb's references: (n, classes, 2)."""
        cols = []
        for c in CLASSES[mb]:
            zr = z[mb][ref[mb][c]]
            p = torch.sigmoid(judge(za.repeat_interleave(REF, 0), zr.repeat(len(za), 1)))
            cols.append(p.reshape(len(za), REF, 2).mean(1))
        return torch.stack(cols, 1).numpy()

    for ma in CLASSES:
        for mb in CLASSES:
            # Image-level: every dice test image, or the first 300 test images of each digit class that has a
            # relative among the targets (digits 0-6 against dice), against mb's references.
            if ma == 'dice':
                rows = np.arange(len(lab[ma]))
            else:
                rows = np.sort(np.concatenate([idx[ma][c][:300] for c in (range(7) if mb == 'dice' else nc.DIGITS)]))
            P, y = probs(z[ma][torch.as_tensor(rows)], mb), lab[ma][rows]
            cols = np.array(CLASSES[mb])
            for h, rel in enumerate(('next', 'same')):
                if rel not in heads:
                    continue
                target = y + 1 if rel == 'next' else y
                pred = cols[P[..., h].argmax(1)]
                valid = np.isin(target, cols)
                # Key accuracy by the dice count involved (source dice, or target dice for digit->dice next).
                dice_count = target if (ma == 'digit' and mb == 'dice') else y
                per = {int(c): float((pred[valid & (dice_count == c)] == target[valid & (dice_count == c)]).mean())
                       for c in np.unique(dice_count[valid]) if (ma == 'dice' or mb == 'dice') and c in nc.FACES}
                key = f'{rel}/{ma}->{mb}'
                if ma == mb:
                    out['image_level'][key] = {'accuracy': float((pred[valid] == target[valid]).mean())}
                else:
                    out['image_level'][key] = {'per_dice_count': {str(c): v for c, v in per.items()}, 'groups': grouped(per, pset)}
                M = np.stack([P[y == a, :, h].mean(0) for a in np.unique(y)])
                out['class_matrices'][key] = np.round(M, 4).tolist()
                if ma == 'dice' and mb == 'digit':
                    tgt = (lambda a: a + 1) if rel == 'next' else (lambda a: a)
                    out['class_offsets'][key] = {str(a): int(cols[int(np.argmax(M[i]))] - tgt(a)) for i, a in enumerate(np.unique(y))}
    # Representation-level readouts against digit training centroids (cosine).
    out['digit_centroid_accuracy'] = float((nearest(z['digit'], cents) == lab['digit']).mean())
    pd = nearest(z['dice'], cents)
    per = {int(c): float((pd[lab['dice'] == c] == c).mean()) for c in nc.FACES}
    out['dice_nearest_digit_centroid'] = {'per_count': {str(c): v for c, v in per.items()}, 'groups': grouped(per, pset)}
    if judge_kind == 'structured' and succession:
        comp = {}
        for k in (1, 2, 3):
            zk = z['digit']
            for _ in range(k):
                zk = judge.S(zk)
            ok = lab['digit'] + k <= 9
            comp[str(k)] = float((nearest(zk[torch.as_tensor(ok)], cents) == lab['digit'][ok] + k).mean())
        out['digit_composition'] = comp
        ps = nearest(judge.S(z['dice']), cents)
        per = {int(c): float((ps[lab['dice'] == c] == c + 1).mean()) for c in nc.FACES}
        out['dice_successor_to_digit_centroid'] = {'per_dice_count': {str(c): v for c, v in per.items()}, 'groups': grouped(per, pset)}
        out['relation_scales'] = {'bias': judge.bias.tolist(), 'softplus_scale': F.softplus(judge.scale).tolist()}
    return out


def load_data():
    raw, _ = nc.load_mnist(ROOT / 'data/mnist')
    train, _ = nc.mnist_split(raw['train_labels'])
    dtr, dte = nc.build_dice('standard', 'train'), nc.build_dice('standard', 'test')
    return {'digit_train': raw['train_images'][train].astype(np.float32) / 255, 'digit_train_labels': raw['train_labels'][train],
            'digit_test': raw['test_images'].astype(np.float32) / 255, 'digit_test_labels': raw['test_labels'],
            'dice_train': dtr['images'], 'dice_train_labels': dtr['labels'],
            'dice_test': dte['images'], 'dice_test_labels': dte['labels']}


def main():
    global DATA
    t0, c0 = time.perf_counter(), time.process_time()
    DATA = load_data()
    prep = time.perf_counter() - t0
    jobs = [(s, a) for a in ARMS for s in SEEDS]
    with mp.get_context('fork').Pool(WORKERS) as pool:
        done = pool.map(fit, jobs, chunksize=1)
    res = {'status': 'exploratory pilot; no frozen contract', 'doc': 'docs/number-cross-relation-pilot.md',
           'config': {'seeds': SEEDS, 'arms': {k: list(v) for k, v in ARMS.items()},
                      'pair_sets': {k: list(v) for k, v in PAIR_SETS.items()}, 'groups': GROUPS,
                      'equivalents_per_count_per_modality': FEW, 'steps': STEPS, 'pairs_per_task_per_step': PAIRS,
                      'lr': LR, 'latent_dim': Z, 'reference_images_per_class': REF,
                      'digit_test_images_per_class_for_pair_questions': 300},
           'runs': {f'{s}/{a}': r for (s, a), r, _ in done}}
    OUT.parent.mkdir(exist_ok=True)
    Path(f'{OUT}-results.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    Path(f'{OUT}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing; scientific results are in -results.json',
        'workers': WORKERS, 'data_preparation_seconds': prep, 'wall_seconds': time.perf_counter() - t0,
        'main_cpu_seconds': time.process_time() - c0, 'fits': {f'{s}/{a}': t for (s, a), _, t in done},
        'runner_sha256': nc.sha256(__file__), 'module_sha256': nc.sha256(nc.__file__),
        'torch': torch.__version__, 'numpy': np.__version__}, indent=1) + '\n')


if __name__ == '__main__':
    main()
