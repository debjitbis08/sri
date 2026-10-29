"""Imagery probe on Part D checkpoints (docs/number-construction-v1-spec.md, "Imagery probe on Part D checkpoints").

A core layer is clamped to a chosen state and the energy is solved for everything else, including a sense's outer
layer, which perception normally clamps to the encoder's features. The solved features are read out by a decoder
trained after the fact on the frozen encoder, by nearest neighbours among held-out inputs, by independent counters,
and by the system's own reading of the decoded input.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_imagery pilot
Environment: NCI_RUNS (runs directory name, default number-construction-v1-part-d-pilot), NSY_WORKERS, NSY_SMOKE=1.
"""
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch
from torch import nn
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D

nc, ns = X.nc, X.ns
ROOT = X.ROOT
RUN_NAME = os.environ.get('NCI_RUNS', 'number-construction-v1-part-d-pilot')
RUN_DIR = ROOT / 'runs' / RUN_NAME
NAME = f'number-construction-v1-imagery-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX = ROOT / 'reports' / NAME
SIZES = P.SIZES
O, M, I = SIZES
STATE = sum(SIZES)
CORE_MIDDLE, CORE_INNER = slice(STATE + O, STATE + O + M), slice(STATE + O + M, 2 * STATE)
DEC_STEPS, DEC_BATCH = (50 if X.SMOKE else 3000), 128
N_RANDOM = 5
LANGS = P.LANGS
FIG_ARMS = ('D-base', 'D-varied+rehearsal')


class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(O, 256), nn.ReLU(), nn.Linear(256, 32 * 7 * 7), nn.ReLU())
        self.up1, self.up2 = nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ConvTranspose2d(16, 1, 4, 2, 1)

    def forward(self, f):
        h = self.fc(f).view(-1, 32, 7, 7)
        return torch.sigmoid(self.up2(F.relu(self.up1(h))))[:, 0]


class SequenceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(O, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, f):
        return self.net(f)


class Reader(nn.Module):
    """Count reader trained on decoder reconstructions of real collections (image) or taps (sequence)."""
    def __init__(self, medium):
        super().__init__()
        self.medium = medium
        if medium == 'image':
            self.net = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
                                     nn.Linear(32 * 49, 128), nn.ReLU(), nn.Linear(128, 9))
        else:
            self.net = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 9))

    def forward(self, x):
        return self.net(x[:, None] if self.medium == 'image' else x)


def reverse_solve(sri, sense, clamp, value):
    """Clamp the core layer `clamp` (a slice of the joint state) to `value` and solve every other state; returns the
    sense's outer-layer features."""
    Q = sri.hessian(sense)
    idx = torch.arange(3 * STATE)
    c = idx[clamp]
    free = idx[~torch.isin(idx, c)]
    y = -torch.linalg.solve(Q[free][:, free], Q[free][:, c] @ value.T).T
    return y[:, :O]  # the sense's outer layer comes first in the free state


def count_dots(x, threshold=.35):
    return int(ndimage.label(x > threshold)[1])


def train_decoder(dec, feats_fn, steps, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(dec.parameters(), 1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        f, x = feats_fn(rng)
        loss = F.mse_loss(dec(f), x)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def run(path):
    t0, c0 = time.perf_counter(), time.process_time()
    ck = torch.load(path, weights_only=False)
    arm, seed = ck['arm'], ck['seed']
    enc, sri = X.make_model(seed)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); sri.load_state_dict(ck['sri'])
    for p in [*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()]:
        p.requires_grad_(False)
    E = P.EVAL_PER
    sv = {m: D.B.Solver(sri, m) for m in ('image', 'sequence')}
    feat = lambda x, m: enc[m](torch.from_numpy(np.ascontiguousarray(x)))
    state = lambda x, m: sri.split(sv[m].solve(feat(x, m)))
    # held-out inputs and forward prototypes
    held = {'collections': {n: P.DOTS['succ'][n] for n in ns.NUMBERS}, 'taps': {n: D.TAPS[('eval', n)] for n in ns.NUMBERS},
            **{f'forms/{lang}': {n: X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]] for n in ns.NUMBERS} for lang in (*LANGS, P.TRANSFER)}}
    medium = lambda kind: 'sequence' if kind == 'taps' else 'image'
    with torch.no_grad():
        st = {k: {n: state(x, medium(k)) for n, x in v.items()} for k, v in held.items()}
    proto = {k: {n: s['core_inner'].mean(0) for n, s in v.items()} for k, v in st.items()}
    coll_mid = {n: st['collections'][n]['core'][:, O:O + M].mean(0) for n in ns.NUMBERS}
    A, t = sri.A.detach(), sri.t.detach()
    targets = {}  # name: {n: (clamp slice, state)}
    for k in ('collections', 'taps', *(f'forms/{lang}' for lang in LANGS)):
        targets[k] = {n: (CORE_INNER, proto[k][n]) for n in ns.NUMBERS}
    pc = proto['collections']
    targets['imagined_successor'] = {n: (CORE_INNER, pc[n - 1] @ A.T + t) for n in range(2, 10)}
    K = torch.eye(I) + A.T @ A
    targets['gap'] = {n: (CORE_INNER, torch.linalg.solve(K, pc[n - 1] @ A.T + t + (pc[n + 1] - t) @ A)) for n in range(2, 9)}
    targets['middle_collections'] = {n: (CORE_MIDDLE, coll_mid[n]) for n in ns.NUMBERS}
    g = torch.Generator().manual_seed(seed)
    targets['random'] = {n: (CORE_INNER, (lambda r: r / r.norm() * pc[n].norm())(torch.randn(I, generator=g))) for n in ns.NUMBERS}
    # decoders, trained on frozen features of training inputs the sense has met
    forms_train = [(lang, X.DATA[(lang, 'train')][0]) for lang in LANGS]

    def image_batch(rng):
        h = DEC_BATCH // 2
        e, k = rng.integers(len(P.DOTS['train']), size=h), rng.integers(9, size=h)
        li = rng.integers(len(forms_train), size=DEC_BATCH - h)
        x = np.concatenate([P.DOTS['train'][e, k], np.stack([forms_train[l][1][rng.integers(len(forms_train[l][1]))] for l in li])])
        with torch.no_grad():
            f = feat(x, 'image')
        return f, torch.from_numpy(x)

    def seq_batch(rng):
        x = D.TAPS['train'][rng.integers(len(D.TAPS['train']), size=DEC_BATCH)]
        with torch.no_grad():
            f = feat(x, 'sequence')
        return f, torch.from_numpy(x)
    dec = {'image': ImageDecoder(), 'sequence': SequenceDecoder()}
    dec_loss = {m: train_decoder(dec[m], fn, DEC_STEPS, seed) for m, fn in (('image', image_batch), ('sequence', seq_batch))}

    def reader_batch(m):
        def fn(rng):
            n = rng.integers(1, 10, size=DEC_BATCH)
            if m == 'image':
                x = P.DOTS['train'][rng.integers(len(P.DOTS['train']), size=DEC_BATCH), n - 1]
            else:
                x = D.TAPS['train'][[rng.choice(D.TAPS['train_idx'][k]) for k in n]]
            with torch.no_grad():
                rec = dec[m](feat(x, m))
            return rec, torch.from_numpy(n - 1)
        return fn
    reader = {m: Reader(m) for m in ('image', 'sequence')}
    for m in reader:
        torch.manual_seed(seed)
        opt = torch.optim.Adam(reader[m].parameters(), 1e-3)
        rng, fn = np.random.default_rng(seed + 1), reader_batch(m)
        for _ in range(DEC_STEPS // 2):
            rec, y = fn(rng)
            loss = F.cross_entropy(reader[m](rec), y)
            opt.zero_grad(); loss.backward(); opt.step()
    read = lambda x, m: reader[m](torch.from_numpy(np.ascontiguousarray(x))).argmax(1).numpy() + 1
    with torch.no_grad():
        fnorm = {m: float(fn(np.random.default_rng(1))[0].norm(dim=1).mean()) for m, fn in (('image', image_batch), ('sequence', seq_batch))}
        counter = {'image': count_dots, 'sequence': ns.count_pulses}
        reader_kind = {'image': 'collections', 'sequence': 'taps'}
        protos_c = torch.stack([pc[n] for n in ns.NUMBERS])

        def own_reading(x, m):
            c = state(x, m)['core_inner']
            lg = P.same_logits(sri, c[:, None].expand(-1, 9, -1).reshape(-1, I), protos_c[None].expand(len(c), -1, -1).reshape(-1, I)).reshape(len(c), 9)
            return lg.argmax(1).numpy() + 1
        # validation: decoder reconstructions of real held-out inputs
        valid = {}
        for m in ('image', 'sequence'):
            k = reader_kind[m]
            xs = np.concatenate([held[k][n] for n in ns.NUMBERS]); ys = np.repeat(ns.NUMBERS, E)
            rec = dec[m](feat(xs, m)).numpy()
            valid[m] = {'counter_on_real': float(np.mean([counter[m](x) == y for x, y in zip(xs, ys)])),
                        'counter_on_reconstruction': float(np.mean([counter[m](x) == y for x, y in zip(rec, ys)])),
                        'own_reading_on_reconstruction': float((own_reading(rec, m) == ys).mean()),
                        'reader_on_reconstruction': float((read(rec, m) == ys).mean()),
                        'reconstruction_mse': float(((rec - xs) ** 2).mean())}
        # gallery for nearest neighbours (cosine on features of each sense)
        gal = {}
        for m in ('image', 'sequence'):
            kinds = [k for k in held if medium(k) == m]
            fs, lab, xs = [], [], []
            for k in kinds:
                for n in ns.NUMBERS:
                    fs.append(feat(held[k][n], m)); lab += [(k, n)] * len(held[k][n]); xs.append(held[k][n])
            gal[m] = (F.normalize(torch.cat(fs), dim=1), lab, np.concatenate(xs))
        out, images = {}, {}
        for name, per_n in targets.items():
            for m in ('image', 'sequence'):
                rows = {'counter_raw': [], 'counter_scaled': [], 'reader_raw': [], 'reader_scaled': [], 'own_reading_scaled': [],
                        'nn_top1_count': [], 'nn_top5_count_majority': [], 'nn_top1_kind': [], 'feature_norm_ratio': []}
                readings = {}
                for n, (clamp, v) in per_n.items():
                    fh = reverse_solve(sri, m, clamp, v[None])
                    scaled = fh / fh.norm(dim=1, keepdim=True) * fnorm[m]
                    xr, xs_ = dec[m](fh).numpy()[0], dec[m](scaled).numpy()[0]
                    images[(name, m, n)] = xs_
                    rows['counter_raw'].append(counter[m](xr) == n); rows['counter_scaled'].append(counter[m](xs_) == n)
                    rr, rs = read(np.stack([xr, xs_]), m)
                    rows['reader_raw'].append(rr == n); rows['reader_scaled'].append(rs == n); readings[str(n)] = int(rs)
                    rows['own_reading_scaled'].append(int(own_reading(xs_[None], m)[0]) == n)
                    sims = (gal[m][0] @ F.normalize(fh, dim=1)[0]).numpy()
                    top = np.argsort(-sims)[:5]
                    labs = [gal[m][1][j] for j in top]
                    images[(name, m, n, 'nn')] = gal[m][2][top[0]]
                    rows['nn_top1_count'].append(labs[0][1] == n)
                    counts = [l[1] for l in labs]
                    rows['nn_top5_count_majority'].append(max(set(counts), key=counts.count) == n)
                    rows['nn_top1_kind'].append(labs[0][0])
                    rows['feature_norm_ratio'].append(float(fh.norm() / fnorm[m]))
                kinds = rows.pop('nn_top1_kind')
                res = {k: float(np.mean(v)) for k, v in rows.items()}
                res['nn_top1_kind'] = {k: kinds.count(k) / len(kinds) for k in sorted(set(kinds))}
                res['counter_scaled_by_count'] = {str(n): int(counter[m](images[(name, m, n)])) for n in per_n}
                res['reader_scaled_by_count'] = readings
                out[f'{name}/{m}'] = res
    fig = None
    if arm in FIG_ARMS:
        fig = figures(images, targets, f'{arm}-seed{seed}')
    return f'{arm}/seed{seed}', {'validation': valid, 'decoder_final_mse': dec_loss, 'mean_feature_norm': fnorm, 'probe': out, 'figures': fig,
                                 'checkpoint_sha256': nc.sha256(path)}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def figures(images, targets, tag):
    """Grids: one row per clamped target, one column per count; images as pixels, taps as traces."""
    names = [f'{t} {v}' for t in targets for v in ('decoded', 'nearest')]
    S, pad, lab = 56, 4, 190
    img = Image.new('L', (lab + 9 * (S + pad), len(names) * (S + pad) + 20), 255)
    seq = Image.new('L', (lab + 9 * (120 + pad), len(names) * (40 + pad) + 20), 255)
    di, ds = ImageDraw.Draw(img), ImageDraw.Draw(seq)
    for col in range(9):
        di.text((lab + col * (S + pad) + S // 2 - 3, 4), str(col + 1), fill=0)
        ds.text((lab + col * (120 + pad) + 57, 4), str(col + 1), fill=0)
    for r, label in enumerate(names):
        name, view = label.rsplit(' ', 1)
        key = (lambda m, n: (name, m, n)) if view == 'decoded' else (lambda m, n: (name, m, n, 'nn'))
        y0, y1 = 20 + r * (S + pad), 20 + r * (40 + pad)
        di.text((4, y0 + S // 2 - 6), label, fill=0); ds.text((4, y1 + 14), label, fill=0)
        for n in targets[name]:
            x = images[key('image', n)]
            tile = Image.fromarray((255 - np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)
            img.paste(tile, (lab + (n - 1) * (S + pad), y0))
            s = images[key('sequence', n)]
            x0 = lab + (n - 1) * (120 + pad)
            pts = [(x0 + j * 120 / 128, y1 + 36 - 32 * float(np.clip(v, -.1, 1.1))) for j, v in enumerate(s)]
            ds.rectangle([x0, y1, x0 + 120, y1 + 38], outline=200)
            ds.line(pts, fill=0)
    paths = {}
    for kind, im in (('image', img), ('taps', seq)):
        p = ROOT / 'reports' / f'{NAME}-{tag}-{kind}.png'
        im.save(p, optimize=True)
        paths[kind] = str(p.relative_to(ROOT))
    return paths


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    P.prepare(); D.prepare()
    phases = {'prepare_wall_seconds': time.perf_counter() - wall0}
    paths = sorted(RUN_DIR.glob('*.pt'))
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(paths))) as pool:
        done = pool.map(run, paths)
    phases['probe_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(P.SPEC.relative_to(ROOT)), 'status': 'imagery probe on Part D checkpoints (pilot)',
           'config': {'runs': str(RUN_DIR.relative_to(ROOT)), 'decoder_steps': DEC_STEPS, 'decoder_batch': DEC_BATCH, 'random_states_per_count': 1,
                      'targets': ['collections', 'taps', *(f'forms/{l}' for l in LANGS), 'imagined_successor', 'gap', 'middle_collections', 'random']},
           'results': {k: r for k, r, _ in sorted(done)}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'part_d_sha256': nc.sha256(D.__file__), 'spec_sha256': nc.sha256(P.SPEC),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'phases': phases, 'tasks': worker,
        'data_bytes': data['data_bytes'], 'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
