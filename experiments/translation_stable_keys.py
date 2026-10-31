"""Stable key pathway for translation memory v1 (docs/translation-memory-v1-draft.md, "Stable key revision").

A separate key encoder is trained for symbol identity on background symbols, then fully frozen: no batch statistics, no
centring, no map; the key is the unit-norm output of a fixed network applied to the image. The host is not involved.

Symbols: the task's filtered pool is the test domain and never enters training or validation. Further Unicode blocks are
thinned against the pool and each other over all pairs of covering fonts (the same glyph drawn in a shared font is caught,
not only the same codepoint), then split by a fixed stream into validation and background.

Stages (argv[1]):
  select   seed 299: every architecture x key width, validation recognition at each training checkpoint, the selection rule
  confirm  the selected setting on seed 298 as well; the frozen gate on the test lexicons of seeds 299 and 298, with one
           stored rendering per word (the rendering the cue buffer holds) against independent evaluation renderings

Usage: NFI_DEVICE=cuda /data/venvs/sri-cuda/bin/python -m experiments.translation_stable_keys select
"""
import gzip
import io
import json
import resource
import sys
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from experiments import device as DEV
from experiments import durable as DUR
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import translation_memory as TM
from experiments import translation_store as TS

nc = X.nc
SMOKE = X.SMOKE
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'select'
BLOCKS = {'latin-extended': (0x100, 0x24F), 'ipa': (0x250, 0x2AF), 'greek': (0x370, 0x3FF), 'cyrillic': (0x400, 0x52F),
          'armenian': (0x531, 0x587), 'hebrew': (0x5D0, 0x5EA), 'arabic': (0x620, 0x64A), 'devanagari': (0x904, 0x939),
          'thai': (0xE01, 0xE30), 'georgian': (0x10A0, 0x10FF), 'canadian': (0x1401, 0x166C), 'math': (0x2200, 0x22FF),
          'technical': (0x2300, 0x23FF), 'box': (0x2500, 0x257F), 'block-elements': (0x2580, 0x259F), 'math-a': (0x27C0, 0x27EF),
          'arrows-b': (0x2900, 0x297F), 'math-b': (0x2980, 0x29FF), 'math-supplement': (0x2A00, 0x2AFF), 'symbols-arrows': (0x2B00, 0x2BFF)}
N_VAL = 200
R_BG, R_VAL_EVAL = (2, 2) if SMOKE else (32, 5)
CLASSES, VIEWS = 64, 2
LR = 1e-3
TEMPERATURE = 0.1
ARCHS = {'base': (16, 128), 'wide': (32, 256)}  # convolution width, hidden width
KEY_WIDTHS = (32, 64)
CHECKPOINTS = (20, 40) if SMOKE else (1000, 2000, 4000, 8000)
VAL_DRAWS, VAL_WORDS = 5, 120
TOLERANCE = 0.01
GATE_WRITE, GATE_FINAL = 0.90, 0.80
SEL_SEED, CONF_SEED = 299, 298
NAME = 'translation-memory-v1-stable-keys' + ('-smoke' if SMOKE else '')
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME


# ---------------------------------------------------------------- symbols

def fonts_covering(cps):
    """{codepoint: every font index drawing it as a non-blank glyph different from the font's missing-glyph box}."""
    have = {cp: [] for cp in cps}
    for i, spec in enumerate(PS.FONT_LIST):
        missing = PS._mask(spec, '￿')
        for cp in cps:
            m = PS._mask(spec, chr(cp))
            if m.any() and not np.array_equal(m, missing):
                have[cp].append(i)
    return {cp: f for cp, f in have.items() if f}


def extra_candidates():
    out = []
    for lo, hi in BLOCKS.values():
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            if unicodedata.category(ch)[0] not in 'LSP' or cp in PS.EMOJI:
                continue
            if any(w in unicodedata.name(ch, 'DIGIT') + ' ' for w in PS.NAME_EXCLUDED):
                continue
            out.append(cp)
    return out


def symbol_splits():
    """{'pool': the test domain, 'validation': [...], 'background': [...], 'fonts': {cp: [its one font]}}, cached."""
    path = RUNS / 'symbols.json'
    if path.exists():
        z = json.loads(path.read_text())
        return {**z, 'fonts': {int(k): v for k, v in z['fonts'].items()}}
    pool = PS.survivors()
    all_fonts = fonts_covering(sorted(pool) + extra_candidates())
    vecs = np.concatenate([np.stack([PS.canonical(cp, [i]) for i in all_fonts[cp]]) for cp in pool])  # every kept glyph, every font
    kept = []
    for cp in sorted(all_fonts):
        if cp in pool:
            continue
        v = np.stack([PS.canonical(cp, [i]) for i in all_fonts[cp]])
        # for unit vectors, distance >= t is the same as cosine <= 1 - t^2 / 2
        if (v @ vecs.T).max() <= 1 - PS.DISTANCE_THRESHOLD ** 2 / 2:
            kept.append(cp)
            vecs = np.concatenate([vecs, v])
    perm = nc.stream('tk_split').permutation(kept)
    val, bg = [int(c) for c in perm[:N_VAL]], [int(c) for c in perm[N_VAL:]]
    if SMOKE:
        bg = bg[:100]
    fonts = {cp: [all_fonts[cp][int(nc.stream('pair_font', cp).integers(len(all_fonts[cp])))]] for cp in val + bg}
    fonts.update({int(cp): f for cp, f in pool.items()})
    out = {'pool': sorted(int(c) for c in pool), 'validation': val, 'background': bg, 'fonts': fonts, 'candidates': len(all_fonts) - len(pool)}
    DUR.atomic_write(path, json.dumps(out).encode())
    return out


def rendered(seed, syms):
    """Background (train split) and validation (one key rendering, independent evaluation renderings) images, cached."""
    path = RUNS / 'material' / f'seed{seed}.npz'
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    r = lambda cp, split, n: PS.renderings(cp, syms['fonts'][cp], seed, split, n)
    out = {'bg': np.stack([r(cp, 'train', R_BG) for cp in syms['background']]),
           'val_key': np.stack([r(cp, 'train', 1)[0] for cp in syms['validation']]),
           'val_eval': np.stack([r(cp, 'eval', R_VAL_EVAL) for cp in syms['validation']])}
    buf = io.BytesIO()
    np.savez_compressed(buf, **out)
    DUR.atomic_write(path, buf.getvalue())
    return out


# ---------------------------------------------------------------- the key encoder

class KeyEncoder(nn.Module):
    """A fixed function of the image once trained: two convolutions, two linear layers, unit-norm output; no running statistics."""

    def __init__(self, width, hidden, key):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, width, 3, 2, 1), nn.Conv2d(width, 2 * width, 3, 2, 1)
        self.fc, self.out = nn.Linear(2 * width * 49, hidden), nn.Linear(hidden, key)

    def forward(self, x):
        h = F.relu(self.c2(F.relu(self.c1(x[:, None]))))
        return F.normalize(self.out(F.relu(self.fc(h.flatten(1)))), dim=1)


def build(arch, key, seed):
    with torch.device('cpu'):
        torch.manual_seed(int(nc.stream('tk_init', seed, arch, key).integers(2 ** 62)))
        model = KeyEncoder(*ARCHS[arch], key)
    return model.to(DEV.NAME) if DEV.ACTIVE else model


def supcon(z, labels):
    n = len(z)
    self_mask = torch.eye(n, dtype=torch.bool)
    sim = (z @ z.T / TEMPERATURE).masked_fill(self_mask, float('-inf'))
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = (labels[:, None] == labels[None]) & ~self_mask
    return -(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1)).mean()


@torch.no_grad()
def encode(model, x):
    model.eval()
    x = np.ascontiguousarray(x.reshape(-1, 28, 28), dtype=np.float32)
    return torch.cat([model(torch.from_numpy(x[i:i + 512])) for i in range(0, len(x), 512)])


def recognition(keys, queries):
    """keys (n, d), queries (n, r, d): the nearest key's identity per word; ties go to the lower row, deterministically."""
    near = (queries.reshape(-1, queries.shape[-1]) @ keys.T).argmax(1)
    return (near == torch.arange(len(keys), device=near.device).repeat_interleave(queries.shape[1])).float().reshape(len(keys), -1).mean(1)


def growth(keys, queries, sizes):
    """Recognition of the first m words among the first m keys, for each candidate-set size m."""
    return {str(m): float(recognition(keys[:m], queries[:m]).mean()) for m in sizes}


def validation_reading(model, mat, seed):
    """Mean over seeded draws of VAL_WORDS validation symbols: recognition among those candidates, and the 20-to-120 growth."""
    return reading(encode(model, mat['val_key']), encode(model, mat['val_eval']).reshape(N_VAL, mat['val_eval'].shape[1], -1), seed)


def reading(keys, queries, seed):
    rng = nc.stream('tk_val_draws', seed)
    rec, grow = [], []
    for _ in range(VAL_DRAWS):
        pick = torch.from_numpy(rng.permutation(N_VAL)[:VAL_WORDS]).to(keys.device)
        k, q = keys[pick], queries[pick]
        rec.append(float(recognition(k, q).mean()))
        grow.append(growth(k, q, range(20, VAL_WORDS + 1, 20)))
    return {'recognition': float(np.mean(rec)), 'draws': rec, 'growth': {m: float(np.mean([g[m] for g in grow])) for m in grow[0]}}


def train(arch, key, seed, syms):
    """One fit, resumable as a whole: validation reading and a saved state at every checkpoint."""
    path = RUNS / 'fits' / f'{arch}-key{key}-seed{seed}.pt'
    if path.exists():
        return torch.load(path, weights_only=False)
    t0 = time.perf_counter()
    mat = rendered(seed, syms)
    render_s = time.perf_counter() - t0
    model = build(arch, key, seed)
    opt = torch.optim.Adam(model.parameters(), LR)
    rng = nc.stream('tk_batches', seed, arch, key)
    bg = mat['bg']
    n_cls, n_r = bg.shape[:2]
    ckpts, losses, t1 = {}, [], time.perf_counter()
    for step in range(1, max(CHECKPOINTS) + 1):
        cls = rng.choice(n_cls, CLASSES, replace=False)
        view = np.stack([rng.choice(n_r, VIEWS, replace=False) for _ in cls])
        x = bg[cls[:, None], view].reshape(-1, 28, 28)
        model.train()
        loss = supcon(model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))), torch.from_numpy(np.repeat(np.arange(CLASSES), VIEWS)))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
        if step in CHECKPOINTS:
            ckpts[step] = {'state': DEV.to_cpu(model.state_dict()), 'validation': validation_reading(model, mat, seed), 'loss_recent': float(np.mean(losses[-200:]))}
            ckpts[step]['state'] = {k: v.clone() for k, v in ckpts[step]['state'].items()}
    out = {'arch': arch, 'key': key, 'seed': seed, 'checkpoints': ckpts, 'parameters': sum(p.numel() for p in model.parameters()),
           'train_seconds': time.perf_counter() - t1, 'render_seconds': render_s}
    bio = io.BytesIO()
    torch.save(out, bio)
    DUR.atomic_write(path, bio.getvalue())
    return out


def untrained_and_pixels(mat_key, mat_eval, seed, read):
    """Cheaper alternatives on the same images: raw pixels (cosine) and the untrained base encoder."""
    flat = lambda x: F.normalize(torch.from_numpy(np.ascontiguousarray(x.reshape(-1, 784), dtype=np.float32)), dim=1)
    px = read(flat(mat_key), flat(mat_eval).reshape(len(mat_key), mat_eval.shape[1], -1))
    m = build('base', 32, seed)
    un = read(encode(m, mat_key), encode(m, mat_eval).reshape(len(mat_key), mat_eval.shape[1], -1))
    return {'pixels': px, 'untrained_base': un}


# ---------------------------------------------------------------- stages

def save_json(path, obj):
    DUR.atomic_write(path, (json.dumps(obj, indent=1, sort_keys=True, default=float) + '\n').encode())


def select():
    t0 = time.perf_counter()
    syms = symbol_splits()
    symbols_s = time.perf_counter() - t0
    rows, timing = [], {}
    for arch in ARCHS:
        for key in KEY_WIDTHS:
            fit = train(arch, key, SEL_SEED, syms)
            timing[f'{arch}-key{key}'] = {'train_seconds': fit['train_seconds'], 'render_seconds': fit['render_seconds']}
            for step, c in fit['checkpoints'].items():
                rows.append({'arch': arch, 'key': key, 'steps': step, 'parameters': fit['parameters'], 'bytes': fit['parameters'] * 4,
                             'validation': c['validation'], 'loss_recent': c['loss_recent']})
            print(arch, key, {s: round(c['validation']['recognition'], 3) for s, c in fit['checkpoints'].items()}, flush=True)
    best = max(r['validation']['recognition'] for r in rows)
    within = [r for r in rows if r['validation']['recognition'] >= best - TOLERANCE]
    chosen = min(within, key=lambda r: (r['bytes'], r['steps']))
    mat = rendered(SEL_SEED, syms)
    alt = untrained_and_pixels(mat['val_key'], mat['val_eval'], SEL_SEED, lambda k, q: reading(k, q, SEL_SEED))
    out = {'status': 'selection on validation symbols only; test symbols untouched',
           'symbols': {'candidates_outside_pool': syms['candidates'], 'pool': len(syms['pool']), 'validation': len(syms['validation']), 'background': len(syms['background'])},
           'grid': rows, 'rule': f'largest validation recognition; within {TOLERANCE}: fewest bytes, then fewest steps',
           'chosen': {k: chosen[k] for k in ('arch', 'key', 'steps', 'bytes')}, 'chosen_validation': chosen['validation'], 'alternatives_validation': alt}
    save_json(f'{REPORT}-selection.json', out)
    save_json(f'{REPORT}-selection-runtime.json', {'scope': 'nondeterministic timing', 'wall_seconds': time.perf_counter() - t0, 'symbol_filter_seconds': symbols_s,
                                                   'fits': timing, **DEV.info(), 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                   'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))})
    print(json.dumps({'chosen': out['chosen'], 'validation': chosen['validation']['recognition'], 'growth': chosen['validation']['growth'],
                      'pixels': alt['pixels']['recognition'], 'untrained': alt['untrained_base']['recognition'], 'symbols': out['symbols']}))


def stored_renderings(seed):
    """The rendering the cue buffer holds for each word (the host stage's first presentation)."""
    if SMOKE:
        return [0] * TM.N_WORDS
    with gzip.open(X.ROOT / 'reports' / f'{TM.NAME}-results.json.gz', 'rt') as f:
        res = json.load(f)['results']
    per_arm = {arm: {e['word']: e['render'] for e in res[f'{arm}/seed{seed}']['branches']['main']['events'] if e['kind'] == 'insert'} for arm in TM.ARMS}
    first = per_arm[TM.ARMS[0]]
    assert all(v == first for v in per_arm.values()), 'buffered renderings differ across arms'
    return [first[w] for w in range(TM.N_WORDS)]


def confirm():
    t0 = time.perf_counter()
    syms = symbol_splits()
    sel = json.loads(open(f'{REPORT}-selection.json').read())
    c = sel['chosen']
    out = {'chosen': c, 'gate': {'write_point': GATE_WRITE, 'final': GATE_FINAL}, 'seeds': {}}
    timing = {}
    for seed in (SEL_SEED, CONF_SEED):
        fit = train(c['arch'], c['key'], seed, syms)
        timing[str(seed)] = {'train_seconds': fit['train_seconds'], 'render_seconds': fit['render_seconds']}
        state = fit['checkpoints'][c['steps']]['state']
        mat = TM.material(seed)
        assert not set(mat['src']) & (set(syms['validation']) | set(syms['background'])), 'test symbols in training or validation'
        assert set(mat['src']) <= set(syms['pool'])
        stored = stored_renderings(seed)
        key_imgs = np.stack([mat['src_train'][w, stored[w]] for w in range(TM.N_WORDS)])
        query_imgs = mat['src_eval']
        reads, digests = [], []
        for point in range(1, TM.BATCHES + 2):  # the fixed-candidate test: reloaded from the saved state at every checkpoint
            model = build(c['arch'], c['key'], seed)
            model.load_state_dict(state)
            for p in model.parameters():
                p.requires_grad_(False)
            k, q = encode(model, key_imgs), encode(model, query_imgs).reshape(TM.N_WORDS, TM.R_EVAL, -1)
            acc = recognition(k, q)
            reads.append({'point': point, 'recognition': float(acc.mean()), 'by_batch': [float(acc[TM.WORDS * b:TM.WORDS * (b + 1)].mean()) for b in range(TM.BATCHES)]})
            digests.append(TS.digest(k, q))
        fixed = reads[0]['recognition']
        pix = untrained_and_pixels(key_imgs, query_imgs, seed, lambda a, b: {'recognition': float(recognition(a, b).mean())})
        out['seeds'][str(seed)] = {'fixed_candidates': reads, 'identical_at_every_checkpoint': len(set(digests)) == 1, 'recognition_120': fixed,
                                   'growth_teaching_order': growth(k, q, range(TM.WORDS, TM.N_WORDS + 1, TM.WORDS)),
                                   'validation': fit['checkpoints'][c['steps']]['validation'], 'alternatives': pix}
    pooled = float(np.mean([out['seeds'][str(s)]['recognition_120'] for s in (SEL_SEED, CONF_SEED)]))
    out['pooled_recognition_120'] = pooled
    out['gate_passes'] = pooled >= GATE_WRITE and pooled >= GATE_FINAL
    out['persistent_bytes'] = c['bytes']
    save_json(f'{REPORT}-confirmation.json', out)
    save_json(f'{REPORT}-confirmation-runtime.json', {'scope': 'nondeterministic timing', 'wall_seconds': time.perf_counter() - t0, 'fits': timing, **DEV.info(),
                                                      'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                                      'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))})
    print(json.dumps({k: out[k] for k in ('pooled_recognition_120', 'gate_passes', 'persistent_bytes')}))
    for s in (SEL_SEED, CONF_SEED):
        r = out['seeds'][str(s)]
        print(s, round(r['recognition_120'], 3), r['identical_at_every_checkpoint'], r['growth_teaching_order'], {k: round(v['recognition'], 3) for k, v in r['alternatives'].items()})


def check(seed):
    """The unchanged perception gate on one further seed: its own encoder at the selected setting, its own lexicon."""
    t0 = time.perf_counter()
    syms = symbol_splits()
    c = json.loads(open(f'{REPORT}-selection.json').read())['chosen']
    fit = train(c['arch'], c['key'], seed, syms)
    mat = TM.material(seed)
    assert not set(mat['src']) & (set(syms['validation']) | set(syms['background'])) and set(mat['src']) <= set(syms['pool'])
    stored = stored_renderings(seed)
    model = build(c['arch'], c['key'], seed)
    model.load_state_dict(fit['checkpoints'][c['steps']]['state'])
    k = encode(model, np.stack([mat['src_train'][w, stored[w]] for w in range(TM.N_WORDS)]))
    q = encode(model, mat['src_eval']).reshape(TM.N_WORDS, TM.R_EVAL, -1)
    acc = recognition(k, q)
    out = {'seed': seed, 'chosen': c, 'recognition_120': float(acc.mean()), 'gate_passes': float(acc.mean()) >= GATE_WRITE,
           'by_batch': [float(acc[TM.WORDS * b:TM.WORDS * (b + 1)].mean()) for b in range(TM.BATCHES)],
           'growth_teaching_order': growth(k, q, range(TM.WORDS, TM.N_WORDS + 1, TM.WORDS)),
           'validation': fit['checkpoints'][c['steps']]['validation']['recognition']}
    save_json(f'{REPORT}-check-seed{seed}.json', out)
    save_json(f'{REPORT}-check-seed{seed}-runtime.json', {'scope': 'nondeterministic timing', 'wall_seconds': time.perf_counter() - t0, 'train_seconds': fit['train_seconds'],
                                                         'render_seconds': fit['render_seconds'], **DEV.info(), 'revision': X.git('rev-parse', 'HEAD')})
    print(json.dumps({k_: out[k_] for k_ in ('seed', 'recognition_120', 'gate_passes', 'validation', 'growth_teaching_order')}))


if __name__ == '__main__':
    torch.set_num_threads(4)
    DEV.install()
    if STAGE == 'check':
        check(int(sys.argv[2]))
    else:
        {'select': select, 'confirm': confirm}[STAGE]()
