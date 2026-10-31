"""Translation memory v1, host stage (docs/translation-memory-v1-draft.md).

The host never reads the memory: it is frozen in the consolidation, gate and sentence phases, so its trajectory is the
same whatever memory is attached. This stage trains the host and saves its state at every consolidation point; the
memory stage (experiments/translation_memory_pass.py) replays those states for every memory variant and pilot setting.

Per arm and seed (one durable task): blocks 1-3 of the limited-memory curriculum (Option A schedule, no rehearsal), then
six batches of 20 words, each 2,000 acquisition updates of 4 context episodes and 16 word trials, then 2,000
context-only updates. A word trial: a source rendering scored against 32 target candidates (the update's distinct
targets and distractors from all 150 target words) by the matching head; cross-entropy on the true target.
Branches: `main` (the system's host) and `frozen-encoder` (image-to-code path frozen after batch 1's acquisition).

Saved per branch: the host state at the end of each batch's acquisition (consolidation points 1-6) and after the final
context-only block (point 7); the cue buffer's events (first presentation of each word inserts one rendering; first
in, first out at 40 words); per-word exposures and last presentation step.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 TM_SEEDS=299,298 /data/venvs/sri-cuda/bin/python -m experiments.translation_memory pilot host
(NSY_SMOKE=1 for a short CPU run; rerun the same command to resume.)
"""
import copy
import io
import os
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import arbitrary_pairings as AP
from experiments import durable as DUR
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import sequence_pilot as SP
from experiments import transition_pilot as TP
from experiments import translation_store as TS

nc = X.nc
SMOKE = X.SMOKE
CURRENT = LM.CURRENT
N_SRC = N_TGT = 150
BATCHES, WORDS = 6, 20
N_WORDS = BATCHES * WORDS
ANCHOR_SYMBOLS, ANCHOR_RENDERS = 32, 2
L_ACQ = 20 if SMOKE else 2000
L_FINAL = 20 if SMOKE else 2000
TRIALS, CANDIDATES = 16, 32
R_TRAIN, R_EVAL = (4, 2) if SMOKE else (64, 5)
CAPACITY = 40
CURVE = (0, 5, 10, 20) if SMOKE else (0, 25, 50, 100, 250, 500, 1000, 2000)
BRANCHES = ('main', 'frozen-encoder')
ARMS = tuple(os.environ.get('TM_ARMS', 'mixed,sparse48').split(','))
SEEDS = tuple(int(s) for s in os.environ.get('TM_SEEDS', ','.join(map(str, range(360, 370)))).split(','))
NAME = 'translation-memory-v1-host' + ('-smoke' if SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME
SPEC = X.ROOT / 'docs' / 'translation-memory-v1-draft.md'
STATE = {}  # the active word trials, read by the extra loss


# ---------------------------------------------------------------- material

def material(seed):
    """Alphabets, lexicon, batches and renderings for a seed (identical across arms and branches).

    Words are indexed 0..119 in teaching order (batch b holds words 20b..20b+19); word w has source codepoint src[w] and
    target index tgt[w] into the target alphabet T (150 symbols; the 30 never used remain distractors)."""
    if not AP.POOL:
        AP.POOL.update(PS.survivors())
    pool = AP.POOL
    rng = nc.stream('tm_alphabets', seed)
    perm = [int(c) for c in rng.permutation(sorted(pool))]
    S, T, rest = perm[:N_SRC], perm[N_SRC:N_SRC + N_TGT], perm[N_SRC + N_TGT:]
    assert len(rest) >= ANCHOR_SYMBOLS
    anchors = rest[:ANCHOR_SYMBOLS]
    lrng = nc.stream('tm_lexicon', seed)
    src = [S[int(i)] for i in lrng.permutation(N_SRC)[:N_WORDS]]
    tgt = [int(i) for i in lrng.permutation(N_TGT)[:N_WORDS]]
    path = RUNS / 'material' / f'seed{seed}.npz'
    if path.exists():
        z = np.load(path)
        imgs = {k: z[k] for k in z.files}
    else:
        r = lambda cp, split, n: PS.renderings(cp, pool[cp], seed, split, n)
        imgs = {'src_train': np.stack([r(cp, 'train', R_TRAIN) for cp in src]),
                'src_eval': np.stack([r(cp, 'eval', R_EVAL) for cp in src]),
                'tgt_train': np.stack([r(cp, 'train', R_TRAIN) for cp in T]),
                'anchor_imgs': np.stack([r(cp, 'train', ANCHOR_RENDERS) for cp in anchors])}
        buf = io.BytesIO()
        np.savez_compressed(buf, **imgs)
        DUR.atomic_write(path, buf.getvalue())
    return {'S': S, 'T': T, 'anchors': anchors, 'src': src, 'tgt': tgt, **imgs}


def batch_words(b):
    return list(range(WORDS * b, WORDS * (b + 1)))


# ---------------------------------------------------------------- word trials

def install_loss():
    if STATE.get('wrapped'):
        return
    base = TP.EXTRA_LOSS

    def extra(model, mat, c, cand):
        loss = base(model, mat, c, cand)
        s = STATE.get('trials')
        if s is None:
            return loss
        enc = STATE['enc']
        solver = P.Coupled(model)
        q = solver.solve(enc['image'](torch.from_numpy(s['src'])))[:, P.INNER]
        k = solver.solve(enc['image'](torch.from_numpy(s['cand'])))[:, P.INNER]
        logits = G.pair(model, q, k[None].expand(len(q), -1, -1), 1)
        l = F.cross_entropy(logits, torch.from_numpy(s['label']))
        STATE['losses'].append(float(l.detach()))
        return loss + l

    TP.EXTRA_LOSS = extra
    STATE['wrapped'] = True


def trials(mat, words, rng):
    """TRIALS word trials from `words`: source renderings, CANDIDATES target candidates (each trial's target among them),
    labels (candidate position) and the words drawn."""
    w = [words[int(i)] for i in rng.integers(len(words), size=TRIALS)]
    j = rng.integers(R_TRAIN, size=TRIALS)
    targets = list(dict.fromkeys(mat['tgt'][x] for x in w))
    others = [t for t in rng.permutation(N_TGT).tolist() if t not in targets][:CANDIDATES - len(targets)]
    cands = targets + others
    cj = rng.integers(R_TRAIN, size=CANDIDATES)
    return {'src': mat['src_train'][w, j], 'cand': mat['tgt_train'][cands, cj],
            'label': np.array([cands.index(mat['tgt'][x]) for x in w]), 'words': w, 'renders': j.tolist()}


@torch.no_grad()
def host_logits(enc, model, mat, x):
    """Host scores of renderings x (n, 28, 28) against every target word (its training rendering 0): (n, 150)."""
    enc['image'].eval()
    code = G.codes_fn(enc, model)
    q = code(x, 'image')
    k = code(mat['tgt_train'][:, 0], 'image')
    out = G.pair(model, q, k[None].expand(len(q), -1, -1), 1)
    enc['image'].train()
    return out


def host_accuracy(enc, model, mat, words):
    x = mat['src_eval'][words].reshape(-1, 28, 28)
    pred = host_logits(enc, model, mat, x).argmax(1).cpu().numpy()
    return float((pred == np.repeat([mat['tgt'][w] for w in words], R_EVAL)).mean())


# ---------------------------------------------------------------- host trajectory

def host_state(enc, model):
    return {'image': copy.deepcopy(enc['image'].state_dict()), 'sequence': copy.deepcopy(enc['sequence'].state_dict()),
            'sri': copy.deepcopy(model.state_dict())}


def path_params(enc, model):
    return [*enc['image'].parameters(), *model.proj['image'].parameters()]


class Buffer:
    """Cue buffer events: one entry per word, inserted at its first presentation in its batch; first in, first out."""

    def __init__(self, capacity):
        self.capacity, self.entries, self.events = capacity, [], []

    def present(self, word, render, step):
        if any(e['word'] == word for e in self.entries):
            return
        if len(self.entries) >= self.capacity:
            old = self.entries.pop(0)
            self.events.append({'kind': 'exit', 'word': old['word'], 'step': step})
        e = {'word': word, 'render': int(render), 'entry_step': step}
        self.entries.append(e)
        self.events.append({'kind': 'insert', **e})

    def snapshot(self):
        return [dict(e) for e in self.entries]


def acquisition(enc, model, opt, rng, mat, seed, b, buf, log, steps, words):
    """steps updates of context episodes plus (when words is not None) word trials from `words`."""
    crng = nc.stream('tm_context', seed, b)
    trng = nc.stream('tm_trials', seed, b)
    curve = []
    if words is not None and 0 in CURVE:
        curve.append({'offset': 0, 'acc': host_accuracy(enc, model, mat, words)})
    for step in range(steps):
        mat_ctx = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
        STATE['trials'] = None
        if words is not None:
            tr = trials(mat, words, trng)
            STATE['trials'] = tr
            for w, j in zip(tr['words'], tr['renders']):
                log['exposures'][w] += 1
                log['last_presented'][w] = CURRENT['step'] + 1
                buf.present(w, j, CURRENT['step'] + 1)
        LM.train_update(enc, model, opt, mat_ctx, rng, True)
        if words is not None and (step + 1) in CURVE:
            curve.append({'offset': step + 1, 'acc': host_accuracy(enc, model, mat, words)})
    STATE['trials'] = None
    return curve


def run_branch(name, start, arm, seed, mat, withheld):
    """From `start` (the parent at the end of block 3, or the end of batch 1 for frozen-encoder) to the end."""
    enc, model, opt, rng = AP.thaw(start['host'], arm, seed)
    STATE['enc'] = enc
    buf = copy.deepcopy(start['buf'])
    log = copy.deepcopy(start['log'])
    points = list(start['points'])
    first = start['next_batch']
    if name == 'frozen-encoder':
        for p in path_params(enc, model):
            p.requires_grad_(False)
        path0 = TS.digest(*path_params(enc, model))
    for b in range(first, BATCHES):
        STATE['losses'] = []
        curve = acquisition(enc, model, opt, rng, mat, seed, b, buf, log, L_ACQ, batch_words(b))
        points.append({'point': b + 1, 'step': CURRENT['step'], 'host': host_state(enc, model), 'buffer': buf.snapshot(),
                       'curve': curve, 'trial_loss_mean': float(np.mean(STATE['losses']))})
        if name == 'main' and b == 0:
            start_b1 = {'host': AP.freeze(enc, model, opt, rng), 'buf': copy.deepcopy(buf), 'log': copy.deepcopy(log),
                        'points': copy.deepcopy(points), 'next_batch': 1}
    acquisition(enc, model, opt, rng, mat, seed, 'final', buf, log, L_FINAL, None)
    points.append({'point': BATCHES + 1, 'step': CURRENT['step'], 'host': host_state(enc, model), 'buffer': buf.snapshot()})
    out = {'events': buf.events, 'exposures': log['exposures'], 'last_presented': log['last_presented'],
           'snapshot': LM.snapshot(enc, model, withheld, seed, name, False)}
    if name == 'frozen-encoder':
        out['path_unchanged'] = TS.digest(*path_params(enc, model)) == path0
    return out, points, (start_b1 if name == 'main' else None)


def run_task(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    install_loss()
    withheld = D.withheld_counts(seed)
    mat = material(seed)
    enc, model, opt = LM.setup(arm, seed)
    CURRENT['end'] = None  # Option A: the inherited schedules end at update 0.75 * 3 * block
    rng = nc.stream('partA_train', seed, 'A-order')
    for lang in P.LANGS:  # blocks 1-3
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    out = {'parent': LM.snapshot(enc, model, withheld, seed, 'block3', False), 'parent_step': CURRENT['step'],
           'lexicon': {'src': mat['src'], 'tgt': mat['tgt'], 'S': mat['S'], 'T': mat['T'], 'anchors': mat['anchors']},
           'branches': {}}
    start = {'host': AP.freeze(enc, model, opt, rng), 'buf': Buffer(CAPACITY),
             'log': {'exposures': [0] * N_WORDS, 'last_presented': [None] * N_WORDS}, 'points': [], 'next_batch': 0}
    states = {}
    rec, points, b1 = run_branch('main', start, arm, seed, mat, withheld)
    out['branches']['main'], states['main'] = rec, points
    rec, points, _ = run_branch('frozen-encoder', b1, arm, seed, mat, withheld)
    out['branches']['frozen-encoder'], states['frozen-encoder'] = rec, points
    for name in states:  # curves and steps go into the results; the host states into the checkpoint file
        out['branches'][name]['points'] = [{k: v for k, v in p.items() if k != 'host'} for p in states[name]]
    path = RUNS / 'states' / f'{arm}-seed{seed}.pt'
    bio = io.BytesIO()
    torch.save({name: [{'point': p['point'], 'step': p['step'], 'host': p['host']} for p in pts] for name, pts in states.items()}, bio)
    DUR.atomic_write(path, bio.getvalue())
    out['checkpoint'] = {'file': str(path.relative_to(X.ROOT)), 'sha256': nc.sha256(path)}
    return f'{arm}/seed{seed}', out, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    AP.NAME, AP.RUNS, AP.run_task = NAME, RUNS, run_task
    D.SPEC = SPEC
    LM.install()
    D.ARMS = {a: (False, 'taps') for a in ARMS}
    D.SEEDS = SEEDS
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, RUNS
    D.run = DUR.durable(AP.run, RUNS / 'done', '{0}-seed{1}')
    D.main()
    path = X.ROOT / 'reports' / f'{NAME}-runtime.json'
    rt = __import__('json').loads(path.read_text())
    rt.update({'threads_per_worker': AP.FI.THREADS, **AP.DEV.info()})
    path.write_text(__import__('json').dumps(rt, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
