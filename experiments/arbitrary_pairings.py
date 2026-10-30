"""Arbitrary pairings v1 (docs/arbitrary-pairings-v1-contract.md): can the memory part learn pairs that no structure predicts?

Per arm and seed (one durable task): blocks 1-3 of the limited-memory curriculum (no rehearsal), the parent at the end of
block 3, then branches that start from it:
  P-AB  block 4 teaches set A (random Unicode symbols, each paired with a count), block 5 teaches set B
  P-A   the same block 4 (shared), then block 5 continues with no pairs      (decay and drift without interference)
  C     blocks 4-5 continue with no pairs                                    (matched control for the damage of pairing)
  P-A'  (sparse48 and mixed only) block 4 with a second assignment of counts to A's symbols (learning check)
A continuation trains on the limited-memory delay's material (context episodes, no pairs). A pair block adds NAP_BATCH
pair items to that material: a symbol rendering, a collection of its assigned count, and the matching loss.

Schedule option (NAP_OPTION): B (the study) lets RigL and the mixed-reset resets run to the end of block 5, with RigL's
cosine fraction annealing to zero there; A (the debugging baseline) keeps the inherited end at update 0.75 * 3 * block.
Pair-set size NAP_N (a multiple of nine) and pairing block length NAP_L come from the seed-299 pilot.

Usage: NAP_OPTION=A NAP_N=18 NAP_L=1000 NFI_DEVICE=cuda NSY_WORKERS=6 NAP_SEEDS=299 \
       /data/venvs/sri-cuda/bin/python -m experiments.arbitrary_pairings pilot pairs
"""
import copy
import os
import resource
import time

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import generic_relations as G
from experiments import sequence_pilot as SP
from experiments import transition_pilot as TP
from experiments import fan_in_pilot as FI
from experiments import device as DEV
from experiments import durable as DUR
from experiments import limited_memory as LM
from experiments import pairing_symbols as PS

nc, ns = X.nc, X.ns
CURRENT = LM.CURRENT
OPTION = os.environ.get('NAP_OPTION', 'B')
N = int(os.environ.get('NAP_N', '18' if not X.SMOKE else '9'))
L = int(os.environ.get('NAP_L', '2000' if not X.SMOKE else '20'))
BATCH = int(os.environ.get('NAP_BATCH', '16'))
EVERY = 250 if not X.SMOKE else 10
EARLY = (25, 50, 100, 150, 200) if not X.SMOKE else ()  # finer points early in a block: the pilot's curves were flat from the first point at 250
TRAIN_RENDERS, EVAL_RENDERS = int(os.environ.get("NAP_TRAIN_RENDERS", "64")), 5
ARMS = tuple(os.environ.get('NAP_ARMS', 'sparse48,dense,mixed,mixed-h125,mixed-h500,mixed-reset').split(','))
SEEDS = tuple(int(s) for s in os.environ.get('NAP_SEEDS', ','.join(map(str, range(320, 330)))).split(','))
CHECK_ARMS = ('sparse48', 'mixed')  # arms that also train block 4 on the second assignment A'
NAME = f'arbitrary-pairings-v1-pairs-{OPTION}-N{N}-L{L}' + ('-smoke' if X.SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME
POOL = {}


# ---------------------------------------------------------------- pair material

def build_sets(seed):
    """Symbol sets for a seed (identical across arms), their training renderings (registered as a pseudo-language so the
    existing image path serves them) and evaluation renderings."""
    if not POOL:
        POOL.update(PS.survivors())
    sets = PS.draw_sets(seed, N, POOL)
    cps = list(sets['A']) + list(sets['B'])
    train = np.stack([PS.renderings(cp, POOL[cp], seed, 'train', TRAIN_RENDERS) for cp in cps])  # (2N, R, 28, 28)
    ev = {name: np.stack([PS.renderings(cp, POOL[cp], seed, 'eval', EVAL_RENDERS) for cp in sets[name]]) for name in ('A', 'B')}
    X.DATA[('pairsym', 'train')] = (train.reshape(-1, 28, 28), np.zeros(len(cps) * TRAIN_RENDERS, int))
    row = {cp: k for k, cp in enumerate(cps)}
    return sets, row, ev


def pair_items(items, match, taught, row, rng):
    """Append BATCH pair items (symbol rendering, collection of its count, matching) to a batch's material."""
    cps = list(taught)
    for _ in range(BATCH):
        cp = cps[int(rng.integers(len(cps)))]
        n = taught[cp]
        f = len(items); items.append(('form', 'pairsym', n, row[cp] * TRAIN_RENDERS + int(rng.integers(TRAIN_RENDERS))))
        c = len(items); items.append(('coll', None, n, int(rng.integers(len(P.DOTS['train'])))))
        match.append((f, c, n))


@torch.no_grad()
def recall(enc, model, taught, ev, seed, tag):
    """Matching accuracy over the taught symbols' evaluation renderings (nine candidates in seeded random positions, the same
    in every arm and at every evaluation); returns the accuracy, the picks and the candidate positions' seed tag."""
    enc['image'].eval(); enc['sequence'].eval()
    code = G.codes_fn(enc, model)
    cps = list(taught)
    n, E = ev.shape[:2]
    q = code(ev.reshape(-1, 28, 28), 'image')
    rng = nc.stream('ap_eval_candidates', seed, tag)
    order = np.stack([rng.permutation(9) for _ in range(n * E)])
    j = rng.integers(P.EVAL_PER, size=n * E)
    imgs = np.stack([P.DOTS['succ'][int(k) + 1][j[t]] for t in range(n * E) for k in order[t]])
    cc = code(imgs, 'image').reshape(n * E, 9, -1)
    pick = order[np.arange(n * E), G.pair(model, q, cc, 1).argmax(1).numpy()] + 1
    truth = np.repeat([taught[cp] for cp in cps], E)
    enc['image'].train(); enc['sequence'].train()
    return float((pick == truth).mean()), pick


@torch.no_grad()
def lookup_recall(enc, model, taught, row, ev, seed):
    """Reference decoders on the item code, training renderings of the taught symbols against the evaluation ones: 1-nearest-neighbor
    count accuracy (`count`) and 1-nearest-neighbor symbol-identity accuracy (`identity`)."""
    code = G.codes_fn(enc, model)
    cps = list(taught)
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, TRAIN_RENDERS, 28, 28)
    bank = code(np.concatenate([tr[row[cp]] for cp in cps]), 'image').double()
    ident = np.repeat(np.arange(len(cps)), TRAIN_RENDERS)
    q = code(ev.reshape(-1, 28, 28), 'image').double()
    nn_ = torch.cdist(q, bank).argmin(1).numpy()
    counts = np.array([taught[cp] for cp in cps])
    return {'count': float((counts[ident[nn_]] == np.repeat(counts, ev.shape[1])).mean()),
            'identity': float((ident[nn_] == np.repeat(np.arange(len(cps)), ev.shape[1])).mean())}


def checks(enc, model, taught, row, ev, seed, tag):
    acc, pick = recall(enc, model, taught, ev, seed, tag)
    counts = np.array([taught[cp] for cp in taught])
    truth = np.repeat(counts, ev.shape[1])
    # rescoring the same picks against independently permuted labels: the mean and spread over 100 permutations
    prng = nc.stream('ap_rescore', seed, tag)
    resc = [float((pick == np.repeat(counts[prng.permutation(len(counts))], ev.shape[1])).mean()) for _ in range(100)]
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, TRAIN_RENDERS, 28, 28)
    ev_train = np.stack([tr[row[cp]][:ev.shape[1]] for cp in taught])  # the exact images seen in training
    out = {'recall': acc, 'recall_on_training_renderings': recall(enc, model, taught, ev_train, seed, tag)[0],
           'rescored_shuffled_mean': float(np.mean(resc)), 'rescored_shuffled_sd': float(np.std(resc)),
           'lookup': lookup_recall(enc, model, taught, row, ev, seed)}
    for name, parts in (('mem_silenced', ('mem',)), ('struct_silenced', ('struct',)), ('both_silenced', ('mem', 'struct'))):
        with LM.silenced(model, parts):
            out[name] = recall(enc, model, taught, ev, seed, tag)[0]
    return out


# ---------------------------------------------------------------- state and blocks

def freeze(enc, model, opt, rng):
    return copy.deepcopy({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': model.state_dict(),
                          'opt': opt.state_dict(), 'rng': rng, 'step': CURRENT['step']})


def thaw(state, arm, seed):
    CURRENT.update(arm=arm, seed=seed, step=state['step'], r=2. ** (-1. / LM.HALF_LIFE[arm]) if arm in LM.HALF_LIFE else None)
    torch.manual_seed(seed)
    enc, model = X.make_model(seed)
    s = copy.deepcopy(state)
    model.load_state_dict(s['sri']); enc['image'].load_state_dict(s['image']); enc['sequence'].load_state_dict(s['sequence'])
    opt = torch.optim.Adam([*enc['image'].parameters(), *enc['sequence'].parameters(), *model.parameters()], P.LR)
    opt.load_state_dict(s['opt'])
    return enc, model, opt, s['rng']


def block(enc, model, opt, rng, seed, number, steps, taught, row, watch, curve, set_name=None):
    """steps updates of the continuation (context episodes) plus pair items when `taught` is given; `watch` maps a set name to
    (taught, evaluation renderings) and is recorded into `curve` every EVERY updates."""
    crng = nc.stream('ap_context', seed, number)  # the same context episodes in every branch
    prng = nc.stream('ap_pairs', seed, number, set_name or 'none')
    t0 = CURRENT['step']
    curve.append({'update': t0, 'offset': 0, 'block': number, **{k: recall(enc, model, t, ev, seed, k)[0] for k, (t, ev) in watch.items()}})
    for step in range(steps):
        items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
        if taught is not None:
            pair_items(items, match, taught, row, prng)
        LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True)
        if (step + 1) % EVERY == 0 or (step + 1) in EARLY:
            curve.append({'update': CURRENT['step'], 'offset': step + 1, 'block': number,
                          **{k: recall(enc, model, t, ev, seed, k)[0] for k, (t, ev) in watch.items()}})


def run_task(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    withheld = D.withheld_counts(seed)
    sets, row, ev = build_sets(seed)
    A, B, A2 = sets['A'], sets['B'], sets["A'"]
    enc, model, opt = LM.setup(arm, seed)
    CURRENT['end'] = (3 * D.STEPS + 2 * L) if OPTION == 'B' else None
    rng = nc.stream('partA_train', seed, 'A-order')
    out = {'withheld': {k: list(v) for k, v in withheld.items()}, 'sets': {k: {str(cp): n for cp, n in v.items()} for k, v in sets.items()},
           'option': OPTION, 'N': N, 'L': L, 'branches': {}}
    for b, lang in enumerate(P.LANGS):  # blocks 1-3: the limited-memory curriculum, no rehearsal
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    out['parent'] = LM.snapshot(enc, model, withheld, seed, 'block3', False)
    parent = freeze(enc, model, opt, rng)
    watchA, watchB = {'A': (A, ev['A'])}, {'A': (A, ev['A']), 'B': (B, ev['B'])}

    def end_of(name, e, m, taught_sets):
        rec = {'snapshot': LM.snapshot(e, m, withheld, seed, name, False)}
        rec['pairs'] = {k: checks(e, m, t, row, evs, seed, k) for k, (t, evs) in taught_sets.items()}
        return rec

    enc, model, opt, rng = thaw(parent, arm, seed)  # P-A, then P-AB and P-A from the end of block 4
    curve4 = []
    block(enc, model, opt, rng, seed, 4, L, A, row, watchA, curve4, 'A')
    b4 = freeze(enc, model, opt, rng)
    out['branches']['block4_A'] = {'curve': curve4, **end_of('b4', enc, model, watchA)}
    for name, taught, w in (('P-AB', B, watchB), ('P-A', None, watchA)):
        enc, model, opt, rng = thaw(b4, arm, seed)
        curve = []
        block(enc, model, opt, rng, seed, 5, L, taught, row, w, curve, 'B')
        out['branches'][name] = {'curve': curve, **end_of(name, enc, model, w)}
    enc, model, opt, rng = thaw(parent, arm, seed)  # C: no pairs in blocks 4-5
    curve = []
    block(enc, model, opt, rng, seed, 4, L, None, row, watchA, curve)
    block(enc, model, opt, rng, seed, 5, L, None, row, watchA, curve)
    out['branches']['C'] = {'curve': curve, **end_of('C', enc, model, {})}
    if arm in CHECK_ARMS:  # learning check: the same symbols with a second assignment of counts
        enc, model, opt, rng = thaw(parent, arm, seed)
        curve = []
        watch2 = {"A'": (A2, ev['A'])}
        block(enc, model, opt, rng, seed, 4, L, A2, row, watch2, curve, "A'")
        out['branches']["P-A'"] = {'curve': curve, **end_of("A'", enc, model, watch2)}
    return f'{arm}/seed{seed}', out, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def run(task):
    torch.set_num_threads(FI.THREADS)
    DEV.install()
    key, res, rt = run_task(task)
    if DEV.ACTIVE:
        rt['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    return key, res, rt


def main():
    LM.install()
    D.ARMS = {a: (False, 'taps') for a in ARMS}
    D.SEEDS = SEEDS
    D.NAME = NAME
    D.PREFIX, D.RUNS = X.ROOT / 'reports' / NAME, RUNS
    D.run = DUR.durable(run, RUNS / 'done', '{0}-seed{1}')
    D.main()
    path = X.ROOT / 'reports' / f'{NAME}-runtime.json'
    rt = __import__('json').loads(path.read_text())
    rt.update({'threads_per_worker': FI.THREADS, 'option': OPTION, 'N': N, 'L': L, **DEV.info()})
    path.write_text(__import__('json').dumps(rt, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
