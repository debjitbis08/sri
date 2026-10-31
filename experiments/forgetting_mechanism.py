"""Forgetting mechanism v1 (docs/forgetting-mechanism-v1-contract.md): which changes make an acquired association inaccessible?

Per arm and seed (one durable task): blocks 1-3 and block 4 (teach set A) exactly as in arbitrary pairings, Option A (RigL and
resets have stopped; only gradients, Adam state and, for `mixed-h125`, explicit decay change weights). The end-of-block-4 state is
frozen and every branch starts from it for NAP_L updates:
  noupdate        nothing changes (evaluation only)
  zero-grad       opt.step() with explicit zero gradients (inherited Adam moments only), no decay
  decay-only      (mixed-h125) explicit decay each update, no gradient steps
  ctx             context episodes, everything trainable, decay as the arm's recipe
  ctx-nodecay     (mixed-h125) ctx with decay disabled
  ctx-enc-frozen  ctx with both encoders frozen
  ctx-read-frozen ctx with every non-encoder parameter of the model frozen (decay disabled)
  hold            context + A pair items every update, as in block 4
  rehearse-10     context + A pair items on every 10th update
then, from the end of ctx, three relearning continuations (original A, A with reassigned counts, never-seen B).

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.forgetting_mechanism pilot pairs
(NSY_SMOKE=1 for a short CPU run; NAP_ARMS and NAP_SEEDS override the defaults below.)
"""
import os

SMOKE = os.environ.get('NSY_SMOKE') == '1'
for key, value in (('NAP_OPTION', 'A'), ('NAP_ARMS', 'sparse48,dense,mixed,mixed-h125'),
                   ('NAP_SEEDS', ','.join(map(str, range(330, 340))))) + (() if SMOKE else (('NAP_N', '45'), ('NAP_L', '2000'))):
    os.environ.setdefault(key, value)

import copy
import resource
import time

import numpy as np
import torch

from experiments import arbitrary_pairings as AP
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import sequence_pilot as SP

nc = X.nc
CURRENT = LM.CURRENT
L, N, EVAL_RENDERS = AP.L, AP.N, AP.EVAL_RENDERS
RELEARN = 250 if not SMOKE else 10
REHEARSE_EVERY = 10
PROBE_LAMBDA = 1.0
NAME = 'forgetting-mechanism-v1' + ('-smoke' if SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME


def points(length, base):
    return sorted({p for p in base if p <= length} | {length})


POINTS = points(L, (0, 10, 25, 50, 100, 150, 200, 250, *range(500, L + 1, 250)))
RELEARN_POINTS = points(RELEARN, (0, 25, 50, 100))
SWAP_POINTS = [p for p in (50, 250, L) if p in POINTS]

SPECS = {  # train, zero_grad, decay_only, freeze, decay (explicit decay active), pairs
    'noupdate': dict(decay=False),
    'zero-grad': dict(zero_grad=True, decay=False),
    'decay-only': dict(decay_only=True, decay=True),
    'ctx': dict(train=True, decay=True),
    'ctx-nodecay': dict(train=True, decay=False),
    'ctx-enc-frozen': dict(train=True, freeze='enc', decay=True),
    'ctx-read-frozen': dict(train=True, freeze='read', decay=False),
    'hold': dict(train=True, decay=True, pairs='hold'),
    'rehearse-10': dict(train=True, decay=True, pairs='rehearse'),
}
RELEARN_ON = True  # the momentum-reset study switches this off
DECAY_ONLY_ARMS = ('mixed-h125',)  # branches that exist only where explicit decay exists
DECAY_BRANCHES = ('decay-only', 'ctx-nodecay')


def group_of(name):
    if name.startswith(('proj.', 'token', 'scale')):
        return 'projections'
    if '.0.struct' in name:
        return 'first_struct'
    if '.0.mem' in name:
        return 'first_mem'
    if '.0.' in name:
        return 'first_dense'
    return 'later_layers'


def apply_adam(opt, mode):
    """'m0': zero `exp_avg` of every initialized parameter state, keeping `exp_avg_sq` and `step`, and check bitwise that parameters,
    second moments and step counters are unchanged and the first moments are exactly zero. 'fresh': clear the whole state."""
    params = [p for g in opt.param_groups for p in g['params']]
    if mode == 'fresh':
        opt.state.clear()
        return {'mode': mode, 'n_params': len(params)}
    kept = {p: (p.detach().clone(), opt.state[p]['exp_avg_sq'].clone(), opt.state[p]['step'].clone()) for p in params if opt.state.get(p)}
    for p in kept:
        opt.state[p]['exp_avg'].zero_()
    ok = all(torch.equal(p.detach(), a) and torch.equal(opt.state[p]['exp_avg_sq'], b) and torch.equal(opt.state[p]['step'], c)
             and not bool(opt.state[p]['exp_avg'].any()) for p, (a, b, c) in kept.items())
    return {'mode': mode, 'n_initialized': len(kept), 'n_uninitialized': len(params) - len(kept), 'unchanged_and_zeroed': bool(ok)}


def named_params(enc, model):
    out = {f'image_encoder/{n}': p for n, p in enc['image'].named_parameters()}
    out.update({f'sequence_encoder/{n}': p for n, p in enc['sequence'].named_parameters()})
    out.update({f'model/{n}': p for n, p in model.named_parameters()})
    return out


def part_of(key):
    return key.split('/')[0] if not key.startswith('model/') else group_of(key[len('model/'):])


def movement(params, start):
    """Relative L2 change from the end of block 4 per group."""
    num, den = {}, {}
    for k, p in params.items():
        g = part_of(k)
        num[g] = num.get(g, 0.) + float(((p.detach() - start[k]) ** 2).sum())
        den[g] = den.get(g, 0.) + float((start[k] ** 2).sum())
    return {g: float(np.sqrt(num[g] / den[g])) for g in num}


@torch.no_grad()
def decoders(enc, model, A, row, ev, bank4):
    """1-NN symbol identity and count accuracy with a rebuilt bank (current code) and with the bank frozen at the end of
    block 4, and a ridge linear probe from the current code to the count (fit on the training renderings)."""
    code = G.codes_fn(enc, model)
    cps = list(A)
    R = AP.TRAIN_RENDERS
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, R, 28, 28)
    bank = code(np.concatenate([tr[row[cp]] for cp in cps]), 'image').double()
    q = code(ev.reshape(-1, 28, 28), 'image').double()
    ident, qid = np.repeat(np.arange(len(cps)), R), np.repeat(np.arange(len(cps)), ev.shape[1])
    counts = np.array([A[cp] for cp in cps])
    out = {}
    for tag, b in (('rebuilt', bank), ('frozen_bank', bank4)):
        nn_ = torch.cdist(q, b).argmin(1).numpy()
        out[f'{tag}/identity'] = float((ident[nn_] == qid).mean())
        out[f'{tag}/count'] = float((counts[ident[nn_]] == counts[qid]).mean())
    mu, sd = bank.mean(0), bank.std(0) + 1e-8
    Xtr, Xte = (bank - mu) / sd, (q - mu) / sd
    Y = torch.nn.functional.one_hot(torch.from_numpy(counts[ident] - 1), 9).double()
    ym = Y.mean(0)
    W = torch.linalg.solve(Xtr.T @ Xtr + PROBE_LAMBDA * torch.eye(Xtr.shape[1], dtype=torch.double), Xtr.T @ (Y - ym))
    pred = ((Xte @ W + ym).argmax(1) + 1).numpy()
    out['probe/count'] = float((pred == counts[qid]).mean())
    return out


def measure(enc, model, ctx, params=None):
    A, ev, ev_train, seed, row, bank4, start = (ctx[k] for k in ('A', 'ev', 'ev_train', 'seed', 'row', 'bank4', 'start'))
    out = {'recall': AP.recall(enc, model, A, ev, seed, 'A')[0], 'recall_train': AP.recall(enc, model, A, ev_train, seed, 'A')[0],
           **decoders(enc, model, A, row, ev, bank4)}
    if params is not None:
        out['movement'] = movement(params, start)
    return out


def run_branch(name, b4, arm, seed, ctx, ref, withheld, spec):
    """One branch from the end of block 4; returns the record and the final frozen state."""
    t0 = time.perf_counter()
    enc, model, opt, rng = AP.thaw(b4, arm, seed)
    if not spec.get('decay'):
        CURRENT['r'] = None
    params = named_params(enc, model)
    ctx = dict(ctx, start={k: p.detach().clone() for k, p in params.items()})
    frozen = {'enc': [k for k in params if not k.startswith('model/')], 'read': [k for k in params if k.startswith('model/')]}.get(spec.get('freeze'), [])
    for k in frozen:
        params[k].requires_grad_(False)
    crng, prng = nc.stream('ap_context', seed, 5), nc.stream('ap_pairs', seed, 5, 'A-' + name)
    curve, losses = [], []
    adam_check = apply_adam(opt, spec['adam']) if spec.get('adam') else None

    def record(k):
        rec = {'offset': k, **measure(enc, model, ctx, params)}
        if name == 'ctx' and k in SWAP_POINTS:
            rec['swap/current_enc_b4_readers'] = AP.recall(enc, ref[1], ctx['A'], ctx['ev'], seed, 'A')[0]
            rec['swap/b4_enc_current_readers'] = AP.recall(ref[0], model, ctx['A'], ctx['ev'], seed, 'A')[0]
        curve.append(rec)

    record(0)
    for step in range(L):
        if spec.get('train'):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            if spec.get('pairs') == 'hold' or (spec.get('pairs') == 'rehearse' and (step + 1) % REHEARSE_EVERY == 0):
                AP.pair_items(items, match, ctx['A'], ctx['row'], prng)
            losses.append(LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True))
        elif spec.get('zero_grad'):
            for p in opt.param_groups[0]['params']:
                p.grad = torch.zeros_like(p)
            opt.step()
            opt.zero_grad()
        elif spec.get('decay_only'):
            LM.decay_step(model, CURRENT['r'])
        if step + 1 in POINTS:
            record(step + 1)
    rec = {'curve': curve, 'movement_final': movement(params, ctx['start']),
           'frozen_unchanged': all(bool(torch.equal(params[k].detach(), ctx['start'][k])) for k in frozen),
           'n_frozen_tensors': len(frozen)}
    if adam_check is not None:
        rec['adam_check'] = adam_check
    if losses:
        rec['losses'] = losses
    if name != 'noupdate':
        rec['snapshot'] = LM.snapshot(enc, model, withheld, seed, name, False)
    rec['wall_seconds'] = time.perf_counter() - t0
    return rec, AP.freeze(enc, model, opt, rng)


def relearn(end, arm, seed, sets, row, ev):
    """From the end of ctx: 250 updates of context plus pairs of the original A, A with reassigned counts, and B."""
    out = {}
    for name, taught, e, tag in (('original', sets['A'], ev['A'], 'A'), ('reassigned', sets["A'"], ev['A'], "A'"), ('B', sets['B'], ev['B'], 'B')):
        enc, model, opt, rng = AP.thaw(end, arm, seed)
        crng, prng = nc.stream('ap_context', seed, 6), nc.stream('ap_pairs', seed, 6, name)
        curve = [{'offset': 0, 'recall': AP.recall(enc, model, taught, e, seed, tag)[0]}]
        for step in range(RELEARN):
            items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
            AP.pair_items(items, match, taught, row, prng)
            LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True)
            if step + 1 in RELEARN_POINTS:
                curve.append({'offset': step + 1, 'recall': AP.recall(enc, model, taught, e, seed, tag)[0]})
        out[name] = curve
    return out


def run_task(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    withheld = D.withheld_counts(seed)
    sets, row, ev = AP.build_sets(seed)
    A = sets['A']
    enc, model, opt = LM.setup(arm, seed)
    CURRENT['end'] = None  # Option A: the inherited schedule, RigL and resets stop at update 4,500
    rng = nc.stream('partA_train', seed, 'A-order')
    out = {'withheld': {k: list(v) for k, v in withheld.items()}, 'sets': {k: {str(cp): n for cp, n in v.items()} for k, v in sets.items()},
           'N': N, 'L': L, 'points': POINTS, 'relearn_points': RELEARN_POINTS, 'branches': {}}
    for b, lang in enumerate(P.LANGS):  # blocks 1-3: the limited-memory curriculum, no rehearsal
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    curve4 = []
    AP.block(enc, model, opt, rng, seed, 4, L, A, row, {'A': (A, ev['A'])}, curve4, 'A')
    out['block4'] = {'curve': curve4, 'snapshot': LM.snapshot(enc, model, withheld, seed, 'b4', False),
                     'pairs': AP.checks(enc, model, A, row, ev['A'], seed, 'A')}
    b4 = AP.freeze(enc, model, opt, rng)
    ref_enc, ref_model, _, _ = AP.thaw(b4, arm, seed)  # the end-of-block-4 state, kept for the component swap and the frozen bank
    code4 = G.codes_fn(ref_enc, ref_model)
    R = AP.TRAIN_RENDERS
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, R, 28, 28)
    with torch.no_grad():
        bank4 = code4(np.concatenate([tr[row[cp]] for cp in A]), 'image').double()
    ctx = {'A': A, 'ev': ev['A'], 'ev_train': np.stack([tr[row[cp]][:EVAL_RENDERS] for cp in A]), 'seed': seed, 'row': row, 'bank4': bank4}
    end = None
    for name, spec in SPECS.items():
        if name in DECAY_BRANCHES and arm not in DECAY_ONLY_ARMS:
            continue
        out['branches'][name], state = run_branch(name, b4, arm, seed, ctx, (ref_enc, ref_model), withheld, spec)
        if name == 'ctx':
            end = state
    if RELEARN_ON:
        out['relearn'] = relearn(end, arm, seed, sets, row, ev)
    return f'{arm}/seed{seed}', out, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


if __name__ == '__main__':
    AP.NAME, AP.RUNS, AP.run_task = NAME, RUNS, run_task
    AP.main()
