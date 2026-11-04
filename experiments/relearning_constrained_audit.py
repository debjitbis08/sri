"""Feasibility audit of the constrained update proposed in docs/relearning-loss-analysis-v1.md (section 4), at the relearning transition.
(docs/relearning-constrained-audit-v1.md)

From the captured pre-relearning state per seed (experiments.relearning_branches), for each of the first 20 relearning updates:
  delta0   Adam's actual proposed parameter displacement from all existing losses (clipped gradient, standard moment update), over every trainable
           parameter (image projection head, taps encoder, core)
  S, E     on the batch's context episodes, which pair a collection with taps of the same count at every step:
             S = mean_i ||c_i - mean(c)||^2 over the context collections' codes      (collection spread)
             E = mean_i ||c_i - t_i||^2 over the same-step collection/taps pairs      (cross-sense mismatch)
  a, b     grad log S, grad E
  delta    the smallest Euclidean correction of delta0 with a.delta = 0 and b.delta <= 0:
             delta1 = delta0 - a (a.delta0)/(a.a); n = b - a (a.b)/(a.a)
             delta = delta1 if b.delta1 <= 0 else delta1 - n (b.delta1)/(n.n)
           degenerate if S < 1e-6 or a.a < 1e-24; the second constraint is treated as dependent if n.n < 1e-12 b.b
Measured per update, for delta0 and delta:
  first order: a.x, b.x; the directional derivative of the sequence loss, of English matching (rows whose query is an English form), of all
               matching, and of the total loss; norm retained and cos(delta, delta0); cos(a, b); which constraints were active
  finite steps: at s * x for s in 0.5, 1, 2, re-evaluated on the same batch with the same draws: the change of log S and of E, of the sequence loss,
               English matching and the total loss; and, on fixed held-out inputs (evaluation collections and taps, counts 1-9), the change of log
               collection spread and of the mean same-count collection/taps squared distance
Two paths of 20 updates, both seeing identical batches and draws:
  audit      the proposals are measured but delta0 is applied (the original relearning)
  corrected  delta is applied; Adam's moments are updated from the raw clipped gradient as usual (the step is corrected, not the gradient)
             reads (experiments.relearning_branches.Reads, plus composition) at updates 0, 10 and 20
The corrected path is a 20-update feasibility trace, not a continuation.

`continuation` (step 3, after the gate passed; docs/relearning-constrained-audit-v1.md): the corrected path alone for 200 updates, reads every 10, the
full corrected step at s = 1 with no backtracking (declared from the audit: the realized batch changes of log S and E stayed within about 1% of the
proposals'), realized changes logged every update. Compared with stage A's `unchanged` and `seq-cand-stop` arms, which start from the same state
and see identical batches. Report: reports/relearning-constrained-continuation-v1.json(.gz).

Declared before the run, the gate for one constrained continuation (step 3 of the analysis), each in every seed:
  G1  useful update survives: median over the 20 audit updates of ||delta|| / ||delta0|| >= 0.5
  G2  learning direction survives: summed over the 20 audit updates, the corrected step's first-order decrease of the sequence loss and of English
      matching is each >= 0.5 of the proposal's (when the proposal's is a decrease)
  G3  the protection holds at the actual step: summed over the corrected path's 20 updates, the realized batch |change of log S| <= 0.25 of the audit
      path's, and the realized batch change of E <= max(0, 0.25 x the audit path's); and the held-out collection spread at update 20 has fallen by
      at most a quarter of the audit path's fall
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_constrained_audit [continuation]
"""
import copy
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_remedies as RR
sys.argv = _argv
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

RM = RR.RM
RB, SC, S0, P, D, X, DEV, LM, SP = RM.RB, RM.SC, RM.S0, RM.P, RM.D, RM.X, RM.DEV, RM.LM, RM.SP
NAME = 'relearning-constrained-audit-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS, READS = 20, (0, 10, 20)
CONT = len(sys.argv) > 1 and sys.argv[1] == 'continuation'
if CONT:
    NAME = 'relearning-constrained-continuation-v1'
    RUNS, REPORT = X.ROOT / 'runs' / NAME, X.ROOT / 'reports' / NAME
    STEPS, READS = 200, tuple(range(0, 201, 10))
SCALES = (0.5, 1.0, 2.0)


def context_pairs(material):
    """(collection item, taps item) for every step of the context episodes, which follow the form episodes in the sequences."""
    return [(st['coll'], st['other']) for steps in material[3][material[4]:] for _, st in steps]


def geometry(c, pairs):
    ci = torch.tensor([p[0] for p in pairs]); ti = torch.tensor([p[1] for p in pairs])
    cc = c[ci]
    S = ((cc - cc.mean(0)) ** 2).sum(-1).mean()
    E = ((cc - c[ti]) ** 2).sum(-1).mean()
    return S, E


def english_matching(model, material, c, cand):
    items, match = material[0], material[2]
    m = [x for x in match if items[x[0]][0] == 'form' and items[x[0]][1] == 'english']
    fi = torch.tensor([x[0] for x in m]); cj = torch.tensor([x[1] for x in m]); nn_ = torch.tensor([x[2] for x in m]) - 1
    C = cand[None].expand(len(m), -1, -1).clone()
    C[torch.arange(len(m)), nn_] = c[cj]
    logits = P.same_logits(model, c[fi][:, None].expand_as(C).reshape(-1, D.SIZES[2]), C.reshape(-1, D.SIZES[2])).reshape(len(m), 9)
    return F.cross_entropy(logits, nn_)


def evaluate(enc, model, material, rng, pairs):
    """Losses and batch geometry at the current parameters, with the given draws; no gradient."""
    with torch.no_grad():
        L, c, cand, _ = RM.forward(enc, model, material, copy.deepcopy(rng))
        S, E = geometry(c, pairs)
        return {'log_S': float(S.log()), 'E': float(E), 'sequence': float(L['sequence']), 'english_matching': float(english_matching(model, material, c, cand)),
                'matching': float(L['match']), 'total': float(sum(L.values()))}


def held_out(enc, model):
    with torch.no_grad():
        f = RM.fixed_codes(enc, model)
        col, taps = f['collections'], f['taps']
        return {'log_collection_spread': float(((col - col.mean(0)) ** 2).sum(-1).mean().log()),
                'collection_taps_mismatch': float(((col - taps) ** 2).sum(-1).mean()),
                'log_taps_spread': float(((taps - taps.mean(0)) ** 2).sum(-1).mean().log())}


def correct(d0, a, b):
    aa = float(a @ a)
    info = {'a_norm': aa ** .5, 'b_norm': float(b.norm()), 'cos_a_b': float(a @ b / (a.norm() * b.norm() + 1e-30))}
    if aa < 1e-24:
        return None, {**info, 'degenerate': True}
    d1 = d0 - a * (a @ d0) / aa
    n = b - a * (a @ b) / aa
    info['b_dot_delta1'] = float(b @ d1)
    info['dependent'] = bool(float(n @ n) < 1e-12 * float(b @ b))
    info['mismatch_constraint_active'] = bool(float(b @ d1) > 0 and not info['dependent'])
    d = d1 - n * (b @ d1) / (n @ n) if info['mismatch_constraint_active'] else d1
    return d, {**info, 'degenerate': False}


def step(enc, model, opt, params, material, rng, corrected, finite=True):
    """One relearning update as run (RM.forward, clip 5, Adam), with the proposal audited; applies delta0 or, if corrected, delta."""
    pairs = context_pairs(material)
    rng0 = copy.deepcopy(rng)
    t_aud = time.perf_counter()
    L, c, cand, _ = RM.forward(enc, model, material, rng)
    total = sum(L.values())
    S, E = geometry(c, pairs)
    if float(S) < 1e-6:
        raise RuntimeError('degenerate context spread')
    grad = lambda l: torch.cat([(g if g is not None else torch.zeros_like(p_)).reshape(-1)
                                for p_, g in zip(params, torch.autograd.grad(l, params, retain_graph=True, allow_unused=True))])
    g = {'sequence': grad(L['sequence']), 'english_matching': grad(english_matching(model, material, c, cand)), 'matching': grad(L['match']),
         'total': grad(total)}
    a, b = grad(S.log()), grad(E)
    t_aud = time.perf_counter() - t_aud
    theta = p2v(params).detach().clone()
    opt.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    d0 = p2v(params).detach() - theta
    t_c = time.perf_counter()
    d, info = correct(d0, a, b)
    t_c = time.perf_counter() - t_c
    row = {'losses': {k: float(v.detach()) for k, v in L.items()}, 'S': float(S), 'E': float(E), 'constraints': info,
           'seconds': {'extra_gradients': t_aud, 'correction': t_c}}
    if d is None:
        row['degenerate'] = True
        return row
    row['delta0_norm'], row['delta_norm'] = float(d0.norm()), float(d.norm())
    row['retained'] = row['delta_norm'] / max(row['delta0_norm'], 1e-30)
    row['cos_delta_delta0'] = float(d @ d0 / (d.norm() * d0.norm() + 1e-30))
    row['first_order'] = {name: {'a': float(a @ x), 'b': float(b @ x), **{k: float(v @ x) for k, v in g.items()}} for name, x in (('delta0', d0), ('delta', d))}
    if not finite:
        v2p(theta, params)
        base = evaluate(enc, model, material, rng0, pairs)
        v2p(theta + d, params)
        e = evaluate(enc, model, material, rng0, pairs)
        row['finite'] = {'delta': {'1.0': {k: e[k] - base[k] for k in e}}}
        return row
    t_f = time.perf_counter()
    v2p(theta, params)
    base, base_h = evaluate(enc, model, material, rng0, pairs), held_out(enc, model)
    fin = {}
    for name, x in (('delta0', d0), ('delta', d)):
        fin[name] = {}
        for s in SCALES:
            v2p(theta + s * x, params)
            e, h = evaluate(enc, model, material, rng0, pairs), held_out(enc, model)
            fin[name][str(s)] = {**{k: e[k] - base[k] for k in e}, **{f'held_out/{k}': h[k] - base_h[k] for k in h}}
    row['finite'] = fin
    row['seconds']['finite_checks'] = time.perf_counter() - t_f
    v2p(theta + (d if corrected else d0), params)
    return row


def path(name, box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    params = [p_ for g_ in opt.param_groups for p_ in g_['params']]
    withheld = box['withheld']
    rows, track, held = [], {}, {0: held_out(enc, model)}
    for t in range(STEPS + 1):
        if name == 'corrected' and t in READS:
            track[t] = RR.read_all(reads, enc, model)
        if t == STEPS:
            break
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        LM.CURRENT['step'] += 1
        rows.append(step(enc, model, opt, params, material, st['rng'], name == 'corrected', not CONT))
        held[t + 1] = held_out(enc, model)
    return {'per_update': rows, 'held_out': held, 'track': track}


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RUNS
    box = RB.capture(seed)
    out = {'delay_state_max_abs_difference_from_step8': RB.state_difference(
        box['weights'], torch.load(SC.RUNS / RB.ENC / 'checkpoints' / f'sparse48-seed{seed}-delay.pt', weights_only=False, map_location='cpu'))}
    out['capture_seconds'] = time.perf_counter() - t0
    reads = RB.Reads(seed, box['withheld'])
    for name in (('corrected',) if CONT else ('audit', 'corrected')):
        b0 = time.perf_counter()
        out[name] = path(name, box, reads)
        out[name]['seconds'] = time.perf_counter() - b0
    out['parameters'] = int(sum(p_.numel() for p_ in [*box['enc']['image'].parameters(), *box['enc']['sequence'].parameters(), *box['model'].parameters()]))
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    print('seed', seed, f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, RB.SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'constrained continuation, 200 updates' if CONT else 'feasibility audit of the constrained update; 20 updates per path, no continuation', 'seeds': RB.SEEDS,
                                    'steps': STEPS, 'scales': SCALES, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'constrained continuation' if CONT else 'constrained-update audit', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
