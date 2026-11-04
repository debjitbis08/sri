"""Does selectively removing one loss path's contribution to the applied update protect the decisions attributed to it, while leaving useful
acquisition intact? (docs/relearning-path-removal-v1.md; proposal and predictions in docs/relearning-margins-v1.md)

The path: form rows' sequence-prediction gradient through the candidate collections (relearning_margins: `seq form via candidates`). From the captured
pre-relearning state per seed, 20 relearning updates, identical batches and draws, three arms:
  unchanged          the original relearning (stage A's reference)
  form-cand-stop     ordinary stop-gradient of that path: form rows' candidates detached in the sequence loss; clipping and Adam on the resulting gradient
  form-cand-removed  the path is removed from the applied update only: on this arm's own trajectory, the clip factor and Adam's second moment are
                     computed from the full gradient (all paths, as unchanged would); the first moment receives the clipped full gradient minus the
                     clipped path gradient. Not a replay of the original run's recorded clip factors or denominators, which would be a different experiment.
Context rows' sequence gradients (both sides), form rows' prediction path and every other loss are kept in all arms.
Reads at 0, 5, 10, 15, 20: experiments.relearning_branches.Reads (interpreter, memory present with actual retrieval, English matching) and
sequence_pilot.composition (sequence and composition, including English and other forms' sequence prediction). Logged per update: losses, the clip
factor, and the path's share of the clipped full gradient.
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_path_removal
"""
import copy
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_margins as RMg
from experiments import relearning_remedies as RR
sys.argv = _argv
import torch

RM = RMg.RM
RB, SC, S0, P, D, X, DEV, LM, SP = RM.RB, RM.SC, RM.S0, RM.P, RM.D, RM.X, RM.DEV, RM.LM, RM.SP
NAME = 'relearning-path-removal-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS, READS = 20, (0, 5, 10, 15, 20)
ARMS = ('unchanged', 'form-cand-stop', 'form-cand-removed')
PATH = 'seq form via candidates'


def removed_step(enc, model, opt, params, material, rng):
    """One update with the path removed from the first moment only; clip factor and second moment from the full gradient."""
    Lc, value = RMg.component_losses(enc, model, material, rng)
    full = sum(Lc.values())  # its gradient is the full gradient: the two form-row sequence paths sum to the form rows' sequence gradient
    g = torch.autograd.grad(full, params, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(Lc[PATH], params, allow_unused=True)
    g = [x if x is not None else torch.zeros_like(p_) for p_, x in zip(params, g)]
    gp = [x if x is not None else torch.zeros_like(p_) for p_, x in zip(params, gp)]
    total = float(torch.sqrt(sum((x ** 2).sum() for x in g)))
    clip = min(1., 5. / (total + 1e-6))
    group = opt.param_groups[0]
    b1, b2 = group['betas']; lr, eps = group['lr'], group['eps']
    with torch.no_grad():
        for p_, x, xp in zip(params, g, gp):
            s = opt.state[p_]
            s['step'] = s['step'] + 1
            step = float(s['step'])
            s['exp_avg'].mul_(b1).add_((1 - b1) * clip * (x - xp))
            s['exp_avg_sq'].mul_(b2).add_((1 - b2) * (clip * x) ** 2)
            den = s['exp_avg_sq'].sqrt() / (1 - b2 ** step) ** 0.5 + eps
            p_.add_(-lr / (1 - b1 ** step) * s['exp_avg'] / den)
    share = float(torch.sqrt(sum((x ** 2).sum() for x in gp)) / total)
    return {'losses': {k: float(v.detach()) for k, v in Lc.items()}, 'value': value, 'clip': clip, 'path_share_of_gradient': share}


def arm(name, box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    params = [p_ for g_ in opt.param_groups for p_ in g_['params']]
    withheld = box['withheld']
    track, log = {}, []
    for t in range(STEPS + 1):
        if t in READS:
            track[t] = RR.read_all(reads, enc, model)
        if t == STEPS:
            break
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        LM.CURRENT['step'] += 1
        if name == 'form-cand-removed':
            log.append(removed_step(enc, model, opt, params, material, st['rng']))
            continue
        L, _, _, _ = RM.forward(enc, model, material, st['rng'], True, 'cand' if name == 'form-cand-stop' else None)
        loss = L['match'] + L['next'] + L['compare'] + L['sequence']
        opt.zero_grad()
        loss.backward()
        total = float(torch.nn.utils.clip_grad_norm_(params, 5.0))
        opt.step()
        log.append({'losses': {k: float(v.detach()) for k, v in L.items()}, 'clip': min(1., 5. / (total + 1e-6))})
    return track, log


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
    for a in ARMS:
        b0 = time.perf_counter()
        track, log = arm(a, box, reads)
        out[a] = {'track': track, 'per_update': log, 'seconds': time.perf_counter() - b0}
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
    S0.save_json(f'{REPORT}.json', {'status': '20-update diagnostic of removing one loss path from the applied update', 'seeds': RB.SEEDS, 'arms': ARMS,
                                    'steps': STEPS, 'reads_at': READS, 'path': PATH, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning path removal', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
