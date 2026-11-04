"""Stage A of the relearning remedies plan (docs/relearning-remedies-plan.md): four arms of 200 relearning updates from the captured pre-relearning state,
changing only the sequence-prediction loss.
  unchanged        reference (reproduces the original relearning)
  seq-cand-stop    sequence prediction's gradient into its candidates stopped, on all sequence rows; other losses unchanged
  seq-relative     logits -exp(scale) * s0 * ||p - c_k||^2 / (s + eps), s the row's nine candidates' mean squared distance from their centroid
                   (gradient through s), s0 the mean of s over the first relearning batch's rows at the captured state (no gradient),
                   eps = 1e-6 * s0; scale keeps training
  seq-weight-0.1   the original sequence loss at weight 0.1
Reads every 10 updates: experiments.relearning_branches' reads (interpreter, memory-present, English) plus sequence prediction and composition
(sequence_pilot.composition), and the training rows' mean candidate spread.
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_remedies
"""
import copy
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_mechanism as RM
sys.argv = _argv
import numpy as np
import torch
import torch.nn.functional as F

RB, SC, S0, P, D, X, DEV, LM, SP, TP = RM.RB, RM.SC, RM.S0, RM.P, RM.D, RM.X, RM.DEV, RM.LM, RM.SP, RM.TP
NAME = 'relearning-remedies-a-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS, EVERY = 200, 10
ARMS = ('unchanged', 'seq-cand-stop', 'seq-relative', 'seq-weight-0.1')
WEIGHT = 0.1


def seq_rows(material):
    seqs = material[3]
    rows = []
    for steps in seqs:
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
    return rows


def spread(Cs):
    return ((Cs - Cs.mean(1, keepdim=True)) ** 2).sum(-1).mean(1)


def sequence_loss(model, material, c, cand, arm, s0):
    rows = seq_rows(material)
    if not rows:
        return torch.zeros(()), None
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt = torch.tensor([r[3] for r in rows]) - 1
    Cs = cand[None].expand(len(rows), -1, -1).clone()
    Cs[torch.arange(len(rows)), cnt] = c[tgt]
    pred = model.predict(c[prev], c[cur])
    s = spread(Cs)
    if arm == 'seq-cand-stop':
        Cs = Cs.detach()
    d2 = (pred[:, None] - Cs).pow(2).sum(-1)
    if arm == 'seq-relative':
        logits = -model.scale.exp() * s0 * d2 / (s + 1e-6 * s0)[:, None]
    else:
        logits = -model.scale.exp() * d2  # TP.predict_logits
    loss = F.cross_entropy(logits, cnt)
    if arm == 'seq-weight-0.1':
        loss = WEIGHT * loss
    return loss, float(s.detach().mean())


def first_batch_spread(enc, model, material, rng):
    with torch.no_grad():
        _, c, cand, _ = RM.forward(enc, model, material, copy.deepcopy(rng))
        rows = seq_rows(material)
        cnt = torch.tensor([r[3] for r in rows]) - 1
        tgt = torch.tensor([r[2] for r in rows])
        Cs = cand[None].expand(len(rows), -1, -1).clone()
        Cs[torch.arange(len(rows)), cnt] = c[tgt]
        return float(spread(Cs).mean())


def read_all(reads, enc, model):
    out = reads(enc, model)
    enc['image'].eval(); enc['sequence'].eval()
    out['composition'] = SP.composition(enc, model)
    enc['image'].train(); enc['sequence'].train()
    return out


def branch(arm, box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    withheld = box['withheld']
    params = [p_ for g in opt.param_groups for p_ in g['params']]
    track, losses, spreads, s0 = {}, [], [], None
    for t in range(STEPS + 1):
        if t % EVERY == 0:
            track[t] = read_all(reads, enc, model)
        if t == STEPS:
            break
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        if t == 0:
            s0 = first_batch_spread(enc, model, material, st['rng'])
        LM.CURRENT['step'] += 1
        L, c, cand, _ = RM.forward(enc, model, material, st['rng'])
        if arm != 'unchanged':
            L['sequence'], s_mean = sequence_loss(model, material, c, cand, arm, s0)
        else:
            s_mean = None
        loss = L['match'] + L['next'] + L['compare'] + L['sequence']
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        losses.append([round(float(v.detach()), 4) for v in L.values()])
        spreads.append(s_mean)
    return track, losses, spreads, s0


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RUNS
    box = RB.capture(seed)
    out = {'delay_state_max_abs_difference_from_step8': RB.state_difference(
        box['weights'], torch.load(SC.RUNS / RB.ENC / 'checkpoints' / f'sparse48-seed{seed}-delay.pt', weights_only=False, map_location='cpu'))}
    reads = RB.Reads(seed, box['withheld'])
    for arm in ARMS:
        b0 = time.perf_counter()
        track, losses, spreads, s0 = branch(arm, box, reads)
        out[arm] = {'track': track, 'loss_every_update': losses, 'training_candidate_spread': spreads, 's0': s0, 'seconds': time.perf_counter() - b0}
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
    S0.save_json(f'{REPORT}.json', {'status': 'stage A screen of the remedies plan; only the sequence-prediction loss changes', 'seeds': RB.SEEDS, 'arms': ARMS,
                                    'steps': STEPS, 'every': EVERY, 'weight': WEIGHT, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'remedies stage A', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
