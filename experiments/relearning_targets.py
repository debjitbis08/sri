"""Where do the forgotten forms' sequence predictions sit, and which way does the loss pull the collections they are scored against?
(docs/relearning-mechanism-v1.md, last section)

At the first relearning batch of the captured pre-relearning state (experiments.relearning_branches), for the sequence-prediction rows of form
episodes and, as reference, of context episodes:
  norms of the predictions, of the target collections and of the other candidates; target probability; distance to target and to the nearest candidate
  the descent direction on each candidate code (minus its gradient): its cosine with -c (towards the origin: shrinking) and with p - c (towards the
  prediction), separately for targets and for the other candidates; and the share of all collection-code gradient that comes from form rows
No training. Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_targets
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
REPORT = X.ROOT / 'reports' / 'relearning-targets-v1'


def rows_of(material):
    seqs, n_form = material[3], material[4]
    rows, form = [], []
    for e, steps in enumerate(seqs):
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next, kind))
                    form.append(e < n_form)
    return rows, np.array(form)


def task(seed):
    t0 = time.perf_counter()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RM.RUNS
    box = RB.capture(seed)
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    material = (*SP.batch_material(LM.rehearsal_batch(copy.deepcopy(eps0), copy.deepcopy(buffer), copy.deepcopy(st['reh_rng'])),
                                   D.context_episodes(copy.deepcopy(st['crng']), D.N_CONTEXT), 'taps'), len(eps0))
    items = material[0]
    solvers = {'image': P.Coupled(model), 'sequence': D.Sequence(model)}
    c = D.encode_items(enc, model, items, solvers)
    rng = copy.deepcopy(st['rng'])
    ci = rng.integers(len(P.DOTS['train']), size=9)
    cand = solvers['image'].solve(enc['image'](torch.from_numpy(np.stack([P.DOTS['train'][ci[k], k] for k in range(9)]))))[:, P.INNER]
    rows, form = rows_of(material)
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt = torch.tensor([r[3] for r in rows]) - 1
    C = cand[None].expand(len(rows), -1, -1).clone()
    C[torch.arange(len(rows)), cnt] = c[tgt]
    C = C.detach().requires_grad_(True)  # a leaf per row and candidate, to read the gradient on each candidate code
    pred = model.predict(c[prev], c[cur])
    logits = TP.predict_logits(model, pred, C)
    loss = F.cross_entropy(logits, cnt)
    gC, = torch.autograd.grad(loss, C)
    with torch.no_grad():
        prob = logits.softmax(1)
        p, Cd = pred.detach(), C.detach()
        desc = -gC  # the descent direction on each candidate code
        to_origin = F.cosine_similarity(desc, -Cd, dim=-1)
        to_pred = F.cosine_similarity(desc, p[:, None] - Cd, dim=-1)
        mag = desc.norm(dim=-1)
        is_t = torch.zeros_like(prob, dtype=torch.bool); is_t[torch.arange(len(rows)), cnt] = True
        dist = (p[:, None] - Cd).norm(dim=-1)
        out = {}
        for name, m in (('form_rows', torch.as_tensor(form)), ('context_rows', ~torch.as_tensor(form))):
            if not m.any():
                continue
            w = mag[m]
            def wmean(x, sel):  # gradient-magnitude-weighted mean over the selected candidate codes
                ww = w[sel]
                return float((x[m][sel] * ww).sum() / ww.sum())
            out[name] = {'rows': int(m.sum()), 'prediction_norm': float(p[m].norm(dim=1).mean()), 'target_norm': float(Cd[m][is_t[m]].norm(dim=-1).mean()),
                         'other_candidate_norm': float(Cd[m][~is_t[m]].norm(dim=-1).mean()), 'target_probability': float(prob[m][is_t[m]].mean()),
                         'distance_to_target': float(dist[m][is_t[m]].mean()), 'distance_to_nearest_candidate': float(dist[m].min(1).values.mean()),
                         'descent_cos_to_origin': {'targets': wmean(to_origin, is_t[m]), 'others': wmean(to_origin, ~is_t[m])},
                         'descent_cos_to_prediction': {'targets': wmean(to_pred, is_t[m]), 'others': wmean(to_pred, ~is_t[m])},
                         'gradient_share': float((mag[m] ** 2).sum() / (mag ** 2).sum())}
    return seed, out, time.perf_counter() - t0


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, RB.SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'no training; first relearning batch from the captured state', 'seeds': RB.SEEDS, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning targets', 'wall_seconds': time.perf_counter() - t0, 'task_wall_seconds': {f'seed{s}': w for s, _, w in res}}))


if __name__ == '__main__':
    main()
