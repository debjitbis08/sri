"""The precise route of the relearning disruption (docs/relearning-mechanism-v1.md).

The sequence-prediction logits are -exp(scale) * ||pred - cand||^2 (transition_pilot.predict_logits): unbounded squared distances. Forms and collections
share the image code projection proj['image'] (code = proj[medium](encoder output)); taps have proj['sequence']. Hypothesis under test: forgotten forms
predict codes far from their target collections; the cheapest descent on a squared-distance loss is to shrink every image code, which contracts the
collections with the forms, while taps keep their own projection.

From the same captured pre-relearning state per seed (experiments.relearning_branches):
  radial         at the first relearning batch, for each loss: the cosine between its gradient on proj['image'] and the projection's own weights (a
                 gradient parallel to the weights asks for a uniform shrink of all image codes), and the directional derivative along that shrink;
                 the same for the scale; on a delay batch for reference
  branches, 50 updates each, one change at a time:
    as-run           unchanged (for the movement decomposition)
    seq-no-cand      the form rows' sequence-prediction gradient is stopped into the candidate collections (the targets cannot be moved by it)
    seq-no-input     the form rows' sequence-prediction gradient is stopped into the codes the prediction is made from
    scale-frozen     the learned inverse temperature of the sequence loss is held fixed
  per update (first 20): the cosine of the actual proj['image'] update with -weights, and each loss's gradient radial cosine
  at 0, 10, 20, 50: codes of fixed inputs against their own update-0 codes in the same run: the best uniform scale a, and the share of the movement
  a uniform scale cannot explain; and the branches' reads
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_mechanism
"""
import copy
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_dynamics as RD
sys.argv = _argv
import numpy as np
import torch
import torch.nn.functional as F

RB, SC, S0, P, D, X, nc, DEV, LM, SP, TP = RD.RB, RD.SC, RD.S0, RD.P, RD.D, RD.X, RD.nc, RD.DEV, RD.LM, RD.SP, RD.TP
NAME = 'relearning-mechanism-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS, LOGGED = 50, 20
BRANCHES = ('as-run', 'seq-no-cand', 'seq-no-input', 'scale-frozen')
LOSSES = RD.LOSSES


def forward(enc, model, material, rng, compare=True, stop=None):
    """RD.update's losses, without the step; stop: 'cand' or 'input' stops the form rows' sequence-prediction gradient on that side."""
    items, pairs, match = material[:3]
    solvers = {'image': P.Coupled(model)}
    if any(it[0] == 'taps' for it in items):
        solvers['sequence'] = D.Sequence(model)
    c = D.encode_items(enc, model, items, solvers)
    ci = rng.integers(len(P.DOTS['train']), size=9)
    cand = solvers['image'].solve(enc['image'](torch.from_numpy(np.stack([P.DOTS['train'][ci[k], k] for k in range(9)]))))[:, P.INNER]
    fi = torch.tensor([m[0] for m in match]); cj = torch.tensor([m[1] for m in match]); nn_ = torch.tensor([m[2] for m in match]) - 1
    C = cand[None].expand(len(match), -1, -1).clone()
    C[torch.arange(len(match)), nn_] = c[cj]
    logits = P.same_logits(model, c[fi][:, None].expand_as(C).reshape(-1, D.SIZES[2]), C.reshape(-1, D.SIZES[2])).reshape(len(match), 9)
    loss_match = F.cross_entropy(logits, nn_)
    a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
    kinds = [items[p[0]][0] for p in pairs]
    by_kind = {k: [i for i, it in enumerate(items) if it[0] == k] for k in ('form', 'coll', 'taps')}
    na, nb = [], []
    for k in kinds:
        pool = by_kind[k]
        for _ in range(20):
            u, v = rng.choice(pool, 2)
            if items[v][2] != items[u][2] + 1:
                na.append(u); nb.append(v)
                break
    a2 = torch.cat([a, torch.tensor(na, dtype=torch.long)]); b2 = torch.cat([b, torch.tensor(nb, dtype=torch.long)])
    target = torch.cat([torch.ones(len(a)), torch.zeros(len(na))])
    loss_next = F.binary_cross_entropy_with_logits(model.relation_logits(c[a2], c[b2], 0), target)
    loss_cmp = torch.zeros(())
    if compare:
        pool = np.array(by_kind['coll'] + by_kind['taps'])
        u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
        cnt = np.array([it[2] for it in items])
        keep_ = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
        u, v = u[keep_], v[keep_]
        loss_cmp = F.binary_cross_entropy_with_logits(model.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
    # sequence prediction, as SP.sequence_loss, with optional stops on the form rows
    seqs, n_form = material[3], material[4]
    rows, form = [], []
    for e, steps in enumerate(seqs):
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
                    form.append(e < n_form)
    loss_t = torch.zeros(())
    if rows:
        prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
        cnt_ = torch.tensor([r[3] for r in rows]) - 1
        Cs = cand[None].expand(len(rows), -1, -1).clone()
        Cs[torch.arange(len(rows)), cnt_] = c[tgt]
        fm = torch.tensor(form)
        pred = model.predict(c[prev], c[cur])
        if stop == 'input':
            pred = torch.where(fm[:, None], model.predict(c[prev].detach(), c[cur].detach()), pred)
        if stop == 'cand':
            Cs = torch.where(fm[:, None, None], Cs.detach(), Cs)
        loss_t = F.cross_entropy(TP.predict_logits(model, pred, Cs), cnt_)
    return {'match': loss_match, 'next': loss_next, 'compare': loss_cmp, 'sequence': loss_t}, c, cand, items


def radial(enc, model, material, rng):
    """Per loss: cos(grad of proj['image'], its weights) and the directional derivative along a uniform shrink of all image codes; and d/d scale."""
    L, c, cand, items = forward(enc, model, material, copy.deepcopy(rng))
    W = [model.proj['image'].weight, model.proj['image'].bias]
    out = {}
    for name, l in L.items():
        if not l.requires_grad:
            continue
        g = torch.autograd.grad(l, W + [model.scale], retain_graph=True, allow_unused=True)
        gw = torch.cat([(x if x is not None else torch.zeros_like(w)).reshape(-1) for x, w in zip(g[:2], W)])
        w = torch.cat([x.detach().reshape(-1) for x in W])
        out[name] = {'loss': float(l.detach()), 'grad_norm': float(gw.norm()),
                     'cos_grad_weights': float(gw @ w / (gw.norm() * w.norm() + 1e-12)),  # +1: descent shrinks every image code uniformly
                     'd_loss_d_log_code_scale': float(gw @ w),  # d L / d log(alpha) for codes scaled by alpha (proj is linear)
                     'd_loss_d_scale': float(g[2]) if g[2] is not None else 0.}
    with torch.no_grad():
        img = [i for i, it in enumerate(items) if it[0] != 'taps']
        out['codes'] = {'image_items_mean_norm': float(c[img].norm(dim=1).mean()), 'candidates_mean_norm': float(cand.norm(dim=1).mean())}
    return out


def fixed_codes(enc, model):
    with torch.no_grad():
        code = lambda x, m: D.codes_for(enc, model, np.ascontiguousarray(x, dtype=np.float32), m)
        return {'collections': torch.cat([code(P.DOTS['succ'][n], 'image') for n in range(1, 10)]),
                'forms': torch.cat([code(X.DATA[('english', 'eval')][0][P.EVAL[('english', n, 'idx')]], 'image') for n in range(1, 10)]),
                'taps': torch.cat([code(D.TAPS[('eval', n)], 'sequence') for n in range(1, 10)])}


def movement(c0, ct):
    out = {}
    for k in c0:
        x, y = c0[k], ct[k]
        a = float((x * y).sum() / (x * x).sum())
        total = float((y - x).norm())
        out[k] = {'uniform_scale': a, 'unexplained_share': float((y - a * x).norm()) / max(total, 1e-12), 'relative_movement': total / float(x.norm())}
    return out


def branch(name, box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    if name == 'scale-frozen':
        model.scale.requires_grad_(False)
    withheld = box['withheld']
    params = [p_ for g in opt.param_groups for p_ in g['params']]
    W = [model.proj['image'].weight, model.proj['image'].bias]
    c0 = fixed_codes(enc, model)
    track, perupdate = {}, []
    stop = {'seq-no-cand': 'cand', 'seq-no-input': 'input'}.get(name)
    for t in range(STEPS + 1):
        if t in (0, 10, 20, 50):
            track[t] = {'reads': reads(enc, model), 'movement': movement(c0, fixed_codes(enc, model))}
        if t == STEPS:
            break
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        LM.CURRENT['step'] += 1
        Ls, _, _, _ = forward(enc, model, material, st['rng'], True, stop)
        loss = sum(Ls.values())
        row = None
        if t < LOGGED:
            w = torch.cat([x.detach().reshape(-1) for x in W])
            row = {'loss': {k: float(v.detach()) for k, v in Ls.items()}, 'scale': float(model.scale.detach()), 'cos_grad_weights': {}}
            for k, v in Ls.items():
                if v.requires_grad:
                    g = torch.autograd.grad(v, W, retain_graph=True, allow_unused=True)
                    gw = torch.cat([(x if x is not None else torch.zeros_like(y)).reshape(-1) for x, y in zip(g, W)])
                    row['cos_grad_weights'][k] = float(gw @ w / (gw.norm() * w.norm() + 1e-12))
            before = w.clone()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if row is not None:
            dw = torch.cat([x.detach().reshape(-1) for x in W]) - before
            row['cos_update_minus_weights'] = float(-(dw @ before) / (dw.norm() * before.norm() + 1e-12))  # +1: the step shrinks proj['image']
            row['proj_image_norm_ratio'] = float(torch.cat([x.detach().reshape(-1) for x in W]).norm() / before.norm())
            perupdate.append(row)
    model.scale.requires_grad_(True)
    return track, perupdate


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RUNS
    box = RB.capture(seed)
    out = {}
    # radial tests at the captured state: the first relearning batch, and a delay batch from the same state
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    LM.CURRENT['step'] += 1
    m_relearn = (*SP.batch_material(LM.rehearsal_batch(copy.deepcopy(eps0), copy.deepcopy(buffer), copy.deepcopy(st['reh_rng'])),
                                    D.context_episodes(copy.deepcopy(st['crng']), D.N_CONTEXT), 'taps'), len(eps0))
    m_delay = (*SP.batch_material([], D.context_episodes(copy.deepcopy(st['drng']), D.N_CONTEXT), 'taps'), 0)
    out['radial'] = {'relearn_batch': radial(enc, model, m_relearn, st['rng']), 'delay_batch': radial(enc, model, m_delay, st['rng'])}
    reads = RB.Reads(seed, box['withheld'])
    for name in BRANCHES:
        track, perupdate = branch(name, box, reads)
        out[name] = {'track': track, 'per_update': perupdate}
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
    S0.save_json(f'{REPORT}.json', {'status': 'mechanism diagnostics from one captured pre-relearning state per seed', 'seeds': RB.SEEDS, 'branches': BRANCHES,
                                    'steps': STEPS, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning mechanism', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
