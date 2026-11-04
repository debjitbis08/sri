"""What happens in the first relearning updates, which losses contribute, and does reverse validity recover with longer learning? (docs/relearning-dynamics-v1.md)

From the same captured pre-relearning state per seed as experiments.relearning_branches (reproduced exactly):
  as-run-1000    relearning as run, 1,000 updates; gradients and updates logged over the first 20
  no-forms-20    the delay's material, 20 updates, logged: a reference for step sizes from the same optimizer state
  no-form-seq    relearning as run, with the sequence-prediction rows of form episodes (English and rehearsed scripts) removed, 200 updates
  no-form-match  relearning as run, with the matching rows whose query is a form removed, 200 updates
The ablations keep every batch draw and the original denominators: removed rows contribute zero to a mean over the same count, so context terms keep
their weight. Removing a loss also changes the combined gradient and the clipping, so each is a test of that loss's contribution, not an isolation.
Logged per update (first 20): per loss (matching, succession, comparison, sequence) and per parameter group, the gradient norm; cosines between the
losses' gradients; the total norm and the clipping factor; and the actual parameter update relative to each group's norm.
Reads every 10 updates as experiments.relearning_branches.
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_dynamics
"""
import os
import sys
import time

_argv = sys.argv
sys.argv = [_argv[0]]
from experiments import relearning_branches as RB
sys.argv = _argv
import numpy as np
import torch
import torch.nn.functional as F

SC, S0, P, D, X, nc, DEV, LM, SP, SRD = RB.SC, RB.S0, RB.P, RB.D, RB.X, RB.nc, RB.DEV, RB.LM, RB.SP, RB.SRD
TP = LM.TP
NAME = 'relearning-dynamics-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
LOGGED = 20
LOSSES = ('match', 'next', 'compare', 'sequence')
BRANCHES = {'as-run-1000': 1000, 'no-forms-20': 20, 'no-form-seq': 200, 'no-form-match': 200}


def groups(enc, model):
    out = {'image projection': list(enc['image'].parameters()), 'taps encoder': list(enc['sequence'].parameters())}
    for prefix, name in (('proj', 'core projection'), ('token', 'relation tokens'), ('g', 'comparator g'), ('P', 'predictor P'), ('scale', 'scale')):
        out[name] = [p for n, p in model.named_parameters() if n.split('.')[0] == prefix]
    return out


def sequence_loss(model, material, c, cand, keep_forms=True):
    """SP.sequence_loss; with keep_forms False, rows from form episodes contribute zero to a mean over the same number of rows."""
    items, _, _, seqs = material[:4]
    n_form_eps = material[4]
    rows, from_form = [], []
    for e, steps in enumerate(seqs):
        for t in range(1, len(steps) - 1):
            n_next, nxt = steps[t + 1]
            if 'coll' not in nxt:
                continue
            for kind in steps[t][1]:
                if kind in steps[t - 1][1]:
                    rows.append((steps[t - 1][1][kind], steps[t][1][kind], nxt['coll'], n_next))
                    from_form.append(e < n_form_eps)
    if not rows:
        return torch.zeros(())
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt = torch.tensor([r[3] for r in rows]) - 1
    C = cand[None].expand(len(rows), -1, -1).clone()
    C[torch.arange(len(rows)), cnt] = c[tgt]
    logits = TP.predict_logits(model, model.predict(c[prev], c[cur]), C)
    if keep_forms:
        return F.cross_entropy(logits, cnt)
    keep = torch.tensor([not f for f in from_form], dtype=torch.float32)
    return (F.cross_entropy(logits, cnt, reduction='none') * keep).sum() / len(rows)


def update(enc, model, opt, material, rng, compare, ablate=None, log=None, grp=None):
    """LM.train_update for sparse48 at relearning (no RigL, no decay) with TP.train_update's losses and draws, copied; `ablate` removes form rows."""
    LM.CURRENT['step'] += 1
    assert LM.CURRENT.get('r') is None and LM.CURRENT['step'] > (LM.CURRENT.get('end') or LM.FI.END_FRACTION * 3 * D.STEPS)
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
    if ablate == 'form-match':
        keep = torch.tensor([items[m[0]][0] != 'form' for m in match], dtype=torch.float32)
        loss_match = (F.cross_entropy(logits, nn_, reduction='none') * keep).sum() / len(match)
    else:
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
    loss_t = sequence_loss(model, material, c, cand, keep_forms=ablate != 'form-seq')
    loss = loss_match + loss_next + loss_cmp + loss_t
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('relearning dynamics update')
    params = [p_ for g in opt.param_groups for p_ in g['params']]
    if log is not None:  # per-loss gradients, without touching .grad
        per = {}
        for name, l in zip(LOSSES, (loss_match, loss_next, loss_cmp, loss_t)):
            gs = torch.autograd.grad(l, params, retain_graph=True, allow_unused=True) if l.requires_grad else [None] * len(params)
            per[name] = {id(p_): (g if g is not None else torch.zeros_like(p_)) for p_, g in zip(params, gs)}
        before = {id(p_): p_.detach().clone() for p_ in params}
    opt.zero_grad()
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    if log is not None:
        flat = lambda d, ps: torch.cat([d[id(p_)].reshape(-1) for p_ in ps]) if ps else torch.zeros(1)
        row = {'losses': [float(x.detach()) for x in (loss_match, loss_next, loss_cmp, loss_t)], 'total_grad_norm': float(total),
               'clip_factor': float(min(1., 5.0 / (float(total) + 1e-6))), 'groups': {}, 'cosines': {}}
        for gname, ps in grp.items():
            if not ps:
                continue
            pn = float(torch.sqrt(sum((p_.detach() ** 2).sum() for p_ in ps)))
            un = float(torch.sqrt(sum(((p_.detach() - before[id(p_)]) ** 2).sum() for p_ in ps)))
            row['groups'][gname] = {'update_over_param': un / max(pn, 1e-12), 'param_norm': pn,
                                    **{f'grad_{n}': float(flat(per[n], ps).norm()) for n in LOSSES}}
        allp = params
        vec = {n: flat(per[n], allp) for n in LOSSES}
        for i, a_ in enumerate(LOSSES):
            for b_ in LOSSES[i + 1:]:
                na_, nb_ = vec[a_].norm(), vec[b_].norm()
                row['cosines'][f'{a_}|{b_}'] = float(vec[a_] @ vec[b_] / (na_ * nb_)) if na_ > 0 and nb_ > 0 else None
        for gname in ('image projection', 'core projection', 'comparator g'):
            ps = grp[gname]
            v = {n: flat(per[n], ps) for n in ('match', 'sequence')}
            nm, ns_ = v['match'].norm(), v['sequence'].norm()
            row['cosines'][f'match|sequence in {gname}'] = float(v['match'] @ v['sequence'] / (nm * ns_)) if nm > 0 and ns_ > 0 else None
        log.append(row)
    return [round(float(x.detach()), 4) for x in (loss_match, loss_next, loss_cmp, loss_t)]


def run_branch(name, box, reads, steps):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    grp = groups(enc, model)
    withheld = box['withheld']
    track, losses, log = {}, [], []
    ablate = {'no-form-seq': 'form-seq', 'no-form-match': 'form-match'}.get(name)
    final200 = None
    for t in range(steps + 1):
        if t % RB.EVERY == 0 and name != 'no-forms-20':
            track[t] = reads(enc, model)
        if t == 200:
            final200 = {k: {n: (v.detach().clone() if torch.is_tensor(v) else v) for n, v in m.state_dict().items()}  # cloned: state_dict holds live tensors
                        for k, m in (('image', enc['image']), ('sequence', enc['sequence']), ('sri', model))}
        if t == steps:
            break
        if name == 'no-forms-20':
            eps = []
            material = SP.batch_material(eps, D.context_episodes(st['drng'], D.N_CONTEXT), 'taps')
        else:
            eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
            eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
            material = SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps')
        material = (*material, len(eps))  # the number of form episodes, which come first in the sequences
        losses.append(update(enc, model, opt, material, st['rng'], True, ablate, log if t < LOGGED else None, grp))
    return track, losses, log, final200


def task(seed):
    t0, c0 = time.perf_counter(), time.process_time()
    torch.set_num_threads(LM.FI.THREADS)
    DEV.install()
    RB.RUNS = RUNS
    box = RB.capture(seed)
    delay_ck = torch.load(SC.RUNS / RB.ENC / 'checkpoints' / f'sparse48-seed{seed}-delay.pt', weights_only=False, map_location='cpu')
    out = {'delay_state_max_abs_difference_from_step8': RB.state_difference(box['weights'], delay_ck)}
    reads = RB.Reads(seed, box['withheld'])
    ref = torch.load(SC.RUNS / RB.ENC / 'checkpoints' / f'sparse48-seed{seed}-relearn.pt', weights_only=False, map_location='cpu')
    for name, steps in BRANCHES.items():
        b0 = time.perf_counter()
        track, losses, log, final200 = run_branch(name, box, reads, steps)
        out[name] = {'track': track, 'loss_every_update': losses, 'log_first_updates': log, 'seconds': time.perf_counter() - b0}
        if name == 'as-run-1000':
            out[name]['update200_state_max_abs_difference_from_step8_relearn'] = RB.state_difference(final200, ref)
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    print('seed', seed, 'delay check', out['delay_state_max_abs_difference_from_step8'], 'relearn@200 check',
          out['as-run-1000']['update200_state_max_abs_difference_from_step8_relearn'], f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, RB.SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'diagnostic continuations and logging from one captured pre-relearning state per seed; no reverse-direction training',
                                    'encoder': RB.ENC, 'seeds': RB.SEEDS, 'branches': BRANCHES, 'logged_updates': LOGGED, 'every': RB.EVERY,
                                    'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning dynamics', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
