"""Which updates make the core's previously correct decisions wrong in the first relearning updates, and through which parameters and loss paths?
(docs/relearning-margins-v1.md)

Decision margins on fixed evaluation examples (diagnosis only, never trained on), for each query: m = score of the correct candidate - max over the
wrong candidates; the answer is correct when m > 0.
  c->t fresh     an evaluation collection against nine evaluation taps, g(collection, taps)   (the read memory's recalled values need)
  c->t stored    memory's stored dot values against nine taps, the same order
  t->c           evaluation taps against nine evaluation collections, g(taps, collection)     (the trained order)
  seq/comp       predicted next code (k = 1, 2, 3 steps of model.predict) against the nine evaluation collections, score -||pred - c||^2
                 (the read of sequence_pilot.composition by distance; exp(scale) left out, as it does not change the ranking)
The draws are those of experiments.relearning_branches.Reads (collections_taps_diagnosis.fixed_draws), so the reads equal stage A's.

From the captured pre-relearning state per seed, the original relearning (stage A's `unchanged`) is rerun for 20 updates, instrumented:
  the update is split exactly into parts by loss path. Adam's step is -lr/bc1 * m_t / den_t, with m_t = b1 m_{t-1} + (1-b1) clip * sum_k g_k; given the
  actual denominator den_t (from the actual second moment) it is the sum over k of -lr/bc1 * M_k / den_t, where each M_k is that loss path's own
  moment, M_k = b1 M_k + (1-b1) clip g_k, and M_history starts as Adam's moment at the captured state and only decays. Paths:
    history, matching (form queries), matching (context), succession, comparison,
    sequence prediction, form rows, through the candidates | through the prediction, sequence prediction, context rows
  each part is split by parameter group: image head, image projection, taps encoder, taps projection, comparator (g and its tokens), predictor P,
  scale. The parts sum to the actual displacement (checked every update).
  for every output (correct score and max wrong score of every query) and every (path, group) part d: the change attributed to it,
    first order          grad o(theta_t) . d
    path integral        mean of grad o(theta_t + s Delta) . d at the two Gauss-Legendre points s = 1/2 -+ 1/(2 sqrt 3) along the actual step
  computed as Jacobian-vector products (double backward). Validated against the actual change of every output over each update and summed over
  updates 0-9 and 0-19.
  memory A/B     memory and core together: symbol -> taps on the values memory retrieves for the stage 1 queries, stage 1's candidates; a query whose
                 retrieval abstains counts as wrong whatever its margin
Lost decisions: queries with m > 0 at update 0 and m <= 0 at update 10 (20). Reported: their mean attributed change of the correct score, the max wrong
score and the margin by path and group, against queries that stay correct.
No new learning policy; no remedy.
Usage: NFI_DEVICE=cuda RB_WORKERS=3 /data/venvs/sri-cuda/bin/python -m experiments.relearning_margins
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
NAME = 'relearning-margins-v1'
RUNS = X.ROOT / 'runs' / NAME
REPORT = X.ROOT / 'reports' / NAME
STEPS = 20
PATHS = ('history', 'match form', 'match context', 'succession', 'comparison', 'seq form via candidates', 'seq form via prediction', 'seq context')
GAUSS = (0.5 - 0.5 / 3 ** 0.5, 0.5 + 0.5 / 3 ** 0.5)
SEQ_READS = (('collections', 'up', 1), ('collections', 'down', 1), ('taps', 'up', 1), ('collections', 'up', 2), ('collections', 'up', 3), ('taps', 'up', 2))


def param_groups(enc, model, params):
    g = {'image head': list(enc['image'].parameters()), 'image projection': list(model.proj['image'].parameters()),
         'taps encoder': list(enc['sequence'].parameters()), 'taps projection': list(model.proj['sequence'].parameters()),
         'comparator': [*model.g.parameters(), model.token.weight], 'predictor': list(model.P.parameters()), 'scale': [model.scale]}
    ids = {id(p_): name for name, ps in g.items() for p_ in ps}
    assert len(ids) == len(params) and all(id(p_) in ids for p_ in params), 'groups must cover the optimizer parameters exactly'
    return [ids[id(p_)] for p_ in params]


class Fixed:
    """The evaluation inputs and draws, as experiments.relearning_branches.Reads reads them."""

    def __init__(self, reads):
        dr = reads.ctd_draws
        E, T = P.EVAL_PER, len(D.TAPS[('eval', 1)])
        self.E, self.T = E, T
        self.values = reads.values
        self.qcounts = np.concatenate([np.repeat(np.arange(1, 10), E), reads.vcounts])
        nq = len(self.qcounts)
        self.order = torch.from_numpy(dr['order'][:nq]); self.tj = torch.from_numpy(dr['taps_j'][:nq])
        self.y_ct = torch.from_numpy(np.array([int(np.flatnonzero(dr['order'][i] + 1 == self.qcounts[i])[0]) for i in range(nq)]))
        self.ro = torch.from_numpy(dr['rev_order']); self.cj = torch.from_numpy(dr['coll_j'])
        tc = np.repeat(np.arange(1, 10), T)
        self.y_tc = torch.from_numpy(np.array([int(np.flatnonzero(dr['rev_order'][i] + 1 == tc[i])[0]) for i in range(len(tc))]))
        self.n_fresh = 9 * E
        # memory and core together: symbol -> taps on the values memory actually retrieves, with stage 1's candidate draws; abstentions count wrong
        self.mem = {}
        for name, st in reads.sets.items():
            order = st['draws']['order']
            self.mem[name] = {'value': st['value'], 'ok': np.asarray(st['ok'], bool), 'order': torch.from_numpy(order), 'tj': torch.from_numpy(st['draws']['tj']),
                              'y': torch.from_numpy(np.array([int(np.flatnonzero(order[i] + 1 == st['counts'][i])[0]) for i in range(len(order))]))}


def scores(enc, model, fx):
    """name -> (scores (n, 9), correct index (n,)); differentiable in every parameter."""
    # D.codes_for without its no_grad
    code = lambda x, m: model.split(D.B.Solver(model, m).solve(enc[m](torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))))['core_inner']
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in range(1, 10)])  # (9, E, d)
    taps = torch.stack([code(D.TAPS[('eval', n)], 'sequence') for n in range(1, 10)])  # (9, T, d)
    vals = model.split(D.B.Solver(model, 'image').solve(enc['image'].from_codes(fx.values)))['core_inner']
    d = colls.shape[-1]
    qc = torch.cat([colls.reshape(-1, d), vals])
    cand = taps[fx.order, fx.tj]  # (nq, 9, d)
    s_ct = P.same_logits(model, qc[:, None].expand_as(cand).reshape(-1, d), cand.reshape(-1, d)).reshape(len(qc), 9)
    tq = taps.reshape(-1, d)
    cc = colls[fx.ro, fx.cj]
    s_tc = P.same_logits(model, tq[:, None].expand_as(cc).reshape(-1, d), cc.reshape(-1, d)).reshape(len(tq), 9)
    out = {'c->t fresh': (s_ct[:fx.n_fresh], fx.y_ct[:fx.n_fresh]), 'c->t stored': (s_ct[fx.n_fresh:], fx.y_ct[fx.n_fresh:]), 't->c': (s_tc, fx.y_tc)}
    collsE = colls.transpose(0, 1)  # (E, 9, d): row i holds the i-th evaluation collection of every count, as composition reads
    codes = {'collections': colls, 'taps': taps}
    for kind, direction, k in SEQ_READS:
        sign = 1 if direction == 'up' else -1
        S, Y = [], []
        for n in range(1, 10):
            m = n + sign * k
            if not (1 <= n - sign <= 9 and 1 <= m <= 9):
                continue
            prev, cur = codes[kind][n - sign - 1], codes[kind][n - 1]
            for _ in range(k):
                prev, cur = cur, model.predict(prev, cur)
            S.append(-(cur[:, None] - collsE).pow(2).sum(-1)); Y.append(torch.full((len(cur),), m - 1))
        out[f'{"seq" if k == 1 else "comp"} {kind} {direction} {k}'] = (torch.cat(S), torch.cat(Y))
    for name, mm in fx.mem.items():
        Q = model.split(D.B.Solver(model, 'image').solve(enc['image'].from_codes(mm['value'])))['core_inner']
        tc = taps[mm['order'], mm['tj']]
        out[f'memory {name} symbol->taps'] = (P.same_logits(model, Q[:, None].expand_as(tc).reshape(-1, d), tc.reshape(-1, d)).reshape(len(Q), 9), mm['y'])
    return out


def outputs(enc, model, fx):
    """One flat vector: per read, the correct scores then the max wrong scores; and the layout."""
    parts, layout, pos = [], {}, 0
    for name, (S, y) in scores(enc, model, fx).items():
        ar = torch.arange(len(y))
        correct = S[ar, y]
        wrong = S.masked_fill(F.one_hot(y, 9).bool().to(S.device), float('-inf')).max(1).values
        parts += [correct, wrong]
        layout[name] = (pos, len(y)); pos += 2 * len(y)
    return torch.cat(parts), layout


def jvps(enc, model, params, fx, dirs):
    """outputs and their directional derivatives along each direction (lists of per-parameter tensors), by double backward."""
    out, layout = outputs(enc, model, fx)
    u = torch.zeros_like(out, requires_grad=True)
    g = torch.autograd.grad(out, params, grad_outputs=u, create_graph=True, allow_unused=True)
    res = []
    for v in dirs:
        terms = [(gi * vi).sum() for gi, vi in zip(g, v) if gi is not None and vi is not None]
        res.append(torch.autograd.grad(sum(terms), u, retain_graph=True)[0].detach() if terms else torch.zeros_like(out))
    return out.detach(), torch.stack(res), layout


def component_losses(enc, model, material, rng):
    """RM.forward's losses split by path, with the same draws; the split losses' gradients sum to the total gradient."""
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
    ce = F.cross_entropy(logits, nn_, reduction='none')
    qform = torch.tensor([items[m[0]][0] == 'form' for m in match])
    L = {'match form': (ce * qform).sum() / len(match), 'match context': (ce * ~qform).sum() / len(match)}
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
    L['succession'] = F.binary_cross_entropy_with_logits(model.relation_logits(c[a2], c[b2], 0), target)
    pool = np.array(by_kind['coll'] + by_kind['taps'])
    u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
    cnt = np.array([it[2] for it in items])
    keep_ = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
    u, v = u[keep_], v[keep_]
    L['comparison'] = F.binary_cross_entropy_with_logits(model.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
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
    prev, cur, tgt = (torch.tensor([r[j] for r in rows]) for j in range(3))
    cnt_ = torch.tensor([r[3] for r in rows]) - 1
    Cs = cand[None].expand(len(rows), -1, -1).clone()
    Cs[torch.arange(len(rows)), cnt_] = c[tgt]
    fm = torch.tensor(form)
    pred = model.predict(c[prev], c[cur])
    ce_ = lambda p_, C_: F.cross_entropy(TP.predict_logits(model, p_, C_), cnt_, reduction='none')
    full = ce_(pred, Cs)
    N = len(rows)
    # the temperature's gradient is counted once, in the prediction path
    ce_c = F.cross_entropy(-model.scale.detach().exp() * (pred.detach()[:, None] - Cs).pow(2).sum(-1), cnt_, reduction='none')
    L['seq form via candidates'] = (ce_c * fm).sum() / N
    L['seq form via prediction'] = (ce_(pred, Cs.detach()) * fm).sum() / N
    L['seq context'] = (full * ~fm).sum() / N
    value = float(L['match form'] + L['match context'] + L['succession'] + L['comparison'] + (full * fm).sum() / N + L['seq context'])
    return L, value


def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])


def instrumented(box, reads):
    enc, model, opt, st, buffer, eps0 = RB.restore(box)
    params = [p_ for g_ in opt.param_groups for p_ in g_['params']]
    gname = param_groups(enc, model, params)
    groups = sorted(set(gname), key=gname.index)
    fx = Fixed(reads)
    withheld = box['withheld']
    b1, b2 = opt.param_groups[0]['betas']; lr, eps_ = opt.param_groups[0]['lr'], opt.param_groups[0]['eps']
    M = {k: [torch.zeros_like(p_) for p_ in params] for k in PATHS}
    M['history'] = [opt.state[p_]['exp_avg'].detach().clone() for p_ in params]
    enc['image'].eval(); enc['sequence'].eval()
    o_prev, layout = outputs(enc, model, fx)
    o_prev = o_prev.detach()
    enc['image'].train(); enc['sequence'].train()
    traj = [o_prev.cpu().numpy()]
    att = {'first': [], 'path': []}
    checks = []
    for t in range(STEPS):
        eps = eps0 if t == 0 else P.episodes(P.LANGS[0], withheld[P.LANGS[0]], st['srng'], D.BATCH_EP)
        eps = LM.rehearsal_batch(eps, buffer, st['reh_rng'])
        material = (*SP.batch_material(eps, D.context_episodes(st['crng'], D.N_CONTEXT), 'taps'), len(eps))
        LM.CURRENT['step'] += 1
        rng0 = copy.deepcopy(st['rng'])
        L, _, _, _ = RM.forward(enc, model, material, st['rng'])  # the update exactly as stage A's unchanged arm
        loss = L['match'] + L['next'] + L['compare'] + L['sequence']
        Lc, value = component_losses(enc, model, material, copy.deepcopy(rng0))
        gk = {k: [g if g is not None else torch.zeros_like(p_) for p_, g in zip(params, torch.autograd.grad(l, params, retain_graph=True, allow_unused=True))]
              for k, l in Lc.items()}
        theta = [p_.detach().clone() for p_ in params]
        opt.zero_grad()
        loss.backward()
        total = float(torch.nn.utils.clip_grad_norm_(params, 5.0))
        gsum_err = float((flat([p_.grad for p_ in params]) - flat([sum(gk[k][i] for k in gk) for i in range(len(params))]) * min(1., 5. / (total + 1e-6))).norm()
                         / flat([p_.grad for p_ in params]).norm())
        opt.step()
        delta = [p_.detach() - th for p_, th in zip(params, theta)]
        clip = min(1., 5. / (total + 1e-6))
        parts = {}
        for k in PATHS:
            for i, p_ in enumerate(params):
                M[k][i] = b1 * M[k][i] + ((1 - b1) * clip * gk[k][i] if k != 'history' else 0.)
        for i, p_ in enumerate(params):
            s_ = opt.state[p_]
            step = float(s_['step'])
            den = (s_['exp_avg_sq'].sqrt() / (1 - b2 ** step) ** 0.5) + eps_
            for k in PATHS:
                parts.setdefault(k, []).append(-lr / (1 - b1 ** step) * M[k][i] / den)
        recon = [sum(parts[k][i] for k in PATHS) for i in range(len(params))]
        recon_err = float((flat(recon) - flat(delta)).norm() / flat(delta).norm())
        dirs = [[(parts[k][i] if gname[i] == g_ else None) for i in range(len(params))] for k in PATHS for g_ in groups]
        enc['image'].eval(); enc['sequence'].eval()
        res = {}
        for label, ss in (('first', (0.,)), ('path', GAUSS)):
            acc = None
            for s in ss:
                with torch.no_grad():
                    for p_, th, dl in zip(params, theta, delta):
                        p_.copy_(th + s * dl)
                _, J, _ = jvps(enc, model, params, fx, dirs)
                acc = J if acc is None else acc + J
            res[label] = (acc / len(ss)).reshape(len(PATHS), len(groups), -1)
        with torch.no_grad():
            for p_, th, dl in zip(params, theta, delta):
                p_.copy_(th + dl)
            o_new = outputs(enc, model, fx)[0].detach()
        enc['image'].train(); enc['sequence'].train()
        actual = o_new - o_prev
        row = {'losses': {k: float(v.detach()) for k, v in L.items()}, 'component_loss_value_minus_total': value - float(loss.detach()),
               'gradient_sum_relative_error': gsum_err, 'displacement_reconstruction_relative_error': recon_err, 'clip': clip,
               'relative_error': {}}
        for label in ('first', 'path'):
            est = res[label].sum((0, 1))
            row['relative_error'][label] = {}
            for name, (pos, n) in layout.items():
                sl_c, sl_w = slice(pos, pos + n), slice(pos + n, pos + 2 * n)
                e = lambda sl: float((est[sl] - actual[sl]).norm() / (actual[sl].norm() + 1e-12))
                em = float(((est[sl_c] - est[sl_w]) - (actual[sl_c] - actual[sl_w])).norm() / ((actual[sl_c] - actual[sl_w]).norm() + 1e-12))
                row['relative_error'][label][name] = {'correct': e(sl_c), 'max_wrong': e(sl_w), 'margin': em}
            att[label].append(res[label].cpu().numpy())
        checks.append(row)
        o_prev = o_new
        traj.append(o_new.cpu().numpy())
    return np.stack(traj), {k: np.stack(v) for k, v in att.items()}, checks, layout, groups, fx


def summarize(traj, att, layout, groups, fx):
    """traj (21, n_out); att[label] (20, paths, groups, n_out)."""
    out = {}
    for name, (pos, n) in layout.items():
        c = traj[:, pos:pos + n]; w = traj[:, pos + n:pos + 2 * n]
        m = c - w
        ok = fx.mem[name.split()[1]]['ok'] if name.startswith('memory') else np.ones(n, bool)  # retrieval outcomes: abstentions are wrong
        r = {'accuracy': [float(((m[t] > 0) & ok).mean()) for t in range(len(m))], 'queries': n, 'cumulative': {}}
        if name.startswith('memory'):
            r['retrieval_ok'] = ok.tolist()
        for T_ in (10, 20):
            act = {'correct': c[T_] - c[0], 'max_wrong': w[T_] - w[0], 'margin': m[T_] - m[0]}
            sets = {'lost': ok & (m[0] > 0) & (m[T_] <= 0), 'kept': ok & (m[0] > 0) & (m[T_] > 0), 'all': ok}
            cell = {}
            for label in ('first', 'path'):
                A = att[label][:T_]
                ac = A[..., pos:pos + n].sum(0); aw = A[..., pos + n:pos + 2 * n].sum(0)  # (paths, groups, n)
                est = {'correct': ac, 'max_wrong': aw, 'margin': ac - aw}
                cell[label] = {}
                for sname, sel in sets.items():
                    if not sel.any():
                        continue
                    cell[label][sname] = {q: {'actual_mean': float(act[q][sel].mean()), 'estimated_mean': float(est[q][..., sel].sum((0, 1)).mean()),
                                              'relative_error': float(np.linalg.norm(est[q][..., sel].sum((0, 1)) - act[q][sel]) / (np.linalg.norm(act[q][sel]) + 1e-12)),
                                              'by_path_and_group': est[q][..., sel].mean(-1).tolist()}
                                          for q in ('correct', 'max_wrong', 'margin')}
                    cell[label][sname]['n'] = int(sel.sum())
            r['cumulative'][str(T_)] = cell
        out[name] = r
    return out


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
    b0 = time.perf_counter()
    traj, att, checks, layout, groups, fx = instrumented(box, reads)
    out.update({'paths': PATHS, 'groups': groups, 'per_update_checks': checks, 'reads': summarize(traj, att, layout, groups, fx),
                'instrumented_seconds': time.perf_counter() - b0})
    np.savez_compressed(RUNS / f'margins-seed{seed}.npz', traj=traj, first=att['first'].astype(np.float32), path=att['path'].astype(np.float32))
    rt = {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}
    print('seed', seed, f'{rt["wall_seconds"]:.0f} s', flush=True)
    return seed, out, rt


def main():
    t0 = time.perf_counter()
    S0.prepare()
    LM.install()
    RUNS.mkdir(parents=True, exist_ok=True)
    ctx = __import__('multiprocessing').get_context('fork')
    with ctx.Pool(int(os.environ.get('RB_WORKERS', 3))) as pool:
        res = pool.map(task, RB.SEEDS, chunksize=1)
    S0.save_json(f'{REPORT}.json', {'status': 'margin decomposition of the first 20 relearning updates; diagnostic, no new learning policy', 'seeds': RB.SEEDS,
                                    'steps': STEPS, 'paths': PATHS, 'gauss_points': GAUSS, 'results': {f'seed{s}': r for s, r, _ in res}})
    S0.save_json(f'{REPORT}-runtime.json', S0.runtime({'phase': 'relearning margins', 'wall_seconds': time.perf_counter() - t0,
                                                      'tasks': {f'seed{s}': t for s, _, t in res}, 'aggregate_worker_cpu_seconds': float(sum(t['cpu_seconds'] for _, _, t in res))}))


if __name__ == '__main__':
    main()
