"""Number construction v1, Part A pilot: counting episodes grounded in collections (docs/number-construction-v1-spec.md,
"Part A protocol").

Languages L1-L3 are taught in blocks by counting episodes over persistent dot collections; two counts per later
language are never directly paired with a collection. Arms share the same evidence and differ in presentation
(episodes, order, shuffled), persistent state between steps, core-as-teacher distillation and language scheduling.
Evaluation: isolated collection matching (acquisition), relational context, succession, reverse direction,
retention, confidence signals and transfer learning cost.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_part_a pilot
"""
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments.number_construction_b1_ambiguous import total_energy

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
SPEC = ROOT / 'docs/number-construction-v1-spec.md'
SEED = int(os.environ.get('NCA_SEED', '100'))
NAME = f'number-construction-v1-part-a-{X.MODE}' + (f'-seed{SEED}' if SEED != 100 else '') + ('-smoke' if X.SMOKE else '')
PREFIX, RUNS = ROOT / 'reports' / NAME, ROOT / 'runs' / NAME
LANGS, TRANSFER = ('english', 'chinese', 'devanagari_digits'), 'thai_digits'
ARMS = {'A-episode': ('episodes', True, False, 'blocked'), 'A-order': ('episodes', False, False, 'blocked'),
        'A-shuffled': ('shuffled', False, False, 'blocked'), 'A-episode+teach': ('episodes', True, True, 'blocked'),
        'A-episode-interleaved': ('episodes', True, False, 'interleaved')}  # (presentation, persistent, teach, schedule)
STEPS = 20 if X.SMOKE else int(os.environ.get('NCA_STEPS', '2000' if X.MODE == 'pilot2' else '1000'))  # updates per block
BATCH_EP, MU, LR, DESC = 16, 1.0, 1e-3, .3
N_DOT_EPISODES = 200 if X.SMOKE else 3000
EVAL_PER = 4 if X.SMOKE else 30  # evaluation forms per count per language
TRANSFER_SHOTS, TRANSFER_STEPS, TRANSFER_AT = 4, (10 if X.SMOKE else 200), ((0, 10) if X.SMOKE else (0, 25, 50, 100, 200))
SIZES = net.SIZES
INNER = slice(SIZES[1] + SIZES[2] + SIZES[0] + SIZES[1], SIZES[1] + SIZES[2] + SIZES[0] + SIZES[1] + SIZES[2])
DOTS, EVAL = {}, {}


# ---------------------------------------------------------------- data

def dot_episode(j):
    # pilot 2: half the episodes keep total area fixed, so ink does not give the count in training either
    mode = 'area' if X.MODE == 'pilot2' and j % 2 else 'radius'
    return np.stack(ns.render_dots_incremental(nc.stream('partA_dots', 'train', j), mode=mode))


def withheld_counts():
    rng = nc.stream('partA_withheld', SEED)
    pairs = [(a, b) for a in range(2, 9) for b in range(a + 2, 9)]
    while True:
        w2 = pairs[rng.integers(len(pairs))]
        w3 = pairs[rng.integers(len(pairs))]
        if not set(w2) & set(w3):
            return {LANGS[0]: (), LANGS[1]: w2, LANGS[2]: w3}


def eval_item(args):
    lang, n, i = args
    order = nc.stream('partA_order', lang, n, i).permutation(9)
    cand = np.stack([ns.render_dots(int(k) + 1, nc.stream('partA_cand', lang, n, i, int(k)), mode='area') for k in order])
    side = 1 if n <= 2 else -1 if n >= 8 else (1 if nc.stream('partA_side', lang, n, i).random() < .5 else -1)
    ctx = {s: ns.render_dots(n + s, nc.stream('partA_ctx', lang, n, i, 'prev' if s < 0 else 'next'), mode='area') if 1 <= n + s <= 9 else None
           for s in (-1, 1)}
    return (lang, n, i), cand, order + 1, side, ctx


def prepare():
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        DOTS['train'] = np.stack(pool.map(dot_episode, range(N_DOT_EPISODES), chunksize=32))
        jobs = [(lang, n, i) for lang in (*LANGS, TRANSFER) for n in ns.NUMBERS for i in range(EVAL_PER)]
        for key, cand, counts, side, ctx in pool.imap_unordered(eval_item, jobs, chunksize=32):
            EVAL[key] = (cand, counts, side, ctx)
    for lang in (*LANGS, TRANSFER):
        y = X.DATA[(lang, 'eval')][1]
        for n in ns.NUMBERS:
            EVAL[(lang, n, 'idx')] = nc.stream('partA_eval_forms', lang, n).permutation(np.flatnonzero(y == n))[:EVAL_PER]
    # succession and reverse-direction test material
    DOTS['succ'] = {n: np.stack([ns.render_dots(n, nc.stream('partA_succ_dots', n, j), mode='area') for j in range(EVAL_PER)]) for n in ns.NUMBERS}


def episodes(lang, withheld, rng, n_ep, pool_idx=None):
    """n_ep counting episodes in one language: lists of steps (count, dot episode index or -1 if hidden, form index, descending)."""
    y = X.DATA[(lang, 'train')][1]
    forms = {n: np.flatnonzero(y == n) if pool_idx is None else pool_idx[n] for n in ns.NUMBERS}
    out = []
    for _ in range(n_ep):
        length = int(rng.integers(3, 10))
        start = int(rng.integers(1, 11 - length))
        counts = list(range(start, start + length))
        desc = rng.random() < DESC
        if desc:
            counts = counts[::-1]
        e = int(rng.integers(len(DOTS['train'])))
        out.append({'lang': lang, 'desc': desc, 'steps': [(n, -1 if n in withheld else e, int(rng.choice(forms[n]))) for n in counts]})
    return out


def schedule(withheld):
    """Episode batches per block, identical across arms (blocked order); interleaving and shuffling rearrange them."""
    blocks = []
    for b, lang in enumerate(LANGS):
        rng = nc.stream('partA_schedule', SEED, lang)
        blocks.append([episodes(lang, withheld[lang], rng, BATCH_EP) for _ in range(STEPS)])
    return blocks


def batch_material(eps):
    """Items (form or collection images with counts) and relation pairs for a list of episodes."""
    items, pairs, match = [], [], []
    for ep in eps:
        prev = None
        for n, e, fi in ep['steps']:
            fidx = len(items); items.append(('form', ep['lang'], n, fi))
            cidx = None
            if e >= 0:
                cidx = len(items); items.append(('coll', None, n, e))
                match.append((fidx, cidx, n))
            if prev is not None:
                pf, pc = prev
                a, b = (pf, fidx) if not ep['desc'] else (fidx, pf)
                pairs.append((a, b))
                if cidx is not None and pc is not None:
                    pairs.append((pc, cidx) if not ep['desc'] else (cidx, pc))
            prev = (fidx, cidx)
    return items, pairs, match


def images(items):
    x = np.empty((len(items), 28, 28), np.float32)
    for k, (kind, lang, n, i) in enumerate(items):
        x[k] = X.DATA[(lang, 'train')][0][i] if kind == 'form' else DOTS['train'][i, n - 1]
    return x


# ---------------------------------------------------------------- model pieces

class Coupled:
    """Differentiable solves for the image medium, with optional successor coupling to a fixed previous state."""

    def __init__(self, sri):
        Q = sri.hessian('image')
        o = SIZES[0]
        self.Qyy, self.Qyx = Q[o:, o:], Q[o:, :o]
        self.sel = torch.zeros(self.Qyy.shape[0], SIZES[2]); self.sel[INNER] = torch.eye(SIZES[2])
        self.A, self.t, self.L = sri.A, sri.t, {}

    def factor(self, kind):
        if kind not in self.L:
            K = self.Qyy if kind == 'none' else self.Qyy + MU * self.sel @ (torch.eye(SIZES[2]) if kind == 'asc' else self.A.T @ self.A) @ self.sel.T
            self.L[kind] = torch.linalg.cholesky(K)
        return self.L[kind]

    def solve(self, f, kind='none', prev=None):
        rhs = -(f @ self.Qyx.T)
        if kind == 'asc':
            rhs = rhs + MU * (prev @ self.A.T + self.t) @ self.sel.T
        elif kind == 'desc':
            rhs = rhs + MU * ((prev - self.t) @ self.A) @ self.sel.T
        return torch.cholesky_solve(rhs.T, self.factor(kind)).T


def same_logits(sri, a, b):
    return sri.relation_logits(a, b, 1)


def train_update(enc, sri, opt, eps_or_material, persistent, teach, rng):
    if isinstance(eps_or_material, tuple):
        items, pairs, match, eps = (*eps_or_material, None)
    else:
        eps = eps_or_material
        items, pairs, match = batch_material(eps)
    f = enc['image'](torch.from_numpy(images(items)))
    solver = Coupled(sri)
    y_iso = solver.solve(f)
    c = y_iso[:, INNER]
    c_used = c
    loss_teach = torch.zeros(())
    if persistent:
        # forms solved in episode order with coupling to the previous step's state (detached). Pilot 1: the previous
        # form's episode state (compounds along the episode). Pilot 2: the previous step's collection when visible,
        # otherwise the previous form's episode state (chains of at most two links).
        c_ep = c.clone()
        pos = 0
        form_rows = []
        for ep in eps:
            prev_state = None
            for n, e, fi in ep['steps']:
                fidx = pos; pos += 1 + (e >= 0)
                if prev_state is not None:
                    kind = 'desc' if ep['desc'] else 'asc'
                    form_rows.append((fidx, kind, prev_state))
                    c_ep_f = solver.solve(f[fidx:fidx + 1], kind, prev_state)[:, INNER]
                    c_ep = torch.cat([c_ep[:fidx], c_ep_f, c_ep[fidx + 1:]])
                grounded = X.MODE == 'pilot2' and e >= 0
                prev_state = (c[fidx + 1:fidx + 2] if grounded else c_ep[fidx:fidx + 1]).detach()
        c_used = c_ep
        if teach:
            idx = [r[0] for r in form_rows]
            loss_teach = (c[idx] - c_ep[idx].detach()).pow(2).sum(1).mean()
    counts = torch.tensor([it[2] for it in items])
    # matching: candidates are one collection per count (shared in the batch), the step's own at its count
    ci = rng.integers(len(DOTS['train']), size=9)
    cand_x = np.stack([DOTS['train'][ci[k], k] for k in range(9)])
    cand = solver.solve(enc['image'](torch.from_numpy(cand_x)))[:, INNER]
    loss_match = torch.zeros(())
    if match:
        fi = torch.tensor([m[0] for m in match]); cj = torch.tensor([m[1] for m in match]); nn_ = torch.tensor([m[2] for m in match]) - 1
        C = cand[None].expand(len(match), -1, -1).clone()
        C[torch.arange(len(match)), nn_] = c_used[cj]
        logits = same_logits(sri, c_used[fi][:, None].expand_as(C).reshape(-1, SIZES[2]), C.reshape(-1, SIZES[2])).reshape(len(match), 9)
        loss_match = F.cross_entropy(logits, nn_)
    loss_next = torch.zeros(())
    if pairs:
        a = torch.tensor([p[0] for p in pairs]); b = torch.tensor([p[1] for p in pairs])
        kinds = [items[p[0]][0] for p in pairs]
        # negatives: same-kind pairs whose counts do not differ by +1
        na, nb = [], []
        by_kind = {k: [i for i, it in enumerate(items) if it[0] == k] for k in ('form', 'coll')}
        for k in kinds:
            pool = by_kind[k]
            for _ in range(20):
                u, v = rng.choice(pool, 2)
                if items[v][2] != items[u][2] + 1:
                    na.append(u); nb.append(v)
                    break
        a2 = torch.cat([a, torch.tensor(na, dtype=torch.long)]); b2 = torch.cat([b, torch.tensor(nb, dtype=torch.long)])
        target = torch.cat([torch.ones(len(a)), torch.zeros(len(na))])
        loss_next = F.binary_cross_entropy_with_logits(sri.relation_logits(c_used[a2], c_used[b2], 0), target)
    loss = loss_match + loss_next + loss_teach
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('part A update')
    opt.zero_grad()
    loss.backward()
    if X.MODE == 'pilot2':
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    return [round(float(v), 4) for v in (loss_match, loss_next, loss_teach)]


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(enc, sri, withheld, langs=LANGS, full=False):
    enc['image'].eval()
    sv = B.Solver(sri, 'image')
    st = lambda x: sri.split(sv.solve(enc['image'](torch.from_numpy(np.ascontiguousarray(x)))))
    out = {'isolated': {}}
    rows = []
    for lang in langs:
        for n in ns.NUMBERS:
            x = X.DATA[(lang, 'eval')][0][EVAL[(lang, n, 'idx')]]
            f = enc['image'](torch.from_numpy(x)); y = sv.solve(f); c = sri.split(y)['core_inner']
            cand = np.stack([EVAL[(lang, n, i)][0] for i in range(EVAL_PER)])
            counts = np.stack([EVAL[(lang, n, i)][1] for i in range(EVAL_PER)])
            cc = st(cand.reshape(-1, 28, 28))['core_inner'].reshape(EVAL_PER, 9, -1)
            logit = same_logits(sri, c[:, None].expand_as(cc).reshape(-1, SIZES[2]), cc.reshape(-1, SIZES[2])).reshape(EVAL_PER, 9)
            top = logit.topk(2, 1)
            pick = counts[np.arange(EVAL_PER), top.indices[:, 0].numpy()]
            rows.append({'lang': lang, 'n': n, 'withheld': n in withheld.get(lang, ()), 'f': f, 'y': y, 'c': c, 'cc': cc, 'counts': counts,
                         'pick': pick, 'margin': (top.values[:, 0] - top.values[:, 1]).numpy(), 'best': top.values[:, 0].numpy(),
                         'energy': (sv.energy(f, y) / f.norm(dim=1) ** 2).numpy(), 'fnorm': f.norm(dim=1).numpy()})
    for lang in langs:
        for grp in ('paired', 'withheld'):
            rs = [r for r in rows if r['lang'] == lang and r['withheld'] == (grp == 'withheld')]
            if rs:
                out['isolated'][f'{lang}/{grp}'] = float(np.mean(np.concatenate([r['pick'] == r['n'] for r in rs])))
    if not full:
        enc['image'].train()
        return out
    A, t = sri.A, sri.t
    # relational context for every form (reported for withheld and paired): two-sided and one-sided, mu = MU
    rel = {'two_sided': [], 'swap': [], 'restricted': [], 'side': [], 'oracle_one_sided': []}
    for r in rows:
        lang, n = r['lang'], r['n']
        ctx = {s: np.stack([EVAL[(lang, n, i)][3][s] for i in range(EVAL_PER)]) if 1 <= n + s <= 9 else None for s in (-1, 1)}
        cs = {s: st(v)['core_inner'] if v is not None else None for s, v in ctx.items()}
        P = cs[-1] if cs[-1] is not None else [None] * EVAL_PER
        N = cs[1] if cs[1] is not None else [None] * EVAL_PER
        yr = sv.revise(r['f'], MU, A, t, P, N, 'eval')
        def pick9(cv):
            lg = same_logits(sri, cv[:, None].expand_as(r['cc']).reshape(-1, SIZES[2]), r['cc'].reshape(-1, SIZES[2])).reshape(EVAL_PER, 9)
            return r['counts'][np.arange(EVAL_PER), lg.argmax(1).numpy()]
        r['revised'] = pick9(sri.split(yr)['core_inner'])
        other = [k for k in ns.NUMBERS if k != n]
        n_swap = int(nc.stream('partA_swap', lang, n).choice(other))
        fs = enc['image'](torch.from_numpy(X.DATA[(lang, 'eval')][0][EVAL[(lang, n_swap, 'idx')]]))
        r['swap'] = pick9(sri.split(sv.revise(fs, MU, A, t, P, N, 'eval'))['core_inner'])
        r['n_swap'] = n_swap
        # one-sided, side unknown
        sides = np.array([EVAL[(lang, n, i)][2] for i in range(EVAL_PER)])
        m = n + sides
        cm = torch.stack([cs[int(s)][i] for i, s in enumerate(sides)])
        lg = same_logits(sri, r['c'][:, None].expand_as(r['cc']).reshape(-1, SIZES[2]), r['cc'].reshape(-1, SIZES[2])).reshape(EVAL_PER, 9).numpy()
        lo = lg[np.arange(EVAL_PER), [list(row).index(k) for row, k in zip(r['counts'], m - 1)]]
        hi = lg[np.arange(EVAL_PER), [list(row).index(k) for row, k in zip(r['counts'], m + 1)]]
        r['restricted'] = np.where(hi > lo, m + 1, m - 1)
        yu = sv.revise(r['f'], MU, A, t, cm, [None] * EVAL_PER, 'eval_up')
        yd = sv.revise(r['f'], MU, A, t, [None] * EVAL_PER, cm, 'eval_down')
        eu, ed = total_energy(sv, r['f'], yu, MU, A, t, cm, 1), total_energy(sv, r['f'], yd, MU, A, t, cm, -1)
        r['side'] = np.where((eu <= ed).numpy(), m + 1, m - 1)
    for grp in ('paired', 'withheld'):
        rs = [r for r in rows if r['withheld'] == (grp == 'withheld') and r['lang'] != LANGS[0]] if grp == 'withheld' else \
             [r for r in rows if not r['withheld'] and 2 <= r['n'] <= 8]
        if not rs:
            continue
        cat = lambda k: np.concatenate([r[k] for r in rs])
        num = np.concatenate([[r['n']] * EVAL_PER for r in rs]); nsw = np.concatenate([[r['n_swap']] * EVAL_PER for r in rs])
        out[f'context/{grp}'] = {'isolated': float((cat('pick') == num).mean()), 'two_sided_revised': float((cat('revised') == num).mean()),
                                 'two_sided_swap_follows_context': float((cat('swap') == num).mean()),
                                 'two_sided_swap_follows_item': float((cat('swap') == nsw).mean()),
                                 'one_sided_restricted': float((cat('restricted') == num).mean()),
                                 'one_sided_energy_side': float((cat('side') == num).mean())}
    # confidence hooks over L1-L3 forms
    num = np.concatenate([[r['n']] * EVAL_PER for r in rows])
    wh = np.concatenate([[r['withheld']] * EVAL_PER for r in rows])
    err = np.concatenate([r['pick'] for r in rows]) != num
    agree = np.concatenate([r['pick'] == r['revised'] for r in rows])
    sig = {'margin': np.concatenate([r['margin'] for r in rows]), 'best': np.concatenate([r['best'] for r in rows]),
           'energy': np.concatenate([r['energy'] for r in rows]), 'fnorm': np.concatenate([r['fnorm'] for r in rows])}
    out['confidence'] = {'error_rate': float(err.mean()),
                         'auroc_error': {k: B.auroc(-v if k in ('margin', 'best') else v, err) for k, v in sig.items()},
                         'auroc_withheld': {k: B.auroc(-v if k in ('margin', 'best') else v, wh) for k, v in sig.items()},
                         'agreement_rate': float(agree.mean()),
                         'accuracy_when_agree': float((~err)[agree].mean()) if agree.any() else None,
                         'accuracy_when_disagree': float((~err)[~agree].mean()) if (~agree).any() else None,
                         'auroc_error_from_disagreement': B.auroc((~agree).astype(float), err)}
    # succession on held-out collections and forms: next(n, n+1) scored above next(n, n+k), k != 1
    succ = {}
    for name, get in (('collections', lambda n: DOTS['succ'][n]),) + tuple((lang, (lambda L: lambda n: X.DATA[(L, 'eval')][0][EVAL[(L, n, 'idx')]])(lang)) for lang in langs):
        codes = {n: st(get(n))['core_inner'] for n in ns.NUMBERS}
        wins = []
        for n in range(1, 9):
            pos_l = sri.relation_logits(codes[n], codes[n + 1], 0)
            for k in ns.NUMBERS:
                if k != n + 1 and k != n:
                    wins.append((pos_l > sri.relation_logits(codes[n], codes[k], 0)).float().mean().item())
        succ[name] = float(np.mean(wins))
    out['succession_pairwise'] = succ
    # reverse direction: collection -> one of the language's nine forms
    rev = {}
    for lang in langs:
        forms = st(np.stack([X.DATA[(lang, 'eval')][0][EVAL[(lang, k, 'idx')][0]] for k in ns.NUMBERS]))['core_inner']
        hits = []
        for n in ns.NUMBERS:
            cd = st(DOTS['succ'][n])['core_inner']
            lg = same_logits(sri, cd[:, None].expand(-1, 9, -1).reshape(-1, SIZES[2]), forms[None].expand(len(cd), -1, -1).reshape(-1, SIZES[2])).reshape(len(cd), 9)
            hits.append((lg.argmax(1) == n - 1).float().numpy())
        rev[lang] = float(np.concatenate(hits).mean())
    out['reverse_direction'] = rev
    enc['image'].train()
    return out


# ---------------------------------------------------------------- one arm

def run(arm):
    t0, c0 = time.perf_counter(), time.process_time()
    presentation, persistent, teach, sched = ARMS[arm]
    withheld = withheld_counts()
    torch.manual_seed(SEED)
    enc, sri = X.make_model(SEED)
    params = [*enc['image'].parameters(), *sri.parameters()]
    opt = torch.optim.Adam(params, LR)
    blocks = schedule(withheld)
    if sched == 'interleaved':
        all_eps = [ep for blk in blocks for batch in blk for ep in batch]
        perm = nc.stream('partA_interleave', SEED).permutation(len(all_eps))
        flat = [all_eps[i] for i in perm]
        blocks = [[flat[(b * STEPS + s) * BATCH_EP:(b * STEPS + s + 1) * BATCH_EP] for s in range(STEPS)] for b in range(3)]
    rng = nc.stream('partA_train', SEED, arm)
    losses, block_eval, cpu = [], [], {'train': 0., 'eval': 0.}
    for b, blk in enumerate(blocks):
        batches = blk
        if presentation == 'shuffled':
            mats = [batch_material(batch) for batch in blk]
            # rebuild shuffled batches: the same matching steps and pairs, mixed across the block
            match_all = [(m[0][a], m[0][c], n) for m in mats for a, c, n in m[2]]
            pairs_all = [(m[0][a], m[0][bb]) for m in mats for a, bb in m[1]]
            pm, pp = rng.permutation(len(match_all)), rng.permutation(len(pairs_all))
            batches, im, ip = [], 0, 0
            for m in mats:
                k_m, k_p = len(m[2]), len(m[1])
                items, match, pairs = [], [], []
                for j in pm[im:im + k_m]:
                    fi_, ci_, n = match_all[j]
                    match.append((len(items), len(items) + 1, n)); items += [fi_, ci_]
                for j in pp[ip:ip + k_p]:
                    a_, b_ = pairs_all[j]
                    pairs.append((len(items), len(items) + 1)); items += [a_, b_]
                im += k_m; ip += k_p
                batches.append((items, pairs, match))
        e0 = time.process_time()
        for step, batch in enumerate(batches):
            ls = train_update(enc, sri, opt, batch, persistent and presentation == 'episodes', teach, rng)
            if step % 100 == 99:
                losses.append([b, step + 1, *ls])
        cpu['train'] += time.process_time() - e0
        e0 = time.process_time()
        block_eval.append(evaluate(enc, sri, withheld, full=X.MODE == 'pilot2'))
        cpu['eval'] += time.process_time() - e0
    e0 = time.process_time()
    final = evaluate(enc, sri, withheld, full=True)
    cpu['eval'] += time.process_time() - e0
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f'{arm}.pt'
    torch.save({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': sri.state_dict(),
                'sources': list(LANGS), 'labelled': {l: [n for n in ns.NUMBERS if n not in withheld[l]] for l in LANGS}}, path)
    # transfer: thai_digits, 4 examples per count, all counts paired, same procedure for every arm
    e0 = time.process_time()
    y = X.DATA[(TRANSFER, 'train')][1]
    shots = {n: nc.stream('partA_transfer_shots', n).permutation(np.flatnonzero(y == n))[:TRANSFER_SHOTS] for n in ns.NUMBERS}
    trng = nc.stream('partA_transfer', SEED)
    topt = torch.optim.Adam(params, LR)
    transfer = {}
    for step in range(TRANSFER_STEPS + 1):
        if step in TRANSFER_AT:
            ev = evaluate(enc, sri, {}, langs=(TRANSFER,))
            transfer[str(step)] = ev['isolated'][f'{TRANSFER}/paired']
        if step == TRANSFER_STEPS:
            break
        train_update(enc, sri, topt, episodes(TRANSFER, (), trng, BATCH_EP, shots), False, False, trng)
    transfer['retention_after'] = evaluate(enc, sri, withheld)['isolated']
    cpu['transfer'] = time.process_time() - e0
    return arm, {'withheld': {k: list(v) for k, v in withheld.items()}, 'loss_every_100': losses, 'after_block': block_eval,
                 'final': final, 'transfer': transfer, 'checkpoint': {'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)}}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **{f'{k}_cpu_seconds': v for k, v in cpu.items()},
        'updates': 3 * STEPS + TRANSFER_STEPS, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    t = time.perf_counter()
    prepare()
    phases = {'data_wall_seconds': t - wall0, 'render_wall_seconds': time.perf_counter() - t}
    t = time.perf_counter()
    done = []
    with mp.get_context('fork').Pool(min(X.WORKERS, len(ARMS))) as pool:
        for item in pool.imap_unordered(run, list(ARMS)):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(ARMS)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['arms_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(SPEC.relative_to(ROOT)), 'status': 'Part A pilot: not headline evidence',
           'config': {'seed': SEED, 'langs': LANGS, 'transfer': TRANSFER, 'arms': ARMS, 'steps_per_block': STEPS, 'batch_episodes': BATCH_EP,
                      'mu': MU, 'lr': LR, 'descending': DESC, 'dot_episodes': N_DOT_EPISODES, 'eval_per_count': EVAL_PER,
                      'transfer_shots': TRANSFER_SHOTS, 'transfer_steps': TRANSFER_STEPS, 'sizes': SIZES},
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__), 'b1_sha256': nc.sha256(B.__file__),
        'experiment_sha256': nc.sha256(X.__file__), 'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]),
        'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith(('NSY_', 'NCA_'))},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_parent_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
