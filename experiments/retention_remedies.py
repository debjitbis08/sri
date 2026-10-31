"""Retention remedies v1 (docs/retention-remedies-v1-contract.md): can arbitrary facts marked for preservation survive 2,000 updates of other
material, at what cost to transfer, new learning, compute and storage?

Per arm and seed: blocks 1-4 as in the forgetting-mechanism runner (Option A), the end-of-block-4 state frozen as the common start, a marked set M
(two of the five symbols per count) and the rest U. Host branches run 2,000 context updates from the start:
  ctx / ctx-m0               context episodes, without and with the first-moment reset
  hold-M                     context + M pair items every update (full rendering pool; unbounded reference)
  rehearse-M / rehearse-M-m0 context + K stored renderings per M pair, BUFFER_BATCH buffer items every REHEARSE_EVERY-th update
  der-M                      rehearse-M plus a penalty keeping the nine canonical-candidate scores of buffered renderings near their stored values
  stable-path-replay         image encoder and model.proj['image'] frozen; stored item codes train the reader (cross-entropy over the candidates)
each followed by NEW_UPDATES updates teaching the never-seen set B. Static branches need no context training (the destination is never updated
by it): frozen-snapshot (the end-of-block-4 pathway) and consolidation into a head on the frozen image-to-code path, with cues from a stored
cache (K = 4 or 1 codes per M pair), random codes, or renderings of symbols outside A and B, and with a nine-way or an anchored (code-space) head.
Cues and teacher outputs are deleted after distillation; the destination is evaluated from its weights and the frozen path alone.

Usage: NFI_DEVICE=cuda NSY_WORKERS=6 /data/venvs/sri-cuda/bin/python -m experiments.retention_remedies pilot pairs
(NSY_SMOKE=1 for a short CPU run; NAP_ARMS and NAP_SEEDS override the defaults below; hyperparameters via RR_* variables for the pilot.)
"""
import os

SMOKE = os.environ.get('NSY_SMOKE') == '1'
for key, value in (('NAP_OPTION', 'A'), ('NAP_ARMS', 'sparse48,mixed,mixed-h125'), ('NAP_SEEDS', ','.join(map(str, range(350, 360))))) + \
        (() if SMOKE else (('NAP_N', '45'), ('NAP_L', '2000'))):
    os.environ.setdefault(key, value)

import copy
import hashlib
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments import arbitrary_pairings as AP
from experiments import forgetting_mechanism as FM
from experiments import generic_relations as G
from experiments import limited_memory as LM
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import number_systems as X
from experiments import pairing_symbols as PS
from experiments import sequence_pilot as SP
from experiments import transition_pilot as TP

nc = X.nc
CURRENT = LM.CURRENT
L, N, R = AP.L, AP.N, AP.TRAIN_RENDERS
K = int(os.environ.get('RR_K', '4'))
PER_COUNT = max(1, min(2, N // 9 - 1))  # marked pairs per count (2 at N = 45); needs N >= 18 so that U is not empty
REHEARSE_EVERY, BUFFER_BATCH = int(os.environ.get('RR_EVERY', '10')), int(os.environ.get('RR_BATCH', '16'))
DER_ALPHA = float(os.environ.get('RR_DER_ALPHA', '0.01'))  # the pilot's rule-selected weight; `der-M-strong` uses 0.1
HEAD_H, DISTILL_STEPS = int(os.environ.get('RR_HEAD', '64')), int(os.environ.get('RR_STEPS', '1000'))
DISTILL_LR, DISTILL_WD = float(os.environ.get('RR_LR', '0.01')), float(os.environ.get('RR_WD', '0.0'))
HP = {False: (HEAD_H, DISTILL_STEPS, DISTILL_WD), True: (32, 3000, 0.001)}  # pilot selections (seed 299): nine-way head, anchored head  # (head width, steps, weight decay) for the nine-way and the anchored head
NEW_UPDATES = 250 if not SMOKE else 10
POINTS = FM.points(L, (0, 50, 250, 500, 1000, 2000))
NAME = 'retention-remedies-v1' + ('-smoke' if SMOKE else '') + os.environ.get('NFI_TAG', '')
RUNS = X.ROOT / 'runs' / NAME

HOST = {  # name -> spec; `arms` limits where the branch exists
    'ctx': dict(),
    'ctx-m0': dict(adam='m0', arms=('mixed', 'mixed-h125')),
    'hold-M': dict(pairs='hold'),
    'rehearse-M': dict(pairs='buffer'),
    'rehearse-M-m0': dict(pairs='buffer', adam='m0', arms=('mixed',)),
    'der-M': dict(pairs='buffer', der=True, alpha=0.01),
    'der-M-strong': dict(pairs='buffer', der=True, alpha=0.1),
    'stable-path-replay': dict(freeze_path=True, stored_codes=True, arms=('mixed',)),
}
STATE = {}  # the active replay state read by the extra loss
PIXEL_BYTES = 28 * 28 * 4


# ---------------------------------------------------------------- helpers

def n_bytes(*mods):
    return int(sum(p.numel() for m in mods for p in m.parameters()) * 4)


def digest(params):
    h = hashlib.sha256()
    for p in params:
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def path_params(enc, model):
    return [*enc['image'].parameters(), *model.proj['image'].parameters()]


def marked_set(seed, A):
    rng = nc.stream('rr_marked', seed)
    M = {}
    for n in range(1, 10):
        syms = [cp for cp, c in A.items() if c == n]
        for i in rng.permutation(len(syms))[:PER_COUNT]:
            M[syms[int(i)]] = n
    return M


def canonical_images():
    return np.stack([P.DOTS['succ'][n][0] for n in range(1, 10)])


@torch.no_grad()
def codes(enc, model, x):
    enc['image'].eval()
    out = G.codes_fn(enc, model)(np.ascontiguousarray(x, dtype=np.float32), 'image')
    enc['image'].train()
    return out


def scores(model, q, cand):
    """Nine candidate scores per query code (G.pair, head 1): q (n, d), cand (9, d) -> (n, 9)."""
    return G.pair(model, q, cand[None].expand(len(q), -1, -1), 1)


class Head(torch.nn.Module):
    def __init__(self, d, out, h):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Linear(h, out))

    def forward(self, x):
        return self.net(x)


def distill(cues, soft, anchored, canon, seed, tag):
    """Train a head on (cue code -> teacher response): KL to the teacher's softmax over the nine counts, or, anchored, mean squared error to the
    expected canonical code in the frozen path's code space. Full batch, DISTILL_STEPS Adam steps, weight decay DISTILL_WD."""
    h, steps, wd = HP[anchored]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(nc.stream('rr_head', seed, tag).integers(2 ** 31)))
        head = Head(cues.shape[1], cues.shape[1] if anchored else 9, h)
    opt = torch.optim.Adam(head.parameters(), DISTILL_LR, weight_decay=wd)
    target = soft @ canon if anchored else soft
    for _ in range(steps):
        out = head(cues)
        loss = F.mse_loss(out, target) if anchored else F.kl_div(F.log_softmax(out, 1), target, reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
    return head, float(loss.detach())


@torch.no_grad()
def answer(head, q, canon, anchored):
    out = head(q)
    return (torch.cdist(out, canon).argmin(1) + 1) if anchored else (out.argmax(1) + 1)


# ---------------------------------------------------------------- extra loss (DER penalty, stored-code replay)

def install_extra_loss():
    if STATE.get('wrapped'):
        return
    base = TP.EXTRA_LOSS

    def extra(model, material, c, cand):
        loss = base(model, material, c, cand)
        s = STATE.get('active')
        if s is None:
            return loss
        if s['kind'] == 'der':
            sol = P.Coupled(model)
            cc = sol.solve(s['enc']['image'](torch.from_numpy(s['canon'])))[:, P.INNER]
            q = c[torch.tensor(s['rows'])]
            pen = F.mse_loss(scores(model, q, cc), s['stored'])
            s['pen'].append(float(pen.detach()))
            return loss + s['alpha'] * pen
        q, tgt = s['codes'], s['counts'] - 1  # stored codes against the step's candidate collections
        logits = model.relation_logits(q[:, None].expand(-1, 9, -1).reshape(-1, q.shape[1]), cand[None].expand(len(q), -1, -1).reshape(-1, q.shape[1]), 1)
        l = F.cross_entropy(logits.reshape(len(q), 9), tgt)
        s['pen'].append(float(l.detach()))
        return loss + l

    TP.EXTRA_LOSS = extra
    STATE['wrapped'] = True


def buffer_items(items, match, M, row, rng, buf):
    """BUFFER_BATCH items drawn from the stored renderings of M pairs, appended as pair items; returns (item row, symbol, rendering index)."""
    cps, out = list(M), []
    for _ in range(BUFFER_BATCH):
        cp = cps[int(rng.integers(len(cps)))]
        j = int(buf[cp][int(rng.integers(len(buf[cp])))])
        f = len(items); items.append(('form', 'pairsym', M[cp], row[cp] * R + j))
        c = len(items); items.append(('coll', None, M[cp], int(rng.integers(len(P.DOTS['train'])))))
        match.append((f, c, M[cp])); out.append((f, cp, j))
    return out


# ---------------------------------------------------------------- host branches

def recall_MU(enc, model, ctx, seed):
    return {'M': AP.recall(enc, model, ctx['M'], ctx['ev_M'], seed, 'M')[0], 'U': AP.recall(enc, model, ctx['U'], ctx['ev_U'], seed, 'U')[0]}


def run_branch(name, spec, b4, arm, seed, ctx, withheld):
    t0 = time.perf_counter()
    enc, model, opt, rng = AP.thaw(b4, arm, seed)
    path0 = digest(path_params(enc, model))
    if spec.get('freeze_path'):
        for p in path_params(enc, model):
            p.requires_grad_(False)
    crng, prng = nc.stream('ap_context', seed, 5), nc.stream('ap_pairs', seed, 5, 'rr-' + name)
    adam_check = FM.apply_adam(opt, spec['adam']) if spec.get('adam') else None
    curve, losses, pen = [], [], []
    M, buf = ctx['M'], ctx['buf']
    record = lambda k: curve.append({'offset': k, **recall_MU(enc, model, ctx, seed)})
    record(0)
    for step in range(L):
        items, pairs, match, *rest = SP.batch_material([], D.context_episodes(crng, D.N_CONTEXT), 'taps')
        STATE['active'] = None
        mode = spec.get('pairs')
        if mode == 'hold':
            AP.pair_items(items, match, M, ctx['row'], prng)
        elif (mode == 'buffer' or spec.get('stored_codes')) and (step + 1) % REHEARSE_EVERY == 0:
            if mode == 'buffer':
                drawn = buffer_items(items, match, M, ctx['row'], prng, buf)
                if spec.get('der'):
                    STATE['active'] = dict(kind='der', alpha=spec.get('alpha', DER_ALPHA), enc=enc, canon=ctx['canon_imgs'], pen=pen, rows=[f for f, _, _ in drawn],
                                           stored=torch.stack([ctx['stored_scores'][(cp, j)] for _, cp, j in drawn]))
            else:
                cps = list(M)
                pick = [cps[int(prng.integers(len(cps)))] for _ in range(BUFFER_BATCH)]
                jj = [int(buf[cp][int(prng.integers(len(buf[cp])))]) for cp in pick]
                STATE['active'] = dict(kind='codes', pen=pen, codes=torch.stack([ctx['stored_codes'][(cp, j)] for cp, j in zip(pick, jj)]),
                                       counts=torch.tensor([M[cp] for cp in pick]))
        losses.append(LM.train_update(enc, model, opt, (items, pairs, match, *rest), rng, True))
        if step + 1 in POINTS:
            record(step + 1)
    STATE['active'] = None
    rec = {'curve': curve, 'losses': losses, 'wall_seconds': time.perf_counter() - t0, 'snapshot': LM.snapshot(enc, model, withheld, seed, name, False)}
    if adam_check is not None:
        rec['adam_check'] = adam_check
    if spec.get('freeze_path'):
        rec['path_unchanged'] = digest(path_params(enc, model)) == path0
    if pen:
        rec['replay_penalty_mean'] = float(np.mean(pen))
    end = AP.freeze(enc, model, opt, rng)
    # new learning: NEW_UPDATES updates teaching the never-seen set B (the path stays frozen where the branch froze it)
    enc2, model2, opt2, rng2 = AP.thaw(end, arm, seed)
    if spec.get('freeze_path'):
        for p in path_params(enc2, model2):
            p.requires_grad_(False)
    curveB = []
    AP.block(enc2, model2, opt2, rng2, seed, 6, NEW_UPDATES, ctx['B'], ctx['row'], {'B': (ctx['B'], ctx['ev_B'])}, curveB, 'B')
    rec['new_B'] = curveB
    rec['wall_seconds_total'] = time.perf_counter() - t0
    return rec


# ---------------------------------------------------------------- static branches (teacher = the end-of-block-4 network)

def static_branches(ref_enc, ref_model, ctx, seed, sets):
    """Frozen snapshot and consolidation destinations; cues and teacher outputs exist only inside this function's distillation."""
    out = {}
    M, U, row = ctx['M'], ctx['U'], ctx['row']
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, R, 28, 28)
    canon = codes(ref_enc, ref_model, ctx['canon_imgs'])  # (9, d), the teacher's code of one canonical collection per count
    d = canon.shape[1]
    ev_codes = {s: codes(ref_enc, ref_model, ctx[f'ev_{s}'].reshape(-1, 28, 28)) for s in ('M', 'U')}
    truth = {s: torch.tensor(np.repeat([ctx[s][cp] for cp in ctx[s]], ctx[f'ev_{s}'].shape[1])) for s in ('M', 'U')}
    with torch.no_grad():
        t_pred = {s: scores(ref_model, ev_codes[s], canon).argmax(1) + 1 for s in ('M', 'U')}
    teacher = {'original_M': AP.recall(ref_enc, ref_model, M, ctx['ev_M'], seed, 'M')[0], 'original_U': AP.recall(ref_enc, ref_model, U, ctx['ev_U'], seed, 'U')[0],
               **{f'canonical_{s}': float((t_pred[s] == truth[s]).float().mean()) for s in ('M', 'U')}}
    out['teacher'] = teacher
    path_b = n_bytes(ref_model.proj['image']) + n_bytes(ref_enc['image'])
    out['frozen-snapshot'] = {'recall_M': teacher['original_M'], 'recall_U': teacher['original_U'], 'persistent_bytes': n_bytes(ref_enc['image']) + n_bytes(ref_model)}

    cache = [(cp, j) for cp in M for j in ctx['buf'][cp]]
    cache_imgs = {cp: tr[row[cp]][list(ctx['buf'][cp])] for cp in M}
    cue_cache = codes(ref_enc, ref_model, np.concatenate([cache_imgs[cp] for cp in M]))  # (|M|*K, d)
    first = np.arange(len(M)) * K  # the first stored cue of each M pair
    rng = nc.stream('rr_cues', seed)
    ctx_codes = codes(ref_enc, ref_model, np.stack([P.DOTS['train'][int(i)][int(n)] for i, n in zip(rng.integers(len(P.DOTS['train']), size=500), rng.integers(9, size=500))]))
    mu, sd = ctx_codes.mean(0), ctx_codes.std(0)
    rand = torch.tensor(nc.stream('rr_random', seed).standard_normal((len(cue_cache), d)), dtype=torch.float32) * sd + mu
    unused = [cp for cp in AP.POOL if cp not in sets['A'] and cp not in sets['B']]
    pick = [unused[int(i)] for i in nc.stream('rr_other', seed).permutation(len(unused))[:len(cue_cache)]]
    other = codes(ref_enc, ref_model, np.concatenate([PS.renderings(cp, AP.POOL[cp], seed, 'train', 1) for cp in pick]))
    sources = {'cache': cue_cache, 'cache1': cue_cache[first], 'random': rand, 'other': other}
    for tag, cues in sources.items():
        for anchored in (False, True):
            if anchored and tag not in ('cache', 'cache1'):
                continue
            t0 = time.perf_counter()
            with torch.no_grad():
                soft = F.softmax(scores(ref_model, cues, canon), 1)
            head, final_loss = distill(cues, soft, anchored, canon, seed, f'{tag}-{anchored}')
            fit = float((answer(head, cues, canon, anchored) == (soft.argmax(1) + 1)).float().mean())
            d_hash = digest(head.parameters())
            temp_bytes = int(cues.numel() * 4 + soft.numel() * 4 + canon.numel() * 4)
            soft = None  # teacher outputs are not used again; evaluation below uses only the head and the frozen path
            rec = {'persistent_bytes': n_bytes(head) + path_b, 'head_bytes': n_bytes(head), 'temporary_bytes': temp_bytes, 'final_loss': final_loss,
                   'fit_to_teacher_on_cues': fit, 'hash_after': d_hash, 'wall_seconds': time.perf_counter() - t0}
            for s in ('M', 'U'):
                pred = answer(head, ev_codes[s], canon, anchored)
                rec[f'acc_{s}'] = float((pred == truth[s]).float().mean())
                rec[f'agreement_{s}'] = float((pred == t_pred[s]).float().mean())
            rec['hash_unchanged'] = digest(head.parameters()) == d_hash
            out[f'consolidate-{tag}' + ('-anchor' if anchored else '')] = rec
    out['storage'] = {'rendering_bytes': PIXEL_BYTES, 'cue_bytes': d * 4, 'd': d, 'path_bytes': path_b, 'model_bytes': n_bytes(ref_enc['image'], ref_enc['sequence'], ref_model),
                      'buffer_bytes': len(cache) * PIXEL_BYTES, 'der_extra_bytes': len(cache) * 9 * 4 + 9 * PIXEL_BYTES, 'stored_code_bytes': len(cache) * d * 4}
    return out


# ---------------------------------------------------------------- task

def prepare(arm, seed):
    """Blocks 1-4, the frozen end-of-block-4 state, the marked set, the buffer and the teacher-side stores."""
    install_extra_loss()
    withheld = D.withheld_counts(seed)
    sets, row, ev = AP.build_sets(seed)
    A = sets['A']
    enc, model, opt = LM.setup(arm, seed)
    CURRENT['end'] = None
    rng = nc.stream('partA_train', seed, 'A-order')
    for b, lang in enumerate(P.LANGS):  # blocks 1-3
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        for _ in range(D.STEPS):
            eps = P.episodes(lang, withheld[lang], srng, D.BATCH_EP)
            LM.train_update(enc, model, opt, SP.batch_material(eps, D.context_episodes(crng, D.N_CONTEXT), 'taps'), rng, True)
    curve4 = []
    AP.block(enc, model, opt, rng, seed, 4, L, A, row, {'A': (A, ev['A'])}, curve4, 'A')
    b4 = AP.freeze(enc, model, opt, rng)
    M = marked_set(seed, A)
    U = {cp: n for cp, n in A.items() if cp not in M}
    order = list(A)
    idx = lambda S: [order.index(cp) for cp in S]
    ref_enc, ref_model, _, _ = AP.thaw(b4, arm, seed)
    tr = X.DATA[('pairsym', 'train')][0].reshape(-1, R, 28, 28)
    brng = nc.stream('rr_buffer', seed)
    buf = {cp: sorted(int(j) for j in brng.choice(R, K, replace=False)) for cp in M}
    ctx = {'M': M, 'U': U, 'ev_M': ev['A'][idx(M)], 'ev_U': ev['A'][idx(U)], 'row': row, 'buf': buf, 'B': sets['B'], 'ev_B': ev['B'], 'canon_imgs': canonical_images()}
    canon = codes(ref_enc, ref_model, ctx['canon_imgs'])
    ctx['stored_codes'] = {(cp, j): codes(ref_enc, ref_model, tr[row[cp]][j][None])[0] for cp in M for j in buf[cp]}
    with torch.no_grad():
        ctx['stored_scores'] = {k: scores(ref_model, v[None], canon)[0] for k, v in ctx['stored_codes'].items()}
    return dict(withheld=withheld, sets=sets, M=M, buf=buf, b4=b4, ref=(ref_enc, ref_model), ctx=ctx, curve4=curve4)


def run_task(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    pre = prepare(arm, seed)
    withheld, sets, M, buf, b4, (ref_enc, ref_model), ctx, curve4 = (pre[k] for k in ('withheld', 'sets', 'M', 'buf', 'b4', 'ref', 'ctx', 'curve4'))
    out = {'N': N, 'L': L, 'K': K, 'points': POINTS, 'sets': {k: {str(cp): n for cp, n in v.items()} for k, v in sets.items()},
           'marked': {str(cp): n for cp, n in M.items()}, 'buffer': {str(cp): j for cp, j in buf.items()},
           'hyper': dict(K=K, every=REHEARSE_EVERY, batch=BUFFER_BATCH, der_alpha=DER_ALPHA, head_ninefold=HP[False], head_anchored=HP[True], lr=DISTILL_LR),
           'block4': {'curve': curve4, 'recall': recall_MU(ref_enc, ref_model, ctx, seed)}, 'branches': {}}
    out['static'] = static_branches(ref_enc, ref_model, ctx, seed, sets)
    for name, spec in HOST.items():
        if arm in spec.get('arms', (arm,)):
            out['branches'][name] = run_branch(name, spec, b4, arm, seed, ctx, withheld)
    return f'{arm}/seed{seed}', out, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


if __name__ == '__main__':
    AP.NAME, AP.RUNS, AP.run_task = NAME, RUNS, run_task
    AP.main()
