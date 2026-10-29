"""Number construction v1, Part D pilot: senses, varied contexts and rehearsal (docs/number-construction-v1-spec.md,
"Part D protocol").

Part A's A-order arm, with two changes tested separately and together:
  rehearsal  in the Chinese and Devanagari blocks, 4 of each batch's 16 language episodes come from a bounded buffer
             of earlier-language episodes;
  contexts   4 extra counting episodes per batch. Varied: taps (a heard count, through the sequence sense) paired with
             collections, plus a comparison loss ("which is more") on collections and taps. Repeated (matched control):
             a second collection paired with the first, no comparison loss. Neither ever shows a form.
Taps never meet forms and comparisons never involve forms, so taps-form matching and form comparisons are untaught
compositions. Seeds 100-102; checkpoints feed the imagery probe (experiments/number_construction_imagery.py).

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_part_d pilot
Environment: NCD_SEEDS (100,101,102), NCA_STEPS (2000 updates per block), NSY_WORKERS, NSY_SMOKE=1.
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
from experiments import number_construction_part_a as P

P.PILOT2 = True  # Part A pilot 2 data (half fixed-area training collections) and gradient clipping
nc, ns = X.nc, X.ns
ROOT = X.ROOT
SPEC = P.SPEC
SEEDS = tuple(int(s) for s in os.environ.get('NCD_SEEDS', '100,101,102').split(','))
NAME = f'number-construction-v1-part-d-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX, RUNS = ROOT / 'reports' / NAME, ROOT / 'runs' / NAME
LANGS, TRANSFER, SIZES = P.LANGS, P.TRANSFER, P.SIZES
STEPS = 20 if X.SMOKE else int(os.environ.get('NCA_STEPS', '2000'))
BATCH_EP, N_REHEARSE, N_CONTEXT, BUFFER, N_COMPARE = 16, 4, 4, 500, 64
TAPS_PER = (40, 4) if X.SMOKE else (1000, 30)  # train pool and evaluation queries per count
ARMS = {'D-base': (False, None), 'D-rehearsal': (True, None), 'D-repeated+rehearsal': (True, 'dots'),
        'D-varied+rehearsal': (True, 'taps'), 'D-varied': (False, 'taps')}  # (rehearsal, context episodes)
TAPS = {}


# ---------------------------------------------------------------- data

def taps_pool(args):
    split, n = args
    per = TAPS_PER[0] if split == 'train' else TAPS_PER[1]
    return split, n, np.stack([ns.render_taps(n, nc.stream('partD_taps', split, n, i)) for i in range(per)])


def prepare():
    with mp.get_context('fork').Pool(X.WORKERS) as pool:
        for split, n, x in pool.map(taps_pool, [(s, n) for s in ('train', 'eval') for n in ns.NUMBERS]):
            TAPS[(split, n)] = x
        for key, cand, counts, side, ctx in pool.imap_unordered(P.eval_item, [('taps', n, i) for n in ns.NUMBERS for i in range(P.EVAL_PER)]):
            P.EVAL[key] = (cand, counts, side, ctx)
    TAPS['train'] = np.concatenate([TAPS[('train', n)] for n in ns.NUMBERS])
    TAPS['train_idx'] = {n: np.arange(TAPS_PER[0]) + (n - 1) * TAPS_PER[0] for n in ns.NUMBERS}


def withheld_counts(seed):
    """Part A's withholding for the same seed."""
    rng = nc.stream('partA_withheld', seed)
    pairs = [(a, b) for a in range(2, 9) for b in range(a + 2, 9)]
    while True:
        w2, w3 = pairs[rng.integers(len(pairs))], pairs[rng.integers(len(pairs))]
        if not set(w2) & set(w3):
            return {LANGS[0]: (), LANGS[1]: w2, LANGS[2]: w3}


def context_episodes(rng, n_ep):
    """Counting episodes without forms: each step pairs the episode's collection with a second item of the same count,
    either taps or another persistent collection; both are drawn so the two context kinds share everything else."""
    out = []
    for _ in range(n_ep):
        length = int(rng.integers(3, 10))
        start = int(rng.integers(1, 11 - length))
        counts = list(range(start, start + length))
        desc = rng.random() < P.DESC
        if desc:
            counts = counts[::-1]
        e, e2 = rng.choice(len(P.DOTS['train']), 2, replace=False)
        out.append({'desc': desc, 'steps': [(n, int(e), int(e2), int(rng.choice(TAPS['train_idx'][n]))) for n in counts]})
    return out


def batch_material(lang_eps, ctx_eps, context):
    items, pairs, match = P.batch_material(lang_eps)
    for ep in ctx_eps:
        prev = None
        for n, e, e2, ti in ep['steps']:
            a = len(items)
            items.append(('taps', None, n, ti) if context == 'taps' else ('coll', None, n, e2))
            items.append(('coll', None, n, e))
            match.append((a, a + 1, n))
            if prev is not None:
                for u, v in ((prev, a), (prev + 1, a + 1)):
                    pairs.append((u, v) if not ep['desc'] else (v, u))
            prev = a
    return items, pairs, match


# ---------------------------------------------------------------- training

def encode_items(enc, sri, items, solvers):
    """Core inner codes for items in order: images through the image sense, taps through the sequence sense."""
    img = [k for k, it in enumerate(items) if it[0] != 'taps']
    seq = [k for k, it in enumerate(items) if it[0] == 'taps']
    parts = [solvers['image'].solve(enc['image'](torch.from_numpy(P.images([items[k] for k in img]))))[:, P.INNER]]
    if seq:
        x = TAPS['train'][[items[k][3] for k in seq]]
        parts.append(solvers['sequence'].solve(enc['sequence'](torch.from_numpy(x)))[:, P.INNER])
    c = torch.cat(parts)
    return c[torch.from_numpy(np.argsort(np.array(img + seq)))]


def train_update(enc, sri, opt, material, rng, compare):
    """Part A's matching and succession losses (same random draws when there are no taps), plus optional comparison."""
    items, pairs, match = material
    solvers = {'image': P.Coupled(sri)}
    if any(it[0] == 'taps' for it in items):
        solvers['sequence'] = Sequence(sri)
    c = encode_items(enc, sri, items, solvers)
    ci = rng.integers(len(P.DOTS['train']), size=9)
    cand = solvers['image'].solve(enc['image'](torch.from_numpy(np.stack([P.DOTS['train'][ci[k], k] for k in range(9)]))))[:, P.INNER]
    fi = torch.tensor([m[0] for m in match]); cj = torch.tensor([m[1] for m in match]); nn_ = torch.tensor([m[2] for m in match]) - 1
    C = cand[None].expand(len(match), -1, -1).clone()
    C[torch.arange(len(match)), nn_] = c[cj]
    logits = P.same_logits(sri, c[fi][:, None].expand_as(C).reshape(-1, SIZES[2]), C.reshape(-1, SIZES[2])).reshape(len(match), 9)
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
    loss_next = F.binary_cross_entropy_with_logits(sri.relation_logits(c[a2], c[b2], 0), target)
    loss_cmp = torch.zeros(())
    if compare:
        pool = np.array(by_kind['coll'] + by_kind['taps'])
        u, v = pool[rng.integers(len(pool), size=4 * N_COMPARE)], pool[rng.integers(len(pool), size=4 * N_COMPARE)]
        cnt = np.array([it[2] for it in items])
        keep = np.flatnonzero(cnt[u] != cnt[v])[:N_COMPARE]
        u, v = u[keep], v[keep]
        loss_cmp = F.binary_cross_entropy_with_logits(sri.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
    loss = loss_match + loss_next + loss_cmp
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('part D update')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    return [round(float(v.detach()), 4) for v in (loss_match, loss_next, loss_cmp)]


class Sequence(P.Coupled):
    def __init__(self, sri):
        Q = sri.hessian('sequence')
        o = SIZES[0]
        self.Qyy, self.Qyx, self.L, self.mu = Q[o:, o:], Q[o:, :o], {}, P.MU


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def codes_for(enc, sri, x, medium):
    sv = B.Solver(sri, medium)
    return sri.split(sv.solve(enc[medium](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner']


def pick(sri, q, cands):
    """q: (n, d) queries, cands: (n, 9, d); index of the best candidate under the same head."""
    lg = P.same_logits(sri, q[:, None].expand_as(cands).reshape(-1, SIZES[2]), cands.reshape(-1, SIZES[2])).reshape(len(q), 9)
    return lg.argmax(1).numpy()


@torch.no_grad()
def evaluate_senses(enc, sri, withheld, full):
    """Taps acquisition, untaught taps-form compositions, comparison transfer to forms, and (full) coherence detection."""
    enc['image'].eval(); enc['sequence'].eval()
    E = P.EVAL_PER
    out = {}
    taps = {n: codes_for(enc, sri, TAPS[('eval', n)], 'sequence') for n in ns.NUMBERS}
    # taps -> collections: the trained pairing (shared Part A candidate format)
    hits = []
    for n in ns.NUMBERS:
        cand = np.stack([P.EVAL[('taps', n, i)][0] for i in range(E)])
        counts = np.stack([P.EVAL[('taps', n, i)][1] for i in range(E)])
        cc = codes_for(enc, sri, cand.reshape(-1, 28, 28), 'image').reshape(E, 9, -1)
        hits.append(counts[np.arange(E), pick(sri, taps[n], cc)] == n)
    out['taps_to_collection'] = float(np.concatenate(hits).mean())
    # untaught: taps <-> forms of each language (galleries of one form per count, matched by item index)
    forms = {lang: {n: codes_for(enc, sri, X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]], 'image') for n in ns.NUMBERS}
             for lang in (*LANGS, TRANSFER)}
    cross = {}
    for lang in (*LANGS, TRANSFER):
        fg = torch.stack([forms[lang][k] for k in ns.NUMBERS], 1)  # (E, 9, d)
        tg = torch.stack([taps[k] for k in ns.NUMBERS], 1)
        for n in ns.NUMBERS:
            grp = 'withheld' if n in withheld.get(lang, ()) else 'paired'
            cross.setdefault(f'taps_to_form/{lang}/{grp}', []).append(pick(sri, taps[n], fg) == n - 1)
            cross.setdefault(f'form_to_taps/{lang}/{grp}', []).append(pick(sri, forms[lang][n], tg) == n - 1)
    out['cross_sense'] = {k: float(np.concatenate(v).mean()) for k, v in cross.items()}
    # comparison head on held-out material: trained on collections and taps only
    def cmp(codes, select=lambda a, b: True):
        acc = []
        for a in ns.NUMBERS:
            for b in ns.NUMBERS:
                if a != b and select(a, b):
                    acc.append(((sri.relation_logits(codes[a], codes[b], 2) > 0).numpy() == (a > b)).mean())
        return float(np.mean(acc)) if acc else None
    colls = {n: codes_for(enc, sri, P.DOTS['succ'][n], 'image') for n in ns.NUMBERS}
    comp = {'collections': cmp(colls), 'taps': cmp(taps)}
    for lang in (*LANGS, TRANSFER):
        w = set(withheld.get(lang, ()))
        comp[f'forms/{lang}'] = cmp(forms[lang])
        if w:
            comp[f'forms/{lang}/involving_withheld'] = cmp(forms[lang], lambda a, b: a in w or b in w)
    out['comparison'] = comp
    if full:
        # coherence detection with no training for it: 5-step ascending runs, one interior step replaced by another count;
        # score = summed successor residuals ||c_{k+1} - S(c_k)||^2 over the run
        det = {}
        for name, codes in (('collections', colls), ('taps', taps), *((f'forms/{lang}', forms[lang]) for lang in (*LANGS, TRANSFER))):
            rng = nc.stream('partD_coherence', name)
            scores, labels = [], []
            for start in range(1, 6):
                run = [codes[start + k] for k in range(5)]
                for bad in (False, True):
                    seq = list(run)
                    if bad:
                        pos = int(rng.integers(1, 4))
                        alt = [m for m in ns.NUMBERS if m != start + pos]
                        seq[pos] = torch.stack([codes[int(rng.choice(alt))][i] for i in range(E)])
                    res = sum((seq[k + 1] - sri.S(seq[k])).pow(2).sum(1) for k in range(4))
                    scores.append(res.numpy()); labels.append(np.full(E, bad))
            det[name] = B.auroc(np.concatenate(scores), np.concatenate(labels))
        out['coherence_auroc_successor_residual'] = det
    enc['image'].train(); enc['sequence'].train()
    return out


# ---------------------------------------------------------------- one run

def run(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    rehearse, context = ARMS[arm]
    withheld = withheld_counts(seed)
    torch.manual_seed(seed)
    enc, sri = X.make_model(seed)
    params = [*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()]
    opt = torch.optim.Adam(params, P.LR)
    rng = nc.stream('partA_train', seed, 'A-order')  # Part A's A-order draws, so D-base replicates it
    reh_rng, buffer = nc.stream('partD_rehearsal', seed), {}
    losses, block_eval, cpu = [], [], {'train': 0., 'eval': 0.}
    for b, lang in enumerate(LANGS):
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        lang_batches = [P.episodes(lang, withheld[lang], srng, BATCH_EP) for _ in range(STEPS)]
        e0 = time.process_time()
        for step, eps in enumerate(lang_batches):
            ctx = context_episodes(crng, N_CONTEXT)  # drawn in every arm, so schedules match
            if rehearse and buffer:
                earlier = list(buffer)
                old = [buffer[earlier[int(reh_rng.integers(len(earlier)))]][int(reh_rng.integers(BUFFER))] for _ in range(N_REHEARSE)]
                eps = eps[:BATCH_EP - N_REHEARSE] + old
            ls = train_update(enc, sri, opt, batch_material(eps, ctx if context else [], context), rng, context == 'taps')
            if step % 100 == 99:
                losses.append([b, step + 1, *ls])
        cpu['train'] += time.process_time() - e0
        pool = [ep for batch in lang_batches for ep in batch]  # bounded buffer of this language's episodes
        buffer[lang] = [pool[i] for i in nc.stream('partD_buffer', seed, lang).choice(len(pool), BUFFER, replace=len(pool) < BUFFER)]
        e0 = time.process_time()
        ev = P.evaluate(enc, sri, withheld, full=True)
        ev['senses'] = evaluate_senses(enc, sri, withheld, full=b == len(LANGS) - 1)
        block_eval.append(ev)
        cpu['eval'] += time.process_time() - e0
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f'{arm}-seed{seed}.pt'
    torch.save({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': sri.state_dict(),
                'arm': arm, 'seed': seed, 'withheld': {k: list(v) for k, v in withheld.items()}}, path)
    # transfer: Part A's procedure (thai_digits, 4 shots per count, language episodes only, same for every arm)
    e0 = time.process_time()
    y = X.DATA[(TRANSFER, 'train')][1]
    shots = {n: nc.stream('partA_transfer_shots', n).permutation(np.flatnonzero(y == n))[:P.TRANSFER_SHOTS] for n in ns.NUMBERS}
    trng = nc.stream('partA_transfer', seed)
    topt = torch.optim.Adam(params, P.LR)
    transfer = {}
    for step in range(P.TRANSFER_STEPS + 1):
        if step in P.TRANSFER_AT:
            transfer[str(step)] = P.evaluate(enc, sri, {}, langs=(TRANSFER,))['isolated'][f'{TRANSFER}/paired']
        if step == P.TRANSFER_STEPS:
            break
        P.train_update(enc, sri, topt, P.episodes(TRANSFER, (), trng, BATCH_EP, shots), False, False, trng)
    transfer['retention_after'] = P.evaluate(enc, sri, withheld)['isolated']
    cpu['transfer'] = time.process_time() - e0
    return f'{arm}/seed{seed}', {'withheld': {k: list(v) for k, v in withheld.items()}, 'loss_every_100': losses, 'after_block': block_eval,
                                 'transfer': transfer, 'checkpoint': {'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)}}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **{f'{k}_cpu_seconds': v for k, v in cpu.items()},
        'updates': 3 * STEPS + P.TRANSFER_STEPS, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    t = time.perf_counter()
    P.prepare()
    prepare()
    phases = {'data_wall_seconds': t - wall0, 'render_wall_seconds': time.perf_counter() - t}
    t = time.perf_counter()
    tasks = [(arm, s) for s in SEEDS for arm in ARMS]
    done = []
    with mp.get_context('fork').Pool(min(X.WORKERS, len(tasks))) as pool:
        for item in pool.imap_unordered(run, tasks):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(tasks)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['runs_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(SPEC.relative_to(ROOT)), 'status': 'Part D pilot: not headline evidence',
           'config': {'seeds': SEEDS, 'langs': LANGS, 'transfer': TRANSFER, 'arms': ARMS, 'steps_per_block': STEPS, 'batch_episodes': BATCH_EP,
                      'rehearsal_episodes': N_REHEARSE, 'context_episodes': N_CONTEXT, 'buffer_per_language': BUFFER,
                      'comparison_pairs': N_COMPARE, 'taps_per_count': TAPS_PER, 'mu': P.MU, 'lr': P.LR, 'eval_per_count': P.EVAL_PER},
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__), 'part_a_sha256': nc.sha256(P.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ns, X.ncore, X.net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]),
        'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith(('NSY_', 'NCA_', 'NCD_'))},
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
