"""Number construction v1, multi-step teaching pilot (docs/number-construction-v1-spec.md, "Multi-step teaching").

D-varied+rehearsal from Part D, plus an explicit multi-step objective (the learned successor applied k = 1-3 times to an
item at count n must match the episode's collection at n + k) and/or equality pairs for the comparison head. Added
losses draw from their own streams, so all arms see identical batches and identical draws for the existing losses.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_multistep pilot
Then: NCC_RUNS=number-construction-v1-multistep-pilot for experiments.number_construction_composition and
experiments.number_construction_projection.
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
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D

nc, ns = X.nc, X.ns
ROOT = X.ROOT
SPEC = P.SPEC
SEEDS = D.SEEDS
NAME = f'number-construction-v1-multistep-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX, RUNS = ROOT / 'reports' / NAME, ROOT / 'runs' / NAME
LANGS, TRANSFER, SIZES = P.LANGS, P.TRANSFER, P.SIZES
STEPS = D.STEPS
N_MULTI, N_EQUAL, KS = 64, 16, (1, 2, 3)
ARMS = {'M-control': (False, False), 'M-multistep': (True, False), 'M-equality': (False, True),
        'M-multistep+equality': (True, True)}  # (multi-step objective, equality pairs); all are D-varied+rehearsal


def multistep_pairs(lang_eps, ctx_eps):
    """(start item, endpoint collection, k, endpoint count) for steps k = 1-3 apart in an episode whose higher-count step
    has a visible collection. Item indices follow D.batch_material's layout for the taps context."""
    out, pos = [], 0
    episodes = []
    for ep in lang_eps:
        steps = []
        for n, e, fi in ep['steps']:
            starts = [pos] + ([pos + 1] if e >= 0 else [])
            steps.append((n, starts, pos + 1 if e >= 0 else None))
            pos += 1 + (e >= 0)
        episodes.append(steps)
    for ep in ctx_eps:
        steps = []
        for n, e, e2, ti in ep['steps']:
            steps.append((n, [pos, pos + 1], pos + 1))
            pos += 2
        episodes.append(steps)
    for steps in episodes:
        for lo_n, lo_items, _ in steps:
            for hi_n, _, hi_coll in steps:
                k = hi_n - lo_n
                if k in KS and hi_coll is not None:
                    out += [(s, hi_coll, k, hi_n) for s in lo_items]
    return out


def train_update(enc, sri, opt, material, multi, rng, mrng, erng, use_multi, use_equal):
    """D.train_update for the taps context with comparison, plus the multi-step and equality losses."""
    items, pairs, match = material
    solvers = {'image': P.Coupled(sri), 'sequence': D.Sequence(sri)}
    c = D.encode_items(enc, sri, items, solvers)
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
    pool = np.array(by_kind['coll'] + by_kind['taps'])
    u, v = pool[rng.integers(len(pool), size=4 * D.N_COMPARE)], pool[rng.integers(len(pool), size=4 * D.N_COMPARE)]
    cnt = np.array([it[2] for it in items])
    keep = np.flatnonzero(cnt[u] != cnt[v])[:D.N_COMPARE]
    u, v = u[keep], v[keep]
    loss_cmp = F.binary_cross_entropy_with_logits(sri.relation_logits(c[u], c[v], 2), torch.from_numpy((cnt[u] > cnt[v]).astype(np.float32)))
    loss_multi, loss_eq = torch.zeros(()), torch.zeros(())
    if use_multi and multi:
        sel = [multi[i] for i in mrng.choice(len(multi), min(N_MULTI, len(multi)), replace=False)]
        rows = []
        for k in KS:
            idx = [r for r in sel if r[2] == k]
            if not idx:
                continue
            s = c[torch.tensor([r[0] for r in idx])]
            for _ in range(k):
                s = sri.S(s)
            tgt = torch.tensor([r[3] for r in idx]) - 1
            Ck = cand[None].expand(len(idx), -1, -1).clone()
            Ck[torch.arange(len(idx)), tgt] = c[torch.tensor([r[1] for r in idx])]
            lg = P.same_logits(sri, s[:, None].expand_as(Ck).reshape(-1, SIZES[2]), Ck.reshape(-1, SIZES[2])).reshape(len(idx), 9)
            rows.append(F.cross_entropy(lg, tgt, reduction='sum'))
        loss_multi = sum(rows) / len(sel)
    if use_equal:
        eu, ev = [], []
        for _ in range(20 * N_EQUAL):
            x, y = pool[erng.integers(len(pool))], pool[erng.integers(len(pool))]
            if x != y and cnt[x] == cnt[y]:
                eu.append(x); ev.append(y)
                if len(eu) == N_EQUAL:
                    break
        if eu:
            eu, ev = torch.tensor(eu), torch.tensor(ev)
            lg = torch.cat([sri.relation_logits(c[eu], c[ev], 2), sri.relation_logits(c[ev], c[eu], 2)])
            loss_eq = F.binary_cross_entropy_with_logits(lg, torch.zeros(len(lg)))
    loss = loss_match + loss_next + loss_cmp + loss_multi + loss_eq
    if not torch.isfinite(loss):
        raise nc.NumericalFailure('multi-step update')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], 5.0)
    opt.step()
    return [round(float(x.detach()), 4) for x in (loss_match, loss_next, loss_cmp, loss_multi, loss_eq)]


def run(task):
    arm, seed = task
    t0, c0 = time.perf_counter(), time.process_time()
    use_multi, use_equal = ARMS[arm]
    withheld = D.withheld_counts(seed)
    torch.manual_seed(seed)
    enc, sri = X.make_model(seed)
    params = [*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()]
    opt = torch.optim.Adam(params, P.LR)
    rng = nc.stream('partA_train', seed, 'A-order')  # Part D's draws, so M-control replicates D-varied+rehearsal
    reh_rng, buffer = nc.stream('partD_rehearsal', seed), {}
    mrng, erng = nc.stream('multistep_pairs', seed), nc.stream('multistep_equal', seed)
    losses, block_eval, cpu = [], [], {'train': 0., 'eval': 0.}
    for b, lang in enumerate(LANGS):
        srng, crng = nc.stream('partA_schedule', seed, lang), nc.stream('partD_context', seed, lang)
        lang_batches = [P.episodes(lang, withheld[lang], srng, D.BATCH_EP) for _ in range(STEPS)]
        e0 = time.process_time()
        for step, eps in enumerate(lang_batches):
            ctx = D.context_episodes(crng, D.N_CONTEXT)
            if buffer:
                earlier = list(buffer)
                old = [buffer[earlier[int(reh_rng.integers(len(earlier)))]][int(reh_rng.integers(D.BUFFER))] for _ in range(D.N_REHEARSE)]
                eps = eps[:D.BATCH_EP - D.N_REHEARSE] + old
            material = D.batch_material(eps, ctx, 'taps')
            ls = train_update(enc, sri, opt, material, multistep_pairs(eps, ctx), rng, mrng, erng, use_multi, use_equal)
            if step % 100 == 99:
                losses.append([b, step + 1, *ls])
        cpu['train'] += time.process_time() - e0
        pool = [ep for batch in lang_batches for ep in batch]
        buffer[lang] = [pool[i] for i in nc.stream('partD_buffer', seed, lang).choice(len(pool), D.BUFFER, replace=len(pool) < D.BUFFER)]
        e0 = time.process_time()
        ev = P.evaluate(enc, sri, withheld, full=True)
        ev['senses'] = D.evaluate_senses(enc, sri, withheld, full=b == len(LANGS) - 1)
        block_eval.append(ev)
        cpu['eval'] += time.process_time() - e0
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f'{arm}-seed{seed}.pt'
    torch.save({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': sri.state_dict(),
                'arm': arm, 'seed': seed, 'withheld': {k: list(v) for k, v in withheld.items()}}, path)
    return f'{arm}/seed{seed}', {'withheld': {k: list(v) for k, v in withheld.items()}, 'loss_every_100': losses, 'after_block': block_eval,
                                 'checkpoint': {'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)}}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, **{f'{k}_cpu_seconds': v for k, v in cpu.items()},
        'updates': 3 * STEPS, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    t = time.perf_counter()
    P.prepare(); D.prepare()
    phases = {'data_wall_seconds': t - wall0, 'render_wall_seconds': time.perf_counter() - t}
    t = time.perf_counter()
    tasks = [(arm, s) for s in SEEDS for arm in ARMS]
    done = []
    with mp.get_context('fork').Pool(min(X.WORKERS, len(tasks))) as pool:
        for item in pool.imap_unordered(run, tasks):
            done.append(item)
            print(f'done {item[0]} ({len(done)}/{len(tasks)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
    phases['runs_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(SPEC.relative_to(ROOT)), 'status': 'multi-step teaching pilot: not headline evidence',
           'config': {'seeds': SEEDS, 'arms': ARMS, 'base': 'D-varied+rehearsal', 'steps_per_block': STEPS, 'multistep_pairs': N_MULTI,
                      'multistep_k': KS, 'equality_pairs': N_EQUAL},
           'results': {k: r for k, r, _ in sorted(done, key=lambda d: d[0])}}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__), 'part_d_sha256': nc.sha256(D.__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]),
        'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith(('NSY_', 'NCA_', 'NCD_'))},
        'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'threads_per_worker': 1,
        'phases': phases, 'data_bytes': data['data_bytes'], 'tasks': worker,
        'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        'rss_convention': 'Linux ru_maxrss in KiB; children value is the largest single worker process'},
        indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
