"""Number systems meta v1 pilot (docs/number-systems-meta-v1-spec.md, draft).

Source training with bounded adaptation episodes (first-order MAML) against ordinary training at an
equal encoding budget, then few-shot adaptation to the final held-out representations.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_systems_meta pilot
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

from experiments import number_systems as X
from experiments import number_systems_reuse_cost as R

nc, ncore, ns, net = X.nc, X.ncore, X.ns, X.net
ROOT = X.ROOT
SPEC = ROOT / 'docs/number-systems-meta-v1-spec.md'
NAME = f'number-systems-meta-v1-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX, RUNS = ROOT / 'reports' / NAME, ROOT / 'runs' / NAME

DEV = ('banana', 'french', 'devanagari_digits')
SOURCES = tuple(s for s in X.SOURCES if s not in DEV)
ARMS = {'ordinary': (False, False), 'ordinary+dev': (False, False), 'meta': (True, False), 'meta-I': (True, True)}  # (episodes, incompatible)
STEP_ENC = X.K * (X.BATCH + 6 * X.PAIRS)
BUDGET_STEPS = 20 if X.SMOKE else 3000
BUDGET = BUDGET_STEPS * STEP_ENC  # training encodings per arm
EVERY_EPISODE, B, K_SHOT, LR_INNER, LAMBDA, PRESERVE = 4, 10, 8, 1e-2, 1.0, 2
SHOTS = (2,) if X.SMOKE else (2, 8, 32)
SCORE_AT = (0, 5, 10) if X.SMOKE else (0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200)
RETAIN = 0.9


# ---------------------------------------------------------------- source training

def episode(enc, sri, opt, codes, d, idx_d, order, labelled, src_idx, rng):
    """One first-order MAML episode on development representation d; returns (outer loss, encodings)."""
    sat = [*enc['image'].parameters(), *sri.satellite_parameters('image')]
    saved = [p.detach().clone() for p in sat]
    support = {c: rng.choice(v, K_SHOT, replace=False) for c, v in idx_d.items()}
    query = {c: np.setdiff1d(v, support[c]) for c, v in idx_d.items()}
    nums = tuple(X.numbers(d))
    inner = torch.optim.Adam(sat, LR_INNER)
    enc_count = 0
    for _ in range(B):
        req, tasks = X.task_batch(rng, d, support, nums, order[d])
        loss = X.batch_loss(sri, codes, X.encode(enc, sri, {d: req}, 'train')[d], tasks)
        inner.zero_grad()
        loss.backward()
        inner.step()
        enc_count += len(req)
    opt.zero_grad()  # discard gradients the inner loop left on shared parameters
    groups, tasks = {}, {}
    groups[d], tasks[d] = X.task_batch(rng, d, query, nums, order[d])
    images = [s for s in src_idx if X.MEDIUM[s] == 'image']
    for s in rng.choice(images, PRESERVE, replace=False).tolist():
        groups[s], tasks[s] = X.task_batch(rng, s, src_idx[s], labelled[s], order[s])
    st = X.encode(enc, sri, groups, 'train')
    new = X.batch_loss(sri, codes, st[d], tasks[d])
    keep = sum(X.batch_loss(sri, codes, st[s], tasks[s]) for s in groups if s != d)
    outer = new + LAMBDA * keep
    if not torch.isfinite(outer):
        raise nc.NumericalFailure('episode')
    outer.backward()
    with torch.no_grad():
        for p, q in zip(sat, saved):
            p.copy_(q)
    opt.step()
    return (round(new.item(), 4), round(keep.item(), 4)), enc_count + sum(len(g) for g in groups.values())


def source_fit(arm):
    episodes, incompatible = ARMS[arm]
    t0, c0 = time.perf_counter(), time.process_time()
    labelled = X.assign_labels(list(SOURCES), X.SEED)
    reps = list(SOURCES) + (list(DEV) if arm != 'ordinary' else [])
    order = X.orders(reps, incompatible, X.SEED)
    for d in DEV:
        labelled[d] = tuple(X.numbers(d))
    enc, sri = X.make_model(X.SEED)
    codes = ncore.label_codes(net.SIZES[0])
    opt = torch.optim.Adam([*enc['image'].parameters(), *enc['sequence'].parameters(), *sri.parameters()], X.LR)
    idx = {r: {c: np.flatnonzero(X.DATA[(r, 'train')][1] == c) for c in X.numbers(r)} for r in reps}
    rng = nc.stream('meta_train', X.SEED, arm)
    train_reps = reps if arm == 'ordinary+dev' else list(SOURCES)
    used, steps, n_episodes, losses, episode_losses = 0, 0, 0, [], []
    cost = {'source_steps_cpu': 0., 'episode_cpu': 0.}
    while True:
        if episodes and steps % EVERY_EPISODE == 0 and steps > 0 and used + (B + 1 + PRESERVE) * (X.BATCH + 6 * X.PAIRS) <= BUDGET:
            e0 = time.process_time()
            d = DEV[rng.integers(len(DEV))]
            ls, n = episode(enc, sri, opt, codes, d, idx[d], order, labelled, {s: idx[s] for s in SOURCES}, rng)
            cost['episode_cpu'] += time.process_time() - e0
            used += n; n_episodes += 1
            if n_episodes % 25 == 0:
                episode_losses.append(ls)
        if used + STEP_ENC > BUDGET:
            break
        e0 = time.process_time()
        l, n = X.train_steps(enc, sri, opt, train_reps, idx, labelled, order, 1, rng, codes)
        cost['source_steps_cpu'] += time.process_time() - e0
        used += n; steps += 1
        if steps % 100 == 0:
            losses.append(l[-1])
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f'{arm}.pt'
    torch.save({'image': enc['image'].state_dict(), 'sequence': enc['sequence'].state_dict(), 'sri': sri.state_dict(),
                'sources': list(SOURCES), 'labelled': {s: labelled[s] for s in SOURCES}}, path)
    e0 = time.process_time()
    ev = X.evaluate_source(enc, sri, codes, list(SOURCES), {s: labelled[s] for s in SOURCES})
    eval_cpu = time.process_time() - e0
    return ('source', arm), {
        'source_steps': steps, 'episodes': n_episodes, 'training_encodings': used, 'budget': BUDGET,
        'loss_every_100_steps': losses, 'episode_losses_every_25': episode_losses,
        'incompatible_orders': {r: v for r, v in order.items()} if incompatible else None,
        'checkpoint': {'file': str(path.relative_to(ROOT)), 'sha256': nc.sha256(path)}, 'eval': ev}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'eval_cpu_seconds': eval_cpu,
        'source_step_cpu_seconds': cost['source_steps_cpu'], 'episode_cpu_seconds': cost['episode_cpu'],
        'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


# ---------------------------------------------------------------- few-shot evaluation

def adapt(job):
    arm, held, k = job
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri = X.make_model(X.SEED)
    sources, labelled = [], {}
    if arm is not None:
        ck = torch.load(RUNS / f'{arm}.pt', weights_only=False)
        enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
        sri.load_state_dict(ck['sri'])
        sources, labelled = ck['sources'], {r: tuple(v) for r, v in ck['labelled'].items()}
    codes = ncore.label_codes(net.SIZES[0])
    med = X.MEDIUM[held]
    held_idx = R.subset(held, R.HELD_EVAL, 'held')
    y_held = X.DATA[(held, 'eval')][1][held_idx]
    src_groups = {r: R.subset(r, R.SOURCE_EVAL, 'source') for r in sources}
    train_y = X.DATA[(held, 'train')][1]
    shots = {c: nc.stream('reuse_shots', held, int(c)).permutation(np.flatnonzero(train_y == c))[:k] for c in X.numbers(held)}
    curve, eval_cpu = [], 0.

    def evaluate(n):
        nonlocal eval_cpu
        e0 = time.process_time()
        with torch.no_grad():
            acc, per = R.held_accuracy(X.encode(enc, sri, {held: held_idx}, 'eval')[held], y_held, codes)
            point = {'step': n, 'held_accuracy': acc, 'held_by_number': per}
            if sources:
                point['retention_shared'], point['retention_per_source'] = R.source_retention(enc, sri, sources, labelled, src_groups, codes, med, False)
        eval_cpu += time.process_time() - e0
        point['train_cpu_seconds'] = time.process_time() - c0 - eval_cpu
        curve.append(point)

    # Interface adaptation of the shared satellite; sources are read through the same (adapted) satellite.
    opt = torch.optim.Adam([*enc[med].parameters(), *sri.satellite_parameters(med)], LR_INNER)
    evaluate(0)
    losses, encodings = X.train_steps(enc, sri, opt, [held], {held: shots}, {held: tuple(X.numbers(held))},
                                      {held: {n: n for n in X.numbers(held)}}, SCORE_AT[-1],
                                      nc.stream('meta_adapt', X.SEED, str(arm), held, k), codes,
                                      lambda n: evaluate(n) if n in SCORE_AT else None)
    return ('adapt', str(arm), held, k), {'examples': k * len(X.numbers(held)), 'curve': curve, 'loss_every_100': losses}, {
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0, 'eval_cpu_seconds': eval_cpu,
        'updates': SCORE_AT[-1], 'training_encodings': encodings, 'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def costs(results):
    """Criterion per held-out: no-source accuracy at the largest k after the full budget."""
    crit = {h: results[f'None/{h}/{SHOTS[-1]}']['curve'][-1]['held_accuracy'] for h in X.HELD_OUT}
    out = {'criterion': crit, 'runs': {}}
    for key, r in results.items():
        arm, held, k = key.split('/')
        c, before = r['curve'], r['curve'][0]
        hit = lambda ok: next((p['step'] for p in c if ok(p)), None)
        reach = {'accuracy_only': hit(lambda p: p['held_accuracy'] >= crit[held])}
        run = {'best_accuracy': max(p['held_accuracy'] for p in c)}
        if 'retention_shared' in before:
            # Retention over the sources sharing the adapted satellite (the affected medium) and per source.
            hit_med = [s for s in before['retention_per_source'] if X.MEDIUM[s] == X.MEDIUM[held]]
            affected = lambda p: float(np.mean([p['retention_per_source'][s] for s in hit_med]))
            reach['with_retention'] = hit(lambda p: p['held_accuracy'] >= crit[held] and affected(p) >= RETAIN * affected(before))
            run['final_retention_ratio'] = c[-1]['retention_shared'] / before['retention_shared']
            run['final_retention_ratio_affected_medium'] = affected(c[-1]) / affected(before)
            run['final_retention_per_source'] = {s: (c[-1]['retention_per_source'][s], v) for s, v in before['retention_per_source'].items()}
        out['runs'][key] = {'steps_to_criterion': reach, **run}
    return out


def run(job):
    return source_fit(job[1]) if job[0] == 'source' else adapt(job[1:])


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    phases = {'data_wall_seconds': time.perf_counter() - wall0}
    ctx = mp.get_context('fork')
    done = []
    for phase, jobs in (('source', [('source', a) for a in ARMS]),
                        ('adaptation', [('adapt', a, h, k) for a in [*ARMS, None] for h in X.HELD_OUT for k in sorted(SHOTS, reverse=True)])):
        t = time.perf_counter()
        with ctx.Pool(min(X.WORKERS, len(jobs))) as pool:
            for item in pool.imap_unordered(run, jobs):
                done.append(item)
                print(f'done {item[0]} ({len(done)}) at {time.perf_counter() - wall0:.0f} s', flush=True)
        phases[f'{phase}_wall_seconds'] = time.perf_counter() - t
    key = lambda k: '/'.join(map(str, k))
    adapt_res = {key(k[1:]): r for k, r, _ in done if k[0] != 'source'}
    res = {'spec': str(SPEC.relative_to(ROOT)), 'status': 'pilot: not headline evidence',
           'config': {'seed': X.SEED, 'dev': DEV, 'sources': SOURCES, 'held_out': X.HELD_OUT, 'arms': ARMS,
                      'budget_encodings': BUDGET, 'budget_steps': BUDGET_STEPS, 'every_episode': EVERY_EPISODE, 'inner_steps': B,
                      'k_shot_episode': K_SHOT, 'lr_inner': LR_INNER, 'lambda': LAMBDA, 'preserve_sources': PRESERVE, 'lr': X.LR,
                      'shots': SHOTS, 'score_at': SCORE_AT, 'retain': RETAIN, 'held_eval_per_number': R.HELD_EVAL,
                      'source_eval_per_number': R.SOURCE_EVAL},
           'sources': {k[1]: r for k, r, _ in done if k[0] == 'source'},
           'adaptation': dict(sorted(adapt_res.items())), 'cost_to_criterion': costs(adapt_res)}
    PREFIX.parent.mkdir(exist_ok=True)
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {key(k): rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'spec_sha256': nc.sha256(SPEC), 'runner_sha256': nc.sha256(__file__), 'experiment_sha256': nc.sha256(X.__file__),
        'reuse_cost_sha256': nc.sha256(R.__file__),
        'module_sha256': {m.__name__: nc.sha256(m.__file__) for m in (nc, ncore, ns, net)},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'environment_overrides': {k: v for k, v in os.environ.items() if k.startswith('NSY_')},
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
