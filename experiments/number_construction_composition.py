"""Relational composition test on Part D checkpoints (docs/number-construction-v1-spec.md, "Relational composition test
on Part D checkpoints").

A start item, a hidden action (add or remove k), and a claim that the result is more or fewer than a visible reference.
Composed verdict: the learned successor applied k times (its inverse for remove), then the comparison head. Compared
with the comparison-only shortcut (start against reference) and with the shown result. No training.

Usage: NSY_WORKERS=11 .venv/bin/python -m experiments.number_construction_composition pilot
"""
from collections import defaultdict
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
from experiments import number_construction_b1 as B
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D

nc, ns = X.nc, X.ns
ROOT = X.ROOT
RUN_DIR = ROOT / 'runs' / 'number-construction-v1-part-d-pilot'
NAME = f'number-construction-v1-composition-{X.MODE}' + ('-smoke' if X.SMOKE else '')
PREFIX = ROOT / 'reports' / NAME
KINDS = ('collections', 'taps', 'forms/english', 'forms/chinese', 'forms/devanagari_digits')
KS = (1, 2, 3)
MAX_COND = 1e3  # removal (inverse-successor inference) is not evaluated above this condition number


def items(kind):
    if kind == 'collections':
        return {n: P.DOTS['succ'][n] for n in ns.NUMBERS}
    if kind == 'taps':
        return {n: D.TAPS[('eval', n)] for n in ns.NUMBERS}
    lang = kind.split('/')[1]
    return {n: X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]] for n in ns.NUMBERS}


@torch.no_grad()
def run(path):
    t0, c0 = time.perf_counter(), time.process_time()
    ck = torch.load(path, weights_only=False)
    arm, seed, withheld = ck['arm'], ck['seed'], ck['withheld']
    enc, sri = X.make_model(seed)
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence']); sri.load_state_dict(ck['sri'])
    sv = {m: B.Solver(sri, m) for m in ('image', 'sequence')}
    codes = {}
    for kind in KINDS:
        m = 'sequence' if kind == 'taps' else 'image'
        codes[kind] = {n: sri.split(sv[m].solve(enc[m](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner'] for n, x in items(kind).items()}
    A, t = sri.A, sri.t
    sv_A = torch.linalg.svdvals(A)
    cond = float(sv_A[0] / sv_A[-1])
    Ainv = torch.linalg.inv(A) if cond <= MAX_COND else None
    removal = Ainv is not None and bool(torch.isfinite(Ainv).all())  # no pseudoinverse substitute (spec amendment)
    more = lambda a, b: (sri.relation_logits(a, b, 2) > 0).numpy()
    is_withheld = lambda kind, n: kind.startswith('forms/') and n in withheld.get(kind.split('/')[1], [])
    proto = torch.stack([codes['collections'][n].mean(0) for n in ns.NUMBERS])
    acc = defaultdict(lambda: [0, 0])
    cache = {}

    def state(sk, n, j):
        """S^j applied to the start codes; negative j is inverse-successor inference (None if removal is not evaluated)."""
        if j < 0 and not removal:
            return None
        if (sk, n, j) not in cache:
            c = codes[sk][n]
            for _ in range(abs(j)):
                c = c @ A.T + t if j > 0 else (c - t) @ Ainv.T
            cache[(sk, n, j)] = c
        return cache[(sk, n, j)]

    def add(key, hits):
        acc[key][0] += int(hits.sum()); acc[key][1] += hits.size

    cases = [(sk, n, action, k) for sk in KINDS for n in ns.NUMBERS for action in ('add', 'remove') for k in KS
             if 1 <= (n + k if action == 'add' else n - k) <= 9 and (action == 'add' or removal)]
    sign = {'add': 1, 'remove': -1}
    # does S^k(c) keep the right count (nearest collection prototype, same head)
    for sk, n, action, k in cases:
        c = state(sk, n, sign[action] * k)
        lg = P.same_logits(sri, c[:, None].expand(-1, 9, -1).reshape(-1, c.shape[1]),
                           proto[None].expand(len(c), -1, -1).reshape(-1, c.shape[1])).reshape(len(c), 9)
        add(('result_count', sk, action, k), lg.argmax(1).numpy() + 1 == n + sign[action] * k)
    # level 1: shown outcome, successor residual for the true result against a result of another count
    auc = {}
    for sk in KINDS:
        rng = nc.stream('composition_level1', sk)
        scores, labels = [], []
        for kind, n, action, k in cases:
            if kind != sk:
                continue
            c, mres = state(sk, n, sign[action] * k), n + sign[action] * k
            wrong = int(rng.choice([x for x in ns.NUMBERS if x != mres]))
            for o, bad in ((codes[sk][mres], False), (codes[sk][wrong], True)):
                scores.append((torch.roll(o, 2, 0) - c).pow(2).sum(1).numpy()); labels.append(np.full(len(c), bad))
        auc[sk] = B.auroc(np.concatenate(scores), np.concatenate(labels))
    # level 2: hidden result against a visible reference
    for sk, n, action, k in cases:
        j = sign[action] * k
        mres = n + j
        c, start, shown = state(sk, n, j), codes[sk][n], torch.roll(codes[sk][mres], 2, 0)
        wrong_j = j + sign[action] if 1 <= n + j + sign[action] <= 9 else (j - sign[action] if k > 1 else None)
        interventions = {'identity': 0, 'reversed': -j, 'wrong_count': wrong_j}
        for rk in KINDS:
            cross = (sk == 'taps') != (rk == 'taps')
            for r in ns.NUMBERS:
                ref = codes[rk][r] if rk != sk else torch.roll(codes[rk][r], 1, 0)
                wh = is_withheld(sk, n) or is_withheld(rk, r)
                stratum = 'ref_eq_start' if r == n else 'ref_eq_result' if r == mres else 'neither'
                for claim in ('more', 'fewer'):
                    expect = lambda m_: m_ > r if claim == 'more' else m_ < r
                    truth = expect(mres)
                    crit = 'critical' if truth != expect(n) else 'noncritical'
                    verdict = (lambda a: more(a, ref)) if claim == 'more' else (lambda a: more(ref, a))
                    composed = verdict(c)
                    hits = {'composed': composed == truth, 'shortcut': verdict(start) == truth, 'shown': verdict(shown) == truth,
                            'action_claim_rule': np.full(len(c), ((action == 'add') == (claim == 'more')) == truth)}
                    for meth, h in hits.items():
                        for key in (('all',), ('all', stratum), (crit,), (crit, stratum), (crit, stratum, k), ('all', k)):
                            add((meth, action, *key), h)
                        if wh:
                            add((meth, action, 'all', 'withheld'), h)
                        if cross:
                            add((meth, action, 'all', 'cross_sense'), h)
                    for name, jj in interventions.items():
                        if jj is None or not 1 <= n + jj <= 9 or expect(n + jj) == truth:
                            continue
                        alt = state(sk, n, jj)
                        if alt is None:
                            continue
                        add(('intervention', action, name, 'composed'), composed == truth)
                        add(('intervention', action, name, 'intervened'), verdict(alt) == truth)
    out = {'/'.join(map(str, k)): v[0] / v[1] for k, v in sorted(acc.items(), key=lambda kv: [str(x) for x in kv[0]])}
    out['level1_auroc_shown_outcome'] = auc
    out['n_items'] = {'/'.join(map(str, k)): v[1] for k, v in acc.items() if k[0] == 'composed' and len(k) == 3}
    out['successor_matrix'] = {'singular_values_min_max': [float(sv_A[-1]), float(sv_A[0])], 'condition_number': cond, 'removal_evaluated': removal}
    return f'{arm}/seed{seed}', out, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                       'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    P.prepare(); D.prepare()
    phases = {'prepare_wall_seconds': time.perf_counter() - wall0}
    paths = sorted(RUN_DIR.glob('*.pt'))
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(paths))) as pool:
        done = pool.map(run, paths)
    phases['analysis_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(P.SPEC.relative_to(ROOT)), 'status': 'composition test on Part D checkpoints (pilot, no training)',
           'config': {'runs': str(RUN_DIR.relative_to(ROOT)), 'kinds': KINDS, 'steps': KS, 'eval_per_count': P.EVAL_PER},
           'checkpoint_sha256': {p.name: nc.sha256(p) for p in paths},
           'results': {k: r for k, r, _ in sorted(done)}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'part_d_sha256': nc.sha256(D.__file__), 'spec_sha256': nc.sha256(P.SPEC),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'phases': phases, 'tasks': worker,
        'data_bytes': data['data_bytes'], 'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
