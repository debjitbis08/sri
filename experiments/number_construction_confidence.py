"""Confidence analysis on Part A checkpoints (docs/number-construction-v1-spec.md, "Confidence analysis on Part A
checkpoints").

Can agreement between instinct (isolated collection matching) and relational inference (two-sided revision) be
anticipated from pre-revision features, and does a predictor of it gate revision usefully? Stable pilot 3 arms,
seeds 100-102, end-of-training checkpoints.

Usage: NSY_WORKERS=6 .venv/bin/python -m experiments.number_construction_confidence pilot3
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

from experiments import number_construction_part_a as P

X, B, nc, ns = P.X, P.B, P.nc, P.ns
ROOT = X.ROOT
PREFIX = ROOT / 'reports' / 'number-construction-v1-part-a-confidence'
RUNS = {(arm, seed): ROOT / 'runs' / f'number-construction-v1-part-a-pilot3{"" if seed == 100 else f"-seed{seed}"}' / f'{arm}.pt'
        for arm in ('A-order', 'A-shuffled') for seed in (100, 101, 102)}
LANGS = ('devanagari_digits', 'chinese')
C_COST = 0.02
FEATURES = ('margin', 'best', 'entropy', 'energy', 'fnorm')


def fit(Xf, y, steps=500):
    lin = torch.nn.Linear(Xf.shape[1], 1)
    with torch.no_grad():
        lin.weight.zero_(); lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), 0.05)
    for _ in range(steps):
        opt.zero_grad()
        (torch.nn.functional.binary_cross_entropy_with_logits(lin(Xf)[:, 0], y) + 1e-3 * lin.weight.pow(2).sum()).backward()
        opt.step()
    return lambda Z: torch.sigmoid(lin(Z)[:, 0]).detach().numpy()


def run(key):
    arm, seed = key
    t0, c0 = time.perf_counter(), time.process_time()
    enc, sri = X.make_model(seed)
    ck = torch.load(RUNS[key], weights_only=False)
    enc['image'].load_state_dict(ck['image']); sri.load_state_dict(ck['sri'])
    enc['image'].eval()
    rows = {k: [] for k in ('lang', 'n', 'half', 'correct', 'agree', 'revised_correct', *FEATURES, 'rel_check')}
    with torch.no_grad():
        sv = B.Solver(sri, 'image')
        st = lambda x: sri.split(sv.solve(enc['image'](torch.from_numpy(np.ascontiguousarray(x)))))
        A, t = sri.A, sri.t
        for lang in LANGS:
            for n in ns.NUMBERS:
                if not 2 <= n <= 8:
                    continue  # two-sided context only
                x = X.DATA[(lang, 'eval')][0][P.EVAL[(lang, n, 'idx')]]
                f = enc['image'](torch.from_numpy(x)); y = sv.solve(f); c = sri.split(y)['core_inner']
                cand = np.stack([P.EVAL[(lang, n, i)][0] for i in range(P.EVAL_PER)])
                counts = np.stack([P.EVAL[(lang, n, i)][1] for i in range(P.EVAL_PER)])
                cc = st(cand.reshape(-1, 28, 28))['core_inner'].reshape(P.EVAL_PER, 9, -1)
                lg = P.same_logits(sri, c[:, None].expand_as(cc).reshape(-1, P.SIZES[2]), cc.reshape(-1, P.SIZES[2])).reshape(P.EVAL_PER, 9)
                top = lg.topk(2, 1)
                pick = counts[np.arange(P.EVAL_PER), top.indices[:, 0].numpy()]
                prob = torch.softmax(lg, 1)
                lo = st(np.stack([P.EVAL[(lang, n, i)][3][-1] for i in range(P.EVAL_PER)]))['core_inner']
                hi = st(np.stack([P.EVAL[(lang, n, i)][3][1] for i in range(P.EVAL_PER)]))['core_inner']
                yr = sv.revise(f, P.MU, A, t, lo, hi, 'conf')
                cr = sri.split(yr)['core_inner']
                lgr = P.same_logits(sri, cr[:, None].expand_as(cc).reshape(-1, P.SIZES[2]), cc.reshape(-1, P.SIZES[2])).reshape(P.EVAL_PER, 9)
                rpick = counts[np.arange(P.EVAL_PER), lgr.argmax(1).numpy()]
                rows['lang'] += [lang] * P.EVAL_PER; rows['n'] += [n] * P.EVAL_PER
                rows['half'] += [0] * (P.EVAL_PER // 2) + [1] * (P.EVAL_PER - P.EVAL_PER // 2)
                rows['correct'].append(pick == n); rows['agree'].append(pick == rpick); rows['revised_correct'].append(rpick == n)
                rows['margin'].append((top.values[:, 0] - top.values[:, 1]).numpy()); rows['best'].append(top.values[:, 0].numpy())
                rows['entropy'].append(-(prob * prob.clamp_min(1e-12).log()).sum(1).numpy())
                rows['energy'].append((sv.energy(f, y) / f.norm(dim=1) ** 2).numpy()); rows['fnorm'].append(f.norm(dim=1).numpy())
                # cheap relational check without a revision solve: consistency of the isolated code with both neighbours
                rows['rel_check'].append(((c - (lo @ A.T + t)).pow(2).sum(1) + (hi - (c @ A.T + t)).pow(2).sum(1)).numpy())
    lang = np.array(rows['lang']); half = np.array(rows['half'])
    cat = {k: np.concatenate(v) for k, v in rows.items() if k not in ('lang', 'n', 'half')}
    out = {}
    for g in LANGS:
        m = lang == g
        res = {'isolated_accuracy': float(cat['correct'][m].mean()), 'revised_accuracy': float(cat['revised_correct'][m].mean()),
               'agreement_rate': float(cat['agree'][m].mean()),
               'accuracy_when_agree': float(cat['correct'][m & cat['agree']].mean()) if (m & cat['agree']).any() else None,
               'accuracy_when_disagree': float(cat['correct'][m & ~cat['agree']].mean()) if (m & ~cat['agree']).any() else None}
        for fname, feats in (('pre_revision', FEATURES), ('pre_revision_plus_rel_check', (*FEATURES, 'rel_check'))):
            F_ = np.stack([cat[k] for k in feats], 1)
            a, b = m & (half == 0), m & (half == 1)
            mu, sd = F_[a].mean(0), F_[a].std(0) + 1e-9
            Z = torch.from_numpy((F_ - mu) / sd).float()
            r = {}
            for target in ('agree', 'correct'):
                pred = fit(Z[torch.from_numpy(a)], torch.from_numpy(cat[target][a].astype(np.float32)))
                p = pred(Z)
                r[f'auroc_{target}'] = B.auroc(p[b], cat[target][b])
                if target == 'agree':
                    # gate: revise when predicted agreement is low; threshold chosen on half A
                    imm, rev = cat['correct'], cat['revised_correct']
                    obj = lambda th, sel: np.where(p[sel] < th, rev[sel], imm[sel]).mean() - C_COST * (p[sel] < th).mean()
                    cands = np.unique(np.r_[p[a], 0., 1.01])
                    th = max(cands, key=lambda u: obj(u, a))
                    go = p[b] < th
                    rate = float(go.mean())
                    r['selection'] = {'immediate': float(imm[b].mean()), 'always_revise': float(rev[b].mean()),
                                      'gated': float(np.where(go, rev[b], imm[b]).mean()), 'gated_rate': rate,
                                      'random_at_gated_rate': float(imm[b].mean() + rate * (rev[b].mean() - imm[b].mean())),
                                      'oracle': float((imm[b] | rev[b]).mean())}
            res[fname] = r
        res['single_feature_auroc_agree'] = {k: B.auroc(cat[k][m], cat['agree'][m]) for k in (*FEATURES, 'rel_check')}
        out[g] = res
    return f'{arm}/seed{seed}', out, {'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0,
                                       'peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    wall0, cpu0 = time.perf_counter(), time.process_time()
    data = X.prepare()
    P.prepare()
    phases = {'prepare_wall_seconds': time.perf_counter() - wall0}
    t = time.perf_counter()
    with mp.get_context('fork').Pool(min(X.WORKERS, len(RUNS))) as pool:
        done = pool.map(run, list(RUNS))
    phases['analysis_wall_seconds'] = time.perf_counter() - t
    res = {'spec': str(P.SPEC.relative_to(ROOT)), 'status': 'post-hoc confidence analysis of Part A pilot 3 checkpoints',
           'config': {'runs': {f'{a}/seed{s}': str(p.relative_to(ROOT)) for (a, s), p in RUNS.items()}, 'langs': LANGS, 'features': FEATURES,
                      'c_cost': C_COST, 'mu': P.MU, 'eval_per_count': P.EVAL_PER},
           'checkpoint_sha256': {f'{a}/seed{s}': nc.sha256(p) for (a, s), p in RUNS.items()},
           'results': {k: r for k, r, _ in sorted(done)}}
    with gzip.open(f'{PREFIX}-results.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True, default=float) + '\n')
    worker = {k: rt for k, _, rt in done}
    Path(f'{PREFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in -results.json.gz',
        'runner_sha256': nc.sha256(__file__), 'part_a_sha256': nc.sha256(P.__file__), 'spec_sha256': nc.sha256(P.SPEC),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'workers': X.WORKERS, 'phases': phases, 'tasks': worker,
        'data_bytes': data['data_bytes'], 'worker_cpu_seconds_total': sum(r['cpu_seconds'] for r in worker.values()),
        'wall_seconds': time.perf_counter() - wall0, 'parent_cpu_seconds': time.process_time() - cpu0,
        'peak_rss_children_max_kb': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}, indent=1, sort_keys=True, default=float) + '\n')


if __name__ == '__main__':
    main()
