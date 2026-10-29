"""Cross-sense confirmation v1 analysis (docs/cross-sense-confirmation-v1-contract.md).

Primary: form -> taps matching for withheld Chinese forms at the end of training, SRI minus standard, paired by seed
(200-209). Decision rule fixed in the contract: bootstrap 95% interval of the mean paired difference (10,000 resamples,
generator seed 0) and a practical margin of 0.10.
Secondary measures come from the runners' results. Diagnostics recompute the primary readout from the checkpoints
(checked against the runner's value) and add the hubness of the taps gallery and a hubness-corrected (CSLS-style)
readout.

Usage: .venv/bin/python -m experiments.cross_sense_confirmation_analysis
"""
import gzip
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import cross_sense_confirmation as CS
from src import standard_net as std

ROOT = X.ROOT
CONTRACT = ROOT / 'docs/cross-sense-confirmation-v1-contract.md'
MARGIN, BOOT, CSLS_K = .10, 10000, 10
LANG = 'chinese'
SEEDS = CS.SEEDS[:2] if X.SMOKE else CS.SEEDS
SUFFIX = '-smoke' if X.SMOKE else ''
SECONDARY = ['form_to_taps/devanagari_digits/withheld', 'form_to_taps/chinese/paired', 'form_to_taps/devanagari_digits/paired',
             'taps_to_form/chinese/withheld', 'taps_to_form/devanagari_digits/withheld']


def results(arch):
    with gzip.open(ROOT / 'reports' / f'{CS.name(arch)}-results.json.gz', 'rt') as f:
        return json.load(f)['results']


def decide(d):
    d = np.asarray(d)
    rng = np.random.default_rng(0)
    means = d[rng.integers(len(d), size=(BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if lo > 0 and d.mean() >= MARGIN:
        out = 'confirmed'
    elif hi < MARGIN:
        out = 'not confirmed'
    else:
        out = 'inconclusive'
    k, n = int((d > 0).sum()), int((d != 0).sum())
    from math import comb
    p = min(1., 2 * sum(comb(n, j) for j in range(max(k, n - k), n + 1)) / 2 ** n) if n else None
    return {'mean': float(d.mean()), 'ci95': [float(lo), float(hi)], 'positive_seeds': k, 'nonzero_seeds': n,
            'sign_test_two_sided_p': p, 'outcome': out}


@torch.no_grad()
def diagnostics(arch, seed):
    ck = torch.load(ROOT / 'runs' / CS.name(arch) / f'{CS.ARM}-seed{seed}.pt', weights_only=False)
    if arch == 'sri':
        enc, model = X.make_model(seed)
        model.load_state_dict(ck['sri'])
        codes = lambda x, m: model.split(B.Solver(model, m).solve(enc[m](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner']
    else:
        enc, model = SB.make_model(seed)
        model.load_state_dict(ck['sri'])
        codes = lambda x, m: std.StandardSolver(model, m).code(enc[m](torch.from_numpy(np.ascontiguousarray(x))))
    for e in enc.values():
        e.load_state_dict(ck['image'] if e is enc['image'] else ck['sequence']); e.eval()
    withheld = set(ck['withheld'].get(LANG, ()))
    E = P.EVAL_PER
    taps = torch.stack([codes(D.TAPS[('eval', n)], 'sequence') for n in X.ns.NUMBERS], 1)  # (E, 9, d)
    forms = torch.stack([codes(X.DATA[(LANG, 'eval')][0][P.EVAL[(LANG, n, 'idx')]], 'image') for n in X.ns.NUMBERS], 1)
    d = taps.shape[-1]
    # logits[i, n, k]: form of count n (item i) against taps of count k (item i), as in evaluate_senses
    logit = P.same_logits(model, forms[:, :, None].expand(-1, -1, 9, -1).reshape(-1, d), taps[:, None].expand(-1, 9, -1, -1).reshape(-1, d)).reshape(E, 9, 9)
    # CSLS-style correction: subtract each gallery element's mean of its top-k logits over all form queries of the item
    allq = P.same_logits(model, forms.reshape(-1, d)[:, None].expand(-1, E * 9, -1).reshape(-1, d),
                         taps.reshape(-1, d)[None].expand(E * 9, -1, -1).reshape(-1, d)).reshape(E * 9, E * 9)  # query x gallery
    r = allq.topk(CSLS_K, 0).values.mean(0).reshape(E, 9)  # per gallery element (item, count)
    csls = logit - r[:, None, :]
    out = {}
    for name, score in (('plain', logit), ('csls', csls)):
        pick = score.argmax(2).numpy()  # (E, 9)
        rows = [n - 1 for n in sorted(withheld)]
        out[f'form_to_taps_withheld_{name}'] = float(np.mean([pick[:, j] == j for j in rows]))
        out[f'form_to_taps_all_{name}'] = float((pick == np.arange(9)).mean())
        counts = np.bincount(pick.ravel(), minlength=9) / pick.size
        out[f'pick_share_max_{name}'] = float(counts.max())
        out[f'pick_entropy_bits_{name}'] = float(-(counts[counts > 0] * np.log2(counts[counts > 0])).sum())
        if name == 'plain':
            out['confusion_withheld'] = {str(j + 1): np.bincount(pick[:, j], minlength=9).tolist() for j in rows}
    # k-occurrence skew of taps codes among nearest neighbours of form codes (Radovanovic et al. 2010), k = 5
    nn5 = allq.topk(5, 1).indices.numpy().ravel()
    occ = np.bincount(nn5, minlength=E * 9).astype(float)
    out['k5_occurrence_skewness'] = float(((occ - occ.mean()) ** 3).mean() / occ.std() ** 3) if occ.std() > 0 else None
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    X.prepare(); P.prepare(); D.prepare()
    t_prep = time.perf_counter() - t0
    R = {a: results(a) for a in ('sri', 'standard')}
    key = f'{CS.ARM}/seed{{}}'
    get = lambda a, s, k: R[a][key.format(s)]['after_block'][-1]['senses']['cross_sense'][k]
    prim = {a: [get(a, s, 'form_to_taps/chinese/withheld') for s in SEEDS] for a in R}
    diff = [x - y for x, y in zip(prim['sri'], prim['standard'])]
    sec = {k: {a: [get(a, s, k) for s in SEEDS] for a in R} for k in SECONDARY}
    sec['taps_to_collection'] = {a: [R[a][key.format(s)]['after_block'][-1]['senses']['taps_to_collection'] for s in SEEDS] for a in R}
    sec['isolated_chinese_withheld_end'] = {a: [R[a][key.format(s)]['after_block'][-1]['isolated']['chinese/withheld'] for s in SEEDS] for a in R}
    t1 = time.perf_counter()
    diag = {a: {str(s): diagnostics(a, s) for s in SEEDS} for a in R}
    t_diag = time.perf_counter() - t1
    check = max(abs(diag[a][str(s)]['form_to_taps_withheld_plain'] - get(a, s, 'form_to_taps/chinese/withheld')) for a in R for s in SEEDS)
    res = {'contract': str(CONTRACT.relative_to(ROOT)), 'seeds': SEEDS, 'arm': CS.ARM, 'margin': MARGIN, 'bootstrap': BOOT,
           'primary': {'measure': 'form_to_taps/chinese/withheld', 'sri': prim['sri'], 'standard': prim['standard'],
                       'paired_difference': diff, **decide(diff)},
           'secondary': {k: {**v, 'paired_difference_mean': float(np.mean(np.subtract(v['sri'], v['standard'])))} for k, v in sec.items()},
           'diagnostics': diag, 'recomputation_max_abs_error': check,
           'withheld': {s: R['sri'][key.format(s)]['withheld'] for s in SEEDS}}
    with gzip.open(ROOT / 'reports' / f'cross-sense-confirmation-v1-results{SUFFIX}.json.gz', 'wt') as f:
        f.write(json.dumps(res, indent=1, sort_keys=True) + '\n')
    (ROOT / 'reports' / f'cross-sense-confirmation-v1-analysis{SUFFIX}-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing and environment; scientific results are in cross-sense-confirmation-v1-results.json.gz',
        'contract_sha256': X.nc.sha256(CONTRACT),
        'module_sha256': {p: X.nc.sha256(ROOT / p) for p in ('experiments/cross_sense_confirmation.py', 'experiments/cross_sense_confirmation_analysis.py',
                                                              'experiments/standard_baseline.py', 'src/standard_net.py', 'experiments/number_construction_part_d.py')},
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
        'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'workers_for_data': X.WORKERS,
        'phases': {'data_wall_seconds': t_prep, 'diagnostics_wall_seconds': t_diag},
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: res[k] for k in ('primary', 'recomputation_max_abs_error')}, indent=1))
    for k, v in res['secondary'].items():
        print(k, f"SRI {np.mean(v['sri']):.2f} standard {np.mean(v['standard']):.2f} diff {v['paired_difference_mean']:+.2f}")
    for a in R:
        dg = diag[a]
        print(a, {m: round(float(np.mean([dg[str(s)][m] for s in SEEDS])), 3) for m in
                  ('form_to_taps_withheld_csls', 'pick_share_max_plain', 'pick_share_max_csls', 'k5_occurrence_skewness')})


if __name__ == '__main__':
    main()
