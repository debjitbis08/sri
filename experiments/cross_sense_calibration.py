"""Cross-sense confirmation v1 follow-up: hubness correction from a frozen, independent calibration set
(docs/cross-sense-confirmation-v1-results.md, "Follow-up: frozen calibration").

The v1 diagnostic estimated each taps candidate's CSLS penalty from all evaluation form queries at once. Here the
penalty is estimated from a fixed calibration set of training-split forms (never evaluation items), frozen before any
test query, and each test query is then scored alone: score(q, g) = same(q, g) - r(g), with r(g) the mean of g's top-10
scores over the calibration queries. Calibration sets: 270 Chinese training forms (same language), or 270 Devanagari
training forms (other language), sampled without labels by a fixed generator.

Criterion fixed before running: the frozen correction "works under the intended conditions" if the paired difference
SRI (plain) - standard (frozen correction, same-language calibration) has a 95% bootstrap upper bound below 0.10.

Usage: .venv/bin/python -m experiments.cross_sense_calibration
"""
import gzip
import json
import sys
import time

import numpy as np
import torch

from experiments import number_systems as X
from experiments import number_construction_b1 as B
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline as SB
from experiments import cross_sense_confirmation as CS
from experiments import cross_sense_confirmation_analysis as CA
from src import standard_net as std

ROOT = X.ROOT
N_CAL, K = 270, 10
CALIBRATION = {'same_language': 'chinese', 'other_language': 'devanagari_digits'}


def calibration_images():
    out = {}
    for name, lang in CALIBRATION.items():
        pool = X.DATA[(lang, 'train')][0]
        out[name] = pool[np.random.default_rng(0).choice(len(pool), N_CAL, replace=False)]
    return out


@torch.no_grad()
def evaluate(arch, seed, cal):
    ck = torch.load(ROOT / 'runs' / CS.name(arch) / f'{CS.ARM}-seed{seed}.pt', weights_only=False)
    enc, model = (X.make_model if arch == 'sri' else SB.make_model)(seed)
    model.load_state_dict(ck['sri'])
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    for e in enc.values():
        e.eval()
    if arch == 'sri':
        codes = lambda x, m: model.split(B.Solver(model, m).solve(enc[m](torch.from_numpy(np.ascontiguousarray(x)))))['core_inner']
    else:
        codes = lambda x, m: std.StandardSolver(model, m).code(enc[m](torch.from_numpy(np.ascontiguousarray(x))))
    E, d = P.EVAL_PER, model.sizes[2]
    taps = torch.stack([codes(D.TAPS[('eval', n)], 'sequence') for n in X.ns.NUMBERS], 1)  # (E, 9, d)
    forms = torch.stack([codes(X.DATA[(CA.LANG, 'eval')][0][P.EVAL[(CA.LANG, n, 'idx')]], 'image') for n in X.ns.NUMBERS], 1)
    score = lambda q, g: P.same_logits(model, q[:, None].expand(-1, len(g), -1).reshape(-1, d), g[None].expand(len(q), -1, -1).reshape(-1, d)).reshape(len(q), len(g))
    logit = torch.stack([score(forms[i], taps[i]) for i in range(E)])  # (E, 9 queries, 9 candidates)
    rows = [n - 1 for n in sorted(ck['withheld'].get(CA.LANG, ()))]
    acc = lambda s: float(np.mean([(s.argmax(2).numpy()[:, j] == j) for j in rows]))
    out = {'plain': acc(logit)}
    for name, x in cal.items():
        cq = codes(x, 'image')
        r = score(cq, taps.reshape(-1, d)).topk(K, 0).values.mean(0).reshape(E, 9)  # frozen penalty per candidate
        out[f'frozen_{name}'] = acc(logit - r[:, None, :])
    return out


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    X.prepare(); P.prepare(); D.prepare()
    cal = calibration_images()
    res = {a: {str(s): evaluate(a, s, cal) for s in CS.SEEDS} for a in ('sri', 'standard')}
    v = lambda a, k: np.array([res[a][str(s)][k] for s in CS.SEEDS])
    summary = {}
    for k in ('plain', *(f'frozen_{n}' for n in CALIBRATION)):
        summary[k] = {'sri': float(v('sri', k).mean()), 'standard': float(v('standard', k).mean())}
    crit = CA.decide(v('sri', 'plain') - v('standard', 'frozen_same_language'))
    crit['works'] = crit['ci95'][1] < .10
    same = CA.decide(v('sri', 'frozen_same_language') - v('standard', 'frozen_same_language'))
    out = {'seeds': CS.SEEDS, 'n_calibration': N_CAL, 'k': K, 'calibration': CALIBRATION, 'per_seed': res, 'means': summary,
           'criterion_sri_plain_minus_standard_frozen_same': crit, 'both_frozen_same_sri_minus_standard': same}
    with gzip.open(ROOT / 'reports' / 'cross-sense-confirmation-v1-calibration-results.json.gz', 'wt') as f:
        f.write(json.dumps(out, indent=1, sort_keys=True) + '\n')
    (ROOT / 'reports' / 'cross-sense-confirmation-v1-calibration-runtime.json').write_text(json.dumps({
        'scope': 'nondeterministic timing; results in cross-sense-confirmation-v1-calibration-results.json.gz',
        'module_sha256': X.nc.sha256(__file__), 'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')),
        'command': ' '.join([sys.executable, *sys.argv]), 'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0},
        indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('means', 'criterion_sri_plain_minus_standard_frozen_same', 'both_frozen_same_sri_minus_standard')}, indent=1))
    print('standard per seed', {k: [round(res['standard'][str(s)][k], 2) for s in CS.SEEDS] for k in res['standard']['200']})


if __name__ == '__main__':
    main()
