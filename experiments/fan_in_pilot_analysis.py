"""Fan-in pressure pilot, stage 1 analysis (docs/fan-in-pressure-proposal.md).

1. Exactness: the dense configuration must reproduce the sequence pilot's D-varied+rehearsal results exactly.
2. Measures: the sequence pilot's comparison measures for every configuration (generic measures recomputed from
   checkpoints), outcome labels against the dense rerun with the audit's margins, paired by seed.
3. Decision: a configuration is underfit if any taught measure's mean falls by more than the margin; it advances if it
   is not underfit, better on at least two primary measures and worse on none.
4. Arrangement probe: ridge regression of count from collection codes (fitted on collections only), applied unchanged
   to taps and form codes; mean absolute count error and pairwise order accuracy.
5. Diagnostics: effective edges per limited layer, code participation ratio, reader units never active; for learned
   selection, the fraction of units whose dimension changed and the fraction reading the same dimension of both operands.
6. Single-coordinate probe ('erasure' onward): per code coordinate, an affine count map fitted on collections only and
   applied unchanged to taps and forms; the distribution over all coordinates (proposal, "Erasure, moving, sparsity and
   rate").

Confirmation ('confirm', seeds 200-209): no exactness check or stage-1 decision; outcomes use the paired 95% t-interval
rule of the proposal ("Confirmation after stage 1") and the declared contrasts.

Usage: .venv/bin/python -m experiments.fan_in_pilot_analysis pilot [confirm|rigl|rigl1]
"""
import gzip
import json
import sys
import time

import numpy as np
from scipy import stats
import torch

from experiments import number_systems as X
from experiments import number_construction_part_a as P
from experiments import number_construction_part_d as D
from experiments import standard_baseline_compare as C
from experiments import generic_relations as G
from experiments import sequence_pilot as SP
from experiments import fan_in_pilot as FI

ROOT, ns = X.ROOT, X.ns
CONFIRM = FI.STAGE != 'stage1'  # confirmation-style analysis: ten seeds, interval rule, declared contrasts
SEEDS = FI.CONFIRM_SEEDS if CONFIRM else C.SEEDS
OUT = 'fan-in-pilot-v1' + ('' if FI.STAGE == 'stage1' else f'-{FI.STAGE}') + ('-smoke' if X.SMOKE else '')
CONTRASTS = {'confirm': (('read-q16', 'dense'), ('read-q16-untied', 'dense'), ('read-q16-untied', 'read-q16'),
                         ('learn-q16-tied', 'read-q16'), ('learn-q16-untied', 'read-q16-untied')),
             'rigl': (('enc-q16', 'dense'), ('enc-q16-rigl', 'dense'), ('enc-q16-rigl', 'enc-q16'),
                      ('read-q4-untied', 'dense'), ('read-q4-untied-rigl', 'dense'), ('read-q4-untied-rigl', 'read-q4-untied')),
             'rigl1': (('read-q16-untied', 'dense'), ('read-q16-untied-rigl', 'dense'), ('read-q16-untied-rigl', 'read-q16-untied')),
             'fixed': (('read-q16-untied-rigl', 'read-q16-untied'), ('read-q16-untied-rigl', 'dense'),
                       ('read-q16-fixed', 'read-q16-untied-rigl'), ('read-q16-fixed75', 'read-q16-untied-rigl'),
                       ('read-q16-fixed', 'dense'), ('read-q16-fixed75', 'dense'), ('read-q16-fixed', 'read-q16-fixed75')),
             'timing': (('read-q16-gradual', 'read-q16-fixed'), ('read-q16-fixed-rigld', 'read-q16-fixed'),
                        ('read-q16-gradual-rigld', 'read-q16-fixed-rigld'), ('read-q16-gradual-rigld', 'read-q16-gradual'),
                        ('read-q16-gradual', 'read-q16-untied-rigl'), ('read-q16-fixed-rigld', 'read-q16-untied-rigl'),
                        ('read-q16-gradual-rigld', 'read-q16-untied-rigl')),
             'reset': tuple((a, b) for a in ('read-q16-untied-riglP', 'read-q16-untied-riglg', 'read-q16-untied-riglcarry',
                                             'read-q16-untied-rewire', 'read-q16-untied-dead')
                            for b in ('read-q16-untied', 'read-q16-untied-rigl')),
             'erasure': (('read-q16-untied-inplace', 'read-q16-untied'), ('read-q16-untied-inplace', 'read-q16-untied-rewire'),
                         ('read-q16-untied-rewirecarry', 'read-q16-untied'), ('read-q16-untied-rewirecarry', 'read-q16-untied-rewire'),
                         ('dense-inplace', 'dense'), ('read-q16-untied-rewire-half', 'read-q16-untied-rewire'),
                         ('read-q16-untied-rewire-double', 'read-q16-untied-rewire'), ('read-q16-untied-rewire-half', 'read-q16-untied'),
                         ('read-q16-untied-rewire-double', 'read-q16-untied')),
             }.get(FI.STAGE, ())
RERUN = {'timing': ('dense', 'read-q16-untied-rigl', 'read-q16-fixed'), 'reset': ('dense', 'read-q16-untied', 'read-q16-untied-rigl'),
         'erasure': ('dense', 'read-q16-untied', 'read-q16-untied-rewire')}
RERUN_OF = {'timing': 'fixed', 'reset': 'fixed', 'erasure': 'reset'}  # stage the reruns must reproduce
RIDGE = 1e-3
THAI = str(max(P.TRANSFER_AT)) if X.SMOKE else '200'  # smoke runs stop transfer early
PRIMARY = ('Chinese withheld, own block', 'Devanagari withheld, own block', 'taps → Chinese form, withheld',
           'taps → Devanagari form, withheld', 'form → taps, Chinese withheld', 'Comparison, Chinese forms (untaught)',
           'Comparison, Devanagari forms (untaught)', 'Comparison, English forms (untaught)')
TAUGHT = ('Chinese paired, own block', 'taps → collection (taught)', 'Comparison, collections', 'Succession pairwise, collections')


def classify_ci(diffs, kind):
    """Confirmation rule: better/worse if |mean| > margin and the 95% paired t-interval excludes 0; equivalent if the
    interval lies within +-margin; otherwise inconclusive."""
    if kind not in C.MARGIN:
        return 'descriptive'
    d, m = np.asarray(diffs, float), C.MARGIN[kind]
    half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = d.mean() - half, d.mean() + half
    if abs(d.mean()) > m and (lo > 0 or hi < 0):
        return 'better' if d.mean() > 0 else 'worse'
    if -m <= lo and hi <= m:
        return 'equivalent'
    return 'inconclusive'


def interval(diffs):
    d = np.asarray(diffs, float)
    half = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean()), float(d.mean() - half), float(d.mean() + half)]


def rigl_changes(seed, config, enc, model):
    """RigL arms: per limited layer, the fraction of final connections not in the initial mask, the number of grown
    connections, and for readers the mean overlap of the two operands' dimension sets (initial and final)."""
    start = FI.limited_layers(*FI.make_model(seed, config))
    out = {}
    for name, lin in FI.rigl_layers(enc, model).items():
        m0, m = start[name].mask, lin.mask
        d = {'changed': float(((m > 0) & (m0 == 0)).sum() / (m > 0).sum()), 'grown': int(lin.swaps)}
        if name.startswith(('g/', 'P/')):
            ov = lambda M: float((M[:, :FI.CODE] * M[:, FI.CODE:2 * FI.CODE]).sum(1).mean())
            d.update({'overlap_initial': ov(m0), 'overlap_final': ov(m)})
        out[name] = d
    return out


def selection(seed, config, model, enc=None):
    """Learned arms: fraction of reader units whose dimension changed from the start, and reading the same dimension of
    both operands, for g and P; RigL arms: rigl_changes."""
    if config in FI.RIGL_BASE:
        return rigl_changes(seed, config, enc, model)
    if FI.CONFIGS[config][2] not in FI.RANDOM_OF:
        return {}
    _, start = FI.make_model(seed, config)
    out = {}
    for name in ('g', 'P'):
        a, b = getattr(model, name)[0].selections()
        a0, b0 = getattr(start, name)[0].selections()
        out[name] = {'changed': float(((a != a0) | (b != b0)).float().mean()), 'same_dimension': float((a == b).float().mean())}
    return out


def load(seed, config):
    ck = torch.load(ROOT / 'runs' / FI.NAME / f'{config}-seed{seed}.pt', weights_only=False)
    enc, model = FI.make_model(seed, config)
    model.load_state_dict(ck['sri'])
    enc['image'].load_state_dict(ck['image']); enc['sequence'].load_state_dict(ck['sequence'])
    enc['image'].eval(); enc['sequence'].eval()
    return enc, model, ck['withheld'], seed, config


@torch.no_grad()
def probe(enc, model, withheld):
    code = G.codes_fn(enc, model)
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS])  # (9, E, d)
    sets = {'taps': torch.stack([code(D.TAPS[('eval', n)], 'sequence') for n in ns.NUMBERS]),
            **{f'forms/{L}': torch.stack([code(X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]], 'image') for n in ns.NUMBERS])
               for L in P.LANGS}}
    counts = torch.arange(1., 10.)
    Xc = torch.cat([colls.reshape(-1, colls.shape[-1]), torch.ones(colls.shape[0] * colls.shape[1], 1)], 1).double()
    y = counts[:, None].expand(-1, colls.shape[1]).reshape(-1).double()
    w = torch.linalg.solve(Xc.T @ Xc + RIDGE * len(Xc) * torch.eye(Xc.shape[1], dtype=torch.float64), Xc.T @ y)
    pred = lambda c: (torch.cat([c, torch.ones(*c.shape[:-1], 1)], -1).double() @ w).numpy()  # (9, E)
    out = {}
    for name, codes in (('collections', colls), *sets.items()):
        p = pred(codes)
        order = [np.mean((p[a] > p[b]) == (a > b)) for a in range(9) for b in range(9) if a != b]
        out[name] = {'mae': float(np.abs(p - np.arange(1, 10)[:, None]).mean()), 'order': float(np.mean(order))}
        L = name.split('/')[-1]
        if name.startswith('forms/') and withheld.get(L):
            rows = [n - 1 for n in withheld[L]]
            out[name]['mae_withheld'] = float(np.abs(p[rows] - np.array(withheld[L])[:, None]).mean())
    return out


@torch.no_grad()
def coordinate_probe(enc, model):
    """Per code coordinate j: the least-squares affine map from coordinate j of collection codes to count, fitted on
    collections only and applied unchanged to taps and form codes. Returns, per coordinate, the collection R^2 and the
    pairwise order accuracy on each set (constant coordinates: R^2 0, order 0.5). All coordinates are kept; none is
    selected."""
    code = G.codes_fn(enc, model)
    colls = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS]).double()  # (9, E, d)
    sets = {'taps': torch.stack([code(D.TAPS[('eval', n)], 'sequence') for n in ns.NUMBERS]).double(),
            **{f'forms/{L}': torch.stack([code(X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]], 'image') for n in ns.NUMBERS]).double()
               for L in P.LANGS}}
    y = torch.arange(1., 10., dtype=torch.float64)[:, None].expand(-1, colls.shape[1])
    out = {'r2': [], **{k: [] for k in ('collections', *sets)}}
    for j in range(colls.shape[-1]):
        x = colls[..., j]
        xc, yc = x - x.mean(), y - y.mean()
        var = (xc ** 2).mean()
        a = float((xc * yc).mean() / var) if var > 1e-12 else 0.
        b = float(y.mean() - a * x.mean())
        out['r2'].append(float((xc * yc).mean() ** 2 / (var * (yc ** 2).mean())) if var > 1e-12 else 0.)
        for k, v in (('collections', colls), *sets.items()):
            p = (a * v[..., j] + b).numpy()
            out[k].append(float(np.mean([np.mean((p[u] > p[w]) == (u > w)) for u in range(9) for w in range(9) if u != w])))
    return out


def live_readers(active, mask):
    """Live units of a masked reader layer: fraction reading the same dimension of both operands, and the share of their
    code connections on the most-read dimension (and that dimension). Dead units are left out: with zero gradient,
    RigL's top-k puts them all on one dimension by tie-breaking."""
    m = mask[active][:, :2 * FI.CODE]
    a, b = m[:, :FI.CODE], m[:, FI.CODE:]
    reads = (a + b).sum(0)
    return {'tied': float((a * b).sum(1).gt(0).float().mean()), 'modal_share': float(reads.max() / reads.sum()),
            'modal_dim': int(reads.argmax())}


def code_structure(colls, forms, axis):
    """colls (9, E, d); forms {name: (9, E, d)}. Collection variance on the first principal direction and on `axis`, their
    cosine, and each form set's mean offset from the collections' mean outside `axis`, in collection SDs along `axis`."""
    C = colls.reshape(-1, colls.shape[-1]).double()
    C = C - C.mean(0)
    ev, evec = torch.linalg.eigh(torch.cov(C.T))
    pc1 = evec[:, -1]
    if axis is None:
        axis = int(pc1.abs().argmax())
    other = [i for i in range(colls.shape[-1]) if i != axis]
    cm, sd = colls.reshape(-1, colls.shape[-1]).mean(0), colls[..., axis].std()
    return {'axis': axis, 'pc1_share': float(ev[-1] / ev.sum()), 'axis_share': float(C[:, axis].var() / C.var(0).sum()),
            'pc1_axis_cosine': float(pc1[axis].abs()),
            'offset_off_axis': {k: float((v.reshape(-1, v.shape[-1]).mean(0) - cm)[other].norm() / sd) for k, v in forms.items()}}


@torch.no_grad()
def diagnostics(enc, model, seed, config):
    code = G.codes_fn(enc, model)
    colls9 = torch.stack([code(P.DOTS['succ'][n], 'image') for n in ns.NUMBERS])  # (9, E, d)
    by_lang = {L: torch.stack([code(X.DATA[(L, 'eval')][0][P.EVAL[(L, n, 'idx')]], 'image') for n in ns.NUMBERS]) for L in P.LANGS}
    colls = colls9.reshape(-1, colls9.shape[-1])
    forms = torch.cat([v.reshape(-1, v.shape[-1]) for v in by_lang.values()])
    allc = torch.cat([colls, forms])
    ev = torch.linalg.eigvalsh(torch.cov(colls.T.double())).clamp(min=0)
    rng = np.random.default_rng(0)
    i, j = rng.integers(len(allc), size=(2, 4000))
    a, b = allc[i], allc[j]
    g0, p0 = model.g[0], model.P[0]
    active_g = sum((g0(torch.cat([a, b, model.token.weight[h].expand(len(a), -1)], 1)) > 0).any(0) for h in range(3)) > 0
    active_p = (p0(torch.cat([a, b], 1)) > 0).any(0)
    edges = {k: int(lin.mask.sum()) if hasattr(lin, 'mask') else lin.weight.numel() for k, lin in FI.limited_layers(enc, model).items()}
    readers = {n: live_readers(act, lin.mask) for n, act, lin in (('g', active_g, g0), ('P', active_p, p0))
               if hasattr(lin, 'mask') and not bool(lin.mask.all())}  # an all-ones mask is a dense reader
    axis = readers['g']['modal_dim'] if readers else None  # dense: the coordinate with the largest first-direction loading
    return {'participation_ratio_collections': float(ev.sum() ** 2 / (ev ** 2).sum()),
            'g_first_never_active': float(1 - active_g.float().mean()), 'P_first_never_active': float(1 - active_p.float().mean()),
            'edges': edges, 'selection': selection(seed, config, model, enc), 'live_readers': readers,
            'code_structure': code_structure(colls9, by_lang, axis), 'coordinate_probe': coordinate_probe(enc, model)}


def dig(d, path):
    for p in path:
        d = d[p]
    return d


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    run_rt = json.loads((ROOT / 'reports' / f'{FI.NAME}-runtime.json').read_text())
    FI.CHANNELS_LAST = run_rt.get('channels_last_image_encoder', False)  # rebuild models with the run's settings
    torch.set_num_threads(run_rt.get('threads_per_worker', 1))
    X.prepare(); P.prepare(); D.prepare()
    R = C.load(FI.NAME)
    exact = None
    if FI.STAGE in RERUN:  # rerun arms must reproduce an earlier stage exactly
        S = C.load(FI.NAME.replace(f'-{FI.STAGE}', f'-{RERUN_OF[FI.STAGE]}'))
        same = lambda a, b: {k: v for k, v in a.items() if k != 'checkpoint'} == {k: v for k, v in b.items() if k != 'checkpoint'}
        exact = {f'{c}/seed{s}': same(R[f'{c}/seed{s}'], S[f'{c}/seed{s}']) for c in RERUN[FI.STAGE] for s in SEEDS}
    if not CONFIRM:
        S = C.load(SP.NAME)
        same = lambda a, b: {k: v for k, v in a.items() if k != 'checkpoint'} == {k: v for k, v in b.items() if k != 'checkpoint'}
        exact = {s: same(R[f'dense/seed{s}'], S[f'{FI.BASE_ARM}/seed{s}']) for s in SEEDS}
    F, PR, DG = {}, {}, {}
    for cfg in FI.CONFIGS:
        for s in SEEDS:
            enc, model, withheld, _, _ = load(s, cfg)
            f = G.generic_extras(enc, model, withheld)
            f['coherence'] = G.generic_coherence(enc, model)
            F[cfg, s], PR[cfg, s], DG[cfg, s] = f, probe(enc, model, withheld), diagnostics(enc, model, s, cfg)
    iso = lambda key, block=-1: lambda c, s: R[f'{c}/seed{s}']['after_block'][block]['isolated'][key]
    sen = lambda *p: lambda c, s: dig(R[f'{c}/seed{s}']['after_block'][-1]['senses'], p)
    fr = lambda *p: lambda c, s: dig(F[c, s], p)
    measures = [
        ('Chinese withheld, own block', iso('chinese/withheld', 1), 'acc'),
        ('Devanagari withheld, own block', iso('devanagari_digits/withheld', 2), 'acc'),
        ('taps → Chinese form, withheld', sen('cross_sense', 'taps_to_form/chinese/withheld'), 'acc'),
        ('taps → Devanagari form, withheld', sen('cross_sense', 'taps_to_form/devanagari_digits/withheld'), 'acc'),
        ('form → taps, Chinese withheld', sen('cross_sense', 'form_to_taps/chinese/withheld'), 'acc'),
        ('Comparison, Chinese forms (untaught)', sen('comparison', 'forms/chinese'), 'acc'),
        ('Comparison, Devanagari forms (untaught)', sen('comparison', 'forms/devanagari_digits'), 'acc'),
        ('Comparison, English forms (untaught)', sen('comparison', 'forms/english'), 'acc'),
        ('Chinese paired, own block', iso('chinese/paired', 1), 'acc'),
        ('taps → collection (taught)', sen('taps_to_collection'), 'acc'),
        ('Comparison, collections', sen('comparison', 'collections'), 'acc'),
        ('Succession pairwise, collections', fr('succession_pairwise', 'collections'), 'acc'),
        ('Chinese paired, end', iso('chinese/paired'), 'acc'),
        ('English, end', iso('english/paired'), 'acc'),
        ('Judgment revision: isolated, withheld', fr('context/withheld', 'isolated'), 'acc'),
        ('Judgment revision: revised, withheld', fr('context/withheld', 'two_sided_revised'), 'acc'),
        ('Next-judgment coherence AUROC, Chinese', fr('coherence', 'forms/chinese'), 'auroc'),
        ('Next-judgment coherence AUROC, English', fr('coherence', 'forms/english'), 'auroc'),
        ('Thai, 4 shots, after 200 updates', lambda c, s: R[f'{c}/seed{s}']['transfer'][THAI], 'acc'),
        ('Probe order, Chinese forms', lambda c, s: PR[c, s]['forms/chinese']['order'], 'acc'),
        ('Probe order, Devanagari forms', lambda c, s: PR[c, s]['forms/devanagari_digits']['order'], 'acc'),
        ('Probe order, English forms', lambda c, s: PR[c, s]['forms/english']['order'], 'acc'),
        ('Probe order, taps', lambda c, s: PR[c, s]['taps']['order'], 'acc'),
        ('Probe MAE, Chinese withheld forms', lambda c, s: PR[c, s]['forms/chinese']['mae_withheld'], 'descriptive'),
        ('Probe MAE, Devanagari withheld forms', lambda c, s: PR[c, s]['forms/devanagari_digits']['mae_withheld'], 'descriptive'),
        ('Code participation ratio (collections)', lambda c, s: DG[c, s]['participation_ratio_collections'], 'descriptive'),
        ('g first layer, never active', lambda c, s: DG[c, s]['g_first_never_active'], 'descriptive'),
        ('P first layer, never active', lambda c, s: DG[c, s]['P_first_never_active'], 'descriptive'),
        *[(f'{n} live units: {label}', lambda c, s, n=n, k=k: DG[c, s]['live_readers'].get(n, {}).get(k, np.nan), 'descriptive')
          for n in ('g', 'P') for k, label in (('tied', 'same dimension of both operands'), ('modal_share', 'share on most-read dimension'))],
        ('Code: collection variance on first direction', lambda c, s: DG[c, s]['code_structure']['pc1_share'], 'descriptive'),
        ('Code: collection variance on reader axis', lambda c, s: DG[c, s]['code_structure']['axis_share'], 'descriptive'),
        ('Code: cosine of first direction with reader axis', lambda c, s: DG[c, s]['code_structure']['pc1_axis_cosine'], 'descriptive'),
        ('Single coordinates: mean collection R²', lambda c, s: np.mean(DG[c, s]['coordinate_probe']['r2']), 'descriptive'),
        ('Single coordinates: count with collection R² ≥ 0.5', lambda c, s: np.sum(np.array(DG[c, s]['coordinate_probe']['r2']) >= .5), 'descriptive'),
        *[(f'Single coordinates: mean order, {k}', lambda c, s, k=k: np.mean(DG[c, s]['coordinate_probe'][k]), 'descriptive')
          for k in ('collections', 'taps', *[f'forms/{L}' for L in P.LANGS])],
        *[(f'Code: {L} forms offset off axis', lambda c, s, L=L: DG[c, s]['code_structure']['offset_off_axis'][L], 'descriptive')
          for L in P.LANGS],
    ]
    rows = []
    cls = classify_ci if CONFIRM else C.classify
    for name, get, kind in measures:
        v = {c: [get(c, s) for s in SEEDS] for c in FI.CONFIGS}
        rows.append({'measure': name, 'kind': kind, 'values': v,
                     'vs_dense': {c: cls(np.subtract(v[c], v['dense']), kind) for c in FI.CONFIGS if c != 'dense'}})
        if CONFIRM:
            rows[-1]['contrasts'] = {f'{a} - {b}': {'outcome': cls(np.subtract(v[a], v[b]), kind), 'mean_ci': interval(np.subtract(v[a], v[b]))}
                                     for a, b in CONTRASTS}
    by = {r['measure']: r for r in rows}
    decision = {}
    for c in FI.CONFIGS:
        if c == 'dense' or CONFIRM:
            continue
        drops = [m for m in TAUGHT if np.mean(by[m]['values'][c]) - np.mean(by[m]['values']['dense']) < -C.MARGIN['acc']]
        better = [m for m in PRIMARY if by[m]['vs_dense'][c] == 'better']
        worse = [m for m in PRIMARY if by[m]['vs_dense'][c] == 'worse']
        decision[c] = {'underfit_on': drops, 'better_on': better, 'worse_on': worse,
                       'advance': not drops and len(better) >= 2 and not worse}
    out = {'seeds': SEEDS, 'margins': C.MARGIN, 'configs': FI.CONFIGS, 'dense_reproduces_sequence_pilot': exact,
           'measures': rows, 'decision': decision, 'probe': {f'{c}/seed{s}': PR[c, s] for c, s in PR},
           'diagnostics': {f'{c}/seed{s}': DG[c, s] for c, s in DG}}
    with gzip.open(ROOT / 'reports' / f'{OUT}-comparison.json.gz', 'wt') as f:
        f.write(json.dumps(out, indent=1, default=float) + '\n')
    (ROOT / 'reports' / f'{OUT}-comparison-runtime.json').write_text(json.dumps({
        'scope': f'nondeterministic timing; results in {OUT}-comparison.json.gz', 'module_sha256': X.nc.sha256(__file__),
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain')), 'command': ' '.join([sys.executable, *sys.argv]),
        'wall_seconds': time.perf_counter() - t0, 'cpu_seconds': time.process_time() - c0}, indent=1) + '\n')
    cfgs = [c for c in FI.CONFIGS if c != 'dense']
    print('reproduces earlier runs:', exact if exact is None else all(exact.values()), exact)
    print('| Measure | dense | ' + ' | '.join(cfgs) + ' |\n|' + ' --- |' * (len(cfgs) + 2))
    for r in rows:
        cell = lambda c: f"{np.mean(r['values'][c]):.2f}" + ('' if r['kind'] == 'descriptive' else f" {r['vs_dense'][c][:3]}")
        print(f"| {r['measure']} | {np.mean(r['values']['dense']):.2f} | " + ' | '.join(cell(c) for c in cfgs) + ' |')
    if CONFIRM:
        print('\n| Measure | ' + ' | '.join(f'{a} − {b}' for a, b in CONTRASTS) + ' |\n|' + ' --- |' * (len(CONTRASTS) + 1))
        for r in rows:
            if r['kind'] != 'descriptive':
                print(f"| {r['measure']} | " + ' | '.join(
                    '{:+.2f} [{:+.2f}, {:+.2f}] {}'.format(*r['contrasts'][f'{a} - {b}']['mean_ci'], r['contrasts'][f'{a} - {b}']['outcome'][:3])
                    for a, b in CONTRASTS) + ' |')
        for c in FI.CONFIGS:
            sel = [DG[c, s]['selection'] for s in SEEDS if DG[c, s]['selection']]
            if sel:
                print(c, {f'{n}/{k}': round(float(np.mean([x[n][k] for x in sel])), 3) for n in sel[0] for k in sel[0][n]})
    print('\nedges', {c: DG[c, SEEDS[0]]['edges'] for c in FI.CONFIGS})
    for c, d in decision.items():
        print(c, d)


if __name__ == '__main__':
    main()
