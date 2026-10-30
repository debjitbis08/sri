"""Analysis of the limited-memory study under the protocol of docs/limited-memory-proposal.md ("Selection and confirmation",
"Checkpoint fork for handoff").

main      per-arm endpoints (a)-(d), the selection rule, contrasts against the declared comparators, relearning curve
          descriptives, reader-disruption descriptives
confirm   the same endpoints and the declared contrasts of the confirmation run; decisions are made on this run alone
fork      decay against no decay from the common parent: full cohort primary, the memory-dependent subset conditional

Every declared arm and all ten seeds are required before any selection, contrast or fork reading. With
NLM_DESCRIPTIVE=1 an incomplete cohort gives per-arm means only, labeled descriptive, in a separate file: no selection,
no intervals, no outcomes.

Usage: .venv/bin/python -m experiments.limited_memory_analysis pilot main|confirm|fork [tag-suffix via NFI_TAG]
Timing goes to <name>-analysis-runtime.json, never into the analysis JSON.
"""
import gzip
import json
import os
import sys
import time

import numpy as np

from experiments import limited_memory as LM
from experiments import fan_in_pilot_analysis as AN
from experiments import number_systems as X

ROOT, P = X.ROOT, LM.P
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
MARGIN = .05
MIN_ELIGIBLE = 5
DESCRIPTIVE = os.environ.get('NLM_DESCRIPTIVE') == '1'
OUT = ROOT / 'reports' / f'{LM.NAME}-analysis{"-descriptive" if DESCRIPTIVE else ""}.json'
RUNTIME = ROOT / 'reports' / f'{LM.NAME}-analysis{"-descriptive" if DESCRIPTIVE else ""}-runtime.json'


def require_complete(R):
    """The declared cohort: every arm of the stage, every seed. Returns the missing keys."""
    missing = [f'{a}/seed{s}' for a in LM.arms_for_stage() for s in LM.SEEDS if f'{a}/seed{s}' not in R]
    if missing and not DESCRIPTIVE:
        raise SystemExit(f'incomplete cohort: {len(missing)} of {len(LM.arms_for_stage()) * len(LM.SEEDS)} runs missing '
                         f'(first: {missing[:3]}). Rerun the stage, or set NLM_DESCRIPTIVE=1 for a labeled descriptive summary.')
    return missing


def load(name):
    return json.load(gzip.open(ROOT / 'reports' / f'{name}-results.json.gz', 'rt'))['results']


def a_endpoint(res):
    """(a) at the end of block 3: order accuracy of the collection-fitted count map, averaged over the 16 coordinates within
    taps, Chinese forms and Devanagari forms, then over the three sets with equal weight."""
    probe = res['snapshots']['block3']['probe']
    return float(np.mean([np.mean(probe[k]) for k in TRANSFER_SETS]))


def b_endpoint(res):
    return res['snapshots']['block2']['evaluate']['isolated']['chinese/paired']


def c_endpoint(res):
    fw = res['snapshots']['block3']['four_way']
    return fw['mem_silenced']['matching']['chinese/paired']['all'] if 'mem_silenced' in fw else None


def relearning(res):
    t = sorted(res['relearning'], key=int)
    a = [res['relearning'][k]['matching']['english/paired']['all'] for k in t]
    x = np.array([int(k) for k in t], float)
    gain = float(np.trapezoid(np.array(a) - a[0], x) / x[-1])
    return {'A0': a[0], 'A_final': a[-1], 'G': gain, 'curve': dict(zip(t, a))}


def measures(res):
    s = res['snapshots']
    out = {'a_avg_coordinate_transfer': a_endpoint(res), 'b_chinese_paired_block2': b_endpoint(res), 'c_structure_only_chinese': c_endpoint(res),
           'english_matching_after_delay': s['delay']['four_way']['full']['matching']['english/paired']['all']}
    fw = s['block3']['four_way']
    if 'mem_silenced' in fw:  # interaction residual f(s,m) - f(s,0) - f(0,m) + f(0,0) of Chinese paired matching
        cz = lambda k: fw[k]['matching']['chinese/paired']['all']
        out['interaction_chinese_block3'] = cz('full') - cz('mem_silenced') - cz('struct_silenced') + cz('both_silenced')
    else:
        out['interaction_chinese_block3'] = None
    out.update({f'transfer8/{k}': v for k, v in fw['full']['transfer8'].items()})
    out.update({f'd_{k}': v for k, v in relearning(res).items() if k != 'curve'})
    return out


def paired(R, a, b, name, seeds, kind='acc'):
    va = [measures(R[f'{a}/seed{s}'])[name] for s in seeds]
    vb = [measures(R[f'{b}/seed{s}'])[name] for s in seeds]
    if any(v is None for v in va + vb):  # e.g. structure-only accuracy is undefined for arms without a memory/structure split
        return {'outcome': 'inapplicable', 'reason': f'{name} is not defined for {a if None in va else b}'}
    d = np.subtract(va, vb)
    return {'outcome': AN.classify_ci(d, kind), 'mean_ci': AN.interval(d)}


def main_like(stage):
    R = load(LM.NAME)
    missing = require_complete(R)
    arms = [a for a in LM.arms_for_stage() if any(f'{a}/seed{s}' in R for s in LM.SEEDS)]
    seeds = [s for s in LM.SEEDS if all(f'{a}/seed{s}' in R for a in arms)] if missing else list(LM.SEEDS)
    if not seeds:
        raise SystemExit('no seed has every arm; nothing to summarize')
    M = {a: {s: measures(R[f'{a}/seed{s}']) for s in seeds} for a in arms}
    mean = lambda a, k: float(np.mean([M[a][s][k] for s in seeds]))
    names = list(M[arms[0]][seeds[0]])
    out = {'status': 'descriptive: incomplete cohort, no selection, intervals or outcomes' if missing else 'complete cohort',
           'missing_runs': missing, 'seeds': seeds, 'arms': arms,
           'means': {a: {k: mean(a, k) for k in names if M[a][seeds[0]][k] is not None} for a in arms}}
    if missing:
        return out
    decaying = [a for a in LM.HALF_LIFE if a in arms]
    if stage == 'main' and decaying and 'sparse48' in arms:
        gate = {a: mean('sparse48', 'b_chinese_paired_block2') - mean(a, 'b_chinese_paired_block2') <= MARGIN for a in decaying}
        ok = [a for a in decaying if gate[a]]
        # highest seed-mean of (a); ties go to the longer half-life
        sel = max(ok, key=lambda a: (round(mean(a, 'a_avg_coordinate_transfer'), 12), LM.HALF_LIFE[a])) if ok else None
        out['selection'] = {'metric': 'a_avg_coordinate_transfer', 'not_underfit': gate, 'selected': sel,
                            'means': {a: mean(a, 'a_avg_coordinate_transfer') for a in decaying}}
    contrasts = [(a, b) for a in decaying for b in ('sparse48', 'sparse64', 'mixed') if b in arms]
    contrasts += [('mixed', b) for b in ('dense', 'sparse64') if b in arms and 'mixed' in arms]
    contrasts += [(a, 'mixed-reset') for a in decaying if 'mixed-reset' in arms]
    if stage == 'confirm' and 'mixed' in arms:
        contrasts = [(a, b) for a in decaying for b in ('sparse48', 'mixed', 'mixed-reset') if b in arms]
    tested = [k for k in names if k.startswith('transfer8/') or k in ('a_avg_coordinate_transfer', 'b_chinese_paired_block2',
                                                                      'c_structure_only_chinese', 'english_matching_after_delay')]
    out['contrasts'] = {f'{a} - {b}': {k: paired(R, a, b, k, seeds) for k in tested} for a, b in contrasts}
    out['relearning_descriptive'] = {a: {k: float(np.mean([M[a][s][f'd_{k}'] for s in seeds])) for k in ('A0', 'A_final', 'G')} for a in arms}
    out['disruption_descriptive'] = {a: {pt: {k: float(np.mean([R[f'{a}/seed{s}']['snapshots'][pt]['disruption'][k] for s in seeds]))
                                              for k in R[f'{a}/seed{seeds[0]}']['snapshots'][pt]['disruption']}
                                         for pt in ('block3', 'delay', 'relearn')} for a in arms}
    return out


def fork_like():
    R = load(LM.NAME)
    missing = require_complete(R)
    branches = list(LM.FORK)
    seeds = [s for s in LM.SEEDS if all(f'{b}/seed{s}' in R for b in branches)]
    if not seeds:
        raise SystemExit('no seed has every branch; nothing to summarize')
    cell = lambda row, part, half='odd': row[part]['chinese/paired'][half]
    parent = {s: R[f'fork-hinf/seed{s}']['track'][0] for s in seeds}  # identical in every branch
    drop_even = {s: cell(parent[s], 'full', 'even') - cell(parent[s], 'mem_silenced', 'even') for s in seeds}
    eligible = [s for s in seeds if drop_even[s] > MARGIN]  # applied once, at the parent, on separate (even) items

    def series(b, s, key):
        out = []
        for row in R[f'{b}/seed{s}']['track']:
            struct_only = cell(row, 'mem_silenced')
            out.append(struct_only if key == 'structure_only' else cell(row, 'full') - struct_only)
        return np.array(out)

    res = {'seeds': seeds, 'eligible': eligible, 'n_eligible': len(eligible), 'n_total': len(seeds), 'drop_even_at_parent': drop_even,
           'conditional_inferential': len(eligible) >= MIN_ELIGIBLE, 'updates': [r['update'] for r in R['fork-hinf/seed%d' % seeds[0]]['track']],
           'status': 'descriptive: incomplete cohort, no intervals or outcomes' if missing else 'complete cohort',
           'missing_runs': missing, 'branches': {}}
    for b in branches[1:]:
        if missing:  # descriptive: mean change from the no-decay branch, nothing inferential
            res['branches'][b] = {key: {'mean_over_points': np.mean([series(b, s, key) - series('fork-hinf', s, key) for s in seeds], 0).tolist()}
                                  for key in ('structure_only', 'dependence')}
            continue
        res['branches'][b] = {}
        for key in ('structure_only', 'dependence'):
            for name, group in (('full_cohort', seeds), ('eligible', eligible)):
                if len(group) < 2:
                    continue
                d = np.array([series(b, s, key) - series('fork-hinf', s, key) for s in group])  # (seeds, points)
                last = d[:, -1]
                cell_out = {'mean_ci_at_end': AN.interval(last) if len(group) >= MIN_ELIGIBLE or name == 'full_cohort' else None,
                            'mean_over_points': d.mean(0).tolist()}
                if name == 'full_cohort' or res['conditional_inferential']:
                    cell_out['outcome'] = AN.classify_ci(last, 'acc')
                res['branches'][b][f'{key}/{name}'] = cell_out
        pair = res['branches'][b]
        for name in ('full_cohort', 'eligible'):
            so, dep = pair.get(f'structure_only/{name}'), pair.get(f'dependence/{name}')
            if so and dep and 'outcome' in so and 'outcome' in dep:
                pair[f'consistent_with_handoff/{name}'] = so['outcome'] == 'better' and dep['outcome'] == 'worse'
    return res


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    out = fork_like() if LM.STAGE == 'fork' else main_like(LM.STAGE)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    results = ROOT / 'reports' / f'{LM.NAME}-results.json.gz'
    RUNTIME.write_text(json.dumps({
        'scope': 'nondeterministic timing; the analysis JSON holds the scientific content only',
        'measured': 'loading the results file, computing all endpoints and contrasts, writing the analysis JSON',
        'wall_seconds': time.perf_counter() - t0, 'process_cpu_seconds': time.process_time() - c0,
        'input': {str(results.relative_to(ROOT)): LM.nc.sha256(results)},
        'sources_sha256': {m: LM.nc.sha256(ROOT / 'experiments' / f'{m}.py') for m in ('limited_memory_analysis', 'limited_memory')},
        'output_sha256': LM.nc.sha256(OUT), 'command': ' '.join(sys.argv), 'descriptive': DESCRIPTIVE,
        'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}, indent=1, sort_keys=True) + '\n')
    print('wrote', OUT.relative_to(ROOT), '|', out['status'])
    if LM.STAGE == 'fork':
        print(f"eligible {out['n_eligible']}/{out['n_total']}; conditional inferential: {out['conditional_inferential']}")
        for b, v in out['branches'].items():
            print(b, {k: (x.get('outcome'), x.get('mean_ci_at_end') and [round(c, 3) for c in x['mean_ci_at_end']]) for k, x in v.items() if isinstance(x, dict)},
                  {k: x for k, x in v.items() if k.startswith('consistent')})
    else:
        print('selection:', out.get('selection'))
        for a, m in out['means'].items():
            print(f"{a:14s} a={m['a_avg_coordinate_transfer']:.3f} b={m['b_chinese_paired_block2']:.3f} c={m.get('c_structure_only_chinese', float('nan')):.3f} "
                  f"G={m['d_G']:.3f}")


if __name__ == '__main__':
    main()
