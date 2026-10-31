"""Analysis of retention remedies v1 under the frozen contract (docs/retention-remedies-v1-contract.md, "Readings").

Ten seeds (350-359), arms sparse48, mixed, mixed-h125. Paired 95% t-intervals (fan_in_pilot_analysis.interval). The contract's stronger rule: an effect is
"established meaningful" if the lower bound exceeds +0.05; noninferior if the lower bound exceeds -0.05. "Not beneficial" is read as: a meaningful benefit is
excluded (upper bound at most +0.05). The inherited classify_ci outcome is reported alongside. Timing-based resource readings go to the -analysis-runtime.json.

Usage: .venv/bin/python -m experiments.retention_remedies_analysis
"""
import gzip
import json
import time

import numpy as np

from experiments import fan_in_pilot_analysis as AN
from experiments import number_systems as X

ROOT = X.ROOT
SEEDS = list(range(350, 360))
ARMS = ('sparse48', 'mixed', 'mixed-h125')
MARGIN = .05
TRANSFER_SETS = ('taps', 'forms/chinese', 'forms/devanagari_digits')
OUT = ROOT / 'reports' / 'retention-remedies-v1-analysis.json'
RUNTIME = ROOT / 'reports' / 'retention-remedies-v1-analysis-runtime.json'
CONSOLIDATION = ('consolidate-cache', 'consolidate-cache1', 'consolidate-random', 'consolidate-other', 'consolidate-cache-anchor', 'consolidate-cache1-anchor')
BUDGET_FRACTION = .25


def strong(d):
    d = np.asarray(d, float)
    m, lo, hi = AN.interval(d)
    outcome = 'established meaningful' if lo > MARGIN else ('not beneficial' if hi <= MARGIN else 'inconclusive')
    return {'mean_ci': [m, lo, hi], 'outcome': outcome, 'inherited_rule': AN.classify_ci(d, 'acc'), 'noninferior': bool(lo > -MARGIN)}


def ci(d):
    return {'mean_ci': AN.interval(np.asarray(d, float)), 'noninferior': bool(AN.interval(np.asarray(d, float))[1] > -MARGIN)}


def a_of(snap):
    return float(np.mean([np.mean(snap['probe'][k]) for k in TRANSFER_SETS]))


def context_perf(snap):
    e = snap['evaluate']
    return float(np.mean([e['senses']['taps_to_collection'], e['senses']['comparison']['collections'], e['succession_pairwise']['collections']]))


def at(rec, t, key):
    return next(p[key] for p in rec['curve'] if p['offset'] == t)


def main():
    t0, c0 = time.perf_counter(), time.process_time()
    R = json.load(gzip.open(ROOT / 'reports' / 'retention-remedies-v1-results.json.gz', 'rt'))['results']
    missing = [f'{a}/seed{s}' for a in ARMS for s in SEEDS if f'{a}/seed{s}' not in R]
    if missing:
        raise SystemExit(f'incomplete cohort: {missing}')
    rec = lambda a, s: R[f'{a}/seed{s}']
    out = {'seeds': SEEDS, 'arms': ARMS, 'readings': {}, 'descriptive': {}}
    timing = {}

    # 1. checks
    checks = {}
    for a in ARMS:
        c = {'destination_hash_unchanged': True, 'path_unchanged': True, 'buffer_within_M': True, 'marked_balanced_2_per_count': True, 'ctx_declines_on_M': None}
        for s in SEEDS:
            r = rec(a, s)
            c['destination_hash_unchanged'] &= all(r['static'][k]['hash_unchanged'] for k in CONSOLIDATION)
            if 'stable-path-replay' in r['branches']:
                c['path_unchanged'] &= bool(r['branches']['stable-path-replay']['path_unchanged'])
            c['buffer_within_M'] &= set(r['buffer']) <= set(r['marked']) and all(len(v) == 4 for v in r['buffer'].values())
            counts = list(r['marked'].values())
            c['marked_balanced_2_per_count'] &= len(counts) == 18 and all(counts.count(n) == 2 for n in range(1, 10))
        c['ctx_declines_on_M'] = ci([at(rec(a, s)['branches']['ctx'], 2000, 'M') - at(rec(a, s)['branches']['ctx'], 0, 'M') for s in SEEDS])
        checks[a] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in c.items()}
    out['readings']['1_checks'] = checks

    def m_end(a, s, b):
        r = rec(a, s)
        if b in r['branches']:
            return at(r['branches'][b], 2000, 'M')
        if b == 'frozen-snapshot':
            return r['static'][b]['recall_M']
        return r['static'][b]['acc_M']

    def u_end(a, s, b):
        r = rec(a, s)
        if b in r['branches']:
            return at(r['branches'][b], 2000, 'U')
        if b == 'frozen-snapshot':
            return r['static'][b]['recall_U']
        return r['static'][b]['acc_U']

    def m_start(a, s, b):
        r = rec(a, s)
        if b in r['branches']:
            return at(r['branches'][b], 0, 'M')
        if b == 'frozen-snapshot':
            return r['static']['teacher']['original_M']
        return r['static']['teacher']['canonical_M']

    branches = {a: [b for b in rec(a, SEEDS[0])['branches'] if b != 'ctx'] + ['frozen-snapshot', *CONSOLIDATION] for a in ARMS}
    vec = lambda f, a, b: np.array([f(a, s, b) for s in SEEDS])

    # 2. retention benefit, primary contrast and compression
    out['readings']['2_retention_benefit'] = {a: {b: strong(vec(m_end, a, b) - vec(m_end, a, 'ctx')) for b in branches[a]} for a in ARMS}
    prim = vec(m_end, 'mixed', 'consolidate-cache') - vec(m_end, 'mixed', 'frozen-snapshot')
    pb = [rec('mixed', s)['static']['consolidate-cache']['persistent_bytes'] / rec('mixed', s)['static']['frozen-snapshot']['persistent_bytes'] for s in SEEDS]
    fmt = vec(lambda a_, s_, b_: rec(a_, s_)['static']['teacher']['canonical_M'], 'mixed', None) - vec(lambda a_, s_, b_: rec(a_, s_)['static']['teacher']['original_M'], 'mixed', None)
    dist = vec(m_end, 'mixed', 'consolidate-cache') - vec(lambda a_, s_, b_: rec(a_, s_)['static']['teacher']['canonical_M'], 'mixed', None)
    out['post_hoc'] = {'primary_decomposition': {'note': 'post hoc: the destination is scored nine-way on canonical candidates, the snapshot by the original matching protocol; '
                                                         'the primary difference = evaluation format (teacher canonical - teacher original) + distillation (destination - teacher canonical)',
                                                 'format': ci(fmt), 'distillation': ci(dist)}}
    out['readings']['2_primary'] = {'consolidate-cache_minus_frozen-snapshot_on_M': ci(prim), 'noninferior': bool(AN.interval(prim)[1] > -MARGIN),
                                    'inherited_rule': AN.classify_ci(prim, 'acc'),
                                    'compression_persistent_bytes_fraction_of_snapshot': float(np.mean(pb)),
                                    'bytes': {'consolidate-cache': rec('mixed', SEEDS[0])['static']['consolidate-cache']['persistent_bytes'],
                                              'frozen-snapshot': rec('mixed', SEEDS[0])['static']['frozen-snapshot']['persistent_bytes']}}

    # 3. preservation
    pres = {}
    for a in ARMS:
        pres[a] = {}
        for b in branches[a]:
            if b == 'frozen-snapshot':
                pres[a][b] = {'outcome': 'preserved by construction (no update)'}
                continue
            d = vec(m_end, a, b) - vec(m_start, a, b)
            iv = AN.interval(d)
            benefit = out['readings']['2_retention_benefit'][a][b]['outcome'] == 'established meaningful'
            pres[a][b] = {'mean_ci': iv, 'outcome': 'preserved' if iv[1] > -MARGIN else ('partial rescue' if benefit else 'inconclusive')}
        d = vec(m_end, a, 'ctx') - vec(m_start, a, 'ctx')
        pres[a]['ctx'] = {'mean_ci': AN.interval(d)}
    out['readings']['3_preservation'] = pres

    # 4. selectivity
    out['readings']['4_selectivity'] = {a: {b: ci(vec(m_end, a, b) - vec(u_end, a, b)) for b in ['ctx'] + branches[a]} for a in ARMS}

    # 5. cue sources (mixed prespecified; other arms descriptive)
    cue = {}
    for a in ARMS:
        cue[a] = {f'consolidate-cache - {x}': strong(vec(m_end, a, 'consolidate-cache') - vec(m_end, a, x)) for x in ('consolidate-random', 'consolidate-other', 'consolidate-cache1')}
        cue[a]['status'] = 'prespecified' if a == 'mixed' else 'descriptive'
    out['readings']['5_cue_sources'] = cue

    # 6. cost gates
    tr = lambda a, s, b: a_of(rec(a, s)['branches'][b]['snapshot'])
    newB = lambda a, s, b: rec(a, s)['branches'][b]['new_B'][-1]['B']
    cp = lambda a, s, b: context_perf(rec(a, s)['branches'][b]['snapshot'])
    gates = {}
    for a in ARMS:
        gates[a] = {}
        for b in ['ctx'] + [x for x in rec(a, SEEDS[0])['branches'] if x != 'ctx']:
            g = {'i_transfer_vs_sparse48_ctx': ci(vec(tr, a, b) - vec(tr, 'sparse48', 'ctx'))}
            if b != 'ctx':
                g['ii_new_learning_B'] = ci(vec(newB, a, b) - vec(newB, a, 'ctx'))
                g['iii_context_performance'] = ci(vec(cp, a, b) - vec(cp, a, 'ctx'))
            g['all_noninferior'] = all(v['noninferior'] for v in g.values() if isinstance(v, dict))
            gates[a][b] = g
        gates[a]['consolidation_branches'] = {'note': 'the host is the ctx branch (consolidation does not touch it): gates (ii) and (iii) are zero by construction; gate (i) is ctx\'s',
                                              'i_transfer_vs_sparse48_ctx': gates[a]['ctx']['i_transfer_vs_sparse48_ctx']}
    out['readings']['6_cost_gates'] = gates

    # 7. resources (bytes; time in the runtime file)
    st = rec('mixed', SEEDS[0])['static']['storage']
    budget = st['model_bytes']
    pool = 18 * 64 * st['rendering_bytes']
    pbytes = {'ctx-m0': 0, 'hold-M': pool, 'rehearse-M': st['buffer_bytes'], 'rehearse-M-m0': st['buffer_bytes'], 'der-M': st['buffer_bytes'] + st['der_extra_bytes'],
              'der-M-strong': st['buffer_bytes'] + st['der_extra_bytes'], 'stable-path-replay': st['stored_code_bytes'],
              'frozen-snapshot': rec('mixed', SEEDS[0])['static']['frozen-snapshot']['persistent_bytes'],
              **{k: rec('mixed', SEEDS[0])['static'][k]['persistent_bytes'] for k in CONSOLIDATION}}
    tbytes = {k: rec('mixed', SEEDS[0])['static'][k]['temporary_bytes'] for k in CONSOLIDATION}
    out['readings']['7_resources_bytes'] = {'budget_bytes': budget, 'persistent_bytes': pbytes, 'within_budget': {k: v <= budget for k, v in pbytes.items()},
                                            'temporary_bytes': tbytes, 'note': 'hold-M counts the full rendering pool of the marked pairs (18 x 64 renderings)'}
    for a in ARMS:
        ctxw = np.array([rec(a, s)['branches']['ctx']['wall_seconds'] for s in SEEDS])
        timing[a] = {b: float(np.mean((np.array([rec(a, s)['branches'][b]['wall_seconds'] for s in SEEDS]) - ctxw) / ctxw))
                     for b in rec(a, SEEDS[0])['branches'] if b != 'ctx'}
        timing[a].update({k: float(np.mean([rec(a, s)['static'][k]['wall_seconds'] for s in SEEDS] / ctxw)) for k in CONSOLIDATION})
    within_time = {a: {b: v <= BUDGET_FRACTION for b, v in t.items()} for a, t in timing.items()}

    # 8. verdicts
    ver = {'useful_remedy': {a: {b: 'exploratory'  # the primary contrast is the noninferiority to frozen-snapshot (reading 2), not a benefit over ctx
                                 for b, v in out['readings']['2_retention_benefit'][a].items() if v['outcome'] == 'established meaningful'} for a in ARMS}}
    cand = {}
    for a in ARMS:
        cand[a] = {}
        for b in CONSOLIDATION:
            ok_p = pres[a][b]['outcome'] == 'preserved'
            ok_g = gates[a]['ctx']['i_transfer_vs_sparse48_ctx']['noninferior']
            ok_r = pbytes[b] <= budget and within_time[a][b]
            cand[a][b] = {'preserved': ok_p, 'host_gates': ok_g, 'resources_within_budget': bool(ok_r), 'candidate': bool(ok_p and ok_g and ok_r)}
    ver['protected_store_candidate'] = cand
    ver['first_moment_reset'] = {'rehearse-M-m0_minus_rehearse-M_on_M_2000': ci(vec(m_end, 'mixed', 'rehearse-M-m0') - vec(m_end, 'mixed', 'rehearse-M')),
                                 'ctx-m0_minus_ctx_D50': {a: ci([(at(rec(a, s)['branches']['ctx-m0'], 50, 'M') - at(rec(a, s)['branches']['ctx-m0'], 0, 'M'))
                                                                 - (at(rec(a, s)['branches']['ctx'], 50, 'M') - at(rec(a, s)['branches']['ctx'], 0, 'M')) for s in SEEDS])
                                                          for a in ('mixed', 'mixed-h125')}}
    out['readings']['8_verdicts'] = ver

    # 9. descriptive
    desc = {}
    for a in ARMS:
        r0 = rec(a, SEEDS[0])
        desc[a] = {'M_trajectory': {b: {str(t): float(np.mean([at(rec(a, s)['branches'][b], t, 'M') for s in SEEDS])) for t in r0['points']} for b in r0['branches']},
                   'U_trajectory': {b: {str(t): float(np.mean([at(rec(a, s)['branches'][b], t, 'U') for s in SEEDS])) for t in r0['points']} for b in r0['branches']},
                   'teacher': {k: float(np.mean([rec(a, s)['static']['teacher'][k] for s in SEEDS])) for k in r0['static']['teacher']},
                   'static': {b: {k: float(np.mean([rec(a, s)['static'][b][k] for s in SEEDS])) for k in ('acc_M', 'acc_U', 'agreement_M', 'agreement_U', 'fit_to_teacher_on_cues')}
                              for b in CONSOLIDATION},
                   'B_after_250': {b: float(np.mean([newB(a, s, b) for s in SEEDS])) for b in r0['branches']},
                   'transfer_a': {b: float(np.mean([tr(a, s, b) for s in SEEDS])) for b in r0['branches']},
                   'context_performance': {b: float(np.mean([cp(a, s, b) for s in SEEDS])) for b in r0['branches']}}
    out['descriptive'] = desc
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + '\n')
    RUNTIME.write_text(json.dumps({'command': 'retention_remedies_analysis', 'wall_seconds': time.perf_counter() - t0, 'process_cpu_seconds': time.process_time() - c0,
                                   'extra_time_fraction_over_ctx': timing, 'within_25_percent': within_time,
                                   'scope': 'nondeterministic timing (reading 7 time part); the analysis JSON holds the deterministic content',
                                   'revision': X.git('rev-parse', 'HEAD'), 'dirty': bool(X.git('status', '--porcelain'))}, indent=1) + '\n')
    print('wrote', OUT.relative_to(ROOT))
    print('checks', {a: {k: (v if not isinstance(v, dict) else round(v['mean_ci'][0], 3)) for k, v in c.items()} for a, c in checks.items()})
    p = out['readings']['2_primary']
    print('PRIMARY cache - snapshot', [round(x, 3) for x in p['consolidate-cache_minus_frozen-snapshot_on_M']['mean_ci']], 'NI', p['noninferior'], 'bytes fraction', round(p['compression_persistent_bytes_fraction_of_snapshot'], 3))
    for a in ARMS:
        print('==', a)
        for b, v in out['readings']['2_retention_benefit'][a].items():
            print(f'  {b:26s} benefit {v["mean_ci"][0]:+.3f} [{v["mean_ci"][1]:+.3f},{v["mean_ci"][2]:+.3f}] {v["outcome"]:22s} pres {pres[a][b]["outcome"]:12s} sel {out["readings"]["4_selectivity"][a][b]["mean_ci"][0]:+.3f}'
                  + (f'  gates {gates[a][b]["all_noninferior"]}' if b in gates[a] else ''))
        print('  cues', {k: (round(v['mean_ci'][0], 3), v['outcome']) for k, v in cue[a].items() if k != 'status'})
        print('  candidates', {b: v['candidate'] for b, v in cand[a].items()})
    print('time fraction', {a: {b: round(v, 3) for b, v in t.items()} for a, t in timing.items()})


if __name__ == '__main__':
    main()
