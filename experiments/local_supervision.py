"""Local learning supervision v1: conditions O, I (v1 references), IB and O+ with the block learner fixed.

Run `.venv/bin/python -m experiments.local_supervision [--out-dir DIR] [--workers N]`.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import unittest

from experiments.mechanism_learning import (
    ALL_Q, FORWARD, GATE_TEMPLATE, MECHANISM_ITEMS, NEGATIVE, POSITIVE, TEACHERS, Evaluator, Index, atom_hash,
    build_battery, canonical, ceilings, completion_status, decode, dump, ev_fwd, local_gate, members, mirror,
    oracle_gate, ratio, reproduce_frozen, sha, write_rows_report,
)
from src.local_supervision import (
    CONDITIONS, SLOTS, axis_phase, coverage_subset, feedback_cost, make_simulator, observation_keys, oplus_order,
    pool_timelines, project, timeline,
)
from src.mechanism_learning import (
    N_VECTORS, QUERIES, TEMPLATES, BlockLearner, Memory, VersionSpace, hypothesis_index, popcount, vector,
)
from src.shared_input_mediation import A, B


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'local-learning-supervision-v1'
CONTRACT = 'docs/local-learning-supervision-v1-contract.md'
V1 = 'learned-input-mediation-v1'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
CHECKPOINTS = {'O': (0, 5, 10, 15), 'I': (0, 5, 10, 15), 'IB': (0, 5, 10, 15),
               'O+': (0, 5, 10, 15, 18, 21, 24, 27)}
FINAL = {c: ks[-1] for c, ks in CHECKPOINTS.items()}
V1_ARMS = ('O', 'I')
SIG = {'O': ('B1',), 'O+': ('B1',), 'I': ('B1', 'B6', 'B7'), 'IB': ('B1', 'B6x', 'B7x')}
RULES = [
    'battery, teachers, templates, learners and update rules exactly as learned-input-mediation v1',
    'O: B1 on 15 axis queries; O+: B1 on 15 axis then 12 coverage queries; '
    'IB: B1, B6x(v) ascending allowed A, B7x(v) ascending allowed B; I: B1, B6, B7 (v1 order)',
    'B6x(v): deduplicated set of b in terminals under do(A=v); B7x(v): set of a under do(B=v)',
    'coverage: per split, 3 mixed-evidence queries per (e1,e2) by smallest sha256 of '
    '"local-learning-supervision-v1:coverage:{split}:{e1},{e2},{da},{db}" (hex string), tie by query ID',
    'O+ order: frozen axis order, then coverage by smallest sha256 of '
    '"local-learning-supervision-v1:order:{split}:{order}:{query_id}", tie by query ID',
    'held-out partitions: within (30), mixed24 (mixed minus coverage), mixed36 (O, I, IB only)',
    'separate learner simulation services for O, I, IB and O+, each shared across teachers, splits and orders',
    'evaluations: one per (candidate, supplied key) prediction; candidate_losses: 27 per local experience',
    'timelines: block learner, own map and B1 only, before own update and after every later update',
]
COST_DEFINITIONS = dict(
    experiences='feedback maps processed', entries='typed feedback entries',
    supervised_slots={'per_key_type': SLOTS, 'note': 'nominal alphabet sizes; encoding-space counts, not bits'},
    positive_tuples='sum of deduplicated feedback-set sizes',
    clamp_runs='one per B6, B7, B6x or B7x entry (an intervention with complete-set feedback)',
    canonical_requests='one per B1 entry',
)


# ---------------------------------------------------------------- manifest

def coverage_plan():
    splits = json.loads((ROOT / SPLITS).read_text())
    plan = {}
    for s in splits['splits']:
        cov = coverage_subset(s['seed'], s['mixed_evidence'], QUERIES)
        plan[str(s['seed'])] = dict(
            coverage=cov, mixed24=[q for q in s['mixed_evidence'] if q not in cov],
            oplus_orders=[oplus_order(s['seed'], o, order, cov) for o, order in enumerate(s['orders'])])
    return plan


def manifest():
    v1_files = [f'reports/{V1}-{k}.json' for k in ('manifest', 'battery-summary', 'ceiling', 'learning', 'runtime')]
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                v1_battery_sha256=json.loads((ROOT / f'reports/{V1}-battery-summary.json').read_text())['battery_sha256'],
                v1_learning_report_sha256=sha(f'reports/{V1}-learning.json'),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                negative_controls=list(NEGATIVE), gate_templates=GATE_TEMPLATE,
                conditions=list(CONDITIONS), checkpoints={c: list(k) for c, k in CHECKPOINTS.items()},
                observation_key_schemas={
                    'O': ['B1'], 'O+': ['B1'], 'I': ['B1', 'B6(v) for v in allowed A', 'B7(v) for v in allowed B'],
                    'IB': ['B1', 'B6x(v) for v in ascending allowed A', 'B7x(v) for v in ascending allowed B']},
                coverage=coverage_plan(), cost_definitions=COST_DEFINITIONS, rules=RULES,
                prior_artifacts={p: sha(p) for p in [
                    f'docs/{V1}-contract.md', *v1_files, SPLITS, 'src/mechanism_learning.py',
                    'src/shared_input_mediation.py', 'experiments/mechanism_learning.py',
                    'tests/test_mechanism_learning.py']})
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    frozen = ROOT / 'reports' / f'{PREFIX}-manifest.json'
    ref = path if path.exists() else frozen if frozen.exists() else None
    if ref is not None and json.loads(ref.read_text()) != m:
        sys.exit('manifest differs from the frozen file; refusing to run')
    if not path.exists():
        dump(path, m)
    return m


# ---------------------------------------------------------------- projected evaluator views

_PMASK = {}


def pmask(m):
    """27-bit (i1, i2, s) mask to the 3-bit mask of s values present."""
    if m not in _PMASK:
        r = 0
        for code in range(27):
            if m >> code & 1:
                r |= 1 << (code % 3)
        _PMASK[m] = r
    return _PMASK[m]


class Projected:
    """Evaluator-only projected values of every B6/B7 atom, and their value partitions."""
    def __init__(self, ix):
        self.ix = ix
        self.table, self.part = {}, {}
        for a, k in enumerate(ix.keys):
            if k[0] in ('B6', 'B7'):
                col = [pmask(ix.table[h][a]) for h in range(N_VECTORS)]
                self.table[a] = col
                d = defaultdict(int)
                for h, v in enumerate(col):
                    d[v] |= 1 << h
                self.part[a] = dict(d)

    def value(self, a, h):
        return self.table[a][h]


def atom_of(ix, qi, key):
    if key[0] == 'B1':
        return ix.key_index[('B1', qi)]
    return ix.key_index[(key[0].rstrip('x'), qi, key[1])]


def value(ix, pj, a, key, h):
    return pj.value(a, h) if key[0] in ('B6x', 'B7x') else ix.table[h][a]


def partition(ix, pj, a, key):
    return pj.part[a] if key[0] in ('B6x', 'B7x') else ix.part[a]


def map_ok(ix, pj, h, th, qi, cond):
    x = QUERIES[qi]
    return all(value(ix, pj, a, k, h) == value(ix, pj, a, k, th)
               for k in observation_keys(x, cond) for a in [atom_of(ix, qi, k)])


def b1_ok(ix, h, th, qi):
    a = ix.key_index[('B1', qi)]
    return ix.table[h][a] == ix.table[th][a]


def feedback(ix, th, qi, cond):
    """Teacher feedback map for one training query; projection happens before anything leaves here."""
    x = QUERIES[qi]
    out = {}
    for key in observation_keys(x, cond):
        m = ix.table[th][atom_of(ix, qi, key)]
        if key[0] == 'B1':
            out[key] = frozenset((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if m >> (3 * (a + 1) + (b + 1)) & 1)
        else:
            trip = frozenset((i, j, s) for i in (-1, 0, 1) for j in (-1, 0, 1) for s in (-1, 0, 1)
                             if m >> (9 * (i + 1) + 3 * (j + 1) + (s + 1)) & 1)
            out[key] = project(trip) if key[0] in ('B6x', 'B7x') else trip
    return out


def ib_boundary_ok(fb):
    for key, f in fb.items():
        if key[0] not in ('B1', 'B6x', 'B7x'):
            return False
        if key[0] != 'B1' and not (isinstance(f, frozenset) and all(type(v) is int and v in (-1, 0, 1) for v in f)):
            return False
    return True


def projection_oracle_gate(bat):
    """B6x/B7x: independent oracle terminals vs projected battery masks vs the IB simulation service."""
    from tests.test_shared_input_mediation import oracle
    sim = make_simulator('IB')
    sample = sorted(set(bat['teacher_idx'].values()) | {44 * j for j in range(50)})
    ix_key = {k: a for a, k in enumerate(bat['keys'])}
    bad, checks = [], 0
    for h in sample:
        w = vector(h)
        for qi, x in enumerate(QUERIES):
            for c, other, item in ((A, B, 'B6'), (B, A, 'B7')):
                for v in x.allowed(c):
                    orc = frozenset(t[other] for t in oracle(x, w, ((c, v),))[1])
                    m = bat['table'][h][ix_key[(item, qi, v)]]
                    via_battery = frozenset(s - 1 for s in range(3) if pmask(m) >> s & 1)
                    via_service = sim.predict(w, x, (item + 'x', v))
                    checks += 1
                    if not orc == via_battery == via_service:
                        bad.append([h, qi, item, v])
    return dict(hypotheses=len(sample), checks=checks, mismatches=bad[:20], mismatch_count=len(bad), passed=not bad)


# ---------------------------------------------------------------- ceilings

def class_report(ix, th, cls):
    """v1 section 5 report for one complete-information class (the body of v1 `ceilings`)."""
    mech_atoms = ix.atoms(MECHANISM_ITEMS, ALL_Q)
    fwd = {t: 0 for t in FORWARD}
    for a in ix.atoms(['B8E'], ALL_Q):
        if ix.keys[a][1] in FORWARD:
            fwd[ix.keys[a][1]] |= ix.part[a].get(1, 0)
    mem = members(cls)
    groups = defaultdict(list)
    for h in mem:
        groups[tuple(ix.table[h][a] for a in mech_atoms)].append(h)
    tkey = tuple(ix.table[th][a] for a in mech_atoms)
    varying = Counter()
    for a in mech_atoms:
        if len({ix.table[h][a] for h in mem}) > 1:
            varying[ix.keys[a][0]] += 1
    b8_wrong = {}
    for tmpl in TEMPLATES:
        ats = [a for a in mech_atoms if ix.keys[a][0] in ('B8E', 'B8W') and ix.keys[a][1] == tmpl]
        b8_wrong[tmpl] = sum(any(ix.table[h][a] != ix.table[th][a] for a in ats) for h in mem)
    pairs = []
    for g, hs in sorted(groups.items(), key=lambda kv: min(kv[1])):
        if g == tkey:
            continue
        h = min(hs)
        diff = [a for a in mech_atoms if ix.table[h][a] != ix.table[th][a]]
        pairs.append(dict(teacher=list(vector(th)), member=list(vector(h)), member_index=h, subclass_size=len(hs),
                          differing_atoms=len(diff), differing_by_item=dict(Counter(ix.keys[a][0] for a in diff)),
                          examples=[dict(atom=list(ix.keys[a]), teacher=decode(ix.keys[a], ix.table[th][a]),
                                         member=decode(ix.keys[a], ix.table[h][a])) for a in diff[:8]]))
    false_fwd = {t: popcount(cls & b) for t, b in fwd.items()}
    false_fwd['union'] = popcount(cls & (fwd['SF'] | fwd['CF']))
    mir = hypothesis_index(mirror(vector(th)))
    return dict(
        class_size=len(mem), mechanism_subclasses=len(groups),
        teacher_subclass_size=len(groups[tkey]), mechanism_homogeneous=len(groups) == 1,
        separating_atoms_by_item=dict(varying), members_with_B8_differences_by_template=b8_wrong,
        members_with_forward_witness=false_fwd,
        core_sign_mirror=dict(index=mir, in_class=bool(cls >> mir & 1),
                              mechanism_equivalent=mir in groups[tkey], is_teacher=mir == th),
        indistinguishable_different_mechanism=pairs[:10], further_pairs_not_listed=max(0, len(pairs) - 10))


def signature_class(ix, pj, th, cond):
    cls = ix.equal_on(ix.atoms(['B1'], ALL_Q), th)
    if cond == 'I':
        cls &= ix.equal_on(ix.atoms(['B6', 'B7'], ALL_Q), th)
    elif cond == 'IB':
        for a in ix.atoms(['B6', 'B7'], ALL_Q):
            cls &= pj.part[a][pj.value(a, th)]
    return cls


def all_ceilings(ix, pj, bat):
    v1_frozen = json.loads((ROOT / f'reports/{V1}-ceiling.json').read_text())['ceilings']
    v1_now = ceilings(ix, bat)
    out, repro = {}, {}
    for n, th in bat['teacher_idx'].items():
        out[n] = {c: class_report(ix, th, signature_class(ix, pj, th, c)) for c in ('O', 'I', 'IB')}
        out[n]['O+'] = dict(same_as='O', note='O+ supplies B1 only; its complete-information class is that of O')
        repro[n] = {arm: json.loads(json.dumps(out[n][arm])) == v1_frozen[n][arm] == json.loads(json.dumps(v1_now[n][arm]))
                    for arm in V1_ARMS}
    return out, dict(per_teacher=repro, passed=all(all(v.values()) for v in repro.values()))


# ---------------------------------------------------------------- learning

def ib_memory(ix, pj, th, mem, Q):
    c = Counter()
    for qi in Q:
        x = QUERIES[qi]
        for key in observation_keys(x, 'IB')[1:]:
            a = atom_of(ix, qi, key)
            pred = mem.predict(x, key)
            c[f'{key[0]}_n'] += 1
            if pred is None:
                c[f'{key[0]}_unknown'] += 1
            else:
                c[f'{key[0]}_correct'] += sum(1 << (v + 1) for v in pred) == pj.value(a, th)
    return dict(c)


def run_stream(cond, n, th, split, o, order, sim, ix, pj, ev, cov, checks):
    """One learner stream. Returns rows, per-step states, run summary, timelines, costs."""
    K = CHECKPOINTS[cond]
    L = dict(full=VersionSpace(sim), positive=VersionSpace(sim, positive_only=True),
             local=BlockLearner(sim), memory=Memory())
    parts = dict(within=split['within_context'], mixed24=cov['mixed24'])
    if cond != 'O+':
        parts['mixed36'] = split['mixed_evidence']
    arm = 'I' if cond == 'I' else 'O'
    states = [dict(full=L['full'].bits, positive=L['positive'].bits, failed=(False, False),
                   w=L['local'].w, memory={})]
    own, b1 = defaultdict(list), defaultdict(list)       # item step -> post-update correctness list
    before = {}
    ret = Counter()
    cost, cost_rows, rows = Counter(), [], []
    history, done = [], 0
    sim_start = sim.requests
    for k in K:
        while done < k:
            qi = order[done]
            x = QUERIES[qi]
            fb = feedback(ix, th, qi, cond)
            if cond == 'IB':
                checks['ib_boundary_violations'] += not ib_boundary_ok(fb)
            h_before = hypothesis_index(L['local'].w)
            before[done] = (map_ok(ix, pj, h_before, th, qi, cond), b1_ok(ix, h_before, th, qi))
            prev_ok = [map_ok(ix, pj, h_before, th, s, cond) for s in history]
            for learner in L.values():
                status = learner.update(x, fb)
                checks['inconsistent_feedback'] += status == 'inconsistent_feedback'
            cost.update(feedback_cost(fb))
            history.append(qi)
            done += 1
            h_local = hypothesis_index(L['local'].w)
            for t, s in enumerate(history):
                own[t].append(map_ok(ix, pj, h_local, th, s, cond))
                b1[t].append(b1_ok(ix, h_local, th, s))
            if done >= 2:                                   # v1 pooled retention metric, unchanged
                for s, was in zip(history[:-1], prev_ok):
                    now = map_ok(ix, pj, h_local, th, s, cond)
                    ret['complete_map_checks'] += 1
                    ret['complete_map_retained'] += now
                    ret['b1_checks'] += 1
                    ret['b1_retained'] += b1_ok(ix, h_local, th, s)
                    ret['lost_previously_correct'] += was and not now
            for vsn in ('full', 'positive'):
                V = L[vsn].bits
                checks[f'{vsn}_teacher_eliminated'] += not V >> th & 1
                for s in history:
                    for key in observation_keys(QUERIES[s], cond):
                        a = atom_of(ix, s, key)
                        T = value(ix, pj, a, key, th)
                        part = partition(ix, pj, a, key)
                        if vsn == 'full':
                            bad = V & ~part[T]
                        else:
                            sup = 0
                            for val, bits in part.items():
                                if val & T == T:
                                    sup |= bits
                            bad = V & ~sup
                        checks[f'{vsn}_retention_violations'] += bad != 0
            states.append(dict(full=L['full'].bits, positive=L['positive'].bits,
                               failed=(L['full'].failed, L['positive'].failed), w=L['local'].w,
                               memory=dict(L['memory'].table)))
        seen = order[:k]
        h_local = hypothesis_index(L['local'].w)
        pl = [('seen', seen)]
        if cond == 'O+':
            pl += [('seen_axis', order[:min(k, 15)]), ('seen_coverage', order[15:k])]
        pl += [(p, Q) for p, Q in parts.items()]
        for part, Q in pl:
            base = [n, cond, split['seed'], o, k, part]
            for vsn in ('full', 'positive'):
                r = ev.vs(n, L[vsn].bits, Q)
                checks['containment_violations'] += r['containment_violations']
                rows.append(base + [vsn, r])
            rows.append(base + ['local', ev.single(n, h_local, Q)])
            rows.append(base + ['zero', ev.single(n, hypothesis_index((0,) * 7), Q)])
            rows.append(base + ['teacher', ev.single(n, th, Q)])
            mrow = ev.memory(n, L['memory'], arm, Q)
            if cond == 'IB':
                mrow = dict(mrow, **ib_memory(ix, pj, th, L['memory'], Q))
            rows.append(base + ['memory', mrow])
        cost_rows.append(dict(checkpoint=k, **dict(sorted(cost.items())),
                              local_evaluations=L['local'].evaluations, candidate_losses=27 * done,
                              local_commits=L['local'].commits,
                              local_coefficients_changed=L['local'].coefficients_changed,
                              full_evaluations=L['full'].evaluations, positive_evaluations=L['positive'].evaluations,
                              simulation_requests=sim.requests - sim_start, local_index=h_local,
                              local_w=list(L['local'].w)))
    run = dict(teacher=n, arm=cond, split=split['seed'], order=o,
               final_survivors={v: popcount(L[v].bits) for v in ('full', 'positive')},
               local_w=list(L['local'].w), local_commits=L['local'].commits,
               memory_entries=len(L['memory'].table),
               full_V_hash=hashlib.sha256(str(L['full'].bits).encode()).hexdigest()[:16])
    tl = [dict(step=t + 1, query=order[t],
               own=dict(before=before[t][0], after=''.join('1' if v else '0' for v in own[t])),
               b1=dict(before=before[t][1], after=''.join('1' if v else '0' for v in b1[t])))
          for t in range(done)]
    work = dict(full=L['full'].evaluations, positive=L['positive'].evaluations, local=L['local'].evaluations,
                commits=L['local'].commits, coefficients_changed=L['local'].coefficients_changed)
    return dict(rows=rows, states=states, run=run, timelines=tl, cost=cost_rows, retention=dict(ret),
                work=work, final=dict(full=L['full'].bits, positive=L['positive'].bits),
                local=dict(w=L['local'].w, index=hypothesis_index(L['local'].w)))


def learning(ix, pj, bat, splits, plan, ev):
    sims = {c: make_simulator(c) for c in CONDITIONS}
    rows, runs, streams, checks = [], [], [], Counter()
    final_V = defaultdict(set)
    work = defaultdict(Counter)
    for n in TEACHERS:
        th = bat['teacher_idx'][n]
        for split in splits:
            cov = plan[str(split['seed'])]
            for o, order in enumerate(split['orders']):
                res = {}
                for cond in ('O', 'I', 'IB', 'O+'):
                    stream = cov['oplus_orders'][o] if cond == 'O+' else order
                    r = run_stream(cond, n, th, split, o, stream, sims[cond], ix, pj, ev, cov, checks)
                    res[cond] = r
                    rows += r['rows']
                    runs.append(r['run'])
                    for v in ('full', 'positive'):
                        final_V[(n, cond, split['seed'], v)].add(r['final'][v])
                    work[f'{cond}_full']['evaluations'] += r['work']['full']
                    work[f'{cond}_positive']['evaluations'] += r['work']['positive']
                    work[f'{cond}_local']['evaluations'] += r['work']['local']
                    work[f'{cond}_local']['commits'] += r['work']['commits']
                    work[f'{cond}_local']['coefficients_changed'] += r['work']['coefficients_changed']
                    work[f'{cond}_local']['candidate_losses'] += 27 * len(stream)
                    streams.append(dict(teacher=n, condition=cond, split=split['seed'], order=o, queries=stream,
                                        final_w=list(r['local']['w']), final_index=r['local']['index'],
                                        cost=r['cost'], timelines=r['timelines'], v1_retention=r['retention']))
                # per-prefix subset chain, O+ prefix identity and O+ monotonicity
                for t in range(16):
                    for v in ('full', 'positive'):
                        vi, vib, vo = (res[c]['states'][t][v] for c in ('I', 'IB', 'O'))
                        checks['subset_checks'] += 1
                        checks['V_I_not_subset_V_IB'] += bool(vi & ~vib)
                        checks['V_IB_not_subset_V_O'] += bool(vib & ~vo)
                    checks['oplus_prefix_checks'] += 1
                    checks['oplus_prefix_mismatch'] += res['O+']['states'][t] != res['O']['states'][t]
                ost = res['O+']['states']
                for t in range(16, len(ost)):
                    for v in ('full', 'positive'):
                        checks['oplus_monotone_checks'] += 1
                        checks['oplus_not_monotone'] += bool(ost[t][v] & ~ost[t - 1][v])
    for key, vs in final_V.items():
        checks['order_dependent_final_V'] += len(vs) != 1
    for cond, sim in sims.items():
        work[f'simulator_{cond}'] = Counter(requests=sim.requests, unique=len(sim.memo))
    return rows, runs, streams, dict(checks), {k: dict(v) for k, v in work.items()}


# ---------------------------------------------------------------- reproduction of v1

def reproduce_v1(rows, runs, streams, work):
    v1 = json.loads((ROOT / f'reports/{V1}-learning.json').read_text())
    ref = {tuple(r[:7]): r[7] for r in v1['rows']}
    mine = {}
    for r in rows:
        if r[1] in V1_ARMS and r[5] in ('seen', 'within', 'mixed36'):
            key = tuple(r[:5]) + ('mixed' if r[5] == 'mixed36' else r[5], r[6])
            mine[key] = r[7]
    row_mismatch = [list(k) for k in sorted(set(ref) | set(mine), key=str) if ref.get(k) != mine.get(k)]
    v1_runs = {(r['teacher'], r['arm'], r['split'], r['order']): r for r in v1['runs']}
    my_runs = {(r['teacher'], r['arm'], r['split'], r['order']): r for r in runs if r['arm'] in V1_ARMS}
    run_mismatch = [list(k) for k in sorted(set(v1_runs) | set(my_runs)) if v1_runs.get(k) != my_runs.get(k)]
    work_mismatch = []
    for arm in V1_ARMS:
        for part in ('full', 'positive'):
            if v1['work'][f'{arm}_{part}'] != {'evaluations': work[f'{arm}_{part}']['evaluations']}:
                work_mismatch.append(f'{arm}_{part}')
        mine_local = {k: work[f'{arm}_local'][k] for k in ('evaluations', 'commits', 'coefficients_changed')}
        if v1['work'][f'{arm}_local'] != mine_local:
            work_mismatch.append(f'{arm}_local')
        if v1['work'][f'simulator_{arm}'] != work[f'simulator_{arm}']:
            work_mismatch.append(f'simulator_{arm}')
    ret_mismatch = []
    for n in TEACHERS:
        for arm in V1_ARMS:
            pooled_ret = Counter()
            for s in streams:
                if s['teacher'] == n and s['condition'] == arm:
                    pooled_ret.update(s['v1_retention'])
            ref_ret = {k: v for k, v in v1['targets'][n][f'local_{arm}']['retention'].items() if not k.endswith('_rate')}
            if dict(pooled_ret) != ref_ret:
                ret_mismatch.append([n, arm])
    return dict(rows_compared=len(ref), row_mismatches=len(row_mismatch), row_mismatch_examples=row_mismatch[:10],
                run_mismatches=run_mismatch[:10], work_mismatches=work_mismatch,
                v1_retention_mismatches=ret_mismatch,
                passed=not row_mismatch and not run_mismatch and not work_mismatch and not ret_mismatch)


# ---------------------------------------------------------------- targets and summaries

def behaviour(ix, n, th, h, w, gate_cache):
    positive_atoms = [a for a, k in enumerate(ix.keys) if k[0] == 'B8W' and ix.table[th][a] == 1]
    g = local_gate(tuple(w), gate_cache)
    wit = all(ix.table[h][a] == 1 for a in positive_atoms)
    false_w = sum(ix.table[h][a] == 1 and ix.table[th][a] == 0 for a, k in enumerate(ix.keys) if k[0] == 'B8W')
    passes = all(g[tm] for tm in GATE_TEMPLATE[n])
    return dict(reproduces_positive_witnesses=wit, false_witness_atoms=false_w, gate=g,
                gate_passed_teacher_template=passes, recovered=wit and passes)


def timeline_summaries(stream, view, upto=None):
    out = []
    for item in stream['timelines']:
        after = [c == '1' for c in item[view]['after']]
        if upto is not None:
            after = after[:upto - item['step'] + 1]
            if not after:
                continue
        out.append(timeline(item[view]['before'], after))
    return out


def min_prefix_accuracy(stream, view):
    T = len(stream['timelines'])
    accs = []
    for t in range(1, T + 1):
        items = [it for it in stream['timelines'] if it['step'] <= t]
        accs.append(sum(it[view]['after'][t - it['step']] == '1' for it in items) / len(items))
    return round(min(accs), 6)


def oplus_axis(stream, view):
    axis = [it for it in stream['timelines'] if it['step'] <= 15]
    at15 = [it[view]['after'][15 - it['step']] == '1' for it in axis]
    during = [[c == '1' for c in it[view]['after'][16 - it['step']:]] for it in axis]
    at27 = [it[view]['after'][-1] == '1' for it in axis]
    return axis_phase(at15, during, at27)


class RowIndex:
    """Rows grouped by (teacher, condition, checkpoint, partition, learner) for pooling."""
    def __init__(self, rows):
        self.g = defaultdict(list)
        for r in rows:
            self.g[(r[0], r[1], r[4], r[5], r[6])].append(r)

    def __call__(self, n, cond, learner, k, part, splits=None, orders=None):
        out = Counter()
        for r in self.g[(n, cond, k, part, learner)]:
            if (orders is None or r[3] in orders) and (splits is None or r[2] in splits):
                out.update(r[7])
        return dict(out)


def targets(rows, ix, bat, ceil, streams, splits, plan, gate_cache):
    split_ids = [s['seed'] for s in splits]
    pooled = RowIndex(rows)
    out = {}
    for n in TEACHERS:
        th = bat['teacher_idx'][n]
        mir = hypothesis_index(mirror(vector(th)))
        t = {}
        for cond in CONDITIONS:
            kf = FINAL[cond]
            S = [s for s in streams if s['teacher'] == n and s['condition'] == cond]
            c = {}
            # L2: outputs on common partitions
            parts = {}
            for part in ('within', 'mixed24', 'mixed36'):
                if cond == 'O+' and part == 'mixed36':
                    continue
                lc = pooled(n, cond, 'local', kf, part)
                z = pooled(n, cond, 'zero', kf, part)
                parts[part] = dict(exact=lc.get('exact', 0), n=lc.get('n', 0), accuracy=ratio(lc.get('exact', 0), lc.get('n', 0)),
                                   zero_exact=z.get('exact', 0), zero_accuracy=ratio(z.get('exact', 0), z.get('n', 0)),
                                   worst_split=min((ratio(pooled(n, cond, 'local', kf, part, splits=[sid]).get('exact', 0),
                                                          pooled(n, cond, 'local', kf, part, splits=[sid]).get('n', 0)) or 0)
                                                   for sid in split_ids))
            l2 = all(parts[p]['n'] > 0 and parts[p]['exact'] >= 0.8 * parts[p]['n'] and parts[p]['exact'] > parts[p]['zero_exact']
                     for p in ('within', 'mixed24'))
            c['L2_outputs'] = dict(partitions=parts, passed=l2)
            # L1: behavioural recovery at the final checkpoint
            beh = []
            for s in S:
                b = behaviour(ix, n, th, s['final_index'], s['final_w'], gate_cache)
                beh.append(dict(split=s['split'], order=s['order'], w=s['final_w'], **b,
                                converged_to=('teacher' if s['final_index'] == th else
                                              'mirror' if s['final_index'] == mir else 'other')))
            rec = sum(b['recovered'] for b in beh)
            per_split = {sid: sum(b['recovered'] for b in beh if b['split'] == sid) for sid in split_ids}
            c['L1_recovery'] = dict(recovered=rec, runs=len(beh), per_split=per_split,
                                    worst_split=min(per_split.values()), v1_level_13_of_25=rec >= 13,
                                    convergence=dict(Counter(b['converged_to'] for b in beh)),
                                    distinct_final_vectors=len({tuple(b['w']) for b in beh}),
                                    runs_detail=beh, passed=rec >= 23 and min(per_split.values()) >= 4)
            # L3: end-of-stream preservation on the own map; B1-only reported alongside
            views = {}
            for view in ('own', 'b1'):
                summ = [x for s in S for x in timeline_summaries(s, view)]
                pool = pool_timelines(summ)
                pool['per_split_final_retention'] = {
                    sid: pool_timelines([x for s in S if s['split'] == sid for x in timeline_summaries(s, view)])['final_retention']
                    for sid in split_ids}
                mins = [min_prefix_accuracy(s, view) for s in S]
                pool['min_prefix_seen_accuracy'] = dict(min=min(mins), median=statistics.median(mins), per_stream=mins)
                views[view] = pool
            v1r = Counter()
            for s in S:
                v1r.update(s['v1_retention'])
            own = views['own']
            l3 = (own['presented'] > 0 and own['ever_acquired'] > 0
                  and own['final_correct'] >= 0.95 * own['presented'] and own['forgotten'] <= 0.05 * own['ever_acquired'])
            c['L3_preservation'] = dict(own_map=own, b1_only=views['b1'], passed=l3,
                                        v1_pooled_retention=dict(v1r, complete_map_rate=ratio(v1r['complete_map_retained'], v1r['complete_map_checks']),
                                                                 b1_rate=ratio(v1r['b1_retained'], v1r['b1_checks']),
                                                                 note='v1 throughout-training metric; not a forgetting rate'))
            if cond == 'O+':
                c['oplus_coverage_phase'] = {}
                for view in ('own', 'b1'):
                    ax = Counter()
                    for s in S:
                        ax.update(oplus_axis(s, view))
                    c['oplus_coverage_phase'][view] = dict(ax)
                c['oplus_at_15'] = dict(
                    L1_recovered=sum(behaviour(ix, n, th, s['cost'][3]['local_index'], s['cost'][3]['local_w'], gate_cache)['recovered']
                                     for s in S),
                    mixed24=pooled(n, cond, 'local', 15, 'mixed24').get('exact', 0),
                    within=pooled(n, cond, 'local', 15, 'within').get('exact', 0),
                    own_map=pool_timelines([x for s in S for x in timeline_summaries(s, 'own', upto=15)]))
            # recovery against cost, per checkpoint
            curve = []
            for i, k in enumerate(CHECKPOINTS[cond]):
                cs = [s['cost'][i] for s in S]
                curve.append(dict(checkpoint=k, recovered=sum(
                    behaviour(ix, n, th, x['local_index'], x['local_w'], gate_cache)['recovered'] for x in cs),
                    **{f'mean_{m}': round(sum(x.get(m, 0) for x in cs) / len(cs), 6)
                       for m in ('experiences', 'entries', 'supervised_slots', 'positive_tuples', 'clamp_runs',
                                 'canonical_requests', 'local_evaluations', 'candidate_losses')}))
            c['recovery_by_cost'] = curve
            c['suffices'] = c['L1_recovery']['passed'] and c['L2_outputs']['passed'] and l3
            # version spaces: output and mechanism identification, negative controls
            per = []
            for sid in split_ids:
                f = pooled(n, cond, 'full', kf, 'within', splits=[sid], orders=[0])
                p = pooled(n, cond, 'positive', kf, 'within', splits=[sid], orders=[0])
                per.append(dict(split=sid, survivors=f['survivors'], mech_equivalent=f['mech_equivalent_all_81'],
                                all_equivalent=f['mech_equivalent_all_81'] == f['survivors'],
                                false_forward=dict(SF=f['forward_attribution_SF'], CF=f['forward_attribution_CF'],
                                                   union=f['forward_attribution_union']),
                                mechanism_uncertain_within=f['mechanism_uncertain_survivors'],
                                positive_only_survivors=p['survivors'],
                                positive_only_false_forward_union=p['forward_attribution_union']))
            outs = {}
            for part in ('within', 'mixed24'):
                f = pooled(n, cond, 'full', kf, part, orders=[0])
                outs[part] = dict(certified=f['certified'], n=f['n'], coverage=ratio(f['certified'], f['n']),
                                  certified_incorrect=f['certified_incorrect'])
            c['version_space'] = dict(per_split=per, outputs=outs)
            lf = sum(bool(ev_fwd(ix, s['final_index'])) for s in S)
            c['local_false_forward_runs'] = dict(runs=lf, of=len(S))
            t[cond] = c
        # V1 / V2 targets
        sig = ceil[n]
        if n == 'S2':
            v = {}
            for cond in ('O+', 'IB'):
                zero = all(p['false_forward']['union'] == 0 for p in t[cond]['version_space']['per_split'])
                cls = sig['O' if cond == 'O+' else 'IB']
                obstructed = cls['members_with_forward_witness']['union'] > 0
                v[cond] = dict(zero_false_forward_every_split=zero,
                               status='pass' if zero else 'unattainable_by_identifiability' if obstructed else 'fail')
            t['V1_S2_coverage'] = v
        if n in POSITIVE:
            t['V2_no_regression'] = {cond: dict(all_splits_equivalent=all(p['all_equivalent'] for p in t[cond]['version_space']['per_split']))
                                     for cond in ('O+', 'IB')}
            for cond in ('O+', 'IB'):
                t['V2_no_regression'][cond]['passed'] = t['V2_no_regression'][cond]['all_splits_equivalent']
        else:
            for cond in CONDITIONS:
                t[cond]['target_applies'] = False
                t[cond]['note'] = 'L1-L3 are declared for S1 and C2 only; reported, not scored'
        out[n] = t
    out['primary_answer'] = dict(
        conditions_sufficing_for_S1_and_C2=[c for c in CONDITIONS if out['S1'][c]['suffices'] and out['C2'][c]['suffices']],
        per_teacher={n: [c for c in CONDITIONS if out[n][c]['suffices']] for n in POSITIVE})
    return out


# ---------------------------------------------------------------- main

def run_tests():
    names = ['tests.test_local_supervision', 'tests.test_mechanism_learning']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(modules=names, run=res.testsRun, failures=len(res.failures), errors=len(res.errors),
                passed=res.wasSuccessful())


ROW_HEAD = ['teacher', 'condition', 'split', 'order', 'checkpoint', 'partition', 'learner', 'counts']


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    splits = json.loads((ROOT / SPLITS).read_text())
    if [[q.e1, q.e2, q.da, q.db] for q in QUERIES] != splits['queries']:
        sys.exit('query order differs from the frozen split file')
    timing = {}
    t0 = time.perf_counter()
    bat = build_battery(args.workers)
    timing['battery'] = time.perf_counter() - t0
    v1s = json.loads((ROOT / f'reports/{V1}-battery-summary.json').read_text())
    hashes = [atom_hash(v) for v in bat['table']]
    battery_sha = hashlib.sha256(''.join(hashes).encode()).hexdigest()
    battery_gate = dict(battery_sha256=battery_sha == v1s['battery_sha256'],
                        atom_schema_sha256=bat['schema_hash'] == v1s['atom_schema_sha256'],
                        hypothesis_hashes=hashes == v1s['hypothesis_hashes'])
    battery_gate['passed'] = all(battery_gate.values())
    if not battery_gate['passed']:
        sys.exit(f'battery differs from v1: {battery_gate}')
    t0 = time.perf_counter()
    og = oracle_gate(bat)
    rep = reproduce_frozen(bat)
    pg = projection_oracle_gate(bat)
    timing['oracle_projection_and_reproduction'] = time.perf_counter() - t0
    if not (og['passed'] and rep['passed'] and pg['passed']):
        sys.exit('oracle, projection or frozen-reproduction gate failed before ceilings')
    t0 = time.perf_counter()
    ix = Index(bat)
    pj = Projected(ix)
    ceil, ceil_repro = all_ceilings(ix, pj, bat)
    timing['ceilings'] = time.perf_counter() - t0
    dump(out / f'{PREFIX}-ceiling.json', dict(experiment=PREFIX, manifest_sha256=m['sha256'],
                                              atom_schema_sha256=bat['schema_hash'], ceilings=ceil,
                                              v1_reproduction=ceil_repro))
    if not ceil_repro['passed']:
        sys.exit('v1 ceilings not reproduced; stopping (see ceiling report)')
    plan = coverage_plan()
    plan_ok = plan == m['coverage'] and all(
        not set(p['coverage']) & set(p['mixed24']) and len(p['coverage']) == 12 and len(p['mixed24']) == 24
        for p in plan.values())
    t0 = time.perf_counter()
    ev = Evaluator(ix, bat)
    rows, runs, streams, checks, work = learning(ix, pj, bat, splits['splits'], plan, ev)
    timing['learning_and_evaluation'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    repro = reproduce_v1(rows, runs, streams, work)
    gate_cache = {}
    tg = targets(rows, ix, bat, ceil, streams, splits['splits'], plan, gate_cache)
    timing['reproduction_targets_and_local_gates'] = time.perf_counter() - t0
    gate = dict(battery=battery_gate['passed'], oracle=og['passed'], frozen_reproduction=rep['passed'],
                projection_oracle=pg['passed'], v1_ceilings=ceil_repro['passed'], v1_learning=repro['passed'],
                coverage_plan=plan_ok,
                teacher_eliminated=checks.get('full_teacher_eliminated', 0) + checks.get('positive_teacher_eliminated', 0),
                full_retention_violations=checks.get('full_retention_violations', 0),
                positive_retention_violations=checks.get('positive_retention_violations', 0),
                containment_violations=checks.get('containment_violations', 0),
                order_dependent_final_V=checks.get('order_dependent_final_V', 0),
                V_I_not_subset_V_IB=checks.get('V_I_not_subset_V_IB', 0),
                V_IB_not_subset_V_O=checks.get('V_IB_not_subset_V_O', 0),
                oplus_prefix_mismatch=checks.get('oplus_prefix_mismatch', 0),
                oplus_not_monotone=checks.get('oplus_not_monotone', 0),
                ib_boundary_violations=checks.get('ib_boundary_violations', 0),
                inconsistent_feedback=checks.get('inconsistent_feedback', 0))
    bools = ('battery', 'oracle', 'frozen_reproduction', 'projection_oracle', 'v1_ceilings', 'v1_learning',
             'coverage_plan')
    gate['passed'] = all(gate[k] for k in bools) and all(v == 0 for k, v in gate.items() if k not in bools)
    t0 = time.perf_counter()
    tests = run_tests()
    timing['tests'] = time.perf_counter() - t0
    head = dict(experiment=PREFIX, contract=CONTRACT, manifest_sha256=m['sha256'],
                completion=completion_status(gate, tests), correctness_gate=gate, checks=checks,
                oracle_verification=og, projection_oracle=pg, frozen_reproduction=rep, v1_reproduction=repro,
                tests=tests, targets=tg, work=work, runs=runs, streams=streams,
                logical_storage=dict(version_space_bits=N_VECTORS, local_model='7 ternary coefficients',
                                     memory='stored (query, key) entries; see runs'),
                provenance=dict(python=platform.python_version(), implementation=platform.python_implementation(),
                                source_sha256={p: sha(p) for p in (
                                    'src/local_supervision.py', 'experiments/local_supervision.py',
                                    'tests/test_local_supervision.py', 'src/mechanism_learning.py',
                                    'experiments/mechanism_learning.py', CONTRACT)}),
                row_columns=ROW_HEAD)
    write_rows_report(out / f'{PREFIX}-learning.json', head, rows)
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: round(v, 3) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3), workers=args.workers,
        peak_rss_kib=dict(main=ru, largest_child=rc,
                          method='getrusage ru_maxrss: main process peak, and the largest single terminated '
                                 'child; not the sum over concurrent workers'),
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(head['completion'], json.dumps(gate))


if __name__ == '__main__':
    main()
