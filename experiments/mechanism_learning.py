"""Learned input mediation v1: manifest, battery, ceilings, arm O and arm I learning.

Run `.venv/bin/python -m experiments.mechanism_learning [--out-dir DIR] [--workers N]`.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import unittest

from src.mechanism_learning import (
    BASE_KEYS, FORWARD, MECHANISM_ITEMS, N_VECTORS, QUERIES, TEMPLATES, BlockLearner, Memory, Simulator,
    VersionSpace, atom_hash, atom_vector, compute_hypothesis, decode, hypothesis_index, observation_keys,
    popcount, vector, witness_keys,
)
from src.shared_input_mediation import COEFFS


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'learned-input-mediation-v1'
CONTRACT = 'docs/learned-input-mediation-v1-contract.md'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
TEACHERS = {
    'S1': (3, 3, 3, 0, 3, -3, 0),
    'C2': (3, 0, 0, -3, 3, -3, -3),
    'S2': (3, 3, 3, 0, 3, 3, 0),
    'T1': (3, 0, 0, -3, 3, -3, 0),
}
POSITIVE, NEGATIVE = ('S1', 'C2'), ('S2', 'T1')
GATE_TEMPLATE = {'S1': ['SF'], 'S2': ['SF'], 'C2': ['CF'], 'T1': ['SF', 'CF']}
TEMPLATE_SPEC = {
    'SF': dict(source='A', target='B', donor='i1', recipient='i1', links=['w_1A', 'w_1B']),
    'SR': dict(source='B', target='A', donor='i1', recipient='i1', links=['w_1B', 'w_1A']),
    'CF': dict(source='A', target='B', donor='i1', recipient='i2', links=['w_1A', 'w_12', 'w_2B']),
    'CR': dict(source='B', target='A', donor='i2', recipient='i1', links=['w_2B', 'w_12', 'w_1A']),
}
CHECKPOINTS = (0, 5, 10, 15)
OBSERVATIONS = {'O': ['B1'], 'I': ['B1', 'B6', 'B7']}
RULES = [
    'battery items B1-B8 as contract section 4; atoms keyed (item, query, arguments); set values as bitmasks',
    'B1-B5 masks over (a,b); B6 over (i1,i2,b) under do(A=v); B7 over (i1,i2,a) under do(B=v)',
    'B8E: any tuple (v,v2,r,r2) qualifies under 8(a)-(d) for the template on the query; r,r2 range over {-1,0,1}',
    'B8W: fixed union of the four teachers qualifying tuples over all four templates, frozen before the full battery',
    'mechanism atoms are B2-B8; observational signature arm O = B1; arm I = B1, B6, B7 (all 81 queries)',
    'one feedback map per training query is one experience; full filter equality on every key; positive-only containment',
    'block learner: blocks (w_1A,w_2A),(w_1B,w_2B),(w_KA,w_KB),(w_12); 26 neighbours plus unchanged; strict improvement',
    'memory answers only observed (query, key); zero baseline is all-zero coefficients',
    'splits, orders and query IDs imported from the frozen v1 split manifest; no resampling',
]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1) + '\n')


def manifest():
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                negative_controls=list(NEGATIVE), gate_templates=GATE_TEMPLATE, chain_templates=TEMPLATE_SPEC,
                checkpoints=list(CHECKPOINTS), arms_observation_types=OBSERVATIONS, rules=RULES,
                prior_artifacts={p: sha(p) for p in (
                    'reports/shared-input-mediation-v1-gate.json', 'reports/shared-input-mediation-v1-manifest.json',
                    'reports/learned-reciprocal-participation-v1-phase-a.json', SPLITS,
                    'src/shared_input_mediation.py', 'src/reciprocal_participation.py',
                    'experiments/shared_input_mediation.py')})
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != m:
            sys.exit('manifest differs from the frozen file; refusing to run')
    else:
        dump(path, m)
    return m


# ---------------------------------------------------------------- battery

def _worker(h):
    return compute_hypothesis(h)


def build_battery(workers):
    """Teachers first (their tuples fix the B8W schema), then all 2,187 hypotheses."""
    teacher_idx = {n: hypothesis_index(w) for n, w in TEACHERS.items()}
    teacher_raw = {n: compute_hypothesis(h) for n, h in teacher_idx.items()}
    wkeys = witness_keys([teacher_raw[n][1] for n in TEACHERS])
    schema = [list(k) for k in BASE_KEYS] + [list(k) for k in wkeys]
    schema_hash = hashlib.sha256(canonical(schema).encode()).hexdigest()
    if workers > 1:
        with Pool(workers) as pool:
            raw = pool.map(_worker, range(N_VECTORS), chunksize=16)
    else:
        raw = [compute_hypothesis(h) for h in range(N_VECTORS)]
    table = [atom_vector(v, t, wkeys) for v, t, _ in raw]
    tuples = [t for _, t, _ in raw]
    stats = dict(inference_requests=sum(s['requests'] for _, _, s in raw),
                 unique_inferences=sum(s['unique'] for _, _, s in raw))
    stats['cache_hits'] = stats['inference_requests'] - stats['unique_inferences']
    for n, h in teacher_idx.items():
        if table[h] != atom_vector(teacher_raw[n][0], teacher_raw[n][1], wkeys):
            sys.exit(f'teacher battery for {n} differs between passes')
    keys = [tuple(k) for k in BASE_KEYS] + list(wkeys)
    return dict(keys=keys, wkeys=wkeys, schema=schema, schema_hash=schema_hash, table=table, tuples=tuples,
                teacher_idx=teacher_idx, stats=stats)


def atom_query(key):
    return key[2] if key[0] in ('B5', 'B8E', 'B8W') else key[1]


def oracle_gate(bat):
    """Independent-oracle verification of every atom for the teachers and the 50 declared indices."""
    from tests.test_mechanism_learning import oracle_atoms
    sample = sorted(set(bat['teacher_idx'].values()) | {44 * j for j in range(50)})
    mismatched, tuple_mismatch = [], []
    for h in sample:
        vec, tuples = oracle_atoms(vector(h), bat['wkeys'])
        if tuple(vec) != bat['table'][h]:
            mismatched.append(h)
        if {k: sorted(v) for k, v in tuples.items()} != {k: sorted(v) for k, v in bat['tuples'][h].items()}:
            tuple_mismatch.append(h)
    return dict(hypotheses=len(sample), indices=sample, atoms_per_hypothesis=len(bat['keys']),
                atom_mismatches=mismatched, qualifying_tuple_set_mismatches=tuple_mismatch,
                passed=not mismatched and not tuple_mismatch)


def reproduce_frozen(bat):
    """Shared-input results for S1, C2, S2 and v1 conditions 1-7 for T1."""
    from experiments.shared_input_mediation import Runner, condition_3_prime, conditions_1_to_7
    si = json.loads((ROOT / 'reports/shared-input-mediation-v1-gate.json').read_text())
    v1 = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-phase-a.json').read_text())
    native = {'S1': ('SF', 'SR'), 'S2': ('SF', 'SR'), 'C2': ('CF', 'CR')}
    out = {}
    for n, w in TEACHERS.items():
        cond, _, _ = conditions_1_to_7(w)
        if n == 'T1':
            out[n] = dict(conditions_1_to_7_match_v1=cond == v1['teachers']['T1']['conditions'])
            continue
        frozen = si['teachers'][n]['conditions']
        c3p = condition_3_prime(w, Runner())
        fwd, rev = native[n]
        tup = bat['tuples'][bat['teacher_idx'][n]]
        out[n] = dict(
            conditions_1_to_7_match=all(cond[k] == frozen[k] for k in cond),
            condition_3prime_match=c3p == frozen['3prime_satellite_feedback_changes_B'],
            forward_tuples_match=sorted(map(list, tup.get(fwd, []))) == sorted(frozen['8_causal_chain_A_input_B']['qualifying']),
            reverse_tuples_match=sorted(map(list, tup.get(rev, []))) == sorted(si['teachers'][n]['reverse_chain_B_input_A']['qualifying']))
    passed = all(all(v.values()) for v in out.values())
    return dict(teachers=out, passed=passed)


class Index:
    """Evaluator-only views of the battery: per-atom value partitions as hypothesis bitsets."""
    def __init__(self, bat):
        self.keys = bat['keys']
        self.table = bat['table']
        self.part = []
        for a in range(len(self.keys)):
            d = defaultdict(int)
            for h in range(N_VECTORS):
                d[self.table[h][a]] |= 1 << h
            self.part.append(dict(d))
        self.by_item_query = defaultdict(list)
        for a, k in enumerate(self.keys):
            self.by_item_query[(k[0], atom_query(k))].append(a)
        self.key_index = {k: a for a, k in enumerate(self.keys)}

    def atoms(self, items, queries):
        return [a for it in items for q in queries for a in self.by_item_query[(it, q)]]

    def eq(self, a, h):
        return self.part[a][self.table[h][a]]

    def equal_on(self, atoms, h):
        bits = (1 << N_VECTORS) - 1
        for a in atoms:
            bits &= self.eq(a, h)
        return bits

    def any_true(self, atoms):
        bits = 0
        for a in atoms:
            bits |= self.part[a].get(1, 0)
        return bits


ALL_Q = list(range(len(QUERIES)))
SIG_ITEMS = {'O': ('B1',), 'I': ('B1', 'B6', 'B7')}


def members(bits):
    return [h for h in range(N_VECTORS) if bits >> h & 1]


def mirror(w):
    return w[:4] + (-w[4], -w[5]) + w[6:]


def ceilings(ix, bat):
    mech_atoms = ix.atoms(MECHANISM_ITEMS, ALL_Q)
    fwd = {t: 0 for t in FORWARD}
    for a in ix.atoms(['B8E'], ALL_Q):
        if ix.keys[a][1] in FORWARD:
            fwd[ix.keys[a][1]] |= ix.part[a].get(1, 0)
    out = {}
    for n, th in bat['teacher_idx'].items():
        out[n] = {}
        for arm, items in SIG_ITEMS.items():
            cls = ix.equal_on(ix.atoms(items, ALL_Q), th)
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
                                  differing_atoms=len(diff),
                                  differing_by_item=dict(Counter(ix.keys[a][0] for a in diff)),
                                  examples=[dict(atom=list(ix.keys[a]), teacher=decode(ix.keys[a], ix.table[th][a]),
                                                 member=decode(ix.keys[a], ix.table[h][a])) for a in diff[:8]]))
            false_fwd = {t: popcount(cls & b) for t, b in fwd.items()}
            false_fwd['union'] = popcount(cls & (fwd['SF'] | fwd['CF']))
            mir = hypothesis_index(mirror(vector(th)))
            out[n][arm] = dict(
                class_size=len(mem), mechanism_subclasses=len(groups),
                teacher_subclass_size=len(groups[tkey]), mechanism_homogeneous=len(groups) == 1,
                separating_atoms_by_item=dict(varying), members_with_B8_differences_by_template=b8_wrong,
                members_with_forward_witness=false_fwd,
                core_sign_mirror=dict(index=mir, in_class=bool(cls >> mir & 1),
                                      mechanism_equivalent=mir in groups[tkey], is_teacher=mir == th),
                indistinguishable_different_mechanism=pairs[:10], further_pairs_not_listed=max(0, len(pairs) - 10))
    return out


# ---------------------------------------------------------------- evaluation

def b5_noop(key, teacher_w):
    return key[0] == 'B5' and teacher_w[COEFFS.index(key[1])] == 0


ITEM_ROWS = ('B2', 'B3', 'B4', 'B5g', 'B5n', 'B6', 'B7', 'B8E', 'B8W')


def item_label(key, tw):
    if key[0] == 'B5':
        return 'B5n' if b5_noop(key, tw) else 'B5g'
    return key[0]


class Evaluator:
    def __init__(self, ix, bat):
        self.ix, self.bat = ix, bat
        self.cache = {}
        self.fwd = {t: 0 for t in FORWARD}
        for a in ix.atoms(['B8E'], ALL_Q):
            if ix.keys[a][1] in FORWARD:
                self.fwd[ix.keys[a][1]] |= ix.part[a].get(1, 0)
        self.mech_all = {}
        for n, th in bat['teacher_idx'].items():
            self.mech_all[n] = ix.equal_on(ix.atoms(MECHANISM_ITEMS, ALL_Q), th)

    def vs(self, n, V, Q):
        key = (n, V, tuple(Q))
        if key in self.cache:
            return self.cache[key]
        ix, th = self.ix, self.bat['teacher_idx'][n]
        tw = TEACHERS[n]
        c = Counter()
        violations = 0
        for a in ix.atoms(['B1'], Q):
            T = ix.table[th][a]
            vals = [v for v, b in ix.part[a].items() if b & V]
            certain, possible = 511, 0
            for v in vals:
                certain &= v
                possible |= v
            c['n'] += 1
            c['certified'] += len(vals) == 1
            c['certified_correct'] += vals == [T]
            c['certified_incorrect'] += len(vals) == 1 and vals[0] != T
            c['possible_tp'] += popcount(possible & T)
            c['possible_fp'] += popcount(possible & ~T)
            c['certain_tp'] += popcount(certain & T)
            c['certain_fn'] += popcount(T & ~certain)
            violations += bool(certain & ~T) or bool(T & ~possible)
        for a in ix.atoms(MECHANISM_ITEMS, Q):
            k = ix.keys[a]
            T = ix.table[th][a]
            vals = [v for v, b in ix.part[a].items() if b & V]
            lab = item_label(k, tw)
            c[f'{lab}_atoms'] += 1
            c[f'{lab}_certified'] += len(vals) == 1
            c[f'{lab}_certified_correct'] += vals == [T]
            c[f'{lab}_certified_incorrect'] += len(vals) == 1 and vals[0] != T
            if k[0] in ('B8E', 'B8W'):
                c[f'{lab}_teacher_true'] += T == 1
                c[f'{lab}_certified_true_when_teacher_true'] += T == 1 and vals == [1]
                c[f'{lab}_teacher_false'] += T == 0
                c[f'{lab}_possibly_true_when_teacher_false'] += T == 0 and 1 in vals
                violations += T not in vals
            else:
                certain, possible = -1, 0
                for v in vals:
                    certain &= v
                    possible |= v
                violations += bool(certain & ~T) or bool(T & ~possible)
        obs = ix.equal_on(ix.atoms(['B1'], Q), th)
        mech = ix.equal_on(ix.atoms(MECHANISM_ITEMS, Q), th)
        c['survivors'] = popcount(V)
        c['teacher_in_V'] = V >> th & 1
        c['mech_equivalent_on_partition'] = popcount(V & mech)
        c['mech_equivalent_all_81'] = popcount(V & self.mech_all[n])
        c['b1_agreeing_survivors'] = popcount(V & obs)
        c['mechanism_uncertain_survivors'] = popcount(V & obs & ~mech)
        c['output_disagreeing_survivors'] = popcount(V & ~obs)
        c['forward_attribution_SF'] = popcount(V & self.fwd['SF'])
        c['forward_attribution_CF'] = popcount(V & self.fwd['CF'])
        c['forward_attribution_union'] = popcount(V & (self.fwd['SF'] | self.fwd['CF']))
        c['containment_violations'] = violations
        self.cache[key] = dict(c)
        return self.cache[key]

    def single(self, n, h, Q):
        ix, th, tw = self.ix, self.bat['teacher_idx'][n], TEACHERS[n]
        c = Counter()
        for a in ix.atoms(['B1'], Q):
            T, P = ix.table[th][a], ix.table[h][a]
            c['n'] += 1
            c['exact'] += P == T
            c['tp'] += popcount(P & T)
            c['fp'] += popcount(P & ~T)
            c['fn'] += popcount(T & ~P)
            c['signature_exact'] += signatures(P) == signatures(T)
            act = Counter(len(s) for s in signatures(P))
            c['predicted_no_activity'] += act[0]
            c['predicted_single'] += act[1]
            c['predicted_joint'] += act[2]
        for a in ix.atoms(MECHANISM_ITEMS, Q):
            k = ix.keys[a]
            T, P = ix.table[th][a], ix.table[h][a]
            lab = item_label(k, tw)
            c[f'{lab}_atoms'] += 1
            c[f'{lab}_correct'] += P == T
            if k[0] in ('B8E', 'B8W'):
                c[f'{lab}_teacher_true'] += T == 1
                c[f'{lab}_true_when_teacher_true'] += T == 1 and P == 1
                c[f'{lab}_teacher_false'] += T == 0
                c[f'{lab}_true_when_teacher_false'] += T == 0 and P == 1
        c['forward_witness_SF'] = self.fwd['SF'] >> h & 1
        c['forward_witness_CF'] = self.fwd['CF'] >> h & 1
        return dict(c)

    def memory(self, n, mem, arm, Q):
        ix, th = self.ix, self.bat['teacher_idx'][n]
        c = Counter()
        for qi in Q:
            x = QUERIES[qi]
            for key in observation_keys(x, 'I'):
                item = key[0]
                a = ix.key_index[('B1', qi)] if item == 'B1' else ix.key_index[(item, qi, key[1])]
                pred = mem.predict(x, key) if key in observation_keys(x, arm) else None
                c[f'{item}_n'] += 1
                if pred is None:
                    c[f'{item}_unknown'] += 1
                else:
                    T = ix.table[th][a]
                    enc = sum(1 << _code(item, e) for e in pred)
                    c[f'{item}_correct'] += enc == T
        return dict(c)


def _code(item, e):
    if item == 'B1':
        return 3 * (e[0] + 1) + (e[1] + 1)
    return 9 * (e[0] + 1) + 3 * (e[1] + 1) + (e[2] + 1)


def signatures(mask):
    return frozenset(('A' if a else '') + ('B' if b else '') for a in (-1, 0, 1) for b in (-1, 0, 1)
                     if mask >> (3 * (a + 1) + (b + 1)) & 1)


def map_atoms(ix, qi, keys):
    return [ix.key_index[('B1', qi)] if k[0] == 'B1' else ix.key_index[(k[0], qi, k[1])] for k in keys]


def feedback_map(ix, th, qi, arm):
    x = QUERIES[qi]
    out = {}
    for key in observation_keys(x, arm):
        a = map_atoms(ix, qi, [key])[0]
        m = ix.table[th][a]
        if key[0] == 'B1':
            out[key] = frozenset((a_, b_) for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) if m >> _code('B1', (a_, b_)) & 1)
        else:
            out[key] = frozenset((i, j, s) for i in (-1, 0, 1) for j in (-1, 0, 1) for s in (-1, 0, 1)
                                 if m >> _code('B6', (i, j, s)) & 1)
    return out


VS_COLS = None


def learning(ix, bat, splits, ev):
    rows, runs, checks = [], [], Counter()
    final_V = defaultdict(set)
    local_final = []
    sims = {arm: Simulator(arm) for arm in ('O', 'I')}
    work = defaultdict(Counter)
    for n in TEACHERS:
        th = bat['teacher_idx'][n]
        for split in splits:
            parts = dict(within=split['within_context'], mixed=split['mixed_evidence'])
            prefix_bits = {}
            for o, order in enumerate(split['orders']):
                for arm in ('O', 'I'):
                    sim = sims[arm]
                    L = dict(full=VersionSpace(sim), positive=VersionSpace(sim, positive_only=True),
                             local=BlockLearner(sim), memory=Memory())
                    history = []
                    ret = Counter()
                    done = 0
                    for k in CHECKPOINTS:
                        while done < k:
                            qi = order[done]
                            x = QUERIES[qi]
                            fb = feedback_map(ix, th, qi, arm)
                            prev_ok = [_map_ok(ix, L['local'].w, th, s, arm) for s in history]
                            for name, learner in L.items():
                                status = learner.update(x, fb)
                                checks['inconsistent_feedback'] += status == 'inconsistent_feedback'
                            history.append(qi)
                            done += 1
                            h_local = hypothesis_index(L['local'].w)
                            if done >= 2:
                                for s, was in zip(history[:-1], prev_ok):
                                    now = _map_ok(ix, L['local'].w, th, s, arm)
                                    ret['complete_map_checks'] += 1
                                    ret['complete_map_retained'] += now
                                    ret['b1_checks'] += 1
                                    ret['b1_retained'] += ix.table[h_local][ix.key_index[('B1', s)]] == ix.table[th][ix.key_index[('B1', s)]]
                                    ret['lost_previously_correct'] += was and not now
                            for vsn in ('full', 'positive'):
                                V = L[vsn].bits
                                checks[f'{vsn}_teacher_eliminated'] += not V >> th & 1
                                for s in history:
                                    for a in map_atoms(ix, s, observation_keys(QUERIES[s], arm)):
                                        T = ix.table[th][a]
                                        if vsn == 'full':
                                            bad = V & ~ix.part[a][T]
                                        else:
                                            sup = 0
                                            for val, b in ix.part[a].items():
                                                if val & T == T:
                                                    sup |= b
                                            bad = V & ~sup
                                        checks[f'{vsn}_retention_violations'] += bad != 0
                        prefix_bits[(o, arm, k)] = {v: L[v].bits for v in ('full', 'positive')}
                        if arm == 'I':
                            for v in ('full', 'positive'):
                                checks['armI_subset_checks'] += 1
                                checks['armI_not_subset_of_armO'] += bool(prefix_bits[(o, 'I', k)][v] & ~prefix_bits[(o, 'O', k)][v])
                        seen = order[:k]
                        h_local = hypothesis_index(L['local'].w)
                        for part, Q in (('seen', seen), ('within', parts['within']), ('mixed', parts['mixed'])):
                            base = [n, arm, split['seed'], o, k, part]
                            for vsn in ('full', 'positive'):
                                r = ev.vs(n, L[vsn].bits, Q)
                                checks['containment_violations'] += r['containment_violations']
                                rows.append(base + [vsn, r])
                            rows.append(base + ['local', ev.single(n, h_local, Q)])
                            rows.append(base + ['zero', ev.single(n, hypothesis_index((0,) * 7), Q)])
                            rows.append(base + ['teacher', ev.single(n, th, Q)])
                            rows.append(base + ['memory', ev.memory(n, L['memory'], arm, Q)])
                    for v in ('full', 'positive'):
                        final_V[(n, arm, split['seed'], v)].add(L[v].bits)
                        work[f'{arm}_{v}']['evaluations'] += L[v].evaluations
                    work[f'{arm}_local']['evaluations'] += L['local'].evaluations
                    work[f'{arm}_local']['commits'] += L['local'].commits
                    work[f'{arm}_local']['coefficients_changed'] += L['local'].coefficients_changed
                    local_final.append(dict(teacher=n, arm=arm, split=split['seed'], order=o, w=list(L['local'].w),
                                            index=h_local, retention=dict(ret)))
                    runs.append(dict(teacher=n, arm=arm, split=split['seed'], order=o,
                                     final_survivors={v: popcount(L[v].bits) for v in ('full', 'positive')},
                                     local_w=list(L['local'].w), local_commits=L['local'].commits,
                                     memory_entries=len(L['memory'].table),
                                     full_V_hash=hashlib.sha256(str(L['full'].bits).encode()).hexdigest()[:16]))
    for key, vs in final_V.items():
        checks['order_dependent_final_V'] += len(vs) != 1
    for arm, sim in sims.items():
        work[f'simulator_{arm}'] = Counter(requests=sim.requests, unique=len(sim.memo))
    return rows, runs, dict(checks), local_final, {k: dict(v) for k, v in work.items()}, final_V


def _map_ok(ix, w, th, s, arm):
    h = hypothesis_index(w)
    return all(ix.table[h][a] == ix.table[th][a] for a in map_atoms(ix, s, observation_keys(QUERIES[s], arm)))


# ---------------------------------------------------------------- aggregation and targets

def pooled(rows, n, arm, learner, k, part, splits=None, orders=None):
    out = Counter()
    for r in rows:
        if r[0] == n and r[1] == arm and r[4] == k and r[5] == part and r[6] == learner \
                and (orders is None or r[3] in orders) and (splits is None or r[2] in splits):
            out.update(r[7])
    return dict(out)


def ratio(a, b):
    return None if not b else round(a / b, 6)


def local_gate(w, cache):
    from experiments.shared_input_mediation import Runner, chain, condition_3_prime, conditions_1_to_7
    if w in cache:
        return cache[w]
    cond, _, _ = conditions_1_to_7(w)
    runner = Runner()
    c3p = condition_3_prime(w, runner)
    res = {}
    for tmpl in ('SF', 'CF'):
        spec = TEMPLATE_SPEC[tmpl]
        ch = chain(w, dict(donor=spec['donor'], recipient=spec['recipient'], links=spec['links']), 'forward', runner)
        res[tmpl] = all(v['passed'] for v in cond.values()) and c3p['passed'] and ch['passed']
    cache[w] = dict(res, failed_1_to_7=[k for k, v in cond.items() if not v['passed']],
                    condition_3prime=c3p['passed'])
    return cache[w]


def targets(rows, ix, bat, ceil, local_final, splits):
    gate_cache = {}
    out = {}
    split_ids = [s['seed'] for s in splits]
    for n in TEACHERS:
        th = bat['teacher_idx'][n]
        t = {}
        # 1. output identification, arm O, one V per split (order 0)
        for arm in ('O', 'I'):
            for part in ('within', 'mixed'):
                c = pooled(rows, n, arm, 'full', 15, part, orders=[0])
                t[f'output_identification_{arm}_{part}'] = dict(
                    certified=c['certified'], n=c['n'], coverage=ratio(c['certified'], c['n']),
                    certified_incorrect=c['certified_incorrect'],
                    passed=c['n'] > 0 and c['certified'] >= 0.8 * c['n'] and c['certified_incorrect'] == 0,
                    positive_only=dict(certified=pooled(rows, n, arm, 'positive', 15, part, orders=[0])['certified']))
        # 2-3. mechanism identification per split (all 81 queries)
        for arm in ('O', 'I'):
            per_split = []
            for sid in split_ids:
                c = pooled(rows, n, arm, 'full', 15, 'within', splits=[sid], orders=[0])
                p = pooled(rows, n, arm, 'positive', 15, 'within', splits=[sid], orders=[0])
                per_split.append(dict(split=sid, survivors=c['survivors'], mech_equivalent=c['mech_equivalent_all_81'],
                                      fraction=ratio(c['mech_equivalent_all_81'], c['survivors']),
                                      all_equivalent=c['mech_equivalent_all_81'] == c['survivors'],
                                      positive_only_survivors=p['survivors'],
                                      positive_only_mech_equivalent=p['mech_equivalent_all_81']))
            hetero = not ceil[n][arm]['mechanism_homogeneous']
            ok = sum(s['all_equivalent'] for s in per_split)
            t[f'mechanism_identification_{arm}'] = dict(
                splits_all_survivors_equivalent=ok, per_split=per_split,
                status=('pass' if ok >= 4 else 'unattainable_by_identifiability' if hetero else 'fail'))
        o, i = t['mechanism_identification_O'], t['mechanism_identification_I']
        t['arm_I_strictly_improves'] = dict(
            split_count=i['splits_all_survivors_equivalent'] > o['splits_all_survivors_equivalent'],
            fraction_any_split=any((b['fraction'] or 0) > (a['fraction'] or 0) for a, b in zip(o['per_split'], i['per_split'])),
            fraction_all_splits_at_least=all((b['fraction'] or 0) >= (a['fraction'] or 0) for a, b in zip(o['per_split'], i['per_split'])))
        # 4. local learner per arm
        positive_atoms = [a for a, k in enumerate(ix.keys) if k[0] == 'B8W' and ix.table[th][a] == 1]
        for arm in ('O', 'I'):
            loc = {}
            for part in ('within', 'mixed'):
                c = pooled(rows, n, arm, 'local', 15, part)
                z = pooled(rows, n, arm, 'zero', 15, part)
                loc[part] = dict(exact=c['exact'], n=c['n'], accuracy=ratio(c['exact'], c['n']),
                                 zero_accuracy=ratio(z['exact'], z['n']),
                                 passed=c['n'] > 0 and c['exact'] >= 0.8 * c['n'] and c['exact'] > z['exact'])
            ret = Counter()
            runs = [r for r in local_final if r['teacher'] == n and r['arm'] == arm]
            for r in runs:
                ret.update(r['retention'])
            behaviour = []
            for r in runs:
                g = local_gate(tuple(r['w']), gate_cache)
                wit = all(ix.table[r['index']][a] == 1 for a in positive_atoms)
                false_w = sum(ix.table[r['index']][a] == 1 and ix.table[th][a] == 0
                              for a, k in enumerate(ix.keys) if k[0] == 'B8W')
                passes = all(g[tm] for tm in GATE_TEMPLATE[n])
                behaviour.append(dict(split=r['split'], order=r['order'], w=r['w'], reproduces_positive_witnesses=wit,
                                      false_witness_atoms=false_w, gate=g, gate_passed_teacher_template=passes,
                                      recovered=wit and passes))
            recovered = sum(b['recovered'] for b in behaviour)
            retention = ratio(ret['complete_map_retained'], ret['complete_map_checks'])
            t[f'local_{arm}'] = dict(
                partitions=loc, retention=dict(ret, complete_map_rate=retention,
                                               b1_rate=ratio(ret['b1_retained'], ret['b1_checks'])),
                retention_passed=retention is not None and retention >= 0.95,
                runs_recovering_behaviour=recovered, positive_teacher_witness_atoms=len(positive_atoms),
                behaviour_by_split={s: sum(b['recovered'] for b in behaviour if b['split'] == s) for s in split_ids},
                runs=behaviour,
                passed=all(v['passed'] for v in loc.values()) and retention is not None and retention >= 0.95
                and recovered >= 13)
        # 5. negative controls
        if n in NEGATIVE:
            teacher_fwd = {tm: any(ix.table[th][a] == 1 for a in ix.atoms(['B8E'], ALL_Q) if ix.keys[a][1] == tm)
                           for tm in FORWARD}
            neg = dict(teacher_forward_witness=teacher_fwd, assumption_refuted=any(teacher_fwd.values()))
            for arm in ('O', 'I'):
                per = []
                for sid in split_ids:
                    c = pooled(rows, n, arm, 'full', 15, 'within', splits=[sid], orders=[0])
                    per.append(dict(split=sid, survivors=c['survivors'], SF=c['forward_attribution_SF'],
                                    CF=c['forward_attribution_CF'], union=c['forward_attribution_union']))
                obstructed = ceil[n][arm]['members_with_forward_witness']['union'] > 0
                zero = all(p['union'] == 0 for p in per)
                lf = [r for r in local_final if r['teacher'] == n and r['arm'] == arm]
                local_rate = sum(bool(ev_fwd(ix, r['index'])) for r in lf)
                neg[arm] = dict(per_split=per, zero_false_attribution=zero,
                                local_runs_with_forward_witness=local_rate, local_runs=len(lf),
                                status=('invalid_assumption_refuted' if neg['assumption_refuted'] else
                                        'pass' if zero else 'unattainable_by_identifiability' if obstructed else 'fail'))
            t['negative_control'] = neg
        if n not in POSITIVE:
            for k in list(t):
                if k != 'negative_control':
                    t[k] = dict(t[k], target_applies=False,
                                note='targets 1-4 are declared for S1 and C2 only; reported, not scored')
        out[n] = t
    return out


def ev_fwd(ix, h):
    return any(ix.table[h][a] == 1 for a in ix.atoms(['B8E'], ALL_Q) if ix.keys[a][1] in FORWARD)


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_mechanism_learning')
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(run=res.testsRun, failures=len(res.failures), errors=len(res.errors), passed=res.wasSuccessful())


def completion_status(gate, tests):
    if not gate['passed']:
        return 'stopped_failed_correctness_gate'
    return 'complete' if tests['passed'] else 'stopped_failed_tests'


ROW_HEAD = ['teacher', 'arm', 'split', 'order', 'checkpoint', 'partition', 'learner', 'counts']


def write_rows_report(path, head, rows):
    text = json.dumps(head, indent=1)
    body = ',\n'.join('  ' + json.dumps(r, sort_keys=True) for r in rows)
    path.write_text(text[:-2] + ',\n "rows": [\n' + body + '\n ]\n}\n')


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
    timing = {}
    t0 = time.perf_counter()
    bat = build_battery(args.workers)
    timing['battery'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    og = oracle_gate(bat)
    rep = reproduce_frozen(bat)
    timing['oracle_and_reproduction'] = time.perf_counter() - t0
    teacher_entries = {n: {json.dumps(list(k)): decode(k, bat['table'][h][a]) for a, k in enumerate(bat['keys'])}
                       for n, h in bat['teacher_idx'].items()}
    summary = dict(experiment=PREFIX, manifest_sha256=m['sha256'], atom_schema=bat['schema'],
                   atom_schema_sha256=bat['schema_hash'], atoms=len(bat['keys']),
                   fixed_witness_atoms=len(bat['wkeys']), cache=bat['stats'],
                   hypothesis_hashes=[atom_hash(v) for v in bat['table']],
                   battery_sha256=hashlib.sha256(''.join(atom_hash(v) for v in bat['table']).encode()).hexdigest(),
                   teacher_indices=bat['teacher_idx'],
                   teacher_qualifying_tuples={n: {t: [list(x) for x in v] for t, v in bat['tuples'][h].items()}
                                              for n, h in bat['teacher_idx'].items()},
                   teacher_entries=teacher_entries, oracle_verification=og, frozen_reproduction=rep,
                   reconstruction='src.mechanism_learning.reconstruct(h, witness keys) regenerates any vector')
    dump(out / f'{PREFIX}-battery-summary.json', summary)
    if not (og['passed'] and rep['passed']):
        sys.exit('correctness gate failed before ceilings; stopping (see battery summary)')
    t0 = time.perf_counter()
    ix = Index(bat)
    ceil = ceilings(ix, bat)
    timing['ceilings'] = time.perf_counter() - t0
    dump(out / f'{PREFIX}-ceiling.json', dict(experiment=PREFIX, manifest_sha256=m['sha256'],
                                              atom_schema_sha256=bat['schema_hash'], ceilings=ceil))
    splits = json.loads((ROOT / SPLITS).read_text())
    if [[q.e1, q.e2, q.da, q.db] for q in QUERIES] != splits['queries']:
        sys.exit('query order differs from the frozen split file')
    t0 = time.perf_counter()
    ev = Evaluator(ix, bat)
    rows, runs, checks, local_final, work, _ = learning(ix, bat, splits['splits'], ev)
    timing['learning_and_evaluation'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tg = targets(rows, ix, bat, ceil, local_final, splits['splits'])
    timing['targets_and_local_gates'] = time.perf_counter() - t0
    gate = dict(oracle=og['passed'], frozen_reproduction=rep['passed'],
                teacher_eliminated=checks.get('full_teacher_eliminated', 0) + checks.get('positive_teacher_eliminated', 0),
                full_retention_violations=checks.get('full_retention_violations', 0),
                positive_retention_violations=checks.get('positive_retention_violations', 0),
                containment_violations=checks.get('containment_violations', 0),
                order_dependent_final_V=checks.get('order_dependent_final_V', 0),
                armI_not_subset_of_armO=checks.get('armI_not_subset_of_armO', 0),
                inconsistent_feedback=checks.get('inconsistent_feedback', 0))
    gate['passed'] = gate['oracle'] and gate['frozen_reproduction'] and all(
        v == 0 for k, v in gate.items() if k not in ('oracle', 'frozen_reproduction'))
    t0 = time.perf_counter()
    tests = run_tests()
    timing['tests'] = time.perf_counter() - t0
    head = dict(experiment=PREFIX, contract=CONTRACT, manifest_sha256=m['sha256'],
                completion=completion_status(gate, tests),
                correctness_gate=gate, checks=checks, tests=tests, targets=tg, work=work, runs=runs,
                logical_storage=dict(version_space_bits=N_VECTORS, local_model='7 ternary coefficients',
                                     memory='stored (query, key) entries; see runs'),
                provenance=dict(python=platform.python_version(), implementation=platform.python_implementation(),
                                source_sha256={p: sha(p) for p in (
                                    'src/mechanism_learning.py', 'experiments/mechanism_learning.py',
                                    'tests/test_mechanism_learning.py', CONTRACT)}),
                row_columns=ROW_HEAD)
    write_rows_report(out / f'{PREFIX}-learning.json', head, rows)
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rc = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: round(v, 3) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3), workers=args.workers,
        peak_rss_kib=dict(main=ru, largest_child=rc), python=platform.python_version(),
        machine=platform.machine(), platform=platform.platform(), out_dir=str(out),
        git_revision=git('rev-parse', 'HEAD'), dirty_worktree=bool(git('status', '--porcelain'))))
    print(head['completion'], json.dumps(gate))


if __name__ == '__main__':
    main()
