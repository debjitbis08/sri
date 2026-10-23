"""Shared input interpretation v1: frozen manifest, mechanism gate (no learning).

Run `.venv/bin/python -m experiments.shared_input_mediation [--out-dir DIR]`.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

from src.shared_input_mediation import (
    A, B, COEFFS, DOMAINS, I1, I2, K, QUERIES, all_states, clamp_sets, controls, core_disconnected, explore,
    influence_witnesses, signature, start_state, zeroed,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'shared-input-mediation-v1'
CONTRACT = 'docs/shared-input-mediation-v1-contract.md'
COEFF_ORDER = ('w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12')
T1, T3 = (3, 0, 0, -3, 3, -3), (0, 3, -3, 0, -3, 3)
TEACHERS = {
    'S1': (3, 3, 3, 0, 3, -3, 0),
    'S2': (3, 3, 3, 0, 3, 3, 0),
    'S3': (3, 3, -3, 0, 3, -3, 0),
    'C1': T1 + (3,),
    'C2': T1 + (-3,),
    'C3': T3 + (3,),
}
REGRESSION = {'T0': (3, 0, 0, -3, 0, 0, 0), 'T1': T1 + (0,), 'T2': (3, 0, 0, -3, 3, 3, 0), 'T3': T3 + (0,)}
# Chains: donor input coordinate (receives the source satellite's feedback), recipient input
# coordinate (influences the target satellite), and the declared chain-link coefficient indices.
CHAINS = {
    'forward': {  # A -> input -> B
        'S1': dict(donor='i1', recipient='i1', links=['w_1A', 'w_1B']),
        'S2': dict(donor='i1', recipient='i1', links=['w_1A', 'w_1B']),
        'S3': dict(donor='i1', recipient='i1', links=['w_1A', 'w_1B']),
        'C1': dict(donor='i1', recipient='i2', links=['w_1A', 'w_12', 'w_2B']),
        'C2': dict(donor='i1', recipient='i2', links=['w_1A', 'w_12', 'w_2B']),
        'C3': dict(donor='i2', recipient='i1', links=['w_2A', 'w_12', 'w_1B']),
    },
    'reverse': {  # B -> input -> A (A/B and donor/recipient swapped)
        'S1': dict(donor='i1', recipient='i1', links=['w_1B', 'w_1A']),
        'S2': dict(donor='i1', recipient='i1', links=['w_1B', 'w_1A']),
        'S3': dict(donor='i1', recipient='i1', links=['w_1B', 'w_1A']),
        'C1': dict(donor='i2', recipient='i1', links=['w_2B', 'w_12', 'w_1A']),
        'C2': dict(donor='i2', recipient='i1', links=['w_2B', 'w_12', 'w_1A']),
        'C3': dict(donor='i1', recipient='i2', links=['w_1B', 'w_12', 'w_2A']),
    },
}
INTERVENTION_RULES = [
    'every intervention is a fresh exploration from the canonical start (e1,e2,0,0,0) with only the named coordinates replaced and frozen',
    'do(A=v): A starts at v and is frozen; ordered pairs v != v over D_A, all enumerated',
    'do(i_l=r): only the recipient input coordinate replaced and frozen; r in U_v^l, r2 in U_v2^l, r != r2; A fixed at v in both runs',
    'do(I=u): both input coordinates replaced by u and frozen; evidence unchanged',
    'NF: input updates use only E_I = 2((i1-e1)^2+(i2-e2)^2) - w_12 i1 i2; K, A, B use full E_C differences',
    'edge ablation: exactly the named coefficient set to zero before exploration',
    'core disconnected: w_KA = w_KB = 0, w_12 kept',
    'all comparisons are set comparisons over complete terminal sets',
]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1) + '\n')


def manifest():
    body = dict(
        schema=PREFIX, coefficient_order=list(COEFF_ORDER),
        teachers={k: list(v) for k, v in TEACHERS.items()},
        regression_controls={k: list(v) for k, v in REGRESSION.items()},
        chains=CHAINS, query_order='(e1, e2, domain_A, domain_B), v1 order',
        queries=[[q.e1, q.e2, q.da, q.db] for q in QUERIES], intervention_rules=INTERVENTION_RULES,
        contract=CONTRACT, contract_sha256=hashlib.sha256((ROOT / CONTRACT).read_bytes()).hexdigest())
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    m = manifest()
    path = out / f'{PREFIX}-manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != m:
            sys.exit('manifest differs from the frozen file (teachers, rules or contract changed); refusing to run')
    else:
        dump(path, m)
    return m


EXAMPLES = 3
COORD = dict(i1=I1, i2=I2)
INPUT_VECTORS = [(u1, u2) for u1 in (-1, 0, 1) for u2 in (-1, 0, 1)]


class Runner:
    """Every exploration used for scoring 3' and 8, memoized and recorded for oracle re-checking."""
    def __init__(self):
        self.cache = {}

    def run(self, x, w, clamps=(), nf=False):
        key = (x, tuple(w), tuple(clamps), nf)
        if key not in self.cache:
            self.cache[key] = explore(x, w, start=start_state(x, clamps), frozen=tuple(c for c, _ in clamps), nf=nf)
        return self.cache[key]


def ys(s):
    return sorted(list(p) for p in s)


def sigs(Y):
    return frozenset(signature(a, b) for a, b in Y)


# ---------------------------------------------------------------- conditions 1-7 (v1 section 5 verbatim)

def conditions_1_to_7(w):
    """Same definitions and output structure as the v1 phase-A gate."""
    counts = Counter()
    ex = Counter()
    fixed_k_match = {kv: 0 for kv in (-1, 0, 1)}
    unique, nonempty = {}, set()
    wd = core_disconnected(w)
    rows = []
    for qi, q in enumerate(QUERIES):
        r = explore(q, w)
        c = controls(q, w)
        Y, T = r.Y, r.terminals
        counts['domain_violations'] += sum(z[A] not in DOMAINS[q.da] or z[B] not in DOMAINS[q.db] for z in r.reachable)
        counts['evidence_mutations'] += r.query != q
        counts['cyclic_graphs'] += not r.acyclic
        tsigs = frozenset(signature(t[A], t[B]) for t in T)
        nonempty |= {g for g in tsigs if g}
        if len(tsigs) == 1 and next(iter(tsigs)):
            unique[qi] = next(iter(tsigs))
        revised = [t for t in T if t[:2] != (q.e1, q.e2)]
        counts['revision_possible_queries'] += bool(revised)
        counts['revision_necessary_queries'] += len(revised) == len(T)
        for key, targets, neighbors in (('forward_by_input', (A, B), (I1, I2)), ('forward_by_core', (A, B), (K,)),
                                        ('reverse_on_input', (I1, I2), (A, B)), ('core_responds', (K,), (A, B))):
            counts[key + '_queries'] += bool(influence_witnesses(q, w, r.reachable, targets, neighbors))
        for key, other in (('feedback', c['input_frozen']), ('core', c['core_disconnected'])):
            if Y != other:
                counts[f'{key}_participation' if sigs(Y) != sigs(other) else f'{key}_sign_only'] += 1
        for kv in (-1, 0, 1):
            fixed_k_match[kv] += c[f'core_fixed_{kv}'] == Y
        counts['mixed_sign_joint'] += any(a * b == -1 for a, b in Y)
        a_marg, b_marg = {a for a, _ in Y}, {b for _, b in Y}
        counts['cross_satellite_exclusion'] += any((a, b) not in Y for a in a_marg for b in b_marg)
        counts['domain_pairs_excluded_by_dynamics'] += len(DOMAINS[q.da]) * len(DOMAINS[q.db]) - len(Y)
        for key, clamped in (('clamp_B_to_A', B), ('clamp_A_to_B', A)):
            full_sets, cut_sets = clamp_sets(q, w, clamped), clamp_sets(q, wd, clamped)
            vals = sorted(full_sets)
            for i, v in enumerate(vals):
                for v2 in vals[i + 1:]:
                    if full_sets[v] != full_sets[v2]:
                        mediated = cut_sets[v] == cut_sets[v2]
                        counts[f'{key}_core_mediated' if mediated else f'{key}_not_core_mediated'] += 1
        rows.append(dict(query=qi, x=q.to_json(), Y=ys(Y), terminals=sorted(list(t) for t in T),
                         signatures=sorted(tsigs), controls={k: ys(v) for k, v in c.items()}))
    differ = {k: sum(row['controls'][k] != row['Y'] for row in rows) for k in rows[0]['controls']}
    part = {k: sum(sorted({signature(a, b) for a, b in row['controls'][k]})
                   != sorted({signature(a, b) for a, b in row['Y']}) for row in rows) for k in rows[0]['controls']}
    distinct_unique = sorted(set(unique.values()))
    cond = {
        '1_state_conditioned_influence': dict(
            passed=bool(counts['forward_by_input_queries'] + counts['forward_by_core_queries'])
            and bool(counts['reverse_on_input_queries']),
            queries_with_satellite_move_influenced_by_input=counts['forward_by_input_queries'],
            queries_with_satellite_move_influenced_by_core=counts['forward_by_core_queries'],
            queries_with_input_move_influenced_by_satellite=counts['reverse_on_input_queries'],
            queries_with_core_move_influenced_by_satellite=counts['core_responds_queries'],
            witness_pairs_co_reachable_from_same_query=True),
        '2_realized_input_revision': dict(
            passed=counts['revision_possible_queries'] > 0,
            queries_revision_possible=counts['revision_possible_queries'],
            queries_revision_at_every_terminal=counts['revision_necessary_queries']),
        '3_feedback_necessity': dict(
            passed=differ['input_frozen'] > 0, queries_Y_differs_from_input_frozen=differ['input_frozen'],
            participation_changes=counts['feedback_participation'], sign_only_changes=counts['feedback_sign_only']),
        '4_core_contribution': dict(
            passed=differ['core_disconnected'] > 0 and part['core_disconnected'] > 0
            and all(m < len(QUERIES) for m in fixed_k_match.values()),
            queries_Y_differs_from_core_disconnected=differ['core_disconnected'],
            queries_participation_differs=part['core_disconnected'],
            participation_changes=counts['core_participation'], sign_only_changes=counts['core_sign_only'],
            fixed_k_queries_reproduced={str(k): v for k, v in fixed_k_match.items()},
            some_fixed_k_reproduces_all=any(m == len(QUERIES) for m in fixed_k_match.values())),
        '5_selective_participation': dict(
            passed=len(nonempty) >= 2 and len(distinct_unique) >= 2,
            distinct_nonempty_signatures=sorted(nonempty),
            queries_with_unique_nonempty_signature=len(unique),
            distinct_unique_nonempty_signatures=distinct_unique,
            unique_signature_examples={g: [q for q, h in sorted(unique.items()) if h == g][:EXAMPLES]
                                       for g in distinct_unique},
            mixed_sign_joint_queries=counts['mixed_sign_joint']),
        '6_local_exclusion_preservation': dict(
            passed=counts['domain_violations'] == 0 and counts['evidence_mutations'] == 0,
            domain_violations=counts['domain_violations'], evidence_mutations=counts['evidence_mutations'],
            queries_with_cross_satellite_exclusion=counts['cross_satellite_exclusion'],
            domain_pairs_excluded_by_dynamics=counts['domain_pairs_excluded_by_dynamics']),
        '7_cross_satellite_influence_through_core': dict(
            passed=counts['clamp_B_to_A_core_mediated'] > 0 and counts['clamp_A_to_B_core_mediated'] > 0,
            B_clamp_pairs_changing_A_core_mediated=counts['clamp_B_to_A_core_mediated'],
            B_clamp_pairs_changing_A_not_core_mediated=counts['clamp_B_to_A_not_core_mediated'],
            A_clamp_pairs_changing_B_core_mediated=counts['clamp_A_to_B_core_mediated'],
            A_clamp_pairs_changing_B_not_core_mediated=counts['clamp_A_to_B_not_core_mediated'],
            note='clamps are initialization interventions, not canonical-start trajectories'),
    }
    return cond, rows, dict(controls_queries_with_different_Y=differ,
                            controls_queries_with_different_participation=part,
                            cyclic_graphs=counts['cyclic_graphs'])


# ---------------------------------------------------------------- condition 3' and condition 8

def condition_3_prime(w, runner):
    queries, activity, sign_only, frozen_diff = [], 0, 0, 0
    for qi, x in enumerate(QUERIES):
        full, nf = runner.run(x, w), runner.run(x, w, nf=True)
        pb, pb_nf = full.projection(B), nf.projection(B)
        frozen_diff += full.projection(B) != explore(x, w, frozen=(I1, I2)).projection(B)
        if pb != pb_nf:
            queries.append(qi)
            if {b != 0 for b in pb} != {b != 0 for b in pb_nf}:
                activity += 1
            else:
                sign_only += 1
    return dict(passed=bool(queries), queries_B_projection_differs_under_NF=len(queries),
                activity_set_changes=activity, sign_only_changes=sign_only, query_ids=queries,
                diagnostic_queries_B_projection_differs_under_full_input_freeze=frozen_diff)


def proj_contrast(r1, r2, coord):
    return r1.projection(coord) != r2.projection(coord)


def chain(w, ch, direction, runner):
    """Condition 8 (or its reverse): exhaustive over queries, ordered source-clamp pairs and recipient values."""
    src, tgt = (A, B) if direction == 'forward' else (B, A)
    j, l = COORD[ch['donor']], COORD[ch['recipient']]
    links = ch['links']
    off_chain = [COEFFS[i] for i, v in enumerate(w) if v and COEFFS[i] not in links]
    records, qualifying = [], []
    totals = Counter()
    for qi, x in enumerate(QUERIES):
        e_clamp = ((I1, x.e1), (I2, x.e2))
        for v in x.allowed(src):
            for v2 in x.allowed(src):
                if v == v2:
                    continue
                Tv, Tv2 = runner.run(x, w, ((src, v),)), runner.run(x, w, ((src, v2),))
                a_ok = proj_contrast(Tv, Tv2, j) and proj_contrast(Tv, Tv2, l)
                c_full = proj_contrast(Tv, Tv2, tgt)
                c_nf = not proj_contrast(runner.run(x, w, ((src, v),), nf=True),
                                         runner.run(x, w, ((src, v2),), nf=True), tgt)
                c_ie = not proj_contrast(runner.run(x, w, ((src, v),) + e_clamp),
                                         runner.run(x, w, ((src, v2),) + e_clamp), tgt)
                d = {}
                for link in links:
                    wz = zeroed(w, link)
                    d[link] = not proj_contrast(runner.run(x, wz, ((src, v),)), runner.run(x, wz, ((src, v2),)), tgt)
                b_pairs = []
                for r in sorted(Tv.projection(l)):
                    for r2 in sorted(Tv2.projection(l)):
                        if r == r2:
                            continue
                        Rr = runner.run(x, w, ((src, v), (l, r)))
                        Rr2 = runner.run(x, w, ((src, v), (l, r2)))
                        b_pairs.append(dict(r=r, r2=r2, b_ok=proj_contrast(Rr, Rr2, tgt)))
                b_any = any(p['b_ok'] for p in b_pairs)
                mask = dict(a=a_ok, b_any=b_any, c_full=c_full, c_NF=c_nf, c_I_equals_e=c_ie,
                            d=all(d.values()), d_links=d)
                for k in ('a', 'b_any', 'c_full', 'c_NF', 'c_I_equals_e', 'd'):
                    totals[k] += mask[k]
                totals['pairs'] += 1
                rec = dict(query=qi, v=v, v2=v2, mask=mask, recipient_pairs=b_pairs)
                ok = a_ok and c_full and c_nf and c_ie and all(d.values())
                if c_full:
                    # (e) reported only: core disconnected, off-chain deletions, every common input vector
                    wd = core_disconnected(w)
                    rec['e'] = dict(
                        survives_core_disconnected=proj_contrast(runner.run(x, wd, ((src, v),)),
                                                                 runner.run(x, wd, ((src, v2),)), tgt),
                        survives_off_chain_deletion={
                            c: proj_contrast(runner.run(x, zeroed(w, c), ((src, v),)),
                                             runner.run(x, zeroed(w, c), ((src, v2),)), tgt) for c in off_chain},
                        contrast_with_input_clamped_at={
                            f'{u[0]},{u[1]}': proj_contrast(runner.run(x, w, ((src, v), (I1, u[0]), (I2, u[1]))),
                                                            runner.run(x, w, ((src, v2), (I1, u[0]), (I2, u[1]))), tgt)
                            for u in INPUT_VECTORS},
                        contrast_with_input_clamped_at_core_disconnected={
                            f'{u[0]},{u[1]}': proj_contrast(runner.run(x, wd, ((src, v), (I1, u[0]), (I2, u[1]))),
                                                            runner.run(x, wd, ((src, v2), (I1, u[0]), (I2, u[1]))), tgt)
                            for u in INPUT_VECTORS})
                if ok:
                    for p in b_pairs:
                        if p['b_ok']:
                            qualifying.append((qi, v, v2, p['r'], p['r2']))
                records.append(rec)
    joint = Counter()
    for rec in records:
        m = rec['mask']
        if not m['c_full']:
            continue
        joint['c_full'] += 1
        joint['c_full_and_NF_vanishes'] += m['c_NF']
        joint['c_full_and_I_equals_e_vanishes'] += m['c_I_equals_e']
        if m['c_NF'] and m['c_I_equals_e']:
            joint['+both_vanish'] += 1
            if m['a']:
                joint['+a'] += 1
                if m['d']:
                    joint['+d'] += 1
                    joint['+b'] += m['b_any']
        joint['contrast_survives_core_disconnected'] += rec['e']['survives_core_disconnected']
    out = dict(direction=direction, donor=ch['donor'], recipient=ch['recipient'], links=links,
               joint_pair_counts=dict(joint),
               off_chain_coefficients=off_chain, passed=bool(qualifying), qualifying_tuples=len(qualifying),
               qualifying=[list(t) for t in qualifying], subcondition_pair_counts=dict(totals),
               pairs=records)
    if qualifying:
        qi, v, v2, r, r2 = qualifying[0]
        x = QUERIES[qi]
        Tv, Tv2 = runner.run(x, w, ((src, v),)), runner.run(x, w, ((src, v2),))
        origin = lambda T, val: sorted(list(t) for t in T.terminals if t[l] == val)[0]
        rec = next(rr for rr in records if (rr['query'], rr['v'], rr['v2']) == (qi, v, v2))
        out['example'] = dict(
            query=qi, x=x.to_json(), source_clamps=[v, v2], recipient_values=[r, r2],
            source_terminals=[sorted(list(t) for t in Tv.terminals), sorted(list(t) for t in Tv2.terminals)],
            recipient_value_origins=[origin(Tv, r), origin(Tv2, r2)],
            target_sets=[sorted(Tv.projection(tgt)), sorted(Tv2.projection(tgt))],
            recipient_clamp_target_sets=[sorted(runner.run(x, w, ((src, v), (l, r))).projection(tgt)),
                                         sorted(runner.run(x, w, ((src, v), (l, r2))).projection(tgt))],
            NF_target_sets=[sorted(runner.run(x, w, ((src, v),), nf=True).projection(tgt)),
                            sorted(runner.run(x, w, ((src, v2),), nf=True).projection(tgt))],
            mask=rec['mask'], e=rec.get('e'),
            path_to_first_terminal=Tv.shortest_path(sorted(Tv.terminals)[0]))
    return out


def ablation_table(w):
    out = {}
    nonzero = [c for i, c in enumerate(COEFFS) if w[i]]
    for name in nonzero + ['input_frozen', 'NF']:
        cnt = Counter()
        for x in QUERIES:
            full = explore(x, w)
            if name == 'input_frozen':
                r = explore(x, w, frozen=(I1, I2))
            elif name == 'NF':
                r = explore(x, w, nf=True)
            else:
                r = explore(x, zeroed(w, name))
            cnt['Y'] += r.Y != full.Y
            cnt['A_projection'] += r.projection(A) != full.projection(A)
            cnt['B_projection'] += r.projection(B) != full.projection(B)
            cnt['signatures'] += sigs(r.Y) != sigs(full.Y)
        out[name] = dict(cnt)
    return out


def evaluate_teacher(name, w, runner):
    cond, rows, extra = conditions_1_to_7(w)
    cond['3prime_satellite_feedback_changes_B'] = condition_3_prime(w, runner)
    fwd = chain(w, CHAINS['forward'][name], 'forward', runner)
    rev = chain(w, CHAINS['reverse'][name], 'reverse', runner)
    cond['8_causal_chain_A_input_B'] = {k: v for k, v in fwd.items() if k != 'pairs'}
    q3 = set(cond['3prime_satellite_feedback_changes_B']['query_ids'])
    cond['8_causal_chain_A_input_B']['condition_3prime_queries_supporting_8'] = sorted(
        {t[0] for t in fwd['qualifying']} & q3)
    failed = [k for k, v in cond.items() if not v['passed']]
    return dict(teacher=name, w=dict(zip(COEFFS, w)), family=name[0], conditions=cond,
                gate_passed=not failed, failed_conditions=failed,
                reverse_chain_B_input_A={k: v for k, v in rev.items() if k != 'pairs'},
                link_ablation_table=ablation_table(w), **extra), rows, dict(forward=fwd['pairs'], reverse=rev['pairs'])


def failure_certificates(teachers):
    out = {}
    for name, t in teachers.items():
        out[name] = {}
        for cond in t['failed_conditions']:
            c = t['conditions'][cond]
            out[name][cond] = {k: v for k, v in c.items()
                               if k not in ('passed', 'qualifying', 'example', 'query_ids')}
    return out


def regression(frozen_v1):
    """v1 teachers rerun in the new module; conditions 1-7 must equal the frozen v1 phase-A values."""
    out = {}
    for name, w in REGRESSION.items():
        cond, _, _ = conditions_1_to_7(w)
        old = frozen_v1['teachers'][name]['conditions']
        out[name] = dict(matches_frozen_v1=cond == old,
                         mismatched_conditions=[k for k in cond if cond[k] != old.get(k)])
    return out


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_shared_input_mediation')
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(run=res.testsRun, failures=len(res.failures), errors=len(res.errors), passed=res.wasSuccessful())


SOURCES = ('src/shared_input_mediation.py', 'experiments/shared_input_mediation.py',
           'tests/test_shared_input_mediation.py', CONTRACT, 'src/reciprocal_participation.py',
           'reports/learned-reciprocal-participation-v1-phase-a.json')


def gate(man):
    runner = Runner()
    teachers, per_query, archives = {}, {}, {}
    for name, w in TEACHERS.items():
        teachers[name], per_query[name], archives[name] = evaluate_teacher(name, w, runner)
    v1_report = json.loads((ROOT / 'reports/learned-reciprocal-participation-v1-phase-a.json').read_text())
    passing = [n for n, t in teachers.items() if t['gate_passed']]
    return dict(
        experiment=PREFIX, contract=CONTRACT, contract_sha256_at_freeze=man['contract_sha256'],
        manifest=dict(path=f'reports/{PREFIX}-manifest.json', sha256=man['sha256']),
        status='mechanism_witness_found_for_declared_teachers' if passing else 'chain_refuted_for_declared_teachers',
        passing_teachers=passing,
        family_results={f: {n: t['gate_passed'] for n, t in teachers.items() if n[0] == f} for f in 'SC'},
        regression_controls_v1=regression(v1_report),
        failure_certificates=failure_certificates(teachers),
        teachers=teachers, condition_8_archive=archives, per_query=per_query,
        scoring_runs=len(runner.cache)), runner


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    report, _ = gate(m)
    report['tests'] = run_tests()
    report['provenance'] = dict(
        python=platform.python_version(), implementation=platform.python_implementation(),
        command='.venv/bin/python -m experiments.shared_input_mediation',
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES})
    dump(out / f'{PREFIX}-gate.json', report)
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds_total_including_preprocessing=round(time.perf_counter() - started, 3),
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), git_revision=git('rev-parse', 'HEAD'), dirty_worktree=bool(git('status', '--porcelain'))))
    print(report['status'], report['passing_teachers'],
          {n: t['failed_conditions'] for n, t in report['teachers'].items()},
          {n: r['matches_frozen_v1'] for n, r in report['regression_controls_v1'].items()}, report['tests'])


if __name__ == '__main__':
    main()
