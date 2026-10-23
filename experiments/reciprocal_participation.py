"""Learned reciprocal participation v1: frozen manifest and phase A feasibility gate.

Run `.venv/bin/python -m experiments.reciprocal_participation [--out-dir DIR]`. Phase B (learning)
is conditional on a teacher among T1-T3 passing the phase A gate; none did, so it is not implemented. Scientific reports are deterministic; volatile
timings and git state go to the runtime file. Pass --out-dir to regenerate into a temporary
directory and compare bytes with the frozen reports.
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

from src.reciprocal_participation import (
    A, B, CHECKPOINTS, COEFFS, COORDS, DOMAINS, I1, I2, K, N_HYPOTHESES, QUERIES, SCHEMA_VERSION, SPLIT_SEEDS,
    TEACHERS, all_states, clamp_sets, controls, core_disconnected, explore, hypothesis, influence_witnesses,
    make_split, signature,
)


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'learned-reciprocal-participation-v1'
COMMAND = '.venv/bin/python -m experiments.reciprocal_participation'
SOURCES = ('src/reciprocal_participation.py', 'experiments/reciprocal_participation.py',
           'tests/test_reciprocal_participation.py', 'docs/learned-reciprocal-participation-v1-contract.md')
GATE_TEACHERS = ('T1', 'T2', 'T3')
EXAMPLES = 3


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + '\n')


def manifest():
    body = dict(schema=SCHEMA_VERSION, query_order='(e1, e2, domain_A, domain_B)',
                queries=[[q.e1, q.e2, q.da, q.db] for q in QUERIES], checkpoints=list(CHECKPOINTS),
                splits=[make_split(s) for s in SPLIT_SEEDS])
    return dict(body, sha256=hashlib.sha256(canonical(body).encode()).hexdigest())


def freeze_manifest(out):
    """Write once; refuse to proceed if a frozen manifest exists and regeneration differs."""
    m = manifest()
    path = out / f'{PREFIX}-splits.json'
    if path.exists():
        if json.loads(path.read_text()) != m:
            sys.exit('split manifest differs from the frozen file; refusing to run')
    else:
        dump(path, m)
    return m


# ---------------------------------------------------------------- phase A

def ys(s):
    return sorted(list(p) for p in s)


def sigs(Y):
    return frozenset(signature(a, b) for a, b in Y)


def phase_a_teacher(name, w):
    rows, counts = [], Counter()
    ex = {k: [] for k in ('forward_by_input', 'forward_by_core', 'reverse_on_input', 'core_responds',
                          'input_revision', 'feedback_participation', 'feedback_sign_only',
                          'core_participation', 'core_sign_only', 'clamp_B_to_A_core_mediated',
                          'clamp_A_to_B_core_mediated', 'clamp_B_to_A_not_core_mediated',
                          'clamp_A_to_B_not_core_mediated', 'mixed_sign_joint', 'independent_product_mismatch',
                          'cross_satellite_exclusion')}
    fixed_k_match = {kv: 0 for kv in (-1, 0, 1)}
    unique_signature_queries = {}
    nonempty_signatures = set()
    wd = core_disconnected(w)

    def note(key, item):
        counts[key] += 1
        if len(ex[key]) < EXAMPLES:
            ex[key].append(item)

    for qi, q in enumerate(QUERIES):
        r = explore(q, w)
        c = controls(q, w)
        Y, T = r.Y, r.terminals
        counts['cyclic_graphs'] += not r.acyclic
        counts['domain_violations'] += sum(z[A] not in DOMAINS[q.da] or z[B] not in DOMAINS[q.db] for z in r.reachable)
        counts['evidence_mutations'] += r.query != q
        tsigs = frozenset(signature(t[A], t[B]) for t in T)
        nonempty_signatures |= {s for s in tsigs if s}
        if len(tsigs) == 1 and next(iter(tsigs)):
            unique_signature_queries[qi] = next(iter(tsigs))
        revised = [t for t in T if t[:2] != (q.e1, q.e2)]
        rev_possible, rev_necessary = bool(revised), len(revised) == len(T)
        counts['revision_possible_queries'] += rev_possible
        counts['revision_necessary_queries'] += rev_necessary
        if rev_possible:
            note('input_revision', dict(query=qi, x=q.to_json(), terminal=revised[0],
                                        path=r.shortest_path(revised[0]), revised_at_every_terminal=rev_necessary))
        # 1. state-conditioned influence, both states reachable from this query
        for key, targets, neighbors in (('forward_by_input', (A, B), (I1, I2)), ('forward_by_core', (A, B), (K,)),
                                        ('reverse_on_input', (I1, I2), (A, B)), ('core_responds', (K,), (A, B))):
            found = influence_witnesses(q, w, r.reachable, targets, neighbors)
            counts[key + '_queries'] += bool(found)
            for f in found:
                note(key, dict(query=qi, x=q.to_json(), **f))
        # 3. feedback necessity and 4. core contribution
        for key, other in (('feedback', c['input_frozen']), ('core', c['core_disconnected'])):
            if Y != other:
                part = sigs(Y) != sigs(other)
                note(f'{key}_participation' if part else f'{key}_sign_only',
                     dict(query=qi, x=q.to_json(), full=ys(Y), control=ys(other)))
        for kv in (-1, 0, 1):
            fixed_k_match[kv] += c[f'core_fixed_{kv}'] == Y
        if c['independent_product'] != Y:
            note('independent_product_mismatch', dict(query=qi, x=q.to_json(), full=ys(Y),
                                                      product=ys(c['independent_product'])))
        # 5. mixed-sign joint activity
        if any(a * b == -1 for a, b in Y):
            note('mixed_sign_joint', dict(query=qi, x=q.to_json(), Y=ys(Y)))
        # 6. dynamical exclusions (not local-domain safety)
        a_marg, b_marg = {a for a, _ in Y}, {b for _, b in Y}
        cross = sorted((a, b) for a in a_marg for b in b_marg if (a, b) not in Y)
        if cross:
            note('cross_satellite_exclusion', dict(query=qi, x=q.to_json(), Y=ys(Y), excluded=cross))
        domain_excluded = len(DOMAINS[q.da]) * len(DOMAINS[q.db]) - len(Y)
        counts['domain_pairs_excluded_by_dynamics'] += domain_excluded
        # 7. cross-satellite influence through K (clamp interventions)
        for key, clamped in (('clamp_B_to_A', B), ('clamp_A_to_B', A)):
            full_sets, cut_sets = clamp_sets(q, w, clamped), clamp_sets(q, wd, clamped)
            vals = sorted(full_sets)
            for i, v in enumerate(vals):
                for v2 in vals[i + 1:]:
                    if full_sets[v] != full_sets[v2]:
                        mediated = cut_sets[v] == cut_sets[v2]
                        note(f'{key}_core_mediated' if mediated else f'{key}_not_core_mediated',
                             dict(query=qi, x=q.to_json(), clamp_values=[v, v2],
                                  sets=[sorted(full_sets[v]), sorted(full_sets[v2])],
                                  core_removed_sets=[sorted(cut_sets[v]), sorted(cut_sets[v2])]))
        rows.append(dict(
            query=qi, x=q.to_json(), Y=ys(Y), terminals=sorted(list(t) for t in T),
            signatures=sorted(tsigs), terminal_multiplicity=len(T),
            possible_active=[s for s in 'AB' if any(s in g for g in tsigs)],
            necessary_active=[s for s in 'AB' if all(s in g for g in tsigs)],
            shortest_path_lengths=sorted(len(r.shortest_path(t)) for t in T), longest_path=r.longest_path,
            reachable=len(r.reachable), edges=r.edges, inspections=r.inspections,
            input_revision=dict(possible=rev_possible, necessary=rev_necessary),
            controls={k: ys(v) for k, v in c.items()}))
    # absence outside co-reachable pairs is checked over all allowed states (does not count for the gate)
    if not counts['forward_by_input_queries'] + counts['forward_by_core_queries'] or not counts['reverse_on_input_queries']:
        counts['influence_in_unrestricted_state_space'] = sum(
            bool(influence_witnesses(q, w, all_states(q), (A, B), (I1, I2, K)))
            + bool(influence_witnesses(q, w, all_states(q), (I1, I2), (A, B))) for q in QUERIES)
    distinct_unique = sorted(set(unique_signature_queries.values()))
    differing_controls = {k: sum(row['controls'][k] != row['Y'] for row in rows) for k in rows[0]['controls']}
    participation_controls = {k: sum(sorted({signature(a, b) for a, b in row['controls'][k]})
                                     != sorted({signature(a, b) for a, b in row['Y']}) for row in rows)
                              for k in rows[0]['controls']}
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
            passed=differing_controls['input_frozen'] > 0,
            queries_Y_differs_from_input_frozen=differing_controls['input_frozen'],
            participation_changes=counts['feedback_participation'], sign_only_changes=counts['feedback_sign_only']),
        '4_core_contribution': dict(
            passed=differing_controls['core_disconnected'] > 0 and participation_controls['core_disconnected'] > 0
            and all(m < len(QUERIES) for m in fixed_k_match.values()),
            queries_Y_differs_from_core_disconnected=differing_controls['core_disconnected'],
            queries_participation_differs=participation_controls['core_disconnected'],
            participation_changes=counts['core_participation'], sign_only_changes=counts['core_sign_only'],
            fixed_k_queries_reproduced={str(k): v for k, v in fixed_k_match.items()},
            some_fixed_k_reproduces_all=any(m == len(QUERIES) for m in fixed_k_match.values())),
        '5_selective_participation': dict(
            passed=len(nonempty_signatures) >= 2 and len(distinct_unique) >= 2,
            distinct_nonempty_signatures=sorted(nonempty_signatures),
            queries_with_unique_nonempty_signature=len(unique_signature_queries),
            distinct_unique_nonempty_signatures=distinct_unique,
            unique_signature_examples={s: [q for q, g in sorted(unique_signature_queries.items()) if g == s][:EXAMPLES]
                                       for s in distinct_unique},
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
    summary = dict(
        teacher=name, w=dict(zip(COEFFS, w)), conditions=cond,
        gate_passed=all(v['passed'] for v in cond.values()),
        failed_conditions=[k for k, v in cond.items() if not v['passed']],
        cyclic_graphs=counts['cyclic_graphs'],
        controls_queries_with_different_Y=differing_controls,
        controls_queries_with_different_participation=participation_controls,
        input_feedback_removed_equals_input_frozen=differing_controls['input_feedback_removed']
        == 0 and all(row['controls']['input_feedback_removed'] == row['controls']['input_frozen'] for row in rows),
        independent_product_mismatches=counts['independent_product_mismatch'],
        terminal_multiplicity=dict(Counter(row['terminal_multiplicity'] for row in rows)),
        response_set_sizes=dict(Counter(len(row['Y']) for row in rows)),
        work=dict(reachable_states=sum(r['reachable'] for r in rows), edges=sum(r['edges'] for r in rows),
                  inspections=sum(r['inspections'] for r in rows)),
        witnesses=ex, witness_counts={k: counts[k] for k in ex},
    )
    if 'influence_in_unrestricted_state_space' in counts:
        summary['influence_in_unrestricted_state_space'] = counts['influence_in_unrestricted_state_space']
    for k in summary['terminal_multiplicity'].copy():
        summary['terminal_multiplicity'][str(k)] = summary['terminal_multiplicity'].pop(k)
    for k in summary['response_set_sizes'].copy():
        summary['response_set_sizes'][str(k)] = summary['response_set_sizes'].pop(k)
    return summary, rows


def phase_a():
    teachers, per_query = {}, {}
    for name, w in TEACHERS.items():
        teachers[name], per_query[name] = phase_a_teacher(name, w)
    qualifying = [t for t in GATE_TEACHERS if teachers[t]['gate_passed']]
    return dict(teachers=teachers, qualifying_teachers=qualifying,
                status='mechanism_feasible' if qualifying else 'mechanism_refuted_in_declared_family',
                per_query=per_query)


def failure_certificates(teachers, per_query):
    """Exhaustive statements for every failed condition of every teacher."""
    out = {}
    for name, t in teachers.items():
        cert = {}
        for cond in t['failed_conditions']:
            c = t['conditions'][cond]
            if cond == '3_feedback_necessity':
                same = sum(row['Y'] == row['controls']['input_frozen'] for row in per_query[name])
                cert[cond] = (f'Y equals the input-frozen response set on {same} of {len(QUERIES)} queries; '
                              f'input revision occurs on {t["conditions"]["2_realized_input_revision"]["queries_revision_possible"]}')
            elif cond == '4_core_contribution':
                cert[cond] = (f'Y differs from core-disconnected on {c["queries_Y_differs_from_core_disconnected"]} queries, '
                              f'participation on {c["queries_participation_differs"]}; fixed k reproduces '
                              f'{c["fixed_k_queries_reproduced"]} of {len(QUERIES)} queries')
            else:
                cert[cond] = {k: v for k, v in c.items() if k != 'passed'}
        out[name] = cert
    return out


def single_satellite_inputs(w):
    """Each input coordinate couples to at most one satellite."""
    return (w[0] == 0 or w[2] == 0) and (w[1] == 0 or w[3] == 0)


def post_gate_diagnostic():
    """Scope of the condition-3 failure over the whole coefficient family. Not a teacher search:
    no hypothesis here is evaluated against the gate or used for phase B."""
    feedback, single, single_feedback = 0, 0, 0
    for h in range(N_HYPOTHESES):
        w = hypothesis(h)
        d = any(explore(q, w).Y != explore(q, w, frozen=(I1, I2)).Y for q in QUERIES)
        feedback += d
        if single_satellite_inputs(w):
            single += 1
            single_feedback += d
    return dict(hypotheses=N_HYPOTHESES, with_feedback_effect_on_some_query=feedback,
                each_input_coordinate_coupled_to_at_most_one_satellite=single,
                of_which_with_feedback_effect=single_feedback,
                declared_teachers_with_that_property=[k for k, w in TEACHERS.items() if single_satellite_inputs(w)],
                note='post-gate scope diagnostic; not a teacher search, no gate evaluation, no phase B')


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_reciprocal_participation')
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(run=res.testsRun, failures=len(res.failures), errors=len(res.errors), passed=res.wasSuccessful())


def provenance():
    return dict(python=platform.python_version(), implementation=platform.python_implementation(), command=COMMAND,
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES})


def main(argv=None):
    start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    man = freeze_manifest(out)
    t0 = time.perf_counter()
    a = phase_a()
    phase_a_seconds = time.perf_counter() - t0
    certs = failure_certificates(a['teachers'], a['per_query'])
    diagnostic = post_gate_diagnostic() if not a['qualifying_teachers'] else None
    tests = run_tests()
    report_a = dict(experiment=SCHEMA_VERSION, phase='A', contract='docs/learned-reciprocal-participation-v1-contract.md',
                    status=a['status'], qualifying_teachers=a['qualifying_teachers'],
                    gate_rule='one teacher among T1-T3 must pass conditions 1-7',
                    split_manifest=dict(path=f'reports/{PREFIX}-splits.json', sha256=man['sha256']),
                    tests=tests, schema=dict(coords=list(COORDS), domains=[list(d) for d in DOMAINS],
                                             coefficients=list(COEFFS), values=[-3, 0, 3]),
                    phase_b='not run: no teacher among T1-T3 passed the gate (contract section 5 stop rule)'
                    if not a['qualifying_teachers'] else 'gate passed',
                    failure_certificates=certs, post_gate_structural_diagnostic=diagnostic,
                    teachers=a['teachers'], per_query=a['per_query'], provenance=provenance())
    dump(out / f'{PREFIX}-phase-a.json', report_a)
    timings = dict(phase_a=round(phase_a_seconds, 3))
    if a['qualifying_teachers']:
        # Contract section 7 is implemented only after a passing gate; this gate failed.
        sys.exit('phase A gate passed: phase B must be implemented before continuing')
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    timings['total_including_preprocessing'] = round(time.perf_counter() - start, 3)
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds=timings, python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        command=COMMAND, out_dir=str(out), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(a['status'], a['qualifying_teachers'],
          {t: v['failed_conditions'] for t, v in a['teachers'].items()}, tests, timings)


if __name__ == '__main__':
    main()
