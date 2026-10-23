"""Core abstraction v1: Stage A representability ceiling and Stage B information ceiling.

Run `.venv/bin/python -m experiments.core_abstraction [--out-dir DIR] [--workers N] [--manifest-only]`.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

from experiments.mechanism_learning import NEGATIVE, POSITIVE, TEACHERS, canonical, completion_status, dump, sha
from src.core_abstraction import (
    C0, FULL, INPUT_FROZEN, K_CLAMPS, NF, OUTCOMES, PAIRS, Responses, acquire, assemble, assess, assignment,
    blocks, index, mirror_core, parts, potential, potential_closed_form, sign_class, wake_targets, waking_vector,
    witness_controls,
)
from src.mechanism_learning import pair_mask
from src.shared_input_mediation import N_VECTORS, QUERIES, TERNARY, vector


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'core-abstraction-v1'
CONTRACT = 'docs/core-abstraction-v1-contract.md'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
RULES = [
    'dynamics: src/shared_input_mediation.explore unchanged; full reachable-terminal sets; Y projected to (a,b)',
    'coefficients (w_1A,w_2A | w_1B,w_2B | w_KA,w_KB | w_12) = (p_A | p_B | c | r); only c varies in Stage A',
    'waking vectors: A (p,0,0,0,0,r), B (0,0,p,0,0,r); waking target = teacher block in that vector, own marginal',
    'assignment: ascending training IDs, even positions -> A (8), odd -> B (7); swapped condition exchanges them',
    'filter: nine-bit mask per satellite over lexicographic blocks (-3<0<3); keep blocks reproducing the marginal; '
    'run in each of the five frozen orders restricted to the assignment; masks must agree',
    'assemblies ordered by (index p_A, index p_B); cores in the same lexicographic order; c0 = (0,0)',
    'predicates R, P, X, core-dependent and route-specific witnesses, outcomes: contract section 5.1',
    'witness cuts: (0,c_B) and (c_A,0); route: input clamped to evidence, learned vs c0 recipient marginal differ',
    'possible: some surviving assembly qualifies; robust: every surviving assembly (including interfering) qualifies',
    'Stage B: classes by identical (V_A,V_B,r) over all 2,187 vectors per split/assignment; members scored by '
    'assess(detail=False) against their own targets; compatible core = intersection over members with a '
    'qualifying core on that assembly',
    'response encoding in reports: 9-bit pair masks over 3(a+1)+(b+1)',
]


def manifest():
    splits = json.loads((ROOT / SPLITS).read_text())['splits']
    runs = {f'{s["seed"]}/{"swapped" if sw else "base"}': dict(zip(('A', 'B'), assignment(s['train'], sw)))
            for s in splits for sw in (False, True)}
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                specificity_controls=list(NEGATIVE), r={k: v[6] for k, v in TEACHERS.items()},
                candidate_order=[list(p) for p in PAIRS], assignments=runs,
                modes=[FULL, INPUT_FROZEN, NF] + [list(k) for k in K_CLAMPS], outcomes=list(OUTCOMES), rules=RULES,
                prior_artifacts={p: sha(p) for p in [
                    SPLITS, 'src/shared_input_mediation.py', 'src/reciprocal_participation.py',
                    'tests/test_shared_input_mediation.py', 'src/mechanism_learning.py',
                    'experiments/mechanism_learning.py', 'reports/shared-input-mediation-v1-manifest.json',
                    'reports/learned-input-mediation-v1-manifest.json', 'reports/local-update-diagnosis-v1-manifest.json',
                    'docs/local-update-diagnosis-v1-results.md']})
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


def evaluation_run(split, r, wake):
    return dict(train=split['train'], mixed=split['mixed_evidence'], within=split['within_context'], wake=wake, r=r)


def acquire_all_orders(resp, split, ids_a, r, wake):
    """Filter in every frozen order; returns (common masks or None, evaluations, repetition_idempotent)."""
    masks, evals, idem = set(), 0, True
    for order in split['orders']:
        fa, fb = acquire(resp, r, order, ids_a, wake)
        before = (fa.mask, fb.mask)
        for qi in order:                                     # ordinary repetition: one further pass
            sat = 'A' if qi in ids_a else 'B'
            (fa if sat == 'A' else fb).update(qi, wake[(sat, qi)])
        idem &= (fa.mask, fb.mask) == before
        masks.add(before)
        evals += fa.evaluations + fb.evaluations
    return (masks.pop() if len(masks) == 1 else None), evals, idem


# ---------------------------------------------------------------- Stage A

def stage_a_run(resp, name, split, swapped):
    w = TEACHERS[name]
    pa, pb, ct, r = parts(w)
    ids_a, ids_b = assignment(split['train'], swapped)
    wake = wake_targets(resp, w, ids_a, ids_b)
    masks, evals, idem = acquire_all_orders(resp, split, ids_a, r, wake)
    if masks is None:
        return dict(teacher=name, split=split['seed'], swapped=swapped, order_independent=False)
    va, vb = blocks(masks[0]), blocks(masks[1])
    original = all(resp.marginal(qi, waking_vector(s, p, r), s) == wake[(s, qi)]
                   for s, vs in (('A', va), ('B', vb)) for p in vs for (s2, qi) in wake if s2 == s)
    run = evaluation_run(split, r, wake)
    assemblies = []
    for p_a in va:
        for p_b in vb:
            rec = assess(resp, run, w, p_a, p_b)
            ctrl = []
            for key, tier in (('qualifying', 'route_specific_witnesses'), ('route_unresolved', 'core_dependent_witnesses')):
                for c in rec[key]:
                    for s, qi in rec['cores'][f'{c[0]},{c[1]}'][tier]:
                        ctrl.append(dict(core=c, tier=key, satellite=s, query=qi,
                                         controls=witness_controls(resp, run, w, p_a, p_b, tuple(c), s, qi)))
            true = (p_a, p_b) == (pa, pb)
            rec.update(p_A=list(p_a), p_B=list(p_b), true_assembly=true, witness_controls=ctrl,
                       qualifying_in_teacher_sign_class=[c for c in rec['qualifying']
                                                         if sign_class(tuple(c)) == sign_class(ct)],
                       qualifying_without_within_regression=[
                           c for c in rec['qualifying']
                           if not any(rec['cores'][f'{c[0]},{c[1]}']['satellites'][s]['within']['losses']
                                      for s in ('A', 'B'))])
            assemblies.append(rec)
    out = Counter(a['outcome'] for a in assemblies)
    beyond = [a for a in assemblies if a['qualifying'] and (not a['true_assembly']
              or len(a['qualifying_in_teacher_sign_class']) < len(a['qualifying']))]
    true_rec = next(a for a in assemblies if a['true_assembly'])
    return dict(
        teacher=name, split=split['seed'], swapped=swapped, A_queries=ids_a, B_queries=ids_b,
        order_independent=True, repetition_idempotent=idem, original_context_retention=original,
        V_A=[list(p) for p in va], V_B=[list(p) for p in vb], teacher_blocks_survive=pa in va and pb in vb,
        acquisition_evaluations=evals, n_assemblies=len(assemblies),
        outcomes={o: out[o] for o in OUTCOMES},
        no_preserving_core=sum(a['no_preserving_core'] for a in assemblies),
        possible=any(a['qualifying'] for a in assemblies), robust=all(a['qualifying'] for a in assemblies),
        possible_route_unresolved=any(a['route_unresolved'] for a in assemblies),
        true_assembly=dict(outcome=true_rec['outcome'], qualifying=true_rec['qualifying'],
                           in_teacher_sign_class=true_rec['qualifying_in_teacher_sign_class']),
        qualifying_beyond_teacher_core_recovery=len(beyond),
        memorization=dict(stored_entries=len(wake), answers_on_assembled_queries=0),
        assemblies=assemblies)


def stage_a_summary(runs):
    out = {}
    for name in TEACHERS:
        R = [r for r in runs if r['teacher'] == name]
        fixed = {}
        for c in PAIRS:
            key = f'{c[0]},{c[1]}'
            hits = [[c[0], c[1]] in a['qualifying'] for r in R for a in r['assemblies']]
            fixed[key] = dict(
                runs_some_assembly=sum(any([c[0], c[1]] in a['qualifying'] for a in r['assemblies']) for r in R),
                runs_true_assembly=sum(any([c[0], c[1]] in a['qualifying'] for a in r['assemblies'] if a['true_assembly'])
                                       for r in R),
                runs_every_assembly=sum(all([c[0], c[1]] in a['qualifying'] for a in r['assemblies']) for r in R),
                assemblies_qualifying=sum(hits), assemblies=len(hits))
        out[name] = dict(
            runs=len(R), possible=sum(r['possible'] for r in R), robust=sum(r['robust'] for r in R),
            possible_route_unresolved=sum(r['possible_route_unresolved'] for r in R),
            outcomes={o: sum(r['outcomes'][o] for r in R) for o in OUTCOMES},
            assemblies=sum(r['n_assemblies'] for r in R),
            assemblies_no_preserving_core=sum(r['no_preserving_core'] for r in R),
            true_assembly_outcomes=dict(Counter(r['true_assembly']['outcome'] for r in R)),
            runs_qualifying_beyond_teacher_core_recovery=sum(bool(r['qualifying_beyond_teacher_core_recovery'])
                                                             for r in R),
            private_mask_sizes=dict(Counter(f'{len(r["V_A"])}x{len(r["V_B"])}' for r in R)),
            fixed_cores=fixed, per_run={f'{r["split"]}/{"swapped" if r["swapped"] else "base"}':
                                        dict(possible=r['possible'], robust=r['robust'], outcomes=r['outcomes'])
                                        for r in R})
        out[name]['existence'] = out[name]['possible'] > 0
    out['existence_for_both_primary'] = all(out[n]['existence'] for n in POSITIVE)
    return out


# ---------------------------------------------------------------- Stage B

def stage_b_run(args):
    split, swapped = args
    t0 = time.perf_counter()
    resp = Responses()
    ids_a, ids_b = assignment(split['train'], swapped)
    order = split['orders'][0]
    state = {}
    for h in range(N_VECTORS):
        w = vector(h)
        r = w[6]
        fa, fb = acquire(resp, r, order, ids_a, wake_targets(resp, w, ids_a, ids_b))
        state[h] = (fa.mask, fb.mask, r)
    groups = defaultdict(list)
    for h, s in state.items():
        groups[s].append(h)
    structure_ok = True
    classes, census = [], Counter()
    teacher_idx = {index(w): n for n, w in TEACHERS.items()}
    assess_calls = 0
    for (ma, mb, r), members in sorted(groups.items()):
        va, vb = blocks(ma), blocks(mb)
        expect = sorted(index(assemble(p, q, c, r)) for p in va for q in vb for c in PAIRS)
        structure_ok &= expect == sorted(members)
        census[f'{len(va)}x{len(vb)}'] += 1
        rep = vector(members[0])
        run = evaluation_run(split, r, wake_targets(resp, rep, ids_a, ids_b))
        qual = {}
        for h in members:
            for p in va:
                for q in vb:
                    qual[(h, p, q)] = assess(resp, run, vector(h), p, q, detail=False)
                    assess_calls += 1
        per_assembly = []
        for p in va:
            for q in vb:
                sets = {h: frozenset(map(tuple, qual[(h, p, q)]['qualifying'])) for h in members}
                withq = [h for h in members if sets[h]]
                inter = frozenset.intersection(*(sets[h] for h in withq)) if withq else frozenset()
                cert = None
                for i, h1 in enumerate(withq):
                    for h2 in withq[i + 1:]:
                        if not sets[h1] & sets[h2]:
                            same_private = vector(h1)[:4] == vector(h2)[:4]
                            cert = dict(members=[h1, h2], qualifying=[sorted(sets[h1]), sorted(sets[h2])],
                                        same_private_blocks=same_private)
                            break
                    if cert:
                        break
                per_assembly.append(dict(p_A=list(p), p_B=list(q),
                                         interference=qual[(members[0], p, q)]['outcome'] == 'assembly_interference',
                                         members_with_qualifying=len(withq),
                                         compatible_cores=sorted(map(list, inter)) if withq else None,
                                         incompatibility_certificate=cert))
        rec = dict(V_A=[list(p) for p in va], V_B=[list(q) for q in vb], r=r, members=len(members),
                   members_with_any_qualifying=sum(any(qual[(h, p, q)]['qualifying'] for p in va for q in vb)
                                                   for h in members),
                   assemblies=per_assembly)
        named = [teacher_idx[h] for h in members if h in teacher_idx]
        if named:
            rec['teachers'] = named
            mats = []
            for n in named:
                tpa, tpb, _, _ = parts(TEACHERS[n])
                for p in va:
                    for q in vb:
                        mat = {}
                        for c in PAIRS:
                            res = qual[(index(assemble(tpa, tpb, c, r)), p, q)]
                            mat[f'{c[0]},{c[1]}'] = ''.join(
                                'Q' if [d[0], d[1]] in res['qualifying'] else 'U' if [d[0], d[1]] in res['route_unresolved']
                                else '-' for d in PAIRS)
                        mats.append(dict(teacher=n, p_A=list(p), p_B=list(q), true_assembly=(p, q) == (tpa, tpb),
                                         member_core_rows=mat))
            rec['teacher_core_matrices'] = mats
        classes.append(rec)
    relevant = [a for c in classes for a in c['assemblies'] if a['members_with_qualifying']]
    summary = dict(
        classes=len(classes), class_census=dict(sorted(census.items())),
        class_structure_verified=structure_ok,
        classes_with_any_qualifying_member=sum(c['members_with_any_qualifying'] > 0 for c in classes),
        assemblies_with_qualifying_members=len(relevant),
        assemblies_with_compatible_core=sum(bool(a['compatible_cores']) for a in relevant),
        incompatibility_certificates=sum(a['incompatibility_certificate'] is not None for a in relevant),
        certificates_same_private_blocks=sum(bool(a['incompatibility_certificate']
                                                  and a['incompatibility_certificate']['same_private_blocks'])
                                             for a in relevant))
    return dict(split=split['seed'], swapped=swapped, summary=summary,
                teacher_classes=[c for c in classes if 'teachers' in c],
                classes=classes, work=dict(assess_calls=assess_calls, simulation_requests=resp.requests,
                                          unique_simulations=len(resp.memo)),
                seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- gates

def oracle_gate(resp):
    from tests.test_core_abstraction import oracle_Y
    bad = [k for k in sorted(resp.memo, key=repr) if resp.memo[k] != oracle_Y(*k)]
    return dict(entries=len(resp.memo), mismatches=len(bad), first=[repr(k) for k in bad[:5]], passed=not bad)


def symmetry_gate():
    resp = Responses()
    bad = 0
    for h in range(N_VECTORS):
        w = vector(h)
        m = assemble(*parts(w)[:2], mirror_core(parts(w)[2]), w[6])
        for qi in range(len(QUERIES)):
            bad += resp.Y(qi, w) != resp.Y(qi, m)
    return dict(cases=N_VECTORS * len(QUERIES), mismatches=bad, passed=bad == 0), resp


def potential_gate():
    bad = sum(potential(c, a, b) != potential_closed_form(c, a, b) for c in PAIRS for a in TERNARY for b in TERNARY)
    return dict(cases=81, mismatches=bad, passed=bad == 0)


def run_tests():
    names = ['tests.test_core_abstraction']
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    with open('/dev/null', 'w') as sink:
        res = unittest.TextTestRunner(stream=sink, verbosity=0).run(suite)
    return dict(modules=names, run=res.testsRun, failures=len(res.failures), errors=len(res.errors),
                passed=res.wasSuccessful())


def main(argv=None):
    started = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(ROOT / 'reports'))
    ap.add_argument('--manifest-only', action='store_true')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    timing = {}
    gate = dict(prior_artifacts=all(sha(p) == h for p, h in m['prior_artifacts'].items()),
                contract=sha(CONTRACT) == m['contract_sha256'])
    if not all(gate.values()):
        sys.exit(f'prior artifacts or contract differ: {gate}')
    splits = json.loads((ROOT / SPLITS).read_text())['splits']

    t0 = time.perf_counter()
    resp_a = Responses()
    runs = [stage_a_run(resp_a, n, s, sw) for n in TEACHERS for s in splits for sw in (False, True)]
    timing['stage_a'] = time.perf_counter() - t0
    gate['order_independent'] = all(r['order_independent'] for r in runs)
    gate['teacher_blocks_survive'] = all(r.get('teacher_blocks_survive') for r in runs)
    gate['original_context_retention'] = all(r.get('original_context_retention') for r in runs)
    gate['repetition_idempotent'] = all(r.get('repetition_idempotent') for r in runs)

    t0 = time.perf_counter()
    og = oracle_gate(resp_a)
    sg, resp_sym = symmetry_gate()
    pg = potential_gate()
    timing['gates'] = time.perf_counter() - t0
    gate.update(oracle=og['passed'], core_sign_symmetry=sg['passed'], core_potential=pg['passed'])

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        info = pool.map(stage_b_run, [(s, sw) for s in splits for sw in (False, True)])
    timing['stage_b'] = time.perf_counter() - t0
    timing['stage_b_per_run'] = [round(i.pop('seconds'), 3) for i in info]
    gate['class_structure'] = all(i['summary']['class_structure_verified'] for i in info)
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
    tests = run_tests()
    completion = completion_status(gate, tests)

    table = {str(h): [pair_mask(resp_sym.Y(qi, vector(h))) for qi in range(len(QUERIES))] for h in range(N_VECTORS)}
    work = dict(
        acquisition=dict(filter_evaluations=sum(r.get('acquisition_evaluations', 0) for r in runs),
                         note='summed over all five orders and the repetition pass, per run'),
        ceiling_search=dict(assemblies=sum(r.get('n_assemblies', 0) for r in runs),
                            core_records=9 * sum(r.get('n_assemblies', 0) for r in runs)),
        evaluation=dict(simulation_requests=resp_a.requests, unique_simulations=len(resp_a.memo),
                        by_mode=dict(Counter(k[2] if isinstance(k[2], str) else f'k={k[2][1]}' for k in resp_a.memo))),
        stage_b={f'{i["split"]}/{"swapped" if i["swapped"] else "base"}': i['work'] for i in info},
        storage=dict(private_candidate_mask_bits=18, executable_coefficients_per_assembly=7,
                     learned_core_coefficients=2, temporary_search_state_per_assembly='9 core records',
                     external_comparator_memory_entries_per_run=15))
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'],
                code={p: sha(p) for p in ('src/core_abstraction.py', 'experiments/core_abstraction.py',
                                          'tests/test_core_abstraction.py')})
    dump(out / f'{PREFIX}-ceiling.json', dict(
        experiment=PREFIX, stage='A', completion=completion, correctness_gate=gate, gate_details=dict(
            oracle=og, core_sign_symmetry=sg, core_potential=pg), tests=tests,
        summary=stage_a_summary([r for r in runs if r['order_independent']]), work=work, provenance=prov,
        runs=runs, response_table_full=table))
    dump(out / f'{PREFIX}-information.json', dict(
        experiment=PREFIX, stage='B', completion=completion, correctness_gate=gate, provenance=prov,
        summary={f'{i["split"]}/{"swapped" if i["swapped"] else "base"}': i['summary'] for i in info},
        teacher_classes={f'{i["split"]}/{"swapped" if i["swapped"] else "base"}': i['teacher_classes'] for i in info},
        classes={f'{i["split"]}/{"swapped" if i["swapped"] else "base"}': i['classes'] for i in info}))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: (round(v, 3) if isinstance(v, float) else v) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3), workers=args.workers,
        peak_rss_kib=dict(main=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          largest_child=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
