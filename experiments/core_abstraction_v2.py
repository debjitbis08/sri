"""Core abstraction v2: context-consistent feasibility (Stage A) and the connected-evidence selector (Stage B).

Run `.venv/bin/python -m experiments.core_abstraction_v2 [--out-dir DIR] [--manifest-only]`.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

from experiments.core_abstraction import acquire_all_orders, evaluation_run, oracle_gate
from experiments.mechanism_learning import NEGATIVE, POSITIVE, TEACHERS, canonical, completion_status, dump, sha
from src.core_abstraction import (
    C0, FULL_MASK, PAIRS, Responses, acquire, assignment, blocks, parts, wake_targets, waking_vector,
)
from src.core_abstraction_v2 import (
    CORE_ORDER, LABELS, acquire_core_masks, connected_targets, consolidate, cores, evaluate, executable_branch,
    in_mask, sign_classes, witness_checks,
)
from src.shared_input_mediation import QUERIES


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'core-abstraction-v2'
CONTRACT = 'docs/core-abstraction-v2-contract.md'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
V1_CEILING = 'reports/core-abstraction-v1-ceiling.json'
ARMS = ('I', 'IR', 'C')
RULES = [
    'dynamics, responses, isolated waking filters and assemblies: v1 src/core_abstraction.py unchanged',
    'isolated acquisition: v1 masks over the five frozen orders; must equal the archived v1 V_A, V_B',
    'IR: re-present the 15 isolated observations once after the first order; state must equal I; core masks full',
    'C: per branch in V_A x V_B, nine-bit core masks C_A, C_B filtered by the addressed satellite marginal under '
    'the full teacher on its assigned training queries; masks must agree over the five frozen orders',
    'branches ordered (index p_A, index p_B); executable branch = first with nonempty C_A & C_B, else p* '
    '(inconsistent_private_assembly); I and IR therefore use p*',
    'installation: first core of C_A & C_B in minimal-change order ' + ' '.join(f'{a},{b}' for a, b in CORE_ORDER),
    'predicates: contract sections 6.1 and 7.1; transfer requires no mixed regression in either satellite',
    'executable success: arm C selector pass on the teacher private blocks; other branches are compensating',
    'response encoding in reports: marginals as sorted lists; mask bits in v1 PAIRS order',
]


def manifest():
    splits = json.loads((ROOT / SPLITS).read_text())['splits']
    runs = {f'{s["seed"]}/{"swapped" if sw else "base"}': dict(zip(('A', 'B'), assignment(s['train'], sw)))
            for s in splits for sw in (False, True)}
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                specificity_controls=list(NEGATIVE), arms=list(ARMS), assignments=runs,
                block_order=[list(p) for p in PAIRS], core_order=[list(c) for c in CORE_ORDER],
                labels=list(LABELS), rules=RULES,
                prior_artifacts={p: sha(p) for p in [
                    SPLITS, 'src/shared_input_mediation.py', 'tests/test_shared_input_mediation.py',
                    'src/core_abstraction.py', 'experiments/core_abstraction.py', 'tests/test_core_abstraction.py',
                    'docs/core-abstraction-v1-contract.md', 'docs/core-abstraction-v1-results.md',
                    'reports/core-abstraction-v1-manifest.json', V1_CEILING,
                    'reports/core-abstraction-v1-information.json']})
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


def key(c):
    return f'{c[0]},{c[1]}'


def labels_for(ev, masks, p, c, true_p, inconsistent, arm, witnesses):
    out = set()
    if inconsistent:
        out.add('inconsistent_private_assembly')
    if p != true_p:
        out.add('wrong_private_branch')
    if len(sign_classes(masks[0] & masks[1])) > 1:
        out.add('core_ambiguity')
    if c == C0:
        out.add('selected_c0')
    if not ev['preservation_nonvacuous']:
        out.add('preservation_vacuous')
    if not ev['preserved']:
        out.add('preservation_lost')
    if arm == 'C' and not in_mask(masks[0] & masks[1], c):
        out.add('connected_acquisition_failed')
    if not ev['transfer']:
        out.add('no_transfer')
    if ev['mixed_regression']:
        out.add('mixed_regression')
    if not ev['core_dependent_gains']:
        out.add('no_core_dependent_gain')
    if ev['core_dependent_gains'] and not any(w['donor_control'] for w in witnesses):
        out.add('donor_control_failed')
    if ev['core_dependent_gains'] and not any(w['comparator_control'] for w in witnesses):
        out.add('comparator_control_failed')
    return sorted(out, key=LABELS.index)


def score(resp, run, teacher, p, c, masks, arm, true_p, inconsistent, memory):
    """Selector record for one installed (branch, core), with witnesses and labels."""
    ev = evaluate(resp, run, teacher, p, c)
    comparators = {'I': consolidate((FULL_MASK, FULL_MASK)), 'IR': consolidate((FULL_MASK, FULL_MASK))}
    wit = [witness_checks(resp, run, teacher, p, c, g, masks, comparators, memory) for g in ev['core_dependent_gains']]
    route = set(ev['route_gains'])
    for w in wit:
        w['v1_route'] = (w['satellite'], w['query']) in route
        w['witness'] = w['donor_control'] and w['comparator_control']
    ca = arm != 'C' or in_mask(masks[0] & masks[1], c)
    base = ev['preservation_nonvacuous'] and ev['preserved'] and ca and ev['transfer']
    return dict(branch=[list(p[0]), list(p[1])], core=list(c), true_branch=p == true_p,
                inconsistent_private_assembly=inconsistent, connected_acquisition=ca,
                passed=base and any(w['witness'] for w in wit),
                passed_v1_route=base and any(w['witness'] and w['v1_route'] for w in wit),
                witnesses=wit, labels=labels_for(ev, masks, p, c, true_p, inconsistent, arm, wit), evaluation=ev)


def roundtrip(vec, resp):
    """Serialize the installed vector alone, reload into a fresh simulator, compare all 81 response sets."""
    w = tuple(json.loads(json.dumps(list(vec))))
    fresh = Responses()
    return all(fresh.Y(qi, w) == resp.Y(qi, vec) for qi in range(len(QUERIES)))


def run_one(resp, name, split, swapped, v1_run):
    w = TEACHERS[name]
    pa, pb, ct, r = parts(w)
    true_p = (pa, pb)
    ids_a, ids_b = assignment(split['train'], swapped)
    wake = wake_targets(resp, w, ids_a, ids_b)
    pmasks, iso_evals, idem = acquire_all_orders(resp, split, ids_a, r, wake)
    checks = dict(private_order_independent=pmasks is not None)
    va, vb = blocks(pmasks[0]), blocks(pmasks[1])
    checks['v1_masks_reproduced'] = ([list(x) for x in va], [list(x) for x in vb]) == (v1_run['V_A'], v1_run['V_B'])
    fa, fb = acquire(resp, r, split['orders'][0], ids_a, wake)                  # IR: one repeated pass
    for qi in split['orders'][0]:
        sat = 'A' if qi in ids_a else 'B'
        (fa if sat == 'A' else fb).update(qi, wake[(sat, qi)])
    checks['IR_equals_I'] = idem and (fa.mask, fb.mask) == pmasks
    branches = [(x, y) for x in va for y in vb]
    obs = connected_targets(resp, w, ids_a, ids_b)
    cm_orders, core_evals = [], 0
    for order in split['orders']:
        cm, ev = acquire_core_masks(resp, branches, r, order, ids_a, obs)
        cm_orders.append(cm)
        core_evals += ev
    checks['core_masks_order_independent'] = all(cm == cm_orders[0] for cm in cm_orders)
    cmasks = cm_orders[0]
    checks['teacher_core_in_true_branch_mask'] = in_mask(cmasks[true_p][0] & cmasks[true_p][1], ct)
    full = {p: (FULL_MASK, FULL_MASK) for p in branches}
    run = evaluation_run(split, r, wake)
    memory = set(wake) | set(obs)

    # Stage A: all branches x all cores (evaluator-only)
    feas = []
    for p in branches:
        cab = cmasks[p][0] & cmasks[p][1]
        recs = {}
        for c in PAIRS:
            ev = evaluate(resp, run, w, p, c)
            recs[key(c)] = dict(connected_acquisition=in_mask(cab, c), **ev)
        feas.append(dict(branch=[list(p[0]), list(p[1])], true_branch=p == true_p,
                         feasible=[list(c) for c in PAIRS if recs[key(c)]['feasible']],
                         feasible_with_CA=[list(c) for c in PAIRS if recs[key(c)]['feasible']
                                           and recs[key(c)]['connected_acquisition']],
                         feasible_route=[list(c) for c in PAIRS if recs[key(c)]['feasible_route']],
                         feasible_route_with_CA=[list(c) for c in PAIRS if recs[key(c)]['feasible_route']
                                                 and recs[key(c)]['connected_acquisition']],
                         cores=recs))

    # Stage B: arms
    arms = {}
    for arm in ARMS:
        masks = cmasks if arm == 'C' else full
        p, inconsistent = executable_branch(branches, masks)
        c = consolidate(masks[p]) if not inconsistent else C0
        vec = (p[0][0], p[0][1], p[1][0], p[1][1], c[0], c[1], r)
        rec = score(resp, run, w, p, c, masks[p], arm, true_p, inconsistent, memory)
        rec.update(private_blocks_unchanged=parts(vec)[:2] == p, serialization_roundtrip=roundtrip(vec, resp),
                   isolated_retention=all(resp.marginal(qi, waking_vector(s, p[0] if s == 'A' else p[1], r), s) == f
                                          for (s, qi), f in wake.items()),
                   executable_success=rec['passed'] and p == true_p,
                   compensating_pass=rec['passed'] and p != true_p)
        arms[arm] = rec

    # reference branches (arm C rule on every branch) and information
    reference, info = [], []
    for p in branches:
        cab = cmasks[p][0] & cmasks[p][1]
        c = consolidate(cmasks[p])
        if c is not None:
            reference.append(score(resp, run, w, p, c, cmasks[p], 'C', true_p, False, memory))
        else:
            reference.append(dict(branch=[list(p[0]), list(p[1])], true_branch=p == true_p,
                                  inconsistent_private_assembly=True, passed=False, passed_v1_route=False))
        disagree = {s: sum(len({resp.marginal(qi, (p[0][0], p[0][1], p[1][0], p[1][1], cc[0], cc[1], r), s)
                                for cc in cores(cab)}) > 1 for qi in run['mixed']) for s in ('A', 'B')}
        info.append(dict(branch=[list(p[0]), list(p[1])], true_branch=p == true_p,
                         C_A=[list(x) for x in cores(cmasks[p][0])], C_B=[list(x) for x in cores(cmasks[p][1])],
                         C_AB=[list(x) for x in cores(cab)], sign_classes=[list(x) for x in sign_classes(cab)],
                         donor_reduction={'A': len(cores(cmasks[p][0])) - len(cores(cab)),
                                          'B': len(cores(cmasks[p][1])) - len(cores(cab))},
                         mixed_disagreement_within_C_AB=disagree))

    # fixed-core controls on the arm-C executable branch and the true branch
    fixed = {}
    for label, p in (('executable', tuple(map(tuple, arms['C']['branch']))), ('true', true_p)):
        cab = cmasks[p][0] & cmasks[p][1]
        fixed[label] = {}
        for c in PAIRS:
            ev = evaluate(resp, run, w, p, c)
            ok = ev['preservation_nonvacuous'] and ev['preserved'] and ev['transfer'] and bool(ev['core_dependent_gains'])
            fixed[label][key(c)] = dict(passed=ok, passed_with_CA=ok and in_mask(cab, c),
                                        passed_v1_route=ok and bool(ev['route_gains']))
    checks['private_blocks_unchanged'] = all(a['private_blocks_unchanged'] for a in arms.values())
    checks['serialization_roundtrip'] = all(a['serialization_roundtrip'] for a in arms.values())
    checks['isolated_retention'] = all(a['isolated_retention'] for a in arms.values())
    checks['donor_clause_consistent'] = all(not x['recipient_only_fails'] or x['recipient_mask_has_alternative']
                                            for a in arms.values() for x in a['witnesses'])
    return dict(teacher=name, split=split['seed'], swapped=swapped, A_queries=ids_a, B_queries=ids_b,
                V_A=[list(x) for x in va], V_B=[list(x) for x in vb], true_branch=[list(pa), list(pb)],
                checks=checks, arms=arms, reference=reference, information=info, fixed_cores=fixed,
                feasibility=feas,
                work=dict(isolated_filter_evaluations=iso_evals, core_filter_evaluations=core_evals,
                          branches=len(branches), stored_mask_bits=18 + 2 * 9 * len(branches),
                          observations=dict(I=15, IR=30, C=30),
                          connected_target_cardinality=sum(len(f) for f in obs.values())))


def summary(runs):
    out = {}
    for name in TEACHERS:
        R = [r for r in runs if r['teacher'] == name]
        s = dict(runs=len(R))
        for arm in ARMS:
            A = [r['arms'][arm] for r in R]
            s[arm] = dict(passed=sum(a['passed'] for a in A), passed_v1_route=sum(a['passed_v1_route'] for a in A),
                          executable_success=sum(a['executable_success'] for a in A),
                          executable_success_v1_route=sum(a['passed_v1_route'] and a['true_branch'] for a in A),
                          compensating_pass=sum(a['compensating_pass'] for a in A),
                          on_true_branch=sum(a['true_branch'] for a in A),
                          inconsistent=sum(a['inconsistent_private_assembly'] for a in A),
                          installed_cores=dict(Counter(key(a['core']) for a in A)),
                          labels=dict(Counter(l for a in A for l in a['labels'])))
        tb = [f for r in R for f in r['feasibility'] if f['true_branch']]
        s['feasibility_true_branch'] = dict(
            runs_any_feasible=sum(bool(f['feasible']) for f in tb),
            runs_any_feasible_with_CA=sum(bool(f['feasible_with_CA']) for f in tb),
            runs_any_feasible_route=sum(bool(f['feasible_route']) for f in tb),
            feasible_sign_classes=dict(Counter(key(min(tuple(c), (-c[0], -c[1]))) for f in tb for c in f['feasible'])))
        allb = [f for r in R for f in r['feasibility']]
        s['feasibility_all_branches'] = dict(branches=len(allb), any_feasible=sum(bool(f['feasible']) for f in allb),
                                             any_feasible_with_CA=sum(bool(f['feasible_with_CA']) for f in allb))
        s['reference_branches'] = dict(
            branches=sum(len(r['reference']) for r in R),
            passed=sum(x['passed'] for r in R for x in r['reference']),
            passed_true_branch=sum(x['passed'] for r in R for x in r['reference'] if x['true_branch']),
            inconsistent=sum(x['inconsistent_private_assembly'] for r in R for x in r['reference']))
        s['fixed_cores'] = {
            lab: {key(c): dict(passed=sum(r['fixed_cores'][lab][key(c)]['passed'] for r in R),
                               passed_with_CA=sum(r['fixed_cores'][lab][key(c)]['passed_with_CA'] for r in R))
                  for c in PAIRS} for lab in ('executable', 'true')}
        cpass = [r for r in R if r['arms']['C']['passed']]
        s['single_fixed_core_explains_C_passes'] = [key(c) for c in PAIRS if cpass and all(
            r['fixed_cores']['executable'][key(c)]['passed'] for r in cpass)]
        s['connected_evidence_changes_core'] = sum(r['arms']['C']['core'] != r['arms']['I']['core'] for r in R)
        out[name] = s
    out['minimal_target_met'] = all(out[n]['C']['executable_success'] > 0 for n in POSITIVE)
    out['minimal_target_met_v1_route'] = all(out[n]['C']['executable_success_v1_route'] > 0 for n in POSITIVE)
    return out


def run_tests():
    names = ['tests.test_core_abstraction_v2', 'tests.test_core_abstraction']
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
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = freeze_manifest(out)
    if args.manifest_only:
        print(m['sha256'], m['contract_sha256'])
        return
    gate = dict(prior_artifacts=all(sha(p) == h for p, h in m['prior_artifacts'].items()),
                contract=sha(CONTRACT) == m['contract_sha256'])
    if not all(gate.values()):
        sys.exit(f'prior artifacts or contract differ: {gate}')
    splits = json.loads((ROOT / SPLITS).read_text())['splits']
    v1 = {(r['teacher'], r['split'], r['swapped']): r for r in json.loads((ROOT / V1_CEILING).read_text())['runs']}
    t0 = time.perf_counter()
    resp = Responses()
    runs = [run_one(resp, n, s, sw, v1[(n, s['seed'], sw)]) for n in TEACHERS for s in splits for sw in (False, True)]
    timing = dict(compute=time.perf_counter() - t0)
    for k in runs[0]['checks']:
        gate[k] = all(r['checks'][k] for r in runs)
    t0 = time.perf_counter()
    og = oracle_gate(resp)
    timing['oracle'] = time.perf_counter() - t0
    gate['oracle'] = og['passed']
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
    tests = run_tests()
    completion = completion_status(gate, tests)
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'],
                code={p: sha(p) for p in ('src/core_abstraction_v2.py', 'experiments/core_abstraction_v2.py',
                                          'tests/test_core_abstraction_v2.py')})
    work = dict(isolated_filter_evaluations=sum(r['work']['isolated_filter_evaluations'] for r in runs),
                core_filter_evaluations=sum(r['work']['core_filter_evaluations'] for r in runs),
                simulation_requests=resp.requests, unique_simulations=len(resp.memo),
                max_stored_mask_bits=max(r['work']['stored_mask_bits'] for r in runs))
    feas = [dict(teacher=r['teacher'], split=r['split'], swapped=r['swapped'], true_branch=r['true_branch'],
                 branches=r.pop('feasibility')) for r in runs]
    dump(out / f'{PREFIX}-feasibility.json', dict(
        experiment=PREFIX, stage='A', completion=completion, correctness_gate=gate, provenance=prov, runs=feas))
    dump(out / f'{PREFIX}-selector.json', dict(
        experiment=PREFIX, stage='B', completion=completion, correctness_gate=gate, oracle=og, tests=tests,
        summary=summary([dict(r, feasibility=f['branches']) for r, f in zip(runs, feas)]), work=work,
        provenance=prov, runs=runs))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: round(v, 3) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
