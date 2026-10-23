"""Shared core abstraction v3: matched-view complementarity ceiling (Stage A), unmatched envelope, executable census (B).

Run `.venv/bin/python -m experiments.shared_core_abstraction_v3 [--out-dir DIR] [--manifest-only]`.
"""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
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
    C0, FULL_MASK, INPUT_FROZEN, PAIRS, SATS, Responses, assemble, assignment, blocks, parts, wake_targets,
    waking_vector, witness_controls,
)
from src.core_abstraction_v2 import CORE_BIT, CORE_ORDER, CoreFilter, connected_targets, cores, evaluate, first_core, in_mask, sign_classes
from src.shared_core_abstraction_v3 import (
    JOINT_STATES, LABELS, SOURCE_STATES, atomic_masks, behavioral, direct_mask, joint_state, members, select_branch,
    select_single, source_state, structural, subset_masks, task_key, witness_conditions,
)
from src.shared_input_mediation import QUERIES


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'shared-core-abstraction-v3'
CONTRACT = 'docs/shared-core-abstraction-v3-contract.md'
SPLITS = 'reports/learned-reciprocal-participation-v1-splits.json'
V2_SELECTOR = 'reports/core-abstraction-v2-selector.json'
LEVELS = ('structural', 'behavioral', 'level3')
RULES = [
    'dynamics, responses, assemblies, isolated waking filters: v1 src/core_abstraction.py unchanged; nine-bit core '
    'filter, minimal-change order and evaluate(): v2 src/core_abstraction_v2.py unchanged',
    'task J: nonempty subset of the split\'s 15 training IDs; both satellites observe their own full-teacher '
    'marginal on every event of J; masks are intersections of atomic one-observation masks',
    'subset bitmask bit j = j-th ascending training ID; task order (|J|, sorted J); witnesses A before B, then '
    'ascending query; envelope pairs ordered (|J_A|+|J_B|, sorted J_A, sorted J_B)',
    'source state by surviving sign classes: 0 empty, 1 identifies, 2-4 ambiguous, 5 uninformative; joint state '
    'empty, two_link, other_single, ambiguous',
    'structural: both sources ambiguous, joint two_link, teacher core in C_AB on the true branch (nonempty C_AB '
    'elsewhere); behavioral: mixed (s,x) with |Pred_s(C_A,x)|>1, |Pred_s(C_B,x)|>1, Pred_s(C_AB,x)={M_s(T,x)}',
    'installed core: first of C_AB in minimal-change order ' + ' '.join(f'{a},{b}' for a, b in CORE_ORDER),
    'level 3 (full witness) at a behavioral (s,x): c0 incorrect, installed exact, both single-link cuts incorrect, '
    'recipient net mixed gain > 0, no mixed regression in either satellite, coupled preservation nonvacuous and held',
    'route criterion (secondary): additionally installed vs c0 recipient marginal differ with inputs frozen',
    'Stage A: true private blocks, assignment-independent; cross-context diagnostic over both satellites on all 15 '
    'training queries',
    'Stage B: v1/v2 isolated acquisition per assignment; per-branch masks for every task; executable branch = first '
    '(index p_A, index p_B) with nonempty C_AB, else first branch with c0 and inconsistent; executable success = '
    'selected branch is the true branch and level 3 holds; full witness on another branch is compensating',
    'source omission end to end: omitted source masks full on every branch; branch = first with nonempty remaining '
    'mask; core = its first core',
    'envelope: true branch, all 2^15 x 2^15 subset pairs; counts by total budget for identical, disjoint and any '
    'overlap; exact via per-source mask groups (convolution) cross-checked by a four-choice joint DP',
]
DISCLOSURES = [
    'v2 known: S1 donor-dependent executable witnesses in 4/10 runs (A own mask full, B pins +-(3,-3)); C2 0/10, '
    'each satellite own assigned marginals pin +-(3,-3); T1 2/10; S2 C_AB holds 7 cores incl c0',
    'v2 known: archived C2 V_A = {(0,-3),(3,-3),(3,0)} with true p_A = (3,0) last in branch order',
    'analytic: structural on the true branch implies C_AB = teacher mirror pair; installed model equals teacher on '
    'all 81 queries; acquisition, preservation, no regression and positive net gain then follow at any '
    'c0-incorrect witness',
    'analytic: condition 6 at fixed assembly equals the behavioral clause; Stage B true-branch masks equal Stage A',
    'no v3 atomic mask, subset census, or complementarity computation was performed before this manifest',
]


def manifest():
    splits = json.loads((ROOT / SPLITS).read_text())['splits']
    runs = {f'{s["seed"]}/{"swapped" if sw else "base"}': dict(zip(('A', 'B'), assignment(s['train'], sw)))
            for s in splits for sw in (False, True)}
    body = dict(schema=PREFIX, contract=CONTRACT, contract_sha256=sha(CONTRACT),
                split_file=SPLITS, split_file_sha256=sha(SPLITS),
                coefficient_order=['w_1A', 'w_2A', 'w_1B', 'w_2B', 'w_KA', 'w_KB', 'w_12'],
                teachers={k: list(v) for k, v in TEACHERS.items()}, positive_teachers=list(POSITIVE),
                specificity_controls=list(NEGATIVE), assignments=runs,
                training_ids={str(s['seed']): sorted(s['train']) for s in splits},
                task_counts=dict(per_split=2 ** 15 - 1, stage_a=4 * 5 * (2 ** 15 - 1), stage_b=40 * (2 ** 15 - 1),
                                 envelope_pairs_per_split=2 ** 30),
                block_order=[list(p) for p in PAIRS], core_order=[list(c) for c in CORE_ORDER],
                source_states=list(SOURCE_STATES), joint_states=list(JOINT_STATES), labels=list(LABELS),
                rules=RULES, disclosures=DISCLOSURES,
                prior_artifacts={p: sha(p) for p in [
                    SPLITS, 'src/shared_input_mediation.py', 'tests/test_shared_input_mediation.py',
                    'src/core_abstraction.py', 'experiments/core_abstraction.py', 'tests/test_core_abstraction.py',
                    'docs/core-abstraction-v1-contract.md', 'docs/core-abstraction-v1-results.md',
                    'reports/core-abstraction-v1-manifest.json', 'reports/core-abstraction-v1-ceiling.json',
                    'src/core_abstraction_v2.py', 'experiments/core_abstraction_v2.py',
                    'tests/test_core_abstraction_v2.py', 'docs/core-abstraction-v2-contract.md',
                    'docs/core-abstraction-v2-results.md', 'reports/core-abstraction-v2-manifest.json',
                    'reports/core-abstraction-v2-feasibility.json', V2_SELECTOR]})
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


def classes(mask):
    return [key(c) for c in sign_classes(mask)]


def popcount(J):
    return bin(J).count('1')


class Context:
    """One teacher/split (and, in Stage B, assignment): targets, evaluation run and memoized pair assessments."""
    def __init__(self, resp, name, split, wake):
        self.resp, self.name, self.w = resp, name, TEACHERS[name]
        pa, pb, self.ct, self.r = parts(self.w)
        self.true_p = (pa, pb)
        self.run = evaluation_run(split, self.r, wake)
        self.pairs, self.evals = {}, {}

    def target(self, s, qi):
        return self.resp.marginal(qi, self.w, s)

    def evaluation(self, p, c):
        if (p, c) not in self.evals:
            self.evals[(p, c)] = evaluate(self.resp, self.run, self.w, p, c)
        return self.evals[(p, c)]

    def correct(self, p, c, gain):
        s, qi = gain
        return self.resp.marginal(qi, assemble(*p, c, self.r), s) == self.target(s, qi)

    def assess(self, p, ma, mb):
        """Levels 1-3 and labels for masks (ma, mb) on branch p; memoized (all inputs are in the key)."""
        k = (p, ma, mb)
        if k in self.pairs:
            return self.pairs[k]
        true = p == self.true_p
        rec = dict(pattern='/'.join((source_state(ma), source_state(mb), joint_state(ma & mb))),
                   structural=structural(ma, mb, self.ct if true else None), core=None, witnesses=[])
        labels = set()
        if not rec['structural']:
            labels.add('not_structural')
        else:
            c = rec['core'] = first_core(ma & mb)
            assert in_mask(ma & mb, c)
            ev = self.evaluation(p, c)
            run_ok = not ev['mixed_regression'] and ev['preserved'] and ev['preservation_nonvacuous']
            for g in behavioral(self.resp, p, self.r, ma, mb, self.run['mixed'], self.target):
                s, qi = g
                wc = witness_conditions(self.resp, p, self.r, c, g, self.target)
                net = s in ev['recipients'] and qi in ev['satellites'][s]['mixed']['gains']
                full = all(wc.values()) and net and run_ok
                route = full and (self.resp.marginal(qi, assemble(*p, c, self.r), s, INPUT_FROZEN)
                                  != self.resp.marginal(qi, assemble(*p, C0, self.r), s, INPUT_FROZEN))
                rec['witnesses'].append(dict(satellite=s, query=qi, recipient_net_gain=net, full=full, route=route,
                                             **wc))
            W = rec['witnesses']
            if not W:
                labels.add('not_behavioral')
            else:
                if not any(x['c0_incorrect'] for x in W):
                    labels.add('c0_already_correct')
                elif not any(x['c0_incorrect'] and x['cuts_lose'] for x in W):
                    labels.add('cut_survives')
                if not any(x['recipient_net_gain'] for x in W):
                    labels.add('no_net_gain')
            if not ev['preservation_nonvacuous']:
                labels.add('preservation_vacuous')
            if not ev['preserved']:
                labels.add('preservation_lost')
            if ev['mixed_regression']:
                labels.add('mixed_regression')
        rec['behavioral'] = bool(rec['witnesses'])
        rec['full'] = [(x['satellite'], x['query']) for x in rec['witnesses'] if x['full']]
        rec['route'] = [(x['satellite'], x['query']) for x in rec['witnesses'] if x['route']]
        rec['level3'] = bool(rec['full'])
        rec['labels'] = labels
        self.pairs[k] = rec
        return rec


def eval_summary(ev):
    return dict(p0_sizes=ev['p0_sizes'], preserved=ev['preserved'], transfer=ev['transfer'],
                mixed_regression=ev['mixed_regression'], joint_mixed_correct=ev['joint_mixed_correct'],
                cross_context_changes=ev['cross_context_changes'],
                satellites={s: dict(mixed_correct=v['mixed']['correct'], net=v['net'],
                                    mixed_gains=len(v['mixed']['gains']), mixed_losses=len(v['mixed']['losses']),
                                    within_correct=v['within']['correct'], within_gains=len(v['within']['gains']),
                                    within_losses=len(v['within']['losses']))
                            for s, v in ev['satellites'].items()})


def roundtrip(vec, resp):
    """Serialize the installed vector alone, reload into a fresh simulator, compare all 81 response sets."""
    w = tuple(json.loads(json.dumps(list(vec))))
    fresh = Responses()
    return all(fresh.Y(qi, w) == resp.Y(qi, vec) for qi in range(len(QUERIES)))


def ablations(resp, ctx, p, c, gain):
    s, qi = gain
    pa, pb = p
    return witness_controls(resp, ctx.run, ctx.w, pa, pb, c, s, qi)


def witness_record(ids, J, gain, mask_a, mask_b):
    return dict(J=members(ids, J), budget=popcount(J), satellite=gain[0], query=gain[1],
                C_A=classes(mask_a), C_B=classes(mask_b))


# ---------------------------------------------------------------- Stage A and envelope

def groups_of(masks, ids):
    """Per-source mask groups over all subsets (incl. empty): counts by budget and canonical subset per (mask, b)."""
    n = len(ids)
    counts, canon = {}, {}
    for J, m in enumerate(masks):
        b = popcount(J)
        counts.setdefault(m, [0] * (n + 1))[b] += 1
        mem = members(ids, J)
        if (m, b) not in canon or mem < canon[(m, b)]:
            canon[(m, b)] = mem
    return counts, canon


def joint_dp(atoms_a, atoms_b, disjoint):
    """Exact counts {(m_A, m_B, total budget): pairs} over source-subset pairs; disjoint drops the 'both' choice."""
    states = {(FULL_MASK, FULL_MASK, 0): 1}
    for a, b in zip(atoms_a, atoms_b):
        new = Counter()
        for (ma, mb, t), n in states.items():
            new[(ma, mb, t)] += n
            new[(ma & a, mb, t + 1)] += n
            new[(ma, mb & b, t + 1)] += n
            if not disjoint:
                new[(ma & a, mb & b, t + 2)] += n
        states = new
    return states


def disjoint_witness(ids, masks_a, masks_b, qualifies, total):
    """First (sorted J_A, sorted J_B) with |J_A|+|J_B| = total, disjoint, qualifying; declared order."""
    n = len(ids)
    for J_A in sorted(range(1 << n), key=lambda J: members(ids, J)):
        need = total - popcount(J_A)
        if need < 0:
            continue
        rest = [j for j in range(n) if not J_A >> j & 1]
        for combo in combinations(rest, need):
            J_B = sum(1 << j for j in combo)
            if qualifies(masks_a[J_A], masks_b[J_B]):
                return dict(J_A=members(ids, J_A), J_B=members(ids, J_B), overlap=0)
    return None


def envelope(ctx, ids, masks_a, masks_b, atoms_a, atoms_b, matched_counts):
    p = ctx.true_p
    n = len(ids)
    ga, ca = groups_of(masks_a, ids)
    gb, cb = groups_of(masks_b, ids)
    conv = Counter()
    for ma, va in ga.items():
        for mb, vb in gb.items():
            for i, x in enumerate(va):
                for j, y in enumerate(vb):
                    if x and y:
                        conv[(ma, mb, i + j)] += x * y
    dp_any = joint_dp(atoms_a, atoms_b, disjoint=False)
    dp_dis = joint_dp(atoms_a, atoms_b, disjoint=True)
    checks = dict(convolution_equals_dp=+conv == +Counter(dp_any),
                  any_total=sum(conv.values()) == 4 ** n, disjoint_total=sum(dp_dis.values()) == 3 ** n,
                  per_source_total=all(sum(map(sum, g.values())) == 2 ** n for g in (ga, gb)))
    level = {}
    for ma in ga:
        for mb in gb:
            rec = ctx.assess(p, ma, mb)
            level[(ma, mb)] = dict(structural=rec['structural'], behavioral=rec['behavioral'], level3=rec['level3'],
                                   route=bool(rec['route']))
    out = dict(checks=checks)
    for lv in LEVELS + ('route',):
        def q(ma, mb, lv=lv):
            return level[(ma, mb)][lv]
        by = {}
        for name, table in (('any', conv), ('disjoint', dp_dis)):
            t = [0] * (2 * n + 1)
            for (ma, mb, tot), cnt in table.items():
                if q(ma, mb):
                    t[tot] += cnt
            by[name] = t
        by['identical'] = matched_counts[lv]
        rec = dict(counts_by_total_budget=by,
                   minimal_total_budget={k: next((i for i, x in enumerate(v) if x), None) for k, v in by.items()},
                   qualifying_mask_pairs=[[classes(ma), classes(mb)] for (ma, mb), v in sorted(level.items())
                                          if v[lv]])
        best = None
        for ma, va in ga.items():
            for mb, vb in gb.items():
                if not q(ma, mb):
                    continue
                for i, x in enumerate(va):
                    for j, y in enumerate(vb):
                        if x and y:
                            cand = (i + j, ca[(ma, i)], cb[(mb, j)])
                            best = cand if best is None or cand < best else best
        rec['canonical_witness_any'] = None if best is None else dict(
            J_A=best[1], J_B=best[2], total=best[0], overlap=len(set(best[1]) & set(best[2])))
        t_dis = rec['minimal_total_budget']['disjoint']
        rec['canonical_witness_disjoint'] = None if t_dis is None else disjoint_witness(
            ids, masks_a, masks_b, q, t_dis)
        out[lv] = rec
    out['closure'] = dict(A=[dict(classes=classes(m), counts_by_budget=v) for m, v in sorted(ga.items())],
                          B=[dict(classes=classes(m), counts_by_budget=v) for m, v in sorted(gb.items())])
    return out


def stage_a(resp, name, split):
    ids = tuple(sorted(split['train']))
    n = len(ids)
    w = TEACHERS[name]
    pa, pb, ct, r = parts(w)
    p = (pa, pb)
    wake = {(s, qi): resp.marginal(qi, waking_vector(s, pa if s == 'A' else pb, r), s) for s in SATS for qi in ids}
    ctx = Context(resp, name, split, wake)
    obs = connected_targets(resp, w, ids, ids)
    atoms = atomic_masks(resp, p, r, ids, obs)
    A, B = [atoms[('A', q)] for q in ids], [atoms[('B', q)] for q in ids]
    mA, mB = subset_masks(A), subset_masks(B)
    flat = [sum(1 << CORE_BIT[c] for c in PAIRS
                if all(resp.marginal(q, assemble(*p, c, r), s) == obs[(s, q)] for s in SATS)) for q in ids]
    mF = subset_masks(flat)
    checks = dict(
        direct_equals_incremental=all(mA[J] == direct_mask(A, J) and mB[J] == direct_mask(B, J) for J in range(1 << n)),
        teacher_core_survives=all(in_mask(m, ct) for m in A + B),
        atoms_mirror_closed=all(in_mask(m, c) == in_mask(m, (-c[0], -c[1])) for m in A + B for c in PAIRS),
        flat_equals_intersection=all(mF[J] == mA[J] & mB[J] for J in range(1 << n)),
        repeated_pass_identical=all(mA[J] & a == mA[J] and mB[J] & b == mB[J]
                                    for J in range(1, 1 << n) for j, (a, b) in enumerate(zip(A, B)) if J >> j & 1))
    budgets = {b: dict(tasks=0, **{lv: 0 for lv in LEVELS}, route=0, patterns=Counter()) for b in range(1, n + 1)}
    first = {lv: None for lv in LEVELS + ('route',)}
    matched = {lv: [0] * (2 * n + 1) for lv in LEVELS + ('route',)}
    lucky = Counter()
    fixed_at_witness = Counter()
    for J in range(1, 1 << n):
        b = popcount(J)
        rec = ctx.assess(p, mA[J], mB[J])
        row = budgets[b]
        row['tasks'] += 1
        row['patterns'][rec['pattern']] += 1
        flags = dict(structural=rec['structural'], behavioral=rec['behavioral'], level3=rec['level3'],
                     route=bool(rec['route']))
        for lv, ok in flags.items():
            if ok:
                row[lv] += 1
                matched[lv][2 * b] += 1
                tk = task_key(ids, J)
                if first[lv] is None or tk < first[lv][0]:
                    first[lv] = (tk, J)
        if rec['level3']:
            g = rec['full'][0]
            ca, cb = first_core(mA[J]), first_core(mB[J])
            la, lb = ctx.correct(p, ca, g), ctx.correct(p, cb, g)
            lucky['A_choice_correct'] += la
            lucky['B_choice_correct'] += lb
            lucky['both_correct'] += la and lb
            lucky['neither_correct'] += not la and not lb
            for c in PAIRS:
                fixed_at_witness[key(c)] += ctx.correct(p, c, g)
    level3_tasks = sum(budgets[b]['level3'] for b in budgets)
    examples = {}
    for lv, f in first.items():
        if f is None:
            examples[lv] = None
            continue
        J = f[1]
        rec = ctx.assess(p, mA[J], mB[J])
        gains = rec['route'] if lv == 'route' else rec['full']
        ex = dict(J=members(ids, J), budget=popcount(J), pattern=rec['pattern'], C_A=classes(mA[J]),
                  C_B=classes(mB[J]), C_AB=classes(mA[J] & mB[J]),
                  installed_core=None if rec['core'] is None else list(rec['core']),
                  witnesses=rec['witnesses'])
        if lv in ('level3', 'route'):
            g = gains[0]
            ex['canonical_witness'] = dict(satellite=g[0], query=g[1],
                                           pred_A=sorted(sorted(m) for m in _preds(ctx, p, g, mA[J])),
                                           pred_B=sorted(sorted(m) for m in _preds(ctx, p, g, mB[J])),
                                           teacher=sorted(ctx.target(*g)),
                                           ablations=ablations(resp, ctx, p, rec['core'], g),
                                           installed=eval_summary(ctx.evaluation(p, rec['core'])))
        examples[lv] = ex
    full = (1 << n) - 1
    fr = ctx.assess(p, mA[full], mB[full])
    fixed = {}
    for c in PAIRS:
        ev = ctx.evaluation(p, c)
        fixed[key(c)] = dict(passed=ev['preservation_nonvacuous'] and ev['preserved'] and ev['transfer']
                             and bool(ev['core_dependent_gains']), transfer=ev['transfer'],
                             joint_mixed_correct=ev['joint_mixed_correct'])
    env = envelope(ctx, ids, mA, mB, A, B, matched)
    checks.update({f'envelope_{k}': v for k, v in env.pop('checks').items()})
    checks['envelope_contains_matched'] = all(
        env[lv]['counts_by_total_budget']['any'][t] >= matched[lv][t] for lv in matched for t in range(2 * n + 1))
    atom_rec = {s: [dict(query=q, classes=classes(atoms[(s, q)]), state=source_state(atoms[(s, q)])) for q in ids]
                for s in SATS}
    informative_identify = {s: all(source_state(atoms[(s, q)]) in ('identifies', 'uninformative') for q in ids)
                            for s in SATS}
    return ctx, mA, mB, dict(
        teacher=name, split=split['seed'], training_ids=list(ids), true_branch=[list(pa), list(pb)],
        checks=checks, atomic_masks=atom_rec, every_informative_atom_identifies=informative_identify,
        budgets={str(b): dict(v, patterns=dict(sorted(v['patterns'].items()))) for b, v in budgets.items()},
        totals={lv: sum(budgets[b][lv] for b in budgets) for lv in LEVELS + ('route',)},
        minimal_budget={lv: next((b for b in budgets if budgets[b][lv]), None) for lv in LEVELS + ('route',)},
        first_examples=examples,
        full_set=dict(pattern=fr['pattern'], C_A=classes(mA[full]), C_B=classes(mB[full]),
                      C_AB=classes(mA[full] & mB[full]), structural=fr['structural'], behavioral=fr['behavioral'],
                      level3=fr['level3']),
        single_source_choices_at_level3_witness=dict(tasks=level3_tasks, **lucky),
        fixed_cores_true_branch=fixed,
        fixed_cores_correct_at_level3_witness=dict(tasks=level3_tasks, **dict(fixed_at_witness)),
        envelope=env, memorization_answers_mixed=0,
        work=dict(atomic_filter_evaluations=2 * n * 9, subset_mask_updates=2 * ((1 << n) - 1),
                  connected_target_cardinality_full_set=sum(len(obs[(s, q)]) for s in SATS for q in ids),
                  distinct_mask_pairs_assessed=len(ctx.pairs)))


def _preds(ctx, p, g, mask):
    s, qi = g
    return {ctx.resp.marginal(qi, assemble(*p, c, ctx.r), s) for c in cores(mask)}


# ---------------------------------------------------------------- Stage B

def stage_b(resp, name, split, swapped, v2run, stage_a_masks):
    ids = tuple(sorted(split['train']))
    n = len(ids)
    w = TEACHERS[name]
    pa, pb, ct, r = parts(w)
    true_p = (pa, pb)
    ids_a, ids_b = assignment(split['train'], swapped)
    wake = wake_targets(resp, w, ids_a, ids_b)
    pmasks, iso_evals, idem = acquire_all_orders(resp, split, ids_a, r, wake)
    checks = dict(private_order_independent=pmasks is not None, private_repetition_idempotent=idem)
    va, vb = blocks(pmasks[0]), blocks(pmasks[1])
    checks['v2_private_masks_reproduced'] = ([list(x) for x in va], [list(x) for x in vb]) == (v2run['V_A'], v2run['V_B'])
    branches = [(x, y) for x in va for y in vb]
    checks['isolated_retention'] = all(
        resp.marginal(qi, waking_vector(s, p[0] if s == 'A' else p[1], r), s) == f
        for p in branches for (s, qi), f in wake.items())
    ctx = Context(resp, name, split, wake)
    obs = connected_targets(resp, w, ids, ids)
    MA, MB, AT = {}, {}, {}
    for p in branches:
        atoms = atomic_masks(resp, p, r, ids, obs)
        AT[p] = atoms
        MA[p] = subset_masks([atoms[('A', q)] for q in ids])
        MB[p] = subset_masks([atoms[('B', q)] for q in ids])
    v2info = {(tuple(i['branch'][0]), tuple(i['branch'][1])): i for i in v2run['information']}
    ok = True
    for p in branches:
        ca = cb = FULL_MASK
        for q in ids_a:
            ca &= AT[p][('A', q)]
        for q in ids_b:
            cb &= AT[p][('B', q)]
        i = v2info[p]
        ok &= (sorted(map(tuple, i['C_A'])), sorted(map(tuple, i['C_B']))) == (sorted(cores(ca)), sorted(cores(cb)))
    checks['v2_core_masks_reproduced_from_atoms'] = ok
    full = (1 << n) - 1
    orders_ok = dup_ok = rep_ok = True
    for p in branches:
        for order in split['orders']:
            fa, fb = CoreFilter('A', p, r, resp), CoreFilter('B', p, r, resp)
            for qi in order:
                fa.update(qi, obs[('A', qi)])
                fb.update(qi, obs[('B', qi)])
            orders_ok &= (fa.mask, fb.mask) == (MA[p][full], MB[p][full])
            for qi in order:
                fa.update(qi, obs[('A', qi)])
                fb.update(qi, obs[('B', qi)])
            rep_ok &= (fa.mask, fb.mask) == (MA[p][full], MB[p][full])
        da, db = CoreFilter('A', p, r, resp), CoreFilter('B', p, r, resp)
        for qi in ids:
            for _ in range(2):
                da.update(qi, obs[('A', qi)])
                db.update(qi, obs[('B', qi)])
        dup_ok &= (da.mask, db.mask) == (MA[p][full], MB[p][full])
    checks.update(core_masks_order_independent=orders_ok, repeated_paired_pass_identical=rep_ok,
                  duplicate_source_idempotent=dup_ok,
                  true_branch_masks_equal_stage_a=MA[true_p] == stage_a_masks[0] and MB[true_p] == stage_a_masks[1],
                  teacher_core_survives_true_branch=all(in_mask(MA[true_p][J] & MB[true_p][J], ct) for J in range(1 << n)))
    budgets = {b: dict(tasks=0, inconsistent=0, true_selected=0, structural=0, behavioral=0, full=0, executable=0,
                       executable_route=0, compensating=0, labels=Counter(), patterns=Counter())
               for b in range(1, n + 1)}
    installed = Counter()
    omission = {k: Counter() for k in ('omit_A', 'omit_B')}
    same_branch = Counter()
    first_exec = None
    for J in range(1, 1 << n):
        b = popcount(J)
        row = budgets[b]
        row['tasks'] += 1
        table = {p: (MA[p][J], MB[p][J]) for p in branches}
        p, inc = select_branch(branches, table)
        rec = ctx.assess(p, *table[p])
        labels = set(rec['labels'])
        if inc:
            labels.add('inconsistent_private_assembly')
            row['inconsistent'] += 1
        if p != true_p:
            labels.add('wrong_private_branch')
        else:
            row['true_selected'] += 1
        c = C0 if inc else rec['core'] if rec['core'] is not None else first_core(table[p][0] & table[p][1])
        installed[(p, c)] += 1
        row['structural'] += rec['structural']
        row['behavioral'] += rec['behavioral']
        row['full'] += rec['level3']
        row['patterns'][rec['pattern']] += 1
        for lab in labels:
            row['labels'][lab] += 1
        exe = not inc and p == true_p and rec['level3']
        row['executable'] += exe
        row['executable_route'] += exe and bool(rec['route'])
        row['compensating'] += (not inc) and p != true_p and rec['level3']
        for name_, keep in (('omit_B', 0), ('omit_A', 1)):
            p1, c1, inc1 = select_single(branches, table, keep)
            o = omission[name_]
            o['branch_changed'] += p1 != p
            o['core_changed'] += c1 != c
            o['inconsistent'] += inc1
            if exe:
                o['executable_tasks'] += 1
                o['witness_correct_without_source'] += ctx.correct(p1, c1, rec['full'][0])
                o['branch_changed_on_executable'] += p1 != p
        if exe:
            g = rec['full'][0]
            same_branch['tasks'] += 1
            same_branch['A_choice_correct'] += ctx.correct(p, first_core(table[p][0]), g)
            same_branch['B_choice_correct'] += ctx.correct(p, first_core(table[p][1]), g)
            tk = task_key(ids, J)
            if first_exec is None or tk < first_exec[0]:
                first_exec = (tk, J)
    example = None
    if first_exec is not None:
        J = first_exec[1]
        rec = ctx.assess(true_p, MA[true_p][J], MB[true_p][J])
        g = rec['full'][0]
        example = dict(witness_record(ids, J, g, MA[true_p][J], MB[true_p][J]), installed_core=list(rec['core']),
                       ablations=ablations(resp, ctx, true_p, rec['core'], g))
    inst = []
    for (p, c), cnt in sorted(installed.items()):
        vec = assemble(*p, c, r)
        inst.append(dict(branch=[list(p[0]), list(p[1])], core=list(c), true_branch=p == true_p, tasks=cnt,
                         private_blocks_unchanged=parts(vec)[:2] == p, serialization_roundtrip=roundtrip(vec, resp),
                         evaluation=eval_summary(ctx.evaluation(p, c))))
    checks['private_blocks_unchanged'] = all(x['private_blocks_unchanged'] for x in inst)
    checks['serialization_roundtrip'] = all(x['serialization_roundtrip'] for x in inst)
    totals = {k: sum(budgets[b][k] for b in budgets)
              for k in ('inconsistent', 'true_selected', 'structural', 'behavioral', 'full', 'executable',
                        'executable_route', 'compensating')}
    return dict(
        teacher=name, split=split['seed'], swapped=swapped, A_queries=ids_a, B_queries=ids_b,
        V_A=[list(x) for x in va], V_B=[list(x) for x in vb], true_branch=[list(pa), list(pb)],
        branches=[[list(p[0]), list(p[1])] for p in branches], checks=checks, totals=totals,
        minimal_executable_budget=next((b for b in budgets if budgets[b]['executable']), None),
        budgets={str(b): dict(v, labels=dict(sorted(v['labels'].items(), key=lambda kv: LABELS.index(kv[0]))),
                              patterns=dict(sorted(v['patterns'].items()))) for b, v in budgets.items()},
        full_set=dict(selected=[list(x) for x in select_branch(branches, {p: (MA[p][full], MB[p][full])
                                                                           for p in branches})[0]],
                      C_AB_by_branch=[[[list(p[0]), list(p[1])], classes(MA[p][full] & MB[p][full])]
                                      for p in branches]),
        installed=inst, source_omission_end_to_end={k: dict(v) for k, v in omission.items()},
        same_branch_single_source_choice_at_executable_witness=dict(same_branch),
        first_executable_example=example, memorization_answers_mixed=0,
        work=dict(isolated_filter_evaluations=iso_evals, atomic_filter_evaluations=len(branches) * 2 * n * 9,
                  subset_mask_updates=len(branches) * 2 * ((1 << n) - 1), branches=len(branches),
                  logical_mask_bits_per_task=2 * 9 * len(branches) + 18,
                  logical_table_bits=2 * 9 * len(branches) * (1 << n),
                  distinct_mask_pairs_assessed=len(ctx.pairs)))


# ---------------------------------------------------------------- summary and driver

def summary(a_runs, b_runs):
    out = {}
    for name in TEACHERS:
        A = [x for x in a_runs if x['teacher'] == name]
        B = [x for x in b_runs if x['teacher'] == name]
        s = dict(stage_a=dict(
            tasks=sum(sum(v['tasks'] for v in x['budgets'].values()) for x in A),
            totals={lv: sum(x['totals'][lv] for x in A) for lv in LEVELS + ('route',)},
            splits_with={lv: sum(x['totals'][lv] > 0 for x in A) for lv in LEVELS + ('route',)},
            minimal_budget={lv: min((x['minimal_budget'][lv] for x in A if x['minimal_budget'][lv]), default=None)
                            for lv in LEVELS + ('route',)},
            full_set_patterns=dict(Counter(x['full_set']['pattern'] for x in A)),
            envelope={lv: {ov: dict(splits=sum(x['envelope'][lv]['minimal_total_budget'][ov] is not None for x in A),
                                    minimal_total_budget=min((x['envelope'][lv]['minimal_total_budget'][ov] for x in A
                                                              if x['envelope'][lv]['minimal_total_budget'][ov]
                                                              is not None), default=None),
                                    pairs=sum(sum(x['envelope'][lv]['counts_by_total_budget'][ov]) for x in A))
                           for ov in ('identical', 'disjoint', 'any')} for lv in LEVELS + ('route',)},
            every_informative_atom_identifies={sat: sum(x['every_informative_atom_identifies'][sat] for x in A)
                                               for sat in SATS}))
        sb = {}
        for sw in (False, True):
            R = [x for x in B if x['swapped'] == sw]
            sb['swapped' if sw else 'base'] = dict(
                tasks=sum(sum(v['tasks'] for v in x['budgets'].values()) for x in R),
                totals={k: sum(x['totals'][k] for x in R) for k in R[0]['totals']},
                runs_with_executable=sum(x['totals']['executable'] > 0 for x in R),
                runs_with_compensating=sum(x['totals']['compensating'] > 0 for x in R),
                minimal_executable_budget=min((x['minimal_executable_budget'] for x in R
                                               if x['minimal_executable_budget']), default=None))
        s['stage_b'] = sb
        s['existence_pass'] = any(sb[k]['totals']['executable'] > 0 for k in sb)
        s['existence_pass_route'] = any(sb[k]['totals']['executable_route'] > 0 for k in sb)
        out[name] = s
    out['two_teacher_target_met'] = all(out[n]['existence_pass'] for n in POSITIVE)
    out['two_teacher_target_met_route'] = all(out[n]['existence_pass_route'] for n in POSITIVE)
    out['primary_partial'] = [n for n in POSITIVE if out[n]['existence_pass']]
    return out


def run_tests():
    names = ['tests.test_shared_core_abstraction_v3', 'tests.test_core_abstraction_v2', 'tests.test_core_abstraction']
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
    v2 = {(r['teacher'], r['split'], r['swapped']): r for r in json.loads((ROOT / V2_SELECTOR).read_text())['runs']}
    resp = Responses()
    timing = {}
    t0 = time.perf_counter()
    a_runs, a_masks = [], {}
    for n in TEACHERS:
        for s in splits:
            _, mA, mB, rec = stage_a(resp, n, s)
            a_runs.append(rec)
            a_masks[(n, s['seed'])] = (mA, mB)
    timing['stage_a_and_envelope'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    b_runs = [stage_b(resp, n, s, sw, v2[(n, s['seed'], sw)], a_masks[(n, s['seed'])])
              for n in TEACHERS for s in splits for sw in (False, True)]
    timing['stage_b'] = time.perf_counter() - t0
    for runs, tag in ((a_runs, 'A'), (b_runs, 'B')):
        for k in runs[0]['checks']:
            gate[f'{tag}:{k}'] = all(r['checks'][k] for r in runs)
    t0 = time.perf_counter()
    og = oracle_gate(resp)
    timing['oracle'] = time.perf_counter() - t0
    gate['oracle'] = og['passed']
    gate['passed'] = all(v for k, v in gate.items() if k != 'passed')
    tests = run_tests()
    completion = completion_status(gate, tests)
    prov = dict(manifest_sha256=m['sha256'], contract_sha256=m['contract_sha256'],
                code={p: sha(p) for p in ('src/shared_core_abstraction_v3.py',
                                          'experiments/shared_core_abstraction_v3.py',
                                          'tests/test_shared_core_abstraction_v3.py')})
    work = dict(simulation_requests=resp.requests, unique_simulations=len(resp.memo),
                stage_a_atomic_filter_evaluations=sum(r['work']['atomic_filter_evaluations'] for r in a_runs),
                stage_b_isolated_filter_evaluations=sum(r['work']['isolated_filter_evaluations'] for r in b_runs),
                stage_b_atomic_filter_evaluations=sum(r['work']['atomic_filter_evaluations'] for r in b_runs),
                stage_b_subset_mask_updates=sum(r['work']['subset_mask_updates'] for r in b_runs),
                max_logical_mask_bits_per_task=max(r['work']['logical_mask_bits_per_task'] for r in b_runs),
                max_logical_table_bits=max(r['work']['logical_table_bits'] for r in b_runs),
                persisted_after_installation='7 coefficients')
    summ = summary(a_runs, b_runs)
    dump(out / f'{PREFIX}-ceiling.json', dict(
        experiment=PREFIX, stage='A', completion=completion, correctness_gate=gate, provenance=prov,
        summary={n: summ[n]['stage_a'] for n in TEACHERS}, runs=a_runs))
    dump(out / f'{PREFIX}-selector.json', dict(
        experiment=PREFIX, stage='B', completion=completion, correctness_gate=gate, oracle=og, tests=tests,
        summary=summ, work=work, provenance=prov, runs=b_runs))
    git = lambda *c: subprocess.run(['git', *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dump(out / f'{PREFIX}-runtime.json', dict(
        seconds={k: round(v, 3) for k, v in timing.items()},
        total_including_preprocessing=round(time.perf_counter() - started, 3),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_rss_scope='whole process',
        python=platform.python_version(), machine=platform.machine(), platform=platform.platform(),
        out_dir=str(out), command=' '.join(sys.argv), git_revision=git('rev-parse', 'HEAD'),
        dirty_worktree=bool(git('status', '--porcelain'))))
    print(completion, json.dumps(gate))


if __name__ == '__main__':
    main()
