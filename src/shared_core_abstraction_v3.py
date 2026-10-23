"""Shared core abstraction v3: matched-view core masks, complementarity predicates and exact subset census.

See docs/shared-core-abstraction-v3-contract.md. Dynamics, responses, assemblies and the nine-bit core
filter are imported unchanged from src/core_abstraction.py (v1) and src/core_abstraction_v2.py (v2).

A task J is a set of training query IDs; both satellites observe their own full-teacher marginal on
every event in J. Masks are intersections of atomic (one-observation) masks, so every subset mask is
computed from the atomic table. Subsets are bitmasks over a declared ID tuple: bit j is ids[j].

Learner-facing code (atomic_masks, subset_masks, select_branch) sees only training query IDs, supplied
marginals, surviving private blocks and r. Everything else here is evaluator code.
"""
from src.core_abstraction import C0, FULL_MASK, SATS, assemble
from src.core_abstraction_v2 import CoreFilter, cores, first_core, in_mask, sign_classes


N_CLASSES = 5
SOURCE_STATES = ('empty', 'identifies', 'ambiguous', 'uninformative')
JOINT_STATES = ('empty', 'two_link', 'other_single', 'ambiguous')
LABELS = ('inconsistent_private_assembly', 'wrong_private_branch', 'not_structural', 'not_behavioral',
          'c0_already_correct', 'cut_survives', 'preservation_vacuous', 'preservation_lost', 'mixed_regression',
          'no_net_gain')


def atomic_masks(resp, p, r, ids, obs):
    """{(sat, qi): core mask after the single observation obs[(sat, qi)] from the full mask} on branch p."""
    out = {}
    for qi in ids:
        for s in SATS:
            f = CoreFilter(s, p, r, resp)
            f.update(qi, obs[(s, qi)])
            out[(s, qi)] = f.mask
    return out


def subset_masks(atoms):
    """Incremental masks for every subset of len(atoms) atoms: out[J] = AND of atoms[j] for bits j in J."""
    out = [FULL_MASK] * (1 << len(atoms))
    for J in range(1, len(out)):
        low = J & -J
        out[J] = out[J ^ low] & atoms[low.bit_length() - 1]
    return out


def direct_mask(atoms, J):
    """Independent reference for subset_masks: reduce over the members of J."""
    m = FULL_MASK
    for j, a in enumerate(atoms):
        if J >> j & 1:
            m &= a
    return m


def members(ids, J):
    return [q for j, q in enumerate(ids) if J >> j & 1]


def task_key(ids, J):
    """Declared task order: budget, then sorted member IDs."""
    return (bin(J).count('1'), sorted(members(ids, J)))


def select_branch(branches, table):
    """v2 executable rule on per-branch (C_A, C_B): first nonempty C_AB, else (first branch, inconsistent)."""
    for p in branches:
        if table[p][0] & table[p][1]:
            return p, False
    return branches[0], True


def select_single(branches, table, keep):
    """End-to-end source omission: only source `keep` (0 = A, 1 = B) constrains; returns branch, core, inconsistent."""
    for p in branches:
        if table[p][keep]:
            return p, first_core(table[p][keep]), False
    return branches[0], C0, True


def source_state(mask):
    n = len(sign_classes(mask))
    return 'empty' if n == 0 else 'identifies' if n == 1 else 'uninformative' if n == N_CLASSES else 'ambiguous'


def joint_state(mask):
    q = sign_classes(mask)
    if not q:
        return 'empty'
    if len(q) > 1:
        return 'ambiguous'
    return 'two_link' if all(q[0]) else 'other_single'


def structural(ma, mb, teacher_core=None):
    """Section 4 structural complementarity; teacher_core None on non-true branches (nonempty C_AB instead)."""
    ok = source_state(ma) == 'ambiguous' and source_state(mb) == 'ambiguous' and joint_state(ma & mb) == 'two_link'
    return ok and (teacher_core is None or in_mask(ma & mb, teacher_core))


def predictions(resp, p, r, s, qi, mask):
    """Pred_s(C, x): the set of complete marginals predicted by the cores in mask (a set of sets)."""
    return frozenset(resp.marginal(qi, assemble(*p, c, r), s) for c in cores(mask))


def behavioral(resp, p, r, ma, mb, mixed, target):
    """All (s, qi), A before B then ascending qi, satisfying section 4's behavioral clause."""
    out = []
    for s in SATS:
        for qi in sorted(mixed):
            if (len(predictions(resp, p, r, s, qi, ma)) > 1 and len(predictions(resp, p, r, s, qi, mb)) > 1
                    and predictions(resp, p, r, s, qi, ma & mb) == {target(s, qi)}):
                out.append((s, qi))
    return out


def witness_conditions(resp, p, r, c, gain, target):
    """Per-(s, x) conditions 1 and 5 of section 7: c0 incorrect, installed exact, each single cut loses it."""
    s, qi = gain
    t = target(s, qi)

    def ok(core):
        return resp.marginal(qi, assemble(*p, core, r), s) == t
    return dict(c0_incorrect=not ok(C0), installed_exact=ok(c),
                cuts_lose=not ok((0, c[1])) and not ok((c[0], 0)))


__all__ = ['JOINT_STATES', 'LABELS', 'N_CLASSES', 'SOURCE_STATES', 'atomic_masks', 'behavioral', 'direct_mask',
           'joint_state', 'members', 'predictions', 'select_branch', 'select_single', 'source_state', 'structural',
           'subset_masks', 'task_key', 'witness_conditions']
