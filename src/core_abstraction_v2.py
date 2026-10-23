"""Core abstraction v2: connected core constraints, branch choice, installation and v2 predicates.

See docs/core-abstraction-v2-contract.md. Dynamics, responses, waking filters and assemblies are
imported unchanged from src/core_abstraction.py (v1). New here: per-branch nine-bit core masks
updated by connected observations, the declared executable-branch rule, the minimal-change
installation order, and the evaluator-only v2 predicates (section 6.1) and witness checks (7.1).

Learner-facing code (CoreFilter, acquire_core_masks, consolidate, executable_branch) sees only
training query IDs, supplied marginals, its surviving private blocks and r.
"""
from src.core_abstraction import C0, FULL_MASK, INPUT_FROZEN, PAIRS, SATS, assemble


CORE_ORDER = tuple(sorted(PAIRS, key=lambda c: (sum(v != 0 for v in c), c)))   # minimal change, then lexicographic
CORE_BIT = {c: i for i, c in enumerate(PAIRS)}                                 # mask bits use v1's PAIRS order
LABELS = ('inconsistent_private_assembly', 'wrong_private_branch', 'core_ambiguity', 'selected_c0',
          'preservation_vacuous', 'preservation_lost', 'connected_acquisition_failed', 'no_transfer',
          'mixed_regression', 'no_core_dependent_gain', 'donor_control_failed', 'comparator_control_failed')


def cores(mask):
    """Cores in a mask, in minimal-change order."""
    return [c for c in CORE_ORDER if mask >> CORE_BIT[c] & 1]


def in_mask(mask, c):
    return bool(mask >> CORE_BIT[c] & 1)


def first_core(mask):
    cs = cores(mask)
    return cs[0] if cs else None


def sign_classes(mask):
    return sorted({min(c, (-c[0], -c[1])) for c in cores(mask)})


class CoreFilter:
    """Nine-bit core mask for one satellite on one fixed private branch."""
    def __init__(self, sat, branch, r, resp):
        if sat not in SATS:
            raise ValueError(sat)
        self.sat, self.branch, self.r, self.resp = sat, branch, r, resp
        self.mask = FULL_MASK
        self.evaluations = 0

    def update(self, qi, target):
        keep = 0
        for c in PAIRS:
            if self.mask >> CORE_BIT[c] & 1:
                self.evaluations += 1
                if self.resp.marginal(qi, assemble(*self.branch, c, self.r), self.sat) == target:
                    keep |= 1 << CORE_BIT[c]
        self.mask = keep


def connected_targets(resp, teacher, ids_a, ids_b):
    """Connected observations: the addressed satellite's marginal under the full teacher."""
    out = {('A', qi): resp.marginal(qi, teacher, 'A') for qi in ids_a}
    out.update({('B', qi): resp.marginal(qi, teacher, 'B') for qi in ids_b})
    return out


def acquire_core_masks(resp, branches, r, order, ids_a, obs):
    """Per-branch (C_A, C_B) masks after the connected observations in `order`; returns masks and work."""
    out, evals = {}, 0
    for p in branches:
        fa, fb = CoreFilter('A', p, r, resp), CoreFilter('B', p, r, resp)
        for qi in order:
            sat = 'A' if qi in ids_a else 'B'
            (fa if sat == 'A' else fb).update(qi, obs[(sat, qi)])
        out[p] = (fa.mask, fb.mask)
        evals += fa.evaluations + fb.evaluations
    return out, evals


def consolidate(masks):
    """Installed core from (C_A, C_B): first of the intersection, or None if empty."""
    return first_core(masks[0] & masks[1])


def executable_branch(branches, masks):
    """First branch (in the given order) with a nonempty C_AB; else the first branch, inconsistent."""
    for p in branches:
        if masks[p][0] & masks[p][1]:
            return p, False
    return branches[0], True


# ---------------------------------------------------------------- evaluator-only predicates

def evaluate(resp, run, teacher, p, c):
    """Section 6.1 quantities for core c on branch p (no CA; that needs the masks)."""
    r = run['r']
    w, w0 = assemble(*p, c, r), assemble(*p, C0, r)

    def t(s, qi):
        return resp.marginal(qi, teacher, s)
    p0 = [(s, qi) for qi in run['train'] for s in SATS if resp.marginal(qi, w0, s) == t(s, qi)]
    p0_sizes = {s: sum(1 for s2, _ in p0 if s2 == s) for s in SATS}
    lost = [[s, qi] for s, qi in p0 if resp.marginal(qi, w, s) != t(s, qi)]
    sats, gains_all = {}, []
    for s in SATS:
        rec = {}
        for part in ('mixed', 'within'):
            now = {qi for qi in run[part] if resp.marginal(qi, w, s) == t(s, qi)}
            base = {qi for qi in run[part] if resp.marginal(qi, w0, s) == t(s, qi)}
            rec[part] = dict(correct=len(now), gains=sorted(now - base), losses=sorted(base - now))
        rec['net'] = len(rec['mixed']['gains']) - len(rec['mixed']['losses'])
        sats[s] = rec
    regress = any(sats[s]['mixed']['losses'] for s in SATS)
    recipients = [s for s in SATS if sats[s]['net'] > 0 and sats[s]['mixed']['gains']]
    transfer = c != C0 and bool(recipients) and not regress
    dep, route = [], []
    for s in recipients:
        for qi in sats[s]['mixed']['gains']:
            cuts = (assemble(*p, (0, c[1]), r), assemble(*p, (c[0], 0), r))
            if all(resp.marginal(qi, u, s) != t(s, qi) for u in cuts):
                dep.append((s, qi))
                if resp.marginal(qi, w, s, INPUT_FROZEN) != resp.marginal(qi, w0, s, INPUT_FROZEN):
                    route.append((s, qi))
    cross = sum(resp.marginal(qi, w, s) != f for (s, qi), f in run['wake'].items())
    preserved = not lost
    nonvacuous = all(p0_sizes[s] > 0 for s in SATS)
    return dict(p0_sizes=p0_sizes, preservation_nonvacuous=nonvacuous, preserved=preserved, lost=lost,
                satellites=sats, mixed_regression=regress, recipients=recipients, transfer=transfer,
                core_dependent_gains=dep, route_gains=route, cross_context_changes=cross,
                joint_mixed_correct=sum(resp.Y(qi, w) == resp.Y(qi, teacher) for qi in run['mixed']),
                feasible=nonvacuous and preserved and transfer and bool(dep),
                feasible_route=nonvacuous and preserved and transfer and bool(route))


def witness_checks(resp, run, teacher, p, c, gain, masks, comparator_cores, memory):
    """Conditions 2 and 3 of section 7.1 for one core-dependent gain (s, qi)."""
    s, qi = gain
    r = run['r']
    target = resp.marginal(qi, teacher, s)
    installed = resp.marginal(qi, assemble(*p, c, r), s)
    own = masks[0] if s == 'A' else masks[1]
    c_s = first_core(own)
    recipient_only_fails = c_s is None or resp.marginal(qi, assemble(*p, c_s, r), s) != target
    alternative = any(resp.marginal(qi, assemble(*p, c2, r), s) != installed for c2 in cores(own))
    comparators = {name: resp.marginal(qi, assemble(*p, cc, r), s) != target for name, cc in comparator_cores.items()}
    memorized = (s, qi) in memory
    return dict(satellite=s, query=qi, recipient_only_core=None if c_s is None else list(c_s),
                recipient_only_fails=recipient_only_fails, recipient_mask_has_alternative=alternative,
                donor_control=recipient_only_fails and alternative,
                comparators_fail=comparators, memorization_answers=memorized,
                comparator_control=all(comparators.values()) and not memorized)


__all__ = ['CORE_BIT', 'CORE_ORDER', 'CoreFilter', 'LABELS', 'acquire_core_masks', 'connected_targets',
           'consolidate', 'cores', 'evaluate', 'executable_branch', 'first_core', 'in_mask', 'sign_classes',
           'witness_checks']
