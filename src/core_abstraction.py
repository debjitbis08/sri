"""Core abstraction v1: waking filters, assemblies, candidate cores and the Stage-A predicates.

See docs/core-abstraction-v1-contract.md. Dynamics are imported unchanged from
src/shared_input_mediation.py. Persistent coefficients are p_A = (w_1A, w_2A), p_B = (w_1B, w_2B),
c = (w_KA, w_KB) and r = w_12. Only c changes during consolidation.

The waking filter sees only (query, addressed satellite's marginal) in its isolated environment.
Everything else here (teacher targets, assessment, witnesses, controls) is evaluator code.
"""
import itertools

from src.shared_input_mediation import A, B, I1, I2, K, QUERIES, TERNARY, check, explore, start_state


VALUES = (-3, 0, 3)
PAIRS = tuple(itertools.product(VALUES, VALUES))      # block and core order: lexicographic, -3 < 0 < 3
C0 = (0, 0)
ZERO_BLOCK = (0, 0)
FULL_MASK = (1 << 9) - 1
SATS = ('A', 'B')
SAT_COORD = {'A': A, 'B': B}
FULL, INPUT_FROZEN, NF = 'full', 'input_frozen', 'nf'
K_CLAMPS = tuple(('k', v) for v in TERNARY)
MODES = (FULL, INPUT_FROZEN, NF) + K_CLAMPS
OUTCOMES = ('assembly_interference', 'no_transfer_core', 'no_core_dependent_witness', 'route_unresolved_only',
            'qualifying')


def index(w):
    """Index of a seven-coefficient vector, consistent with shared_input_mediation.vector."""
    return sum(VALUES.index(v) * 3 ** (6 - j) for j, v in enumerate(check(w)))


def assemble(pa, pb, c, r):
    return check((pa[0], pa[1], pb[0], pb[1], c[0], c[1], r))


def parts(w):
    """(p_A, p_B, c, r) of a seven-coefficient vector."""
    return (w[0], w[1]), (w[2], w[3]), (w[4], w[5]), w[6]


def waking_vector(sat, p, r):
    """Core disconnected and the other satellite's input couplings zeroed; r kept."""
    return assemble(p, ZERO_BLOCK, C0, r) if sat == 'A' else assemble(ZERO_BLOCK, p, C0, r)


def mirror_core(c):
    return (-c[0], -c[1])


def sign_class(c):
    return min(c, mirror_core(c))


def potential(c, a, b):
    """min over k of the core energy k^2 - k(c_A a + c_B b), by enumeration."""
    return min(k * k - k * (c[0] * a + c[1] * b) for k in TERNARY)


def potential_closed_form(c, a, b):
    return min(0, 1 - abs(c[0] * a + c[1] * b))


def assignment(train, swapped=False):
    """Ascending training IDs alternate A, B, A, ... (8 and 7); swapped gives A the odd positions."""
    s = sorted(train)
    even, odd = s[0::2], s[1::2]
    return (odd, even) if swapped else (even, odd)


def simulate(x, w, mode=FULL):
    """Complete response set Y of (a, b) pairs under one declared intervention."""
    if mode == FULL:
        r = explore(x, w)
    elif mode == INPUT_FROZEN:                     # the canonical start already has inputs at evidence
        r = explore(x, w, frozen=(I1, I2))
    elif mode == NF:
        r = explore(x, w, nf=True)
    elif mode in K_CLAMPS:
        r = explore(x, w, start=start_state(x, ((K, mode[1]),)), frozen=(K,))
    else:
        raise ValueError(f'unknown mode {mode}')
    if r.status != 'complete':
        raise AssertionError('uncapped exploration must complete')
    return r.Y


class Responses:
    """Memoized pure simulation: (query index, vector, mode) -> Y, and satellite marginals."""
    def __init__(self):
        self.memo = {}
        self.marg = {}
        self.requests = 0

    def Y(self, qi, w, mode=FULL):
        self.requests += 1
        key = (qi, w, mode)
        y = self.memo.get(key)
        if y is None:
            y = self.memo[key] = simulate(QUERIES[qi], w, mode)
        return y

    def marginal(self, qi, w, sat, mode=FULL):
        key = (qi, w, mode, sat)
        m = self.marg.get(key)
        if m is None:
            c = 0 if sat == 'A' else 1
            m = self.marg[key] = frozenset(y[c] for y in self.Y(qi, w, mode))
        else:
            self.requests += 1
        return m


class WakeFilter:
    """Nine-bit candidate mask for one satellite. Inputs: its own query and marginal only."""
    def __init__(self, sat, r, resp):
        if sat not in SATS:
            raise ValueError(sat)
        self.sat, self.r, self.resp = sat, r, resp
        self.mask = FULL_MASK
        self.evaluations = 0

    def update(self, qi, target):
        keep = 0
        for i, p in enumerate(PAIRS):
            if self.mask >> i & 1:
                self.evaluations += 1
                if self.resp.marginal(qi, waking_vector(self.sat, p, self.r), self.sat) == target:
                    keep |= 1 << i
        self.mask = keep

    def blocks(self):
        return blocks(self.mask)


def blocks(mask):
    return [p for i, p in enumerate(PAIRS) if mask >> i & 1]


def wake_targets(resp, teacher, ids_a, ids_b):
    """Teacher's waking marginals: {(sat, qi): marginal} for each satellite's own queries."""
    pa, pb, _, r = parts(teacher)
    out = {('A', qi): resp.marginal(qi, waking_vector('A', pa, r), 'A') for qi in ids_a}
    out.update({('B', qi): resp.marginal(qi, waking_vector('B', pb, r), 'B') for qi in ids_b})
    return out


def acquire(resp, r, order, ids_a, wake):
    """Run both filters over `order` (training IDs); each sees only its own assigned items."""
    fa, fb = WakeFilter('A', r, resp), WakeFilter('B', r, resp)
    for qi in order:
        if qi in ids_a:
            fa.update(qi, wake[('A', qi)])
        else:
            fb.update(qi, wake[('B', qi)])
    return fa, fb


# ---------------------------------------------------------------- evaluator-only assessment

def assess(resp, run, teacher, pa, pb, detail=True):
    """Stage-A record of one assembly against one teacher's targets (contract section 5.1).

    run: dict(train, mixed, within: query IDs; wake: {(sat, qi): waking target}; r).
    With detail=False, returns only the outcome and qualifying/route-unresolved cores, and stops
    early on assembly interference.
    """
    r = run['r']
    ws = {c: assemble(pa, pb, c, r) for c in PAIRS}

    def m(c, s, qi, mode=FULL):
        return resp.marginal(qi, ws[c], s, mode)

    def t(s, qi):
        return resp.marginal(qi, teacher, s)

    def retained(c):
        return all(m(c, s, qi) == f for (s, qi), f in run['wake'].items())

    r0 = retained(C0)
    if not r0 and not detail:
        return dict(outcome='assembly_interference', qualifying=[], route_unresolved=[])
    tc0 = [(s, qi) for qi in run['train'] for s in SATS if m(C0, s, qi) == t(s, qi)]

    def correct(c, s, ids):
        return frozenset(qi for qi in ids if m(c, s, qi) == t(s, qi))

    base = {(s, part): correct(C0, s, run[part]) for s in SATS for part in ('mixed', 'within')}
    cores = {}
    for c in PAIRS:
        retain = r0 if c == C0 else retained(c)
        preserve = retain and all(m(c, s, qi) == t(s, qi) for s, qi in tc0)
        if not detail and not (r0 and preserve and c[0] and c[1]):
            continue                            # cannot qualify (section 5.1: both links needed)
        sat_rec, recipients = {}, []
        for s in SATS:
            now = correct(c, s, run['mixed'])
            gains, losses = sorted(now - base[(s, 'mixed')]), sorted(base[(s, 'mixed')] - now)
            rec = dict(correct=len(now), gains=gains, losses=losses, net=len(gains) - len(losses))
            if detail:
                wn = correct(c, s, run['within'])
                rec['within'] = dict(correct=len(wn), gains=sorted(wn - base[(s, 'within')]),
                                     losses=sorted(base[(s, 'within')] - wn))
            sat_rec[s] = rec
            if gains and rec['net'] > 0:
                recipients.append(s)
        transfer = c != C0 and bool(recipients)
        dep, route = [], []
        if transfer:
            for s in recipients:
                for qi in sat_rec[s]['gains']:
                    cuts = (assemble(pa, pb, (0, c[1]), r), assemble(pa, pb, (c[0], 0), r))
                    if all(resp.marginal(qi, u, s) != t(s, qi) for u in cuts):
                        dep.append([s, qi])
                        if m(c, s, qi, INPUT_FROZEN) != m(C0, s, qi, INPUT_FROZEN):
                            route.append([s, qi])
        ok = r0 and preserve and transfer
        tier = 'qualifying' if ok and route else 'route_unresolved' if ok and dep else None
        rec = dict(retained=retain, preserved=preserve, transfer=transfer, recipients=recipients,
                   core_dependent_witnesses=dep, route_specific_witnesses=route, tier=tier)
        if detail:
            rec['satellites'] = sat_rec
            rec['joint_mixed_correct'] = sum(resp.Y(qi, ws[c]) == resp.Y(qi, teacher) for qi in run['mixed'])
        cores[c] = rec
    qualifying = [list(c) for c in PAIRS if c in cores and cores[c]['tier'] == 'qualifying']
    unresolved = [list(c) for c in PAIRS if c in cores and cores[c]['tier'] == 'route_unresolved']
    if not r0:
        outcome = 'assembly_interference'
    elif qualifying:
        outcome = 'qualifying'
    elif unresolved:
        outcome = 'route_unresolved_only'
    elif not detail:
        outcome = 'not_qualifying'              # cores were skipped; finer outcomes need detail=True
    elif any(cores[c]['preserved'] and cores[c]['transfer'] for c in PAIRS):
        outcome = 'no_core_dependent_witness'
    else:
        outcome = 'no_transfer_core'
    out = dict(outcome=outcome, qualifying=qualifying, route_unresolved=unresolved)
    if detail:
        out.update(retained_at_c0=r0, teacher_correct_training_at_c0=[[s, qi] for s, qi in tc0],
                   no_preserving_core=not any(cores[c]['preserved'] for c in PAIRS if c != C0),
                   transfer_without_preservation=[list(c) for c in PAIRS
                                                  if cores[c]['transfer'] and not cores[c]['preserved']],
                   cores={f'{c[0]},{c[1]}': rec for c, rec in cores.items()})
    return out


def witness_controls(resp, run, teacher, pa, pb, c, s, qi):
    """Per-witness diagnostics for learned core c versus c0 (contract section 6)."""
    r = run['r']
    target = resp.marginal(qi, teacher, s)

    def entry(w, mode=FULL):
        mg = resp.marginal(qi, w, s, mode)
        return dict(marginal=sorted(mg), teacher_correct=mg == target)
    out = {}
    for name, core in (('learned', c), ('initial', C0)):
        w = assemble(pa, pb, core, r)
        out[name] = dict(full=entry(w), input_frozen=entry(w, INPUT_FROZEN), nf=entry(w, NF),
                         **{f'k_clamped_{v}': entry(w, ('k', v)) for _, v in K_CLAMPS},
                         ablate_A=entry(assemble(ZERO_BLOCK, pb, core, r)),
                         ablate_B=entry(assemble(pa, ZERO_BLOCK, core, r)))
    out['cut_c_A'] = entry(assemble(pa, pb, (0, c[1]), r))
    out['cut_c_B'] = entry(assemble(pa, pb, (c[0], 0), r))
    out['contrasts'] = {k: out['learned'][k]['marginal'] != out['initial'][k]['marginal']
                        for k in ('full', 'input_frozen', 'nf') + tuple(f'k_clamped_{v}' for _, v in K_CLAMPS)}
    return out


__all__ = ['C0', 'FULL', 'FULL_MASK', 'INPUT_FROZEN', 'K_CLAMPS', 'MODES', 'NF', 'OUTCOMES', 'PAIRS', 'Responses',
           'SATS', 'WakeFilter', 'acquire', 'assemble', 'assess', 'assignment', 'blocks', 'index', 'mirror_core',
           'parts', 'potential', 'potential_closed_form', 'sign_class', 'simulate', 'wake_targets',
           'waking_vector', 'witness_controls']
