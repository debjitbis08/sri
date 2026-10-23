"""Phase B of core mediation v1: exact mode-level realizability, not an architecture."""
from fractions import Fraction
from itertools import combinations, permutations, product
from math import prod

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


BOUND = 16
MARGIN = 1
BIG = 256
BRANCH_LIMIT = 4096
CLASSES = ('U', 'D', 'S', 'F', 'T', 'G')


def parameters(cls, n):
    if cls not in CLASSES or n not in (2, 3, 4) or (cls == 'G' and n != 2):
        raise ValueError('outside the declared class/core-size domain')
    names = ([('uA', a) for a in (0, 1)] + [('uB', b) for b in (0, 1)]
             + [('uK', k) for k in range(n)])
    if cls in 'DF':
        names += [('AB', a, b) for a in (0, 1) for b in (0, 1)]
    if cls in 'SF':
        names += [('AK', a, k) for a in (0, 1) for k in range(n)]
        names += [('BK', b, k) for b in (0, 1) for k in range(n)]
    if cls == 'T':
        names += [('ABK', a, b, k) for a in (0, 1) for b in (0, 1) for k in range(n)]
    if cls == 'G':
        names += [('wA',), ('wB',)]
    return tuple(names)


def terms(cls, x):
    """Energy terms of x tagged U (kept without K), K (core unary) or C (scaled)."""
    a, b, k = x
    out = [('U', ('uA', a)), ('U', ('uB', b)), ('K', ('uK', k))]
    if cls in 'DF':
        out.append(('U', ('AB', a, b)))
    if cls in 'SF':
        out += [('C', ('AK', a, k)), ('C', ('BK', b, k))]
    if cls == 'T':
        out.append(('C', ('ABK', a, b, k)))
    if cls == 'G':
        out += [('C', ('wA',))]*(a != k) + [('C', ('wB',))]*(b != k)
    return out


def configurations(n):
    return tuple(product((0, 1), (0, 1), range(n)))


def moves(x, n, satellites_only=False):
    a, b, k = x
    out = [(1-a, b, k), (a, 1-b, k)]
    if not satellites_only:
        out += [(a, b, j) for j in range(n) if j != k]
    return out


class Encoder:
    def __init__(self, cls, n):
        self.cls, self.n = cls, n
        self.names = parameters(cls, n)
        self.index = {name: i for i, name in enumerate(self.names)}

    def vector(self, x, tags=('U', 'K', 'C')):
        v = np.zeros(len(self.names))
        for tag, name in terms(self.cls, x):
            if tag in tags:
                v[self.index[name]] += 1
        return v

    def delta(self, x, y, tags=('U', 'K', 'C')):
        return self.vector(y, tags) - self.vector(x, tags)


def joint_dependent(targets):
    def function_of(position):
        seen = {}
        return all(seen.setdefault(x[position], x[2]) == x[2] for x in targets)
    return not function_of(0) and not function_of(1)


def target_sets(n, level):
    configs = configurations(n)
    sizes = (2,) if level == 1 else (3, 4)
    for size in sizes:
        for chosen in combinations(configs, size):
            if len({x[:2] for x in chosen}) < size:
                continue
            if level == 2 and not joint_dependent(chosen):
                continue
            yield chosen


def symmetry_group(cls, n):
    """Relabelings preserving each class's family of tables, as maps on configurations."""
    group = []
    if cls == 'G':
        for swap, flip in product((0, 1), (0, 1)):
            group.append(lambda x, s=swap, f=flip: _act(x, s, f, f, tuple(
                (k ^ f) for k in range(n))))
    else:
        for swap, fa, fb in product((0, 1), (0, 1), (0, 1)):
            for perm in permutations(range(n)):
                group.append(lambda x, s=swap, p=fa, q=fb, r=perm: _act(x, s, p, q, r))
    return group


def _act(x, swap, flip_a, flip_b, perm):
    a, b, k = x
    a, b = a ^ flip_a, b ^ flip_b
    if swap:
        a, b = b, a
    return a, b, perm[k]


def orbits(cls, n, level):
    group = symmetry_group(cls, n)
    seen = {}
    for targets in target_sets(n, level):
        canonical = min(tuple(sorted(g(x) for x in targets)) for g in group)
        seen.setdefault(canonical, 0)
        seen[canonical] += 1
    return seen


def constraints(cls, n, targets, strict):
    """Hard rows (A theta <= b) and disjunctions of literal rows."""
    enc = Encoder(cls, n)
    hard, disjunctions = [], []
    for x in targets:
        for y in moves(x, n):
            hard.append((-enc.delta(x, y), -MARGIN))            # stable with margin
            hard.append((-enc.delta(x, y, ('C',)), 0))          # R3: Delta C >= 0
    disjunctions.append([(enc.delta(x, y, ('U',)), -MARGIN)     # R1 without K
                         for x in targets for y in moves(x, n, True)])
    for kc in range(n):                                         # R2 clamped core
        disjunctions.append([(enc.delta((x[0], x[1], kc), (y[0], y[1], kc)), -MARGIN)
                             for x in targets for y in moves(x, n, True)])
    if strict:                                                  # R4 exclusivity
        for z in configurations(n):
            if z not in targets:
                disjunctions.append([(enc.delta(z, y), -MARGIN) for y in moves(z, n)])
    return enc, hard, disjunctions


def solve(cls, n, targets, strict, time_limit=60.):
    enc, hard, disjunctions = constraints(cls, n, targets, strict)
    p = len(enc.names)
    literals = [lit for d in disjunctions for lit in d]
    q = len(literals)
    # variables: theta (p), |theta| bounds t (p), literal indicators z (q)
    rows, lo, hi = [], [], []
    for row, rhs in hard:
        rows.append(np.concatenate([row, np.zeros(p+q)])); lo.append(-np.inf); hi.append(rhs)
    for i, (row, rhs) in enumerate(literals):
        z = np.zeros(q); z[i] = BIG
        rows.append(np.concatenate([row, np.zeros(p), z])); lo.append(-np.inf); hi.append(rhs+BIG)
    start = 0
    for d in disjunctions:
        z = np.zeros(q); z[start:start+len(d)] = 1; start += len(d)
        rows.append(np.concatenate([np.zeros(2*p), z])); lo.append(1); hi.append(np.inf)
    for i in range(p):
        for sign in (1, -1):
            r = np.zeros(2*p+q); r[i], r[p+i] = sign, -1
            rows.append(r); lo.append(-np.inf); hi.append(0)
    cost = np.concatenate([np.zeros(p), np.ones(p), np.zeros(q)])
    result = milp(cost, constraints=LinearConstraint(np.array(rows), lo, hi),
                  integrality=np.ones(2*p+q),
                  bounds=Bounds([-BOUND]*p+[0]*p+[0]*q, [BOUND]*p+[BOUND]*p+[1]*q),
                  options=dict(time_limit=time_limit))
    base = dict(cls=cls, n=n, targets=[list(x) for x in targets], strict=strict,
                parameters=p, literals=q, disjunctions=len(disjunctions))
    if result.status == 0:
        theta = [int(round(v)) for v in result.x[:p]]
        tables = dict(zip(enc.names, theta))
        check = verify(cls, n, tables, targets, strict)
        if all(check.values()):
            return dict(base, outcome='realizable', witness=_serial(tables),
                        stored_entries=p, l1=sum(map(abs, theta)), checks=check)
        return dict(base, outcome='unresolved', reason='witness failed exact check',
                    checks=check)
    if result.status != 2:
        return dict(base, outcome='unresolved', reason=result.message)
    branches = prod(len(d) for d in disjunctions)
    if branches > BRANCH_LIMIT:
        return dict(base, outcome='bounded_infeasible', branches=branches,
                    reason='branch count exceeds certificate limit')
    for choice in product(*disjunctions):
        if not farkas(hard + list(choice), p):
            return dict(base, outcome='bounded_infeasible', branches=branches,
                        reason='a branch lacks an exact certificate')
    return dict(base, outcome='impossible', branches=branches, certified_branches=branches)


def farkas(rows, p):
    """Exact certificate y >= 0, A^T y = 0, b^T y < 0 that A theta <= b is infeasible."""
    A = np.array([r for r, _ in rows]); b = np.array([float(s) for _, s in rows])
    m = len(rows)
    lp = linprog(np.zeros(m), A_eq=np.vstack([A.T, b]), b_eq=np.concatenate([np.zeros(p), [-1]]),
                 bounds=[(0, None)]*m, method='highs')
    if lp.status != 0:
        return False
    support = [i for i in range(m) if lp.x[i] > 1e-9]
    y = _exact_solve([[Fraction(int(A[i, j])) for i in support] for j in range(p)]
                     + [[Fraction(int(b[i])) for i in support]],
                     [Fraction(0)]*p + [Fraction(-1)])
    if y is None or min(y) < 0:
        return False
    full = [Fraction(0)]*m
    for i, v in zip(support, y):
        full[i] = v
    ints = [[int(v) for v in r] for r, _ in rows]
    return (all(sum(full[i]*ints[i][j] for i in range(m)) == 0 for j in range(p))
            and sum(full[i]*int(rows[i][1]) for i in range(m)) < 0)


def _exact_solve(M, r):
    """Gaussian elimination over Fractions; free variables are set to zero."""
    rows = [list(row)+[v] for row, v in zip(M, r)]
    cols = len(M[0]) if M else 0
    pivots, rank = [], 0
    for c in range(cols):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        lead = rows[rank][c]
        rows[rank] = [v/lead for v in rows[rank]]
        for i in range(len(rows)):
            if i != rank and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [u-f*v for u, v in zip(rows[i], rows[rank])]
        pivots.append(c); rank += 1
    if any(all(v == 0 for v in row[:-1]) and row[-1] != 0 for row in rows):
        return None
    y = [Fraction(0)]*cols
    for i, c in enumerate(pivots):
        y[c] = rows[i][-1]
    return y


def _serial(tables):
    return {'.'.join(map(str, name)): v for name, v in tables.items()}


# Independent exact evaluator: direct table formulas, not the encoder above.

def energy(cls, tables, x, core=True, coupling_only=False):
    a, b, k = x
    t = lambda *name: tables.get(name, 0)
    coupling = 0
    if cls in 'SF':
        coupling += t('AK', a, k) + t('BK', b, k)
    if cls == 'T':
        coupling += t('ABK', a, b, k)
    if cls == 'G':
        coupling += t('wA')*(a != k) + t('wB')*(b != k)
    if coupling_only:
        return coupling
    local = t('uA', a) + t('uB', b) + t('AB', a, b)
    return local + (t('uK', k) + coupling if core else 0)


def status(cls, tables, x, n, core=True, satellites_only=False, coupling_only=False):
    deltas = [energy(cls, tables, y, core, coupling_only) - energy(cls, tables, x, core, coupling_only)
              for y in moves(x, n, satellites_only or not core)]
    if min(deltas) >= MARGIN:
        return 'stable'
    return 'unstable' if min(deltas) <= -MARGIN else 'marginal'


def verify(cls, n, tables, targets, strict):
    no_collapse = all(energy(cls, tables, y, coupling_only=True)
                      - energy(cls, tables, x, coupling_only=True) >= 0
                      for x in targets for y in moves(x, n))
    out = dict(
        targets_stable=all(status(cls, tables, x, n) == 'stable' for x in targets),
        R1_core_dependent=any(status(cls, tables, x, n, core=False) == 'unstable'
                              for x in targets),
        R2_not_bias=all(any(status(cls, tables, (x[0], x[1], kc), n, satellites_only=True)
                            == 'unstable' for x in targets) for kc in range(n)),
        R3_no_collapse=no_collapse,
        entries_bounded=all(abs(v) <= BOUND for v in tables.values()),
    )
    if strict:
        out['R4_exclusive'] = all(status(cls, tables, z, n) == 'unstable'
                                  for z in configurations(n) if z not in targets)
    return out


def stable_set(cls, n, tables, scale=1):
    """Stable configurations of U + scale*C, for reporting collapse behaviour."""
    scaled = {k: (v*scale if k[0] in ('AK', 'BK', 'ABK', 'wA', 'wB') else v)
              for k, v in tables.items()}
    return [x for x in configurations(n) if status(cls, scaled, x, n) == 'stable']
