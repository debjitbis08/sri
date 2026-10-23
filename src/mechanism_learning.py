"""Learned input mediation v1: intervention battery, chain atoms and learners.

See docs/learned-input-mediation-v1-contract.md. Dynamics, controls and interventions are
imported unchanged from src/shared_input_mediation.py. Hypotheses are the 2,187 seven-coefficient
vectors. The battery is evaluator data; learners see only typed training feedback maps and a
simulation service restricted to their arm's observation types.

Encodings: a set of (a, b) pairs is a 9-bit mask over index 3(a+1)+(b+1); a set of (i1, i2, s)
triples is a 27-bit mask over 9(i1+1)+3(i2+1)+(s+1); Booleans are 0/1.
"""
import hashlib
import itertools
import json

from src.shared_input_mediation import (
    A, B, COEFFS, I1, I2, N_VECTORS, QUERIES, Query, core_disconnected, explore, start_state, vector, zeroed,
)


TERNARY = (-1, 0, 1)
COEFF_INDEX = {c: j for j, c in enumerate(COEFFS)}
COORD = {'A': A, 'B': B, 'i1': I1, 'i2': I2}
TEMPLATES = {
    'SF': dict(source=A, target=B, donor=I1, recipient=I1, links=('w_1A', 'w_1B')),
    'SR': dict(source=B, target=A, donor=I1, recipient=I1, links=('w_1B', 'w_1A')),
    'CF': dict(source=A, target=B, donor=I1, recipient=I2, links=('w_1A', 'w_12', 'w_2B')),
    'CR': dict(source=B, target=A, donor=I2, recipient=I1, links=('w_2B', 'w_12', 'w_1A')),
}
FORWARD, REVERSE = ('SF', 'CF'), ('SR', 'CR')
MECHANISM_ITEMS = ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8E', 'B8W')


def pair_mask(Y):
    m = 0
    for a, b in Y:                      # OR, not sum: several terminals can share a projection
        m |= 1 << (3 * (a + 1) + (b + 1))
    return m


def triple_mask(triples):
    m = 0
    for i, j, s in triples:
        m |= 1 << (9 * (i + 1) + 3 * (j + 1) + (s + 1))
    return m


def unpair(m):
    return sorted([a, b] for a in TERNARY for b in TERNARY if m >> (3 * (a + 1) + (b + 1)) & 1)


def untriple(m):
    return sorted([i, j, s] for i in TERNARY for j in TERNARY for s in TERNARY
                  if m >> (9 * (i + 1) + 3 * (j + 1) + (s + 1)) & 1)


def popcount(m):
    return bin(m).count('1')


class Sim:
    """Memoized pure exploration for one hypothesis: (query, w, clamps, nf) -> terminal states."""
    def __init__(self):
        self.memo = {}
        self.requests = 0

    def terminals(self, x, w, clamps=(), nf=False):
        self.requests += 1
        key = (x, w, clamps, nf)
        if key not in self.memo:
            r = explore(x, w, start=start_state(x, clamps), frozen=tuple(c for c, _ in clamps), nf=nf)
            self.memo[key] = r.terminals
        return self.memo[key]


def proj(terms, coord):
    return frozenset(t[coord] for t in terms)


def chain_tuples(sim, x, w, template):
    """Exhaustive qualifying tuples (v, v2, r, r2) of 8(a)-(d) for one hypothesis, query and template."""
    t = TEMPLATES[template]
    src, tgt, j, l = t['source'], t['target'], t['donor'], t['recipient']
    e_clamp = ((I1, x.e1), (I2, x.e2))
    out = []
    for v in x.allowed(src):
        for v2 in x.allowed(src):
            if v == v2:
                continue
            Tv, Tv2 = sim.terminals(x, w, ((src, v),)), sim.terminals(x, w, ((src, v2),))
            if proj(Tv, tgt) == proj(Tv2, tgt):                                     # (c) full contrast
                continue
            if proj(sim.terminals(x, w, ((src, v),), True), tgt) != proj(sim.terminals(x, w, ((src, v2),), True), tgt):
                continue                                                           # (c) vanishes under NF
            if proj(sim.terminals(x, w, ((src, v),) + e_clamp), tgt) != proj(sim.terminals(x, w, ((src, v2),) + e_clamp), tgt):
                continue                                                           # (c) vanishes at I = e
            if proj(Tv, j) == proj(Tv2, j) or proj(Tv, l) == proj(Tv2, l):         # (a)
                continue
            if any(proj(sim.terminals(x, zeroed(w, link), ((src, v),)), tgt)
                   != proj(sim.terminals(x, zeroed(w, link), ((src, v2),)), tgt) for link in t['links']):
                continue                                                           # (d)
            Ul, Ul2 = proj(Tv, l), proj(Tv2, l)
            for r in TERNARY:
                for r2 in TERNARY:
                    if r == r2 or r not in Ul or r2 not in Ul2:                     # absent origin scores false
                        continue
                    if proj(sim.terminals(x, w, ((src, v), (l, r))), tgt) != proj(sim.terminals(x, w, ((src, v), (l, r2))), tgt):
                        out.append((v, v2, r, r2))                                  # (b)
    return out


def tuple_qualifies(x, w, template, v, v2, r, r2):
    """B8 for one explicit tuple; domain-invalid arguments are rejected."""
    src = TEMPLATES[template]['source']
    if v not in x.allowed(src) or v2 not in x.allowed(src) or v == v2 or r not in TERNARY or r2 not in TERNARY or r == r2:
        raise ValueError('tuple arguments outside their domains')
    return (v, v2, r, r2) in chain_tuples(Sim(), x, w, template)


def base_keys():
    """Battery atom keys other than the fixed-witness atoms, in a fixed order."""
    keys = []
    for qi, x in enumerate(QUERIES):
        keys += [('B1', qi), ('B2', qi), ('B3', qi), ('B4', qi)]
        keys += [('B5', c, qi) for c in COEFFS]
        keys += [('B6', qi, v) for v in x.allowed(A)]
        keys += [('B7', qi, v) for v in x.allowed(B)]
        keys += [('B8E', t, qi) for t in TEMPLATES]
    return keys


BASE_KEYS = base_keys()


def compute_hypothesis(h):
    """Base atom values and exhaustive qualifying tuples for hypothesis index h."""
    w = vector(h)
    sim = Sim()
    values, tuples = [], {}
    for qi, x in enumerate(QUERIES):
        canon = sim.terminals(x, w)
        values.append(pair_mask((t[A], t[B]) for t in canon))
        values.append(pair_mask((t[A], t[B]) for t in sim.terminals(x, w, (), True)))
        values.append(pair_mask((t[A], t[B]) for t in sim.terminals(x, w, ((I1, x.e1), (I2, x.e2)))))
        values.append(pair_mask((t[A], t[B]) for t in sim.terminals(x, core_disconnected(w))))
        for c in COEFFS:
            values.append(pair_mask((t[A], t[B]) for t in sim.terminals(x, zeroed(w, c))))
        for v in x.allowed(A):
            values.append(triple_mask((t[I1], t[I2], t[B]) for t in sim.terminals(x, w, ((A, v),))))
        for v in x.allowed(B):
            values.append(triple_mask((t[I1], t[I2], t[A]) for t in sim.terminals(x, w, ((B, v),))))
        for tmpl in TEMPLATES:
            found = chain_tuples(sim, x, w, tmpl)
            values.append(int(bool(found)))
            for tup in found:
                tuples.setdefault(tmpl, []).append((qi,) + tup)
    return values, tuples, dict(requests=sim.requests, unique=len(sim.memo))


def witness_keys(teacher_tuples):
    """Fixed-witness atom keys: the union of all teachers' qualifying tuples, sorted."""
    keys = set()
    for tuples in teacher_tuples:
        for tmpl, lst in tuples.items():
            keys |= {('B8W', tmpl) + tup for tup in lst}
    return sorted(keys)


def atom_vector(values, tuples, wkeys):
    present = {('B8W', tmpl) + tup for tmpl, lst in tuples.items() for tup in lst}
    return tuple(values) + tuple(int(k in present) for k in wkeys)


def reconstruct(h, wkeys):
    """Deterministic reconstruction interface for audit of per-hypothesis hashes."""
    values, tuples, _ = compute_hypothesis(h)
    return atom_vector(values, tuples, wkeys)


def atom_hash(vec):
    return hashlib.sha256(json.dumps(list(vec), separators=(',', ':')).encode()).hexdigest()


def decode(key, value):
    if key[0] in ('B1', 'B2', 'B3', 'B4', 'B5'):
        return unpair(value)
    if key[0] in ('B6', 'B7'):
        return untriple(value)
    return bool(value)


# ---------------------------------------------------------------- observations and learners

def observation_keys(x, arm):
    """Typed feedback keys for one training query: ('B1',) and, in arm I, ('B6', v) and ('B7', v)."""
    keys = [('B1',)]
    if arm == 'I':
        keys += [('B6', v) for v in x.allowed(A)] + [('B7', v) for v in x.allowed(B)]
    return keys


class Simulator:
    """Simulation service for learners: (candidate, current training query, supplied key) only.

    Pure memoization; it holds no teacher labels, battery, held-out results or B8 schema.
    """
    def __init__(self, arm):
        self.arm = arm
        self.memo = {}
        self.requests = 0

    def predict(self, w, x, key):
        if key not in observation_keys(x, self.arm):
            raise PermissionError(f'observation key {key} not supplied in arm {self.arm}')
        self.requests += 1
        mk = (w, x, key)
        if mk not in self.memo:
            if key[0] == 'B1':
                terms = explore(x, w).terminals
                self.memo[mk] = frozenset((t[A], t[B]) for t in terms)
            else:
                c, other = (A, B) if key[0] == 'B6' else (B, A)
                terms = explore(x, w, start=start_state(x, ((c, key[1]),)), frozen=(c,)).terminals
                self.memo[mk] = frozenset((t[I1], t[I2], t[other]) for t in terms)
        return self.memo[mk]


ALL_BITS = (1 << N_VECTORS) - 1


class VersionSpace:
    """2,187-bit version space. Full: equality on every key; positive-only: containment on every key."""
    def __init__(self, sim, positive_only=False):
        self.sim, self.positive_only = sim, positive_only
        self.bits, self.failed = ALL_BITS, False
        self.evaluations = 0

    def update(self, x, feedback):
        if not isinstance(x, Query):
            raise TypeError('x must be a Query')
        if self.failed:
            return 'inconsistent_feedback'
        nxt = 0
        for h in range(N_VECTORS):
            if not self.bits >> h & 1:
                continue
            w = vector(h)
            ok = True
            for key, f in feedback.items():
                y = self.sim.predict(w, x, key)
                self.evaluations += 1
                if not ((f <= y) if self.positive_only else (y == f)):
                    ok = False
                    break
            if ok:
                nxt |= 1 << h
        if not nxt:
            self.failed = True
            return 'inconsistent_feedback'
        self.bits = nxt
        return 'updated'


BLOCKS = ((0, 1), (2, 3), (4, 5), (6,))


class BlockLearner:
    """Seven coefficients, one block edit per experience, strict improvement on summed set discrepancy."""
    def __init__(self, sim):
        self.sim = sim
        self.w = (0,) * 7
        self.evaluations = self.commits = self.coefficients_changed = 0

    def loss(self, w, x, feedback):
        total = 0
        for key, f in feedback.items():
            total += len(self.sim.predict(w, x, key) ^ f)
            self.evaluations += 1
        return total

    def neighbors(self):
        out = []
        for block in BLOCKS:
            for vals in itertools.product((-3, 0, 3), repeat=len(block)):
                cand = list(self.w)
                for j, v in zip(block, vals):
                    cand[j] = v
                if tuple(cand) != self.w:
                    out.append(tuple(cand))
        return out

    def update(self, x, feedback):
        current = self.loss(self.w, x, feedback)
        cands = [(current, 0, self.w)] + [
            (self.loss(c, x, feedback), sum(a != b for a, b in zip(c, self.w)), c) for c in self.neighbors()]
        best = min(cands)
        if best[0] < current:
            self.w = best[2]
            self.commits += 1
            self.coefficients_changed += best[1]
            return 'updated'
        return 'unchanged'


class Memory:
    """Answers only (query, key) combinations it has observed; everything else is unknown (None)."""
    def __init__(self):
        self.table = {}

    def update(self, x, feedback):
        for key, f in feedback.items():
            self.table[(x, key)] = f
        return 'updated'

    def predict(self, x, key):
        return self.table.get((x, key))


def hypothesis_index(w):
    return sum((v // 3 + 1) * 3 ** (6 - j) for j, v in enumerate(w))
