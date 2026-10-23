"""Phase C of core mediation v1: unit-level modules interacting only through readouts."""
from collections import deque
from itertools import combinations, product

from src.core_mediation import MARGIN, energy as mode_energy, moves


SAT_CODE = ('0000', '1111')
CORE_CODES = {
    2: ('000000', '111111'),
    3: ('00000000', '11110000', '00001111'),
    4: ('0000000000', '1111100000', '0000011111', '1111111111'),
}
SEARCH_LIMIT = 2_000_000
OUTPUT_UNITS = 4


def distance_one_code(n):
    units = 2*n + 2
    return tuple('1'*j + '0'*(units-j) for j in range(n))


def bits(word):
    return sum(int(c) << i for i, c in enumerate(word))


class Module:
    """Hamming-prototype module: distance to nearest prototype and its unique readout."""
    def __init__(self, code):
        self.units = len(code[0])
        if any(len(w) != self.units for w in code):
            raise ValueError('prototypes must share a unit count')
        self.prototypes = tuple(bits(w) for w in code)
        self.distance, self.readout = [], []
        for s in range(1 << self.units):
            d = [bin(s ^ p).count('1') for p in self.prototypes]
            best = min(d)
            self.distance.append(best)
            self.readout.append(d.index(best) if d.count(best) == 1 else None)

    def min_separation(self):
        return min(bin(p ^ q).count('1') for p, q in combinations(self.prototypes, 2))


def term_values(cls, n, tables):
    """Per-term evaluators with the declared ⊥ rule: max attained value plus margin."""
    t = lambda *name: tables.get(name, 0)
    A, B, K = (0, 1), (0, 1), tuple(range(n))
    families = [(lambda a, b, k: t('uA', a), ('a',)), (lambda a, b, k: t('uB', b), ('b',)),
                (lambda a, b, k: t('uK', k), ('k',))]
    if cls in 'DF':
        families.append((lambda a, b, k: t('AB', a, b), ('a', 'b')))
    if cls in 'SF':
        families += [(lambda a, b, k: t('AK', a, k), ('a', 'k')),
                     (lambda a, b, k: t('BK', b, k), ('b', 'k'))]
    if cls == 'T':
        families.append((lambda a, b, k: t('ABK', a, b, k), ('a', 'b', 'k')))
    if cls == 'G':
        families += [(lambda a, b, k: t('wA')*(a != k), ('a', 'k')),
                     (lambda a, b, k: t('wB')*(b != k), ('b', 'k'))]
    out = []
    for f, args in families:
        top = max(f(a, b, k) for a, b, k in product(A, B, K)) + MARGIN
        out.append((f, args, top))
    return out


def readout_energy(cls, n, tables):
    """Coupling-plus-unary energy for every readout triple, including ⊥ (None)."""
    families = term_values(cls, n, tables)
    table = {}
    for a, b, k in product((0, 1, None), (0, 1, None), tuple(range(n)) + (None,)):
        value = dict(a=a, b=b, k=k)
        total = 0
        for f, args, top in families:
            if any(value[x] is None for x in args):
                total += top
            else:
                total += f(a, b, k)
        table[(a, b, k)] = total
    return table


class System:
    """Packed unit state: A bits, then B, then K, then optional output bits."""
    def __init__(self, cls, n, tables, lam, core_code=None, mu=None, pulse=None):
        self.cls, self.n, self.lam, self.mu = cls, n, lam, mu
        self.mods = (Module(SAT_CODE), Module(SAT_CODE), Module(core_code or CORE_CODES[n]))
        self.offsets = (0, 4, 8)
        self.units = 8 + self.mods[2].units + (OUTPUT_UNITS if mu else 0)
        self.coupling = readout_energy(cls, n, tables)
        self.pulse = pulse  # (h, e) bias on A only
        self.successor_cache = {}

    def split(self, s):
        a = s & 0xF
        b = (s >> 4) & 0xF
        k = (s >> 8) & ((1 << self.mods[2].units) - 1)
        o = s >> (8 + self.mods[2].units) if self.mu else None
        return a, b, k, o

    def readouts(self, s):
        a, b, k, _ = self.split(s)
        return self.mods[0].readout[a], self.mods[1].readout[b], self.mods[2].readout[k]

    def output_pattern(self, ra, rb):
        word, mismatch = 0, 0
        for i, r in enumerate((ra, rb)):
            if r is None:
                mismatch += 2
            elif r:
                word |= 0b11 << (2*i)
        return word, mismatch

    def energy(self, s):
        a, b, k, o = self.split(s)
        ra, rb, rk = self.mods[0].readout[a], self.mods[1].readout[b], self.mods[2].readout[k]
        e = self.lam*(self.mods[0].distance[a] + self.mods[1].distance[b]
                      + self.mods[2].distance[k]) + self.coupling[(ra, rb, rk)]
        if self.mu:
            word, mismatch = self.output_pattern(ra, rb)
            mask = 0
            for i, r in enumerate((ra, rb)):
                if r is not None:
                    mask |= 0b11 << (2*i)
            e += self.mu*(bin((o ^ word) & mask).count('1') + mismatch)
        if self.pulse:
            h, target = self.pulse
            e += h*bin(a ^ self.mods[0].prototypes[target]).count('1')
        return e

    def successors(self, s):
        if s not in self.successor_cache:
            e = self.energy(s)
            self.successor_cache[s] = tuple(
                t for t in (s ^ (1 << i) for i in range(self.units)) if self.energy(t) < e)
        return self.successor_cache[s]

    def explore(self, start):
        """All states reachable by strict descent under any schedule, and terminals."""
        seen, queue, terminals = {start}, deque([start]), set()
        while queue:
            s = queue.popleft()
            nxt = self.successors(s)
            if not nxt:
                terminals.add(s)
            for t in nxt:
                if t not in seen:
                    seen.add(t)
                    if len(seen) > SEARCH_LIMIT:
                        return None
                    queue.append(t)
        return seen, terminals

    def encode(self, x, output=True):
        a, b, k = x
        s = (self.mods[0].prototypes[a] | self.mods[1].prototypes[b] << 4
             | self.mods[2].prototypes[k] << 8)
        if self.mu and output:
            s |= self.output_pattern(a, b)[0] << (8 + self.mods[2].units)
        return s


def perturbations(system, s):
    """(label, module, state) for 1-/2-unit flips in one module and full prototype switches."""
    a, b, k, o = system.split(s)
    for m, (word, off) in enumerate(zip((a, b, k), system.offsets)):
        units = system.mods[m].units
        for i in range(units):
            yield 'flip1', m, s ^ (1 << (off+i))
        for i, j in combinations(range(units), 2):
            yield 'flip2', m, s ^ (1 << (off+i)) ^ (1 << (off+j))
        for p in system.mods[m].prototypes:
            if p != word:
                yield 'switch', m, (s & ~(((1 << units)-1) << off)) | (p << off)


def recovery(system, x):
    s0 = system.encode(x)
    r0 = system.readouts(s0)
    rows = []
    for label, m, s in perturbations(system, s0):
        result = system.explore(s)
        if result is None:
            rows.append(dict(kind=label, module=m, status='unresolved'))
            continue
        seen, terminals = result
        finals = {system.readouts(t) for t in terminals}
        other_changed = any(system.readouts(v)[j] != r0[j]
                            for v in seen for j in range(3) if j != m)
        rows.append(dict(kind=label, module=m, status='resolved',
                         readout_recovered=finals == {r0},
                         state_recovered=terminals == {s0},
                         final_readouts=sorted(finals, key=str),
                         other_module_readout_changed=other_changed,
                         visited=len(seen)))
    return rows


def mode_recovery(cls, n, tables, x):
    """Control (i): no internals; a single-module mode move followed by mode descent."""
    rows = []
    for y in moves(x, n):
        seen, queue, terminals = {y}, deque([y]), set()
        while queue:
            z = queue.popleft()
            e = mode_energy(cls, tables, z)
            nxt = [w for w in moves(z, n) if mode_energy(cls, tables, w) < e]
            if not nxt:
                terminals.add(z)
            for w in nxt:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        rows.append(dict(perturbed=list(y), recovered=terminals == {x},
                         final=sorted(map(list, terminals))))
    return rows


def pulse_response(cls, n, tables, lam, x, e, h, mu=None):
    """Pulse A toward mode e, release, and return every final readout."""
    driven = System(cls, n, tables, lam, mu=mu, pulse=(h, e))
    released = System(cls, n, tables, lam, mu=mu)
    start = released.encode(x)
    first = driven.explore(start)
    if first is None:
        return dict(status='unresolved')
    finals, driven_readouts = set(), set()
    for t in first[1]:
        driven_readouts.add(driven.readouts(t))
        second = released.explore(t)
        if second is None:
            return dict(status='unresolved')
        finals |= {released.readouts(u) for u in second[1]}
    return dict(status='resolved', driven=sorted(driven_readouts, key=str),
                final=sorted(finals, key=str))
