"""Graded core interface v1: multilinear coupling on units, two modes per module."""
from collections import deque
from itertools import product

from src.core_mediation import energy as mode_energy


UNITS = 4
SCALE = UNITS**3  # makes multilinear weights and internal terms integers
MODULES = ('A', 'B', 'K')


def popcount(word):
    return bin(word).count('1')


def readout(word):
    x = popcount(word)
    return None if 2*x == UNITS else int(2*x > UNITS)


class GradedSystem:
    """Packed state: A units 0-3, B units 4-7, K units 8-11 (absent if core removed)."""
    def __init__(self, cls, tables, sigma, lam, core=True, clamp=None, pulse=None):
        self.cls, self.sigma, self.lam = cls, sigma, lam
        self.core, self.clamp, self.pulse = core, clamp, pulse
        self.modules = 3 if core else 2
        self.free_units = 4*(2 if clamp is not None or not core else 3)
        # Mode energies at every vertex, already scaled by sigma.
        if core:
            self.vertex = {x: sigma*mode_energy(cls, tables, x) for x in product((0, 1), repeat=3)}
        else:
            self.vertex = {(a, b): sigma*mode_energy(cls, tables, (a, b, 0), core=False)
                           for a, b in product((0, 1), repeat=2)}
        self.cache = {}

    def barriers(self):
        """Per-module lambda: an int applies to all; a pair is (satellites, core)."""
        sat, core = (self.lam, self.lam) if isinstance(self.lam, int) else self.lam
        return (sat, sat, core)[:self.modules]

    def words(self, s):
        return tuple((s >> (4*m)) & 0xF for m in range(self.modules))

    def readouts(self, s):
        return tuple(readout(w) for w in self.words(s))

    def energy(self, s):
        """Exact integer energy times SCALE."""
        words = self.words(s)
        xs = [popcount(w) for w in words]
        internal = SCALE*sum(l*min(x, UNITS-x) for l, x in zip(self.barriers(), xs))
        coupling = 0
        for modes, value in self.vertex.items():
            weight = 1
            for mode, x in zip(modes, xs):
                weight *= x if mode else UNITS-x
            coupling += value*weight*(UNITS**(3-self.modules))
        e = internal + coupling
        if self.pulse:
            h, target = self.pulse
            e += SCALE*h*popcount(words[0] ^ (0xF if target else 0))
        return e

    def successors(self, s):
        if s not in self.cache:
            e = self.energy(s)
            self.cache[s] = tuple(t for t in (s ^ (1 << i) for i in range(self.free_units))
                                  if self.energy(t) < e)
        return self.cache[s]

    def explore(self, start):
        seen, queue, terminals = {start}, deque([start]), set()
        while queue:
            s = queue.popleft()
            nxt = self.successors(s)
            if not nxt:
                terminals.add(s)
            for t in nxt:
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
        return seen, terminals


def encode(modes):
    return sum((0xF if m else 0) << (4*i) for i, m in enumerate(modes))


def with_pulse(system, pulse):
    out = GradedSystem.__new__(GradedSystem)
    out.__dict__.update(system.__dict__, pulse=pulse, cache={})
    return out


def pulse_then_release(system, start_modes, e, h):
    """Every final readout after holding a pulse on A to termination, then releasing."""
    driven = with_pulse(system, (h, e))
    finals = set()
    for t in driven.explore(encode(start_modes))[1]:
        finals |= {system.readouts(u) for u in system.explore(t)[1]}
    return sorted(finals, key=str)


def perturbation_rows(system, x):
    s0 = encode(x)
    r0 = system.readouts(s0)
    rows = []
    for m in range(system.modules):
        off = 4*m
        flips = [('flip1', s0 ^ (1 << (off+i))) for i in range(4)]
        flips += [('flip2', s0 ^ (1 << (off+i)) ^ (1 << (off+j)))
                  for i in range(4) for j in range(i+1, 4)]
        flips += [('switch', s0 ^ (0xF << off))]
        for kind, s in flips:
            seen, terminals = system.explore(s)
            rows.append(dict(kind=kind, module=MODULES[m],
                             state_recovered=terminals == {s0},
                             readout_recovered={system.readouts(t) for t in terminals} == {r0},
                             other_module_readout_changed=any(
                                 system.readouts(v)[j] != r0[j] for v in seen
                                 for j in range(system.modules) if j != m)))
    return rows


def strict_minimum(system, s):
    """Every single flip raises energy; equal-energy neighbours do not pass."""
    e = system.energy(s)
    return all(system.energy(s ^ (1 << i)) > e for i in range(system.free_units))


def census(system, targets):
    """Global enumeration of all unit states: terminals and schedule-independent basins.

    A state commits to readout r if every terminal reachable from it under every
    schedule reads r; otherwise it is ambiguous.
    """
    states = range(1 << system.free_units)
    terminals = [s for s in states if not system.successors(s)]
    spurious = {}
    for s in terminals:
        r = system.readouts(s)
        if not (tuple(r) in targets and s == encode(r)):
            key = '/'.join(map(str, (popcount(w) for w in system.words(s)))) + ' reads ' + str(r)
            spurious[key] = spurious.get(key, 0) + 1
    basins = {}
    for s in states:
        finals = {system.readouts(t) for t in system.explore(s)[1]}
        key = str(next(iter(finals))) if len(finals) == 1 else 'ambiguous'
        basins[key] = basins.get(key, 0) + 1
    return dict(states=len(states), terminals=len(terminals),
                spurious_terminals=sum(spurious.values()),
                spurious_by_popcount_class=dict(sorted(spurious.items())),
                committed_to_target=sum(basins.get(str(tuple(x)), 0) for x in targets),
                basins=dict(sorted(basins.items())))


def evaluate(cls, tables, targets, sigma, lam, h, with_census=True):
    """Criteria P1-P5 for one parameter setting, plus phase D locality records."""
    full = GradedSystem(cls, tables, sigma, lam)
    removed = GradedSystem(cls, tables, sigma, lam, core=False)
    p1 = all(strict_minimum(full, encode(x)) for x in targets)
    rows = [dict(r, target=list(x)) for x in targets for r in perturbation_rows(full, x)]
    p2 = all(r['state_recovered'] for r in rows if r['kind'] == 'flip1')
    pulses = []
    for x in targets:
        for e in (0, 1):
            if e == x[0]:
                continue
            finals = pulse_then_release(full, x, e, h)
            y = finals[0] if len(finals) == 1 else None
            p3 = y is not None and tuple(y) in targets and y[0] == e and y[2] != x[2]
            no_core = pulse_then_release(removed, x[:2], e, h)
            clamped_system = GradedSystem(cls, tables, sigma, lam, clamp=x[2])
            clamped = pulse_then_release(clamped_system, x, e, h)
            p4 = p3 and all(tuple(f[:2]) != tuple(y[:2]) for f in no_core)
            p5 = p3 and all(tuple(f[:2]) != tuple(y[:2]) for f in clamped)
            pulses.append(dict(start=list(x), e=e, final=[list(f) for f in finals],
                               core_removed=[list(f) for f in no_core],
                               core_clamped=[list(f) for f in clamped],
                               P3=p3, P4=p4, P5=p5))
    counts = {}
    for r in rows:
        key = f"{r['kind']}/{r['module']}"
        c = counts.setdefault(key, dict(trials=0, state_recovered=0, readout_recovered=0,
                                        other_module_readout_changed=0))
        c['trials'] += 1
        for f in ('state_recovered', 'readout_recovered', 'other_module_readout_changed'):
            c[f] += r[f]
    return dict(sigma=sigma, lam=lam, h=h, P1=p1, P2=p2,
                joint=p1 and p2 and any(p['P3'] and p['P4'] and p['P5'] for p in pulses),
                pulses=pulses, perturbations=counts,
                census=census(full, targets) if with_census else None)
