"""Taught truth v1 statements (docs/taught-truth-v1.md) on v3's everyday world.

A statement is (a, r, b, v) with one missing slot: v (the value: T, F, U for individuals; always, usually, sometimes,
never for kinds; T, F, U for connectives) or b (fill the tail, for true individual facts). Values are entities, so
connective statements (T, and, U) use them as arguments. Deterministic.
"""
from dataclasses import dataclass, field

import numpy as np

from src import everyday_world_v3 as W3

INDIVIDUAL_VALUES = ('T', 'F', 'U')
KIND_VALUES = ('always', 'usually', 'sometimes', 'never')
VALUES = INDIVIDUAL_VALUES + KIND_VALUES
CONNECTIVES = ('not', 'and', 'or')
KIND_RELS = {'has': W3.W1.PARTS, 'can': W3.W1.ABILITIES, 'lives-in': W3.W1.HOMES, 'eats': W3.W1.FOODS}
DEFAULT = {'always': 'T', 'usually': 'T', 'sometimes': 'U', 'never': 'F'}
ANON_PER_CLASS = 2
ANON_HELD = ('reptile', 'amphibian')      # S3b
S4A_ANIMALS = ('rabbit', 'shark', 'owl', 'ant')  # colourless animals whose unknown colour rows are held out (S4a)

# K3 (Kleene strong) tables; order T > U > F
_RANK = {'F': 0, 'U': 1, 'T': 2}
_INV = {0: 'F', 1: 'U', 2: 'T'}
NOT = {'T': 'F', 'F': 'T', 'U': 'U'}


def k3_and(a, b):
    return _INV[min(_RANK[a], _RANK[b])]


def k3_or(a, b):
    return _INV[max(_RANK[a], _RANK[b])]


def quantity(cls, rel, val):
    k = {'has': 1, 'can': 2, 'lives-in': 3, 'eats': 4}[rel]
    mem = [a for a in W3.ANIMALS if W3.ANIMALS[a][0] == cls]
    n = sum(val in W3.ANIMALS[a][k].split() for a in mem)
    if n == len(mem):
        return 'always'
    if n == 0:
        return 'never'
    return 'usually' if n > len(mem) / 2 else 'sometimes'


@dataclass
class Statements:
    entities: list
    relations: list
    a: np.ndarray = field(default=None)
    r: np.ndarray = field(default=None)
    b: np.ndarray = field(default=None)      # -1 when the tail is the missing slot
    v: np.ndarray = field(default=None)      # value index into VALUES, or -1 when the tail is missing
    gap: np.ndarray = field(default=None)    # 0 = value missing, 1 = tail missing
    kind: np.ndarray = field(default=None)   # 'individual', 'kind', 'anon', 'connective', 'fill'
    test: np.ndarray = field(default=None)
    sub: np.ndarray = field(default=None)
    world_y: np.ndarray = field(default=None)  # world truth for individual statements (1/0), -1 otherwise
    fill_answers: dict = field(default=None)   # fill row -> list of true tail indices

    def index(self, x):
        return self.entities.index(x)


def colourless_animals():
    return [a for a in W3.ANIMALS if a not in W3.W1.COLOURED]


def build():
    world = W3.build()
    classes = list(W3.CLASSES)
    anon = [f'{c}-{i + 1}' for c in classes for i in range(ANON_PER_CLASS)]
    ents = list(world.entities) + [x for x in VALUES + ('nil',) if x not in world.entities] + anon
    ents += [a for a in W3.ANIMALS if a not in ents]
    rels = list(world.relations) + list(CONNECTIVES)
    E = {x: i for i, x in enumerate(ents)}
    R = {x: i for i, x in enumerate(rels)}
    V = {x: i for i, x in enumerate(VALUES)}
    rows = []   # (a, r, b, v, gap, kind, test, sub, world_y)
    # individual statements from the v3 world
    for h, r, t, y, tst, sub in zip(world.h, world.r, world.t, world.y, world.test, world.subset):
        rows.append((E[world.entities[h]], R[world.relations[r]], E[world.entities[t]],
                     V['T' if y else 'F'], 0, 'individual', tst, sub, int(y)))
    # unknown colours for colourless animals
    s4a = [a for a in colourless_animals() if a in S4A_ANIMALS]
    for a in colourless_animals():
        for c in W3.W1.COLOURS:
            rows.append((E[a], R['colour-of'], E[c], V['U'], 0, 'individual', 'S4a' if a in s4a else '', 'unknown-colour', -1))
    # kind statements
    for ci, c in enumerate(classes):
        for rel, vals in KIND_RELS.items():
            for vi, val in enumerate(vals):
                q = quantity(c, rel, val)
                held = (ci + vi + len(rel)) % 4 == 0
                rows.append((E[c], R[rel], E[val], V[q], 0, 'kind', 'S2' if held else '', q, -1))
    # anonymous individuals: is-a facts and default values
    for c in classes:
        for i in range(ANON_PER_CLASS):
            x = f'{c}-{i + 1}'
            held = c in ANON_HELD
            for cl in classes:
                rows.append((E[x], R['is-a'], E[cl], V['T' if cl == c else 'F'], 0, 'anon-isa', '', '', -1))
            rows.append((E[x], R['is-a'], E['animal'], V['T'], 0, 'anon-isa', '', '', -1))
            for rel, vals in KIND_RELS.items():
                for val in vals:
                    q = quantity(c, rel, val)
                    rows.append((E[x], R[rel], E[val], V[DEFAULT[q]], 0, 'anon', 'S3b' if held else '', q, -1))
    # connectives (K3), all trained
    for x in INDIVIDUAL_VALUES:
        rows.append((E[x], R['not'], E['nil'], V[NOT[x]], 0, 'connective', '', 'not', -1))
        for y in INDIVIDUAL_VALUES:
            rows.append((E[x], R['and'], E[y], V[k3_and(x, y)], 0, 'connective', '', 'and', -1))
            rows.append((E[x], R['or'], E[y], V[k3_or(x, y)], 0, 'connective', '', 'or', -1))
    # fill-the-tail statements for trained true individual facts: (a, r, ?) -> all true tails
    fills = {}
    for (a, r, b, v, gap, kind, tst, sub, wy) in rows:
        if kind == 'individual' and tst == '' and v == V['T']:
            fills.setdefault((a, r), []).append(b)
    fill_rows = []
    for (a, r), bs in sorted(fills.items()):
        fill_rows.append((a, r, -1, -1, 1, 'fill', '', '', -1))
    rows += fill_rows
    a, r, b, v, gap, kind, tst, sub, wy = zip(*rows)
    st = Statements(ents, rels, np.array(a), np.array(r), np.array(b), np.array(v), np.array(gap), np.array(kind),
                    np.array(tst), np.array(sub), np.array(wy))
    n_before = len(rows) - len(fill_rows)
    st.fill_answers = {n_before + i: fills[(fa, fr)] for i, (fa, fr, *_rest) in enumerate(fill_rows)}
    return st, world


def default_expectations(cls):
    """For a new member of cls: (rel, value, expected individual value, quantity) over the kind relations."""
    out = []
    for rel, vals in KIND_RELS.items():
        for val in vals:
            q = quantity(cls, rel, val)
            out.append((rel, val, DEFAULT[q], q))
    return out
