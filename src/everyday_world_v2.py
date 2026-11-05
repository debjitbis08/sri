"""Everyday relations v2 world (docs/everyday-relations-v2.md): v1's world plus scales that give adjectives meaning, and
several relations per property (symmetric, inverse, transitive). The tables are the specification; deterministic."""
from dataclasses import dataclass, field

import numpy as np

from src import everyday_world as W1

ANIMALS = dict(W1.ANIMALS, cheetah=('mammal', 'fur tail legs4', 'run', 'land', 'meat'))
SOUNDS, BABIES, COLOURED = W1.SOUNDS, W1.BABIES, W1.COLOURED
CLASSES, PARTS, ABILITIES, HOMES, FOODS = W1.CLASSES, W1.PARTS, W1.ABILITIES, W1.HOMES, W1.FOODS
COLOURS, MIXES, WARM, COOL = W1.COLOURS, W1.MIXES, W1.WARM, W1.COOL

# dimension: (comparative, inverse comparative, things from the top of the scale down, pole 1, pole 2)
SCALES = {
    'size': ('bigger-than', 'smaller-than', W1.SIZE, 'big', 'small'),
    'temperature': ('hotter-than', 'colder-than', ('sun', 'fire', 'soup', 'tea', 'milk', 'water', 'ice', 'snow'), 'hot', 'cold'),
    'speed': ('faster-than', 'slower-than', ('cheetah', 'horse', 'dog', 'cat', 'duck', 'turtle', 'ant'), 'fast', 'slow'),
    'weight': ('heavier-than', 'lighter-than', ('whale', 'elephant', 'horse', 'dog', 'cat', 'mouse', 'feather'), 'heavy', 'light'),
    'loudness': ('louder-than', 'quieter-than', ('thunder', 'lion', 'dog', 'cat', 'bee', 'mouse', 'butterfly'), 'loud', 'quiet'),
    'brightness': ('brighter-than', 'darker-than', ('sun', 'snow', 'sky', 'grass', 'chocolate', 'coal', 'night-sky'), 'bright', 'dark'),
}
ADJECTIVES = tuple(a for s in SCALES.values() for a in s[3:])
ADJACENT_ONLY = ('temperature', 'loudness')   # P4
OPP_FROM_MEANING = ('speed', 'weight')         # P5

PREYS_ON = {'lion': 'cow horse rabbit', 'cheetah': 'rabbit', 'cat': 'mouse', 'owl': 'mouse', 'eagle': 'rabbit snake',
            'snake': 'mouse frog', 'frog': 'housefly bee', 'lizard': 'ant housefly', 'shark': 'tuna salmon',
            'crocodile': 'horse duck', 'robin': 'ant', 'toad': 'housefly'}

NOUN_WORDS, VERB_WORDS = W1.NOUN_WORDS, W1.VERB_WORDS
TREE_A, TREE_B, MALE, b, w = W1.TREE_A, W1.TREE_B, W1.MALE, W1.b, W1.w

SYMMETRIC = ('opposite', 'spouse-of', 'sibling-of', 'same-colour-as', 'same-class-as', 'lives-with')
INVERSE_PAIRS = (('parent-of', 'child-of'), ('baby-of', 'has-baby'), ('preys-on', 'prey-of')) + tuple(
    (SCALES[d][0], SCALES[d][1]) for d in SCALES if d not in ADJACENT_ONLY)
COMPARATIVES = tuple(r for s in SCALES.values() for r in s[:2])

RELATIONS = ('is-a', 'has', 'can', 'lives-in', 'eats', 'says', 'baby-of', 'has-baby', 'colour-of', 'mixed-from',
             'is', 'opposite') + COMPARATIVES + ('parent-of', 'child-of', 'spouse-of', 'sibling-of', 'grandparent-of',
                                                'names', 'can-subject', 'same-colour-as', 'same-class-as', 'lives-with',
                                                'preys-on', 'prey-of')

PROPERTY = {r: 'symmetric' for r in SYMMETRIC}
PROPERTY.update({a: 'inverse-first' for a, _ in INVERSE_PAIRS})
PROPERTY.update({bb: 'inverse-second' for _, bb in INVERSE_PAIRS})
PROPERTY.update({r: 'comparative' for r in COMPARATIVES if r not in PROPERTY})

T1_ANIMALS, T1_RELS, T2_ANIMALS, T2_WORDS = W1.T1_ANIMALS, W1.T1_RELS, W1.T2_ANIMALS, W1.T2_WORDS
T9_ANIMALS, T8_NOUNS = W1.T9_ANIMALS, W1.T8_NOUNS
P6_COLOUR_THINGS = ('banana', 'grass', 'snow', 'grape')   # same-colour-as rows and columns held out; colour-of trained


def pairs_same(groups, items):
    """Ordered pairs (x, y), x != y, sharing at least one group value; groups: item -> set."""
    return {(x, y) for x in items for y in items if x != y and groups[x] & groups[y]}


def tables():
    animals = list(ANIMALS)
    people = list(TREE_A) + list(TREE_B)
    words = [w(x) for x in NOUN_WORDS + VERB_WORDS]
    coloured = list(COLOURED)
    things = []
    for s in SCALES.values():
        for x in s[2]:
            if x not in things:
                things.append(x)
    H, T, F = {}, {}, {}
    H['is-a'] = animals + list(CLASSES) + words
    T['is-a'] = list(CLASSES) + ['animal', 'noun', 'verb']
    F['is-a'] = ({(a, ANIMALS[a][0]) for a in animals} | {(a, 'animal') for a in animals}
                 | {(c, 'animal') for c in CLASSES} | {(w(x), 'noun') for x in NOUN_WORDS}
                 | {(w(x), 'verb') for x in VERB_WORDS})
    H['has'] = animals + list(COLOURS) + people
    T['has'] = list(PARTS) + ['warm', 'cool', 'male', 'female']
    F['has'] = ({(a, p) for a in animals for p in ANIMALS[a][1].split()} | {(c, 'warm') for c in WARM}
                | {(c, 'cool') for c in COOL})
    for p in people:
        a = p if p in TREE_A else TREE_A[TREE_B.index(p)]
        F['has'].add((p, 'male' if a in MALE else 'female'))
    for rel, k, vals in (('can', 2, ABILITIES), ('lives-in', 3, HOMES), ('eats', 4, FOODS)):
        H[rel], T[rel] = animals, list(vals)
        F[rel] = {(a, v) for a in animals for v in ANIMALS[a][k].split()}
    H['says'], T['says'] = list(SOUNDS), sorted(set(SOUNDS.values()))
    F['says'] = set(SOUNDS.items())
    H['baby-of'], T['baby-of'] = list(BABIES), animals
    F['baby-of'] = {(bb, a) for bb, s in BABIES.items() for a in s.split()}
    H['has-baby'], T['has-baby'] = animals, list(BABIES)
    F['has-baby'] = {(a, bb) for bb, a in F['baby-of']}
    H['colour-of'], T['colour-of'] = coloured, list(COLOURS)
    F['colour-of'] = {(x, c) for x, s in COLOURED.items() for c in s.split()}
    H['mixed-from'] = T['mixed-from'] = list(COLOURS)
    F['mixed-from'] = {(c, p) for c, s in MIXES.items() for p in s.split()}
    H['is'], T['is'] = things, list(ADJECTIVES)
    F['is'] = set()
    for comp, inv, order, p1, p2 in SCALES.values():
        F['is'] |= {(x, p1) for x in order[:2]} | {(x, p2) for x in order[-2:]}
        H[comp] = T[comp] = H[inv] = T[inv] = list(order)
        F[comp] = {(order[i], order[j]) for i in range(len(order)) for j in range(len(order)) if i < j}
        F[inv] = {(y, x) for x, y in F[comp]}
    H['opposite'] = T['opposite'] = list(ADJECTIVES)
    F['opposite'] = {(s[3], s[4]) for s in SCALES.values()} | {(s[4], s[3]) for s in SCALES.values()}
    fam = W1.family_facts()
    for rel in fam:
        H[rel] = T[rel] = people
        F[rel] = fam[rel]
    H['names'], T['names'] = words, animals + list(ABILITIES)
    F['names'] = {(w(x), x) for x in NOUN_WORDS + VERB_WORDS}
    H['can-subject'], T['can-subject'] = [w(x) for x in NOUN_WORDS], [w(x) for x in VERB_WORDS]
    F['can-subject'] = {(w(n), w(v)) for n in NOUN_WORDS for v in VERB_WORDS if v in ANIMALS[n][2].split()}
    H['same-colour-as'] = T['same-colour-as'] = coloured
    F['same-colour-as'] = pairs_same({x: set(s.split()) for x, s in COLOURED.items()}, coloured)
    H['same-class-as'] = T['same-class-as'] = animals
    F['same-class-as'] = pairs_same({a: {ANIMALS[a][0]} for a in animals}, animals)
    H['lives-with'] = T['lives-with'] = animals
    F['lives-with'] = pairs_same({a: set(ANIMALS[a][3].split()) for a in animals}, animals)
    H['preys-on'], T['preys-on'] = animals, animals
    F['preys-on'] = {(a, p) for a, s in PREYS_ON.items() for p in s.split()}
    H['prey-of'], T['prey-of'] = animals, animals
    F['prey-of'] = {(p, a) for a, p in F['preys-on']}
    return H, T, F


def third(items, key):
    """Deterministic third of items: every third one in sorted order, offset by a fixed per-relation key."""
    items = sorted(items)
    off = sum(map(ord, key)) % 3
    return set(items[off::3])


def held_rows(H, F):
    """P2: symmetric relation -> held-out heads (whole rows); P3: inverse second relation -> held-out heads."""
    p2, p3 = {}, {}
    for rel in SYMMETRIC:
        heads = {x for x, _ in F[rel]}
        if rel == 'opposite':
            heads = {a for d, s in SCALES.items() if d not in OPP_FROM_MEANING for a in s[3:]}
        if rel == 'same-colour-as':
            heads -= set(P6_COLOUR_THINGS)
        p2[rel] = third(heads, rel)
    for first, second in INVERSE_PAIRS:
        p3[second] = third({x for x, _ in F[second]}, second)
    return p2, p3


def heldout(rel, hd, tl, p2, p3):
    if rel in T1_RELS and hd in T1_ANIMALS:
        return 'P1'
    if rel == 'lives-with' and (hd in T1_ANIMALS or tl in T1_ANIMALS):
        return 'P1-lives-with'
    if rel == 'is-a' and tl == 'animal' and (hd in T2_ANIMALS or hd in T2_WORDS):
        return 'P1-hierarchy'
    if rel == 'same-colour-as' and (hd in P6_COLOUR_THINGS or tl in P6_COLOUR_THINGS):
        return 'P6-colour'
    if rel == 'grandparent-of' and tl == b('colin'):
        return 'P6-grandparent'
    if rel == 'opposite' and {d for d, s in SCALES.items() if hd in s[3:]} & set(OPP_FROM_MEANING):
        return 'P5'
    if rel in COMPARATIVES:
        d = next(d for d, s in SCALES.items() if rel in s[:2])
        order = SCALES[d][2]
        if d in ADJACENT_ONLY and abs(order.index(hd) - order.index(tl)) >= 2:
            return 'P4'
    if rel in p2 and hd in p2[rel]:
        return 'P2'
    if rel in p3 and hd in p3[rel]:
        return 'P3'
    if rel == 'can-subject' and (hd in {w(x) for x in T8_NOUNS}
                                 or (NOUN_WORDS.index(hd[:-2]) + VERB_WORDS.index(tl[:-2])) % 4 == 0):
        return 'P8'
    if rel == 'colour-of' and hd in T9_ANIMALS:
        return 'P7'
    return ''


def class_default(a, rel, tl):
    cls = ANIMALS[a][0]
    mates = [m for m in ANIMALS if ANIMALS[m][0] == cls and m != a and m not in T1_ANIMALS]
    k = {'has': 1, 'can': 2, 'lives-in': 3, 'eats': 4}[rel]
    votes = [tl in ANIMALS[m][k].split() for m in mates]
    if not votes or 2 * sum(votes) == len(votes):
        return None
    return 2 * sum(votes) > len(votes)


@dataclass
class World:
    entities: list
    relations: tuple
    heads: dict
    tails: dict
    facts: dict
    h: np.ndarray = field(default=None)
    r: np.ndarray = field(default=None)
    t: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    test: np.ndarray = field(default=None)
    subset: np.ndarray = field(default=None)

    def index(self, name):
        return self.entities.index(name)


def build():
    H, T, F = tables()
    p2, p3 = held_rows(H, F)
    ents = []
    for rel in RELATIONS:
        for x in H[rel] + T[rel]:
            if x not in ents:
                ents.append(x)
    world = World(ents, RELATIONS, H, T, F)
    rows = []
    for ri, rel in enumerate(RELATIONS):
        for hd in H[rel]:
            for tl in T[rel]:
                y = (hd, tl) in F[rel]
                tst = heldout(rel, hd, tl, p2, p3)
                sub = ''
                if tst == 'P1':
                    d = class_default(hd, rel, tl)
                    sub = 'none' if d is None else ('default' if d == y else 'specific')
                if tst == 'P4':
                    order = next(s[2] for s in SCALES.values() if rel in s[:2])
                    sub = str(abs(order.index(hd) - order.index(tl)))
                rows.append((ents.index(hd), ri, ents.index(tl), y, tst, sub))
    h, r, t, y, tst, sub = zip(*rows)
    world.h, world.r, world.t = np.array(h), np.array(r), np.array(t)
    world.y, world.test, world.subset = np.array(y, dtype=np.float32), np.array(tst), np.array(sub)
    return world


shuffled = W1.shuffled


def fast_mapping_specs():
    """name -> (given [(rel, head, tail, label)], read, ks, n_negative_context). k counts given facts in order; the
    is-a row's false facts (dax, fep) are given with every k (they are the row the held-out lizard had)."""
    specs = {}
    for name, cls, extra in (('dax', 'bird', [('has', 'beak'), ('can', 'sing'), ('lives-in', 'trees')]),
                             ('fep', 'fish', [('has', 'fins'), ('lives-in', 'water'), ('eats', 'meat')])):
        given = [('is-a', name, cls, True)] + [(r, name, v, True) for r, v in extra]
        row = [('is-a', name, c, False) for c in CLASSES if c != cls] + [('is-a', name, x, False) for x in ('noun', 'verb')]
        read = [('is-a', name, 'animal', True)]
        for rel in ('has', 'can', 'lives-in', 'eats'):
            read += [(rel, name, v, lab) for v, lab in W1.bird_majority(cls, rel).items()]
        gk = {(g[0], g[2]) for g in given}
        specs[name] = (given, [x for x in read if (x[0], x[2]) not in gk], (1, 2, 4), row)
    size = W1.SIZE
    lo = size.index('dog')
    given = [('bigger-than', 'zorp', 'cat', True), ('bigger-than', 'dog', 'zorp', True),
             ('bigger-than', 'cat', 'zorp', False), ('bigger-than', 'zorp', 'dog', False)]
    read = []
    for i, a in enumerate(size):
        if a in ('cat', 'dog'):
            continue
        read += [('bigger-than', 'zorp', a, i > lo), ('bigger-than', a, 'zorp', i <= lo),
                 ('smaller-than', 'zorp', a, i <= lo), ('smaller-than', a, 'zorp', i > lo)]
    read += [('smaller-than', 'zorp', 'cat', False), ('smaller-than', 'cat', 'zorp', True),
             ('smaller-than', 'zorp', 'dog', True), ('smaller-than', 'dog', 'zorp', False)]
    specs['zorp'] = (given, read, (1, 2, 4), [])
    temp = SCALES['temperature'][2]
    given = [('is', 'sun', 'toasty', True), ('is', 'snow', 'toasty', False), ('is', 'fire', 'toasty', True),
             ('is', 'ice', 'toasty', False)]
    read = [('opposite', 'toasty', a, a == 'cold') for a in ADJECTIVES] + [('opposite', a, 'toasty', a == 'cold')
                                                                          for a in ADJECTIVES]
    read += [('is', x, 'toasty', x in temp[:2]) for x in temp if x not in ('sun', 'snow', 'fire', 'ice')]
    specs['toasty'] = (given, read, (2, 4), [])
    given = [('child-of', 'kim', 'arthur', True), ('parent-of', 'arthur', 'kim', True),
             ('child-of', 'kim', 'margaret', True), ('parent-of', 'margaret', 'kim', True)]
    read = [('grandparent-of', p, 'kim', p in ('christopher', 'penelope')) for p in TREE_A]
    read += [('child-of', 'kim', p, p in ('arthur', 'margaret')) for p in TREE_A]
    read += [('parent-of', p, 'kim', p in ('arthur', 'margaret')) for p in TREE_A]
    read = [x for x in read if x[:3] not in {g[:3] for g in given}]
    specs['kim'] = (given, read, (1, 2, 4), [])
    return specs


def new_relation_specs():
    size = W1.SIZE
    smaller = [(a, c, size.index(a) > size.index(c)) for a in size for c in size]
    fliers = {a for a in ANIMALS if 'fly' in ANIMALS[a][2].split()}
    an = list(ANIMALS)
    both = [(a, c, a != c and a in fliers and c in fliers) for a in an for c in an]
    return {'smaller-than-v': smaller, 'both-fly': both}
