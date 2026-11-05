"""Everyday relations v1 world (docs/everyday-relations-v1.md): a closed world of facts children learn.

The tables below are the specification. `build()` returns every query (head, relation, tail, label) over each relation's
declared head and tail sets, with the held-out tests marked. Deterministic: no randomness here.
"""
from dataclasses import dataclass, field

import numpy as np

CLASSES = ('mammal', 'bird', 'fish', 'insect', 'reptile', 'amphibian')
PARTS = ('fur', 'feathers', 'scales', 'wings', 'fins', 'beak', 'tail', 'legs0', 'legs2', 'legs4', 'legs6')
ABILITIES = ('fly', 'swim', 'run', 'climb', 'sing')
HOMES = ('water', 'land', 'trees', 'farm')
FOODS = ('meat', 'plants', 'seeds', 'bugs', 'nectar')

# animal: class, parts, abilities, homes, foods
ANIMALS = {
    'dog': ('mammal', 'fur tail legs4', 'run swim', 'land farm', 'meat'),
    'cat': ('mammal', 'fur tail legs4', 'run climb', 'land', 'meat'),
    'cow': ('mammal', 'fur tail legs4', 'run', 'land farm', 'plants'),
    'horse': ('mammal', 'fur tail legs4', 'run', 'land farm', 'plants'),
    'lion': ('mammal', 'fur tail legs4', 'run', 'land', 'meat'),
    'elephant': ('mammal', 'tail legs4', 'run swim', 'land', 'plants'),
    'mouse': ('mammal', 'fur tail legs4', 'run climb', 'land', 'seeds'),
    'rabbit': ('mammal', 'fur tail legs4', 'run', 'land', 'plants'),
    'bat': ('mammal', 'fur wings legs2', 'fly', 'trees', 'bugs'),
    'whale': ('mammal', 'fins tail legs0', 'swim', 'water', 'meat'),
    'robin': ('bird', 'feathers wings beak tail legs2', 'fly sing', 'trees', 'bugs'),
    'sparrow': ('bird', 'feathers wings beak tail legs2', 'fly sing', 'trees', 'seeds'),
    'eagle': ('bird', 'feathers wings beak tail legs2', 'fly', 'trees', 'meat'),
    'owl': ('bird', 'feathers wings beak tail legs2', 'fly', 'trees', 'meat'),
    'duck': ('bird', 'feathers wings beak tail legs2', 'fly swim', 'water farm', 'plants'),
    'penguin': ('bird', 'feathers wings beak tail legs2', 'swim', 'water', 'meat'),
    'chicken': ('bird', 'feathers wings beak tail legs2', 'run', 'farm', 'seeds'),
    'flamingo': ('bird', 'feathers wings beak tail legs2', 'fly', 'water', 'bugs'),
    'crow': ('bird', 'feathers wings beak tail legs2', 'fly', 'trees', 'seeds'),
    'salmon': ('fish', 'scales fins tail legs0', 'swim', 'water', 'bugs'),
    'shark': ('fish', 'scales fins tail legs0', 'swim', 'water', 'meat'),
    'goldfish': ('fish', 'scales fins tail legs0', 'swim', 'water', 'plants'),
    'tuna': ('fish', 'scales fins tail legs0', 'swim', 'water', 'meat'),
    'bee': ('insect', 'wings legs6', 'fly', 'land', 'nectar'),
    'butterfly': ('insect', 'wings legs6', 'fly', 'land', 'nectar'),
    'housefly': ('insect', 'wings legs6', 'fly', 'land', 'plants'),
    'ant': ('insect', 'legs6', 'climb', 'land', 'seeds'),
    'snake': ('reptile', 'scales tail legs0', 'swim climb', 'land', 'meat'),
    'turtle': ('reptile', 'scales tail legs4', 'swim', 'water land', 'plants'),
    'lizard': ('reptile', 'scales tail legs4', 'run climb', 'land', 'bugs'),
    'crocodile': ('reptile', 'scales tail legs4', 'swim', 'water land', 'meat'),
    'frog': ('amphibian', 'legs4', 'swim', 'water land', 'bugs'),
    'toad': ('amphibian', 'legs4', 'run', 'land', 'bugs'),
    'newt': ('amphibian', 'tail legs4', 'swim', 'water land', 'bugs'),
}
SOUNDS = {'dog': 'bark', 'cat': 'meow', 'cow': 'moo', 'horse': 'neigh', 'lion': 'roar', 'elephant': 'trumpet',
          'mouse': 'squeak', 'robin': 'tweet', 'sparrow': 'tweet', 'eagle': 'screech', 'owl': 'hoot', 'duck': 'quack',
          'chicken': 'cluck', 'crow': 'caw', 'bee': 'buzz', 'housefly': 'buzz', 'snake': 'hiss', 'frog': 'croak',
          'toad': 'croak'}
BABIES = {'puppy': 'dog', 'kitten': 'cat', 'calf': 'cow elephant whale', 'foal': 'horse', 'cub': 'lion',
          'chick': 'robin sparrow eagle owl penguin chicken crow flamingo', 'duckling': 'duck',
          'tadpole': 'frog toad', 'caterpillar': 'butterfly'}
COLOURS = ('red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray')
COLOURED = {'banana': 'yellow', 'lemon': 'yellow', 'sun': 'yellow', 'apple': 'red', 'strawberry': 'red',
            'cherry': 'red', 'tomato': 'red', 'grass': 'green', 'leaf': 'green', 'sky': 'blue', 'sea': 'blue',
            'snow': 'white', 'milk': 'white', 'coal': 'black', 'chocolate': 'brown', 'carrot': 'orange',
            'tangerine': 'orange', 'grape': 'purple', 'plum': 'purple', 'cloud': 'gray',
            'elephant': 'gray', 'mouse': 'gray', 'flamingo': 'pink', 'crow': 'black', 'frog': 'green',
            'cow': 'black white', 'penguin': 'black white', 'lion': 'yellow', 'dog': 'brown', 'duck': 'white'}
MIXES = {'orange': 'red yellow', 'green': 'blue yellow', 'purple': 'red blue', 'pink': 'red white',
         'gray': 'black white'}
WARM, COOL = ('red', 'orange', 'yellow'), ('blue', 'green', 'purple')
SIZE = ('whale', 'elephant', 'horse', 'lion', 'dog', 'cat', 'rabbit', 'duck', 'mouse', 'frog', 'bee', 'ant')
OPPOSITES = (('big', 'small'), ('hot', 'cold'), ('wet', 'dry'), ('fast', 'slow'), ('up', 'down'), ('day', 'night'),
             ('happy', 'sad'), ('light', 'dark'))
OPP_ONE_ORDER = 3  # the last 3 pairs are trained in one order only

# Hinton (1986) family tree; tree B is isomorphic (b-names map in order).
TREE_A = ('christopher', 'penelope', 'andrew', 'christine', 'margaret', 'arthur', 'victoria', 'james', 'jennifer',
          'charles', 'colin', 'charlotte')
TREE_B = ('roberto', 'maria', 'pierro', 'francesca', 'gina', 'emilio', 'lucia', 'marco', 'angela', 'tomaso',
          'alfonso', 'sophia')
MALE = {'christopher', 'andrew', 'arthur', 'james', 'charles', 'colin'}
SPOUSES = (('christopher', 'penelope'), ('andrew', 'christine'), ('arthur', 'margaret'), ('james', 'victoria'),
           ('charles', 'jennifer'))
PARENTS = {'arthur': ('christopher', 'penelope'), 'victoria': ('christopher', 'penelope'),
           'james': ('andrew', 'christine'), 'jennifer': ('andrew', 'christine'),
           'colin': ('victoria', 'james'), 'charlotte': ('victoria', 'james')}

NOUN_WORDS = ('dog', 'cat', 'cow', 'horse', 'duck', 'eagle', 'robin', 'salmon', 'shark', 'snake', 'frog', 'bee')
VERB_WORDS = ABILITIES

RELATIONS = ('is-a', 'has', 'can', 'lives-in', 'eats', 'says', 'baby-of', 'colour-of', 'mixed-from', 'bigger-than',
             'opposite', 'parent-of', 'child-of', 'spouse-of', 'sibling-of', 'grandparent-of', 'names', 'can-subject')

# Held-out choices (fixed)
T1_ANIMALS = ('robin', 'salmon', 'dog', 'bee', 'lizard', 'frog')
T1_RELS = ('has', 'can', 'lives-in', 'eats')
T2_ANIMALS = ('sparrow', 'shark', 'horse', 'butterfly', 'turtle', 'toad')
T2_WORDS = ('cat-w', 'cow-w', 'duck-w', 'shark-w', 'run-w', 'sing-w')  # negatives: words are not animals
T9_ANIMALS =('elephant', 'flamingo', 'crow')
T8_NOUNS = ('eagle', 'snake')


def w(x):
    return x + '-w'


def b(name):
    """Tree A name -> tree B name."""
    return TREE_B[TREE_A.index(name)]


def family_facts():
    """Facts over the 24 people, both trees."""
    f = {r: set() for r in ('parent-of', 'child-of', 'spouse-of', 'sibling-of', 'grandparent-of')}
    for m in (lambda x: x, b):
        for x, y in SPOUSES:
            f['spouse-of'] |= {(m(x), m(y)), (m(y), m(x))}
        for c, ps in PARENTS.items():
            for p in ps:
                f['parent-of'].add((m(p), m(c)))
                f['child-of'].add((m(c), m(p)))
        for c1, p1 in PARENTS.items():
            for c2, p2 in PARENTS.items():
                if c1 != c2 and p1 == p2:
                    f['sibling-of'].add((m(c1), m(c2)))
        for c, ps in PARENTS.items():
            for p in ps:
                for g in PARENTS.get(p, ()):
                    f['grandparent-of'].add((m(g), m(c)))
    return f


@dataclass
class World:
    entities: list
    relations: tuple
    heads: dict          # relation -> list of entity names
    tails: dict
    facts: dict          # relation -> set of (head, tail)
    h: np.ndarray = field(default=None)   # query arrays (entity / relation indices), label, test name ('' = trained)
    r: np.ndarray = field(default=None)
    t: np.ndarray = field(default=None)
    y: np.ndarray = field(default=None)
    test: np.ndarray = field(default=None)
    subset: np.ndarray = field(default=None)  # T1: 'default' (label = classmates' majority) or 'specific'

    def index(self, name):
        return self.entities.index(name)


def tables():
    """(heads, tails, facts) per relation."""
    animals = list(ANIMALS)
    people = list(TREE_A) + list(TREE_B)
    adjectives = [a for p in OPPOSITES for a in p]
    words = [w(x) for x in NOUN_WORDS + VERB_WORDS]
    coloured = list(COLOURED)
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
    H['colour-of'], T['colour-of'] = coloured, list(COLOURS)
    F['colour-of'] = {(x, c) for x, s in COLOURED.items() for c in s.split()}
    H['mixed-from'] = T['mixed-from'] = list(COLOURS)
    F['mixed-from'] = {(c, p) for c, s in MIXES.items() for p in s.split()}
    H['bigger-than'] = T['bigger-than'] = list(SIZE)
    F['bigger-than'] = {(SIZE[i], SIZE[j]) for i in range(len(SIZE)) for j in range(len(SIZE)) if i < j}
    H['opposite'] = T['opposite'] = adjectives
    F['opposite'] = {(x, y) for x, y in OPPOSITES} | {(y, x) for x, y in OPPOSITES}
    fam = family_facts()
    for rel in fam:
        H[rel] = T[rel] = people
        F[rel] = fam[rel]
    H['names'], T['names'] = words, animals + list(ABILITIES)
    F['names'] = {(w(x), x) for x in NOUN_WORDS + VERB_WORDS}
    H['can-subject'], T['can-subject'] = [w(x) for x in NOUN_WORDS], [w(x) for x in VERB_WORDS]
    F['can-subject'] = {(w(n), w(v)) for n in NOUN_WORDS for v in VERB_WORDS if v in ANIMALS[n][2].split()}
    return H, T, F


def heldout(rel, hd, tl, H, F):
    """Test name for a query, '' if trained."""
    if rel in T1_RELS and hd in T1_ANIMALS:
        return 'T1'
    if rel == 'is-a' and tl == 'animal' and (hd in T2_ANIMALS or hd in T2_WORDS):
        return 'T2'
    if rel == 'can' and hd == 'bat' and tl == 'fly':
        return 'T3b'
    if rel == 'bigger-than' and abs(SIZE.index(hd) - SIZE.index(tl)) >= 2:
        return 'T4'
    if rel == 'opposite' and (tl, hd) in OPPOSITES[-OPP_ONE_ORDER:]:
        return 'T5'
    if rel == 'sibling-of' and (hd, tl) in ((b('victoria'), b('arthur')), (b('charlotte'), b('colin'))):
        return 'T5'
    if rel == 'child-of' and hd in {b(c) for c in PARENTS}:
        return 'T6'
    if rel == 'grandparent-of' and tl == b('colin'):
        return 'T7'
    if rel == 'can-subject' and (hd in {w(x) for x in T8_NOUNS}
                                 or (NOUN_WORDS.index(hd[:-2]) + VERB_WORDS.index(tl[:-2])) % 4 == 0):
        return 'T8'
    if rel == 'colour-of' and hd in T9_ANIMALS:
        return 'T9'
    return ''


def class_default(a, rel, tl):
    """Majority label among a's classmates other than T1 animals; None if tied or no classmates."""
    cls = ANIMALS[a][0]
    mates = [m for m in ANIMALS if ANIMALS[m][0] == cls and m != a and m not in T1_ANIMALS]
    k = {'has': 1, 'can': 2, 'lives-in': 3, 'eats': 4}[rel]
    votes = [tl in ANIMALS[m][k].split() for m in mates]
    if not votes or 2 * sum(votes) == len(votes):
        return None
    return 2 * sum(votes) > len(votes)


def build():
    H, T, F = tables()
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
                tst = heldout(rel, hd, tl, H, F)
                sub = ''
                if tst == 'T1':
                    d = class_default(hd, rel, tl)
                    sub = 'none' if d is None else ('default' if d == y else 'specific')
                rows.append((ents.index(hd), ri, ents.index(tl), y, tst, sub))
    h, r, t, y, tst, sub = zip(*rows)
    world.h, world.r, world.t = np.array(h), np.array(r), np.array(t)
    world.y, world.test, world.subset = np.array(y, dtype=np.float32), np.array(tst), np.array(sub)
    return world


def shuffled(world, seed):
    """Same queries and positions; each relation's heads and tails relabelled by independent permutations of that
    relation's head and tail sets, so facts of one entity in different relations no longer belong together and
    within-relation structure (order, symmetry, inverses) is broken."""
    rng = np.random.default_rng(seed)
    y = world.y.copy()
    for ri, rel in enumerate(world.relations):
        hs = [world.index(x) for x in world.heads[rel]]
        ts = [world.index(x) for x in world.tails[rel]]
        ph = dict(zip(hs, rng.permutation(hs)))
        pt = dict(zip(ts, rng.permutation(ts)))
        m = world.r == ri
        lab = {(int(a), int(c)): v for a, c, v in zip(world.h[m], world.t[m], world.y[m])}
        y[m] = [lab[(ph[int(a)], pt[int(c)])] for a, c in zip(world.h[m], world.t[m])]
    return y


# ---------------------------------------------------------------- fast mapping (truth defined here)

def bird_majority(cls, rel):
    k = {'has': 1, 'can': 2, 'lives-in': 3, 'eats': 4}[rel]
    mates = [m for m in ANIMALS if ANIMALS[m][0] == cls]
    vals = {'has': PARTS, 'can': ABILITIES, 'lives-in': HOMES, 'eats': FOODS}[rel]
    out = {}
    for v in vals:
        frac = np.mean([v in ANIMALS[m][k].split() for m in mates])
        if frac >= .75 or frac <= .25:
            out[v] = bool(frac >= .75)
    return out


def fast_mapping_specs():
    """name -> (given facts in order [(rel, head, tail, label)], read facts [(rel, head, tail, label)], ks).
    The new entity's own name appears as head or tail."""
    specs = {}
    for name, cls, extra in (('dax', 'bird', [('has', 'beak'), ('can', 'sing'), ('lives-in', 'trees')]),
                             ('fep', 'fish', [('has', 'fins'), ('lives-in', 'water'), ('eats', 'meat')])):
        given = [('is-a', name, cls, True)] + [(r, name, v, True) for r, v in extra]
        read = [('is-a', name, c, c == cls) for c in CLASSES] + [('is-a', name, 'animal', True)]
        for rel in ('has', 'can', 'lives-in', 'eats'):
            read += [(rel, name, v, lab) for v, lab in bird_majority(cls, rel).items()]
        gk = {(g[0], g[2]) for g in given}
        specs[name] = (given, [x for x in read if (x[0], x[2]) not in gk], (1, 2, 4))
    owl_can = ANIMALS['owl'][2].split()
    given = [('names', 'wug-w', 'owl', True), ('is-a', 'wug-w', 'noun', True)]
    read = [('can-subject', 'wug-w', w(v), v in owl_can) for v in VERB_WORDS] + [('is-a', 'wug-w', 'verb', False)]
    specs['wug-w'] = (given, read, (1, 2))
    lo, hi = SIZE.index('dog'), SIZE.index('cat')  # zorp sits between dog and cat
    given = [('bigger-than', 'zorp', 'cat', True), ('bigger-than', 'dog', 'zorp', True),
             ('bigger-than', 'cat', 'zorp', False), ('bigger-than', 'zorp', 'dog', False)]
    read = []
    for i, a in enumerate(SIZE):
        read += [('bigger-than', 'zorp', a, i > lo), ('bigger-than', a, 'zorp', i <= lo)]
    read = [x for x in read if (x[1], x[2]) not in {(g[1], g[2]) for g in given}]
    specs['zorp'] = (given, read, (1, 2, 4))
    given = [('child-of', 'kim', 'arthur', True), ('parent-of', 'arthur', 'kim', True),
             ('child-of', 'kim', 'margaret', True), ('parent-of', 'margaret', 'kim', True)]
    read = [('grandparent-of', p, 'kim', p in ('christopher', 'penelope')) for p in TREE_A]
    read += [('child-of', 'kim', p, p in ('arthur', 'margaret')) for p in TREE_A]
    read += [('parent-of', p, 'kim', p in ('arthur', 'margaret')) for p in TREE_A]
    read = [x for x in read if (x[0], x[1], x[2]) not in {g[:3] for g in given}]
    specs['kim'] = (given, read, (1, 2, 4))
    return specs


def new_relation_specs(world):
    """name -> list of (head, tail, label) over the relation's full query set."""
    size = [(a, c, SIZE.index(a) > SIZE.index(c)) for a in SIZE for c in SIZE]
    an = list(ANIMALS)
    same = [(a, c, a != c and ANIMALS[a][0] == ANIMALS[c][0]) for a in an for c in an]
    return {'smaller-than': size, 'same-class': same}
