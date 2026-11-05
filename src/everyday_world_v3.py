"""Everyday relations v3 world (docs/everyday-relations-v3.md): v2's world plus a two-word phrase domain.

Words: 19 noun words, 5 verb words, 8 adjective words, each with `names` and `is-a` (noun / verb / adjective).
`good-phrase` (w1, w2): word order only, (noun, verb) or (adjective, noun).
`makes-sense` (w1, w2): a good phrase whose meaning holds (the animal can do the action / is the adjective).
v2's `can-subject` is replaced by `makes-sense`. Deterministic; the tables are the specification.
"""
import numpy as np

from src import everyday_world as W1
from src import everyday_world_v2 as W2

NOUN_WORDS = W1.NOUN_WORDS + ('whale', 'elephant', 'lion', 'mouse', 'ant', 'cheetah', 'turtle')
VERB_WORDS = W1.VERB_WORDS
ADJ_WORDS = ('big', 'small', 'fast', 'slow', 'heavy', 'light', 'loud', 'quiet')
w = W1.w
NOUNS, VERBS, ADJS = [w(x) for x in NOUN_WORDS], [w(x) for x in VERB_WORDS], [w(x) for x in ADJ_WORDS]
WORDS = NOUNS + VERBS + ADJS
CATEGORY = {**{x: 'noun' for x in NOUNS}, **{x: 'verb' for x in VERBS}, **{x: 'adjective' for x in ADJS}}

G2_WORDS = ('robin-w', 'shark-w')                     # known only by meaning (names trained)
G3_WORDS = ('cow-w', 'frog-w')                        # known only from use
G3_GIVEN = (('run-w', True), ('swim-w', True))       # good-phrase(word, verb) trained; (run-w, word) trained false

RELATIONS = tuple(r for r in W2.RELATIONS if r != 'can-subject') + ('good-phrase', 'makes-sense')
ANIMALS, CLASSES, SCALES = W2.ANIMALS, W2.CLASSES, W2.SCALES
PROPERTY = W2.PROPERTY


def good(a, b):
    return (CATEGORY[a], CATEGORY[b]) in (('noun', 'verb'), ('adjective', 'noun'))


def tables():
    H, T, F = W2.tables()
    for d in (H, T, F):
        d.pop('can-subject')
    is_facts = F['is']
    H['is-a'] = [x for x in H['is-a'] if not x.endswith('-w')] + WORDS
    T['is-a'] = T['is-a'] + ['adjective']
    F['is-a'] = {f for f in F['is-a'] if not f[0].endswith('-w')} | {(x, CATEGORY[x]) for x in WORDS}
    H['names'] = WORDS
    T['names'] = list(ANIMALS) + list(W1.ABILITIES) + list(ADJ_WORDS)
    F['names'] = {(x, x[:-2]) for x in WORDS}
    H['good-phrase'] = T['good-phrase'] = WORDS
    F['good-phrase'] = {(a, b) for a in WORDS for b in WORDS if good(a, b)}
    H['makes-sense'] = T['makes-sense'] = WORDS
    ms = set()
    for a, b in F['good-phrase']:
        if CATEGORY[a] == 'noun' and b[:-2] in ANIMALS[a[:-2]][2].split():
            ms.add((a, b))
        if CATEGORY[a] == 'adjective' and (b[:-2], a[:-2]) in is_facts:
            ms.add((a, b))
    F['makes-sense'] = ms
    return H, T, F


def heldout(rel, hd, tl, p2, p3):
    word_rels = ('good-phrase', 'makes-sense', 'is-a', 'names')
    if rel in word_rels and (hd in G2_WORDS or tl in G2_WORDS):
        return '' if rel == 'names' else 'G2'
    if rel in word_rels and (hd in G3_WORDS or tl in G3_WORDS):
        if rel == 'good-phrase' and hd in G3_WORDS and tl in [v for v, _ in G3_GIVEN]:
            return ''
        if rel == 'good-phrase' and tl in G3_WORDS and hd == G3_GIVEN[0][0]:
            return ''
        return 'G3-names' if rel == 'names' else 'G3'
    if rel == 'good-phrase' and (WORDS.index(hd) + WORDS.index(tl)) % 4 == 0:
        return 'G1'
    if rel == 'makes-sense' and good(hd, tl) and (7 * WORDS.index(hd) + WORDS.index(tl)) % 4 == 1:
        return 'G4'
    return W2.heldout(rel, hd, tl, p2, p3)


def build():
    H, T, F = tables()
    p2, p3 = W2.held_rows(H, F)
    ents = []
    for rel in RELATIONS:
        for x in H[rel] + T[rel]:
            if x not in ents:
                ents.append(x)
    world = W2.World(ents, RELATIONS, H, T, F)
    rows = []
    for ri, rel in enumerate(RELATIONS):
        for hd in H[rel]:
            for tl in T[rel]:
                y = (hd, tl) in F[rel]
                tst = heldout(rel, hd, tl, p2, p3)
                sub = ''
                if tst == 'P1':
                    d = W2.class_default(hd, rel, tl)
                    sub = 'none' if d is None else ('default' if d == y else 'specific')
                if tst == 'P4':
                    order = next(s[2] for s in SCALES.values() if rel in s[:2])
                    sub = str(abs(order.index(hd) - order.index(tl)))
                if tst == 'G4':
                    sub = 'sensible' if y else 'silly'
                if tst in ('G2', 'G3'):
                    sub = rel
                rows.append((ents.index(hd), ri, ents.index(tl), y, tst, sub))
    h, r, t, y, tst, sub = zip(*rows)
    world.h, world.r, world.t = np.array(h), np.array(r), np.array(t)
    world.y, world.test, world.subset = np.array(y, dtype=np.float32), np.array(tst), np.array(sub)
    return world


shuffled = W1.shuffled


def fast_mapping_specs():
    """name -> (given, read, ks). No exclusions are given; dax/fep read the other classes as tests."""
    specs = {}
    for name, (given, read, ks, row) in W2.fast_mapping_specs().items():
        specs[name] = (given, read + [x for x in row if x[2] in CLASSES], ks)
    owl = ANIMALS['owl'][2].split()
    read = ([('good-phrase', 'blick-w', v, True) for v in VERBS] + [('good-phrase', v, 'blick-w', False) for v in VERBS]
            + [('good-phrase', a, 'blick-w', True) for a in ADJS] + [('good-phrase', 'blick-w', a, False) for a in ADJS]
            + [('is-a', 'blick-w', c, c == 'noun') for c in ('noun', 'verb', 'adjective')]
            + [('makes-sense', 'blick-w', v, v[:-2] in owl) for v in VERBS])
    specs['blick-w'] = ([('names', 'blick-w', 'owl', True)], read, (1,))
    given = [('good-phrase', 'zib-w', 'run-w', True), ('good-phrase', 'zib-w', 'swim-w', True),
             ('good-phrase', 'run-w', 'zib-w', False)]
    read = ([('good-phrase', 'zib-w', v, True) for v in VERBS if v not in ('run-w', 'swim-w')]
            + [('good-phrase', v, 'zib-w', False) for v in VERBS if v != 'run-w']
            + [('good-phrase', a, 'zib-w', True) for a in ADJS] + [('good-phrase', 'zib-w', a, False) for a in ADJS]
            + [('is-a', 'zib-w', c, c == 'noun') for c in ('noun', 'verb', 'adjective')])
    specs['zib-w'] = (given, read, (2, 3))
    return specs


new_relation_specs = W2.new_relation_specs
