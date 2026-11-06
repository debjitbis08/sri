"""Multi-step reader v1 statements (docs/multistep-reader-v1.md): taught truth v1's statements plus chains.

L1 closures (trained on 1-2 steps, tested on 3+): kind-of (via is-a and subkind-of), ancestor-of (via parent-of, with 3
new 6-generation lineages), size bigger-than / smaller-than (via just-bigger-than). L2 higher-order relations:
named-can, named-has, named-kind. Deterministic.
"""
import numpy as np

from src import everyday_truth as T

W3, W1 = T.W3, T.W3.W1
SUPER = {'mammal': 'vertebrate', 'bird': 'vertebrate', 'fish': 'vertebrate', 'reptile': 'vertebrate',
         'amphibian': 'vertebrate', 'insect': 'invertebrate', 'vertebrate': 'animal', 'invertebrate': 'animal',
         'animal': 'living-thing'}
KINDS = tuple(W3.CLASSES) + ('vertebrate', 'invertebrate', 'animal', 'living-thing')
NONLIVING = ('sun', 'fire', 'snow', 'ice', 'coal', 'sky', 'thunder', 'water', 'milk', 'tea', 'soup', 'night-sky')
LINEAGES = [[f'line{c}{g}' for g in range(1, 7)] for c in 'abc']
L2_WORDS = ('cat-w', 'eagle-w', 'whale-w', 'lion-w', 'snake-w', 'horse-w')
NEW_RELS = ('subkind-of', 'kind-of', 'ancestor-of', 'just-bigger-than', 'named-can', 'named-has', 'named-kind')


def kind_chain(x):
    """Kinds of an animal with their depth: class 1, vertebrate/invertebrate 2, animal 3, living-thing 4."""
    out, k, d = {}, W3.ANIMALS[x][0], 1
    while True:
        out[k] = d
        if k not in SUPER:
            return out
        k, d = SUPER[k], d + 1


def parents():
    """child -> list of parents over the 42 people."""
    par = {}
    for m in (lambda x: x, W1.b):
        for c, ps in W1.PARENTS.items():
            par[m(c)] = [m(p) for p in ps]
    for line in LINEAGES:
        for g in range(1, len(line)):
            par[line[g]] = [line[g - 1]]
    return par


def ancestors(par):
    """(ancestor, descendant) -> depth."""
    out = {}
    for c in par:
        frontier, d = list(par[c]), 1
        while frontier:
            nxt = []
            for p in frontier:
                if (p, c) not in out:
                    out[(p, c)] = d
                nxt += par.get(p, [])
            frontier, d = nxt, d + 1
    return out


def build():
    st, world = T.build()
    keep = st.kind != 'fill'
    names, rels = list(st.entities), list(st.relations)
    for x in list(KINDS) + list(NONLIVING) + [p for line in LINEAGES for p in line]:
        if x not in names:
            names.append(x)
    rels += list(NEW_RELS)
    E = {x: i for i, x in enumerate(names)}
    R = {x: i for i, x in enumerate(rels)}
    V = {x: i for i, x in enumerate(T.VALUES)}
    rows = [tuple(x[i] for x in (st.a, st.r, st.b, st.v, st.gap, st.kind, st.test, st.sub, st.world_y))
            for i in np.flatnonzero(keep)]
    # size closures: bigger-than / smaller-than at distance >= 3 become L1-size tests
    size = W1.SIZE
    out = []
    for row in rows:
        a, r, b = row[0], row[1], row[2]
        rel = rels[r]
        if rel in ('bigger-than', 'smaller-than') and names[a] in size and names[b] in size:
            d = abs(size.index(names[a]) - size.index(names[b]))
            if d >= 3:
                row = row[:6] + ('L1-size', str(d)) + row[8:]
        out.append(row)
    rows = out

    def add(a, rel, b, y, test='', sub=''):
        rows.append((E[a], R[rel], E[b], V['T' if y else 'F'], 0, 'individual', test, sub, int(y)))

    # just-bigger-than (base, all trained)
    for i, a in enumerate(size):
        for j, b in enumerate(size):
            add(a, 'just-bigger-than', b, j == i + 1)
    # subkind-of (base, all trained)
    for k in KINDS[:-1] + ('living-thing',):
        for s in ('vertebrate', 'invertebrate', 'animal', 'living-thing'):
            add(k, 'subkind-of', s, SUPER.get(k) == s)
    # kind-of: animals (true at depth 1-4) and non-living things (all false)
    for x in W3.ANIMALS:
        ch = kind_chain(x)
        for k in KINDS:
            d = ch.get(k)
            held = k in ('animal', 'living-thing')
            add(x, 'kind-of', k, d is not None, 'L1-kind' if held else '', str(d or (3 if k == 'animal' else 4)) if held else '')
    for x in NONLIVING:
        for k in KINDS:
            held = k in ('animal', 'living-thing')
            add(x, 'kind-of', k, False, 'L1-kind' if held else '', ('3' if k == 'animal' else '4') if held else '')
    # parent-of for lineage people (Hinton people already have rows among themselves)
    par = parents()
    people = list(W1.TREE_A) + list(W1.TREE_B) + [p for line in LINEAGES for p in line]
    lineage = set(p for line in LINEAGES for p in line)
    for a in people:
        for b in people:
            if a in lineage or b in lineage:
                add(a, 'parent-of', b, a in par.get(b, []))
    # ancestor-of over all 42 people; depth >= 3 pairs and their reverses held out; tree-B colin column kept out (P6)
    anc = ancestors(par)
    colin_b = W1.b('colin')
    for a in people:
        for b in people:
            d = anc.get((a, b))
            rd = anc.get((b, a))
            if b == colin_b or a == colin_b:
                test, sub = 'P6-anc', ''
            elif d is not None and d >= 3:
                test, sub = 'L1-anc', str(d)
            elif rd is not None and rd >= 3:
                test, sub = 'L1-anc', str(rd)
            else:
                test, sub = '', ''
            add(a, 'ancestor-of', b, d is not None, test, sub)
    # higher-order relations over noun words
    t1 = set(W3.W1.T1_ANIMALS)
    for wd in W3.NOUNS:
        an = wd[:-2]
        if wd in L2_WORDS:
            test = 'L2-named'
        elif an in t1 or wd in W3.G2_WORDS or wd in W3.G3_WORDS:
            test = 'L2-x'   # would leak held-out facts, or the word's meaning is held out
        else:
            test = ''
        for ab in W1.ABILITIES:
            add(wd, 'named-can', ab, ab in W3.ANIMALS[an][2].split(), test, 'named-can')
        for p in W1.PARTS:
            add(wd, 'named-has', p, p in W3.ANIMALS[an][1].split(), test, 'named-has')
        ch = kind_chain(an)
        for k in KINDS:
            add(wd, 'named-kind', k, k in ch, test, 'named-kind')
    # fill-the-tail rows for trained true individual facts
    fills = {}
    for (a, r, b, v, gap, kind, tst, sub, wy) in rows:
        if kind == 'individual' and tst == '' and v == V['T']:
            fills.setdefault((a, r), []).append(b)
    n_before = len(rows)
    keys = sorted(fills)
    rows += [(a, r, -1, -1, 1, 'fill', '', '', -1) for a, r in keys]
    a, r, b, v, gap, kind, tst, sub, wy = zip(*rows)
    out = T.Statements(names, rels, np.array(a), np.array(r), np.array(b), np.array(v), np.array(gap), np.array(kind),
                       np.array(tst), np.array(sub), np.array(wy))
    out.fill_answers = {n_before + i: fills[k] for i, k in enumerate(keys)}
    return out, world


def chain_length(st):
    """Expected number of produce steps for each statement where it is defined (for the halting read); -1 otherwise."""
    L = np.full(len(st.a), -1)
    rel = np.array(st.relations)[st.r]
    for tst, m in (('L1-kind', rel == 'kind-of'), ('L1-anc', rel == 'ancestor-of'), ('L1-size', np.isin(rel, ('bigger-than', 'smaller-than')))):
        mm = m & (st.sub != '') & (st.test == tst)
        L[mm] = st.sub[mm].astype(int) - 1
    return L
