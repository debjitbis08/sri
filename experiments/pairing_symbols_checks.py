"""Structural checks of experiments/pairing_symbols.py. Usage: .venv/bin/python -m experiments.pairing_symbols_checks"""
import unicodedata
from collections import Counter

import numpy as np

from experiments import pairing_symbols as S


def main():
    pool = S.survivors()
    assert len(pool) >= 2 * 90, len(pool)
    assert all(len(f) >= S.MIN_FONTS for f in pool.values())
    assert all(not any(w in unicodedata.name(chr(cp)) + ' ' for w in S.NAME_EXCLUDED) and cp not in S.EMOJI for cp in pool)
    for n in (18, 45, 90):
        sets = S.draw_sets(321, n, pool)
        assert sets == S.draw_sets(321, n, pool), 'draw is not deterministic'
        a, b, a2 = sets['A'], sets['B'], sets["A'"]
        assert len(a) == len(b) == n and not set(a) & set(b) and set(a) == set(a2)
        for d in (a, b, a2):
            assert set(Counter(d.values()).values()) == {n // 9} and set(d.values()) == set(range(1, 10))
        assert a != a2, "A' must be a different assignment"
    assert S.draw_sets(321, 18, pool)['A'] != S.draw_sets(322, 18, pool)['A']
    cp = next(iter(pool))
    tr, ev = S.renderings(cp, pool[cp], 321, 'train', 4), S.renderings(cp, pool[cp], 321, 'eval', 4)
    assert tr.shape == (4, 28, 28) and tr.max() > .3 and tr.min() >= 0
    assert np.array_equal(tr, S.renderings(cp, pool[cp], 321, 'train', 4)), 'renderings are not reproducible'
    assert not any(np.array_equal(x, y) for x in tr for y in ev), 'train and eval renderings share an image'
    print(f'ok: {len(pool)} symbols; draws, balance, disjointness, determinism and train/eval independence hold')


if __name__ == '__main__':
    main()
