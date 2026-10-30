"""Symbols for the arbitrary-pairings study (docs/arbitrary-pairings-v1-contract.md, "Symbol selection").

Candidates are codepoints of four Unicode blocks. Excluded: characters whose name carries a number or a count (digits,
numerals, dice, tally, dots, braille), emoji presentation characters, and codepoints that any renderer font lacks. The
survivors are rendered at fixed parameters and greedily thinned so no two look alike. A seeded stream then draws two
disjoint sets of N symbols (A and B, N a multiple of nine) and balanced count assignments; a second assignment of A
(A') serves the learning check. Training and evaluation renderings use separate named streams.

Coverage is tested without a font-table library: a codepoint is covered by a font if its glyph differs from the font's
missing-glyph box and is not blank. Requiring all eight fonts leaves 5 of 569 candidates, so a symbol is kept if at least
MIN_FONTS fonts cover it and is rendered only in the fonts that cover it (no missing-glyph box is ever drawn).

Usage: .venv/bin/python -m experiments.pairing_symbols   (prints the survivor count and a few draws)
"""
import unicodedata

import numpy as np
from PIL import Image, ImageDraw

from experiments import number_systems as X

nc, ns = X.nc, X.ns
BLOCKS = {'Geometric Shapes': (0x25A0, 0x25FF), 'Arrows': (0x2190, 0x21FF), 'Dingbats': (0x2700, 0x27BF),
          'Miscellaneous Symbols': (0x2600, 0x26FF)}
NAME_EXCLUDED = ('DIGIT', 'NUMBER', 'ROMAN', 'TALLY', 'DIE ', 'DICE', 'FRACTION', 'BRAILLE', 'DOTS')
EMOJI = {*range(0x23E9, 0x23ED), 0x231A, 0x231B, 0x23F0, 0x23F3, 0x25FD, 0x25FE, 0x2614, 0x2615, *range(0x2648, 0x2654), 0x267F,
         0x2693, 0x26A1, 0x26AA, 0x26AB, 0x26BD, 0x26BE, 0x26C4, 0x26C5, 0x26CE, 0x26D4, 0x26EA, 0x26F2, 0x26F3, 0x26F5, 0x26FA,
         0x26FD, 0x2705, 0x270A, 0x270B, 0x2728, 0x274C, 0x274E, *range(0x2753, 0x2756), 0x2757, *range(0x2795, 0x2798), 0x27B0, 0x27BF}
FONT_LIST = ns.FONTS['latin']
DISTANCE_THRESHOLD = 0.35  # provisional; fixed by the seed-299 pilot before the contract is frozen
CANON_FONTS = 4
MIN_FONTS = 4
SIZE, SS = ns.SIZE, ns.SS


def candidates():
    out = []
    for lo, hi in BLOCKS.values():
        for cp in range(lo, hi + 1):
            try:
                name = unicodedata.name(chr(cp))
            except ValueError:
                continue
            if cp in EMOJI or any(w in name + ' ' for w in NAME_EXCLUDED):
                continue
            out.append(cp)
    return out


def _mask(spec, ch):
    big = Image.new('L', (200, 200), 0)
    ImageDraw.Draw(big).text((20, 20), ch, fill=255, font=ns._font(spec, 96))
    return np.asarray(big)


def covering(cps):
    """{codepoint: indices of the fonts in FONT_LIST that draw it as a non-blank glyph different from the missing-glyph box},
    for codepoints covered by at least MIN_FONTS fonts."""
    have = {cp: [] for cp in cps}
    for i, spec in enumerate(FONT_LIST):
        missing = _mask(spec, '\uffff')
        for cp in cps:
            if (m := _mask(spec, chr(cp))).any() and not np.array_equal(m, missing):
                have[cp].append(i)
    return {cp: f for cp, f in have.items() if len(f) >= MIN_FONTS}


def render(cp, rng, fonts):
    """One variable rendering of a symbol (random covering font, rotation, scale, position, brightness and noise), 28x28."""
    spec = FONT_LIST[fonts[rng.integers(len(fonts))]]
    big = Image.new('L', (400, 200), 0)
    ImageDraw.Draw(big).text((20, 20), chr(cp), fill=255, font=ns._font(spec, 96))
    big = big.crop(big.getbbox()).rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=0)
    w, h = big.size
    box = (SIZE - 3) * SS
    scale = min(min(box / w, box / h) * rng.uniform(.65, 1.), 18 * SS / h)
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    glyph = np.asarray(big.resize((tw, th), Image.LANCZOS), np.float32) / 255
    canvas = np.zeros((SIZE * SS, SIZE * SS), np.float32)
    y, x = rng.integers(SS, SIZE * SS - th - SS + 1), rng.integers(SS, SIZE * SS - tw - SS + 1)
    canvas[y:y + th, x:x + tw] = np.clip(glyph, 0, 1)
    return ns._finish(canvas.reshape(SIZE, SS, SIZE, SS).mean((1, 3)), rng)


def canonical(cp, fonts):
    """Noise-free rendering at fixed parameters, averaged over the first CANON_FONTS covering fonts, unit norm (for the distance filter)."""
    imgs = []
    for spec in [FONT_LIST[i] for i in fonts[:CANON_FONTS]]:
        big = Image.new('L', (400, 200), 0)
        ImageDraw.Draw(big).text((20, 20), chr(cp), fill=255, font=ns._font(spec, 96))
        big = big.crop(big.getbbox())
        w, h = big.size
        scale = min((SIZE - 3) * SS / w, (SIZE - 3) * SS / h, 18 * SS / h)
        tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        glyph = np.asarray(big.resize((tw, th), Image.LANCZOS), np.float32) / 255
        canvas = np.zeros((SIZE * SS, SIZE * SS), np.float32)
        y, x = (SIZE * SS - th) // 2, (SIZE * SS - tw) // 2
        canvas[y:y + th, x:x + tw] = glyph
        imgs.append(canvas.reshape(SIZE, SS, SIZE, SS).mean((1, 3)))
    v = np.mean(imgs, 0).ravel()
    return v / np.linalg.norm(v)


def survivors(threshold=DISTANCE_THRESHOLD):
    """{codepoint: covering fonts} for covered, non-excluded codepoints, thinned greedily (in codepoint order) so every kept
    pair is at least `threshold` apart."""
    cover = covering(candidates())
    kept, vecs = {}, []
    for cp, fonts in cover.items():
        v = canonical(cp, fonts)
        if all(np.linalg.norm(v - u) >= threshold for u in vecs):
            kept[cp] = fonts; vecs.append(v)
    return kept


def draw_sets(seed, n, pool):
    """Disjoint sets A and B of n symbols with balanced count assignments (9 counts, n a multiple of nine), plus a second
    assignment A' of A's symbols. Seeded by the seed alone, so identical across arms."""
    assert n % 9 == 0 and 2 * n <= len(pool)
    rng = nc.stream('pair_sets', seed)
    pick = [int(c) for c in rng.permutation(sorted(pool))[:2 * n]]
    def assign(r):
        return r.permutation(np.repeat(np.arange(1, 10), n // 9))
    a, b = pick[:n], pick[n:]
    return {'A': dict(zip(a, map(int, assign(nc.stream('pair_assign', seed, 'A'))))),
            'B': dict(zip(b, map(int, assign(nc.stream('pair_assign', seed, 'B'))))),
            "A'": dict(zip(a, map(int, assign(nc.stream('pair_assign', seed, "A'")))))}


def renderings(cp, fonts, seed, split, n):
    """n renderings of a symbol from the stream of (seed, symbol, split); 'train' and 'eval' never share a stream."""
    assert split in ('train', 'eval')
    return np.stack([render(cp, nc.stream('pair_render', seed, cp, split, i), fonts) for i in range(n)])


if __name__ == '__main__':
    c = candidates()
    cov = covering(c)
    s = survivors()
    print(f'candidates {len(c)}, covered by at least {MIN_FONTS} of {len(FONT_LIST)} fonts {len(cov)}, '
          f'after distance filter ({DISTANCE_THRESHOLD}) {len(s)}')
    sets = draw_sets(299, 18, s)
    print({k: list(v.items())[:3] for k, v in sets.items()})
