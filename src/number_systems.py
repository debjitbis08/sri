"""Generators for many number representations, organized by Lockhart's representation families.

Draft for docs/number-systems-v1-spec.md. Every representation shows the numbers 1-9 (dice:
1-6). Images are 28 x 28 float32 in [0, 1]; sequences are 128-sample float32 signals.
Families follow Lockhart, *Arithmetic* (2017), chapters "Language" to "India":

- collection: the number as the size of an arbitrarily arranged set (number permanence);
- grouping: repetition bundled into groups marked by a group symbol, with different group
  sizes (five-barred gate 5; Banana People 4; Tree People 7; Hand People 5, gestural);
- stacking: Egyptian strokes, at most four in a row, nine as three rows of three;
- subgroup symbols: Roman numerals in additive form (IIII, VIIII);
- distinct symbols: one arbitrary symbol per number (Chinese, Devanagari, Eastern Arabic and
  Thai digits; MNIST lives in src/number_concept.py);
- names: number words in several languages.

All variation is drawn from named streams; fonts are fixed system fonts listed below.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src import number_concept as nc
from src import number_core as ncore

NUMBERS = tuple(range(1, 10))
SIZE, SS = 28, 4  # canvas and supersampling factor
BRIGHTNESS, NOISE = (.7, 1.), .02

F = '/usr/share/fonts/'
FONTS = {
    'latin': [F + p for p in ('truetype/dejavu/DejaVuSans.ttf', 'truetype/dejavu/DejaVuSerif.ttf',
                              'truetype/dejavu/DejaVuSans-Bold.ttf', 'truetype/liberation/LiberationSans-Regular.ttf',
                              'truetype/liberation/LiberationSerif-Regular.ttf', 'truetype/freefont/FreeSans.ttf',
                              'truetype/freefont/FreeSerif.ttf', 'opentype/fira/FiraSans-Book.otf')],
    'devanagari': [F + p for p in ('truetype/noto/NotoSansDevanagari-Regular.ttf', 'truetype/noto/NotoSansDevanagari-Bold.ttf',
                                   'truetype/noto/NotoSerifDevanagari-Regular.ttf', 'truetype/freefont/FreeSans.ttf',
                                   'truetype/freefont/FreeSerif.ttf')],
    'arabic': [F + p for p in ('truetype/dejavu/DejaVuSans.ttf', 'truetype/noto/NotoSansArabic-Regular.ttf',
                               'truetype/noto/NotoNaskhArabic-Regular.ttf', 'truetype/noto/NotoKufiArabic-Regular.ttf',
                               'truetype/freefont/FreeSerif.ttf')],
    'thai': [F + p for p in ('truetype/noto/NotoSansThai-Regular.ttf', 'truetype/noto/NotoSerifThai-Regular.ttf',
                             'truetype/noto/NotoLoopedThai-Regular.ttf', 'truetype/freefont/FreeSerif.ttf')],
    'cjk': [(F + 'opentype/noto/NotoSansCJK-Regular.ttc', 2), (F + 'opentype/noto/NotoSerifCJK-Regular.ttc', 2),
            (F + 'truetype/arphic/ukai.ttc', 0), (F + 'truetype/droid/DroidSansFallbackFull.ttf', 0)],
}
TEXT = {  # system: (script, strings for 1..9)
    'roman': ('latin', ['I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'VIIII']),
    'chinese': ('cjk', list('一二三四五六七八九')),
    'devanagari_digits': ('devanagari', list('१२३४५६७८९')),
    'arabic_digits': ('arabic', list('١٢٣٤٥٦٧٨٩')),
    'thai_digits': ('thai', list('๑๒๓๔๕๖๗๘๙')),
    'english': ('latin', 'one two three four five six seven eight nine'.split()),
    'french': ('latin', 'un deux trois quatre cinq six sept huit neuf'.split()),
    'german': ('latin', 'eins zwei drei vier fünf sechs sieben acht neun'.split()),
    'hindi': ('devanagari', 'एक दो तीन चार पाँच छह सात आठ नौ'.split()),
}
EGYPTIAN_ROWS = {1: [1], 2: [2], 3: [3], 4: [4], 5: [3, 2], 6: [3, 3], 7: [4, 3], 8: [4, 4], 9: [3, 3, 3]}


# ---------------------------------------------------------------- raster helpers

def _finish(cover, rng):
    x = rng.uniform(*BRIGHTNESS) * cover + rng.normal(0, NOISE, (SIZE, SIZE))
    return np.clip(x, 0, 1).astype(np.float32)


def _place(segs, discs, hw, rng, margin=1.):
    """Shift a drawing (segments and discs in a local frame) to a random position that fits; return coverage."""
    pts = [np.array(segs).reshape(-1, 2)] if len(segs) else []
    if len(discs):
        d = np.array(discs)
        pts.append(np.concatenate([d[:, :2] - d[:, 2:3], d[:, :2] + d[:, 2:3]]))
    p = np.concatenate(pts)
    lo, hi = margin + hw, SIZE - margin - hw
    room = (hi - lo) - (p.max(0) - p.min(0))
    if (room < 0).any():
        raise AssertionError('drawing does not fit the canvas')
    off = lo - p.min(0) + rng.uniform(0, 1, 2) * room
    g = (np.arange(SIZE)[:, None] + nc.SUBPIXEL[None]).reshape(-1) + .5
    py, px = np.meshgrid(g, g, indexing='ij')
    inside = np.zeros(px.shape, bool)
    for x0, y0, x1, y1 in (np.array(segs) + np.r_[off, off] if len(segs) else []):
        vx, vy = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * vx + (py - y0) * vy) / max(vx * vx + vy * vy, 1e-9), 0, 1)
        inside |= (px - x0 - t * vx) ** 2 + (py - y0 - t * vy) ** 2 <= hw ** 2
    for cx, cy, r in (np.array(discs) + np.r_[off, 0] if len(discs) else []):
        inside |= (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
    return inside.reshape(SIZE, SS, SIZE, SS).mean((1, 3))


def _polyline(points):
    return [(*a, *b) for a, b in zip(points[:-1], points[1:])]


# ---------------------------------------------------------------- collections

def render_dots(n, rng, mode=None):
    """n non-overlapping discs at random positions. Half the images keep the radius fixed, half keep total area
    fixed (radius proportional to 1/sqrt(n)), so ink amount alone does not give the number. mode 'radius' or
    'area' forces one of the two (the default draw order is unchanged)."""
    fixed_radius = rng.random() < .5 if mode is None else mode == 'radius'
    r = rng.uniform(1.4, 2.2) if fixed_radius else rng.uniform(2.6, 3.4) / np.sqrt(n)
    r = max(r, .9)
    for _ in range(1_000):  # sequential placement with restarts
        c = []
        for _ in range(n):
            for _ in range(200):
                p = rng.uniform(3 + r, 25 - r, 2)
                if all(np.hypot(*(p - q)) >= 2 * r + 1.2 for q in c):
                    c.append(p)
                    break
            else:
                break
        if len(c) == n:
            c = np.array(c)
            break
    else:
        raise AssertionError('dot placement exhausted')
    discs = [(x, y, r) for x, y in c]
    return _finish(_place([], discs, 0., rng, margin=0.), rng)


def render_dots_incremental(rng, max_n=9, mode='radius'):
    """One collection grown one disc at a time: images for counts 1..max_n with persistent disc positions and one
    brightness, so adding one object adds exactly one disc (removing one removes the last). mode 'radius' keeps the
    radius fixed; 'area' keeps total area fixed (radius R / sqrt(k) at count k, at least 0.9), with spacing set at
    insertion so shrinking discs never overlap."""
    if mode == 'radius':
        R = rng.uniform(1.4, 2.2)
        radius = lambda k: R
    else:
        R = rng.uniform(2.6, 3.4)
        radius = lambda k: max(R / np.sqrt(k), .9)
    for _ in range(1_000):
        c = []
        for k in range(1, max_n + 1):
            r = radius(k)
            for _ in range(200):
                p = rng.uniform(3 + r, 25 - r, 2)
                if all(np.hypot(*(p - q)) >= 2 * r + 1.2 for q in c):
                    c.append(p)
                    break
            else:
                break
        if len(c) == max_n:
            break
    else:
        raise AssertionError('dot placement exhausted')
    g = (np.arange(SIZE)[:, None] + nc.SUBPIXEL[None]).reshape(-1) + .5
    py, px = np.meshgrid(g, g, indexing='ij')
    bright, out = rng.uniform(*BRIGHTNESS), []
    for k in range(1, max_n + 1):
        r = radius(k)
        inside = np.zeros(px.shape, bool)
        for cx, cy in c[:k]:
            inside |= (px - cx) ** 2 + (py - cy) ** 2 <= r ** 2
        cover = inside.reshape(SIZE, SS, SIZE, SS).mean((1, 3))
        out.append(np.clip(bright * cover + rng.normal(0, NOISE, (SIZE, SIZE)), 0, 1).astype(np.float32))
    return out


# ---------------------------------------------------------------- grouping systems

def render_egyptian(n, rng):
    """Stacked strokes, at most four per row; four is sometimes two over two."""
    rows = [2, 2] if n == 4 and rng.random() < .3 else EGYPTIAN_ROWS[n]
    spacing, gap = rng.uniform(2.6, 3.4), rng.uniform(2.2, 3.0)
    length = min(rng.uniform(7, 10), (23 - (len(rows) - 1) * gap) / len(rows))
    segs, y = [], 0.
    for k in rows:
        x0 = -(k - 1) * spacing / 2
        for j in range(k):
            x = x0 + j * spacing + rng.uniform(-.25, .25)
            segs.append((x, y + rng.uniform(-.3, .3), x + rng.uniform(-.4, .4), y + length + rng.uniform(-.3, .3)))
        y += length + gap
    return _finish(_place(segs, [], rng.uniform(.5, .8), rng), rng)


def _banana_glyphs(n, rng):
    """Banana People (group size 4): a square for each bunch of four, a curved stroke for each leftover."""
    groups, left = divmod(n, 4)
    segs, x = [], 0.
    for _ in range(groups):
        s = rng.uniform(6.5, 8)
        segs += _polyline([(x, 0), (x + s, 0), (x + s, s), (x, s), (x, 0)])
        x += s + rng.uniform(1.8, 2.6)
    for _ in range(left):
        h = rng.uniform(8, 10)
        t = np.linspace(-1, 1, 7)
        segs += _polyline([(x + 1.6 * (1 - u * u), h / 2 * (1 + u)) for u in t])
        x += rng.uniform(3.0, 3.6)
    return segs


def render_banana(n, rng):
    return _finish(_place(_banana_glyphs(n, rng), [], rng.uniform(.45, .7), rng), rng)


def render_tree(n, rng):
    """Tree People (group size 7): a tree sign for seven, small rings for leftovers."""
    groups, left = divmod(n, 7)
    segs, x = [], 0.
    for _ in range(groups):
        h, w = rng.uniform(12, 15), rng.uniform(7, 9)
        segs += _polyline([(x, h * .55), (x + w / 2, 0), (x + w, h * .55), (x, h * .55)])  # canopy
        segs.append((x + w / 2, h * .55, x + w / 2, h))  # trunk
        x += w + rng.uniform(1.5, 2.5)
    rr, step = rng.uniform(1.4, 1.8), rng.uniform(4.2, 4.8)
    for j in range(left):  # rings in rows of at most three
        cx, cy = x + rr + (j % 3) * step, 2 + rr + (j // 3) * step
        t = np.linspace(0, 2 * np.pi, 9)
        segs += _polyline([(cx + rr * np.cos(a), cy + rr * np.sin(a)) for a in t])
    return _finish(_place(segs, [], rng.uniform(.4, .6), rng), rng)


def render_tally(n, rng):
    return ncore.render_tally(n, rng)


# ---------------------------------------------------------------- text systems

_font_cache = {}


def _font(spec, size):
    path, index = spec if isinstance(spec, tuple) else (spec, 0)
    key = (path, index, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.RAQM)
    return _font_cache[key]


def render_text(system, n, rng):
    """Render a string with a random font, case (Latin words), rotation, size and position, 4x supersampled."""
    script, strings = TEXT[system]
    s = strings[n - 1]
    if system in ('english', 'french', 'german') and rng.random() < .3:
        s = s.capitalize()
    fonts = FONTS[script]
    spec = fonts[rng.integers(len(fonts))]
    big = Image.new('L', (400, 200), 0)
    ImageDraw.Draw(big).text((20, 20), s, fill=255, font=_font(spec, 96))
    big = big.crop(big.getbbox()).rotate(rng.uniform(-8, 8), resample=Image.BICUBIC, expand=True, fillcolor=0)
    w, h = big.size
    box = (SIZE - 3) * SS
    scale = min(box / w, box / h) * rng.uniform(.65, 1.)
    if system in ('chinese', 'devanagari_digits', 'arabic_digits', 'thai_digits'):
        scale = min(scale, 18 * SS / h)  # single glyphs: cap height so they do not fill the canvas
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    glyph = np.asarray(big.resize((tw, th), Image.LANCZOS), np.float32) / 255
    canvas = np.zeros((SIZE * SS, SIZE * SS), np.float32)
    y = rng.integers(SS, SIZE * SS - th - SS + 1)
    x = rng.integers(SS, SIZE * SS - tw - SS + 1)
    canvas[y:y + th, x:x + tw] = np.clip(glyph, 0, 1)
    return _finish(canvas.reshape(SIZE, SS, SIZE, SS).mean((1, 3)), rng)


# ---------------------------------------------------------------- sequences

def render_hand(n, rng):
    """Hand People (group size 5, gestural): a thump (wide, softer pulse) per handful, a clap (narrow, sharp
    pulse) per leftover, as a 128-sample signal like the Morse signals."""
    u = rng.uniform(3., 5.)
    t, x = int(rng.integers(0, 5)), np.zeros(ncore.MORSE_LEN, np.float32)
    amp = rng.uniform(.7, 1.)
    groups, left = divmod(n, 5)
    for i, kind in enumerate(['thump'] * groups + ['clap'] * left):
        if i:
            t += max(1, int(round(u * 1.5 * (1 + rng.uniform(-.2, .2)))))
        d = max(1, int(round(u * (3 if kind == 'thump' else 1) * (1 + rng.uniform(-.2, .2)))))
        if t + d > ncore.MORSE_LEN:
            raise AssertionError('hand signal overflows the window')
        x[t:t + d] = amp * (.55 if kind == 'thump' else 1.)
        t += d
    return x + rng.normal(0, ncore.MORSE_NOISE, ncore.MORSE_LEN).astype(np.float32)


def render_morse(n, rng):
    return ncore.morse_signal(ncore.MORSE[n], rng)[0]


def render_taps(n, rng, mode=None):
    """A heard count: n identical pulses as a 128-sample signal. 'rate' keeps one pulse length and gap per signal, so
    the signal lengthens with n; 'span' spreads the pulses over a fixed span, so length alone does not give the count.
    Not in SYSTEMS: used by number construction Part D as a second sense, so the default pools are unchanged."""
    rate = rng.random() < .5 if mode is None else mode == 'rate'
    if rate:
        d, gap = rng.uniform(3., 5.), rng.uniform(4., 6.)
    else:
        period = rng.uniform(80., 110.) / n
        d = min(rng.uniform(3., 5.), .45 * period)
        gap = period - d
    t, x, amp = rng.uniform(0., 5.), np.zeros(ncore.MORSE_LEN, np.float32), rng.uniform(.7, 1.)
    for _ in range(n):
        a = int(round(t))
        b = a + max(3, int(round(d * (1 + rng.uniform(-.2, .2)))))
        if b > ncore.MORSE_LEN:
            raise AssertionError('taps overflow the window')
        x[a:b] = amp
        t = b + max(3., gap * (1 + rng.uniform(-.2, .2)))
    return x + rng.normal(0, ncore.MORSE_NOISE, ncore.MORSE_LEN).astype(np.float32)


def count_pulses(x, threshold=.35):
    """Independent reader for tap signals: runs of at least two samples above threshold after 3-sample smoothing."""
    s = np.convolve(x, np.ones(3) / 3, mode='same') > threshold
    edges = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    return int(((ends - starts) >= 2).sum())


# ---------------------------------------------------------------- registry

SYSTEMS = {  # name: (family, medium, renderer)
    'dots': ('collection', 'image', render_dots),
    'tally': ('grouping-5 (gate)', 'image', render_tally),
    'banana': ('grouping-4', 'image', render_banana),
    'tree': ('grouping-7', 'image', render_tree),
    'egyptian': ('stacking', 'image', render_egyptian),
    'roman': ('subgroup symbols', 'image', lambda n, r: render_text('roman', n, r)),
    'chinese': ('distinct symbols', 'image', lambda n, r: render_text('chinese', n, r)),
    'devanagari_digits': ('distinct symbols', 'image', lambda n, r: render_text('devanagari_digits', n, r)),
    'arabic_digits': ('distinct symbols', 'image', lambda n, r: render_text('arabic_digits', n, r)),
    'thai_digits': ('distinct symbols', 'image', lambda n, r: render_text('thai_digits', n, r)),
    'english': ('names', 'image', lambda n, r: render_text('english', n, r)),
    'french': ('names', 'image', lambda n, r: render_text('french', n, r)),
    'german': ('names', 'image', lambda n, r: render_text('german', n, r)),
    'hindi': ('names', 'image', lambda n, r: render_text('hindi', n, r)),
    'hand': ('grouping-5 (gestural)', 'sequence', render_hand),
    'morse': ('code', 'sequence', render_morse),
}
EXISTING = {'mnist': ('distinct symbols', 'image'), 'dice': ('collection', 'image')}  # from src/number_concept.py


def render(system, n, rng):
    return SYSTEMS[system][2](n, rng)


def build(system, split, per_number):
    """Pool for one system: numbers 1-9, one named stream per (system, split, number, index)."""
    xs, ys = [], []
    for n in NUMBERS:
        for i in range(per_number):
            xs.append(render(system, n, nc.stream('system', system, split, n, i)))
            ys.append(n)
    return {'x': np.stack(xs), 'y': np.array(ys)}
