import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import number_concept as nc  # noqa: E402
from src import number_systems as ns  # noqa: E402


class NumberSystemsTests(unittest.TestCase):
    def test_every_system_renders_every_number(self):
        for name, (_, medium, _) in ns.SYSTEMS.items():
            for n in ns.NUMBERS:
                x = ns.render(name, n, nc.stream('test_render', name, n))
                self.assertEqual(x.shape, (28, 28) if medium == 'image' else (128,))
                self.assertEqual(x.dtype, np.float32)
                if medium == 'image':
                    self.assertTrue(0 <= x.min() and x.max() <= 1)
                    self.assertGreater(x.sum(), 1.)

    def test_deterministic_streams(self):
        for name in ns.SYSTEMS:
            a = ns.render(name, 7, nc.stream('test_det', name))
            b = ns.render(name, 7, nc.stream('test_det', name))
            self.assertTrue(np.array_equal(a, b))

    def test_dot_count_is_the_number(self):
        for n in ns.NUMBERS:
            for i in range(20):
                x = ns.render('dots', n, nc.stream('test_dots', n, i))
                self.assertEqual(ndimage.label(x > .35)[1], n)

    def test_incremental_dots_add_one_disc_per_step(self):
        for i in range(10):
            xs = ns.render_dots_incremental(nc.stream('test_dots_incremental', i), mode='radius' if i % 2 else 'area')
            masks = [x > .35 for x in xs]
            self.assertEqual([ndimage.label(m)[1] for m in masks], list(ns.NUMBERS))
            if i % 2:  # fixed radius: the previous discs stay (noise flips a few edge pixels at the threshold)
                self.assertTrue(all((masks[k] & masks[k + 1]).sum() >= .85 * masks[k].sum() for k in range(8)))
            else:  # fixed area: discs shrink but keep their centres, so every earlier centre stays inside a disc
                centres = ndimage.center_of_mass(masks[0], ndimage.label(masks[0])[0], 1)
                self.assertTrue(all(masks[k][int(round(centres[0])), int(round(centres[1]))] for k in range(9)))

    def test_taps_pulse_count_is_the_number(self):
        for mode in ('rate', 'span', None):
            for n in ns.NUMBERS:
                for i in range(20):
                    x = ns.render_taps(n, nc.stream('taps_test', str(mode), n, i), mode=mode)
                    self.assertEqual(x.shape, (128,))
                    self.assertEqual(ns.count_pulses(x), n)

    def test_structural_definitions(self):
        self.assertEqual([sum(ns.EGYPTIAN_ROWS[n]) for n in ns.NUMBERS], list(ns.NUMBERS))
        self.assertTrue(all(max(r) <= 4 for r in ns.EGYPTIAN_ROWS.values()))
        value = {'I': 1, 'V': 5}
        self.assertEqual([sum(value[c] for c in s) for s in ns.TEXT['roman'][1]], list(ns.NUMBERS))
        for name, (_, strings) in ns.TEXT.items():
            self.assertEqual(len(strings), 9, name)
            self.assertEqual(len(set(strings)), 9, name)


if __name__ == '__main__':
    unittest.main()
