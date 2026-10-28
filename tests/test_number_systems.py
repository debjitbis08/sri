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
