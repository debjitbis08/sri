"""Regression tests for src.senses.KeptMemory: the stability invariant of docs/sense-stable-v1-pilot.md, on synthetic codes."""
import unittest

import numpy as np

from src import senses as SN
from src.sparse_local_memory import Memory

D = 8


class FixedGate:
    """A gate whose weights depend on the query: softmax of a fixed linear map of the raw unit codes."""
    def __init__(self, seed):
        self.W = np.random.default_rng(seed).normal(size=(2 * D, 2)).astype(np.float32)

    def __call__(self, raw):
        s = np.concatenate(raw, 1) @ self.W
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def parameters(self):
        return self.W.size


def cluster(rng, centre, n, spread=.3):
    return {m: (centre[m] + spread * rng.normal(size=(n, D))).astype(np.float32) for m in ('a', 'b')}


def snapshot(km):
    return [(m.h, m.a[:m.h].copy(), m.u[:m.h].copy(), m.n[:m.h].copy()) for m in km.parts]


class Invariant(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rng = rng
        self.cA = {m: rng.normal(size=D).astype(np.float32) * 3 for m in ('a', 'b')}
        self.cB = {m: rng.normal(size=D).astype(np.float32) * 3 for m in ('a', 'b')}
        self.cues, self.targets = cluster(rng, self.cA, 12), cluster(rng, self.cA, 12)
        self.queries = {m: (v + .05 * rng.normal(size=v.shape)).astype(np.float32) for m, v in self.cues.items()}  # other views of the cues

    def build(self, frozen):
        km = SN.KeptMemory(('a', 'b'), lambda: Memory(64, 1, .99, 2 * D, 2 * D), frozen_gate=frozen)
        km.open_centre(self.cA, FixedGate(1))
        km.write(self.cues, self.targets, np.zeros(12, int))
        return km

    def read_all(self, km):
        r = km.read(self.queries)
        return r['score'].copy(), r['centre'].copy(), r['unit'].copy(), km.decode(r, self.targets)

    def test_opening_a_centre_and_refitting_leave_old_reads_exactly_unchanged(self):
        km = self.build(frozen=True)
        before, units = self.read_all(km), snapshot(km)
        km.open_centre(self.cB, FixedGate(2))
        km.refit(FixedGate(3))
        after = self.read_all(km)
        for x, y in zip(before, after):
            np.testing.assert_array_equal(x, y)
        for (h0, a0, u0, n0), (h1, a1, u1, n1) in zip(units, snapshot(km)):
            self.assertEqual(h0, h1)
            np.testing.assert_array_equal(a0, a1)
            np.testing.assert_array_equal(u0, u1)
            np.testing.assert_array_equal(n0, n1)

    def test_without_frozen_gates_a_refit_moves_old_scores(self):
        km = self.build(frozen=False)
        before = self.read_all(km)[0]
        km.refit(FixedGate(3))
        self.assertGreater(np.abs(self.read_all(km)[0] - before).max(), 1e-4)

    def test_values_decode_on_the_winning_units_centre(self):
        km = self.build(frozen=True)
        np.testing.assert_array_equal(self.read_all(km)[3], np.arange(12))
        cuesB, targetsB = cluster(self.rng, self.cB, 6), cluster(self.rng, self.cB, 6)
        km.open_centre(self.cB, FixedGate(2))
        km.write(cuesB, targetsB, np.ones(6, int))
        r = km.read(cuesB)
        np.testing.assert_array_equal(r['centre'], 1)
        np.testing.assert_array_equal(km.decode(r, targetsB), np.arange(6))
        np.testing.assert_array_equal(self.read_all(km)[3], np.arange(12))

    def test_writes_never_touch_another_centres_units(self):
        km = self.build(frozen=True)
        km.open_centre(self.cB, FixedGate(2))
        old = snapshot(km)[0]
        km.write(cluster(self.rng, self.cA, 6), cluster(self.rng, self.cA, 6), np.ones(6, int))  # coded and stored under centre 1
        h, a, u, n = snapshot(km)[0]
        self.assertEqual(h, old[0])
        np.testing.assert_array_equal(a, old[1])
        np.testing.assert_array_equal(u, old[2])

    def test_a_write_that_merges_into_an_old_unit_does_change_it(self):
        """The invariant covers unchanged units only: a later write under the same centre can merge into an old unit (the interference case)."""
        km = SN.KeptMemory(('a', 'b'), lambda: Memory(64, 1, -1., 2 * D, 2 * D))  # tau -1: every write merges into the best unit
        km.open_centre(self.cA, FixedGate(1))
        km.write({m: v[:1] for m, v in self.cues.items()}, {m: v[:1] for m, v in self.targets.items()}, [0])
        a0 = km.parts[0].a[0].copy()
        km.write({m: v[1:2] for m, v in self.cues.items()}, {m: v[1:2] for m, v in self.targets.items()}, [0])
        self.assertEqual(km.parts[0].h, 1)
        self.assertGreater(np.abs(km.parts[0].a[0] - a0).max(), 1e-4)

    def test_state_counts_every_unit_array_and_allocation(self):
        km = self.build(frozen=True)
        s = km.state_bytes()
        per = 4 * (2 * D + 2 * D) + 4 * 2 * D + 8
        self.assertEqual(s['units_occupied'], km.parts[0].h * per)
        self.assertEqual(s['units_allocated'], 64 * per)
        self.assertEqual(s['gates'], 4 * 2 * D * 2)


class RawValues(unittest.TestCase):
    """values='raw' (docs/sense-memory-core-v1-design.md): the stored value is the target's raw codes, untouched by centres or gates."""
    def setUp(self):
        rng = np.random.default_rng(4)
        self.rng = rng
        self.cA = {m: rng.normal(size=D).astype(np.float32) * 3 for m in ('a', 'b')}
        self.cB = {m: rng.normal(size=D).astype(np.float32) * 3 for m in ('a', 'b')}
        self.cues, self.targets = cluster(rng, self.cA, 10), cluster(rng, self.cA, 10)

    def build(self, tau=.99):
        km = SN.KeptMemory(('a', 'b'), lambda: Memory(64, 1, tau, 2 * D, 2 * D), values='raw')
        km.open_centre(self.cA, FixedGate(1))
        return km

    def test_a_unit_holds_its_targets_raw_codes_and_reports_where_each_fact_went(self):
        km = self.build()
        w = km.write(self.cues, self.targets, np.zeros(10, int))
        self.assertEqual([k for k, _, _ in w], list(range(10)))
        raw = np.concatenate([self.targets['a'], self.targets['b']], 1)
        for k, j, units in w:
            self.assertEqual(j, 0)
            np.testing.assert_allclose(km.parts[0].u[units[0]] / km.parts[0].n[units[0]], raw[k], rtol=1e-6)  # tau .99: one unit per fact
        r = km.read(self.cues)
        np.testing.assert_allclose(r['value'], raw, rtol=1e-6)

    def test_merged_writes_average_raw_codes(self):
        km = self.build(tau=-1.)
        km.write({m: v[:3] for m, v in self.cues.items()}, {m: v[:3] for m, v in self.targets.items()}, np.zeros(3, int))
        raw = np.concatenate([self.targets['a'][:3], self.targets['b'][:3]], 1)
        np.testing.assert_allclose(km.read({m: v[:1] for m, v in self.cues.items()})['value'][0], raw.mean(0), rtol=1e-5)

    def test_raw_values_are_unchanged_by_new_centres_and_refits(self):
        km = self.build()
        km.write(self.cues, self.targets, np.zeros(10, int))
        before = km.read(self.cues)
        km.open_centre(self.cB, FixedGate(2))
        km.refit(FixedGate(3))
        after = km.read(self.cues)
        for k in ('centre', 'unit', 'score', 'value'):
            np.testing.assert_array_equal(before[k], after[k])

    def test_raw_values_cannot_be_decoded_in_key_coordinates(self):
        km = self.build()
        km.write(self.cues, self.targets, np.zeros(10, int))
        with self.assertRaises(AssertionError):
            km.decode(km.read(self.cues), self.targets)


if __name__ == '__main__':
    unittest.main()
