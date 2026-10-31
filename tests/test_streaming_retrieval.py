import unittest

import numpy as np

from src import streaming_retrieval as sr


def simple_schedule():
    """Word 0 for 40 frames, word 1 for 40, unfamiliar for 40, with one intrusion burst inside word 0."""
    truth = np.r_[np.full(40, 0), np.full(40, 1), np.full(40, sr.REJECT)]
    return {'truth': truth, 'segments': [(0, 40, 0), (40, 80, 1), (80, 120, sr.REJECT)], 'bursts': [(20, 2, 'intrude')]}


class Scores(unittest.TestCase):
    def test_perfect_decisions(self):
        sch = simple_schedule()
        r = sr.score(sch['truth'], sch)
        self.assertEqual(r['wrong_per_1000'], 0)
        self.assertEqual(r['delay_mean'], 0)
        self.assertEqual(r['accept_unfamiliar'], 0)
        self.assertEqual(r['dropout'], 0)
        self.assertEqual(r['settled_correct'], 1)
        self.assertEqual(r['capture_by_kind']['intrude'], 0)

    def test_capture_delay_stale_and_dropout_are_separate(self):
        sch = simple_schedule()
        y = sch['truth'].copy()
        y[20:22] = 5        # captured by the intrusion
        y[40:46] = 0        # six frames late on the real change
        y[80:] = 1          # never rejects
        y[60:62] = sr.REJECT
        r = sr.score(y, sch)
        # the only switches to a wrong identity: 0 -> 5 in the burst. Returning to 0 is correct. Staying on 1 is staleness, not a switch.
        self.assertAlmostEqual(r['wrong_per_1000'], 1000 / 120)
        self.assertEqual(r['capture_by_kind']['intrude'], 1)
        self.assertEqual(r['delay_mean'], 6)
        self.assertEqual(r['accept_unfamiliar'], 1)
        self.assertEqual(r['reject_miss'], 1)
        self.assertAlmostEqual(r['dropout'], 2 / (24 - 10 + 24))

    def test_a_missed_change_is_censored_at_the_dwell(self):
        sch = simple_schedule()
        y = sch['truth'].copy()
        y[40:80] = 0
        r = sr.score(y, sch)
        self.assertEqual(r['delay_mean'], 40)
        self.assertEqual(r['delay_miss'], 1)

    def test_a_decision_must_hold_to_count(self):
        sch = simple_schedule()
        y = sch['truth'].copy()
        y[40:50] = 0
        y[44] = 1           # one correct frame does not count as reaching the word
        self.assertEqual(sr.score(y, sch)['delay_mean'], 10)


class Schedules(unittest.TestCase):
    def test_structure(self):
        sch = sr.make_schedule(np.random.default_rng(0), 30, 10, 20000)
        self.assertEqual(len(sch['truth']), len(sch['shown']))
        for (a, b, word), (c, _, next_word) in zip(sch['segments'], sch['segments'][1:]):
            self.assertEqual(b, c)
            self.assertNotEqual(word, next_word)
            self.assertTrue(sr.MIN_DWELL <= b - a <= sr.MAX_DWELL)
        fam = sch['truth'] != sr.REJECT
        off = (sch['shown'] != sch['truth']) & fam
        self.assertTrue(sch['corrupt'].sum() + off.sum() > 0)
        where = {s: (a, b, w) for a, b, w in sch['segments'] for s in range(a, b, 1)}
        for start, length, kind in sch['bursts']:
            a, b, word = where[start]
            self.assertNotEqual(word, sr.REJECT)
            self.assertGreaterEqual(start - a, sr.ONSET_CLEAR)
            self.assertGreaterEqual(b - start - length, sr.AFTER_CLEAR)
            if kind == 'intrude':
                self.assertTrue((sch['shown'][start:start + length] < 30).all())
                self.assertTrue((sch['shown'][start:start + length] != word).all())
            if kind == 'flash':
                self.assertTrue((sch['shown'][start:start + length] >= 30).all())
            if kind == 'corrupt':
                self.assertTrue(sch['corrupt'][start:start + length].all())
        self.assertAlmostEqual(np.mean([k[2] == sr.REJECT for k in sch['segments']]), sr.P_UNFAMILIAR / (1 + sr.P_UNFAMILIAR), delta=0.05)

    def test_deterministic(self):
        a = sr.make_schedule(np.random.default_rng(3), 30, 10, 2000)
        b = sr.make_schedule(np.random.default_rng(3), 30, 10, 2000)
        np.testing.assert_array_equal(a['shown'], b['shown'])


class Methods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        keys = rng.standard_normal((20, 32))
        self.keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
        self.rng = rng

    def test_nearest_recovers_keys(self):
        i, s = sr.nearest(self.keys @ self.keys.T)
        np.testing.assert_array_equal(i, np.arange(20))
        np.testing.assert_allclose(s, 1)

    def test_ema_equals_nearest_on_the_smoothed_query(self):
        q = self.rng.standard_normal((50, 32))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        S = q @ self.keys.T
        index, score = sr.ema(S, 0.2)
        qbar = q[0].copy()
        for t in range(50):
            qbar += 0.2 * (q[t] - qbar)
            sims = self.keys @ qbar
            self.assertEqual(index[t], sims.argmax())
            self.assertAlmostEqual(score[t], sims.max())

    def test_ema_at_one_is_nearest(self):
        S = self.keys @ self.keys.T
        np.testing.assert_array_equal(sr.ema(S, 1.0)[0], sr.nearest(S)[0])

    def test_hysteresis_holds_against_a_small_margin(self):
        S = np.tile(np.array([[.8, .1]]), (10, 1))
        S = np.r_[S, np.tile(np.array([[.5, .6]]), (5, 1))]
        np.testing.assert_array_equal(sr.hysteresis(S, 1.0, .3, .2)[-1:], [0])
        np.testing.assert_array_equal(sr.hysteresis(S, 1.0, .3, .05)[-1:], [1])

    def test_hysteresis_rejects_below_its_lower_threshold(self):
        S = np.r_[np.tile(np.array([[.8, .1]]), (5, 1)), np.tile(np.array([[.1, .1]]), (5, 1))]
        self.assertEqual(sr.hysteresis(S, 1.0, .3, .1)[-1], sr.REJECT)

    def test_hmm_filter_resists_a_glitch_follows_a_change_and_rejects(self):
        hi, lo = .9, .1
        A, B, none = [hi, lo], [lo, hi], [lo, lo]
        S = np.array([A] * 20 + [B] + [A] * 10 + [B] * 20 + [none] * 20)
        y = sr.hmm_filter(S, 10., 1e-3, .5)
        self.assertEqual(y[20], 0)
        self.assertEqual(y[50], 1)
        self.assertEqual(y[-1], sr.REJECT)

    def test_ops_are_ordered(self):
        self.assertLess(sr.ops('nearest', 120, 32), sr.ops('ema', 120, 32))
        self.assertLess(sr.ops('ema', 120, 32), sr.ops('idp', 120, 32, 1))
        self.assertLess(sr.ops('idp', 120, 32, 1), sr.ops('idp', 120, 32, 2))
        self.assertLess(sr.ops('idp', 120, 32, 2), sr.ops('idp', 120, 32, 2, N=1024))


class PublishedBehaviour(unittest.TestCase):
    """The paper's setting: independent +-1 patterns, N = 1024, tanh(5x), dt = 0.01; saliency weights set directly."""

    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.xi = np.sign(self.rng.standard_normal((8, 1024)))

    def run_alpha(self, alpha, sigma=0., tau=.1, steps=10):
        """alpha (T, P) as similarity^2 with unit gain."""
        return sr.idp(np.sqrt(alpha), self.xi, 1., tau, steps, sigma, self.rng)

    def piece(self, dominant, frames, low=.5):
        a = np.full((frames, 8), low)
        a[:, dominant] = 8.
        return a

    def test_retrieves_the_dominant_memory_and_tracks_changes(self):
        alpha = np.r_[self.piece(0, 100), self.piece(1, 100)]
        index, score = self.run_alpha(alpha, sigma=2.)
        self.assertEqual(index[99], 0)
        self.assertEqual(index[199], 1)
        self.assertGreater(score[99], .9)
        self.assertGreater(score[199], .9)

    def test_noise_shortens_the_switch(self):
        alpha = np.r_[self.piece(0, 100), self.piece(1, 100)]
        quiet = np.argmax(self.run_alpha(alpha)[0][100:] == 1)
        noisy = np.argmax(self.run_alpha(alpha, sigma=2.)[0][100:] == 1)
        self.assertLess(noisy, quiet)

    def test_a_brief_glitch_leaves_the_memory_in_place(self):
        alpha = self.piece(0, 200)
        alpha[100, 0], alpha[100, 1] = .5, 8.
        index, score = self.run_alpha(alpha)
        self.assertEqual(index[99], 0)
        self.assertTrue((index[100:] == 0).all())
        self.assertGreater(score[199], .9)

    def test_vague_input_leaves_the_origin(self):
        alpha = np.full((200, 8), .02)
        _, score = self.run_alpha(alpha)
        self.assertLess(score[-1], .1)

    def test_deterministic_given_the_generator(self):
        alpha = self.piece(0, 20)
        a = sr.idp(np.sqrt(alpha), self.xi, 1., .1, 2, 1., np.random.default_rng(5))
        b = sr.idp(np.sqrt(alpha), self.xi, 1., .1, 2, 1., np.random.default_rng(5))
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_factored_step_matches_the_dense_matrix(self):
        xi = self.xi
        x = self.rng.standard_normal(1024)
        a = self.rng.random(8)
        W = (xi.T * a) @ xi / 1024
        np.testing.assert_allclose(W @ np.tanh(5 * x), xi.T @ (a * (xi @ np.tanh(5 * x) / 1024)))


if __name__ == '__main__':
    unittest.main()
