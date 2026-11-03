import unittest

import numpy as np

from experiments.general_encoder import KindCentres, N, resemblance, ANIMALS_HELD, ANIMALS_SEEN, CENTRINGS, COARSE, DIETS, HELDOUT_ARMS, NOT_ANIMALS, arm_name, centred, coarse_noise, dead_leaves, few_shot, memory_arms, nearest_label, score


class Centring(unittest.TestCase):
    def test_codes_piled_on_one_point_spread_out_when_centred(self):
        rng = np.random.default_rng(0)
        c = np.eye(16, dtype=np.float32)[0]
        z = c + .01 * rng.normal(size=(50, 16)).astype(np.float32)
        z = z / np.linalg.norm(z, axis=1, keepdims=True)
        before = z @ z.T
        y = centred(z, z.mean(0))
        after = y @ y.T
        np.fill_diagonal(before, 0)
        np.fill_diagonal(after, 0)
        self.assertGreater(before.mean(), .9)
        self.assertLess(abs(after.mean()), .1)
        self.assertLess(np.abs(np.linalg.norm(y, axis=1) - 1).max(), 1e-5)

    def test_a_code_at_the_centre_stays_zero_and_finite(self):
        y = centred(np.array([[1., 0.]], np.float32), np.array([1., 0.], np.float32))
        self.assertTrue(np.isfinite(y).all())
        self.assertEqual(float(np.abs(y).sum()), 0.)


class Measures(unittest.TestCase):
    def test_score_counts_abstentions_as_wrong(self):
        got = score(np.array([True, True]), np.array([.9, .1]), np.array([True, False]), np.array([.1, .2]), .5)
        self.assertEqual(got['tau']['address'], .5)
        self.assertEqual(got['answer-all']['address'], 1.)
        self.assertEqual(got['answer-all']['value'], .5)
        self.assertEqual(got['unseen_at_tau'], 0.)

    def test_nearest_label(self):
        K = np.eye(2)
        self.assertEqual(nearest_label(K, np.array([[1., 0.], [1., 0.]]), np.array([3, 4]), np.array([3, 4])), .5)

    def test_few_shot_finds_clusters_and_is_near_chance_on_noise(self):
        rng = np.random.default_rng(1)
        centres = np.linalg.qr(rng.normal(size=(16, 16)))[0][:4]
        y = np.repeat(np.arange(4), 30)
        z = centres[y] + .05 * rng.normal(size=(120, 16))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        self.assertGreater(few_shot(z, y, 1, rng), .95)
        noise = rng.normal(size=(120, 16))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        self.assertLess(few_shot(noise, y, 1, rng), .5)


class Resemblance(unittest.TestCase):
    def test_answers_reaching_the_same_kind_and_category_are_counted(self):
        cue_labels = np.zeros((2, N), np.int64)
        cue_labels[1, 3] = 7
        rows = np.array([N + 3, 3, N + 4])  # kind 1 fact 3 (label 7), kind 0, kind 1 fact 4 (label 0)
        got = resemblance(rows, np.array([.9, .9, .1]), .5, 1, np.array([7, 7, 0]), cue_labels)
        self.assertEqual(got['answered_same_kind'], .5)
        self.assertEqual(got['answered_same_class'], .5)
        self.assertAlmostEqual(got['all_same_class'], 2 / 3)

    def test_no_answers_gives_zero_not_an_error(self):
        got = resemblance(np.array([0]), np.array([.1]), .5, 0, np.array([0]), np.zeros((1, N), np.int64))
        self.assertEqual(got['answered_same_kind'], 0.)


class Centres(unittest.TestCase):
    def kinds(self, rng, n=300):
        a, b = np.zeros(32, np.float32), np.zeros(32, np.float32)
        a[0], b[1] = 5., 5.
        return [x + rng.normal(size=(n, 32)).astype(np.float32) for x in (a, b)]

    def test_two_kinds_arriving_in_turn_give_two_centres_and_are_told_apart(self):
        rng = np.random.default_rng(4)
        ka, kb = self.kinds(rng)
        c = KindCentres(form=100)
        for z in np.concatenate([ka[:200], kb[:200]]):
            c.observe(z)
        self.assertEqual(len(c.centres), 2)
        self.assertTrue((c.assign(ka[200:]) == 0).all())
        self.assertTrue((c.assign(kb[200:]) == 1).all())

    def test_more_of_a_known_kind_makes_no_new_centre_and_frozen_centres_do_not_move(self):
        rng = np.random.default_rng(5)
        ka, _ = self.kinds(rng)
        c = KindCentres(form=100)
        for z in ka[:100]:
            c.observe(z)
        before = c.centres[0].copy()
        for z in ka[100:]:
            c.observe(z)
        self.assertEqual(len(c.centres), 1)
        self.assertLess(len(c.forming), 10)
        np.testing.assert_array_equal(c.centres[0], before)


class Diets(unittest.TestCase):
    def test_dead_leaves_are_images_with_many_grey_levels(self):
        x = dead_leaves(4, np.random.default_rng(2))
        self.assertEqual(x.shape, (4, 28, 28))
        self.assertTrue((x >= 0).all() and (x <= 1).all())
        self.assertGreater(len(np.unique(x[0])), 5)

    def test_coarse_noise_is_blocks(self):
        x = coarse_noise(2, np.random.default_rng(3))
        self.assertEqual(x.shape, (2, 28, 28))
        self.assertTrue((x[0, :4, :4] == x[0, 0, 0]).all())

    def test_diet_names_are_distinct(self):
        self.assertEqual(len(set(DIETS)), len(DIETS))


class HeldOut(unittest.TestCase):
    def test_no_held_category_is_in_a_partial_diet(self):
        for arm in ('no-animals', 'some-animals'):
            self.assertFalse(set(HELDOUT_ARMS[arm]) & set(ANIMALS_HELD))
        self.assertFalse(set(HELDOUT_ARMS['no-animals']) & set(ANIMALS_SEEN))

    def test_the_groups_cover_all_but_people_and_do_not_overlap(self):
        groups = list(NOT_ANIMALS) + list(ANIMALS_SEEN) + list(ANIMALS_HELD)
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(set(range(20)) - set(groups), {COARSE.index('people')})
        self.assertEqual(COARSE[8], 'large_carnivores')
        self.assertEqual(COARSE[15], 'reptiles')
        self.assertEqual(COARSE[19], 'vehicles_2')


class Arms(unittest.TestCase):
    def test_names_are_unique_and_baselines_are_present(self):
        names = [arm_name(a) for a in memory_arms()]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn('pca/code/none', names)
        self.assertIn('glyph-small/key/none', names)
        for c in CENTRINGS:
            self.assertIn(f'natural-deep/key/{c}', names)


if __name__ == '__main__':
    unittest.main()
