import unittest

import numpy as np
import torch

from experiments.natural_memory import LOAD, VIEWS, DeepEncoder, Tagged, auc, fact_index, instance_scores, knn_class, matched_threshold, partners, plan_writes, read_scores, move_memory, sdc_move, set_facts, split_test, taus_for, unit, views


class Views(unittest.TestCase):
    def test_grey_and_colour_keep_their_shape_and_range(self):
        rng = np.random.default_rng(0)
        for x in (torch.rand(8, 28, 28), torch.rand(8, 3, 32, 32)):
            for cfg in VIEWS.values():
                y = views(x, rng, cfg)
                self.assertEqual(y.shape, x.shape)
                self.assertGreaterEqual(float(y.min()), 0.)
                self.assertLessEqual(float(y.max()), 1.)

    def test_a_crop_with_no_jitter_keeps_a_flat_image_and_a_flip_reverses_a_ramp(self):
        flat = torch.full((4, 28, 28), .3)
        y = views(flat, np.random.default_rng(1), dict(area=.5, jitter=0., flip=True, grey=0.))
        self.assertLess(float((y - flat).abs().max()), 1e-5)
        ramp = torch.linspace(0, 1, 28)[None, None].repeat(64, 28, 1)
        y = views(ramp, np.random.default_rng(1), dict(area=1., jitter=0., flip=True, grey=0.))
        rising = y[:, 14, 27] > y[:, 14, 0]
        self.assertTrue(.2 < float(rising.float().mean()) < .8)

    def test_strong_views_differ_more_than_mild(self):
        x = torch.rand(1, 28, 28).repeat(64, 1, 1)
        x = torch.nn.functional.avg_pool2d(x[:, None], 5, 1, 2)[:, 0]
        rng = np.random.default_rng(2)
        self.assertGreater(float((views(x, rng, VIEWS['strong']) - x).abs().mean()), float((views(x, rng, VIEWS['mild']) - x).abs().mean()))


class Measures(unittest.TestCase):
    def test_auc_of_separated_tied_and_reversed_scores(self):
        self.assertEqual(auc([2., 3.], [0., 1.]), 1.)
        self.assertEqual(auc([0., 1.], [2., 3.]), 0.)
        self.assertEqual(auc([1., 1.], [1., 1.]), .5)

    def test_identical_views_are_found(self):
        g = unit(np.random.default_rng(3).normal(size=(50, 16)).astype(np.float32))
        u = unit(np.random.default_rng(4).normal(size=(50, 16)).astype(np.float32))
        got = instance_scores(g, g.copy(), u)
        self.assertEqual(got['top1'], 1.)
        self.assertAlmostEqual(got['own_median'], 1., places=5)
        self.assertEqual(got['auc_unseen'], 1.)
        self.assertEqual(got['own_at_0.95'], 1.)

    def test_scores_do_not_let_an_image_count_as_its_own_other(self):
        g = np.eye(4, dtype=np.float32)
        self.assertLess(instance_scores(g, g.copy(), g.copy())['other_median'], .5)

    def test_neighbours_vote_for_clusters(self):
        rng = np.random.default_rng(5)
        centres = unit(rng.normal(size=(3, 16)))
        y = rng.integers(3, size=300)
        z = unit(centres[y] + .05 * rng.normal(size=(300, 16))).astype(np.float32)
        got = knn_class(z[:200], y[:200], z[200:], y[200:], k=5, classes=3)
        self.assertEqual(got['knn'], 1.)
        self.assertEqual(got['nearest'], 1.)


class Split(unittest.TestCase):
    def test_parts_do_not_overlap_except_gallery_in_queries(self):
        sp = split_test(392)
        self.assertEqual(len(set(sp['special']) & set(sp['write'])), 0)
        self.assertEqual(len(set(sp['special']) & set(sp['query'])), 0)
        self.assertEqual(len(set(sp['write']) & set(sp['query'])), 0)
        self.assertTrue(set(sp['gallery']) <= set(sp['query']))
        self.assertEqual(len(sp['special']) + len(sp['write']) + len(sp['query']), 10000)

    def test_a_seed_gives_its_own_split(self):
        self.assertFalse(np.array_equal(split_test(392)['special'], split_test(393)['special']))


class Networks(unittest.TestCase):
    def test_keys_are_unit_length_for_grey_and_colour(self):
        for channels, x in ((1, torch.rand(3, 28, 28)), (3, torch.rand(3, 3, 32, 32))):
            z = DeepEncoder(channels)(x)
            self.assertEqual(z.shape, (3, 32))
            self.assertLess(float((z.norm(dim=1) - 1).abs().max()), 1e-5)

    def test_partners_are_the_two_views_of_a_base(self):
        self.assertEqual(partners(2).tolist(), [1, 0, 3, 2])


class Memory(unittest.TestCase):
    def test_nearest_agrees_with_the_memorys_own_one_winner_read(self):
        rng = np.random.default_rng(6)
        m = Tagged(64, .85, 16, 4, 8)
        z = unit(rng.normal(size=(40, 16)).astype(np.float32))
        for i, x in enumerate(z):
            m.write_tagged(x, np.eye(4, dtype=np.float32)[i % 4], i % 8)
        Q = unit(z + .05 * rng.normal(size=z.shape).astype(np.float32))
        content, ok = m.read(Q)
        i, s = m.nearest(Q)
        np.testing.assert_array_equal(ok, s >= .85)
        np.testing.assert_allclose(content[ok], (m.u[:m.h] / m.n[:m.h, None])[i[ok]])

    def test_tags_say_what_a_unit_took_and_are_not_read_by_it(self):
        m = Tagged(8, .9, 2, 1, 3)
        z = np.array([1., 0.], np.float32)
        m.write_tagged(z, np.ones(1, np.float32), 2)
        m.write_tagged(z, np.ones(1, np.float32), 2)
        self.assertEqual(m.h, 1)
        self.assertEqual(m.tags[0].tolist(), [0, 0, 2])

    def test_a_write_past_capacity_goes_to_the_nearest_unit(self):
        m = Tagged(1, .9, 2, 1, 2)
        m.write_tagged(np.array([1., 0.], np.float32), np.ones(1, np.float32), 0)
        m.write_tagged(np.array([0., 1.], np.float32), np.ones(1, np.float32), 1)
        self.assertEqual((m.h, m.forced), (1, 1))
        self.assertEqual(m.tags[0].tolist(), [1, 1])

    def test_abstentions_are_wrong_and_coverage_counts_them(self):
        tags = np.array([[5, 0], [0, 5]])
        got = read_scores(np.array([0, 1, 1]), np.array([.9, .9, .1]), .5, np.array([0, 1, 1]), tags, np.array([7, 7, 7]), np.array([7, 8, 7]))
        self.assertAlmostEqual(got['coverage'], 2 / 3)
        self.assertAlmostEqual(got['address'], 2 / 3)
        self.assertAlmostEqual(got['value'], 1 / 3)

    def test_matched_threshold_answers_the_stated_share(self):
        s = np.linspace(0, 1, 10001)
        self.assertAlmostEqual(float((s >= matched_threshold(s, .13)).mean()), .13, places=2)


class Transport(unittest.TestCase):
    def test_sdc_moves_a_point_with_its_neighbours_not_with_far_inputs(self):
        old = unit(np.array([[1., 0., 0.], [0., 0., 1.]], np.float32))
        new = unit(np.array([[.9, .1, 0.], [0., 1., 0.]], np.float32))
        got = sdc_move(np.array([[1., 0., 0.]], np.float32), old, new, sigma=.1)
        np.testing.assert_allclose(got[0], unit(np.array([.9, .1, 0.], np.float32)), atol=1e-4)

    def test_no_movement_leaves_points_in_place(self):
        rng = np.random.default_rng(8)
        old = unit(rng.normal(size=(20, 8)).astype(np.float32))
        p = unit(rng.normal(size=(5, 8)).astype(np.float32))
        np.testing.assert_allclose(sdc_move(p, old, old), p, atol=1e-5)

    def test_a_far_point_gets_a_defined_move(self):
        old = np.array([[1., 0.]], np.float32)
        got = sdc_move(np.array([[-1., 0.]], np.float32), old, np.array([[0., 1.]], np.float32), sigma=.01)
        self.assertTrue(np.isfinite(got).all())

    def test_moving_keeps_unit_size_and_mean_content_form(self):
        m = Tagged(4, .9, 2, 2, 1)
        m.write_tagged(np.array([1., 0.], np.float32), np.array([1., 0.], np.float32), 0)
        m.write_tagged(np.array([1., 0.], np.float32), np.array([1., 0.], np.float32), 0)
        move_memory(m, lambda a: a[:, ::-1])
        np.testing.assert_allclose(m.a[0], [0., 1.], atol=1e-6)
        np.testing.assert_allclose(m.c[0], [0., 2.], atol=1e-6)
        np.testing.assert_allclose(m.u[0], [0., 2.], atol=1e-6)


class Facts(unittest.TestCase):
    def test_every_fact_is_written_as_often_and_in_order_of_a_block(self):
        rng = np.random.default_rng(7)
        plan = plan_writes(rng, np.arange(45), 8, 2000)
        self.assertEqual(len(plan), 250)
        got = np.concatenate(plan)
        self.assertEqual(len(got), 360)
        np.testing.assert_array_equal(np.bincount(got), np.full(45, 8))

    def test_the_sets_do_not_share_a_fact_at_any_level(self):
        for lv in ('instance-45', 'instance-450', 'class'):
            self.assertEqual(len(set(set_facts(lv, 'A')) & set(set_facts(lv, 'B'))), 0)

    def test_table_rows_fill_a_small_load_without_gaps(self):
        rows = fact_index('instance-45', np.r_[set_facts('instance-45', 'A'), set_facts('instance-45', 'B')])
        np.testing.assert_array_equal(rows, np.arange(90))
        self.assertEqual(set_facts('instance-450', 'B')[0], LOAD)

    def test_pixels_keep_one_threshold(self):
        self.assertEqual(len(taus_for('pixels')), 1)
        self.assertGreater(len(taus_for('working')), 1)


class Mixed(unittest.TestCase):
    def test_the_most_similar_version_answers(self):
        from experiments.natural_memory_mixed import pick_answer
        sim = np.array([[.9, .1], [.2, .8]])
        units = np.array([[3, 4], [5, 6]])
        v, i, s = pick_answer(sim, units)
        self.assertEqual(v.tolist(), [0, 1])
        self.assertEqual(i.tolist(), [3, 6])
        self.assertEqual(s.tolist(), [.9, .8])

    def test_both_orders_teach_both_kinds_once_and_run_without_gaps(self):
        from experiments.natural_memory_mixed import timeline
        for order in ('GN', 'NG'):
            tl = timeline(order)
            self.assertEqual(sorted(p[3] for p in tl if p[3]), ['glyph', 'natural'])
            self.assertEqual(tl[0][0], 0)
            for a, b in zip(tl, tl[1:]):
                self.assertEqual(a[1], b[0])
            for lo, hi, stream, taught in tl:
                if taught:
                    self.assertEqual(stream, taught)


class Margin(unittest.TestCase):
    def test_a_blind_code_has_no_margin_and_a_discriminating_one_does(self):
        from experiments.natural_memory import margin_of
        blind = np.array([[.97, .97, .96, .97]], np.float32)
        sharp = np.array([[.95, .10, .20, .05]], np.float32)
        self.assertLess(margin_of(blind, 3)[2][0], .02)
        self.assertGreater(margin_of(sharp, 3)[2][0], .7)

    def test_the_margin_is_best_minus_the_mean_of_the_next_k(self):
        from experiments.natural_memory import margin_of
        i, best, m = margin_of(np.array([[.1, .9, .5, .3]], np.float32), 2)
        self.assertEqual(int(i[0]), 1)
        self.assertAlmostEqual(float(best[0]), .9, places=5)
        self.assertAlmostEqual(float(m[0]), .9 - .4, places=5)

    def test_one_unit_gives_no_margin(self):
        from experiments.natural_memory import margin_of
        self.assertEqual(float(margin_of(np.array([[.5]], np.float32), 10)[2][0]), 0.)


class Sequence(unittest.TestCase):
    def test_collapse_tells_one_point_from_spread_codes(self):
        from experiments.natural_memory_sequence import collapse
        one = np.tile(np.array([[1., 0., 0., 0.]], np.float32), (64, 1))
        self.assertAlmostEqual(collapse(one), 1., places=5)
        spread = unit(np.random.default_rng(9).normal(size=(64, 32)).astype(np.float32))
        self.assertLess(collapse(spread), .1)

    def test_the_reverse_order_reverses_the_seeds_kinds(self):
        from experiments.natural_memory_sequence import KINDS, kind_order
        f, r = kind_order(392, 'forward'), kind_order(392, 'reverse')
        self.assertEqual(f, r[::-1])
        self.assertEqual(sorted(f), sorted(KINDS))

    def test_each_kind_has_its_own_rows(self):
        from experiments.natural_memory_sequence import N, fact_row
        self.assertEqual(fact_row(0, N - 1) + 1, fact_row(1, 0))


if __name__ == '__main__':
    unittest.main()
