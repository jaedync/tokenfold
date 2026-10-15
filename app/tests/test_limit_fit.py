"""Pure fitting helpers for /api/limit-rate (app/limit_fit.py)."""

import random
import unittest

from app import limit_fit as lf


def _gram(rows, ys):
    return lf.gram_vector(rows, ys)


class NnlsTest(unittest.TestCase):

    def test_recovers_exact_coefficients(self):
        rng = random.Random(1)
        rows = [[rng.uniform(0, 10), rng.uniform(0, 5), rng.uniform(0, 2)]
                for _ in range(50)]
        ys = [0.5 * a + 2.0 * b + 0.1 * c for a, b, c in rows]
        ata, aty = lf.normal_equations(rows, ys)
        x = lf.nnls(ata, aty)
        for got, want in zip(x, (0.5, 2.0, 0.1)):
            self.assertAlmostEqual(got, want, places=8)

    def test_negative_truth_clamps_to_zero(self):
        rng = random.Random(2)
        rows = [[rng.uniform(1, 10), rng.uniform(1, 10)] for _ in range(40)]
        ys = [1.0 * a - 0.5 * b for a, b in rows]
        x = lf.nnls(*lf.normal_equations(rows, ys))
        self.assertEqual(x[1], 0.0)
        self.assertGreater(x[0], 0.0)
        # With the second feature at zero, x0 is the 1-D least squares slope.
        sxy = sum(a * y for (a, _), y in zip(rows, ys))
        sxx = sum(a * a for a, _ in rows)
        self.assertAlmostEqual(x[0], max(0.0, sxy / sxx), places=8)

    def test_all_zero_feature_stays_zero(self):
        rows = [[float(i), 0.0] for i in range(1, 10)]
        ys = [2.0 * i for i in range(1, 10)]
        x = lf.nnls(*lf.normal_equations(rows, ys))
        self.assertAlmostEqual(x[0], 2.0, places=9)
        self.assertEqual(x[1], 0.0)

    def test_collinear_features_do_not_raise(self):
        rows = [[float(i), 2.0 * i] for i in range(1, 10)]
        ys = [3.0 * i for i in range(1, 10)]
        x = lf.nnls(*lf.normal_equations(rows, ys))
        self.assertAlmostEqual(x[0] + 2 * x[1], 3.0, places=6)
        self.assertTrue(all(v >= 0 for v in x))

    def test_empty_problem(self):
        self.assertEqual(lf.nnls([], []), [])


class GramVectorTest(unittest.TestCase):

    def test_r2_and_ratio_from_summed_blocks(self):
        rows = [[1.0], [2.0], [3.0]]
        ys = [2.0, 4.1, 5.9]
        g = lf.add_vectors([_gram(rows[:1], ys[:1]), _gram(rows[1:], ys[1:])])
        whole = _gram(rows, ys)
        for a, b in zip(g, whole):
            self.assertAlmostEqual(a, b)
        stats = lf.unpack_gram(g, 1)
        x = lf.nnls(stats.ata, stats.aty)
        self.assertGreater(lf.r_squared(stats, x), 0.99)
        self.assertAlmostEqual(stats.sum_x[0] / stats.sum_y, 6.0 / 12.0)


class BootstrapTest(unittest.TestCase):

    def test_seeded_and_deterministic(self):
        blocks = [[float(i), float(i % 3 + 1)] for i in range(1, 15)]
        stat = lambda v: v[0] / v[1]  # noqa: E731
        a = lf.block_bootstrap(blocks, stat, draws=50, seed=7)
        b = lf.block_bootstrap(blocks, stat, draws=50, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 50)

    def test_none_results_are_dropped(self):
        blocks = [[1.0], [0.0]]
        out = lf.block_bootstrap(blocks, lambda v: v[0] or None, draws=40, seed=1)
        self.assertTrue(all(v is not None for v in out))
        self.assertLess(len(out), 40)

    def test_percentile_linear(self):
        self.assertEqual(lf.percentile([], 50), None)
        self.assertAlmostEqual(lf.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(lf.percentile([1, 2, 3, 4, 5], 5), 1.2)
        self.assertAlmostEqual(lf.percentile([5, 1, 4, 2, 3], 95), 4.8)

    def test_median(self):
        self.assertEqual(lf.median([3, 1, 2]), 2)
        self.assertEqual(lf.median([4, 1, 2, 3]), 2.5)
        self.assertIsNone(lf.median([]))


class IntervalsTest(unittest.TestCase):

    def test_greedy_pairs_at_least_min_apart(self):
        R = 10_000.0
        readings = [(0.0, 1.0, R), (600.0, 2.0, R), (1200.0, 3.0, R),
                    (1800.0, 3.0, R), (2400.0, 5.0, R), (3000.0, 6.0, R),
                    (3600.0, 6.0, R)]
        iv = lf.build_intervals(readings, min_s=1800, max_s=72 * 3600)
        self.assertEqual([(i.ta, i.tb, i.points) for i in iv],
                         [(0.0, 1800.0, 2.0), (1800.0, 3600.0, 3.0)])

    def test_window_change_and_drop_break_interval(self):
        readings = [(0.0, 10.0, 5000.0), (1000.0, 12.0, 5000.0),
                    # resets_at jumps: a new window, no interval across it
                    (2000.0, 1.0, 9000.0), (4000.0, 4.0, 9000.0),
                    # a drop inside one window (granted reset) breaks too
                    (5000.0, 0.0, 9000.0), (7000.0, 2.0, 9000.0)]
        iv = lf.build_intervals(readings, min_s=1500, max_s=72 * 3600)
        self.assertEqual([(i.ta, i.tb) for i in iv],
                         [(2000.0, 4000.0), (5000.0, 7000.0)])

    def test_unknown_reset_breaks_and_long_gaps_drop(self):
        readings = [(0.0, 1.0, None), (4000.0, 2.0, 9e9),
                    (4000.0 + 80 * 3600, 9.0, 9e9)]
        self.assertEqual(lf.build_intervals(readings, 1800, 72 * 3600), [])

    def test_small_jitter_drop_is_tolerated(self):
        readings = [(0.0, 5.0, 9e9), (900.0, 4.6, 9e9), (1900.0, 7.0, 9e9)]
        iv = lf.build_intervals(readings, 1800, 72 * 3600)
        self.assertEqual([(i.ta, i.tb, i.points) for i in iv], [(0.0, 1900.0, 2.0)])


class AssignTest(unittest.TestCase):

    def test_sums_rows_into_half_open_intervals(self):
        iv = [lf.Interval(0.0, 10.0, 1.0), lf.Interval(10.0, 20.0, 1.0),
              lf.Interval(30.0, 40.0, 1.0)]
        # (t, vector): t on a right edge belongs to that interval.
        rows = [(0.0, [9.0]), (5.0, [1.0]), (10.0, [2.0]), (15.0, [3.0]),
                (25.0, [100.0]), (40.0, [4.0]), (41.0, [7.0])]
        sums = lf.assign_to_intervals(iv, rows, width=1)
        self.assertEqual(sums, [[3.0], [3.0], [4.0]])


if __name__ == "__main__":
    unittest.main()
