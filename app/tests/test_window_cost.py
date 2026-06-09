from app.tests._support import TempDBTestCase
from app.tests.test_summarizer_pricing import insert_assistant


class WindowCostFastGeoTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_window_honors_fast(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000, speed="fast")  # $10
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 10)
        self.assertAlmostEqual(got, 10.0, places=2)

    def test_window_mixed_same_model(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000)                            # $5 normal
        insert_assistant(self.conn, "u2", "r2", inp=1_000_000, speed="fast", ts=1781000001.0)  # $10 fast
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 100)
        self.assertAlmostEqual(got, 15.0, places=2)  # must NOT collapse to 2M tokens @ one rate
