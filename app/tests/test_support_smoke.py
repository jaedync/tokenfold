import unittest

from app.tests._support import TempDBTestCase


class SupportSmokeTest(TempDBTestCase):
    def test_temp_db_has_schema(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
        self.assertIn("request_id", cols)
        tbls = {r[0] for r in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("daily_summary", tbls)

    def test_compute_cost_baseline(self):
        from app.pricing import compute_cost
        self.assertAlmostEqual(
            compute_cost("Sonnet 4.6", 1_000_000, 0, 0, 0), 3.0, places=4)
