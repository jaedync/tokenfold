from app.tests._support import TempDBTestCase
from app.tests.test_summarizer_pricing import insert_assistant


class ReconcileTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_dashboard_total_reconciles_with_fast_turns(self):
        # normal Opus 4.8 1M-input ($5) + fast 1M-input ($10) = $15
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000)
        insert_assistant(self.conn, "u2", "r2", inp=1_000_000, speed="fast", ts=1781000100.0)
        from app.summarizer import summarize_days
        import app.aggregator as _agg
        summarize_days(None)
        # Directly clear the module-level cache so build_dashboard_data() does a
        # synchronous rebuild against the freshly-populated temp DB, rather than
        # returning stale data from a previous test run in the same process.
        _agg._cached_data = None
        d = _agg.build_dashboard_data()
        mb_total = sum(m["cost"] for m in d["model_breakdown"])
        daily_total = sum(x["cost"] for x in d["daily"])
        self.assertAlmostEqual(mb_total, 15.0, places=2)
        self.assertAlmostEqual(daily_total, 15.0, places=2)
