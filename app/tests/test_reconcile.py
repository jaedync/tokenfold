from app.tests._support import TempDBTestCase
from app.tests.test_enterprise_only import ins as ins_enterprise


class ReconcileTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_dashboard_total_reconciles_with_fast_turns(self):
        # normal Opus 4.8 1M-input ($5) + fast 1M-input ($10) = $15
        # Both events tagged as enterprise so they pass the compliance filter.
        ins_enterprise(self.conn, "u1", "r1", "ent@acme.io", "enterprise", "Acme",
                       "hpc1", "proj", "s1", inp=1_000_000)
        ins_enterprise(self.conn, "u2", "r2", "ent@acme.io", "enterprise", "Acme",
                       "hpc1", "proj", "s1", inp=1_000_000, ts=1781000100.0)
        # Override speed to "fast" for u2 so pricing test still exercises fast-turn path
        self.conn.execute("UPDATE events SET speed='fast' WHERE uuid='u2'")
        self.conn.commit()
        from app.summarizer import summarize_days
        import app.aggregator as _agg
        summarize_days(None)
        # Directly clear the module-level cache so build_dashboard_data() does a
        # synchronous rebuild against the freshly-populated temp DB, rather than
        # returning stale data from a previous test run in the same process.
        _agg._cached_data.clear()
        d = _agg.build_dashboard_data()
        mb_total = sum(m["cost"] for m in d["model_breakdown"])
        daily_total = sum(x["cost"] for x in d["daily"])
        self.assertAlmostEqual(mb_total, 15.0, places=2)
        self.assertAlmostEqual(daily_total, 15.0, places=2)
