import time

from app.tests._support import TempDBTestCase
from app.tests.test_summarizer_pricing import insert_assistant


def _tag_enterprise(conn):
    """Bring seeded events in-scope for the enterprise-only window cost filter.

    insert_assistant() leaves plan/org/account NULL; compute_window_cost is now
    fail-closed to verified-enterprise usage (plan, org_name, AND account_email
    must all be set), so seed data must be fully tagged.
    """
    conn.execute(
        "UPDATE events SET plan='enterprise', org_name='Acme', "
        "account_email='test@acme.io'"
    )
    conn.commit()


class WindowCostFastGeoTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def test_window_honors_fast(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000, speed="fast")  # $10
        _tag_enterprise(self.conn)
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 10)
        self.assertAlmostEqual(got, 10.0, places=2)

    def test_window_mixed_same_model(self):
        from app.cost_windows import compute_window_cost
        insert_assistant(self.conn, "u1", "r1", inp=1_000_000)                            # $5 normal
        insert_assistant(self.conn, "u2", "r2", inp=1_000_000, speed="fast", ts=1781000001.0)  # $10 fast
        _tag_enterprise(self.conn)
        got = compute_window_cost(self.conn, 1781000000.0 - 10, 1781000000.0 + 100)
        self.assertAlmostEqual(got, 15.0, places=2)  # must NOT collapse to 2M tokens @ one rate


class StreamedChunksMixedSpeedDedupTest(TempDBTestCase):
    """ONE request whose streamed chunks mix speed=NULL and speed='fast' must
    price as fast ($10) in ALL THREE cost paths.

    Streaming chunks repeat token counts; dedup takes MAX(tokens) per request_id.
    A bare (non-aggregated) speed column in that inner GROUP BY takes an
    ARBITRARY row's value in SQLite — so the same request could price at base in
    one path and fast in another. MAX(speed) (NULL-skipping, deterministic) is
    the required semantics everywhere.
    """

    def setUp(self):
        super().setUp()
        self.freeze_pricing()

    def _seed_streamed_fast_request(self):
        """One enterprise Opus 4.8 request, two chunk rows, MAX input = 1M.

        The NULL-speed chunk is inserted FIRST (lowest rowid): observed SQLite
        behavior picks the FIRST row in scan order for a bare column alongside
        multiple MAX() aggregates, so a bare `speed` column resolves to NULL
        here and prices at base — exposing the bug. MAX(speed) skips NULLs and
        deterministically yields 'fast' regardless of chunk order.
        """
        now = time.time()
        # chunk 1: speed=NULL (field absent in this chunk), partial token count
        insert_assistant(self.conn, "u1", "rX", inp=400_000, speed=None,
                         ts=now - 120)
        # chunk 2: speed='fast', full MAX token count
        insert_assistant(self.conn, "u2", "rX", inp=1_000_000, speed="fast",
                         ts=now - 60)
        _tag_enterprise(self.conn)
        return now

    def test_window_cost_prices_mixed_chunks_as_fast(self):
        from app.cost_windows import compute_window_cost
        now = self._seed_streamed_fast_request()
        got = compute_window_cost(self.conn, now - 3600, now + 10)
        self.assertAlmostEqual(
            got, 10.0, places=2,
            msg=f"mixed NULL/fast chunks must price as fast ($10), got {got}")

    def test_rate_limits_week_cost_prices_mixed_chunks_as_fast(self):
        self._seed_streamed_fast_request()
        import app.aggregator as agg
        agg._cached_data = None
        c = self.client()
        rl = c.get("/api/rate-limits").json()["weekly_budget"]
        self.assertAlmostEqual(
            rl["week_cost"], 10.0, places=2,
            msg=f"week_cost must be $10 (fast), got {rl['week_cost']}")

    def test_aggregator_hourly_prices_mixed_chunks_as_fast(self):
        from app.aggregator import _build_hourly
        self._seed_streamed_fast_request()
        hourly = _build_hourly(self.conn)
        total = sum(h["cost"] for h in hourly)
        self.assertAlmostEqual(
            total, 10.0, places=2,
            msg=f"hourly cost sum must be $10 (fast), got {total}")
