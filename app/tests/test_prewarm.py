"""Startup prewarm of the slow /next reads (app/prewarm.py): today's typical
profile, /api/spend-history and the second /api/stats scope."""
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.prewarm import start as START
from app.tests._now_support import WORK, insert_event
from app.tests._support import TempDBTestCase


class _Base(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        import app.prewarm as pw
        import app.spend_history as sh
        import app.today_typical as tt
        self.pw, self.sh, self.tt = pw, sh, tt
        tt.clear_cache()
        sh.clear_cache()
        self.addCleanup(tt.clear_cache)
        self.addCleanup(sh.clear_cache)
        self.addCleanup(lambda: pw.join(5))
        for name, value in (("DASHBOARD_PASSWORD", ""), ("LOCKED_SCOPE", None)):
            p = patch("app.config." + name, value)
            p.start()
            self.addCleanup(p.stop)

    def seed(self):
        from app.config import TZ_NAME
        tz = ZoneInfo(TZ_NAME)
        yesterday = (datetime.now(tz) - timedelta(days=1)).replace(
            hour=1, minute=0, second=0, microsecond=0)
        insert_event(self.conn, "p", session="s1", ts=yesterday.timestamp(), inp=1_000_000)
        insert_event(self.conn, "w", session="s2", ts=yesterday.timestamp(), inp=2_000_000,
                     account=WORK)
        from app.summarizer import summarize_days
        summarize_days(None)


class TypicalWarmTest(_Base):
    def test_warm_builds_the_profile_that_today_reads(self):
        self.seed()
        self.assertTrue(self.tt.warm("personal", time.time(), wait_s=5))
        from app.today import build_today
        with patch.object(self.tt, "build_profile",
                          side_effect=AssertionError("the profile is warm")):
            body = build_today("personal", {"sessions": {}}, time.time())
        self.assertIsNotNone(body["typical_by_now_usd"])

    def test_warm_without_history_is_a_no_op(self):
        self.assertFalse(self.tt.warm("personal", time.time(), wait_s=5))


class SpendHistoryWarmTest(_Base):
    def test_warm_fills_the_route_cache_per_scope(self):
        self.seed()
        self.sh.warm("personal")
        self.sh.warm("enterprise")
        http = self.client()
        self.addCleanup(http.close)
        with patch.object(self.sh, "_spend_history",
                          side_effect=AssertionError("the history is warm")):
            personal = http.get("/api/spend-history?scope=personal")
            work = http.get("/api/spend-history?scope=enterprise")
        self.assertEqual(personal.status_code, 200, personal.text)
        self.assertAlmostEqual(personal.json()["months"][-1]["total"], 5.0, places=2)
        self.assertAlmostEqual(work.json()["months"][-1]["total"], 10.0, places=2)

    def test_stale_history_is_served_while_it_rebuilds(self):
        self.seed()
        self.sh.warm("personal")
        http = self.client()
        self.addCleanup(http.close)
        gate = threading.Event()
        real = self.sh._spend_history

        def slow(*a, **kw):
            gate.wait(5)
            return real(*a, **kw)
        with patch.object(self.sh._HISTORY, "ttl", 0.0), \
                patch.object(self.sh, "_spend_history", side_effect=slow):
            t0 = time.monotonic()
            r = http.get("/api/spend-history?scope=personal")
            self.assertLess(time.monotonic() - t0, 1.0, "a stale value must not wait")
            gate.set()
        self.assertAlmostEqual(r.json()["months"][-1]["total"], 5.0, places=2)


class LedgerWarmTest(_Base):
    def test_warm_caches_the_closed_days_of_the_default_window(self):
        import app.ledger as ledger
        import app.ledger_days as ledger_days
        ledger._cache._entries.clear()
        ledger_days._days._entries.clear()
        self.addCleanup(ledger._cache._entries.clear)
        self.addCleanup(ledger_days._days._entries.clear)
        self.seed()
        ledger.warm("personal")
        ledger._cache._entries.clear()          # the route's 30 s cache is not the point
        scans = []
        real = ledger_days.billable_groups
        with patch.object(ledger_days, "billable_groups",
                          side_effect=lambda *a, **k: scans.append(a[2]) or real(*a, **k)):
            body = self.client().get("/api/ledger?scope=personal").json()
        self.assertEqual(len(body["sessions"]), 1)
        # Only the open day is scanned again: yesterday came from the cache.
        self.assertEqual(len(scans), 1)


class PrewarmTest(_Base):
    def test_start_never_blocks_and_warms_every_scope(self):
        self.seed()
        gate = threading.Event()
        real = self.tt.warm
        seen = []

        def slow(scope, *a, **kw):
            gate.wait(5)
            seen.append(scope)
            return real(scope, *a, **kw)
        with patch.object(self.tt, "warm", side_effect=slow), \
                patch.object(self.pw, "_warm_stats"):
            t0 = time.monotonic()
            thread = START()
            self.assertLess(time.monotonic() - t0, 0.5, "prewarm must not block startup")
            gate.set()
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(seen), ["enterprise", "personal"])
        with patch.object(self.sh, "_spend_history", side_effect=AssertionError("warm")):
            self.sh.cached_history("enterprise")
            self.sh.cached_history("personal")

    def test_a_locked_instance_warms_only_its_scope(self):
        self.seed()
        seen = []
        with patch("app.config.LOCKED_SCOPE", "enterprise"), \
                patch.object(self.tt, "warm", side_effect=lambda s, *a, **k: seen.append(s)), \
                patch.object(self.sh, "warm", side_effect=lambda s: seen.append("sh:" + s)), \
                patch.object(self.pw, "_warm_stats", side_effect=lambda s: seen.append("st:" + s)):
            START().join(10)
        self.assertEqual(seen, ["enterprise", "sh:enterprise"])

    def test_steps_run_in_order(self):
        seen = []
        with patch("app.config.DEFAULT_SCOPE", "personal"), \
                patch.object(self.tt, "warm", side_effect=lambda s, *a, **k: seen.append("typical:" + s)), \
                patch("app.ledger.warm", side_effect=lambda s: seen.append("ledger:" + s)), \
                patch.object(self.sh, "warm", side_effect=lambda s: seen.append("history:" + s)), \
                patch.object(self.pw, "_warm_stats", side_effect=lambda s: seen.append("stats:" + s)):
            START().join(10)
        self.assertEqual(seen, ["typical:personal", "typical:enterprise",
                                "ledger:personal", "ledger:enterprise",
                                "history:personal", "history:enterprise",
                                "stats:enterprise"])

    def test_the_second_stats_scope_is_built_after_the_default(self):
        built = []
        with patch.object(self.tt, "warm"), patch.object(self.sh, "warm"), \
                patch("app.config.DEFAULT_SCOPE", "personal"), \
                patch("app.aggregator.build_dashboard_data",
                      side_effect=lambda s: built.append(s) or {}):
            START().join(10)
        self.assertEqual(built, ["personal", "enterprise"])

    def test_a_failing_step_does_not_stop_the_rest(self):
        self.seed()
        with patch.object(self.tt, "warm", side_effect=RuntimeError("boom")), \
                patch.object(self.pw, "_warm_stats"), \
                self.assertLogs("app.prewarm", "ERROR"):
            START().join(10)
        with patch.object(self.sh, "_spend_history", side_effect=AssertionError("warm")):
            self.sh.cached_history("personal")

    def test_prewarm_never_needs_the_shared_connection(self):
        # Ingest holds the shared connection's write lock for each batch; the
        # prewarm reads on its own read-only connections, so it never waits.
        from app.db import write_txn
        self.seed()
        with patch.object(self.pw, "_warm_stats"):
            with write_txn(self.conn):
                self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('x', '1')")
                thread = START()
                thread.join(10)
                self.assertFalse(thread.is_alive(), "prewarm waited for the writer")
        with patch.object(self.sh, "_spend_history", side_effect=AssertionError("warm")):
            self.sh.cached_history("personal")

    def test_join_stops_between_steps(self):
        gate = threading.Event()
        calls = []

        def slow(scope, *a, **kw):
            calls.append(scope)
            gate.wait(5)
        with patch.object(self.tt, "warm", side_effect=slow), \
                patch.object(self.sh, "warm", side_effect=AssertionError("stopped")), \
                patch.object(self.pw, "_warm_stats", side_effect=AssertionError("stopped")):
            START()
            while not calls:
                time.sleep(0.01)
            threading.Timer(0.05, gate.set).start()
            self.pw.join(5)
        self.assertEqual(calls, ["personal"])


class LifespanTest(_Base):
    def test_lifespan_starts_the_prewarm_and_waits_for_it_before_closing(self):
        import app.limit_rate as lr
        import app.main as main
        order = []
        stubs = [patch.object(main, "load_pricing"),
                 patch.object(main.usage_fetcher, "should_run", return_value=False),
                 patch.object(main, "init_light", new=lambda: asyncio.sleep(0)),
                 patch.object(main, "start_watchdog"), patch.object(main, "stop_watchdog"),
                 patch.object(main, "_backfill_if_needed"),
                 patch("app.aggregator.trigger_eager_rebuild"),
                 patch("app.aggregator.start_sweeps"), patch("app.aggregator.stop_sweeps"),
                 patch.object(lr, "prewarm"), patch.object(lr, "join_refresh"),
                 patch.object(self.pw, "start", side_effect=lambda: order.append("start")),
                 patch.object(self.pw, "join", side_effect=lambda t: order.append("join")),
                 patch.object(main, "close_conn", side_effect=lambda: order.append("close"))]
        for p in stubs:
            p.start()
            self.addCleanup(p.stop)

        async def run():
            async with main.lifespan(main.app):
                pass

        with patch("builtins.print"):
            asyncio.run(run())
        self.assertEqual(order, ["start", "join", "close"])
