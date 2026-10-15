"""Background threads and the shared connection (app/db.py shutdown rules).

Closing a sqlite3 connection while another thread runs a statement on it
crashes the process (segfault). These tests pin the rules that make that
impossible: background work reads on its own connections, writes through
db.background_write (which never opens the database and is skipped once
shutdown starts), and the app joins every background thread before it
closes the shared connection.
"""
import asyncio
import json
import threading
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase


def _meta(key):
    from app.db import read_conn
    with read_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row[0]


class GetConnAfterShutdownTest(TempDBTestCase):
    def test_get_conn_does_not_reopen_after_shutdown_starts(self):
        db = self._db
        db.begin_shutdown()
        db.close_conn()
        with patch.object(db, "_migrate") as migrate:
            with self.assertRaises(db.DatabaseClosed):
                db.get_conn()
        migrate.assert_not_called()
        self.assertIsNone(db._conn)

    def test_open_conn_clears_the_shutdown(self):
        db = self._db
        db.begin_shutdown()
        db.close_conn()
        self.assertIsNotNone(db.open_conn())
        self.assertFalse(db.shutting_down())


class BackgroundWriteTest(TempDBTestCase):
    def test_writes_while_open(self):
        with self._db.background_write() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('bg', '1')")
        self.assertEqual(_meta("bg"), "1")

    def test_skips_after_shutdown_starts(self):
        self._db.begin_shutdown()
        with self._db.background_write() as conn:
            self.assertIsNone(conn)
        self.assertIsNone(_meta("bg"))

    def test_skips_a_closed_connection_without_reopening(self):
        self._db.close_conn()
        with self._db.background_write() as conn:
            self.assertIsNone(conn)
        self.assertIsNone(self._db._conn)

    def test_skips_a_replaced_connection(self):
        generation = self._db.generation()
        self._db.close_conn()
        self._db.get_conn()
        with self._db.background_write(generation) as conn:
            self.assertIsNone(conn)

    def test_close_waits_for_a_running_background_write(self):
        order, inside = [], threading.Event()

        def writer():
            with self._db.background_write() as conn:
                inside.set()
                time.sleep(0.2)
                conn.execute("INSERT INTO meta(key, value) VALUES('bg', '1')")
                order.append("write")

        t = threading.Thread(target=writer)
        t.start()
        inside.wait(5)
        self._db.close_conn()
        order.append("close")
        t.join(5)
        self.assertEqual(order, ["write", "close"])
        self.assertEqual(_meta("bg"), "1")


class LimitRateStoreTest(TempDBTestCase):
    def test_store_after_shutdown_neither_writes_nor_reopens(self):
        import app.limit_rate as lr
        self._db.begin_shutdown()
        self._db.close_conn()
        with patch.object(self._db, "_migrate") as migrate, \
                patch.object(lr.logger, "exception") as logged:
            lr._store({"version": lr.CACHE_VERSION, "computed_at": 1.0, "payload": {}})
        migrate.assert_not_called()
        logged.assert_not_called()
        self.assertIsNone(self._db._conn)
        self.assertIsNone(_meta(lr.CACHE_KEY))


class PricingConnectionTest(TempDBTestCase):
    def test_load_pricing_reads_its_cache_without_the_shared_connection(self):
        import app.pricing as pricing
        cache = {"ts": time.time(), "pricing": {"Test Model": [1, 2, 3, 4]}}
        with self._db.write_txn() as conn:
            conn.execute("INSERT INTO meta(key, value) VALUES('pricing_cache', ?)",
                         (json.dumps(cache),))
        saved = pricing._dynamic_pricing
        self.addCleanup(setattr, pricing, "_dynamic_pricing", saved)
        self._db.close_conn()
        pricing.load_pricing()
        self.assertEqual(pricing._dynamic_pricing, cache["pricing"])
        self.assertIsNone(self._db._conn, "load_pricing opened the shared connection")

    def test_load_pricing_after_shutdown_skips_the_store(self):
        import app.pricing as pricing
        saved = pricing._dynamic_pricing
        self.addCleanup(setattr, pricing, "_dynamic_pricing", saved)
        body = json.dumps({"claude-test-1": {"input_cost_per_token": 1e-6,
                                             "output_cost_per_token": 2e-6}}).encode()

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return body

        self._db.begin_shutdown()
        with patch.dict("os.environ", {"TOKENFOLD_PRICING_FETCH": "1"}), \
                patch("urllib.request.urlopen", return_value=Resp()), \
                patch("builtins.print") as printed:
            pricing.load_pricing()
        printed.assert_not_called()
        self.assertEqual(len(pricing._dynamic_pricing), 1)
        self.assertIsNone(_meta("pricing_cache"))


class AggregatorJoinTest(TempDBTestCase):
    def test_join_rebuild_waits_for_the_worker(self):
        import app.aggregator as agg
        started = threading.Event()

        def slow(scope):
            started.set()
            time.sleep(0.3)
            return {}
        with patch.object(agg, "_build_dashboard_data_inner", side_effect=slow):
            agg.trigger_eager_rebuild()
            started.wait(5)
            agg.join_rebuild(5)
            with agg._cache_lock:
                self.assertFalse(agg._rebuilding)

    def test_stop_sweeps_stops_rescheduling(self):
        import app.aggregator as agg
        agg.stop_sweeps(1)
        self.addCleanup(agg.stop_sweeps, 1)
        with patch.object(agg, "_last_full_sweep", time.time()), \
                patch.object(agg, "checkpoint_wal"):
            agg._run_periodic_sweep()
        self.assertEqual(agg._sweep_timers, [])


class BackgroundCacheJoinTest(TempDBTestCase):
    def test_join_all_waits_for_inflight_builds(self):
        from app import bg_cache
        cache = bg_cache.BackgroundCache(ttl=60, wait_s=0.0)
        done = []

        def build():
            time.sleep(0.2)
            done.append(1)
            return 1
        cache.get("k", build)
        bg_cache.join_all(5)
        self.assertEqual(done, [1])


class LifespanShutdownTest(TempDBTestCase):
    def _stub_startup(self, order):
        import app.aggregator as agg
        import app.bg_cache as bg
        import app.limit_rate as lr
        import app.main as main
        import app.prewarm as pw
        stubs = [patch.object(main, "load_pricing"),
                 patch.object(main.usage_fetcher, "should_run", return_value=False),
                 patch.object(main, "init_light", new=lambda: asyncio.sleep(0)),
                 patch.object(main, "start_watchdog"), patch.object(main, "stop_watchdog"),
                 patch.object(main, "_backfill_if_needed"),
                 patch.object(agg, "trigger_eager_rebuild"),
                 patch.object(agg, "start_sweeps"),
                 patch.object(lr, "prewarm", side_effect=lambda: order.append("lr start")),
                 patch.object(pw, "start", side_effect=lambda: order.append("pw start")),
                 patch.object(main, "begin_shutdown", side_effect=lambda: order.append("signal")),
                 patch.object(agg, "stop_sweeps", side_effect=lambda t: order.append("sweeps")),
                 patch.object(agg, "join_rebuild", side_effect=lambda t: order.append("rebuild")),
                 patch.object(lr, "join_refresh", side_effect=lambda t: order.append("fit")),
                 patch.object(pw, "join", side_effect=lambda t: order.append("prewarm")),
                 patch.object(bg, "join_all", side_effect=lambda t: order.append("bg")),
                 patch.object(main, "close_conn", side_effect=lambda: order.append("close"))]
        for p in stubs:
            p.start()
            self.addCleanup(p.stop)
        return main

    def _run(self, main):
        async def run():
            async with main.lifespan(main.app):
                pass
        with patch("builtins.print"):
            asyncio.run(run())

    def test_shutdown_signals_and_joins_every_thread_before_close(self):
        order = []
        main = self._stub_startup(order)
        self._run(main)
        self.assertEqual(order[0], "signal")
        self.assertEqual(order[-1], "close")
        self.assertEqual(sorted(order[1:-1]), ["bg", "fit", "prewarm", "rebuild", "sweeps"])

    def test_prewarm_is_off_under_tests(self):
        import app.config as cfg
        self.assertFalse(cfg.PREWARM)
        order = []
        self._run(self._stub_startup(order))
        self.assertNotIn("lr start", order)
        self.assertNotIn("pw start", order)

    def test_prewarm_flag_starts_both_prewarms(self):
        order = []
        main = self._stub_startup(order)
        with patch("app.config.PREWARM", True):
            self._run(main)
        self.assertIn("lr start", order)
        self.assertIn("pw start", order)
