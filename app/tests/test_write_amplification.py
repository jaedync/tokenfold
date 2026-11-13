"""Dashboard reads must not churn the disk, and ingest must not rebuild nonstop.

Production, 2026-11-13: the uvicorn process wrote about 14 MB/s, and 99.7% of
it went to SQLite temp files (`/var/tmp/etilqs_*`, opened and unlinked at
once). The dashboard build's GROUP BY and ORDER BY statements spilled their
sorters and temp B-trees out of the 2 MB page cache into those files, about
90 MB per build. Every Pi ingest (about one each 5 s) triggered a full build,
so the cache-rebuild thread ran without a pause at one full core.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import patch

import app.aggregator as agg
import app.db as db
from app.tests._support import TempDBTestCase

_TEMP_STORE_MEMORY = 2


def _open_temp_files() -> list[str]:
    """SQLite temp files open in this process (they are unlinked at open,
    so they show only as /proc/self/fd links)."""
    out = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if "etilqs_" in target:
            out.append(target)
    return out


class TempStoreTest(TempDBTestCase):
    def _fill(self, rows=40000):
        with db.write_txn() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS big(k TEXT, v INTEGER)")
            conn.executemany(
                "INSERT INTO big(k, v) VALUES(?, ?)",
                ((f"{(i * 7919) % rows:08d}-" + "x" * 120, i) for i in range(rows)))

    def test_read_conn_keeps_temp_store_in_memory(self):
        with db.read_conn() as conn:
            self.assertEqual(conn.execute("PRAGMA temp_store").fetchone()[0],
                             _TEMP_STORE_MEMORY)

    def test_shared_conn_keeps_temp_store_in_memory(self):
        self.assertEqual(db.get_conn().execute("PRAGMA temp_store").fetchone()[0],
                         _TEMP_STORE_MEMORY)

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc/self/fd")
    def test_large_sort_on_read_conn_opens_no_temp_file(self):
        """A sort bigger than the page cache stays in RAM: no temp file."""
        self._fill()
        with db.read_conn() as conn:
            conn.execute("PRAGMA cache_size=-256")  # 256 KiB, far below the sort
            cur = conn.execute(
                "SELECT k, COUNT(*), SUM(v) FROM big GROUP BY k ORDER BY SUM(v) DESC")
            self.assertIsNotNone(cur.fetchone())  # the sort is done at the first step
            self.assertEqual(_open_temp_files(), [])
            cur.fetchall()

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc/self/fd")
    def test_probe_sees_a_temp_file_when_temp_store_is_file(self):
        """Proves the probe above: the same sort with temp_store=FILE spills."""
        self._fill()
        with db.read_conn() as conn:
            conn.execute("PRAGMA temp_store=FILE")
            conn.execute("PRAGMA cache_size=-256")
            cur = conn.execute(
                "SELECT k, COUNT(*), SUM(v) FROM big GROUP BY k ORDER BY SUM(v) DESC")
            self.assertIsNotNone(cur.fetchone())
            self.assertNotEqual(_open_temp_files(), [])
            cur.fetchall()


def _wait_not_rebuilding(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with agg._cache_lock:
            if not agg._rebuilding:
                return True
        time.sleep(0.01)
    return False


class RebuildIntervalTest(TempDBTestCase):
    INTERVAL = 0.4

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        with agg._cache_lock:
            agg._warm_scopes.clear()
        orig_inner, orig_interval = agg._build_dashboard_data_inner, agg.REBUILD_MIN_INTERVAL_S
        orig_last = agg._last_build_started
        agg.REBUILD_MIN_INTERVAL_S = self.INTERVAL
        agg._last_build_started = None
        self.starts: list[float] = []
        self.gens: list[int] = []

        def fake_inner(scope):
            self.starts.append(time.monotonic())
            with agg._cache_lock:
                self.gens.append(agg._cache_gen)
            return {"scope": scope}

        agg._build_dashboard_data_inner = fake_inner

        def restore():
            agg.join_rebuild(5.0)
            agg._build_dashboard_data_inner = orig_inner
            agg.REBUILD_MIN_INTERVAL_S = orig_interval
            agg._last_build_started = orig_last
            with agg._cache_lock:
                agg._rebuilding = False
                agg._warm_scopes.clear()
            agg._cached_data.clear()
        self.addCleanup(restore)

    def test_first_invalidation_after_idle_builds_at_once(self):
        t0 = time.monotonic()
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding())
        self.assertEqual(len(self.starts), 1)
        self.assertLess(self.starts[0] - t0, self.INTERVAL / 2)

    def test_steady_ingest_builds_at_most_once_per_interval(self):
        """Invalidations every 20 ms for 1.2 s: builds are spaced by the interval,
        and the last invalidation is still built (no lost generation)."""
        end = time.monotonic() + 1.2
        while time.monotonic() < end:
            agg.trigger_eager_rebuild()
            time.sleep(0.02)
        with agg._cache_lock:
            final_gen = agg._cache_gen
        self.assertTrue(_wait_not_rebuilding(timeout=5.0))
        gaps = [b - a for a, b in zip(self.starts, self.starts[1:])]
        # 1.2 s / 0.4 s: a build at t=0, 0.4, 0.8 and one at 1.2 for the
        # last invalidation. This tight bound also pins the current_gen
        # refresh after the wait: without it each waited build is discarded
        # and repeated, and the count reaches 5 or more.
        self.assertLessEqual(len(self.starts), 4, self.starts)
        self.assertTrue(all(g >= self.INTERVAL * 0.95 for g in gaps), gaps)
        self.assertEqual(self.gens[-1], final_gen)
        self.assertEqual(agg._cached_data.get(agg.DEFAULT_SCOPE), {"scope": agg.DEFAULT_SCOPE})

    def test_readers_get_previous_data_while_the_worker_waits(self):
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding())
        agg.trigger_eager_rebuild()  # inside the interval: the worker waits
        served = agg.build_dashboard_data(agg.DEFAULT_SCOPE)
        self.assertEqual(served, {"scope": agg.DEFAULT_SCOPE})
        self.assertEqual(len(self.starts), 1, "the reader must not build in its own thread")
        self.assertTrue(_wait_not_rebuilding())
        self.assertEqual(len(self.starts), 2)

    def test_immediate_trigger_skips_the_wait(self):
        agg.REBUILD_MIN_INTERVAL_S = 30.0
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding())
        agg.trigger_eager_rebuild()  # an ingest: waits 30 s
        time.sleep(0.05)
        t0 = time.monotonic()
        agg.trigger_eager_rebuild(immediate=True)  # the user's own write
        self.assertTrue(_wait_not_rebuilding(timeout=2.0))
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(len(self.starts), 2)

    def test_budget_post_rebuilds_without_the_wait(self):
        import asyncio
        import app.monthly_budget as mb
        from app.models import EnterpriseBudgetRequest
        with patch.object(agg, "trigger_eager_rebuild") as trig, \
                patch.object(mb, "_require_writable", lambda: None), \
                patch.object(mb, "set_budget", lambda conn, value: None), \
                patch.object(mb, "get_budget", lambda conn: 5.0):
            asyncio.run(mb.post_enterprise_budget(EnterpriseBudgetRequest(budget_usd=5.0)))
        trig.assert_called_once_with(immediate=True)

    def test_shutdown_ends_the_wait(self):
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding())
        agg.REBUILD_MIN_INTERVAL_S = 30.0
        agg.trigger_eager_rebuild()
        time.sleep(0.05)
        db.begin_shutdown()
        try:
            t0 = time.monotonic()
            agg.join_rebuild(5.0)
            self.assertLess(time.monotonic() - t0, 2.0)
            self.assertEqual(len(self.starts), 1, "no build after the shutdown started")
        finally:
            db._shutdown.clear()


class IntervalEnvTest(unittest.TestCase):
    def _parse(self, value):
        env = {} if value is None else {"TOKENFOLD_REBUILD_MIN_INTERVAL_S": value}
        with patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("TOKENFOLD_REBUILD_MIN_INTERVAL_S", None)
            return agg._rebuild_interval_from_env()

    def test_default_and_valid_values(self):
        self.assertEqual(self._parse(None), 30.0)
        self.assertEqual(self._parse(""), 30.0)
        self.assertEqual(self._parse("0"), 0.0)
        self.assertEqual(self._parse("12.5"), 12.5)

    def test_bad_values_fall_back_to_the_default_with_a_warning(self):
        for raw in ("abc", "-1", "nan", "inf"):
            with self.subTest(raw=raw), self.assertLogs(agg.logger, "WARNING"):
                self.assertEqual(self._parse(raw), 30.0)


class TeardownJoinTest(unittest.TestCase):
    """Review repro, 2026-11-13: a worker with a pending generation must not
    outlive the test that started it (it waited the 30 s default, because
    the interval patch reverted before the teardown joined the worker)."""

    def test_teardown_joins_a_worker_with_a_pending_generation(self):
        orig_inner, orig_last = agg._build_dashboard_data_inner, agg._last_build_started
        started = threading.Event()

        def slow(scope):
            started.set()
            time.sleep(0.3)
            return {"scope": scope}

        class Inner(TempDBTestCase):
            def test_pending(self):
                agg._build_dashboard_data_inner = slow
                agg.trigger_eager_rebuild()
                self.assertTrue(started.wait(2.0))
                agg.trigger_eager_rebuild()  # pending while the build runs

        try:
            t0 = time.monotonic()
            result = unittest.TextTestRunner(stream=open(os.devnull, "w")).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(Inner))
            elapsed = time.monotonic() - t0
            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
            with agg._cache_lock:
                thread, rebuilding = agg._rebuild_thread, agg._rebuilding
            self.assertFalse(thread is not None and thread.is_alive(), "worker leaked")
            self.assertFalse(rebuilding)
            self.assertLess(elapsed, 4.0)
        finally:
            agg.join_rebuild(40.0)
            agg._build_dashboard_data_inner = orig_inner
            agg._last_build_started = orig_last


if __name__ == "__main__":
    unittest.main()
