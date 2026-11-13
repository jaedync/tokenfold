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
import time
import unittest

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
        self.assertLessEqual(len(self.starts), 1 + int(1.2 / self.INTERVAL) + 1, self.starts)
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


if __name__ == "__main__":
    unittest.main()
