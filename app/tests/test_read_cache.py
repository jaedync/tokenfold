"""ReadCache (app/read_cache.py): single flight per key, no global build lock."""
import threading
import time
import unittest

from app import read_cache
from app.read_cache import ReadCache


class ReadCacheLockTest(unittest.TestCase):
    def test_a_slow_build_does_not_block_another_key(self):
        cache = ReadCache(ttl=30)
        release, started = threading.Event(), threading.Event()

        def slow():
            started.set()
            release.wait(5)
            return "slow"
        t = threading.Thread(target=cache.get, args=("a", slow))
        t.start()
        started.wait(5)
        t0 = time.monotonic()
        self.assertEqual(cache.get("b", lambda: "fast"), "fast")
        self.assertLess(time.monotonic() - t0, 1.0)
        release.set()
        t.join(5)

    def test_one_build_per_key_at_a_time(self):
        cache = ReadCache(ttl=30)
        calls, gate = [], threading.Event()

        def build():
            calls.append(1)
            gate.wait(5)
            return len(calls)
        threads = [threading.Thread(target=cache.get, args=("k", build)) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.1)
        gate.set()
        for t in threads:
            t.join(5)
        self.assertEqual(calls, [1])
        self.assertEqual(cache.get("k", lambda: 99), 1)

    def test_a_failed_build_is_not_cached(self):
        cache = ReadCache(ttl=30)

        def boom():
            raise RuntimeError("x")
        with self.assertRaises(RuntimeError):
            cache.get("k", boom)
        self.assertEqual(cache.get("k", lambda: 2), 2)

    def test_key_locks_do_not_grow_without_bound(self):
        cache = ReadCache(ttl=30, max_entries=4)
        for i in range(50):
            cache.get(i, lambda i=i: i)
        self.assertLessEqual(len(cache._key_locks), 4)

class ReadCacheStaleTest(unittest.TestCase):
    """stale_s: a value past its ttl is served at once while one refresh runs."""

    def setUp(self):
        # An earlier test's app shutdown leaves the flag set; refreshes need it clear.
        from app import db
        was = db.shutting_down()
        db._shutdown.clear()
        self.addCleanup(lambda: db._shutdown.set() if was else None)

    def _aged(self, cache, key, seconds):
        with cache._lock:
            ts, value = cache._entries[key]
            cache._entries[key] = (ts - seconds, value)

    def test_an_expired_value_is_served_while_one_refresh_runs(self):
        cache = ReadCache(ttl=30, stale_s=600)
        self.assertEqual(cache.get("k", lambda: "old"), "old")
        self._aged(cache, "k", 31)
        gate, calls = threading.Event(), []

        def slow():
            calls.append(1)
            gate.wait(5)
            return "new"
        t0 = time.monotonic()
        self.assertEqual(cache.get("k", slow), "old")
        self.assertEqual(cache.get("k", slow), "old")
        self.assertLess(time.monotonic() - t0, 1.0, "a poll waited for the rebuild")
        gate.set()
        read_cache.join_all(5)
        self.assertEqual(calls, [1])
        self.assertEqual(cache.get("k", lambda: "never"), "new")

    def test_a_value_older_than_ttl_plus_stale_s_waits_for_the_build(self):
        cache = ReadCache(ttl=30, stale_s=60)
        cache.get("k", lambda: "old")
        self._aged(cache, "k", 91)
        self.assertEqual(cache.get("k", lambda: "new"), "new")

    def test_without_stale_s_an_expired_value_is_rebuilt_inline(self):
        cache = ReadCache(ttl=30)
        cache.get("k", lambda: "old")
        self._aged(cache, "k", 31)
        self.assertEqual(cache.get("k", lambda: "new"), "new")

    def test_a_failed_refresh_keeps_the_stale_value(self):
        cache = ReadCache(ttl=30, stale_s=600)
        cache.get("k", lambda: "old")
        self._aged(cache, "k", 31)

        def boom():
            raise RuntimeError("x")
        with self.assertLogs("app.read_cache", level="ERROR"):
            self.assertEqual(cache.get("k", boom), "old")
            read_cache.join_all(5)
        self.assertEqual(cache.get("k", lambda: "new"), "old")
        read_cache.join_all(5)
        self.assertEqual(cache.get("k", lambda: "never"), "new")

    def test_no_refresh_starts_after_the_shutdown_began(self):
        from app import db
        cache = ReadCache(ttl=30, stale_s=600)
        cache.get("k", lambda: "old")
        self._aged(cache, "k", 31)
        calls = []
        db.begin_shutdown()
        try:
            self.assertEqual(cache.get("k", lambda: calls.append(1) or "new"), "old")
            read_cache.join_all(5)
        finally:
            db._shutdown.clear()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
