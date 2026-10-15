"""bg_cache.BackgroundCache: stale-while-revalidate for slow live-view inputs."""
import threading
import time
import unittest

from app.bg_cache import BackgroundCache


class BackgroundCacheTests(unittest.TestCase):
    def test_fast_build_is_returned_on_a_cold_miss(self):
        cache = BackgroundCache(ttl=60, wait_s=2.0)
        self.assertEqual(cache.get("k", lambda: 42), 42)

    def test_fresh_value_is_reused(self):
        cache = BackgroundCache(ttl=60, wait_s=2.0)
        calls = []
        cache.get("k", lambda: calls.append(1) or 1)
        cache.get("k", lambda: calls.append(1) or 2)
        self.assertEqual(len(calls), 1)

    def test_slow_cold_build_returns_none_then_value(self):
        cache = BackgroundCache(ttl=60, wait_s=0.01)
        gate = threading.Event()

        def slow():
            gate.wait(2)
            return "v"
        self.assertIsNone(cache.get("k", slow))
        gate.set()
        deadline = time.time() + 2
        while time.time() < deadline and cache.get("k", slow) is None:
            time.sleep(0.01)
        self.assertEqual(cache.get("k", slow), "v")

    def test_stale_value_is_served_while_rebuilding(self):
        cache = BackgroundCache(ttl=0.0, wait_s=0.0)
        cache.get("k", lambda: "old")
        time.sleep(0.05)               # the first build lands
        gate = threading.Event()

        def slow():
            gate.wait(2)
            return "new"
        self.assertEqual(cache.get("k", slow), "old")
        gate.set()

    def test_value_past_the_stale_limit_waits_for_the_build(self):
        cache = BackgroundCache(ttl=0.0, wait_s=2.0, stale_s=0.0)
        cache.get("k", lambda: "old")
        time.sleep(0.01)
        self.assertEqual(cache.get("k", lambda: "new"), "new")

    def test_value_inside_the_stale_limit_is_served(self):
        cache = BackgroundCache(ttl=0.0, wait_s=0.0, stale_s=60.0)
        cache.get("k", lambda: "old", wait_s=2.0)
        gate = threading.Event()
        self.assertEqual(cache.get("k", lambda: gate.wait(2) and "new"), "old")
        gate.set()

    def test_failed_build_is_not_cached(self):
        cache = BackgroundCache(ttl=60, wait_s=2.0)

        def boom():
            raise RuntimeError("x")
        self.assertIsNone(cache.get("k", boom))
        self.assertEqual(cache.get("k", lambda: 7), 7)


if __name__ == "__main__":
    unittest.main()
