"""ReadCache (app/read_cache.py): single flight per key, no global build lock."""
import threading
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
