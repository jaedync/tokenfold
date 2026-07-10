"""trigger_eager_rebuild correctness under concurrency.

The original implementation no-opped entirely (no version bump, no cache
clear) when a rebuild was already in flight — with machines pushing every
minute, a billing-reading write usually raced an ingest rebuild and the
dashboard served stale data until the NEXT unrelated invalidation. It also
let a completing rebuild write back data built before a later invalidation.
"""

import threading
import time
import unittest

import app.aggregator as agg
from app.tests._support import TempDBTestCase


def _wait_not_rebuilding(timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with agg._cache_lock:
            if not agg._rebuilding:
                return True
        time.sleep(0.02)
    return False


class TriggerEagerRebuildTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._orig_inner = agg._build_dashboard_data_inner
        self.addCleanup(self._restore)

    def _restore(self):
        agg._build_dashboard_data_inner = self._orig_inner
        with agg._cache_lock:
            agg._rebuilding = False
        agg._cached_data.clear()

    def test_invalidates_even_while_rebuild_in_flight(self):
        with agg._cache_lock:
            agg._rebuilding = True
        agg._cached_data["enterprise"] = {"stale": True}
        v0 = agg.get_cache_version()
        agg.trigger_eager_rebuild()
        self.assertEqual(agg._cached_data, {},
                         "cache must clear even mid-rebuild")
        self.assertGreater(agg.get_cache_version(), v0,
                           "version must bump even mid-rebuild")

    def test_stale_rebuild_result_is_discarded(self):
        """A rebuild that was invalidated again while running must not write its
        (pre-invalidation) result into the cache. Under the drain loop the stale
        first result is dropped and replaced by a fresh build for the newer
        generation — the cache never ends up holding the stale one."""
        builds = []

        def fake_inner(scope):
            n = len(builds) + 1
            builds.append(n)
            if n == 1:
                # a second invalidation lands while this first build is running
                agg.trigger_eager_rebuild()
            return {"built_at_call": n}

        agg._build_dashboard_data_inner = fake_inner
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding(), "rebuild never finished")
        from app.config import DEFAULT_SCOPE
        cached = agg._cached_data.get(DEFAULT_SCOPE)
        # The drain loop must rebuild for the newer generation and cache THAT
        # result — never the stale first build (built_at_call == 1).
        self.assertEqual(cached, {"built_at_call": 2},
                         "stale first-build result must be discarded and "
                         "replaced by the follow-up build's fresh result")


class DrainLoopTest(TempDBTestCase):
    """Drain-loop invariants: one worker, no lost rebuilds, coalescing,
    exception safety, version semantics."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        # A real background rebuild spawned by an earlier test's ingest could
        # still be draining; wait it out so _rebuilding starts False and our
        # first trigger_eager_rebuild actually spawns a worker.
        _wait_not_rebuilding()
        with agg._cache_lock:
            agg._rebuilding = False
        self._orig_inner = agg._build_dashboard_data_inner
        self.addCleanup(self._restore)

    def _restore(self):
        agg._build_dashboard_data_inner = self._orig_inner
        with agg._cache_lock:
            agg._rebuilding = False
        agg._cached_data.clear()

    def test_no_lost_rebuild_and_final_gen_wins(self):
        """An invalidation during a build must produce one more build, and the
        cache must end up holding the LAST build's result (built from the final
        generation) — the drain loop rebuilds rather than dropping it."""
        from app.config import DEFAULT_SCOPE

        build_count = [0]
        in_first_build = threading.Event()
        release_first_build = threading.Event()

        def fake_inner(scope):
            build_count[0] += 1
            n = build_count[0]
            if n == 1:
                in_first_build.set()
                # hold the first build open until the test fires a 2nd invalidation
                self.assertTrue(release_first_build.wait(timeout=5.0),
                                "first build never released")
            return {"scope": scope, "n": n}

        agg._build_dashboard_data_inner = fake_inner
        agg.trigger_eager_rebuild()
        self.assertTrue(in_first_build.wait(timeout=5.0),
                        "first build never started")
        # second invalidation lands mid-build
        agg.trigger_eager_rebuild()
        release_first_build.set()

        self.assertTrue(_wait_not_rebuilding(), "drain loop never finished")
        self.assertEqual(build_count[0], 2,
                         "invalidation mid-build must cause exactly one more build")
        cached = agg._cached_data.get(DEFAULT_SCOPE)
        self.assertIsNotNone(cached, "final build result must be cached")
        self.assertEqual(cached["n"], 2,
                         "cache must hold the LAST build (final generation)")

    def test_at_most_one_worker_thread(self):
        """No matter how many invalidations arrive, only ONE worker thread is
        ever spawned — concurrent invalidations must not stack workers."""
        threads_created = []
        orig_thread = threading.Thread

        def counting_thread(*args, **kwargs):
            t = orig_thread(*args, **kwargs)
            threads_created.append(t)
            return t

        in_first_build = threading.Event()
        release_first_build = threading.Event()
        concurrent_entries = [0]
        entry_lock = threading.Lock()

        def fake_inner(scope):
            with entry_lock:
                concurrent_entries[0] += 1
                self.assertEqual(concurrent_entries[0], 1,
                                 "two builds ran concurrently — >1 worker thread")
            if not in_first_build.is_set():
                in_first_build.set()
                self.assertTrue(release_first_build.wait(timeout=5.0),
                                "first build never released")
            with entry_lock:
                concurrent_entries[0] -= 1
            return {"scope": scope}

        agg._build_dashboard_data_inner = fake_inner
        threading.Thread = counting_thread
        try:
            agg.trigger_eager_rebuild()
            self.assertTrue(in_first_build.wait(timeout=5.0),
                            "first build never started")
            # fire several more invalidations while the first build is held
            for _ in range(4):
                agg.trigger_eager_rebuild()
            release_first_build.set()
            self.assertTrue(_wait_not_rebuilding(), "drain loop never finished")
        finally:
            threading.Thread = orig_thread

        self.assertEqual(len(threads_created), 1,
                         "exactly one worker thread must ever be spawned")

    def test_coalescing_n_invalidations_one_followup(self):
        """5 invalidations arriving during one build coalesce into exactly ONE
        follow-up build — 2 builds total, not 6."""
        build_count = [0]
        in_first_build = threading.Event()
        release_first_build = threading.Event()

        def fake_inner(scope):
            build_count[0] += 1
            if build_count[0] == 1:
                in_first_build.set()
                self.assertTrue(release_first_build.wait(timeout=5.0),
                                "first build never released")
            return {"scope": scope}

        agg._build_dashboard_data_inner = fake_inner
        agg.trigger_eager_rebuild()
        self.assertTrue(in_first_build.wait(timeout=5.0),
                        "first build never started")
        for _ in range(5):
            agg.trigger_eager_rebuild()
        release_first_build.set()

        self.assertTrue(_wait_not_rebuilding(), "drain loop never finished")
        self.assertEqual(build_count[0], 2,
                         "5 invalidations mid-build must coalesce to 1 follow-up "
                         "(2 builds total, not 6)")

    def test_exception_clears_rebuilding_flag(self):
        """If the build fn raises, _rebuilding must be cleared (try/finally) so
        future invalidations can rebuild — the flag must not stick True."""
        def boom(scope):
            raise RuntimeError("build blew up")

        agg._build_dashboard_data_inner = boom
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding(),
                        "_rebuilding stuck True after an exception")
        with agg._cache_lock:
            self.assertFalse(agg._rebuilding)

    def test_version_bumps_once_per_invalidation(self):
        """_cache_version bumps exactly once per invalidation call, regardless of
        the drain loop's internal rebuild count."""
        # hold a build so the drain loop is exercised
        in_first_build = threading.Event()
        release_first_build = threading.Event()
        build_count = [0]

        def fake_inner(scope):
            build_count[0] += 1
            if build_count[0] == 1:
                in_first_build.set()
                self.assertTrue(release_first_build.wait(timeout=5.0),
                                "first build never released")
            return {"scope": scope}

        agg._build_dashboard_data_inner = fake_inner
        v0 = agg.get_cache_version()
        agg.trigger_eager_rebuild()
        self.assertTrue(in_first_build.wait(timeout=5.0),
                        "first build never started")
        # three invalidations while the first build is held → +3 versions
        for _ in range(3):
            agg.trigger_eager_rebuild()
        release_first_build.set()
        self.assertTrue(_wait_not_rebuilding(), "drain loop never finished")
        # 1 initial + 3 mid-build = 4 invalidation calls, each bumps once
        self.assertEqual(agg.get_cache_version(), v0 + 4,
                         "version must bump exactly once per invalidation call")


if __name__ == "__main__":
    unittest.main()
