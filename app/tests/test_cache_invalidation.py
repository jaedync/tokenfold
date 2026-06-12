"""trigger_eager_rebuild correctness under concurrency.

The original implementation no-opped entirely (no version bump, no cache
clear) when a rebuild was already in flight — with machines pushing every
minute, a billing-reading write usually raced an ingest rebuild and the
dashboard served stale data until the NEXT unrelated invalidation. It also
let a completing rebuild write back data built before a later invalidation.
"""

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
        """A rebuild that was invalidated again while running must not write
        its (pre-invalidation) result into the cache."""
        calls = []

        def fake_inner(scope):
            if not calls:
                calls.append(1)
                # a second invalidation lands while this build is running
                agg.trigger_eager_rebuild()
            return {"built_at_gen": len(calls)}

        agg._build_dashboard_data_inner = fake_inner
        agg.trigger_eager_rebuild()
        self.assertTrue(_wait_not_rebuilding(), "rebuild never finished")
        from app.config import DEFAULT_SCOPE
        cached = agg._cached_data.get(DEFAULT_SCOPE)
        self.assertIsNone(cached,
                          "first build ran before the second invalidation — "
                          "its result is stale and must be discarded")


if __name__ == "__main__":
    unittest.main()
