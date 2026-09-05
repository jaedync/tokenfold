"""Cache publication, wall-clock freshness, single-flight and transport contracts."""
import inspect
import threading
import unittest
from unittest.mock import patch
from app.tests._support import TempDBTestCase


class DashboardFreshnessTest(TempDBTestCase):
    def test_build_straddling_invalidation_keeps_original_revision(self):
        import app.aggregator as agg
        before = agg.get_cache_version()
        def build(scope):
            with agg._cache_lock:
                agg._cache_gen += 1
                agg._cache_version_bump()
            return {'version': agg.get_cache_version()}
        with patch.object(agg, '_build_dashboard_data_inner', side_effect=build):
            data = agg.build_dashboard_data('personal')
        self.assertEqual(data['version'], before)
        self.assertNotIn('personal', agg._cached_data)

    def test_concurrent_cold_reads_build_once(self):
        import app.aggregator as agg
        entered, release = threading.Event(), threading.Event()
        def build(scope):
            entered.set(); release.wait(2)
            return {'version': agg.get_cache_version()}
        with patch.object(agg, '_build_dashboard_data_inner', side_effect=build) as build_mock:
            threads = [threading.Thread(target=agg.build_dashboard_data, args=('personal',)) for _ in range(2)]
            for thread in threads: thread.start()
            self.assertTrue(entered.wait(1)); release.set()
            for thread in threads: thread.join(3)
        self.assertEqual(build_mock.call_count, 1)

    def test_clock_expiry_triggers_rebuild_without_ingest(self):
        import app.aggregator as agg
        with patch.object(agg, '_build_dashboard_data_inner', return_value={'version':0}):
            agg.build_dashboard_data('personal')
        with patch.object(agg._time, 'monotonic', return_value=10**12), patch.object(agg, 'trigger_eager_rebuild') as rebuild:
            agg.build_dashboard_data('personal')
        rebuild.assert_called_once()

    def test_blocking_routes_are_not_coroutine_handlers(self):
        from app.api import stats
        from app.dashboard import dashboard
        from app.spend_history import spend_history
        from app.served_models import served_models_timeline
        for handler in (stats, dashboard, spend_history, served_models_timeline):
            self.assertFalse(inspect.iscoroutinefunction(handler), handler.__name__)

    def test_continuous_ingest_publishes_intermediate_progress(self):
        import app.aggregator as agg
        first, second, release_first, release_second = (threading.Event() for _ in range(4))
        calls = []
        def build(scope):
            calls.append(scope)
            n = len(calls)
            (first if n == 1 else second).set()
            (release_first if n == 1 else release_second).wait(3)
            return {'n': n, 'version': agg.get_cache_version()}
        with agg._cache_lock:
            agg._warm_scopes.clear()
            agg._cached_data[agg.DEFAULT_SCOPE] = {'n':0, 'version':agg._cache_version}
        with patch.object(agg, '_build_dashboard_data_inner', side_effect=build):
            try:
                agg.trigger_eager_rebuild(); self.assertTrue(first.wait(1))
                agg.trigger_eager_rebuild(); release_first.set()
                self.assertTrue(second.wait(1))
                progress = agg.build_dashboard_data(agg.DEFAULT_SCOPE)
                self.assertEqual(progress['n'], 1)
                self.assertLess(progress['version'], agg.get_cache_version())
            finally:
                release_first.set(); release_second.set()
                import time
                end = time.monotonic() + 3
                while agg._rebuilding and time.monotonic() < end: time.sleep(.01)

    def test_secondary_read_cache_coalesces_and_expires(self):
        from app.read_cache import ReadCache
        cache = ReadCache(ttl=30)
        with patch('app.read_cache.time.monotonic', return_value=0):
            self.assertEqual(cache.get(('personal',), lambda:1), 1)
            self.assertEqual(cache.get(('personal',), lambda:2), 1)
            self.assertEqual(cache.get(('enterprise',), lambda:3), 3)
        with patch('app.read_cache.time.monotonic', return_value=31):
            self.assertEqual(cache.get(('personal',), lambda:2), 2)

    def test_private_json_and_sse_transport_headers(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get('/health')
        self.assertEqual(response.status_code, 200)
        response = client.get('/api/stats/version')
        self.assertEqual(response.headers['cache-control'], 'private, no-store')
        from app.stream import stats_stream
        import asyncio
        response = asyncio.run(stats_stream(None))
        self.assertIn('no-transform', response.headers['cache-control'])
        self.assertEqual(response.headers['content-encoding'], 'identity')
