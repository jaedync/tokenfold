"""Shared test support: an isolated-DB unittest base for tokenfold.

Repoints app.db at a throwaway temp DB by setting the module globals get_conn()
reads (DB_PATH, _conn) at call time — no importlib.reload needed, because every
caller invokes the same get_conn() function object, which reads these globals
each call. Restores originals on teardown so test modules don't leak state.
"""
import os
import tempfile
import unittest


class TempDBTestCase(unittest.TestCase):
    api_key = "test-key"

    def setUp(self):
        import app.config as _config
        import app.db as _db
        self._db = _db
        self._config = _config
        self._saved_db_path = _db.DB_PATH
        self._saved_conn = _db._conn
        self._saved_config_key = _config.STATS_API_KEY

        _db.close_conn()
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        _db.DB_PATH = self.db_path
        _db._conn = None
        # Patch app.config so the shared require_api_key dependency sees the key.
        _config.STATS_API_KEY = self.api_key
        self.conn = _db.get_conn()
        # Reset scope-keyed aggregator cache so tests don't leak scope state
        import app.aggregator as _agg
        _agg._cached_data.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        # A background rebuild worker (spawned by an ingest in this test) may
        # still be mid-drain, running _build_dashboard_data_inner on the shared
        # sqlite connection. close_conn() below would close it out from under
        # that thread → segfault. Wait the worker out, and clear _warm_scopes so
        # a following test's worker can't rebuild THIS test's leaked scopes.
        import app.aggregator as _agg
        import time as _t
        _deadline = _t.time() + 5.0
        while _t.time() < _deadline:
            with _agg._cache_lock:
                if not _agg._rebuilding:
                    break
            _t.sleep(0.01)
        with _agg._cache_lock:
            _agg._warm_scopes.clear()
        self._db.close_conn()
        self._db.DB_PATH = self._saved_db_path
        self._db._conn = self._saved_conn
        self._config.STATS_API_KEY = self._saved_config_key
        # Clear scope cache so this test's data can't leak into subsequent tests
        _agg._cached_data.clear()
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def freeze_pricing(self):
        """Deterministic + offline pricing for summarizer/aggregator tests: clear any
        LiteLLM-fetched rates (fall back to static MODEL_PRICING) and stub load_pricing
        so summarize_days() never hits the network. Restored on teardown."""
        import app.pricing as _p
        import app.summarizer as _s
        saved = (_p._dynamic_pricing, _p.load_pricing, _s.load_pricing)
        _p._dynamic_pricing = {}
        _p.load_pricing = lambda force=False: None
        _s.load_pricing = lambda force=False: None

        def _restore():
            _p._dynamic_pricing, _p.load_pricing, _s.load_pricing = saved
        self.addCleanup(_restore)

    def client(self):
        """FastAPI TestClient bound to the temp DB. Use as `with self.client() as c:`
        so startup/shutdown fire (matches existing DesktopRouteTest)."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
