"""Shared test support: an isolated-DB unittest base for tokenfold.

Repoints app.db at a throwaway temp DB by setting the module globals get_conn()
reads (DB_PATH, _conn) at call time — no importlib.reload needed, because every
caller invokes the same get_conn() function object, which reads these globals
each call. Restores originals on teardown so test modules don't leak state.
"""
import os
import tempfile
import unittest

# Startup prewarm (app/prewarm.py, limit_rate.prewarm) reads the database on
# background threads. It stays off under tests; the tests of the prewarm turn
# it on (app.config.PREWARM) or call it directly. Set before app.config is
# imported here, and patched in setUp for a config imported earlier.
os.environ["TOKENFOLD_PREWARM"] = "0"


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
        # open_conn: a previous test's app shutdown leaves the database closed.
        self.conn = _db.open_conn()
        # Reset scope-keyed aggregator cache so tests don't leak scope state
        import app.aggregator as _agg
        _agg._cached_data.clear()
        _agg._previous_data.clear()
        _agg._cached_at.clear()
        self.addCleanup(self._restore)
        from unittest.mock import patch as _patch
        _prewarm = _patch.object(_config, "PREWARM", False)
        _prewarm.start()
        self.addCleanup(_prewarm.stop)

    def _restore(self):
        # Background threads of this test (the cache rebuild worker an ingest
        # spawned, a limit-rate fit, background cache builds, the prewarm) must
        # not outlive its database. They read on their own connections and
        # write through db.background_write, so a late one cannot crash on the
        # close below; joining them keeps their reads off the next test's
        # database. Clear _warm_scopes so a following test's worker can't
        # rebuild THIS test's leaked scopes.
        import app.aggregator as _agg
        import app.bg_cache as _bg
        import app.limit_rate as _lr
        import app.prewarm as _pw
        _agg.join_rebuild(5.0)
        _lr.join_refresh(5.0)
        _pw.join(5.0)
        _bg.join_all(5.0)
        with _agg._cache_lock:
            _agg._warm_scopes.clear()
        self._db.close_conn()
        self._db.DB_PATH = self._saved_db_path
        self._db._conn = self._saved_conn
        self._config.STATS_API_KEY = self._saved_config_key
        # Clear scope cache so this test's data can't leak into subsequent tests
        _agg._cached_data.clear()
        _agg._previous_data.clear()
        _agg._cached_at.clear()
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
