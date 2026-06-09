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
        import app.db as _db
        import app.ingest as _ingest
        self._db = _db
        self._ingest = _ingest
        self._saved_db_path = _db.DB_PATH
        self._saved_conn = _db._conn
        self._saved_key = _ingest.STATS_API_KEY

        _db.close_conn()
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        _db.DB_PATH = self.db_path
        _db._conn = None
        _ingest.STATS_API_KEY = self.api_key
        self.conn = _db.get_conn()
        self.addCleanup(self._restore)

    def _restore(self):
        self._db.close_conn()
        self._db.DB_PATH = self._saved_db_path
        self._db._conn = self._saved_conn
        self._ingest.STATS_API_KEY = self._saved_key
        try:
            os.unlink(self.db_path)
        except OSError:
            pass

    def client(self):
        """FastAPI TestClient bound to the temp DB. Use as `with self.client() as c:`
        so startup/shutdown fire (matches existing DesktopRouteTest)."""
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
