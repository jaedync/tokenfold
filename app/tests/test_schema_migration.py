import os
import sqlite3
import tempfile
import unittest

from app.tests._support import TempDBTestCase


class SchemaHasSpeedGeoTest(TempDBTestCase):
    def test_fresh_schema_has_speed_geo(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
        self.assertTrue({"speed", "inference_geo"} <= cols, cols)


class MigrateLegacyDBTest(unittest.TestCase):
    def test_migrate_adds_columns_to_legacy_events(self):
        import app.db as db
        fd, p = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        # Pre-migration DB: a realistic events table that has the old columns
        # the SCHEMA indexes reference, just missing speed/inference_geo.
        legacy = sqlite3.connect(p)
        legacy.execute(
            "CREATE TABLE events ("
            "  uuid TEXT PRIMARY KEY, session_id TEXT, type TEXT, ts_epoch REAL, "
            "  day TEXT, request_id TEXT, model TEXT, source_machine TEXT, "
            "  project_dir TEXT, is_human_prompt INTEGER, service_tier TEXT"
            ")")
        legacy.commit()
        legacy.close()
        saved_path, saved_conn = db.DB_PATH, db._conn
        try:
            db.close_conn()
            db.DB_PATH = p
            db._conn = None
            c = db.get_conn()  # must ALTER in the missing columns, not crash
            cols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
            self.assertTrue({"speed", "inference_geo"} <= cols, cols)
        finally:
            db.close_conn()
            db.DB_PATH, db._conn = saved_path, saved_conn
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
