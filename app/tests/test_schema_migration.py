import os
import sqlite3
import tempfile
import unittest

from app.tests._support import TempDBTestCase


class SchemaHasSpeedGeoTest(TempDBTestCase):
    def test_fresh_schema_has_speed_geo(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
        self.assertTrue({"speed", "inference_geo"} <= cols, cols)


class SchemaHasSignatureColumnsTest(TempDBTestCase):
    def test_fresh_schema_has_signature_columns(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
        self.assertTrue({"served_model", "sig_version", "sig_header",
                         "sig_cipher_len", "sig_fields"} <= cols, cols)

    def test_fresh_schema_has_the_partial_served_index(self):
        names = {r[1] for r in self.conn.execute("PRAGMA index_list(events)")}
        self.assertIn("idx_events_served", names)


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
            # Signature capture: the columns AND the partial index over them.
            # The index cannot live in SCHEMA (executescript runs before the
            # ALTERs and would abort on the missing column), so a legacy DB is
            # the only place that ordering bug would ever show up.
            self.assertTrue({"served_model", "sig_version", "sig_header",
                             "sig_cipher_len", "sig_fields"} <= cols, cols)
            names = {r[1] for r in c.execute("PRAGMA index_list(events)")}
            self.assertIn("idx_events_served", names)
        finally:
            db.close_conn()
            db.DB_PATH, db._conn = saved_path, saved_conn
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
