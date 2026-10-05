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

    def test_pi_anthropic_message_index_and_historical_backfill(self):
        import app.db as db
        names = {r[1] for r in self.conn.execute("PRAGMA index_list(events)")}
        self.assertIn("idx_events_pi_anthropic_message", names)
        message_id = "msg_historical"
        request_id = f"pi:{'a' * 40}:{message_id}"
        self.conn.execute(
            "INSERT INTO events(uuid,type,timestamp,ts_epoch,day,source_machine,"
            "model,request_id,source_client,provider) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("pi-row", "assistant", "2026-06-09T12:00:00Z", 1781000000,
             "2026-06-09", "m", "claude-opus-4-8", request_id,
             "pi-agent", "anthropic"),
        )
        self.conn.execute(
            "INSERT INTO daily_summary(day,account_email,cost,updated_at) "
            "VALUES('2026-06-09','personal@example.com',5,'now')"
        )
        self.conn.commit()

        db._migrate(self.conn)

        stored = self.conn.execute(
            "SELECT message_id FROM events WHERE uuid='pi-row'"
        ).fetchone()[0]
        self.assertEqual(stored, message_id)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0],
            0,
            "a historical ID backfill must invalidate derived rollups",
        )


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
