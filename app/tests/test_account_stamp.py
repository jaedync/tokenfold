import os
import sqlite3
import tempfile
import unittest

from app.tests._support import TempDBTestCase


def _arec(uuid="ua"):
    return {"uuid": uuid, "type": "assistant", "timestamp": "2026-06-09T12:00:00Z",
            "sessionId": "s1", "requestId": "r1",
            "message": {"model": "claude-opus-4-8", "id": "m1",
                        "usage": {"input_tokens": 1, "output_tokens": 1}}}


class ExtractStampsAccountTest(unittest.TestCase):
    def test_extract_stamps_account(self):
        from app.ingest import _extract_event, EVENT_COLS
        acct = {"account_email": "me@x.com", "org_name": "Acme",
                "plan": "max", "rate_limit_tier": "max_20x"}
        row = _extract_event(_arec(), "mach", "proj", acct)
        self.assertEqual(row["account_email"], "me@x.com")
        self.assertEqual(row["org_name"], "Acme")
        self.assertEqual(row["plan"], "max")
        self.assertEqual(row["rate_limit_tier"], "max_20x")
        for c in ("account_email", "org_name", "plan", "rate_limit_tier"):
            self.assertIn(c, EVENT_COLS)

    def test_extract_account_absent_is_none(self):
        from app.ingest import _extract_event
        row = _extract_event(_arec(), "mach", "proj")
        self.assertIsNone(row["account_email"])


class DailySummaryHasAccountJsonTest(TempDBTestCase):
    def test_fresh_schema_has_account_json(self):
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(daily_summary)")}
        self.assertIn("account_json", cols)


class MigrateAddsAccountColumnsTest(unittest.TestCase):
    def test_legacy_events_and_daily_summary_migrated(self):
        import app.db as db
        fd, p = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        legacy = sqlite3.connect(p)
        # events with old cols (incl. indexed ones) but no account cols / no speed/geo
        legacy.execute(
            "CREATE TABLE events (uuid TEXT PRIMARY KEY, session_id TEXT, type TEXT, "
            "ts_epoch REAL, day TEXT, request_id TEXT, model TEXT, source_machine TEXT, "
            "project_dir TEXT, is_human_prompt INTEGER, service_tier TEXT)")
        # daily_summary WITHOUT account_json
        legacy.execute("CREATE TABLE daily_summary (day TEXT PRIMARY KEY, cost REAL, "
                       "model_json TEXT, updated_at TEXT)")
        legacy.commit()
        legacy.close()
        saved_path, saved_conn = db.DB_PATH, db._conn
        try:
            db.close_conn()
            db.DB_PATH = p
            db._conn = None
            c = db.get_conn()
            ecols = {r[1] for r in c.execute("PRAGMA table_info(events)")}
            dcols = {r[1] for r in c.execute("PRAGMA table_info(daily_summary)")}
            self.assertTrue(
                {"speed", "inference_geo", "account_email", "org_name",
                 "plan", "rate_limit_tier"} <= ecols, ecols)
            self.assertIn("account_json", dcols)
        finally:
            db.close_conn()
            db.DB_PATH, db._conn = saved_path, saved_conn
            os.unlink(p)


class RouteStampsAccountTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()  # avoid any network from app startup

    def test_route_persists_account(self):
        body = {"machine": "m", "project_dir": "p", "session_file": "s.jsonl",
                "cursor": {"last_line_num": 0}, "events": [_arec()],
                "account_email": "me@x.com", "org_name": "Acme",
                "plan": "max", "rate_limit_tier": "max_20x"}
        c = self.client()
        r = c.post("/api/ingest", json=body, headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        row = self.conn.execute(
            "SELECT account_email, org_name, plan, rate_limit_tier "
            "FROM events WHERE uuid='ua'").fetchone()
        self.assertEqual(tuple(row), ("me@x.com", "Acme", "max", "max_20x"))


if __name__ == "__main__":
    unittest.main()
