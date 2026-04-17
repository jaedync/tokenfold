import sqlite3
import unittest

from app.db import SCHEMA


class DesktopSessionsSchemaTest(unittest.TestCase):
    def test_table_exists_after_schema_apply(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='desktop_sessions'"
        )
        self.assertIsNotNone(cur.fetchone())

    def test_required_columns_present(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(desktop_sessions)")}
        required = {
            "cli_session_id", "desktop_session_id", "source_machine",
            "title", "model", "effort", "permission_mode", "completed_turns",
            "is_archived", "cwd", "origin_cwd",
            "created_at_ms", "last_activity_at_ms",
            "enabled_mcp_tools", "remote_mcp_servers",
            "chrome_permission_mode", "chrome_allowed_domains",
            "updated_at_ms",
        }
        missing = required - cols
        self.assertFalse(missing, f"missing columns: {missing}")


class DesktopModelTest(unittest.TestCase):
    def test_accepts_full_payload(self):
        from app.models import DesktopSessionUpsert

        row = DesktopSessionUpsert(
            cli_session_id="aaaa-1",
            desktop_session_id="local_xxx",
            title="Review MCP",
            model="claude-opus-4-6",
            effort="high",
            permission_mode="default",
            completed_turns=3,
            is_archived=False,
            cwd="/tmp",
            origin_cwd="/tmp",
            created_at_ms=1_700_000_000_000,
            last_activity_at_ms=1_700_000_100_000,
            enabled_mcp_tools={"local:x:y": True},
            remote_mcp_servers=[{"name": "foo"}],
            chrome_permission_mode="allowAll",
            chrome_allowed_domains=["example.com"],
        )
        self.assertEqual(row.cli_session_id, "aaaa-1")
        self.assertEqual(row.is_archived, False)

    def test_minimum_required_is_cli_session_id(self):
        from app.models import DesktopSessionUpsert

        row = DesktopSessionUpsert(cli_session_id="aaaa-2")
        self.assertIsNone(row.title)
        self.assertIsNone(row.last_activity_at_ms)

    def test_missing_cli_session_id_rejected(self):
        from pydantic import ValidationError

        from app.models import DesktopSessionUpsert
        with self.assertRaises(ValidationError):
            DesktopSessionUpsert()

    def test_request_wrapper_accepts_list(self):
        from app.models import DesktopMetadataRequest, DesktopSessionUpsert

        req = DesktopMetadataRequest(
            machine="host1",
            sessions=[
                DesktopSessionUpsert(cli_session_id="a"),
                DesktopSessionUpsert(cli_session_id="b"),
            ],
        )
        self.assertEqual(len(req.sessions), 2)


if __name__ == "__main__":
    unittest.main()
