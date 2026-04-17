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


if __name__ == "__main__":
    unittest.main()
