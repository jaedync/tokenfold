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




class UpsertDesktopSessionsTest(unittest.TestCase):
    def _fresh_conn(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        self.addCleanup(conn.close)
        return conn

    def test_insert_new_session(self):
        from app.desktop_sessions import upsert_desktop_sessions

        conn = self._fresh_conn()
        result = upsert_desktop_sessions(conn, "host1", [{
            "cli_session_id": "s1",
            "title": "First",
            "last_activity_at_ms": 1000,
        }])
        self.assertEqual(result["inserted"], 1)
        self.assertEqual(result["updated"], 0)

        row = conn.execute(
            "SELECT title, source_machine FROM desktop_sessions WHERE cli_session_id='s1'"
        ).fetchone()
        self.assertEqual(row["title"], "First")
        self.assertEqual(row["source_machine"], "host1")

    def test_update_overwrites_when_newer(self):
        from app.desktop_sessions import upsert_desktop_sessions

        conn = self._fresh_conn()
        upsert_desktop_sessions(conn, "host1", [
            {"cli_session_id": "s1", "title": "old", "last_activity_at_ms": 1000},
        ])
        result = upsert_desktop_sessions(conn, "host1", [
            {"cli_session_id": "s1", "title": "new", "last_activity_at_ms": 2000},
        ])
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(result["updated"], 1)

        title = conn.execute(
            "SELECT title FROM desktop_sessions WHERE cli_session_id='s1'"
        ).fetchone()["title"]
        self.assertEqual(title, "new")

    def test_stale_push_ignored(self):
        from app.desktop_sessions import upsert_desktop_sessions

        conn = self._fresh_conn()
        upsert_desktop_sessions(conn, "host1", [
            {"cli_session_id": "s1", "title": "current", "last_activity_at_ms": 2000},
        ])
        result = upsert_desktop_sessions(conn, "host1", [
            {"cli_session_id": "s1", "title": "stale", "last_activity_at_ms": 1000},
        ])
        self.assertEqual(result["ignored_stale"], 1)

        title = conn.execute(
            "SELECT title FROM desktop_sessions WHERE cli_session_id='s1'"
        ).fetchone()["title"]
        self.assertEqual(title, "current")

    def test_partial_update_preserves_existing_fields(self):
        from app.desktop_sessions import upsert_desktop_sessions

        conn = self._fresh_conn()
        upsert_desktop_sessions(conn, "host1", [{
            "cli_session_id": "s1",
            "title": "keep-me",
            "model": "claude-opus-4-6",
            "last_activity_at_ms": 1000,
        }])
        upsert_desktop_sessions(conn, "host1", [{
            "cli_session_id": "s1",
            "last_activity_at_ms": 2000,
        }])
        row = conn.execute(
            "SELECT title, model FROM desktop_sessions WHERE cli_session_id='s1'"
        ).fetchone()
        self.assertEqual(row["title"], "keep-me")
        self.assertEqual(row["model"], "claude-opus-4-6")

    def test_dict_and_list_fields_serialized_as_json(self):
        from app.desktop_sessions import upsert_desktop_sessions

        conn = self._fresh_conn()
        upsert_desktop_sessions(conn, "host1", [{
            "cli_session_id": "s1",
            "enabled_mcp_tools": {"local:x:y": True, "local:a:b": False},
            "remote_mcp_servers": [{"name": "srv1"}],
            "chrome_allowed_domains": ["example.com"],
            "last_activity_at_ms": 1000,
        }])
        row = conn.execute(
            "SELECT enabled_mcp_tools, remote_mcp_servers, chrome_allowed_domains "
            "FROM desktop_sessions WHERE cli_session_id='s1'"
        ).fetchone()
        import json as _json
        self.assertEqual(_json.loads(row["enabled_mcp_tools"])["local:x:y"], True)
        self.assertEqual(_json.loads(row["remote_mcp_servers"])[0]["name"], "srv1")
        self.assertEqual(_json.loads(row["chrome_allowed_domains"]), ["example.com"])



class DesktopRouteTest(unittest.TestCase):
    def setUp(self):
        import os

        # Snapshot env so mutations do not leak to other test modules
        self._env_snapshot = {
            k: os.environ.get(k) for k in ("DB_PATH", "STATS_API_KEY")
        }

    def tearDown(self):
        import importlib
        import os

        import app.config
        import app.db

        # Restore env
        for k, v in self._env_snapshot.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

        # Reload config and db modules so cached values reflect restored env
        importlib.reload(app.config)
        importlib.reload(app.db)

    def _client(self, api_key: str = "test-key"):
        import importlib
        import os
        import tempfile

        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        self.addCleanup(os.unlink, tmp.name)

        os.environ["DB_PATH"] = tmp.name
        os.environ["STATS_API_KEY"] = api_key

        import app.config
        import app.db
        importlib.reload(app.config)
        importlib.reload(app.db)

        from fastapi.testclient import TestClient
        from app.main import app as fastapi_app
        return TestClient(fastapi_app), tmp.name

    def test_route_requires_api_key(self):
        client, _ = self._client()
        with client:
            r = client.post(
                "/api/desktop-metadata",
                json={"machine": "h", "sessions": []},
                headers={"X-API-Key": "wrong"},
            )
        self.assertEqual(r.status_code, 401)
        self.assertIn("detail", r.json())

    def test_route_happy_path(self):
        client, _ = self._client(api_key="kk")
        with client:
            r = client.post(
                "/api/desktop-metadata",
                json={
                    "machine": "host1",
                    "sessions": [
                        {
                            "cli_session_id": "aaaa-1",
                            "title": "T",
                            "last_activity_at_ms": 1000,
                        }
                    ],
                },
                headers={"X-API-Key": "kk"},
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["inserted"], 1)
        self.assertEqual(body["updated"], 0)
        self.assertEqual(body["ignored_stale"], 0)

if __name__ == "__main__":
    unittest.main()
