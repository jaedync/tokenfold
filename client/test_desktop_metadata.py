"""Standalone tests for the desktop metadata helpers in claude-stats-push.py.

Run with:
    python client/test_desktop_metadata.py
"""

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


def _load_module():
    """Load claude-stats-push.py as a module (hyphen in name blocks normal import)."""
    here = pathlib.Path(__file__).resolve().parent
    src = here / "claude-stats-push.py"
    spec = importlib.util.spec_from_file_location("claude_stats_push", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


SAMPLE_SESSION = {
    "sessionId": "local_70eff8b0-29e0-4660-96ad-9e9a8f657825",
    "cliSessionId": "017b6056-de7c-43a3-a313-72073203dc33",
    "cwd": "/Users/jaedy/Documents",
    "originCwd": "/Users/jaedy/Documents",
    "createdAt": 1772763369505,
    "lastActivityAt": 1772763425424,
    "model": "claude-opus-4-6",
    "isArchived": True,
    "title": "Review MCP server configuration",
    "permissionMode": "default",
    "effort": "high",
    "completedTurns": 4,
    "enabledMcpTools": {"local:foo:bar": True, "local:foo:baz": False},
    "remoteMcpServersConfig": [{"name": "srv", "tools": [{"name": "t1"}]}],
    "chromePermissionMode": "ask",
    "chromeAllowedDomains": ["example.com"],
}


class ExtractDesktopSessionTest(unittest.TestCase):
    def _write(self, data):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, tmp)
        tmp.close()
        p = pathlib.Path(tmp.name)
        self.addCleanup(p.unlink, missing_ok=True)
        return p

    def test_full_record_normalized(self):
        p = self._write(SAMPLE_SESSION)
        row = MOD.extract_desktop_session(p)

        self.assertEqual(row["cli_session_id"], "017b6056-de7c-43a3-a313-72073203dc33")
        self.assertEqual(row["desktop_session_id"], "local_70eff8b0-29e0-4660-96ad-9e9a8f657825")
        self.assertEqual(row["title"], "Review MCP server configuration")
        self.assertEqual(row["model"], "claude-opus-4-6")
        self.assertEqual(row["is_archived"], True)
        self.assertEqual(row["created_at_ms"], 1772763369505)
        self.assertEqual(row["last_activity_at_ms"], 1772763425424)
        self.assertEqual(row["enabled_mcp_tools"], SAMPLE_SESSION["enabledMcpTools"])

    def test_missing_cli_session_id_returns_none(self):
        data = dict(SAMPLE_SESSION)
        data.pop("cliSessionId")
        p = self._write(data)
        self.assertIsNone(MOD.extract_desktop_session(p))

    def test_unreadable_file_returns_none(self):
        p = pathlib.Path(tempfile.gettempdir()) / "does-not-exist.json"
        self.assertIsNone(MOD.extract_desktop_session(p))

    def test_malformed_json_returns_none(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write("{not json")
        tmp.close()
        p = pathlib.Path(tmp.name)
        self.addCleanup(p.unlink, missing_ok=True)
        self.assertIsNone(MOD.extract_desktop_session(p))


class FindDesktopSessionsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.session_dir = self.tmp / "acct" / "org"
        self.session_dir.mkdir(parents=True)
        import shutil
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_session(self, cli_id: str, last_activity_ms: int):
        data = dict(SAMPLE_SESSION)
        data["cliSessionId"] = cli_id
        data["lastActivityAt"] = last_activity_ms
        p = self.session_dir / f"local_{cli_id}.json"
        p.write_text(json.dumps(data))

    def test_returns_only_updated_since_cursor(self):
        self._write_session("old-1", 1000)
        self._write_session("new-1", 5000)
        rows = MOD.find_desktop_sessions(self.tmp, cursor_ms=2000)
        ids = sorted(r["cli_session_id"] for r in rows)
        self.assertEqual(ids, ["new-1"])

    def test_no_cursor_returns_all(self):
        self._write_session("a", 1000)
        self._write_session("b", 2000)
        rows = MOD.find_desktop_sessions(self.tmp, cursor_ms=0)
        self.assertEqual(len(rows), 2)

    def test_missing_dir_returns_empty(self):
        rows = MOD.find_desktop_sessions(self.tmp / "nonexistent", cursor_ms=0)
        self.assertEqual(rows, [])


class DesktopCursorTest(unittest.TestCase):
    def test_cursor_key_constant(self):
        self.assertEqual(MOD.DESKTOP_CURSOR_KEY, "__desktop_last_activity_ms")

    def test_read_missing_returns_zero(self):
        self.assertEqual(MOD.read_desktop_cursor({}), 0)

    def test_read_existing(self):
        cursors = {MOD.DESKTOP_CURSOR_KEY: 4242}
        self.assertEqual(MOD.read_desktop_cursor(cursors), 4242)

    def test_write_updates_in_place(self):
        cursors = {"some/path.jsonl": {"line": 10}}
        MOD.write_desktop_cursor(cursors, 9999)
        self.assertEqual(cursors[MOD.DESKTOP_CURSOR_KEY], 9999)
        self.assertEqual(cursors["some/path.jsonl"], {"line": 10})


if __name__ == "__main__":
    unittest.main()
