"""The usage fallback reads the mounted credentials and never refreshes them;
managed source ownership."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
from app import usage_fetcher as f
from app.tests._support import TempDBTestCase


class UsageFetcherTestBase(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.now = 10000
        self.oauth = {"accessToken": "fake-access", "refreshToken": "fake-refresh",
                      "expiresAt": (self.now + 3600) * 1000}
        for name, value in (("_backoff_until", 0), ("_token_state", None)):
            p = patch.object(f, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.creds = patch.object(f, "_read_credentials_file", return_value=({}, self.oauth)).start()
        self.addCleanup(patch.stopall)
        self.http = AsyncMock()
        self.factory = patch.object(f.httpx, "AsyncClient").start()
        self.factory.return_value.__aenter__.return_value = self.http
        self.clock = patch.object(f.time, "time", side_effect=lambda: self.now).start()
        self.printed = patch.object(f, "log").start()

    def usage(self, status, body=None):
        return httpx.Response(status, json=body or {}, request=httpx.Request("GET", f.USAGE_API_URL))


class NoRefreshTest(UsageFetcherTestBase):
    """A refresh-token grant rotates the token server-side and the mount is
    read-only, so a refresh here would spend the token Claude Code and
    Meridian on the host need."""

    def test_module_has_no_refresh_path(self):
        for name in ("_refresh_token_if_needed", "_load_cached_credentials",
                     "_write_credentials_file", "OAUTH_TOKEN_URL"):
            self.assertFalse(hasattr(f, name), name)

    def test_expired_file_token_skips_the_fetch(self):
        self.creds.return_value = ({}, {**self.oauth, "expiresAt": (self.now - 1) * 1000})
        asyncio.run(f._fetch_usage())
        self.http.get.assert_not_called()
        self.http.post.assert_not_called()

    def test_expired_token_is_logged_once_per_change(self):
        self.creds.return_value = ({}, {**self.oauth, "expiresAt": (self.now - 1) * 1000})
        for _ in range(3):
            asyncio.run(f._fetch_usage())
        expired = [c for c in self.printed.call_args_list if "expired" in c.args[0]]
        self.assertEqual(len(expired), 1)

    def test_401_backs_off_without_a_refresh(self):
        self.http.get.return_value = self.usage(401)
        asyncio.run(f._fetch_usage())
        self.http.post.assert_not_called()
        self.assertEqual(f._backoff_until, self.now + f.UNAUTHORIZED_BACKOFF_S)

    def test_429_backs_off(self):
        self.http.get.return_value = self.usage(429)
        asyncio.run(f._fetch_usage())
        self.assertEqual(f._backoff_until, self.now + f.RATE_LIMIT_BACKOFF_S)

    def test_token_comes_from_the_file_only(self):
        fresher = {**self.oauth, "accessToken": "db-access", "expiresAt": (self.now + 7200) * 1000}
        self.conn.execute("INSERT INTO meta VALUES('oauth_credentials',?)", (json.dumps(fresher),))
        self.conn.commit()
        self.assertEqual(f._get_access_token(), "fake-access")

    def test_startup_deletes_the_cached_token_pair(self):
        self.conn.execute("INSERT INTO meta VALUES('oauth_credentials','{}')")
        self.conn.commit()
        self._db._migrate(self.conn)
        self.assertIsNone(self.conn.execute(
            "SELECT 1 FROM meta WHERE key='oauth_credentials'").fetchone())

    def test_the_deleted_token_bytes_leave_the_database_file(self):
        # DELETE alone leaves the row's bytes on a free page of the file
        # (and in every backup made after). secure_delete overwrites them.
        small = "sk-ant-ort01-SMALLSECRETMARKER"
        big = "sk-ant-oat01-" + "BIGSECRETMARKER" * 400  # spills to overflow pages
        self.conn.execute("INSERT INTO meta VALUES('oauth_credentials',?)",
                          (json.dumps({"refreshToken": small, "accessToken": big}),))
        self.conn.commit()
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with open(self._db.DB_PATH, "rb") as fh:
            self.assertIn(b"SMALLSECRETMARKER", fh.read(), "the test row is not in the file")
        before = self.conn.execute("PRAGMA secure_delete").fetchone()[0]
        self._db._migrate(self.conn)
        for suffix in ("", "-wal"):
            try:
                with open(self._db.DB_PATH + suffix, "rb") as fh:
                    data = fh.read()
            except FileNotFoundError:
                continue
            self.assertNotIn(b"SMALLSECRETMARKER", data, f"token bytes stay in the file{suffix}")
            self.assertNotIn(b"BIGSECRETMARKER", data, f"token bytes stay in the file{suffix}")
        self.assertEqual(self.conn.execute("PRAGMA secure_delete").fetchone()[0], before,
                         "secure_delete was not restored")


class ManagedSourceTest(UsageFetcherTestBase):
    def test_managed_source_prevents_server_snapshot_and_history_writes(self):
        managed = {"source": "meridian-oauth", "observed_at_epoch": self.now - 100,
                   "updated_at": "2026-01-01T00:00:00Z", "data": {}}
        self.conn.execute("INSERT INTO meta VALUES('oauth_usage',?)", (json.dumps(managed),))
        self.conn.commit()
        self.http.get.return_value = self.usage(200, {"five_hour": {"utilization": 82}})
        asyncio.run(f._fetch_usage())
        self.assertEqual(json.loads(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0]), managed)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 0)

    def test_transfer_during_server_get_is_guarded_at_write(self):
        from app.claude_usage import store_snapshot

        async def provider_response(*args, **kwargs):
            store_snapshot({"five_hour": {"utilization": 7}}, self.now - 100,
                           "meridian-oauth", metadata={"source_profile": "default"})
            return self.usage(200, {"five_hour": {"utilization": 82}})

        self.http.get.side_effect = provider_response
        asyncio.run(f._fetch_usage())
        self.http.get.assert_awaited_once()
        value = json.loads(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0])
        self.assertEqual(value["source"], "meridian-oauth")
        self.assertEqual(value["data"]["five_hour"]["utilization"], 7)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 1)
