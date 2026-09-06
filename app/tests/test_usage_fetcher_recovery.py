"""Mocked refresh deadlines, credential rotation, managed source ownership."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
from app import usage_fetcher as f
from app.tests._support import TempDBTestCase


class UsageFetcherRecoveryTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.oauth = {"accessToken": "fake-access", "refreshToken": "fake-refresh", "expiresAt": 1}
        for name, value in (("_cached_oauth", None), ("_consecutive_refresh_failures", 0),
                            ("_refresh_identity", None), ("_refresh_retry_at", 0),
                            ("_file_identity", None), ("_backoff_until", 0)):
            p = patch.object(f, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        self.creds = patch.object(f, "_read_credentials_file", return_value=({}, self.oauth)).start()
        self.addCleanup(patch.stopall)
        self.write_file = patch.object(f, "_write_credentials_file").start()
        self.http = AsyncMock()
        self.factory = patch.object(f.httpx, "AsyncClient").start()
        self.factory.return_value.__aenter__.return_value = self.http
        self.now = 10000
        self.clock = patch.object(f.time, "time", side_effect=lambda: self.now).start()
        response = httpx.Response(403, request=httpx.Request("POST", f.OAUTH_TOKEN_URL))
        self.http.post.return_value = response

    def refresh(self):
        asyncio.run(f._refresh_token_if_needed(force=True))

    def test_retry_threshold_has_real_hourly_deadline(self):
        for _ in range(5):
            self.refresh()
        self.assertEqual(self.http.post.call_count, 5)
        self.assertEqual(f._refresh_retry_at, self.now + 3600)
        self.now += 3599
        self.refresh()
        self.assertEqual(self.http.post.call_count, 5)
        self.now += 1
        self.refresh()
        self.assertEqual(self.http.post.call_count, 6)
        self.assertEqual(f._refresh_retry_at, self.now + 3600)

    def test_changed_credentials_recover_even_with_earlier_expiry(self):
        for _ in range(5):
            self.refresh()
        self.creds.return_value = ({}, {**self.oauth, "accessToken": "rotated-access", "refreshToken": "rotated-refresh", "expiresAt": 0})
        self.http.post.return_value = httpx.Response(200, json={"access_token": "recovered", "expires_in": 7200},
                                                   request=httpx.Request("POST", f.OAUTH_TOKEN_URL))
        self.refresh()
        self.assertEqual(self.http.post.call_count, 6)
        self.assertEqual(self.http.post.call_args.kwargs["json"]["refresh_token"], "rotated-refresh")
        self.assertEqual(f._consecutive_refresh_failures, 0)
        self.assertEqual(f._refresh_retry_at, 0)
        self.assertEqual(f._cached_oauth["accessToken"], "recovered")
        # The still-mounted old file must not undo the just-refreshed token.
        self.assertEqual(f._get_oauth()["accessToken"], "recovered")

    def test_missing_access_token_response_also_backs_off(self):
        self.http.post.return_value = httpx.Response(200, json={}, request=httpx.Request("POST", f.OAUTH_TOKEN_URL))
        for _ in range(6):
            self.refresh()
        self.assertEqual(self.http.post.call_count, 5)
        self.assertEqual(f._refresh_retry_at, self.now + 3600)

    def test_managed_source_prevents_server_snapshot_and_history_writes(self):
        managed = {"source": "meridian-oauth", "observed_at_epoch": self.now - 100,
                   "updated_at": "2026-01-01T00:00:00Z", "data": {}}
        self.conn.execute("INSERT INTO meta VALUES('oauth_usage',?)", (json.dumps(managed),))
        self.conn.commit()
        self.http.get.return_value = httpx.Response(200, json={"five_hour": {"utilization": 82}},
                                                  request=httpx.Request("GET", f.USAGE_API_URL))
        asyncio.run(f._fetch_usage())
        self.assertEqual(json.loads(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0]), managed)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 0)

    def test_transfer_during_server_get_is_guarded_at_write(self):
        from app.claude_usage import store_snapshot

        async def provider_response(*args, **kwargs):
            store_snapshot({"five_hour": {"utilization": 7}}, self.now - 100,
                           "meridian-oauth", metadata={"source_profile": "default"})
            return httpx.Response(200, json={"five_hour": {"utilization": 82}},
                                  request=httpx.Request("GET", f.USAGE_API_URL))

        self.http.get.side_effect = provider_response
        asyncio.run(f._fetch_usage())
        self.http.get.assert_awaited_once()
        value = json.loads(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0])
        self.assertEqual(value["source"], "meridian-oauth")
        self.assertEqual(value["data"]["five_hour"]["utilization"], 7)
        self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 1)
