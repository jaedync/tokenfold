"""The push client reads the Claude login and never refreshes or writes it.

A refresh-token grant rotates the token on the server side. Claude Code (and
anything else on the machine that shares ~/.claude/.credentials.json) then
holds a spent refresh token. Claude Code renews the file when it runs, so an
expired token only skips the usage fetch.

Run with:
    python client/test_oauth_readonly.py
"""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("csp_oauth", HERE / "claude-stats-push.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MOD = _load()


def _no_network(*args, **kwargs):
    raise AssertionError("the client made a network call")


class OAuthReadOnlyTest(unittest.TestCase):
    def _creds(self, expires_at_ms):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = pathlib.Path(tmp.name) / ".credentials.json"
        path.write_text(json.dumps({"claudeAiOauth": {
            "accessToken": "access-abc", "refreshToken": "refresh-xyz", "expiresAt": expires_at_ms,
            "subscriptionType": "max",
        }}))
        os.utime(path, (1_700_000_000, 1_700_000_000))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(MOD, "CREDENTIALS_FILE", path).start()
        return path, tmp.name

    def _call(self):
        out = io.StringIO()
        with mock.patch.object(MOD.urllib.request, "urlopen", _no_network), contextlib.redirect_stderr(out):
            token = MOD._get_oauth_token()
        return token, out.getvalue()

    def test_expired_token_skips_without_a_refresh_or_a_write(self):
        path, folder = self._creds(int(time.time() * 1000) - 60_000)
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        token, out = self._call()
        self.assertIsNone(token)
        self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), before, "the credentials file changed")
        self.assertEqual(sorted(os.listdir(folder)), [".credentials.json"], "the client wrote a file beside it")
        self.assertIn("expired", out)
        self.assertNotIn("access-abc", out)
        self.assertNotIn("refresh-xyz", out)

    def test_token_near_expiry_is_still_used_as_is(self):
        # The old code refreshed 5 min before the expiry; the token is valid until then.
        path, _ = self._creds(int(time.time() * 1000) + 60_000)
        before = path.read_bytes()
        token, _ = self._call()
        self.assertEqual(token, "access-abc")
        self.assertEqual(path.read_bytes(), before)

    def test_valid_token_is_returned(self):
        self._creds(int(time.time() * 1000) + 3_600_000)
        token, out = self._call()
        self.assertEqual(token, "access-abc")
        self.assertEqual(out, "")

    def test_missing_file_or_token(self):
        path, _ = self._creds(0)
        path.write_text(json.dumps({"claudeAiOauth": {}}))
        self.assertIsNone(self._call()[0])
        path.unlink()
        self.assertIsNone(self._call()[0])

    def test_no_refresh_grant_or_credentials_write_in_the_client(self):
        for name in ("claude-stats-push.py", "refresh-oauth-token.sh"):
            text = (HERE / name).read_text()
            for needle in ("grant_type", "refresh_token", "oauth/token", "refreshToken"):
                self.assertFalse(needle in text, f"{name} still has {needle!r}")
        push = (HERE / "claude-stats-push.py").read_text()
        self.assertIn("from __future__ import annotations", push)
        self.assertFalse("CREDENTIALS_FILE.with_suffix" in push, "the client writes the credentials file")


if __name__ == "__main__":
    unittest.main()
