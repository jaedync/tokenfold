"""The vendored uploader must distinguish upstream OAuth from ingest auth."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import urllib.error

CLIENT_PATH = Path(__file__).resolve().parents[2] / "client" / "claude-stats-push.py"


class VendoredUsageDiagnosticTest(unittest.TestCase):
    def test_upstream_auth_failure_does_not_blame_ingest_key(self):
        with patch.dict(os.environ, {"TOKENFOLD_URL": "http://localhost:1", "TOKENFOLD_API_KEY": "fake-ingest"}):
            spec = importlib.util.spec_from_file_location("vendored_client", CLIENT_PATH)
            client = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(client)
        for status in (401, 403):
            with self.subTest(status=status), \
                    patch.object(client, "_usage_is_backed_off", return_value=False), \
                    patch.object(client, "_usage_fetch_too_soon", return_value=False), \
                    patch.object(client, "_usage_fetch_stamp"), \
                    patch.object(client, "_get_oauth_token", return_value="fake-oauth"), \
                    patch.object(client, "_get_claude_version", return_value="0.0.0"), \
                    patch.object(client, "err") as log, \
                    patch.object(client.urllib.request, "urlopen", side_effect=urllib.error.HTTPError(
                        "https://api.anthropic.com/api/oauth/usage", status, "denied", {}, None)):
                client._fetch_and_push_usage()
                message = log.call_args.args[0]
                self.assertIn("Anthropic", message)
                self.assertIn("OAuth", message)
                self.assertNotIn("must match", message)
                self.assertNotIn("fake-oauth", message)
                self.assertNotIn("fake-ingest", message)
