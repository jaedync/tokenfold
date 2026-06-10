import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "push", ROOT / "client" / "claude-stats-push.py")
push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(push)


class ReadAccountTest(unittest.TestCase):
    def test_reads_email_org_plan_tier(self):
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            (home / ".claude.json").write_text(json.dumps(
                {"oauthAccount": {"emailAddress": "me@x.com",
                                  "organizationName": "Acme"}}))
            cdir = home / ".claude"
            cdir.mkdir()
            (cdir / ".credentials.json").write_text(json.dumps(
                {"claudeAiOauth": {"subscriptionType": "max",
                                   "rateLimitTier": "max_20x"}}))
            with mock.patch.object(push.Path, "home", staticmethod(lambda: home)):
                acct = push.read_account(home / ".claude")
            self.assertEqual(acct, {"account_email": "me@x.com", "org_name": "Acme",
                                    "plan": "max", "rate_limit_tier": "max_20x",
                                    "org_type": None, "org_uuid": None})

    def test_missing_files_return_nones(self):
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            with mock.patch.object(push.Path, "home", staticmethod(lambda: home)):
                acct = push.read_account(home / ".claude")
            self.assertEqual(acct, {"account_email": None, "org_name": None,
                                    "plan": None, "rate_limit_tier": None,
                                    "org_type": None, "org_uuid": None})


if __name__ == "__main__":
    unittest.main()
