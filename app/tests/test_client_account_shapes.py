"""Tests for read_account() robustness against malformed config shapes.

Fix 1: read_account must not crash (AttributeError/TypeError) when config files
contain non-dict JSON values. All cases must return the default-shaped dict.
"""
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

_EMPTY = {"account_email": None, "org_name": None, "plan": None, "rate_limit_tier": None}


def _make_env(home, claude_json_content=None, credentials_content=None):
    """Write temp home with optional .claude.json and .credentials.json."""
    if claude_json_content is not None:
        (home / ".claude.json").write_text(
            json.dumps(claude_json_content) if not isinstance(claude_json_content, str)
            else claude_json_content
        )
    cdir = home / ".claude"
    cdir.mkdir(exist_ok=True)
    if credentials_content is not None:
        (cdir / ".credentials.json").write_text(
            json.dumps(credentials_content) if not isinstance(credentials_content, str)
            else credentials_content
        )
    return cdir


class ReadAccountShapeTest(unittest.TestCase):
    """Each case must return the default dict without raising."""

    def _call(self, home):
        with mock.patch.object(push.Path, "home", staticmethod(lambda: home)):
            return push.read_account(home / ".claude")

    def test_case1_claude_json_is_list(self):
        """Case 1: .claude.json content is [] (list, not dict) — must not crash."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(home, claude_json_content=[])
            result = self._call(home)
        self.assertEqual(result, _EMPTY)

    def test_case2_claude_json_is_string(self):
        """Case 2: .claude.json content is "hello" (string) — must not crash."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            # Write raw string JSON (json.dumps("hello") -> '"hello"')
            (home / ".claude.json").write_text('"hello"')
            cdir = home / ".claude"
            cdir.mkdir(exist_ok=True)
            result = self._call(home)
        self.assertEqual(result, _EMPTY)

    def test_case3_oauth_account_is_string(self):
        """Case 3: oauthAccount present but a string — must not crash."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(home, claude_json_content={"oauthAccount": "nope"})
            result = self._call(home)
        self.assertEqual(result["account_email"], None)
        self.assertEqual(result["org_name"], None)

    def test_case4_credentials_json_is_null(self):
        """Case 4: .credentials.json content is null (None) — must not crash."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(home, credentials_content=None)
            # Write literal "null" (json.dumps(None) = "null")
            (home / ".claude" / ".credentials.json").write_text("null")
            result = self._call(home)
        self.assertEqual(result["plan"], None)
        self.assertEqual(result["rate_limit_tier"], None)

    def test_case5_claude_ai_oauth_is_list(self):
        """Case 5: claudeAiOauth is a list — must not crash."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(home, credentials_content={"claudeAiOauth": ["item1", "item2"]})
            result = self._call(home)
        self.assertEqual(result["plan"], None)
        self.assertEqual(result["rate_limit_tier"], None)

    def test_case6_well_formed_extracts_correctly(self):
        """Case 6: well-formed input — correct values extracted (positive case)."""
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(
                home,
                claude_json_content={"oauthAccount": {
                    "emailAddress": "jane@example.com",
                    "organizationName": "ExampleCorp",
                }},
                credentials_content={"claudeAiOauth": {
                    "subscriptionType": "pro",
                    "rateLimitTier": "default",
                }},
            )
            result = self._call(home)
        self.assertEqual(result, {
            "account_email": "jane@example.com",
            "org_name": "ExampleCorp",
            "plan": "pro",
            "rate_limit_tier": "default",
        })

    def test_case7_security_no_tokens_in_result(self):
        """Case 7: SECURITY — accessToken/refreshToken must NOT appear in result.

        read_account must read ONLY emailAddress, organizationName,
        subscriptionType, rateLimitTier — never access/refresh tokens.
        """
        with tempfile.TemporaryDirectory() as h:
            home = Path(h)
            cdir = _make_env(
                home,
                credentials_content={"claudeAiOauth": {
                    "accessToken": "SECRET",
                    "refreshToken": "SECRET2",
                    "subscriptionType": "enterprise",
                    "rateLimitTier": "max_20x",
                }},
            )
            result = self._call(home)

        self.assertEqual(result["plan"], "enterprise",
                         "subscriptionType should still be extracted")
        result_repr = repr(result)
        self.assertNotIn("SECRET", result_repr,
                         "accessToken must not appear in read_account result")
        self.assertNotIn("SECRET2", result_repr,
                         "refreshToken must not appear in read_account result")


if __name__ == "__main__":
    unittest.main()
