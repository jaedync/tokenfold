"""GET /next: the new dashboard shell.

Covers dashboard auth, the scope precedence shared with GET / (lock > ?scope=
> tf_scope cookie > default, soft-fail on a bad scope), escaping of the owner
name and the no-store header on the authenticated page.
"""
import base64
import re
import unittest
from unittest.mock import patch

from app.tests._support import TempDBTestCase


def data_attr(html, name):
    match = re.search(rf'data-{name}="([^"]*)"', html)
    return match.group(1) if match else None


class NextPageAuthTest(TempDBTestCase):
    def test_open_when_no_password(self):
        with patch.object(self._config, "DASHBOARD_PASSWORD", ""):
            response = self.client().get("/next")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="view"', response.text)

    def test_requires_basic_auth_when_password_set(self):
        with patch.object(self._config, "DASHBOARD_PASSWORD", "secret"):
            response = self.client().get("/next")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("www-authenticate"), "Basic")

    def test_wrong_password_rejected(self):
        bad = "Basic " + base64.b64encode(b"admin:nope").decode()
        with patch.object(self._config, "DASHBOARD_PASSWORD", "secret"):
            response = self.client().get("/next", headers={"Authorization": bad})
        self.assertEqual(response.status_code, 401)

    def test_right_password_accepted(self):
        good = "Basic " + base64.b64encode(b"admin:secret").decode()
        with patch.object(self._config, "DASHBOARD_PASSWORD", "secret"):
            response = self.client().get("/next", headers={"Authorization": good})
        self.assertEqual(response.status_code, 200)

    def test_page_is_not_cached(self):
        response = self.client().get("/next")
        self.assertEqual(response.headers.get("cache-control"), "private, no-store")


class NextPageScopeTest(TempDBTestCase):
    def get(self, url, cookies=None):
        client = self.client()
        if cookies:
            client.cookies.update(cookies)
        response = client.get(url)
        self.assertEqual(response.status_code, 200)
        return response.text

    def test_query_scope_wins_over_cookie(self):
        html = self.get("/next?scope=personal", {"tf_scope": "enterprise"})
        self.assertEqual(data_attr(html, "scope"), "personal")

    def test_cookie_scope_used_without_query(self):
        html = self.get("/next", {"tf_scope": "personal"})
        self.assertEqual(data_attr(html, "scope"), "personal")

    def test_default_scope_without_query_or_cookie(self):
        with patch("app.next_page.DEFAULT_SCOPE", "personal"):
            html = self.get("/next")
        self.assertEqual(data_attr(html, "scope"), "personal")

    def test_bad_scope_soft_fails_to_default(self):
        with patch("app.next_page.DEFAULT_SCOPE", "enterprise"):
            html = self.get("/next?scope=bogus")
        self.assertEqual(data_attr(html, "scope"), "enterprise")

    def test_lock_wins_and_is_announced(self):
        with patch.object(self._config, "LOCKED_SCOPE", "enterprise"):
            html = self.get("/next?scope=personal", {"tf_scope": "personal"})
        self.assertEqual(data_attr(html, "scope"), "enterprise")
        self.assertEqual(data_attr(html, "scope-locked"), "1")

    def test_unlocked_page_says_so(self):
        html = self.get("/next?scope=personal")
        self.assertEqual(data_attr(html, "scope-locked"), "0")

    def test_owner_is_escaped(self):
        with patch("app.next_page.STATS_OWNER", '<script>x</script>"'):
            html = self.get("/next?scope=personal")
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", data_attr(html, "owner"))

    def test_static_version_is_set(self):
        html = self.get("/next?scope=personal")
        self.assertRegex(data_attr(html, "version") or "", r"^[0-9a-f]{8,}$")
        self.assertIn("/static/next/js/main.js?v=", html)


if __name__ == "__main__":
    unittest.main()
