"""GET / (and its alias /next): the dashboard shell.

Covers dashboard auth, the scope precedence shared with GET /classic (lock >
?scope= > tf_scope cookie > default, soft-fail on a bad scope), escaping of
the owner name, the no-store header on the authenticated page, and the swap
that moved the old dashboard to /classic.
"""
import base64
import re
import unittest
from pathlib import Path
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

    def test_page_sets_a_content_security_policy(self):
        # Same-origin scripts, styles, fonts and fetches only; no inline
        # style or script. The browser checks run with this header on.
        response = self.client().get("/next")
        self.assertEqual(response.headers.get("content-security-policy"),
                         "default-src 'self'; style-src 'self'; img-src 'self' data:")


class HomeSwapTest(TempDBTestCase):
    """/ serves the new dashboard; the old one moved to /classic; /next stays
    as an alias so bookmarks keep working."""

    def test_root_serves_the_new_dashboard(self):
        response = self.client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-view-link="runway"', response.text)
        self.assertEqual(response.headers.get("cache-control"), "private, no-store")
        self.assertEqual(response.headers.get("content-security-policy"),
                         "default-src 'self'; style-src 'self'; img-src 'self' data:")

    def test_next_is_an_alias_of_root(self):
        response = self.client().get("/next")
        self.assertEqual(response.status_code, 200)
        self.assertIn('data-view-link="runway"', response.text)

    def test_classic_serves_the_old_dashboard(self):
        response = self.client().get("/classic")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Human Prompts", response.text)
        self.assertNotIn('data-view-link="runway"', response.text)
        self.assertEqual(response.headers.get("cache-control"), "private, no-store")

    def test_classic_requires_auth_when_password_set(self):
        with patch.object(self._config, "DASHBOARD_PASSWORD", "secret"):
            for path in ("/", "/classic"):
                response = self.client().get(path)
                self.assertEqual(response.status_code, 401, path)

    def test_new_page_links_to_classic_with_its_scope(self):
        html = self.client().get("/?scope=enterprise").text
        self.assertIn('href="/classic?scope=enterprise"', html)
        self.assertNotRegex(html, r'href="/(\?[^"]*)?"')


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
        version = data_attr(html, "version") or ""
        self.assertRegex(version, r"^[0-9a-f]{8,}$")
        self.assertEqual(data_attr(html, "static"), f"/static/next-{version}")


class NextStaticVersionTest(TempDBTestCase):
    """Cache busting: every module and asset URL carries the content hash in
    its path, so a nested ES module import can never be stale after a deploy.
    """

    def setUp(self):
        super().setUp()
        self.http = self.client()
        html = self.http.get("/next?scope=personal").text
        self.html = html
        self.base = data_attr(html, "static")

    def test_every_shell_asset_uses_the_versioned_path(self):
        urls = re.findall(r'(?:href|src)="(/static/next[^"]*)"', self.html)
        self.assertGreaterEqual(len(urls), 5)
        for url in urls:
            self.assertTrue(url.startswith(self.base + "/"), url)
            self.assertNotIn("?v=", url)

    def test_module_and_nested_module_are_served_immutable(self):
        root = Path(__file__).resolve().parents[2] / "static" / "next"
        for rel in ("js/main.js", "js/panels/hero.js", "css/tokenfold.css"):
            r = self.http.get(f"{self.base}/{rel}")
            self.assertEqual(r.status_code, 200, rel)
            self.assertEqual(r.content, (root / rel).read_bytes())
            self.assertIn("immutable", r.headers.get("cache-control", ""))
        self.assertIn("javascript", self.http.get(f"{self.base}/js/main.js").headers["content-type"])

    def test_head_request(self):
        r = self.http.head(f"{self.base}/js/main.js")
        self.assertEqual(r.status_code, 200)

    def test_older_version_still_loads_but_is_not_kept(self):
        r = self.http.get("/static/next-0123456789/js/main.js")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("cache-control"), "no-cache")

    def test_missing_file_bad_version_and_traversal_are_404(self):
        for url in (f"{self.base}/js/nope.js",
                    "/static/next-XYZ!/js/main.js",
                    f"{self.base}/../../app/config.py",
                    f"{self.base}/%2e%2e/%2e%2e/app/config.py",
                    f"{self.base}/js/..%2f..%2f..%2fapp%2fconfig.py"):
            r = self.http.get(url)
            self.assertEqual(r.status_code, 404, url)
            self.assertNotIn(b"STATS_API_KEY", r.content)

    def test_version_follows_content_not_mtime(self):
        import os
        import tempfile
        from app.next_page import _static_version
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "js").mkdir()
            f = folder / "js" / "a.js"
            f.write_text("export const a = 1;")
            v1 = _static_version(folder)
            os.utime(f, (1, 1))
            self.assertEqual(_static_version(folder), v1)
            f.write_text("export const a = 2;")
            os.utime(f, (1, 1))
            self.assertNotEqual(_static_version(folder), v1)


if __name__ == "__main__":
    unittest.main()
