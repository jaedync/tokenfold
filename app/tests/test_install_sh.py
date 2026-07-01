"""GET /install.sh — the unauthenticated one-command install bootstrap.

The served script carries ZERO secrets: the ingest token is supplied by the
operator on the curl command line, never baked in. That is precisely why the
route is unauthenticated. These tests pin that contract (no auth, no key leak),
the request-time URL baking, and the fail-loud 503 on a broken image.
"""

import unittest
from pathlib import Path
from unittest.mock import patch

import app.config
import app.install
from app.tests._support import TempDBTestCase


class InstallScriptTest(TempDBTestCase):

    def test_serves_script_with_baked_host(self):
        """200, real bootstrap body, placeholder gone, baked URL = request host.
        TestClient's default base_url is http://testserver."""
        with self.client() as c:
            r = c.get("/install.sh")
        self.assertEqual(r.status_code, 200)
        self.assertIn("codeload.github.com/jaedync/tokenfold", r.text)
        self.assertNotIn("__TOKENFOLD_URL__", r.text)
        self.assertIn('TOKENFOLD_URL_DEFAULT="http://testserver"', r.text)

    def test_honors_forwarded_proto(self):
        """Behind Caddy (uvicorn without --proxy-headers) the real scheme comes
        from X-Forwarded-Proto, not request.url.scheme."""
        with self.client() as c:
            r = c.get("/install.sh", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(r.status_code, 200)
        self.assertIn('TOKENFOLD_URL_DEFAULT="https://testserver"', r.text)

    def test_forwarded_proto_uses_first_value(self):
        """A proxy chain may append protos ('https, http'); the client-facing
        scheme is the FIRST hop, so only that one is baked."""
        with self.client() as c:
            r = c.get("/install.sh", headers={"X-Forwarded-Proto": "https, http"})
        self.assertEqual(r.status_code, 200)
        self.assertIn('TOKENFOLD_URL_DEFAULT="https://testserver"', r.text)

    def test_content_type_and_no_store(self):
        """Shell MIME so `curl | bash` treats it as a script; no-store so an
        intermediary can't cache a stale baked URL."""
        with self.client() as c:
            r = c.get("/install.sh")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/x-shellscript", r.headers["content-type"])
        self.assertEqual(r.headers.get("cache-control"), "no-store")

    def test_body_never_contains_api_key(self):
        """Unauthenticated exactly because it holds no secret — the ingest key
        (STATS_API_KEY) must never appear in the served body."""
        # _support sets STATS_API_KEY = self.api_key ("test-key") in setUp.
        with self.client() as c:
            r = c.get("/install.sh")
        self.assertEqual(r.status_code, 200)
        self.assertNotIn(self.api_key, r.text)

    def test_no_auth_required_even_when_dashboard_locked(self):
        """A fresh machine has no credentials yet; the route has no auth
        dependency, so it serves 200 even with dashboard Basic auth enabled."""
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", "secret"):
            with self.client() as c:
                r = c.get("/install.sh")  # deliberately no auth= credentials
        self.assertEqual(r.status_code, 200)
        self.assertNotEqual(r.status_code, 401)

    def test_503_when_bootstrap_missing(self):
        """A missing bootstrap.sh is a broken image (server fault) → 503, never
        a silent 200 with an unusable body."""
        missing = Path(self.db_path).with_name("definitely-not-here.sh")
        with patch.object(app.install, "BOOTSTRAP_PATH", missing):
            with self.client() as c:
                r = c.get("/install.sh")
        self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
