"""GET /install.sh — the unauthenticated one-command install bootstrap.

The served script carries ZERO secrets: the ingest token is supplied by the
operator on the curl command line, never baked in. That is precisely why the
route is unauthenticated. These tests pin that contract (no auth, no key leak),
the request-time URL baking, and the fail-loud 503 on a broken image.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import app.config
import app.install
from app.tests._support import TempDBTestCase

ROOT = Path(__file__).resolve().parents[2]  # app/tests/ -> repo root


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


class WatchModePlistTest(unittest.TestCase):
    """Part C: the opt-in launchd watch-mode template + installer wiring.

    Source-level contract (mirrors test_hook_scripts.py): the plist has the
    agreed Label / ProgramArguments / KeepAlive / RunAtLoad / log paths and
    obvious substitution placeholders, and the installer grows a --watch mode
    that path-substitutes and launchctl bootstraps it, macOS-only, without
    changing default (no-flag) behavior."""

    @classmethod
    def setUpClass(cls):
        base = ROOT / "client"
        cls.plist = (base / "com.jaedynchilton.tokenfold-watch.plist").read_text()
        cls.installer_path = base / "install-tokenfold-hook.sh"
        cls.installer = cls.installer_path.read_text()

    # ── plist template ──
    def test_plist_label(self):
        self.assertIn("<string>com.jaedynchilton.tokenfold-watch</string>", self.plist)

    def test_plist_program_arguments(self):
        self.assertIn("<string>/usr/bin/python3</string>", self.plist)
        self.assertIn("tokenfold-push.py", self.plist)
        self.assertIn("<string>--watch</string>", self.plist)

    def test_plist_keepalive_and_runatload(self):
        self.assertIn("<key>KeepAlive</key>", self.plist)
        self.assertIn("<key>RunAtLoad</key>", self.plist)

    def test_plist_logs_to_push_log(self):
        self.assertIn(".tokenfold-push.log", self.plist)
        self.assertIn("<key>StandardOutPath</key>", self.plist)
        self.assertIn("<key>StandardErrorPath</key>", self.plist)

    def test_plist_uses_substitution_placeholders(self):
        # An obvious token the installer substitutes (mirrors bootstrap.sh's
        # __TOKENFOLD_URL__ convention).
        self.assertIn("__TOKENFOLD_HOOKS_DIR__", self.plist)
        self.assertIn("__HOME__", self.plist)

    # ── installer wiring ──
    def test_installer_is_valid_bash(self):
        r = subprocess.run(["bash", "-n", str(self.installer_path)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_installer_has_watch_flag(self):
        self.assertIn("--watch", self.installer)

    def test_installer_references_plist_and_launchagents(self):
        self.assertIn("com.jaedynchilton.tokenfold-watch.plist", self.installer)
        self.assertIn("LaunchAgents", self.installer)

    def test_installer_substitutes_both_placeholders(self):
        self.assertIn("__TOKENFOLD_HOOKS_DIR__", self.installer)
        self.assertIn("__HOME__", self.installer)

    def test_installer_bootstraps_with_bootout_first(self):
        # bootout before bootstrap so a re-run is idempotent, tolerant of "not loaded".
        self.assertIn("launchctl bootstrap", self.installer)
        self.assertIn("launchctl bootout", self.installer)

    def test_installer_watch_is_macos_only(self):
        # Non-macOS must print a clear message and exit nonzero.
        self.assertIn("Darwin", self.installer)


if __name__ == "__main__":
    unittest.main()
