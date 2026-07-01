"""Footer ingest-key reveal on the dashboard.

The ingest token (STATS_API_KEY) is embedded in the dashboard HTML so an
operator can grab it after logging in (click-to-reveal in the footer).

Fail-closed gating: the token is embedded ONLY when dashboard auth is
actually enabled (DASHBOARD_PASSWORD set). An open dashboard must never
leak the machine-ingest key.
"""

import shlex
import unittest
from unittest.mock import patch

import app.config
from app.tests._support import TempDBTestCase


class FooterTokenRevealTest(TempDBTestCase):

    def _get_html(self, password="secret"):
        with patch.object(app.config, "DASHBOARD_USER", "admin"), \
             patch.object(app.config, "DASHBOARD_PASSWORD", password):
            c = self.client()
            if password:
                r = c.get("/", auth=("admin", password))
            else:
                r = c.get("/")
        self.assertEqual(r.status_code, 200)
        return r.text

    def test_token_embedded_when_auth_enabled(self):
        """Logged-in dashboard exposes the ingest key for client onboarding."""
        html = self._get_html()
        self.assertIn(self.api_key, html)
        self.assertIn("ingestKeyBtn", html)  # the reveal UI element

    def test_token_absent_when_dashboard_open(self):
        """No DASHBOARD_PASSWORD => open dashboard => key must NOT be served."""
        html = self._get_html(password="")
        self.assertNotIn(self.api_key, html)
        self.assertNotIn("ingestKeyBtn", html)

    def test_no_reveal_ui_when_key_unset(self):
        """Empty STATS_API_KEY => nothing to reveal, no dangling UI."""
        with patch.object(app.config, "STATS_API_KEY", ""):
            html = self._get_html()
        self.assertNotIn("ingestKeyBtn", html)

    def test_footer_observer_uses_unclipped_sentinel(self):
        """Source-level regression: the footer self-hides via clip-path until
        .in-view, but IntersectionObserver intersects the CLIPPED rect (0x0),
        so observing the footer directly deadlocks at ratio 0 and the footer
        (and the ingest-key button in it) stays invisible forever. The template
        must observe an unclipped sentinel instead of the footer itself."""
        from pathlib import Path
        tpl = (Path(__file__).resolve().parents[2] / "templates" / "dashboard.html").read_text()
        self.assertIn("footerSentinel", tpl)
        self.assertNotIn("obs.observe(footer)", tpl)

    def test_token_is_html_escaped(self):
        """A token containing HTML metacharacters must not break out of the
        attribute it is embedded in (defense-in-depth; real keys are URL-safe)."""
        evil = 'k"><script>alert(1)</script>'
        with patch.object(app.config, "STATS_API_KEY", evil):
            html = self._get_html()
        self.assertNotIn(evil, html)            # raw form never appears
        self.assertIn("&lt;script&gt;", html)    # escaped form does

    # ── one-command install button (sits next to the ingest key) ──────────

    def test_install_command_rendered_when_auth_enabled(self):
        """Logged-in dashboard exposes the curl-pipe-bash one-liner, with the
        real ingest key inline and the server's own base URL baked in."""
        from app.dashboard import build_install_command
        html = self._get_html()
        self.assertIn("installCmdBtn", html)
        self.assertIn("data-install-cmd", html)
        # TestClient reaches us as http://testserver, so external_base_url()
        # bakes exactly that host into the copied command.
        self.assertIn(build_install_command("http://testserver", self.api_key), html)

    def test_install_command_absent_when_dashboard_open(self):
        """No DASHBOARD_PASSWORD => open dashboard => the command (which inlines
        the ingest key) must NOT be served."""
        html = self._get_html(password="")
        self.assertNotIn("installCmdBtn", html)
        self.assertNotIn("/install.sh | bash", html)

    def test_install_command_absent_when_key_unset(self):
        """Empty STATS_API_KEY => no key to inline => no install button."""
        with patch.object(app.config, "STATS_API_KEY", ""):
            html = self._get_html()
        self.assertNotIn("installCmdBtn", html)


class BuildInstallCommandTest(unittest.TestCase):
    """The pure command builder is shell-injection safe by construction."""

    def test_builds_curl_pipe_bash_oneliner(self):
        from app.dashboard import build_install_command
        self.assertEqual(
            build_install_command("https://usage.example.com", "tk_abc"),
            "curl -fsSL https://usage.example.com/install.sh "
            "| bash -s -- --token tk_abc",
        )

    def test_quotes_key_with_single_quote(self):
        """A key with shell metacharacters must be shlex-quoted so it can't
        break out of the --token argument when pasted into a terminal."""
        from app.dashboard import build_install_command
        key = "a'b; rm -rf /"
        cmd = build_install_command("https://x", key)
        self.assertIn(shlex.quote(key), cmd)
        self.assertNotIn("--token a'b", cmd)  # never a bare, shell-breaking token


if __name__ == "__main__":
    unittest.main()
