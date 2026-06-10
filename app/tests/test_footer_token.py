"""Footer ingest-key reveal on the dashboard.

The ingest token (STATS_API_KEY) is embedded in the dashboard HTML so an
operator can grab it after logging in (click-to-reveal in the footer).

Fail-closed gating: the token is embedded ONLY when dashboard auth is
actually enabled (DASHBOARD_PASSWORD set). An open dashboard must never
leak the machine-ingest key.
"""

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

    def test_token_is_html_escaped(self):
        """A token containing HTML metacharacters must not break out of the
        attribute it is embedded in (defense-in-depth; real keys are URL-safe)."""
        evil = 'k"><script>alert(1)</script>'
        with patch.object(app.config, "STATS_API_KEY", evil):
            html = self._get_html()
        self.assertNotIn(evil, html)            # raw form never appears
        self.assertIn("&lt;script&gt;", html)    # escaped form does


if __name__ == "__main__":
    unittest.main()
