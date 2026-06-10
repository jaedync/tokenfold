"""Tests for the site-wide scope toggle UI (T2).

Verifies that:
1. Unlocked instance shows the scope toggle with both buttons; enterprise is active by default.
2. /?scope=personal marks the personal button active and renders PERSONAL badge.
3. Locked instance hides the toggle entirely but keeps the badge; scope=personal soft-fails.
4. JS persistence wiring (TF_LOCKED, tf_scope localStorage guard) is present in the page.
"""

import re
import unittest
from unittest.mock import patch

import app.config as cfg
from app.tests._support import TempDBTestCase


def _active_scope(html):
    """Return the data-scope value of the button that carries scope-btn--active.

    Searches for a <button ...> that contains BOTH 'scope-btn--active' and a
    data-scope="..." attribute.  Returns the scope string or None if not found.
    """
    # Match any <button ...> tag (no newlines that span across buttons)
    for m in re.finditer(r'<button\b([^>]*?)>', html, re.DOTALL):
        attrs = m.group(1)
        if "scope-btn--active" in attrs:
            ds = re.search(r'data-scope="([^"]+)"', attrs)
            if ds:
                return ds.group(1)
    return None


class ScopeToggleUnlockedTest(TempDBTestCase):
    """Toggle visible when LOCKED_SCOPE is None."""

    def test_unlocked_shows_toggle(self):
        """Default / (unlocked) renders scope-toggle with both scope buttons."""
        with patch.object(cfg, "LOCKED_SCOPE", None):
            c = self.client()
            html = c.get("/").text
        self.assertIn('class="scope-toggle"', html)
        self.assertIn('data-scope="enterprise"', html)
        self.assertIn('data-scope="personal"', html)
        self.assertIn("setScope(", html)

    def test_unlocked_enterprise_button_active_by_default(self):
        """Default scope is enterprise — enterprise button carries scope-btn--active."""
        with patch.object(cfg, "LOCKED_SCOPE", None):
            c = self.client()
            html = c.get("/").text
        active = _active_scope(html)
        self.assertEqual(active, "enterprise",
                         f"Expected enterprise button to be active, got: {active!r}")

    def test_personal_scope_marks_personal_button_active(self):
        """/?scope=personal marks personal button scope-btn--active and shows PERSONAL badge."""
        with patch.object(cfg, "LOCKED_SCOPE", None):
            c = self.client()
            html = c.get("/?scope=personal").text
        self.assertIn("PERSONAL", html)
        active = _active_scope(html)
        self.assertEqual(active, "personal",
                         f"Expected personal button to be active, got: {active!r}")


class ScopeToggleLockedTest(TempDBTestCase):
    """Toggle hidden when LOCKED_SCOPE is set."""

    def test_locked_hides_toggle_shows_badge(self):
        """Locked instance: toggle absent, but ENTERPRISE badge present."""
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            html = c.get("/").text
        self.assertNotIn('class="scope-toggle"', html)
        self.assertIn("ENTERPRISE", html)

    def test_locked_ignores_personal_param_no_toggle(self):
        """Locked instance: /?scope=personal soft-fails to ENTERPRISE; toggle still absent."""
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            c = self.client()
            resp = c.get("/?scope=personal")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("ENTERPRISE", resp.text)
        self.assertNotIn('class="scope-toggle"', resp.text)


class ScopeToggleJSWiringTest(TempDBTestCase):
    """JS persistence guards are present in the rendered page."""

    def test_tf_locked_constant_present(self):
        """TF_LOCKED constant is emitted in the page."""
        with patch.object(cfg, "LOCKED_SCOPE", None):
            c = self.client()
            html = c.get("/").text
        self.assertIn("TF_LOCKED", html)

    def test_localstorage_guard_present(self):
        """The tf_scope localStorage key string is present (persistence wiring shipped)."""
        with patch.object(cfg, "LOCKED_SCOPE", None):
            c = self.client()
            html = c.get("/").text
        self.assertIn("tf_scope", html)
        self.assertIn("localStorage.getItem", html)


if __name__ == "__main__":
    unittest.main()


class AutoRefreshScopePinnedTest(unittest.TestCase):
    """Source-level regression for live-update scope bleed: the 30s version-poll
    refetch MUST request the page's own scope. A bare fetch('/api/stats') falls
    back to the server's DEFAULT_SCOPE (personal on ms01) and repaints an
    enterprise view with personal data until manual reload. The footer refresh
    must also not clobber the ingest-key button by rewriting .footer-right."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        cls.tpl = (Path(__file__).resolve().parents[2]
                   / "templates" / "dashboard.html").read_text()

    def test_stats_refetch_pins_scope(self):
        self.assertIn("'/api/stats?scope='", self.tpl)
        self.assertNotIn("fetch('/api/stats')", self.tpl)

    def test_footer_timestamp_update_preserves_ingest_key(self):
        self.assertNotIn("footerRight.textContent", self.tpl)
