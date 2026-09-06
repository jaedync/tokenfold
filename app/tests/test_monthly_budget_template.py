"""Template-level regression tests for the enterprise monthly $ budget card.

These pin the client-side contract added in Task 2:
  * the "Monthly usage limit" gauge card + inline budget edit + empty-state
    affordance live in the template and reuse the shared gauge helpers;
  * personal scope renders NOTHING new (the global invariant);
  * enterprise renders the affordance / card wiring.

Style mirrors test_dashboard_template.py (source regex) plus a rendered-HTML
pass for the personal-vs-enterprise invariant.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.tests._support import TempDBTestCase

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "dashboard.html"


class MonthlyBudgetTemplateSourceTest(unittest.TestCase):
    """Source-level: the feature is present and reuses shared machinery."""

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_monthly_card_label_present(self):
        self.assertIn("Monthly usage limit", self.tpl)

    def test_empty_state_affordance_present(self):
        self.assertIn("Set monthly budget…", self.tpl)

    def test_projection_line_wording(self):
        self.assertIn("Projected by month end:", self.tpl)
        self.assertIn("(window avg)", self.tpl)

    def test_verdict_wording_matches_personal_side(self):
        # Exact strings, including the .proj-warn over-pace treatment.
        self.assertIn('<strong class="proj-warn">over pace</strong>', self.tpl)
        self.assertIn("'on'", self.tpl)  # pace === 'on' branch
        # 'under pace' / 'on pace' literals reused verbatim.
        self.assertIn("under pace", self.tpl)
        self.assertIn("on pace", self.tpl)

    def test_reuses_shared_barColor_no_duplicate(self):
        # barColor defined exactly ONCE (hoisted, shared by oauth + monthly).
        self.assertEqual(self.tpl.count("function barColor("), 1)

    def test_reuses_shared_marker_overlap_pass_no_duplicate(self):
        # The overlap pass is defined once and shared across both cards.
        self.assertEqual(self.tpl.count("function runMarkerOverlapPass("), 1)

    def test_monthly_card_uses_gauge_marker_markup(self):
        # The card must emit the same marker class the overlap pass hides.
        self.assertIn("buildMonthlyBudgetSection", self.tpl)
        self.assertIn("rate-gauge-marker-label", self.tpl)

    def test_post_endpoint_wired(self):
        self.assertIn("'/api/enterprise-budget'", self.tpl)

    def test_commit_refetches_rate_limits(self):
        # On success the card re-runs the section's single rate-limits poll
        # (reused, not a second poller URL) which re-renders the card.
        self.assertIn("function commitBudget(", self.tpl)
        idx = self.tpl.index("function commitBudget(")
        body = self.tpl[idx:idx + 700]
        self.assertIn("pollLimits(true);", body)
        # The shared poll itself feeds renderAllRateLimits.
        self.assertIn("paint(nextState, remembered.memory)", self.tpl)
        self.assertIn("renderAllRateLimits(state, { memory: memory", self.tpl)

    def test_edit_supports_escape_cancel_and_clear(self):
        self.assertIn("wireMonthlyBudgetEdit", self.tpl)
        self.assertIn("'Escape'", self.tpl)
        # empty input -> clears the budget (null POST)
        self.assertIn("commitBudget(null)", self.tpl)

    def test_detail_cells_labels(self):
        self.assertIn("spent · MTD", self.tpl)
        self.assertIn("budget left", self.tpl)

    def test_blur_cancels_not_commits(self):
        # Final-review fix: blur must CANCEL the in-progress edit (restore the
        # display), not commit it. Enter (via commit()) remains the only path
        # that saves — including empty+Enter, which clears the budget.
        self.assertIn("wireMonthlyBudgetEdit", self.tpl)
        idx = self.tpl.index("function wireMonthlyBudgetEdit")
        body = self.tpl[idx:idx + 3200]
        self.assertIn("input.onblur = cancel;", body)
        self.assertNotIn("input.onblur = commit;", body)
        # Enter still commits.
        self.assertIn("commit();", body)

    def test_poll_skips_render_during_open_edit(self):
        # Final-review fix: the 60s poll must not blow away an open
        # .mb-edit-input via innerHTML replace.
        self.assertIn("querySelector('.mb-edit-input')", self.tpl)


class MonthlyBudgetScriptSyntaxTest(unittest.TestCase):
    """Extract the inline <script> block housing the monthly-budget JS and
    verify it is syntactically valid, catching typos node can't otherwise
    see (this JS ships server-rendered, not through a JS test framework)."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("node") is None:
            raise unittest.SkipTest("node not available on PATH")
        tpl = TEMPLATE.read_text()
        scripts = re.findall(r"<script>(.*?)</script>", tpl, re.S)
        # The big inline app script is the one housing wireMonthlyBudgetEdit /
        # pollLimits — pick it by content rather than a hardcoded index so a
        # future extra <script> block doesn't silently break this test.
        candidates = [s for s in scripts if "wireMonthlyBudgetEdit" in s]
        assert candidates, "could not locate the monthly-budget inline script"
        cls.script = candidates[0]
        # Stub the handful of Jinja expressions so node can parse this as
        # plain JS (the template is never executed here, only parsed).
        stubbed = cls.script
        stubbed = stubbed.replace("{{ scope }}", "personal")
        stubbed = stubbed.replace(
            "{{ 'true' if scope_locked else 'false' }}", "false")
        stubbed = stubbed.replace(
            "{{ 'true' if readings_writable else 'false' }}", "false")
        assert "{{" not in stubbed, "unstubbed Jinja expression remains: " + stubbed
        cls.stubbed = stubbed

    def test_script_parses_with_node(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False) as f:
            f.write(self.stubbed)
            path = f.name
        try:
            result = subprocess.run(
                ["node", "--check", path],
                capture_output=True, text=True, timeout=30)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(
            result.returncode, 0,
            f"node --check failed:\n{result.stderr}")


class MonthlyBudgetRenderInvariantTest(TempDBTestCase):
    """Rendered-HTML: personal renders nothing; enterprise renders the card
    scaffolding. Auth on so readings_writable / BR_WRITABLE is true."""

    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self._saved_pw = self._config.DASHBOARD_PASSWORD
        self._saved_user = self._config.DASHBOARD_USER
        self._config.DASHBOARD_PASSWORD = "pw"
        self._config.DASHBOARD_USER = "jaedyn"
        self.addCleanup(self._restore_auth)

    def _restore_auth(self):
        self._config.DASHBOARD_PASSWORD = self._saved_pw
        self._config.DASHBOARD_USER = self._saved_user

    def _get(self, scope):
        c = self.client()
        return c.get(f"/?scope={scope}", auth=("jaedyn", "pw"))

    def test_personal_scope_has_no_monthly_affordance(self):
        """Personal must render zero trace of the monthly feature affordance.

        The JS helpers ship in the shared <script> for both scopes (one
        template), but the runtime gate (TF_SCOPE === 'enterprise') means the
        affordance never activates. We assert the server-rendered scope marker
        is personal so the client gate is inert."""
        r = self._get("personal")
        self.assertEqual(r.status_code, 200)
        self.assertIn("const TF_SCOPE = 'personal'", r.text)

    def test_enterprise_scope_marker_and_writable(self):
        r = self._get("enterprise")
        self.assertEqual(r.status_code, 200)
        self.assertIn("const TF_SCOPE = 'enterprise'", r.text)
        # Auth on -> BR_WRITABLE true -> edit affordance can activate.
        self.assertIn("const BR_WRITABLE = true", r.text)


if __name__ == "__main__":
    unittest.main()
