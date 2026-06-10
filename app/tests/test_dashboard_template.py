"""Source-level regression tests for the dashboard template.

The dashboard is a single server-rendered template with inline JS; these
assertions pin the perf/robustness contracts from the UX overhaul:

* Chart.js is self-hosted (no CDN single point of failure) and deferred,
  with all chart code guarded so tables/heatmaps survive a missing library.
* Pollers and listeners are created once, not once-per-render (leaks).
* Charts update in place (no destroy/recreate animation replay on refresh).
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "dashboard.html"


class DashboardTemplateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    # ── Chart.js self-hosting (P1-1) ──────────────────────────────────────

    def test_chartjs_is_self_hosted(self):
        """No CDN <script> for chart.js; the bundle ships from /static."""
        self.assertNotIn("cdn.jsdelivr.net", self.tpl)
        self.assertIn('defer src="/static/chart.umd.min.js"', self.tpl)

    def test_chartjs_bundle_exists(self):
        bundle = ROOT / "static" / "chart.umd.min.js"
        self.assertTrue(bundle.is_file(), "static/chart.umd.min.js missing")
        self.assertGreater(bundle.stat().st_size, 100_000)
        self.assertIn("Chart.js", bundle.read_text()[:200])

    def test_chart_usage_is_guarded(self):
        """DOM-only renders must survive a failed chart.js load."""
        self.assertIn("typeof Chart !== 'undefined'", self.tpl)
        self.assertIn("function initChartGlobals()", self.tpl)

    # ── Interval / listener leaks (P1-2) ──────────────────────────────────

    def test_rate_limits_poller_outside_render_mode(self):
        """The /api/rate-limits fetch + 60s poll lives in initRateLimits
        (run once at boot), NOT in renderMode (run on every refresh)."""
        self.assertIn("function initRateLimits()", self.tpl)
        self.assertEqual(self.tpl.count("'/api/rate-limits?scope='"), 1)
        render_start = self.tpl.index("function renderMode(")
        init_start = self.tpl.index("function initRateLimits()")
        fetch_pos = self.tpl.index("'/api/rate-limits?scope='")
        self.assertGreater(init_start, render_start,
                           "initRateLimits should be defined after renderMode")
        self.assertGreater(fetch_pos, init_start,
                           "rate-limits fetch must live inside initRateLimits")

    def test_single_global_keydown_handler(self):
        """One Escape handler for all modals — was one new listener per render."""
        self.assertEqual(self.tpl.count("document.addEventListener('keydown'"), 1)

    def test_env_cycler_interval_is_cleared(self):
        self.assertIn("_envCycleTimer", self.tpl)
        self.assertIn("clearInterval(_envCycleTimer)", self.tpl)

    def test_heatmap_tip_listeners_attached_once(self):
        """Tooltip listeners are delegated in initTips, not re-attached inside
        renderHeatmap/renderHourly on every innerHTML swap."""
        self.assertIn("function initTips()", self.tpl)
        for fn in ("renderHeatmap", "renderHourly"):
            start = self.tpl.index("function %s()" % fn)
            end = self.tpl.index("\n}", start)
            self.assertNotIn("addEventListener", self.tpl[start:end],
                             "%s must not attach listeners per render" % fn)

    # ── Chart animations / update-in-place (P1-3) ─────────────────────────

    def test_chart_animations_disabled(self):
        self.assertIn("Chart.defaults.animation = false", self.tpl)

    def test_charts_update_in_place(self):
        """upsertChart registry replaces the old destroy-everything loop."""
        self.assertIn("function upsertChart(", self.tpl)
        self.assertIn("ex.update('none')", self.tpl)
        self.assertNotIn("_tCharts", self.tpl)


if __name__ == "__main__":
    unittest.main()
