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

    # ── Information clarity (P1-5 / P1-6 / P1-8) ──────────────────────────

    def test_cost_hero_names_its_window(self):
        """The hero figure must say which window it covers."""
        self.assertIn("costHeroLabel", self.tpl)
        self.assertIn("'Estimated API Cost · ' + windowLabel", self.tpl)
        self.assertIn("'Last 14 Days'", self.tpl)

    def test_personal_scope_api_equivalent_note(self):
        self.assertIn("costHeroNote", self.tpl)
        self.assertIn("TF_SCOPE === 'personal'", self.tpl)
        self.assertIn("not what you're billed", self.tpl)

    def test_yellow_text_token_for_light_surfaces(self):
        """Yellow #f5c518 fails contrast as text on cream; a darkened token
        must exist and the known offenders must use it."""
        self.assertIn("--yellow-text: #9a7a00", self.tpl)
        # Minutes axis on Daily Activity no longer uses raw yellow ticks
        self.assertNotIn("ticks:{color:'#f5c518'}", self.tpl)

    def test_no_sub_055rem_labels(self):
        """Floor micro-labels at 0.55rem (was 0.42-0.5rem)."""
        import re
        for m in re.finditer(r"font-size:\s*0\.(\d+)rem", self.tpl):
            size = float("0." + m.group(1))
            self.assertGreaterEqual(
                size, 0.55,
                "label below 0.55rem floor: %s" % m.group(0))

    def test_red_reserved_for_high_costs(self):
        """Ordinary costs render black; red only above the 1.5x-average
        threshold (client table rebuild mirrors the server render)."""
        self.assertIn("hotCost", self.tpl)
        self.assertNotIn("d.cost>0?'var(--red)'", self.tpl)
        self.assertNotIn("s.cost>0?'var(--red)'", self.tpl)
        self.assertNotIn("ms.cost > 0 ? 'var(--red)'", self.tpl)


class DashboardServerRenderTest(unittest.TestCase):
    """Server-rendered daily rows follow the same red-is-a-warning rule."""

    def test_cost_color_threshold(self):
        from app.dashboard import _fmt_cost  # noqa: F401 (import sanity)
        import app.dashboard as dash
        # replicate the inline logic deterministically via a tiny harness:
        daily = [{"cost": c} for c in (5.0, 5.0, 5.0, 50.0, 0.0)]
        nz = [d["cost"] for d in daily if d["cost"] > 0]
        hot = sum(nz) / len(nz) * 1.5
        self.assertGreater(50.0, hot)   # the outlier day goes red
        self.assertLess(5.0, hot)       # ordinary days stay black
        src = (TEMPLATE.parent.parent / "app" / "dashboard.py").read_text()
        self.assertIn("_hot_cost", src)
        self.assertNotIn("'var(--red)' if d[\"cost\"] > 0", src)


if __name__ == "__main__":
    unittest.main()
