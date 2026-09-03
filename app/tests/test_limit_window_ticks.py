"""Every usage-limit gauge carries a whole-window projection and boundary
ticks (hour boundaries for 5-hour windows, local midnights for weekly and
monthly ones).

Two layers, mirroring test_monthly_budget_template.py:
  * source-level pins: the shared helpers exist once, sit at the scope every
    gauge can reach, and every gauge (Claude weekly / 5-hour / per-model,
    Codex + OpenCode windows) routes through them;
  * functional: the pure tick helper and the window-pace helper are lifted
    out of the template and executed under node against fixed instants.
"""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "templates" / "dashboard.html"

HOUR_MS = 3600 * 1000
DAY_MS = 24 * HOUR_MS


def _extract_function(src, name):
    """Return the full source of top-level-ish `function name(...) {...}`
    by brace matching (string/comment aware enough for this file)."""
    start = src.index("function " + name + "(")
    i = src.index("{", start)
    depth = 0
    in_str = None
    j = i
    while j < len(src):
        ch = src[j]
        if in_str:
            if ch == "\\":
                j += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch
        elif src.startswith("//", j):
            j = src.index("\n", j)
            continue
        elif src.startswith("/*", j):
            j = src.index("*/", j) + 2
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start:j + 1]
        j += 1
    raise AssertionError("unbalanced braces extracting " + name)


class LimitWindowTicksSourceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tpl = TEMPLATE.read_text()

    def test_tick_helper_defined_once_at_top_level(self):
        self.assertEqual(self.tpl.count("function tfWindowTicks("), 1)
        # Top-level (next to the tz helper it depends on), not buried in a
        # render closure, so tests and future panels can reach it.
        self.assertGreater(self.tpl.index("function tfWindowTicks("),
                           self.tpl.index("function tfChicagoOffsetMin("))
        self.assertLess(self.tpl.index("function tfWindowTicks("),
                        self.tpl.index("function initRateLimits()"))

    def test_gauge_helpers_hoisted_to_shared_scope(self):
        """buildGauge / calcExpectedPct / windowPace must sit in
        renderAllRateLimits scope (next to barColor) so the reported-provider
        builder can use them, not inside the Claude-only OAuth closure."""
        for name in ("function calcExpectedPct(", "function windowPace(",
                     "function buildGauge("):
            self.assertEqual(self.tpl.count(name), 1, name)
        oauth_closure = self.tpl.index(
            "/* ── Personal OAuth limit gauges (present only when wb.oauth")
        bar_color = self.tpl.index("function barColor(")
        for name in ("function calcExpectedPct(", "function windowPace(",
                     "function buildGauge("):
            pos = self.tpl.index(name)
            self.assertGreater(pos, bar_color, name)
            self.assertLess(pos, oauth_closure, name)

    def test_window_pace_is_the_single_d6_guard(self):
        """The stale-reset (D6) rule lives in windowPace, once, instead of
        being re-derived per gauge."""
        block = _extract_function(self.tpl, "windowPace")
        self.assertIn("resetMs <= Date.now()", block)
        self.assertIn("tfWindowTicks(resetMs, periodMs)", block)
        self.assertIn("calcExpectedPct(resetIso, periodMs)", block)
        # The old hand-rolled copies are gone.
        self.assertNotIn("hResetMs <= Date.now()", self.tpl)
        self.assertNotIn("sbResetMs <= Date.now()", self.tpl)
        self.assertNotIn("weeklyDayLines", self.tpl)
        self.assertNotIn("dayLines", self.tpl)

    def test_build_gauge_renders_labeled_and_bare_ticks(self):
        block = _extract_function(self.tpl, "buildGauge")
        self.assertIn("opts.ticks", block)
        self.assertIn('class="rate-gauge-dayline"', block)
        self.assertIn('class="rate-gauge-dayline-label"', block)
        self.assertIn('class="rate-gauge-tick"', block)
        # Remaining never goes negative on an over-100% reading.
        self.assertIn("Math.max(0, 100 - Math.round(pct))", block)

    def test_claude_gauges_pass_ticks(self):
        for call in ("buildGauge('Weekly Limit'", "buildGauge('5-Hour Window'"):
            pos = self.tpl.index(call)
            self.assertIn("ticks:", self.tpl[pos:pos + 400], call)
        start = self.tpl.index("var scopedBuckets = ")
        end = self.tpl.index("// Extra usage block")
        block = self.tpl[start:end]
        self.assertIn("windowPace(sb.resets_at, WEEK_MS, sbPct)", block)
        self.assertIn("ticks: sbPace.ticks", block)
        self.assertIn("projectedPct: sbPace.projected", block)

    def test_reported_provider_windows_use_shared_gauge(self):
        block = _extract_function(self.tpl, "buildReportedProviderGroups")
        self.assertIn("windowPace(", block)
        self.assertIn("buildGauge(", block)
        self.assertIn("ticks: pace.ticks", block)
        self.assertIn("projectedPct: pace.projected", block)
        # No hand-rolled bar/projection markup left in the provider loop.
        self.assertNotIn("provider-limit-projection", block)
        self.assertNotIn('<div class="rate-gauge-track">', block)
        # Never shadow the global `window` object inside the loop.
        self.assertNotIn("var window =", block)

    def test_tick_css_comment_no_longer_monthly_only(self):
        css = self.tpl[self.tpl.index(".rate-gauge-tick {") - 400:
                       self.tpl.index(".rate-gauge-tick {")]
        self.assertNotIn("monthly budget track", css)


class LimitWindowTicksNodeTest(unittest.TestCase):
    """Execute the pure helpers under node against fixed instants.

    Wall clock is America/Chicago (CDT, UTC-5, in September 2026)."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("node") is None:
            raise unittest.SkipTest("node not available on PATH")
        tpl = TEMPLATE.read_text()
        cls.prelude = "\n".join([
            _extract_function(tpl, "tfTzOffsetMin"),
            _extract_function(tpl, "tfChicagoOffsetMin"),
            _extract_function(tpl, "tfWindowTicks"),
            _extract_function(tpl, "calcExpectedPct"),
            _extract_function(tpl, "windowPace"),
        ])

    def _run(self, body):
        script = self.prelude + "\n" + body
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".js", delete=False) as f:
            f.write(script)
            path = f.name
        try:
            result = subprocess.run(
                ["node", path], capture_output=True, text=True, timeout=30)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def _ticks(self, reset_iso, period_ms):
        return self._run(
            "console.log(JSON.stringify(tfWindowTicks("
            f"new Date('{reset_iso}').getTime(), {period_ms})));")

    def test_five_hour_window_gets_hour_ticks_with_hour_labels(self):
        # Resets 05:40Z = 00:40 CDT; window opened 19:40 CDT the day before.
        ticks = self._ticks("2026-09-03T05:40:00Z", 5 * HOUR_MS)
        self.assertEqual([t["label"] for t in ticks],
                         ["8p", "9p", "10p", "11p", "12a"])
        self.assertTrue(all(t["kind"] == "hour" for t in ticks))
        # 20:00 CDT is 20 minutes into a 300-minute window.
        self.assertAlmostEqual(ticks[0]["pct"], 20 / 300 * 100, places=6)
        self.assertAlmostEqual(ticks[-1]["pct"], 260 / 300 * 100, places=6)

    def test_hour_ticks_on_the_border_are_dropped(self):
        # Reset exactly on the hour: the boundary at 100% must not render,
        # and the one at 0% (window start) must not either.
        ticks = self._ticks("2026-09-03T05:00:00Z", 5 * HOUR_MS)
        self.assertEqual([t["label"] for t in ticks],
                         ["8p", "9p", "10p", "11p"])
        self.assertTrue(all(0.5 < t["pct"] < 99.5 for t in ticks))

    def test_weekly_window_gets_midnight_ticks_with_weekday_labels(self):
        # Resets Thu 2026-09-03 03:00 CDT (08:00Z); window opened the
        # previous Thursday 03:00 CDT.
        ticks = self._ticks("2026-09-03T08:00:00Z", 7 * DAY_MS)
        # Thursday's own midnight (3h before reset, 98.2%) is inside the
        # 0.5..99.5 band, so it renders too.
        self.assertEqual([t["label"] for t in ticks],
                         ["Fri", "Sat", "Sun", "Mon", "Tue", "Wed", "Thu"])
        self.assertTrue(all(t["kind"] == "day" for t in ticks))
        # Friday midnight is 21h into a 168h window.
        self.assertAlmostEqual(ticks[0]["pct"], 21 / 168 * 100, places=6)

    def test_monthly_window_ticks_every_day_labels_mondays_only(self):
        ticks = self._ticks("2026-09-16T00:49:00Z", 30 * DAY_MS)
        # Window opened Aug 16 19:49 CDT: midnights Aug 17 .. Sep 15.
        self.assertEqual(len(ticks), 30)
        labeled = [t["label"] for t in ticks if t["label"]]
        self.assertEqual(labeled,
                         ["Aug 17", "Aug 24", "Aug 31", "Sep 7", "Sep 14"])

    def test_edge_ticks_keep_tick_but_drop_caption(self):
        # Codex weekly: window opened Sun 21:32 CDT, so Monday midnight lands
        # 148 minutes in (1.47%): the tick stays, the caption would hang off
        # the left border and is dropped.
        ticks = self._ticks("2026-09-07T02:32:00Z", 7 * DAY_MS)
        self.assertEqual(len(ticks), 7)
        self.assertLess(ticks[0]["pct"], 1.5)
        self.assertIsNone(ticks[0]["label"])
        self.assertEqual([t["label"] for t in ticks[1:]],
                         ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    def test_daily_window_thins_hour_labels(self):
        ticks = self._ticks("2026-09-03T05:00:00Z", 24 * HOUR_MS)
        self.assertEqual(len(ticks), 23)
        labeled = [t["label"] for t in ticks if t["label"]]
        self.assertEqual(labeled, ["3a", "6a", "9a", "12p", "3p", "6p", "9p"])

    def test_garbage_input_yields_no_ticks(self):
        out = self._run(
            "console.log(JSON.stringify(["
            "tfWindowTicks(NaN, 1000), tfWindowTicks(0, 0), "
            "tfWindowTicks(0, -5), tfWindowTicks('x', 1000), "
            "tfWindowTicks(Date.now(), 400 * 24 * 3600 * 1000 * 2)]));")
        self.assertEqual(out, [[], [], [], [], []])

    def test_window_pace_projection_marker_and_ticks(self):
        # now = 2 hours into a 5-hour window (expected 40%), pct 30 -> 75%.
        out = self._run(
            "var NOW = new Date('2026-09-03T02:40:00Z').getTime();"
            "Date.now = function(){ return NOW; };"
            "var p = windowPace('2026-09-03T05:40:00Z', 5 * 3600 * 1000, 30);"
            "console.log(JSON.stringify({e: p.expected, pr: p.projected,"
            " n: p.ticks.length, labels: p.ticks.map(function(t){return t.label;})}));")
        self.assertAlmostEqual(out["e"], 40.0, places=6)
        self.assertAlmostEqual(out["pr"], 75.0, places=6)
        self.assertEqual(out["n"], 5)
        self.assertEqual(out["labels"], ["8p", "9p", "10p", "11p", "12a"])

    def test_window_pace_stale_reset_returns_nothing(self):
        out = self._run(
            "var NOW = new Date('2026-09-03T06:00:00Z').getTime();"
            "Date.now = function(){ return NOW; };"
            "console.log(JSON.stringify(["
            "windowPace('2026-09-03T05:40:00Z', 5 * 3600 * 1000, 30),"
            "windowPace(null, 5 * 3600 * 1000, 30),"
            "windowPace('2026-09-03T09:00:00Z', null, 30),"
            "windowPace('junk', 1000, 30)]));")
        for entry in out:
            self.assertEqual(entry, {"expected": None, "projected": None,
                                     "ticks": []})

    def test_window_pace_no_projection_inside_deadband(self):
        # 1% into the window: expected <= 2 -> marker but no projection.
        out = self._run(
            "var NOW = new Date('2026-09-03T00:43:00Z').getTime();"
            "Date.now = function(){ return NOW; };"
            "var p = windowPace('2026-09-03T05:40:00Z', 5 * 3600 * 1000, 30);"
            "console.log(JSON.stringify({e: p.expected, pr: p.projected}));")
        self.assertAlmostEqual(out["e"], 1.0, places=6)
        self.assertIsNone(out["pr"])


if __name__ == "__main__":
    unittest.main()
