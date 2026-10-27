"""Pure /api/runway rules: binding, waitable, verdict and the dollar math."""
import unittest

from app.runway_rules import (binding, kind_for, usd_left, usd_per_point,
                              verdict, waitable)

NOW = 1_800_000_000.0
H = 3600.0


def _w(id_, kind="long", pct=50.0, projected=None, resets_in=3 * 86400,
       reaches=None, tight=None, state="on", stale=False):
    pace = None if state is None else {
        "pace_state": state, "expected_pct": 50.0, "projected_pct": projected,
        "reaches_100_epoch": reaches, "elapsed_fraction": 0.5}
    return {"id": id_, "kind": kind, "pct": pct, "resets_at": NOW + resets_in,
            "pace": pace, "tight_epoch": tight, "stale": stale}


class BindingTest(unittest.TestCase):
    def test_highest_projected_long_window_binds(self):
        ws = [_w("a", projected=80.0), _w("b", projected=120.0, pct=30.0),
              _w("c", kind="short", projected=300.0)]
        self.assertEqual(binding(ws), "b")

    def test_short_windows_never_bind(self):
        self.assertIsNone(binding([_w("fh", kind="short", projected=150.0)]))

    def test_month_window_can_bind(self):
        self.assertEqual(binding([_w("m", kind="month", projected=90.0)]), "m")

    def test_falls_back_to_pct_without_a_projection(self):
        ws = [_w("a", pct=9.0, state="too_early"),
              _w("b", pct=30.0, state="too_early")]
        self.assertEqual(binding(ws), "b")

    def test_window_without_projection_uses_its_pct(self):
        ws = [_w("a", pct=60.0, state="too_early"), _w("b", projected=40.0)]
        self.assertEqual(binding(ws), "a")

    def test_fresh_windows_beat_stale_ones(self):
        ws = [_w("old", projected=150.0, stale=True), _w("new", projected=20.0)]
        self.assertEqual(binding(ws), "new")

    def test_all_stale_still_binds(self):
        self.assertEqual(binding([_w("old", projected=50.0, stale=True)]), "old")

    def test_missing_pct_never_binds(self):
        self.assertIsNone(binding([_w("x", pct=None, state=None)]))

    def test_empty(self):
        self.assertIsNone(binding([]))


class CalmBindingTest(unittest.TestCase):
    """When no long window is at risk, the hero stays on the main limit."""

    def test_calm_binds_claude_seven_day(self):
        # Production copy, 2026-10-15: 3.7% into the Claude week (too
        # early), OpenCode Go monthly projected 24.4%.
        ws = [_w("claude:seven_day", pct=9.0, state="too_early"),
              _w("opencode-go:monthly", pct=24.0, projected=24.4,
                 state="below"),
              _w("claude:five_hour", kind="short", projected=99.0)]
        self.assertEqual(binding(ws), "claude:seven_day")

    def test_calm_work_scope_binds_the_month_meter(self):
        ws = [_w("claude:month", kind="month", projected=50.0),
              _w("codex:primary", projected=70.0)]
        self.assertEqual(binding(ws), "claude:month")

    def test_projected_90_is_at_risk(self):
        ws = [_w("claude:seven_day", projected=40.0),
              _w("opencode-go:monthly", projected=90.0)]
        self.assertEqual(binding(ws), "opencode-go:monthly")

    def test_tight_window_is_at_risk(self):
        ws = [_w("claude:seven_day", projected=60.0),
              _w("codex:primary", projected=70.0, tight=NOW + H)]
        self.assertEqual(binding(ws), "codex:primary")

    def test_runs_out_window_is_at_risk(self):
        ws = [_w("claude:seven_day", projected=40.0),
              _w("codex:primary", pct=30.0, projected=70.0,
                 reaches=NOW + H, state="above")]
        self.assertEqual(binding(ws), "codex:primary")

    def test_short_window_risk_does_not_count(self):
        ws = [_w("claude:seven_day", pct=5.0, projected=20.0),
              _w("x:monthly", projected=60.0),
              _w("claude:five_hour", kind="short", reaches=NOW + 60,
                 state="above", projected=300.0)]
        self.assertEqual(binding(ws), "claude:seven_day")

    def test_window_that_fills_first_binds(self):
        # Production, 2026-10-26: Codex weekly 16% early in its window
        # projects 267% but fills in 53 h; Claude 7-day (84%, 124%) fills
        # in 22 h. The window that stops work first is the hero.
        ws = [_w("claude:seven_day", pct=84.0, projected=124.4,
                 reaches=NOW + 22 * H, resets_in=55 * H, state="above"),
              _w("codex:secondary", pct=16.0, projected=267.4,
                 reaches=NOW + 53 * H, resets_in=158 * H, state="above")]
        self.assertEqual(binding(ws), "claude:seven_day")

    def test_tight_time_counts_as_the_fill_time(self):
        ws = [_w("a", projected=150.0, reaches=NOW + 30 * H, state="above"),
              _w("b", projected=80.0, tight=NOW + 5 * H)]
        self.assertEqual(binding(ws), "b")

    def test_earlier_of_pace_and_tight_time(self):
        ws = [_w("a", projected=150.0, reaches=NOW + 30 * H,
                 tight=NOW + 2 * H, state="above"),
              _w("b", projected=200.0, reaches=NOW + 10 * H, state="above")]
        self.assertEqual(binding(ws), "a")

    def test_a_window_that_fills_beats_one_that_only_projects_high(self):
        ws = [_w("a", projected=99.0),
              _w("b", projected=95.0, reaches=NOW + 40 * H, state="above")]
        self.assertEqual(binding(ws), "b")

    def test_fill_after_the_reset_does_not_count(self):
        ws = [_w("a", projected=99.0),
              _w("b", projected=95.0, reaches=NOW + 5 * 86400,
                 resets_in=86400, state="above")]
        self.assertEqual(binding(ws), "a")

    def test_stale_window_that_fills_first_loses_to_a_fresh_one(self):
        ws = [_w("old", projected=150.0, reaches=NOW + H, stale=True,
                 state="above"),
              _w("new", projected=120.0, reaches=NOW + 9 * H, state="above")]
        self.assertEqual(binding(ws), "new")

    def test_a_waitable_fill_ranks_after_one_that_stops_work_longer(self):
        # a fills first but resets 30 min later (runway.py sets waitable
        # before binding); b stops work for days.
        ws = [{**_w("a", projected=130.0, reaches=NOW + 22 * H,
                    resets_in=22.5 * H, state="above"), "waitable": True},
              _w("b", projected=120.0, reaches=NOW + 30 * H,
                 resets_in=5 * 86400, state="above")]
        self.assertEqual(binding(ws), "b")

    def test_a_window_full_now_keeps_the_hero_even_when_waitable(self):
        # Work is stopped now; the reset in 40 min does not make it calm.
        ws = [{**_w("a", pct=100.0, projected=130.0, reaches=NOW - H,
                    resets_in=40 * 60, state="above"), "waitable": True},
              _w("b", projected=120.0, reaches=NOW + 3 * 86400,
                 resets_in=5 * 86400, state="above")]
        self.assertEqual(binding(ws, now=NOW), "a")

    def test_only_waitable_fills_still_bind_by_fill_time(self):
        ws = [{**_w("a", projected=130.0, reaches=NOW + 30 * H,
                    resets_in=30.5 * H, state="above"), "waitable": True},
              {**_w("b", projected=120.0, reaches=NOW + 20 * H,
                    resets_in=20.5 * H, state="above"), "waitable": True}]
        self.assertEqual(binding(ws), "b")

    def test_calm_without_a_default_uses_the_rule(self):
        ws = [_w("codex:primary", projected=30.0),
              _w("opencode-go:weekly", projected=50.0)]
        self.assertEqual(binding(ws), "opencode-go:weekly")


class WaitableTest(unittest.TestCase):
    def test_short_window_resetting_within_an_hour(self):
        self.assertTrue(waitable(_w("fh", kind="short", resets_in=59 * 60), NOW))
        self.assertTrue(waitable(_w("fh", kind="short", resets_in=3600), NOW))
        self.assertFalse(waitable(_w("fh", kind="short", resets_in=61 * 60), NOW))

    def test_long_window_that_fills_close_to_its_reset(self):
        # Hits 100% 40 min before the reset: waiting it out is cheap.
        w = _w("sd", resets_in=10 * H, reaches=NOW + 10 * H - 40 * 60,
               state="above")
        self.assertTrue(waitable(w, NOW))

    def test_long_window_that_fills_long_before_reset(self):
        w = _w("sd", resets_in=10 * H, reaches=NOW + 2 * H, state="above")
        self.assertFalse(waitable(w, NOW))

    def test_long_window_uses_tight_time_when_on_pace(self):
        w = _w("sd", resets_in=10 * H, tight=NOW + 10 * H - 30 * 60)
        self.assertTrue(waitable(w, NOW))

    def test_long_window_never_filling_is_not_waitable(self):
        self.assertFalse(waitable(_w("sd"), NOW))

    def test_missing_reset(self):
        w = {**_w("fh", kind="short"), "resets_at": None}
        self.assertFalse(waitable(w, NOW))


class VerdictTest(unittest.TestCase):
    def test_runs_out_when_100_comes_before_reset(self):
        w = _w("a", reaches=NOW + H, state="above", tight=NOW + 2 * H)
        self.assertEqual(verdict(w), "runs_out")

    def test_reaches_after_reset_is_not_runs_out(self):
        w = _w("a", reaches=NOW + 5 * 86400, resets_in=86400, state="above")
        self.assertEqual(verdict(w), "clear")

    def test_tight(self):
        self.assertEqual(verdict(_w("a", tight=NOW + H)), "tight")

    def test_too_early(self):
        self.assertEqual(verdict(_w("a", state="too_early")), "too_early")

    def test_clear(self):
        self.assertEqual(verdict(_w("a", state="below")), "clear")

    def test_unknown_without_pace(self):
        self.assertIsNone(verdict(_w("a", state=None)))


class DollarsTest(unittest.TestCase):
    def test_usd_per_point(self):
        self.assertAlmostEqual(usd_per_point(100.0, 40.0), 2.5)

    def test_low_pct_is_noise(self):
        self.assertIsNone(usd_per_point(10.0, 4.9))
        self.assertIsNone(usd_per_point(None, 40.0))
        self.assertIsNone(usd_per_point(0.0, 40.0))

    def test_usd_left(self):
        self.assertAlmostEqual(usd_left(2.5, 40.0), 150.0)
        self.assertEqual(usd_left(2.5, 120.0), 0.0)
        self.assertIsNone(usd_left(None, 40.0))


class KindTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(kind_for(5 * 3600), "short")
        self.assertEqual(kind_for(86400), "short")
        self.assertEqual(kind_for(7 * 86400), "long")
        self.assertEqual(kind_for(30 * 86400), "long")
        self.assertEqual(kind_for(None), "long")


if __name__ == "__main__":
    unittest.main()
