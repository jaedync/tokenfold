"""One pace rule for every limit gauge. Cases mirror pi-extras'
tests/usage-pace.test.ts so Tokenfold and the Pi `usage` tool cannot disagree."""

import unittest

from app.usage_pace import FIVE_HOUR_S, WEEK_S, tight_epoch, usage_pace

NOW = 1790921400.0          # 2026-10-02T06:10:00Z
RESET = NOW + 3 * 3600      # 09:10:00Z


def pace(used, now=NOW, reset=RESET, window=FIVE_HOUR_S):
    return usage_pace(used, reset, window, now)


class UsagePaceTest(unittest.TestCase):

    def test_projection_and_time_to_100(self):
        p = pace(60)
        self.assertEqual(p["elapsed_fraction"], 0.4)
        self.assertEqual(p["expected_pct"], 40.0)
        self.assertEqual(p["projected_pct"], 150.0)
        self.assertEqual(p["pace_state"], "above")
        # 07:30:00Z = now + 80 min
        self.assertEqual(p["reaches_100_epoch"], NOW + 4800)
        self.assertEqual(p["reaches_100_in_s"], 4800)

    def test_on_pace_band_is_inclusive_90_to_110(self):
        now = RESET - 9000
        for pct, state in [(44.9, "below"), (45, "on"), (50, "on"),
                           (55, "on"), (55.1, "above")]:
            p = pace(pct, now=now)
            self.assertEqual(p["pace_state"], state, pct)
            self.assertEqual(p["reaches_100_epoch"] is not None,
                             state == "above", pct)

    def test_early_or_tiny_usage_gives_no_projection(self):
        for p in (pace(3, now=1790914303.0),          # 04:11:43Z, <5% elapsed
                  pace(60, now=RESET - 18000),
                  pace(0), pace(0.9)):
            self.assertEqual(p["pace_state"], "too_early")
            self.assertIsNone(p["projected_pct"])
            self.assertIsNone(p["reaches_100_epoch"])
        self.assertNotEqual(pace(1, now=RESET - 0.95 * 18000)["pace_state"],
                            "too_early")

    def test_inapplicable_windows_return_none(self):
        bad = [
            dict(reset=None), dict(reset=NOW), dict(reset=NOW - 1),
            dict(reset=NOW + 18000 + 1), dict(reset=float("inf")),
            dict(window=0), dict(window=float("nan")), dict(window=None),
        ]
        for kw in bad:
            self.assertIsNone(pace(50, **kw), kw)
        for used in (None, float("nan"), -1, True, "50"):
            self.assertIsNone(pace(used), used)

    def test_full_and_overage_never_wait_negative_time(self):
        for used in (100, 120):
            p = pace(used)
            self.assertEqual(p["pace_state"], "above")
            self.assertEqual(p["reaches_100_in_s"], 0)
            self.assertLessEqual(p["reaches_100_epoch"], NOW)

    def test_weekly_window_same_rule(self):
        p = usage_pace(60, NOW + 0.5 * WEEK_S, WEEK_S, NOW)
        self.assertEqual(p["elapsed_fraction"], 0.5)
        self.assertEqual(p["projected_pct"], 120.0)
        self.assertEqual(p["pace_state"], "above")


class TightTest(unittest.TestCase):

    def test_tight_only_when_average_is_not_above_and_burn_hits_100_early(self):
        eta = RESET - 600
        self.assertEqual(tight_epoch("on", eta, RESET), eta)
        self.assertEqual(tight_epoch("below", eta, RESET), eta)
        self.assertIsNone(tight_epoch("above", eta, RESET))
        self.assertIsNone(tight_epoch("too_early", eta, RESET))
        self.assertIsNone(tight_epoch("on", RESET + 60, RESET))
        self.assertIsNone(tight_epoch("on", None, RESET))
        self.assertIsNone(tight_epoch(None, eta, RESET))


if __name__ == "__main__":
    unittest.main()


class GrantedResetPaceTest(unittest.TestCase):
    """After a granted mid-window reset the window effectively restarts at the
    grant: 100% must last from the grant to the unchanged resets_at."""

    RESET = NOW + 86400            # one day left in a 7-day window

    def test_granted_start_shortens_the_window(self):
        granted = NOW - 2 * 86400  # grant two days ago -> 3-day effective window
        p = usage_pace(70, self.RESET, WEEK_S, NOW, window_start_epoch=granted)
        self.assertAlmostEqual(p["elapsed_fraction"], 2 / 3)
        self.assertEqual(p["expected_pct"], 66.7)
        self.assertEqual(p["projected_pct"], 105.0)
        self.assertEqual(p["pace_state"], "on")
        # The nominal window would call the same usage "below" (81.7%).
        self.assertEqual(usage_pace(70, self.RESET, WEEK_S, NOW)["pace_state"], "below")

    def test_reaches_100_counts_from_the_grant(self):
        granted = NOW - 86400      # 2-day effective window, half elapsed
        p = usage_pace(80, self.RESET, WEEK_S, NOW, window_start_epoch=granted)
        self.assertEqual(p["pace_state"], "above")
        # 80% in one day -> 100% at grant + 1.25 days
        self.assertEqual(p["reaches_100_epoch"], ((granted + 1.25 * 86400) // 60) * 60)

    def test_start_outside_the_window_falls_back_to_nominal(self):
        nominal = usage_pace(70, self.RESET, WEEK_S, NOW)
        for start in (self.RESET - WEEK_S - 3600,   # before the nominal start
                      NOW + 60,                     # in the future
                      None, float("nan"), "x"):
            self.assertEqual(usage_pace(70, self.RESET, WEEK_S, NOW,
                                        window_start_epoch=start), nominal, start)

    def test_fresh_grant_is_too_early(self):
        p = usage_pace(2, self.RESET, WEEK_S, NOW, window_start_epoch=NOW - 600)
        self.assertEqual(p["pace_state"], "too_early")

