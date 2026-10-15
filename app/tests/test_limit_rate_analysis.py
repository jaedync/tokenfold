"""Exchange-rate analysis on synthetic data with known answers
(app/limit_rate_analysis.py)."""

import random
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app import limit_rate_analysis as lra

PT = ZoneInfo("America/Los_Angeles")
HOUR = 3600.0
DAY = 86400.0
# Monday 2026-09-28 00:00 PT
T0 = datetime(2026, 9, 28, tzinfo=PT).timestamp()
POLL = 120.0


def _is_peak(t):
    d = datetime.fromtimestamp(t, PT)
    return d.weekday() < 5 and 5 <= d.hour < 11


class _World:
    """Simulate requests and both meters with whole-point readings.

    usd_per_pt7 / usd_per_pt5 are the dollars per point of an Opus dollar;
    fable_weight multiplies Fable's limit use per dollar. peak multiplies the
    5-hour burn on weekdays 05:00-11:00 PT.
    """

    def __init__(self, days, *, step_at=None, step_factor=1.0, fable_weight=2.0,
                 peak=1.4, cr_weight=0.5, seed=3, grant_at=None,
                 fable_ps=(0.0, 0.1, 0.25, 0.6)):
        rng = random.Random(seed)
        self.requests = []
        self.readings = {"seven_day": [], "five_hour": []}
        self.granted = []
        acc7 = acc5 = 0.0
        w7_end = T0 + 7 * DAY
        w5_end = None
        t = T0
        end = T0 + days * DAY
        pending = []
        block = None
        while t < end:
            # Work comes in 3-hour sessions with their own model and token
            # mix, as in real use; that variation is what identifies the fit.
            if block != int(t // (3 * HOUR)):
                block = int(t // (3 * HOUR))
                fable_p = rng.choice(fable_ps)
                mix = [rng.uniform(0.3, 1.7) for _ in range(4)]
                load = rng.choice([0.2, 1.0, 2.0, 3.0])
            fam = "fable" if rng.random() < fable_p else "opus"
            busy = load * (0.5 + rng.random())
            usd_in = 0.01 * busy * mix[0]
            usd_out = 0.20 * busy * mix[1]
            usd_cw = 0.15 * busy * mix[2]
            usd_cr = 0.25 * busy * mix[3]
            t_req = t + rng.uniform(0, POLL)
            pending.append((t_req, fam, [usd_in, usd_out, usd_cw, usd_cr]))
            scale = step_factor if (step_at is not None and t_req >= step_at) else 1.0
            fam_w = fable_weight if fam == "fable" else 1.0
            # Limit weight: cache reads count at cr_weight of their price.
            eff = (usd_in + usd_out + usd_cw + cr_weight * usd_cr) * fam_w
            eff /= (0.01 + 0.20 + 0.15 + cr_weight * 0.25) / 0.61
            if grant_at is not None and t < grant_at <= t + POLL:
                acc7 = acc5 = 0.0
                self.granted.append(t + POLL)
            if t >= w7_end:
                acc7 = 0.0
                w7_end += 7 * DAY
            if w5_end is None or t >= w5_end:
                acc5 = 0.0
                w5_end = t + 5 * HOUR
            acc7 += eff / (24.0 / scale)
            acc5 += eff * (1.4 if (peak != 1.0 and _is_peak(t_req)) else 1.0) / (3.6 / scale)
            t += POLL
            self.readings["seven_day"].append((t, float(int(acc7)), w7_end))
            self.readings["five_hour"].append((t, float(int(acc5)), w5_end))
        self.requests = sorted(pending, key=lambda r: r[0])
        self.end = end

    def run(self):
        return lra.analyze(self.readings, self.requests, self.granted, now=self.end)


class SteadyRateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.world = _World(days=14)
        cls.out = cls.world.run()

    def test_usd_per_point_matches_truth_with_band(self):
        upp = self.out["usd_per_point"]
        seven = upp["seven_day"]
        # Opus dollars buy a point at $24; Fable (~24% of spend) weighs 2x.
        truth7 = 24.0 / (1 - 0.2375 + 0.2375 * 2.0)
        self.assertAlmostEqual(seven["value"], truth7, delta=truth7 * 0.05)
        self.assertLessEqual(seven["lo"], seven["value"])
        self.assertGreaterEqual(seven["hi"], seven["value"])
        self.assertIsNotNone(upp["five_hour"]["value"])

    def test_no_steps_without_change(self):
        self.assertEqual(self.out["steps"], [])
        self.assertEqual(self.out["period_start"], (self.world.readings["seven_day"][0][0] // 60) * 60)

    def test_family_weights(self):
        fams = {f["family"]: f for f in self.out["families"]}
        self.assertAlmostEqual(fams["opus"]["usd_per_point"], 24.0, delta=2.0)
        self.assertAlmostEqual(fams["fable"]["usd_per_point"], 12.0, delta=1.5)
        self.assertLess(fams["fable"]["usd_per_point"], fams["opus"]["usd_per_point"])
        self.assertGreater(fams["opus"]["spend_usd"], fams["fable"]["spend_usd"])
        q = self.out["fit_quality"]
        self.assertGreater(q["r2"], 0.8)
        self.assertGreater(q["intervals"], 50)
        self.assertEqual(q["status"], "ok")

    def test_peak_factor_and_hours(self):
        peak = self.out["peak"]
        self.assertEqual(peak["tz"], "America/Los_Angeles")
        self.assertAlmostEqual(peak["factor"], 1.4, delta=0.12)
        self.assertEqual(len(peak["hours"]), 168)
        cells = {(h["dow"], h["hour"]): h["factor"] for h in peak["hours"]}
        self.assertGreater(cells[(1, 7)], 1.2)       # Tuesday 07:00 PT
        self.assertLess(abs(cells[(1, 14)] - 1.0), 0.15)
        sunday = [cells[(6, h)] for h in range(24) if cells[(6, h)] is not None]
        self.assertLess(abs(sum(sunday) / len(sunday) - 1.0), 0.1)

    def test_deterministic(self):
        again = self.world.run()
        self.assertEqual(again["usd_per_point"], self.out["usd_per_point"])
        self.assertEqual(again["families"], self.out["families"])


class CacheReadWeightTest(unittest.TestCase):

    def test_weight_relative_to_output(self):
        # One family only: the weight fit treats other families as one
        # dollar feature, which blurs it when their token mix tracks Opus.
        w = _World(days=14, fable_ps=(0.0,)).run()["cache_read_weight"]
        self.assertEqual(w["family"], "opus")
        # Cache reads count at half their price in this world.
        self.assertAlmostEqual(w["value"], 0.5, delta=0.15)
        self.assertLessEqual(w["lo"], w["value"])
        self.assertGreaterEqual(w["hi"], w["value"])


class RateStepTest(unittest.TestCase):

    def test_rate_change_is_found_and_period_starts_after_it(self):
        step = T0 + 8 * DAY + 3 * HOUR
        out = _World(days=16, step_at=step, step_factor=0.5).run()
        seven = [s for s in out["steps"] if s["bucket"] == "seven_day"]
        self.assertEqual(len(seven), 1, out["steps"])
        s = seven[0]
        self.assertEqual(s["cause"], "rate_change")
        self.assertLess(abs(s["at"] - step), 8 * HOUR)
        self.assertGreater(s["to_usd_per_point"], 1.6 * s["from_usd_per_point"])
        self.assertEqual(out["period_start"], s["at"])
        truth = 48.0 / 1.2375
        self.assertAlmostEqual(out["usd_per_point"]["seven_day"]["value"], truth, delta=truth * 0.07)

    def test_granted_reset_without_rate_change_is_not_a_step(self):
        w = _World(days=14, grant_at=T0 + 9 * DAY + 600)
        out = w.run()
        self.assertTrue(w.granted)
        self.assertEqual(out["steps"], [])

    def test_granted_reset_with_rate_change_is_a_step(self):
        grant = T0 + 8 * DAY + 600
        out = _World(days=16, step_at=grant, step_factor=0.45, grant_at=grant).run()
        seven = [s for s in out["steps"] if s["bucket"] == "seven_day"]
        self.assertEqual([s["cause"] for s in seven], ["granted_reset"])
        self.assertLess(abs(seven[0]["at"] - grant), 300)


class ThinDataTest(unittest.TestCase):

    def test_small_family_gets_null_and_note(self):
        w = _World(days=10, fable_ps=(0.0, 0.0, 0.0, 0.01))
        out = w.run()
        fams = {f["family"]: f for f in out["families"]}
        self.assertLess(fams["fable"]["spend_usd"], 100)
        self.assertGreater(fams["fable"]["spend_usd"], 0)
        self.assertIsNone(fams["fable"]["usd_per_point"])
        self.assertIsNone(fams["fable"]["lo"])
        self.assertEqual(fams["fable"]["note"], "not enough data")
        self.assertIsNotNone(fams["opus"]["usd_per_point"])

    def test_empty_inputs_degrade_to_nulls(self):
        out = lra.analyze({"seven_day": [], "five_hour": []}, [], [], now=T0)
        self.assertIsNone(out["usd_per_point"]["seven_day"]["value"])
        self.assertIsNone(out["usd_per_point"]["five_hour"]["value"])
        self.assertEqual(out["families"], [])
        self.assertEqual(out["steps"], [])
        self.assertIsNone(out["peak"]["factor"])
        self.assertIsNone(out["cache_read_weight"]["value"])
        self.assertEqual(out["fit_quality"]["status"], "insufficient_data")
        self.assertIsNone(out["fit_quality"]["r2"])


class PeakStateTest(unittest.TestCase):

    def test_active_until_and_next_start(self):
        hours = [{"dow": d, "hour": h, "factor": 1.4 if d < 5 and 5 <= h < 11 else 1.0}
                 for d in range(7) for h in range(24)]
        peak = {"factor": 1.4, "tz": "America/Los_Angeles", "hours": hours}
        tue_7 = datetime(2026, 10, 13, 7, 30, tzinfo=PT).timestamp()
        st = lra.peak_state(peak, tue_7)
        self.assertTrue(st["active"])
        self.assertEqual(st["until"], datetime(2026, 10, 13, 11, tzinfo=PT).timestamp())
        sat = datetime(2026, 10, 17, 12, tzinfo=PT).timestamp()
        st = lra.peak_state(peak, sat)
        self.assertFalse(st["active"])
        self.assertIsNone(st["until"])
        self.assertEqual(st["next_start"], datetime(2026, 10, 19, 5, tzinfo=PT).timestamp())

    def test_no_peak_cells(self):
        st = lra.peak_state({"factor": None, "hours": []}, T0)
        self.assertEqual(st, {"active": False, "until": None, "next_start": None})


if __name__ == "__main__":
    unittest.main()
