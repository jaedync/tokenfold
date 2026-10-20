"""Days when the work billing meter rose much more than the recorded spend."""
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.meter_gap import meter_gaps

TZ = ZoneInfo("America/Chicago")


def at(day, hour=0, minute=0):
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=TZ).timestamp()


def daily(**days):
    return [{"day": d.lstrip("_").replace("_", "-"), "by_model": {("Anthropic", "claude-opus-5-5"): usd,
                                                      ("OpenRouter", "glm"): 50.0}}
            for d, usd in days.items()]


class MeterGapTest(unittest.TestCase):
    def test_reports_a_day_the_meter_billed_without_sessions(self):
        readings = [(at("2026-10-16", 23), 1399.0), (at("2026-10-19", 9), 1402.0),
                    (at("2026-10-19", 23, 57), 1753.0), (at("2026-10-20", 8), 1870.0)]
        gaps = meter_gaps(readings, daily(_2026_10_19=126.0, _2026_10_20=15.0, _2026_10_18=0.0),
                          TZ, now=at("2026-10-20", 9))
        self.assertEqual(gaps, [
            {"day": "2026-10-19", "billed": 354.0, "recorded": 126.0},
            {"day": "2026-10-20", "billed": 117.0, "recorded": 15.0},
        ])

    def test_small_or_proportional_differences_are_not_gaps(self):
        readings = [(at("2026-10-14", 23), 1220.0), (at("2026-10-15", 23), 1305.0),
                    (at("2026-10-16", 23), 1325.0)]
        # +6% on a big day and +$19 on a small day: lag and rounding, not missing usage.
        gaps = meter_gaps(readings, daily(_2026_10_15=80.0, _2026_10_16=1.0), TZ,
                          now=at("2026-10-17", 1))
        self.assertEqual(gaps, [])

    def test_days_without_a_reading_on_both_sides_or_with_a_rollover_are_skipped(self):
        readings = [(at("2026-10-31", 22), 2900.0), (at("2026-11-01", 3), 5.0),
                    (at("2026-11-01", 23), 400.0)]
        gaps = meter_gaps(readings, daily(_2026_10_30=0.0, _2026_10_31=0.0, _2026_11_01=10.0),
                          TZ, now=at("2026-11-02", 1))
        self.assertEqual(gaps, [])

    def test_no_readings_no_gaps(self):
        self.assertEqual(meter_gaps([], daily(_2026_10_19=0.0), TZ, now=at("2026-10-20")), [])
        self.assertEqual(meter_gaps([(at("2026-10-19", 1), 1.0)], None, TZ, now=at("2026-10-20")), [])


if __name__ == "__main__":
    unittest.main()
