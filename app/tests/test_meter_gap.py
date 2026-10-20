"""Days when the work billing meter rose much more than the recorded spend."""
import sqlite3
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.meter_gap import meter_gaps, read_meter

TZ = ZoneInfo("America/Chicago")


def at(day, hour=0, minute=0):
    y, m, d = map(int, day.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=TZ).timestamp()


def spend(*events):
    """cost(t0, t1) over recorded calls [(epoch, usd)], like compute_window_cost."""
    return lambda t0, t1: sum(usd for t, usd in events if t0 <= t < t1)


class MeterGapTest(unittest.TestCase):
    def test_reports_a_day_the_meter_billed_without_sessions(self):
        readings = [(at("2026-10-16", 23), 1399.0), (at("2026-10-19", 9), 1402.0),
                    (at("2026-10-19", 23, 57), 1753.0), (at("2026-10-20", 8), 1870.0)]
        cost = spend((at("2026-10-17", 1), 3.0), (at("2026-10-19", 12), 123.0),
                     (at("2026-10-20", 7), 15.0))
        gaps = meter_gaps(readings, ["2026-10-18", "2026-10-19", "2026-10-20"], TZ,
                          at("2026-10-20", 9), cost)
        self.assertEqual(gaps, [
            {"day": "2026-10-19", "billed": 354.0, "recorded": 126.0},
            {"day": "2026-10-20", "billed": 117.0, "recorded": 15.0},
        ])

    def test_spend_the_meter_shows_after_midnight_is_not_a_gap(self):
        # A heavy session at 23:30 reaches the meter at 00:30: the bill and
        # the recorded spend cover the same window, so the day balances.
        readings = [(at("2026-10-14", 23), 1000.0), (at("2026-10-15", 0, 30), 1200.0),
                    (at("2026-10-15", 20), 1210.0)]
        cost = spend((at("2026-10-14", 23, 30), 200.0), (at("2026-10-15", 12), 10.0))
        self.assertEqual(meter_gaps(readings, ["2026-10-15"], TZ, at("2026-10-16"), cost), [])

    def test_small_or_proportional_differences_are_not_gaps(self):
        readings = [(at("2026-10-14", 23), 1220.0), (at("2026-10-15", 23), 1305.0),
                    (at("2026-10-16", 23), 1325.0)]
        # +6% on a big day and +$19 on a small day: lag and rounding, not missing usage.
        cost = spend((at("2026-10-15", 12), 80.0), (at("2026-10-16", 12), 1.0))
        self.assertEqual(meter_gaps(readings, ["2026-10-15", "2026-10-16"], TZ,
                                    at("2026-10-17", 1), cost), [])

    def test_days_without_a_reading_before_them_or_across_a_rollover_are_skipped(self):
        readings = [(at("2026-10-31", 22), 2900.0), (at("2026-11-01", 3), 5.0),
                    (at("2026-11-01", 23), 400.0)]
        gaps = meter_gaps(readings, ["2026-10-30", "2026-10-31", "2026-11-01"], TZ,
                          at("2026-11-02", 1), spend())
        self.assertEqual(gaps, [])

    def test_out_of_order_readings_inside_a_cycle_still_count(self):
        # Two work machines can push readings a little out of order.
        readings = [(at("2026-10-14", 23), 100.0), (at("2026-10-15", 9), 160.0),
                    (at("2026-10-15", 9, 1), 159.5), (at("2026-10-15", 22), 300.0)]
        gaps = meter_gaps(readings, ["2026-10-15"], TZ, at("2026-10-16"), spend())
        self.assertEqual(gaps, [{"day": "2026-10-15", "billed": 200.0, "recorded": 0}])

    def test_no_readings_no_gaps(self):
        self.assertEqual(meter_gaps([], ["2026-10-19"], TZ, at("2026-10-20"), spend()), [])
        self.assertEqual(meter_gaps([(at("2026-10-19", 1), 1.0)], [], TZ, at("2026-10-20"),
                                    spend()), [])


class ReadMeterTest(unittest.TestCase):
    def test_keeps_the_last_reading_before_a_long_quiet_spell(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE extra_usage_readings(fetched_epoch REAL, used_cents REAL)")
        rows = [(at("2026-09-01"), 100), (at("2026-09-02"), 500), (at("2026-10-15", 9), 900)]
        conn.executemany("INSERT INTO extra_usage_readings VALUES(?, ?)", rows)
        self.assertEqual(read_meter(conn, at("2026-10-15"), at("2026-10-16")),
                         [(at("2026-09-02"), 5.0), (at("2026-10-15", 9), 9.0)])


if __name__ == "__main__":
    unittest.main()
