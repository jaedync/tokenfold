"""Days when the work billing meter rose much more than the recorded spend."""
import sqlite3
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.meter_gap import local_days, meter_days, meter_gaps, read_meter

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

    def test_days_with_fewer_than_two_readings_in_reach_are_skipped(self):
        readings = [(at("2026-10-31", 22), 2900.0), (at("2026-11-01", 23), 3000.0)]
        days = meter_days(readings, ["2026-10-30", "2026-10-31", "2026-11-01"], TZ,
                          at("2026-11-02", 1), spend())
        self.assertEqual([d["day"] for d in days], ["2026-11-01"])

    def test_a_rollover_counts_the_new_cycle_from_zero(self):
        # The meter restarts at zero: $5 then $395 more, $400 on Nov 1 (a DST day).
        readings = [(at("2026-10-31", 22), 2900.0), (at("2026-11-01", 3), 5.0),
                    (at("2026-11-01", 23), 400.0)]
        cost = spend((at("2026-11-01", 12), 400.0))
        days = meter_days(readings, ["2026-11-01"], TZ, at("2026-11-02", 1), cost)
        self.assertEqual(days, [{"day": "2026-11-01", "billed": 400.0, "recorded": 400.0}])
        self.assertEqual(meter_gaps(readings, ["2026-11-01"], TZ, at("2026-11-02", 1), cost), [])

    def test_out_of_order_readings_inside_a_cycle_still_count(self):
        # Two work machines can push readings a little out of order.
        readings = [(at("2026-10-14", 23), 100.0), (at("2026-10-15", 9), 160.0),
                    (at("2026-10-15", 9, 1), 159.5), (at("2026-10-15", 22), 300.0)]
        gaps = meter_gaps(readings, ["2026-10-15"], TZ, at("2026-10-16"), spend())
        self.assertEqual(gaps, [{"day": "2026-10-15", "billed": 200.0, "recorded": 0}])

    def test_a_small_fall_across_a_month_start_is_a_rollover(self):
        # A long gap with no readings: the cycle ended at $30, the new one is at $20.
        readings = [(at("2026-10-31", 12), 30.0), (at("2026-11-01", 12), 20.0)]
        days = meter_days(readings, ["2026-11-01"], TZ, at("2026-11-02"), spend())
        self.assertEqual(days[0]["billed"], 20.0)

    def test_local_days_cover_the_range_in_the_local_zone(self):
        self.assertEqual(local_days(at("2026-10-30", 23), at("2026-11-02", 0, 30), TZ),
                         ["2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"])

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
