"""GET /api/runway/trajectory and GET /api/window-history."""
from unittest.mock import patch

import app.config
from app.tests.test_bucket_windows import _ins_event, _ins_pi_event
from app.tests.test_ha_windows import _HABase, _ins_reading, _minute

WEEK_S = 7 * 86400
FIVE_H_S = 5 * 3600
DAY = 86400


class TrajectoryTest(_HABase):
    def _get(self, window="claude:seven_day", scope="personal", status=200):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/runway/trajectory",
                                  params={"scope": scope, "window": window})
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    def test_current_window_points_and_bounds(self):
        self._seed_oauth(sd_pct=30.0)
        start = _minute(self.sd_resets) - WEEK_S
        # The previous window's last reading (old anchor) stays out.
        _ins_reading(self.conn, "seven_day", start - 600, 90.0, start)
        _ins_reading(self.conn, "seven_day", start + 3600, 5.0, self.sd_resets)
        _ins_reading(self.conn, "seven_day", self.now - 60, 30.0, self.sd_resets)
        body = self._get()
        self.assertEqual(set(body), {"points", "resets", "start", "end", "window"})
        # The answer names its window: two windows can share bounds.
        self.assertEqual(body["window"], "claude:seven_day")
        self.assertEqual(body["start"], start)
        self.assertEqual(body["end"], _minute(self.sd_resets))
        self.assertEqual(body["points"], [[_minute(start + 3600), 5.0],
                                          [_minute(self.now - 60), 30.0]])

    def test_points_start_at_a_granted_reset(self):
        grant_t = self.now - DAY
        self._seed_oauth(sd_pct=4.0)
        self._grant("seven_day", grant_t, self.sd_resets, before=97.0, after=1.0)
        body = self._get()
        self.assertEqual(body["start"], _minute(grant_t))
        self.assertEqual(body["points"][0], [_minute(grant_t), 1.0])

    def test_five_hour_boundaries_inside_the_window(self):
        self._seed_oauth()
        a1 = self.now - 2 * DAY
        _ins_reading(self.conn, "five_hour", a1 - 3600, 40.0, a1)
        _ins_reading(self.conn, "five_hour", a1 + 1800, 3.0, a1 + 1800 + FIVE_H_S)
        body = self._get()
        self.assertEqual(body["resets"], [_minute(a1)])
        # The 5 h window itself carries no inner resets.
        self.assertEqual(self._get("claude:five_hour")["resets"], [])

    def test_downsampled_to_400_points(self):
        self._seed_oauth()
        start = _minute(self.sd_resets) - WEEK_S
        step = (self.now - start - 120) / 1000
        for i in range(1000):
            _ins_reading(self.conn, "seven_day", start + 60 + i * step,
                         i / 20, self.sd_resets)
        pts = self._get()["points"]
        self.assertLessEqual(len(pts), 400)
        self.assertEqual(pts[0][1], 0.0)
        self.assertEqual(pts[-1][1], 999 / 20)

    def test_provider_window_has_bounds_only(self):
        import json
        snap = {"provider": "codex", "windows": [{
            "key": "primary", "label": "x", "pct": 10.0,
            "resets_at_epoch": self.now + 3600, "window_seconds": FIVE_H_S}],
            "observed_at_epoch": self.now - 60, "machine": "m"}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('provider_usage', ?)",
            (json.dumps({"scopes": {"personal": {"codex": snap}}}),))
        self.conn.commit()
        body = self._get("codex:primary")
        self.assertEqual(body["points"], [])
        self.assertEqual(body["end"], _minute(self.now + 3600))
        self.assertEqual(body["start"], _minute(self.now + 3600) - FIVE_H_S)

    def test_errors(self):
        self._seed_oauth()
        self._get("claude:seven_day", scope="enterprise", status=404)
        self._get("claude:nope", status=404)
        self._get("bad id", status=400)
        self._get("codex:primary", status=404)


class WindowHistoryTest(_HABase):
    def _get(self, bucket="seven_day", scope="personal", status=200):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/window-history",
                                  params={"scope": scope, "bucket": bucket})
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    def _seed(self):
        a1 = _minute(self.now - 10 * DAY)
        a2, a3 = a1 + WEEK_S, a1 + 2 * WEEK_S
        for t, pct in ((a1 - 6 * DAY, 10.0), (a1 - DAY, 100.0)):
            _ins_reading(self.conn, "seven_day", t, pct, a1)
        for t, pct in ((a2 - 6 * DAY, 5.0), (a2 - 4 * DAY, 80.0),
                       (a2 - 3 * DAY, 0.0), (a2 - 2 * DAY, 0.0),
                       (a2 - DAY, 30.0)):
            _ins_reading(self.conn, "seven_day", t, pct, a2)
        _ins_reading(self.conn, "seven_day", self.now - 3600, 2.0, a3)  # ongoing
        _ins_event(self.conn, "w1", "r1", a1 - 2 * DAY, inp=1_000_000)
        _ins_event(self.conn, "w2", "r2", a2 - 5 * DAY, inp=1_000_000)
        _ins_event(self.conn, "w3", "r3", a2 - 12 * 3600, inp=1_000_000)
        # Not Claude-subscription spend.
        _ins_pi_event(self.conn, "w4", "r4", a2 - 12 * 3600, "openai-codex",
                      "gpt-5", 50.0)
        return a1, a2

    def test_past_windows(self):
        a1, a2 = self._seed()
        windows = self._get()["windows"]
        self.assertEqual(len(windows), 2)  # the ongoing window is left out
        w1, w2 = windows
        self.assertEqual(set(w1), {"start", "end", "peak_pct", "hit_limit",
                                   "granted_resets", "usd", "usd_per_point"})
        self.assertEqual((w1["start"], w1["end"]), (a1 - WEEK_S, a1))
        self.assertEqual(w1["peak_pct"], 100.0)
        self.assertTrue(w1["hit_limit"])
        self.assertEqual(w1["granted_resets"], [])
        self.assertAlmostEqual(w1["usd"], 5.0, places=2)
        self.assertAlmostEqual(w1["usd_per_point"], 0.05, places=4)

        self.assertEqual((w2["start"], w2["end"]), (a1, a2))
        self.assertEqual(w2["peak_pct"], 80.0)
        self.assertFalse(w2["hit_limit"])
        self.assertEqual(w2["granted_resets"], [_minute(a2 - 3 * DAY)])
        self.assertAlmostEqual(w2["usd"], 10.0, places=2)
        # A grant restarts the meter: 80 points before it, 30 after.
        self.assertAlmostEqual(w2["usd_per_point"], 10.0 / 110, places=4)

    def test_stale_replay_is_not_a_window(self):
        a1, a2 = self._seed()
        # An old snapshot (previous anchor) pushed late inside window 2.
        _ins_reading(self.conn, "seven_day", a2 - 2 * DAY + 60, 100.0, a1)
        self.assertEqual(len(self._get()["windows"]), 2)

    def test_scope_and_bucket_checks(self):
        self._get(scope="enterprise", status=404)
        self._get(bucket="BAD!", status=400)
        self.assertEqual(self._get(bucket="five_hour")["windows"], [])


class HistoryDegradeTest(_HABase):
    def test_trajectory_failure_is_empty_not_500(self):
        self._seed_oauth()
        with patch.object(app.config, "LOCKED_SCOPE", None), \
                patch("app.runway.claude_trajectory",
                      side_effect=RuntimeError("boom")):
            r = self.client().get("/api/runway/trajectory", params={
                "scope": "personal", "window": "claude:seven_day"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"points": [], "resets": [],
                                    "start": None, "end": None,
                                    "window": "claude:seven_day"})

    def test_window_history_failure_is_empty_not_500(self):
        with patch.object(app.config, "LOCKED_SCOPE", None), \
                patch("app.runway.window_history",
                      side_effect=RuntimeError("boom")):
            r = self.client().get("/api/window-history",
                                  params={"scope": "personal"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"windows": []})


class MonthTrajectoryTest(_HABase):
    """claude:month (work scope): the billed line from the billing meter."""

    def setUp(self):
        super().setUp()
        from app.monthly_budget import _month_bounds_utc
        self.m_start, self.m_end, self.m_label, _ = _month_bounds_utc(self.now)
        # Readings spread over the part of the month that has passed.
        self.span = self.now - self.m_start

    def _get(self, status=200):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/runway/trajectory", params={
                "scope": "enterprise", "window": "claude:month"})
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    def _meter(self, t, used_usd, limit_usd=2000.0):
        self.conn.execute(
            "INSERT INTO extra_usage_readings(fetched_epoch, machine, used_cents,"
            " limit_cents, utilization) VALUES(?,?,?,?,?)",
            (t, "m", used_usd * 100, limit_usd * 100, None))
        self.conn.commit()

    def _at(self, frac):
        return self.m_start + 30 + frac * (self.span - 60)

    def test_meter_readings_become_percent_of_budget(self):
        self._meter(self.m_start - 3600, 1900.0)  # last month: left out
        self._meter(self._at(0.2), 100.0)
        self._meter(self._at(0.6), 500.0)
        self._meter(self._at(1.0), 1000.0)
        body = self._get()
        self.assertEqual(body["start"], self.m_start)
        self.assertEqual(body["end"], self.m_end)
        self.assertEqual(body["points"], [
            [_minute(self._at(0.2)), 5.0],
            [_minute(self._at(0.6)), 25.0],
            [_minute(self._at(1.0)), 50.0]])
        self.assertEqual(body["resets"], [])

    def test_rollover_inside_the_month_drops_the_old_cycle(self):
        # The meter rolled over a little after the UTC month started.
        self._meter(self._at(0.05), 1950.0)
        self._meter(self._at(0.1), 2.0)
        self._meter(self._at(0.9), 400.0)
        pts = self._get()["points"]
        self.assertEqual([p[1] for p in pts], [0.1, 20.0])

    def test_hand_entered_billing_readings_when_no_meter(self):
        from app.monthly_budget import set_budget
        set_budget(self.conn, 1000.0)
        self.conn.execute(
            "INSERT INTO billing_readings(scope, amount_usd, month, recorded_at,"
            " recorded_epoch) VALUES('enterprise', ?, ?, 'x', ?)",
            (250.0, self.m_label, self._at(0.5)))
        self.conn.execute(
            "INSERT INTO billing_readings(scope, amount_usd, month, recorded_at,"
            " recorded_epoch) VALUES('personal', 900.0, ?, 'x', ?)",
            (self.m_label, self._at(0.6)))
        self.conn.commit()
        self.assertEqual(self._get()["points"], [[_minute(self._at(0.5)), 25.0]])

    def test_no_readings_is_bounds_only(self):
        from app.monthly_budget import set_budget
        set_budget(self.conn, 1000.0)
        body = self._get()
        self.assertEqual(body["points"], [])
        self.assertEqual((body["start"], body["end"]), (self.m_start, self.m_end))

    def test_downsampled_to_400_points(self):
        for i in range(900):
            self._meter(self._at(i / 899), float(i + 1))
        pts = self._get()["points"]
        self.assertLessEqual(len(pts), 400)
        self.assertEqual(pts[-1][1], 45.0)
