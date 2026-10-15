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
        self.assertEqual(set(body), {"points", "resets", "start", "end"})
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
