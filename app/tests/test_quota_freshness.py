"""Quota inference is valid only at a fresh, unexpired observation."""
import json
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _iso, _ins_event


class QuotaFreshnessTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = round(time.time(), 6)
        self.observed = self.now - 600

    def seed(self, observed, reset=None):
        reset = self.now + 3600 if reset is None else reset
        stored = {"data": {k: {"utilization": 50, "resets_at": _iso(reset)}
                           for k in ("five_hour", "seven_day", "seven_day_opus")},
                  "source": "meridian-oauth", "updated_at": _iso(observed) if observed else ""}
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES('oauth_usage',?)", (json.dumps(stored),))
        self.conn.commit()

    def budget(self):
        with patch("app.api.time.time", return_value=self.now):
            return self.client().get("/api/rate-limits?scope=personal").json()["weekly_budget"]

    def test_cost_cutoff_tracks_observation_not_request(self):
        self.seed(self.observed)
        _ins_event(self.conn, "before", "r1", self.observed - 60, inp=1_000_000)
        _ins_event(self.conn, "after", "r2", self.observed + 60, inp=1_000_000)
        b = self.budget()
        self.assertEqual(b["week_cost"], 10)
        for key in ("limit_window", "five_hour_window"):
            self.assertEqual(b["oauth"][key]["cost"], 5)
            self.assertAlmostEqual(b["oauth"][key]["end_epoch"], self.observed, places=5)
            self.assertAlmostEqual(b["oauth"][key]["observed_at_epoch"], self.observed, places=5)
        scoped = b["oauth"]["buckets"][2]
        self.assertEqual(scoped["window_cost"], 5)
        self.assertAlmostEqual(scoped["window_end_epoch"], self.observed, places=5)

    def test_stale_missing_future_expired_omit_inference_not_measured_spend(self):
        _ins_event(self.conn, "recent", "r1", self.now - 60, inp=1_000_000)
        for observed, reset in ((self.now - 3601, None), (None, None),
                                (self.now + 1, None), (self.observed, self.now - 1)):
            with self.subTest(observed=observed, reset=reset):
                self.seed(observed, reset)
                b = self.budget()
                self.assertEqual(b["week_cost"], 5)
                self.assertNotIn("limit_window", b["oauth"])
                self.assertNotIn("five_hour_window", b["oauth"])
                self.assertNotIn("window_cost", b["oauth"]["buckets"][2])

    def test_exact_stale_boundary_and_window_start_gate(self):
        self.seed(self.now - 3600)
        self.assertIn("limit_window", self.budget()["oauth"])
        self.seed(self.observed, self.now + 8 * 86400)
        self.assertNotIn("limit_window", self.budget()["oauth"])

    def test_provider_estimates_share_one_hour_freshness(self):
        from app.provider_usage import _fresh_windows
        snapshot = {"observed_at_epoch": self.now - 3601, "windows": [
            {"key": "weekly", "label": "Weekly", "pct": 50,
             "resets_at_epoch": self.now + 3600, "window_seconds": 604800}]}
        with patch("app.provider_usage._window_reported_cost", return_value=10):
            stale = _fresh_windows(snapshot, "codex", self.now, self.conn, "personal", True)
            self.assertNotIn("estimated_capacity", stale[0])
            snapshot["observed_at_epoch"] = self.now - 3600
            fresh = _fresh_windows(snapshot, "codex", self.now, self.conn, "personal", True)
            self.assertEqual(fresh[0]["estimated_capacity"], 20)

    def managed_reading(self, observed, pct, reset):
        # Freshness assertions need the snapshot's own source, otherwise
        # source isolation alone would make stale suppression pass vacuously.
        from app.limit_readings import record_limit_readings
        record_limit_readings(self.conn, {"seven_day": {
            "utilization": pct, "resets_at": _iso(reset)}},
            observed, "meridian-oauth", strict=True)

    def test_stale_observation_and_expired_bucket_do_not_publish_pace(self):
        for observed, reset in ((self.now - 3601, self.now + 3600),
                                (self.observed, self.now - 1)):
            self.seed(observed, reset)
            self.managed_reading(self.now - 1200, 40, reset)
            self.managed_reading(self.now - 600, 50, reset)
            self.assertNotIn("seven_day", self.budget()["oauth"].get("trend", {}))

    def test_trend_excludes_readings_after_observation(self):
        self.seed(self.observed)
        reset = self.now + 3600
        for t, pct in ((self.observed - 1800, 10), (self.observed - 1200, 20),
                       (self.observed - 600, 30), (self.observed, 40),
                       (self.observed + 300, 99)):
            self.managed_reading(t, pct, reset)
        trend = self.budget()["oauth"]["trend"]["seven_day"]
        self.assertEqual(trend["series"][-1][1], 40)
        self.assertTrue(all(t <= self.observed for t, _ in trend["series"]))
