"""Mixed-source production regression, with real SQLite and HTTP boundaries."""
import json
import time
from unittest.mock import patch

from app.claude_usage import MANAGED_SOURCE
from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _ins_event, _iso

WEEK = 7 * 86400


class QuotaHistorySourceTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = (time.time() // 60) * 60
        self.observed = self.now - 30
        self.reset = self.now + 3600
        self.snapshot()
        _ins_event(self.conn, "before", "r1", self.observed - 600, inp=1_000_000)
        _ins_event(self.conn, "after", "r2", self.observed - 60, inp=1_000_000)

    def snapshot(self, source=MANAGED_SOURCE, pct=14, weekly=True):
        keys = ("five_hour", "seven_day", "seven_day_opus") if weekly else ("five_hour",)
        stored = {"source": source, "observed_at_epoch": self.observed,
                  "updated_at": _iso(self.observed),
                  "data": {key: {"utilization": pct, "resets_at": _iso(self.reset)}
                           for key in keys}}
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES('oauth_usage',?)",
                          (json.dumps(stored),))
        self.conn.commit()

    def row(self, delta, pct, source=MANAGED_SOURCE, bucket="seven_day", reset=None):
        reset = self.reset if reset is None else reset
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch,source,bucket,utilization,"
            "resets_at,resets_at_epoch) VALUES(?,?,?,?,?,?)",
            (self.observed + delta, source, bucket, pct, _iso(reset), reset))
        self.conn.commit()

    def budget(self):
        from app.api import _build_rate_limits
        with patch("app.api.time.time", return_value=self.now):
            response = _build_rate_limits("personal", self.conn)
            return json.loads(response.body)["weekly_budget"]

    def history(self, bucket="seven_day"):
        with patch("app.limit_readings.time.time", return_value=self.now):
            response = self.client().get("/api/limit-history", params={
                "scope": "personal", "bucket": bucket, "hours": 168})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def spend(self):
        from app.spend_history import _spend_history
        with patch("app.spend_history.time.time", return_value=self.now):
            return _spend_history(self.conn, "personal", None)

    def test_live_legacy_straddler_does_not_create_negative_burn(self):
        self.row(-190000, 16, "client")
        self.row(0, 14)
        trend = self.budget()["oauth"]["trend"]["seven_day"]
        self.assertIsNone(trend["burn_6h_pct_per_hr"])
        self.assertIsNone(trend["pace"])
        self.assertEqual(trend["series"], [[(self.observed // 60) * 60, 14]])

    def test_legacy_same_epoch_cannot_replace_managed_series_point(self):
        self.row(-1800, 10)
        self.row(0, 14)
        expected = self.budget()["oauth"]["trend"]
        self.row(0, 90, "client")
        self.assertEqual(self.budget()["oauth"]["trend"], expected)

    def test_foreign_resets_and_recoveries_cannot_change_managed_trend(self):
        for delta, pct in ((-3600, 8), (-1800, 11), (0, 14)):
            self.row(delta, pct)
        expected = self.budget()["oauth"]["trend"]
        self.assertEqual(expected["seven_day"]["burn_6h_pct_per_hr"], 6)
        for delta, pct in ((-3000, 90), (-2400, 0), (-600, 90)):
            self.row(delta, pct, "server")
        self.assertEqual(self.budget()["oauth"]["trend"], expected)

    def test_foreign_grants_cannot_cut_any_current_dollar_window(self):
        for key in ("seven_day", "five_hour", "scoped:opus"):
            self.row(-900, 80, "client", key)
            self.row(-300, 0, "server", key)
            self.row(0, 14, bucket=key)
        oauth = self.budget()["oauth"]
        for field, duration in (("limit_window", WEEK), ("five_hour_window", 5 * 3600)):
            self.assertEqual(oauth[field]["cost"], 10)
            self.assertEqual(oauth[field]["start_epoch"], self.reset - duration)
        scoped = next(b for b in oauth["buckets"] if b["key"] == "scoped:opus")
        self.assertEqual(scoped["window_cost"], 10)
        self.assertEqual(scoped["window_start_epoch"], self.reset - WEEK)

    def test_foreign_sibling_cannot_corroborate_small_managed_drop(self):
        self.snapshot(pct=1)
        for key in ("seven_day", "scoped:opus"):
            self.row(-900, 9, bucket=key)
            self.row(-300, 1, bucket=key)
            self.row(0, 1, bucket=key)
        self.row(-900, 9, "client", "five_hour")
        self.row(-300, 0, "client", "five_hour")
        oauth = self.budget()["oauth"]
        self.assertEqual(oauth["limit_window"]["cost"], 10)
        self.assertEqual(self.history()["resets"], [])

    def test_foreign_recovery_cannot_cancel_real_managed_grant(self):
        self.snapshot(pct=0)
        self.row(-900, 80)
        self.row(-300, 0)
        self.row(-100, 80, "client")
        self.row(0, 0)
        window = self.budget()["oauth"]["limit_window"]
        self.assertEqual(window["start_epoch"], ((self.observed - 300) // 60) * 60)
        self.assertEqual(window["cost"], 5)

    def test_managed_sibling_corroboration_still_cuts_window(self):
        self.snapshot(pct=1)
        for key in ("seven_day", "five_hour"):
            self.row(-900, 9, bucket=key)
            self.row(-300, 0 if key == "five_hour" else 1, bucket=key)
            self.row(0, 0 if key == "five_hour" else 1, bucket=key)
        self.assertEqual(self.budget()["oauth"]["limit_window"]["cost"], 5)
        self.assertEqual(self.history()["resets"][0]["corroborated_by"], "five_hour")

    def test_history_endpoint_excludes_legacy_and_future_rows(self):
        self.row(-900, 80, "client")
        self.row(0, 14)
        self.row(120, 70)
        result = self.history()
        self.assertEqual([r["pct"] for r in result["readings"]], [14])
        self.assertEqual(result["resets"], [])
        self.assertTrue(all(r["resets_at"].endswith(":00+00:00") for r in result["readings"]))

    def test_legacy_snapshot_keeps_legacy_writers_but_excludes_managed(self):
        self.snapshot(source="client")
        self.row(-1800, 10, "server")
        self.row(0, 14, "client")
        expected = self.budget()["oauth"]["trend"]
        self.row(-900, 90)
        self.assertEqual(self.budget()["oauth"]["trend"], expected)
        self.assertEqual([r["pct"] for r in self.history()["readings"]], [10, 14])

    def test_legacy_only_bucket_is_not_exposed_under_managed_owner(self):
        self.row(0, 88, "client", "scoped:legacy")
        self.assertEqual(self.history("scoped:legacy")["readings"], [])

    def test_spend_history_ignores_foreign_boundaries_and_peak_overlay(self):
        self.row(-900, 90, "server")
        self.row(-300, 0, "client")
        self.row(0, 14)
        result = self.spend()
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["peak_pct"], 14)
        self.assertEqual(result["windows"][0]["cost"], 10)
        self.assertEqual(result["months"][0]["total"], 10)

    def test_foreign_earliest_anchor_and_future_anchor_cannot_seed_segments(self):
        self.row(-10 * 86400, 90, "client", reset=self.reset - WEEK)
        self.row(0, 14)
        expected = self.spend()["windows"]
        self.assertEqual(expected[0]["start_epoch"], self.reset - WEEK)
        self.row(120, 0, reset=self.reset + WEEK)
        self.assertEqual(self.spend()["windows"], expected)

    def test_missing_managed_anchor_cannot_fall_back_to_legacy_history(self):
        self.snapshot(weekly=False)
        self.row(-900, 90, "client")
        result = self.spend()
        self.assertNotIn("windows", result)
        self.assertEqual(result["months"][0]["total"], 10)

    def test_source_filtering_never_deletes_or_rewrites_raw_history(self):
        self.row(-900, 90, "client")
        self.row(0, 14)
        before = [tuple(r) for r in self.conn.execute("SELECT * FROM limit_readings ORDER BY id")]
        self.budget()
        self.history()
        self.spend()
        self.assertEqual([tuple(r) for r in self.conn.execute("SELECT * FROM limit_readings ORDER BY id")], before)

    def test_spend_cache_cannot_reuse_legacy_overlay_after_transfer(self):
        self.snapshot(source="client")
        self.row(-900, 90, "client")
        client = self.client()
        url = "/api/spend-history?scope=personal"
        self.assertEqual(client.get(url).json()["windows"][-1]["peak_pct"], 90)
        self.snapshot()
        self.row(0, 14)
        result = client.get(url).json()
        self.assertEqual(result["windows"][-1]["peak_pct"], 14)
        self.assertEqual(result["months"][0]["total"], 10)

    def test_enterprise_endpoints_still_hide_personal_history(self):
        self.row(0, 14)
        client = self.client()
        self.assertEqual(client.get("/api/limit-history?scope=enterprise&bucket=seven_day").status_code, 404)
        self.assertNotIn("windows", client.get("/api/spend-history?scope=enterprise").json())
        self.assertNotIn("oauth", client.get("/api/rate-limits?scope=enterprise").json()["weekly_budget"])
