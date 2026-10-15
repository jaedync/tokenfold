"""GET /api/runway: the BUILD.md windows[] contract over the lean
/api/rate-limits build."""
import json
import time
from unittest.mock import patch

import app.config
from app.monthly_budget import set_budget
from app.tests.test_bucket_windows import _ins_event, _ins_pi_event
from app.tests.test_ha_windows import _HABase, _iso, _minute

WEEK_S = 7 * 86400
FIVE_H_S = 5 * 3600

WINDOW_KEYS = {
    "id", "provider", "key", "label", "kind", "window_s", "pct", "resets_at",
    "window_start", "observed_at", "stale", "age_s", "pace", "tight_epoch",
    "burn_pct_per_h", "usd_spent", "usd_per_point", "usd_left", "waitable",
    "verdict"}
PACE_KEYS = {"pace_state", "expected_pct", "projected_pct",
             "reaches_100_epoch", "elapsed_fraction"}


class RunwayTest(_HABase):
    def _get(self, scope="personal"):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/runway", params={"scope": scope})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _by_id(self, body):
        return {w["id"]: w for w in body["windows"]}

    def _seed_provider(self, pct=20.0, observed=None):
        snap = {"provider": "opencode-go", "windows": [{
            "key": "weekly", "label": "Weekly limit", "pct": pct,
            "resets_at_epoch": self.now + 3 * 86400,
            "window_seconds": WEEK_S}],
            "observed_at_epoch": observed or self.now - 60,
            "machine": "laptop"}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('provider_usage', ?)",
            (json.dumps({"scopes": {"personal": {"opencode-go": snap}}}),))
        self.conn.commit()

    def test_personal_shape_and_claude_dollars(self):
        self._seed_oauth(sd_pct=40.0, fh_pct=20.0)
        _ins_event(self.conn, "n1", "r1", self.now - 2 * 86400, inp=1_000_000)
        _ins_event(self.conn, "n2", "r2", self.now - 600, inp=1_000_000)
        body = self._get()
        self.assertEqual(set(body), {"generated_at", "scope", "binding",
                                     "windows", "budget"})
        self.assertEqual(body["scope"], "personal")
        self.assertIsNone(body["budget"])
        ws = self._by_id(body)
        self.assertEqual(set(ws), {"claude:seven_day", "claude:five_hour"})
        for w in ws.values():
            self.assertEqual(set(w), WINDOW_KEYS)
            self.assertEqual(set(w["pace"]), PACE_KEYS)
        sd = ws["claude:seven_day"]
        self.assertEqual(sd["kind"], "long")
        self.assertEqual(sd["label"], "Claude · 7-day")
        self.assertEqual(sd["window_s"], WEEK_S)
        self.assertEqual(sd["resets_at"], _minute(self.sd_resets))
        self.assertEqual(sd["window_start"], _minute(self.sd_resets) - WEEK_S)
        self.assertFalse(sd["stale"])
        self.assertAlmostEqual(sd["usd_spent"], 10.0, places=2)
        self.assertAlmostEqual(sd["usd_per_point"], 0.25, places=4)
        self.assertAlmostEqual(sd["usd_left"], 15.0, places=2)
        fh = ws["claude:five_hour"]
        self.assertEqual(fh["kind"], "short")
        self.assertAlmostEqual(fh["usd_spent"], 5.0, places=2)
        self.assertEqual(body["binding"], "claude:seven_day")

    def test_granted_reset_moves_window_start(self):
        grant_t = self.now - 86400 + 13.5
        self._seed_oauth(sd_pct=10.0)
        self._grant("seven_day", grant_t, self.sd_resets, before=97.0, after=0.0)
        sd = self._by_id(self._get())["claude:seven_day"]
        self.assertEqual(sd["window_start"], _minute(grant_t))

    def test_stale_snapshot_has_no_dollars(self):
        self._seed_oauth(observed=self.now - 2 * 3600)
        _ins_event(self.conn, "n1", "r1", self.now - 4 * 3600, inp=1_000_000)
        sd = self._by_id(self._get())["claude:seven_day"]
        self.assertTrue(sd["stale"])
        self.assertGreaterEqual(sd["age_s"], 2 * 3600)
        self.assertIsNone(sd["usd_spent"])
        self.assertIsNone(sd["usd_per_point"])
        self.assertIsNone(sd["window_start"])

    def test_provider_window(self):
        self._seed_provider(pct=20.0)
        _ins_pi_event(self.conn, "p1", "q1", self.now - 600, "opencode-go",
                      "glm-5", 4.0)
        w = self._by_id(self._get())["opencode-go:weekly"]
        self.assertEqual(set(w), WINDOW_KEYS)
        self.assertEqual(w["provider"], "opencode-go")
        self.assertEqual(w["label"], "OpenCode Go · Weekly limit")
        self.assertEqual(w["kind"], "long")
        self.assertAlmostEqual(w["usd_spent"], 4.0, places=4)
        self.assertAlmostEqual(w["usd_per_point"], 0.2, places=4)
        self.assertAlmostEqual(w["usd_left"], 16.0, places=4)
        self.assertIsNone(w["burn_pct_per_h"])
        self.assertIsNone(w["tight_epoch"])

    def test_short_window_resetting_soon_is_waitable(self):
        self.fh_resets = self.now + 30 * 60
        self._seed_oauth()
        fh = self._by_id(self._get())["claude:five_hour"]
        self.assertTrue(fh["waitable"])

    def test_enterprise_has_month_window_and_budget(self):
        self._seed_oauth()
        set_budget(self.conn, 1000.0)
        body = self._get("enterprise")
        ws = self._by_id(body)
        self.assertEqual(list(ws), ["claude:month"])
        m = ws["claude:month"]
        self.assertEqual(set(m), WINDOW_KEYS)
        self.assertEqual(m["kind"], "month")
        self.assertEqual(m["usd_per_point"], 10.0)
        self.assertIn(m["pace"]["pace_state"],
                      ("above", "on", "below", "too_early"))
        self.assertEqual(set(body["budget"]), {
            "budget_usd", "spent_usd", "projected_usd", "pace_state",
            "per_workday_left_usd"})
        self.assertEqual(body["budget"]["budget_usd"], 1000.0)
        self.assertEqual(body["binding"], "claude:month")

    def test_enterprise_without_budget_has_no_windows(self):
        self._seed_oauth()
        body = self._get("enterprise")
        self.assertEqual(body["windows"], [])
        self.assertIsNone(body["binding"])
        self.assertIsNone(body["budget"])

    def test_failing_build_degrades_to_empty(self):
        self._seed_oauth()
        with patch("app.runway._rate_limits_payload",
                   side_effect=RuntimeError("boom")):
            body = self._get()
        self.assertEqual(body["windows"], [])
        self.assertIsNone(body["binding"])

    def test_failing_provider_part_keeps_claude(self):
        self._seed_oauth()
        self._seed_provider()
        with patch("app.runway.provider_windows",
                   side_effect=RuntimeError("boom")):
            body = self._get()
        self.assertEqual(set(self._by_id(body)),
                         {"claude:seven_day", "claude:five_hour"})

    def test_invalid_scope(self):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/runway", params={"scope": "nope"})
        self.assertEqual(r.status_code, 400)


class MonthWindowTest(_HABase):
    def test_business_time_walks_over_weekends(self):
        from datetime import datetime, timezone
        from app.runway_windows import business_epoch_after
        fri_noon = datetime(2026, 10, 16, 12, tzinfo=timezone.utc).timestamp()
        mon_noon = datetime(2026, 10, 19, 12, tzinfo=timezone.utc).timestamp()
        self.assertEqual(business_epoch_after(fri_noon, 86400), mon_noon)
        self.assertEqual(business_epoch_after(fri_noon, 3600), fri_noon + 3600)

    def test_over_pace_month_reaches_100(self):
        from app.runway_windows import month_window
        block = {"budget_usd": 1000.0, "mtd_cost": 600.0, "pace": "over",
                 "elapsed_fraction": 0.5, "business_days": 22,
                 "month_end_epoch": self.now + 15 * 86400,
                 "expected_usd": 500.0, "projected_eom_usd": 1200.0,
                 "source": "estimate"}
        w = month_window(block, self.now)
        self.assertEqual(w["pace"]["pace_state"], "above")
        self.assertAlmostEqual(w["pace"]["projected_pct"], 120.0)
        self.assertAlmostEqual(w["pct"], 60.0)
        reaches = w["pace"]["reaches_100_epoch"]
        self.assertIsNotNone(reaches)
        self.assertGreater(reaches, self.now)
        self.assertLess(reaches, block["month_end_epoch"])
