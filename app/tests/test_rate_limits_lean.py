"""_rate_limits_payload(lean=True): the /api/runway input.

/api/runway needs the gauges, pace, trends and window dollars, but not the
rolling-7d chart inputs (week_cost, week_active_s, hourly_costs), the
limit-window active time or the providers' month totals. Those are the
expensive event scans, so the lean build skips them; everything it keeps
must equal the full build.
"""
import json
import time
from datetime import datetime, timezone
from unittest.mock import patch

import app.config
from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _ins_event, _ins_pi_event


def _iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


class LeanRateLimitsTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = time.time()
        sd = self.now + 3 * 86400
        fh = self.now + 2 * 3600
        stored = {"data": {
            "seven_day": {"utilization": 40.0, "resets_at": _iso(sd)},
            "five_hour": {"utilization": 20.0, "resets_at": _iso(fh)},
        }, "updated_at": _iso(self.now - 30)}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('oauth_usage', ?)",
            (json.dumps(stored),))
        self.conn.commit()
        _ins_event(self.conn, "e1", "r1", self.now - 600, inp=1_000_000)
        _ins_pi_event(self.conn, "p1", "q1", self.now - 300, "opencode-go",
                      "glm-5", 2.0)

    def _payload(self, lean):
        from app.api import _rate_limits_payload
        with patch.object(app.config, "LOCKED_SCOPE", None), \
                patch("app.api.time.time", return_value=self.now):
            return _rate_limits_payload("personal", self.conn, lean=lean)

    def test_lean_skips_the_rolling_chart_inputs(self):
        lean = self._payload(True)["weekly_budget"]
        for key in ("week_cost", "week_active_s", "hourly_costs"):
            self.assertNotIn(key, lean)
        self.assertNotIn("active_s", lean["oauth"]["limit_window"])

    def test_lean_keeps_the_gauges_and_window_dollars(self):
        full = self._payload(False)["weekly_budget"]
        lean = self._payload(True)["weekly_budget"]
        self.assertEqual(lean["oauth"]["buckets"], full["oauth"]["buckets"])
        self.assertEqual(lean["oauth"]["limit_window"]["cost"],
                         full["oauth"]["limit_window"]["cost"])
        self.assertEqual(lean["oauth"]["five_hour_window"],
                         full["oauth"]["five_hour_window"])
        self.assertGreater(full["week_cost"], 0)

    def test_full_build_is_unchanged(self):
        from app.api import _build_rate_limits
        with patch.object(app.config, "LOCKED_SCOPE", None):
            body = json.loads(_build_rate_limits("personal", self.conn).body)
        wb = body["weekly_budget"]
        for key in ("week_cost", "week_active_s", "hourly_costs"):
            self.assertIn(key, wb)
        self.assertIn("active_s", wb["oauth"]["limit_window"])


class ProviderBlockMonthCostTest(TempDBTestCase):
    def test_month_cost_can_be_skipped_without_losing_window_dollars(self):
        from app.provider_usage import provider_usage_block
        from app.tests.test_provider_usage import _codex
        now = time.time()
        snap = {**_codex(10, now), "observed_at_epoch": now - 60,
                "machine": "laptop"}
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('provider_usage', ?)",
            (json.dumps({"scopes": {"personal": {"codex": snap}}}),))
        self.conn.commit()
        full = provider_usage_block("personal", now, conn=self.conn)
        lean = provider_usage_block("personal", now, conn=self.conn,
                                    include_month_cost=False)
        self.assertIn("month_cost", full["codex"])
        self.assertNotIn("month_cost", lean["codex"])
        self.assertEqual(lean["codex"]["windows"], full["codex"]["windows"])
