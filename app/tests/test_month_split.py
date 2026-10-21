"""The model split under the work month's billed line (app/month_trajectory.py)."""
import unittest
from unittest.mock import patch

import app.config
from app.cost_windows import (compute_window_cost_by_model,
                              compute_window_cost_by_model_buckets)
from app.month_trajectory import OTHER_MODELS, split_layers
from app.tests._now_support import PERSONAL, WORK, insert_event
from app.tests.test_ha_windows import _HABase

B = 600  # bucket size used by the pure tests
START = 1_000_000.0


def bucket(t):
    return int((t - START) // B)


class SplitLayersTest(unittest.TestCase):
    """split_layers: budget $1000, so $10 is 1%."""

    def test_layers_follow_recorded_spend_and_the_rest_is_unrecorded(self):
        buckets = {bucket(START + 100): {"Opus 5.5": 80.0, "Sonnet 5.5": 20.0},
                   bucket(START + 4000): {"Opus 5.5": 300.0}}
        points = [[START + 3600, 10.0], [START + 7200, 50.0]]
        out = split_layers(points, buckets, START, B, 1000.0)
        self.assertEqual(out["models"], [
            {"model": "Opus 5.5", "pct": [8.0, 38.0]},
            {"model": "Sonnet 5.5", "pct": [2.0, 2.0]}])
        self.assertEqual(out["unrecorded"], [0.0, 10.0])

    def test_a_point_counts_only_buckets_that_ended_before_it(self):
        # The request at START + 650 is in bucket 1 (600..1200): it ends
        # after a point at START + 1000, so that point leaves it out.
        buckets = {bucket(START + 650): {"Opus 5.5": 50.0}}
        out = split_layers([[START + 1000, 5.0], [START + 1200, 5.0]],
                           buckets, START, B, 1000.0)
        self.assertEqual(out["models"][0]["pct"], [0.0, 5.0])
        self.assertEqual(out["unrecorded"], [5.0, 0.0])

    def test_recorded_above_the_line_scales_down_to_it(self):
        buckets = {0: {"Opus 5.5": 150.0, "Sonnet 5.5": 50.0}}
        out = split_layers([[START + 600, 10.0]], buckets, START, B, 1000.0)
        self.assertEqual([m["pct"] for m in out["models"]], [[7.5], [2.5]])
        self.assertEqual(out["unrecorded"], [0.0])

    def test_layers_add_up_to_each_point(self):
        buckets = {i: {"Opus 5.5": 3.0 * i, "Haiku 5.5": 1.0} for i in range(0, 40, 3)}
        points = [[START + i * 700, i * 1.7] for i in range(1, 30)]
        out = split_layers(points, buckets, START, B, 1000.0)
        for j, (_, pct) in enumerate(points):
            total = sum(m["pct"][j] for m in out["models"]) + out["unrecorded"][j]
            self.assertAlmostEqual(total, pct, places=3)

    def test_small_models_share_one_layer(self):
        spend = {f"Model {c}": 100.0 - i for i, c in enumerate("ABCDEFGH")}
        out = split_layers([[START + 600, 90.0]], {0: spend}, START, B, 1000.0)
        names = [m["model"] for m in out["models"]]
        self.assertEqual(names, ["Model A", "Model B", "Model C", "Model D",
                                 "Model E", OTHER_MODELS])
        # F, G and H: $95 + $94 + $93 of a $1000 budget.
        self.assertEqual(out["models"][-1]["pct"], [28.2])
        self.assertEqual(out["unrecorded"], [12.8])

    def test_six_models_keep_their_own_layers(self):
        spend = {f"Model {c}": 10.0 for c in "ABCDEF"}
        out = split_layers([[START + 600, 6.0]], {0: spend}, START, B, 1000.0)
        self.assertNotIn(OTHER_MODELS, [m["model"] for m in out["models"]])
        self.assertEqual(len(out["models"]), 6)

    def test_no_recorded_spend_is_no_split(self):
        self.assertIsNone(split_layers([[START + 600, 6.0]], {}, START, B, 1000.0))


class MonthSplitEndpointTest(_HABase):
    def setUp(self):
        super().setUp()
        from app.monthly_budget import _month_bounds_utc
        self.m_start, _, _, _ = _month_bounds_utc(self.now)
        if self.now - self.m_start < 2 * 3600:
            self.skipTest("the month is too young for 10 min buckets")

    def _get(self):
        with patch.object(app.config, "LOCKED_SCOPE", None):
            r = self.client().get("/api/runway/trajectory", params={
                "scope": "enterprise", "window": "claude:month"})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def _meter(self, t, used_usd, limit_usd=2000.0):
        self.conn.execute(
            "INSERT INTO extra_usage_readings(fetched_epoch, machine, used_cents,"
            " limit_cents, utilization) VALUES(?,?,?,?,?)",
            (t, "m", used_usd * 100, limit_usd * 100, None))
        self.conn.commit()

    def _spend(self, uuid, ts, usd, model="claude-opus-4-8", provider="anthropic",
               account=WORK):
        insert_event(self.conn, uuid, session="s-" + uuid, ts=ts, model=model,
                     client="pi-agent", provider=provider, account=account,
                     reported_total=usd)

    def test_work_claude_spend_splits_the_billed_area(self):
        t0 = self.m_start + 60
        self._spend("a", t0, 200.0)
        self._spend("b", t0 + 5, 100.0, model="claude-sonnet-5-5")
        # A Claude Code row ($5 per 1M input tokens): one Opus layer with the
        # Pi row, which names its provider ("Anthropic / Opus 4.8").
        insert_event(self.conn, "cc", session="cc", ts=t0 + 7, inp=20_000_000, account=WORK)
        # Not billed on the work meter: personal calls and other providers.
        self._spend("c", t0 + 9, 999.0, account=PERSONAL)
        self._spend("d", t0 + 12, 999.0, model="glm-5", provider="opencode-go")
        self._meter(self.now - 60, 500.0)
        body = self._get()
        self.assertEqual([p[1] for p in body["points"]], [25.0])
        split = body["split"]
        self.assertEqual([m["model"] for m in split["models"]], ["Opus 4.8", "Sonnet 5.5"])
        self.assertEqual([m["pct"] for m in split["models"]], [[15.0], [5.0]])
        self.assertEqual(split["unrecorded"], [5.0])

    def test_no_meter_points_is_no_split(self):
        from app.monthly_budget import set_budget
        set_budget(self.conn, 1000.0)
        self._spend("a", self.m_start + 60, 300.0)
        body = self._get()
        self.assertEqual(body["points"], [])
        self.assertIsNone(body["split"])

    def test_a_failed_spend_query_keeps_the_billed_line(self):
        self._meter(self.now - 60, 500.0)
        with patch("app.month_trajectory.compute_window_cost_by_model_buckets",
                   side_effect=RuntimeError("boom")):
            body = self._get()
        self.assertEqual([p[1] for p in body["points"]], [25.0])
        self.assertIsNone(body["split"])

    def test_buckets_sum_to_the_window_total(self):
        # Claude Code rows are priced by the server: the same price rules.
        for i in range(12):
            insert_event(self.conn, f"cc{i}", session="cc", ts=self.m_start + i * 431,
                         inp=200_000 + i, account=WORK)
        end = self.m_start + 12 * 431
        whole = compute_window_cost_by_model(self.conn, self.m_start, end,
                                             "enterprise", anthropic_only=True)
        parts = compute_window_cost_by_model_buckets(
            self.conn, self.m_start, end, 600, "enterprise", anthropic_only=True)
        self.assertEqual(sorted(parts), list(range(8)))  # 11 * 431 s is in bucket 7
        total = {}
        for models in parts.values():
            for m, usd in models.items():
                total[m] = total.get(m, 0.0) + usd
        self.assertEqual(set(total), set(whole))
        for m in whole:
            self.assertAlmostEqual(total[m], whole[m], places=9)


if __name__ == "__main__":
    unittest.main()
