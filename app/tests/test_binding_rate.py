"""binding_rate: $/point of the binding Claude long window (for /api/now)."""
import time
import unittest
from datetime import datetime, timezone

from app.tests._now_support import WORK, insert_event, set_oauth
from app.tests._support import TempDBTestCase


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


class BindingRateTests(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.now = time.time()

    def _build(self, scope="personal"):
        from app.binding_rate import build_binding_rate
        return build_binding_rate(scope, self.now)

    def test_usd_per_point_is_window_cost_over_pct(self):
        set_oauth(self.conn, {
            "seven_day": {"utilization": 10.0,
                          "resets_at": _iso(self.now + 3 * 86400)},
            "five_hour": {"utilization": 50.0,
                          "resets_at": _iso(self.now + 3600)},
        }, observed=self.now - 60)
        # $5 inside the window, $5 before its start (8 days ago): only the
        # first counts. A work row never counts for the personal window.
        insert_event(self.conn, "a", session="s", ts=self.now - 3600, inp=1_000_000)
        insert_event(self.conn, "b", session="s", ts=self.now - 8 * 86400, inp=1_000_000)
        insert_event(self.conn, "c", session="w", ts=self.now - 3600,
                     inp=1_000_000, account=WORK)
        r = self._build()
        self.assertEqual(r["bucket"], "seven_day")
        self.assertAlmostEqual(r["usd_per_point"], 0.5, places=6)

    def test_highest_projected_long_window_binds(self):
        set_oauth(self.conn, {
            "seven_day": {"utilization": 10.0,
                          "resets_at": _iso(self.now + 3 * 86400)},
            "seven_day_opus": {"utilization": 40.0,
                               "resets_at": _iso(self.now + 3 * 86400)},
        }, observed=self.now - 60)
        insert_event(self.conn, "a", session="s", ts=self.now - 3600, inp=1_000_000)
        r = self._build()
        self.assertEqual(r["bucket"], "scoped:opus")
        # Scoped windows count only their family's spend: Opus 4.8 is opus.
        self.assertAlmostEqual(r["usd_per_point"], 5.0 / 40.0, places=6)

    def test_none_for_enterprise_missing_or_tiny_pct(self):
        self.assertIsNone(self._build())                     # no snapshot
        set_oauth(self.conn, {"seven_day": {
            "utilization": 0.0, "resets_at": _iso(self.now + 86400)}},
            observed=self.now - 60)
        self.assertIsNone(self._build())                     # 0 % used
        set_oauth(self.conn, {"seven_day": {
            "utilization": 20.0, "resets_at": _iso(self.now + 86400)}},
            observed=self.now - 60)
        self.assertIsNone(self._build("enterprise"))

    def test_expired_window_is_ignored(self):
        set_oauth(self.conn, {"seven_day": {
            "utilization": 20.0, "resets_at": _iso(self.now - 60)}},
            observed=self.now - 3600)
        self.assertIsNone(self._build())


if __name__ == "__main__":
    unittest.main()
