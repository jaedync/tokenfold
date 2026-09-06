"""Executed regressions for independent quota atomicity/cutoff findings."""
import json
import time
from unittest.mock import patch

from app.tests._support import TempDBTestCase
from app.tests.test_bucket_windows import _ins_event, _ins_reading, _iso


class ResetObservationCutoffTest(TempDBTestCase):
    def setUp(self):
        super().setUp()
        self.freeze_pricing()
        self.observed = (time.time() // 60) * 60 - 570  # second30, always fresh
        self.now = self.observed + 600
        self.reset = self.now + 3600
        self.keys = ("seven_day", "five_hour", "scoped:opus")
        stored = {"data": {key: {"utilization": 50, "resets_at": _iso(self.reset)}
                           for key in ("seven_day", "five_hour", "seven_day_opus")},
                  "source": "meridian-oauth", "updated_at": _iso(self.observed)}
        self.conn.execute("INSERT INTO meta VALUES('oauth_usage',?)", (json.dumps(stored),))
        self.conn.commit()
        _ins_event(self.conn, "before-reset", "r1", self.observed - 600, inp=1_000_000)
        _ins_event(self.conn, "after-reset", "r2", self.observed - 60, inp=1_000_000)

    def windows(self):
        with patch("app.api.time.time", return_value=self.now):
            oauth = self.client().get("/api/rate-limits?scope=personal").json()["weekly_budget"]["oauth"]
        return {"weekly": (oauth["limit_window"]["start_epoch"], oauth["limit_window"]["cost"]),
                "five": (oauth["five_hour_window"]["start_epoch"], oauth["five_hour_window"]["cost"]),
                "scoped": (oauth["buckets"][2]["window_start_epoch"], oauth["buckets"][2]["window_cost"])}

    def reading(self, bucket, delta, pct):
        _ins_reading(self.conn, bucket, self.observed + delta, pct, self.reset)

    def test_later_same_minute_reset_cannot_cut_old_sample_windows(self):
        for bucket in self.keys:
            self.reading(bucket, -90, 80)
        before = self.windows()
        self.assertTrue(all(cost == 10 for _, cost in before.values()))
        for bucket in self.keys:
            self.reading(bucket, 10, 0)  # floors before observed, but was observed later
        self.assertEqual(self.windows(), before)

    def test_later_recovery_cannot_remove_old_sample_reset(self):
        for bucket in self.keys:
            self.reading(bucket, -900, 80)
            self.reading(bucket, -300, 0)
        before = self.windows()
        self.assertTrue(all(cost == 5 for _, cost in before.values()))
        for bucket in self.keys:
            self.reading(bucket, 60, 80)
        self.assertEqual(self.windows(), before)

    def test_later_sibling_recovery_cannot_remove_corroboration(self):
        # Small own drops only qualify through the scoped bucket's grant.
        for bucket in self.keys:
            self.reading(bucket, -900, 9)
            self.reading(bucket, -300, 0 if bucket == "scoped:opus" else 1)
        before = self.windows()
        self.assertTrue(all(cost == 5 for _, cost in before.values()))
        self.reading("scoped:opus", 60, 9)
        self.assertEqual(self.windows(), before)


class ManagedHistoryAtomicityTest(TempDBTestCase):
    def test_history_failure_rolls_back_owner_snapshot_and_partial_rows_then_retries(self):
        from fastapi.testclient import TestClient
        from app.main import app
        from app.claude_usage import store_snapshot
        now = time.time()
        payload = {"machine": "test", "account_class": "personal", "source": "meridian-oauth",
                   "source_profile": "default", "observed_at_epoch": now - 60,
                   "buckets": [{"key": key, "label": key, "pct": 10,
                                "resets_at_epoch": now + 3600}
                               for key in ("five_hour", "seven_day")]}
        client = TestClient(app, raise_server_exceptions=False)
        for failed_bucket in ("five_hour", "seven_day"):
            with self.subTest(failed_bucket=failed_bucket):
                self.conn.execute("DELETE FROM meta")
                self.conn.execute("DELETE FROM limit_readings")
                self.conn.commit()
                store_snapshot({"five_hour": {"utilization": 80}}, now, "client", history=False)
                prior = self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0]
                # SQL-trigger injection exercises the REAL helper's error path,
                # including a failure after the first bucket was already inserted.
                self.conn.execute("CREATE TRIGGER fail_history BEFORE INSERT ON limit_readings "
                                  f"WHEN NEW.bucket='{failed_bucket}' BEGIN "
                                  "SELECT RAISE(FAIL, 'synthetic history failure'); END")
                self.conn.commit()
                try:
                    response = client.post("/api/usage/claude", json=payload,
                                           headers={"X-API-Key": self.api_key})
                    self.assertEqual(response.status_code, 500)
                    self.assertNotIn("synthetic history", response.text)
                    self.assertEqual(self.conn.execute("SELECT value FROM meta WHERE key='oauth_usage'").fetchone()[0], prior)
                    self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 0)
                finally:
                    self.conn.execute("DROP TRIGGER fail_history")
                    self.conn.commit()
                with patch("app.aggregator.trigger_eager_rebuild"):
                    response = client.post("/api/usage/claude", json=payload,
                                           headers={"X-API-Key": self.api_key})
                self.assertEqual(response.json()["status"], "ok")
                self.assertEqual(self.conn.execute("SELECT count(*) FROM limit_readings").fetchone()[0], 2)
