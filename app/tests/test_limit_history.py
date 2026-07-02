"""Workstream C route tests: GET /api/limit-history (C3) and the
POST /api/usage history write (C1 client wiring, C6 bucket validation, C7
live-fixture end-to-end).
"""

import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app.tests._support import TempDBTestCase

FIXTURE_PATH = (Path(__file__).resolve().parent
                / "fixtures" / "oauth_usage_live_2026-07-01.json")


def _fixture():
    return json.loads(FIXTURE_PATH.read_text())


class _LimitHistoryBase(TempDBTestCase):

    def _seed_row(self, bucket, fetched_epoch, utilization,
                  resets_at=None, resets_at_epoch=None, source="server"):
        self.conn.execute(
            "INSERT INTO limit_readings(fetched_epoch, source, bucket, "
            "utilization, resets_at, resets_at_epoch) VALUES(?,?,?,?,?,?)",
            (fetched_epoch, source, bucket, utilization,
             resets_at, resets_at_epoch))
        self.conn.commit()

    def _get(self, qs):
        c = self.client()
        return c.get(f"/api/limit-history?{qs}")


# ---------------------------------------------------------------------------
# C3 — endpoint
# ---------------------------------------------------------------------------

class LimitHistoryEndpointTest(_LimitHistoryBase):

    def test_personal_scope_returns_ordered_scrubbed_readings_and_resets(self):
        now = time.time()
        raw = "2026-07-02T08:00:12.345678+00:00"
        resets_epoch = datetime.fromisoformat(raw).timestamp()
        # Insert OUT of fetched_epoch order; response must be ascending.
        # Second reading is a synthetic mid-window reset (63 -> 2 while
        # prev resets_at is still hours in the future).
        self._seed_row("seven_day", now - 600, 2.0,
                       resets_at=raw, resets_at_epoch=resets_epoch + 7 * 86400)
        self._seed_row("seven_day", now - 1200, 63.0,
                       resets_at=raw, resets_at_epoch=now + 3 * 3600)
        # Other-bucket noise must not leak into the response.
        self._seed_row("five_hour", now - 900, 10.0)

        r = self._get("bucket=seven_day&scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["bucket"], "seven_day")
        readings = body["readings"]
        self.assertEqual(len(readings), 2)
        ts = [x["t"] for x in readings]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(readings[0]["pct"], 63.0)
        # Minute-scrubbed: seconds zeroed, no sub-second component.
        scrubbed = readings[0]["resets_at"]
        self.assertNotIn(".", scrubbed)
        self.assertTrue(scrubbed.endswith(":00+00:00"), scrubbed)
        # The synthetic reset is detected and carries epoch fields only.
        resets = body["resets"]
        self.assertEqual(len(resets), 1)
        self.assertEqual(resets[0]["bucket"], "seven_day")
        self.assertAlmostEqual(resets[0]["at_epoch"], now - 600, places=2)
        for key in ("utilization_before", "utilization_after",
                    "resets_at_epoch_before", "resets_at_epoch_after"):
            self.assertIn(key, resets[0])
        # No raw resets_at strings in the reset events.
        self.assertNotIn("resets_at_before", resets[0])
        self.assertNotIn(raw, r.text)

    def test_reset_event_epochs_never_leak_sub_minute_precision(self):
        """Privacy: resets_at_epoch_before/after must be minute-floored in
        the JSON response — a raw sub-minute offset is account-derived and
        could fingerprint the account across responses."""
        now = time.time()
        raw = "2026-07-02T08:00:12.345678+00:00"
        resets_epoch = datetime.fromisoformat(raw).timestamp()
        self._seed_row("seven_day", now - 1200, 63.0,
                       resets_at=raw, resets_at_epoch=now + 3 * 3600)
        self._seed_row("seven_day", now - 600, 2.0,
                       resets_at=raw, resets_at_epoch=resets_epoch + 7 * 86400)

        r = self._get("bucket=seven_day&scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        resets = r.json()["resets"]
        self.assertEqual(len(resets), 1)
        for key in ("resets_at_epoch_before", "resets_at_epoch_after"):
            value = resets[0][key]
            if value is not None:
                self.assertEqual(value % 60, 0,
                                 f"{key}={value!r} not minute-floored")

    def test_scoped_bucket_key_with_colon_is_valid(self):
        now = time.time()
        self._seed_row("scoped:fable", now - 60, 34.0)
        r = self._get("bucket=scoped:fable&scope=personal")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["readings"]), 1)

    def test_invalid_bucket_rejected(self):
        for bad in ("../etc", "UPPER", "a b", "x" * 65, ""):
            r = self._get(f"bucket={bad}&scope=personal")
            self.assertIn(r.status_code, (400, 422),
                          f"bucket {bad!r} must be rejected")

    def test_bucket_with_trailing_newline_rejected(self):
        """Regression: '$' in the bucket regex matches before a trailing
        newline, so a naive .match() let 'seven_day\\n' through. fullmatch()
        must reject it."""
        r = self._get("bucket=seven_day%0A&scope=personal")
        self.assertIn(r.status_code, (400, 422), r.text)

    def test_hours_clamped_not_errored(self):
        now = time.time()
        # 100 days old: outside even the max window (2160h = 90d).
        self._seed_row("seven_day", now - 100 * 86400, 40.0)
        self._seed_row("seven_day", now - 60, 50.0)
        r = self._get("bucket=seven_day&scope=personal&hours=99999")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["readings"]), 1)  # clamp applied

    def test_hours_default_168(self):
        now = time.time()
        self._seed_row("seven_day", now - 8 * 86400, 40.0)  # outside 168h
        self._seed_row("seven_day", now - 60, 50.0)
        r = self._get("bucket=seven_day&scope=personal")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()["readings"]), 1)

    def test_enterprise_scope_is_404(self):
        now = time.time()
        self._seed_row("seven_day", now - 60, 50.0)
        r = self._get("bucket=seven_day&scope=enterprise")
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("pct", r.text)

    def test_requires_dashboard_auth_when_password_set(self):
        saved_pw = self._config.DASHBOARD_PASSWORD
        saved_user = self._config.DASHBOARD_USER
        self._config.DASHBOARD_PASSWORD = "pw"
        self._config.DASHBOARD_USER = "jaedyn"

        def _restore():
            self._config.DASHBOARD_PASSWORD = saved_pw
            self._config.DASHBOARD_USER = saved_user
        self.addCleanup(_restore)
        r = self._get("bucket=seven_day&scope=personal")
        self.assertEqual(r.status_code, 401)
        c = self.client()
        r2 = c.get("/api/limit-history?bucket=seven_day&scope=personal",
                   auth=("jaedyn", "pw"))
        self.assertEqual(r2.status_code, 200, r2.text)


# ---------------------------------------------------------------------------
# C1 client wiring + C6 + C7 — POST /api/usage boundary
# ---------------------------------------------------------------------------

class StoreUsageLimitHistoryTest(TempDBTestCase):

    def _post_usage(self, usage):
        c = self.client()
        return c.post("/api/usage", json={"usage": usage},
                      headers={"X-API-Key": self.api_key})

    def test_live_fixture_writes_three_client_rows(self):
        r = self._post_usage(_fixture())
        self.assertEqual(r.status_code, 200, r.text)
        rows = self.conn.execute(
            "SELECT bucket, source FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual({x["bucket"] for x in rows},
                         {"five_hour", "seven_day", "scoped:fable"})
        self.assertEqual({x["source"] for x in rows}, {"client"})

    def test_mixed_valid_and_garbage_buckets(self):
        """C6: bucket-level validation is delegated to the normalizer inside
        record_limit_readings — garbage buckets are skipped, valid ones land,
        and the meta snapshot still stores the raw dict verbatim."""
        usage = {
            "five_hour": {"utilization": 42.0,
                          "resets_at": "2026-07-02T07:40:00+00:00"},
            "seven_day": {"utilization": "ninety"},  # garbage
        }
        r = self._post_usage(usage)
        self.assertEqual(r.status_code, 200, r.text)
        rows = self.conn.execute(
            "SELECT bucket, utilization FROM limit_readings").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bucket"], "five_hour")
        self.assertEqual(rows[0]["utilization"], 42.0)
        meta = self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
        self.assertEqual(json.loads(meta["value"])["data"], usage)

    def test_non_dict_usage_still_400s(self):
        r = self._post_usage("garbage")
        self.assertEqual(r.status_code, 400)
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()
