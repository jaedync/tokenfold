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
        # Fix 2: at_epoch is minute-floored like every other epoch field this
        # endpoint emits (resets_at above, resets_at_epoch_before/after
        # below) — it is NOT raw fetched_epoch precision anymore.
        expect_at_epoch = ((now - 600) // 60) * 60.0
        self.assertEqual(resets[0]["at_epoch"], expect_at_epoch)
        self.assertEqual(resets[0]["at_epoch"] % 60, 0)
        for key in ("utilization_before", "utilization_after",
                    "resets_at_epoch_before", "resets_at_epoch_after"):
            self.assertIn(key, resets[0])
        # No raw resets_at strings in the reset events.
        self.assertNotIn("resets_at_before", resets[0])
        self.assertNotIn(raw, r.text)

    def test_reset_event_epochs_never_leak_sub_minute_precision(self):
        """Privacy: at_epoch/resets_at_epoch_before/after must be
        minute-floored in the JSON response — a raw sub-minute offset is
        account-derived and could fingerprint the account across responses
        (Fix 2 extends this to at_epoch, which used to leave the server at
        full fetched_epoch precision)."""
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
        for key in ("at_epoch", "resets_at_epoch_before",
                    "resets_at_epoch_after"):
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
        # 500 days old: outside even the max window (9600h = 400d, F7).
        self._seed_row("seven_day", now - 500 * 86400, 40.0)
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

    # ── enterprise-account stomp guard (2026-07-09 incident) ────────────

    ENTERPRISE_SHAPED = {
        # What an enterprise-account machine actually pushes: every limit
        # bucket null (the account has no Max limits), extra_usage carrying
        # the org's numbers. Normalizes to ZERO usable buckets.
        "five_hour": {"utilization": None,
                      "resets_at": "2026-07-09T22:50:00+00:00"},
        "seven_day": {"utilization": None,
                      "resets_at": "2026-07-16T08:00:00+00:00"},
        "seven_day_opus": None,
        "limits": [{"kind": "weekly_all", "percent": None,
                    "resets_at": "2026-07-16T08:00:00+00:00"}],
        "extra_usage": {"utilization": 34.0, "is_enabled": True},
        "spend": {},
    }

    def test_no_usable_limits_does_not_stomp_snapshot(self):
        """A work machine logged into the enterprise account pushes usage
        fetched with ITS token; the blind REPLACE let that null-limits
        payload stomp the personal snapshot (gauges zeroed, Fable gone,
        extra_usage surfacing enterprise numbers) until the server poller
        healed it. Zero-usable-bucket payloads must never overwrite."""
        good = _fixture()
        self.assertEqual(self._post_usage(good).status_code, 200)
        r = self.client().post(
            "/api/usage",
            json={"machine": "Z000012-Mantle-VM-Dev01",
                  "usage": self.ENTERPRISE_SHAPED},
            headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "ignored_no_limits")
        meta = self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
        self.assertEqual(json.loads(meta["value"])["data"], good)
        # and no readings recorded for the ignored push
        n = self.conn.execute(
            "SELECT COUNT(*) c FROM limit_readings").fetchone()["c"]
        self.assertEqual(n, 3)  # the fixture's rows only

    def test_no_usable_limits_never_creates_snapshot(self):
        """First write on a fresh instance: an enterprise-shaped payload
        must not seed the snapshot either."""
        r = self._post_usage(self.ENTERPRISE_SHAPED)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "ignored_no_limits")
        self.assertIsNone(self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone())

    def test_ignored_push_captures_extra_usage(self):
        """The guard rejects the snapshot write but must RETAIN the org's
        extra_usage block (server-side billing dollars, in cents) in its own
        meta key — that's the only billing-grade number the enterprise side
        ever sees, and dropping the whole body threw it away."""
        payload = dict(self.ENTERPRISE_SHAPED)
        payload["extra_usage"] = {"is_enabled": True, "monthly_limit": 100000,
                                  "used_credits": 34012.5, "utilization": 34.0}
        r = self.client().post(
            "/api/usage",
            json={"machine": "Z000012-Mantle-VM-Dev01", "usage": payload},
            headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "ignored_no_limits")
        self.assertTrue(r.json()["captured_extra_usage"])
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage_enterprise'"
        ).fetchone()
        cap = json.loads(row["value"])
        self.assertEqual(cap["machine"], "Z000012-Mantle-VM-Dev01")
        self.assertEqual(cap["extra_usage"], payload["extra_usage"])
        self.assertIn("updated_at", cap)
        # the personal snapshot is still never touched
        self.assertIsNone(self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone())

    def test_ignored_push_without_extra_usage_keeps_prior_capture(self):
        """A null/absent extra_usage must not clobber a previously captured
        block — an empty push carries nothing worth overwriting with."""
        payload = dict(self.ENTERPRISE_SHAPED)
        payload["extra_usage"] = {"is_enabled": True, "used_credits": 5000}
        self.client().post(
            "/api/usage", json={"machine": "vm-a", "usage": payload},
            headers={"X-API-Key": self.api_key})
        empty = dict(self.ENTERPRISE_SHAPED)
        empty["extra_usage"] = None
        r = self.client().post(
            "/api/usage", json={"machine": "vm-b", "usage": empty},
            headers={"X-API-Key": self.api_key})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(r.json()["captured_extra_usage"])
        cap = json.loads(self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage_enterprise'"
        ).fetchone()["value"])
        self.assertEqual(cap["machine"], "vm-a")
        self.assertEqual(cap["extra_usage"]["used_credits"], 5000)

    def test_newer_capture_overwrites_older(self):
        """Snapshot semantics: the freshest extra_usage wins (same org, the
        meter only moves forward within a billing cycle)."""
        first = dict(self.ENTERPRISE_SHAPED)
        first["extra_usage"] = {"is_enabled": True, "used_credits": 1000}
        second = dict(self.ENTERPRISE_SHAPED)
        second["extra_usage"] = {"is_enabled": True, "used_credits": 2000}
        for machine, usage in (("vm-a", first), ("vm-b", second)):
            self.client().post(
                "/api/usage", json={"machine": machine, "usage": usage},
                headers={"X-API-Key": self.api_key})
        cap = json.loads(self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage_enterprise'"
        ).fetchone()["value"])
        self.assertEqual(cap["machine"], "vm-b")
        self.assertEqual(cap["extra_usage"]["used_credits"], 2000)

    def test_single_usable_bucket_still_stored(self):
        """The guard keys on 'normalizes to zero buckets', not payload
        completeness: one valid bucket is a real (if partial) snapshot."""
        usage = {"five_hour": {"utilization": 12.0,
                               "resets_at": "2026-07-09T22:50:00+00:00"}}
        r = self._post_usage(usage)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["status"], "ok")
        meta = self.conn.execute(
            "SELECT value FROM meta WHERE key='oauth_usage'").fetchone()
        self.assertEqual(json.loads(meta["value"])["data"], usage)


if __name__ == "__main__":
    unittest.main()
